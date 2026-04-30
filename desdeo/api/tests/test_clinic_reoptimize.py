"""Tests for site-level constraint re-optimization against the real clinic problem.

These tests load the actual clinic problem from `experiments/clinic/` and
exercise the full pipeline: site metadata retrieval, constrained variant
creation, conflict validation, and optional RPM re-solve.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from desdeo.api.models import (
    ProblemDB,
    ProblemMetaDataDB,
    RepresentativeNonDominatedSolutions,
    SiteSelectionMetaData,
    SolverSelectionMetadata,
    User,
)
from desdeo.problem import Problem

from .conftest import get_json, login, post_json

CLINIC_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "clinic"
CLINIC_DATA_DIR = CLINIC_DIR / "data"


# --- Helpers --------------------------------------------------------------


def _load_clinic_problem(session: Session, user: User) -> ProblemDB:
    """Insert the clinic problem (86 sites, 36 cities) into the in-memory DB."""
    raw = (CLINIC_DIR / "clinic_problem.json").read_text()
    problem = Problem.model_validate_json(raw, by_name=True)
    problem_db = ProblemDB.from_problem(problem, user=user)
    session.add(problem_db)
    session.commit()
    session.refresh(problem_db)
    return problem_db


def _attach_pareto(session: Session, problem_db: ProblemDB) -> RepresentativeNonDominatedSolutions:
    """Load `clinic_pareto.json` as a representative non-dominated set."""
    pareto = json.loads((CLINIC_DIR / "clinic_pareto.json").read_text())

    metadata_db = ProblemMetaDataDB(problem_id=problem_db.id)
    session.add(metadata_db)
    session.commit()
    session.refresh(metadata_db)

    # Pin a deterministic solver. The auto-picker prefers gurobipy if importable
    # but the test environment lacks a real Gurobi license; use cbc instead.
    session.add(
        SolverSelectionMetadata(
            metadata_id=metadata_db.id,
            solver_string_representation="pyomo_cbc",
        )
    )
    session.commit()

    rep = RepresentativeNonDominatedSolutions(
        metadata_id=metadata_db.id,
        name=pareto["name"],
        description=pareto["description"],
        solution_data=pareto["solution_data"],
        ideal=pareto["ideal"],
        nadir=pareto["nadir"],
    )
    session.add(rep)
    session.commit()
    session.refresh(rep)
    return rep


def _attach_site_selection_metadata(session: Session, problem_db: ProblemDB) -> SiteSelectionMetaData:
    """Build SiteSelectionMetaData from the CSVs shipped with the clinic problem."""
    cities: list[dict] = []
    with (CLINIC_DATA_DIR / "cities.csv").open() as f:
        for row in csv.DictReader(f):
            cities.append(
                {
                    "name": row["city"],
                    "lat": float(row["lat"]),
                    "lon": float(row["long"]),
                    "size": 5.0 + float(row["pop"]) / 1000.0,
                }
            )

    sites: list[dict] = []
    city_coords = {c["name"]: (c["lat"], c["lon"]) for c in cities}
    with (CLINIC_DATA_DIR / "proposedSitesProcessed.csv").open() as f:
        for row in csv.DictReader(f):
            city = row["city"]
            lat, lon = city_coords.get(city, (0.0, 0.0))
            sites.append({"name": row["siteName"], "node": city, "lat": lat, "lon": lon})

    travel: list[list[float]] = []
    with (CLINIC_DATA_DIR / "adjacencyMatrixTravelTime.csv").open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            travel.append([float(v) for v in row[1:]])

    metadata_db = problem_db.problem_metadata
    if metadata_db is None:
        metadata_db = ProblemMetaDataDB(problem_id=problem_db.id)
        session.add(metadata_db)
        session.commit()
        session.refresh(metadata_db)

    site_meta = SiteSelectionMetaData(
        metadata_id=metadata_db.id,
        sites_json=json.dumps(sites),
        nodes_json=json.dumps(cities),
        travel_time_matrix_json=json.dumps(travel),
        site_variable_symbols=[f"sv_{i + 1}" for i in range(len(sites))],
        coverage_variable_symbols=[f"cover_{i + 1}" for i in range(len(cities))],
        coverage_threshold=15.0,
    )
    session.add(site_meta)
    session.commit()
    session.refresh(site_meta)
    return site_meta


@pytest.fixture(name="clinic")
def clinic_fixture(session_and_user: dict):
    """Set up the full clinic problem + pareto + site metadata."""
    session: Session = session_and_user["session"]
    user: User = session_and_user["user"]

    problem_db = _load_clinic_problem(session, user)
    _attach_pareto(session, problem_db)
    site_meta = _attach_site_selection_metadata(session, problem_db)

    return {
        "session": session,
        "user": user,
        "problem_db": problem_db,
        "n_sites": len(site_meta.site_variable_symbols),
    }


def _post_constrained_variant(
    client: TestClient,
    problem_id: int,
    fixings: list[dict],
    token: str,
    max_total_sites: int | None = None,
):
    body: dict = {"variable_fixings": fixings}
    if max_total_sites is not None:
        body["max_total_sites"] = max_total_sites
    return post_json(
        client,
        f"/problem/{problem_id}/constrained_variant",
        body,
        token,
    )


def _solve_rpm(client: TestClient, problem_id: int, aspiration: dict[str, float], token: str):
    return post_json(
        client,
        "/method/rpm/solve",
        {
            "problem_id": problem_id,
            "preference": {
                "preference_type": "reference_point",
                "aspiration_levels": aspiration,
            },
        },
        token,
    )


def _reference_point(rep_pareto: dict, index: int = 0) -> dict[str, float]:
    """Pick a feasible reference point from the loaded pareto solutions."""
    return {k: rep_pareto["solution_data"][k][index] for k in ("f_1", "f_2", "f_3", "f_4", "f_5")}


def _flatten_var(value):
    """Solver may return scalar variables as nested lists; pull out the scalar."""
    while isinstance(value, list) and value:
        value = value[0]
    return value


# --- Test cases -----------------------------------------------------------


def test_get_sites(client: TestClient, clinic: dict):
    """GET /site-selection/sites/{problem_id} returns 86 sites with valid metadata."""
    token = login(client)
    problem_id = clinic["problem_db"].id
    n_sites = clinic["n_sites"]

    response = get_json(client, f"/site-selection/sites/{problem_id}", token)
    assert response.status_code == 200, response.text
    data = response.json()

    assert "sites" in data
    sites = data["sites"]
    assert len(sites) == n_sites

    seen_indices = set()
    for s in sites:
        assert 0 <= s["index"] < n_sites
        seen_indices.add(s["index"])
        assert s["name"], "site name must be non-empty"
        assert s["node"], "site city must be non-empty"
        assert 38.0 < s["lat"] < 42.5, f"unexpected lat {s['lat']} (expecting NW Ohio)"
        assert -85.0 < s["lon"] < -82.0, f"unexpected lon {s['lon']} (expecting NW Ohio)"
        assert s["variable_symbol"].startswith("sv_")
        assert s["variable_symbol"] == f"sv_{s['index'] + 1}"
    assert seen_indices == set(range(n_sites))


def test_get_sites_missing_metadata(client: TestClient, session_and_user: dict):
    """GET /site-selection/sites/{problem_id} returns 404 when no site metadata is loaded."""
    token = login(client)
    # session_and_user already creates dtlz2 / river / forest problems but no site metadata
    response = get_json(client, "/site-selection/sites/1", token)
    assert response.status_code == 404


def test_reoptimize_conflicting_constraints(client: TestClient, clinic: dict):
    """Including and excluding the same site should be rejected with 422."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    payload = [
        {"variable_symbol": "sv_5", "fixed_value": 1.0},
        {"variable_symbol": "sv_5", "fixed_value": 0.0},
    ]
    response = _post_constrained_variant(client, problem_id, payload, token)
    assert response.status_code == 422
    assert "sv_5" in response.json()["detail"]


def test_reoptimize_unknown_site_symbol(client: TestClient, clinic: dict):
    """Site index outside [0, 59] surfaces as an unknown-symbol 422."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    payload = [{"variable_symbol": "sv_99", "fixed_value": 0.0}]
    response = _post_constrained_variant(client, problem_id, payload, token)
    assert response.status_code == 422
    assert "sv_99" in response.json()["detail"]


def test_reoptimize_constraints_added_for_sites(client: TestClient, clinic: dict):
    """Site-level fixings produce the expected number of EQ constraints in the variant."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    payload = [
        {"variable_symbol": "sv_1", "fixed_value": 0.0},
        {"variable_symbol": "sv_6", "fixed_value": 0.0},
        {"variable_symbol": "sv_11", "fixed_value": 0.0},
        {"variable_symbol": "sv_2", "fixed_value": 1.0},
    ]
    response = _post_constrained_variant(client, problem_id, payload, token)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["n_constraints_added"] == 4
    assert data["parent_problem_id"] == problem_id
    assert data["problem_id"] != problem_id


def test_reoptimize_mixed_constraints_added(client: TestClient, clinic: dict):
    """Mixing site (sv_*) and city (cover_*) symbols is accepted; both produce constraints."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    payload = [
        {"variable_symbol": "sv_3", "fixed_value": 1.0},
        {"variable_symbol": "sv_7", "fixed_value": 0.0},
        {"variable_symbol": "cover_2", "fixed_value": 0.0},
        {"variable_symbol": "cover_5", "fixed_value": 1.0},
    ]
    response = _post_constrained_variant(client, problem_id, payload, token)
    assert response.status_code == 200, response.text
    assert response.json()["n_constraints_added"] == 4


def test_reoptimize_backward_compatible_city_only(client: TestClient, clinic: dict):
    """City-only constraints (no site fields) still create a valid variant."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    payload = [
        {"variable_symbol": "cover_1", "fixed_value": 0.0},
        {"variable_symbol": "cover_4", "fixed_value": 1.0},
    ]
    response = _post_constrained_variant(client, problem_id, payload, token)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["n_constraints_added"] == 2


def test_reoptimize_max_total_sites_alone(client: TestClient, clinic: dict):
    """Setting only max_total_sites (no fixings) creates a variant with 0 added EQ constraints."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    response = _post_constrained_variant(client, problem_id, [], token, max_total_sites=5)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["n_constraints_added"] == 0
    assert data["parent_problem_id"] == problem_id


def test_reoptimize_max_total_sites_negative_rejected(client: TestClient, clinic: dict):
    """Negative max_total_sites is rejected with 422."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    response = _post_constrained_variant(client, problem_id, [], token, max_total_sites=-1)
    assert response.status_code == 422
    assert "max_total_sites" in response.json()["detail"]


def test_reoptimize_max_total_sites_with_fixings(client: TestClient, clinic: dict):
    """max_total_sites is independent of variable fixings; both can be supplied at once."""
    token = login(client)
    problem_id = clinic["problem_db"].id

    payload = [
        {"variable_symbol": "sv_1", "fixed_value": 1.0},
        {"variable_symbol": "sv_2", "fixed_value": 0.0},
    ]
    response = _post_constrained_variant(client, problem_id, payload, token, max_total_sites=8)
    assert response.status_code == 200, response.text
    assert response.json()["n_constraints_added"] == 2


# --- Solver-backed tests --------------------------------------------------
# These actually invoke the RPM solver on the (linear, mixed-integer) clinic
# problem. They are slower; group them so the cheap tests above can run alone.


@pytest.mark.slow
def test_reoptimize_exclude_sites_solver(client: TestClient, clinic: dict):
    """Excluding sites forces sv_i = 0 in the optimal solution."""
    token = login(client)
    problem_id = clinic["problem_db"].id
    pareto = json.loads((CLINIC_DIR / "clinic_pareto.json").read_text())

    excluded = [1, 6, 11]  # 1-based sv symbols
    payload = [{"variable_symbol": f"sv_{i}", "fixed_value": 0.0} for i in excluded]
    var_resp = _post_constrained_variant(client, problem_id, payload, token)
    assert var_resp.status_code == 200, var_resp.text
    variant_id = var_resp.json()["problem_id"]

    solve_resp = _solve_rpm(client, variant_id, _reference_point(pareto), token)
    assert solve_resp.status_code == 200, solve_resp.text
    results = solve_resp.json()["solver_results"]
    assert results, "RPM returned no solutions"

    optimal_vars = results[0]["optimal_variables"]
    sv = optimal_vars["sv"]  # tensor-shaped value
    for i in excluded:
        # tensor [60, 1]; flatten and 1-based index
        flat = []

        def _flat(v):
            if isinstance(v, list):
                for x in v:
                    _flat(x)
            else:
                flat.append(v)

        _flat(sv)
        assert flat[i - 1] == pytest.approx(0.0, abs=1e-3), f"site sv_{i} expected 0 after exclusion, got {flat[i - 1]}"


@pytest.mark.slow
def test_reoptimize_include_sites_solver(client: TestClient, clinic: dict):
    """Including sites forces sv_i = 1 in the optimal solution."""
    token = login(client)
    problem_id = clinic["problem_db"].id
    pareto = json.loads((CLINIC_DIR / "clinic_pareto.json").read_text())

    included = [3, 8]
    payload = [{"variable_symbol": f"sv_{i}", "fixed_value": 1.0} for i in included]
    var_resp = _post_constrained_variant(client, problem_id, payload, token)
    assert var_resp.status_code == 200, var_resp.text
    variant_id = var_resp.json()["problem_id"]

    solve_resp = _solve_rpm(client, variant_id, _reference_point(pareto), token)
    assert solve_resp.status_code == 200, solve_resp.text
    results = solve_resp.json()["solver_results"]
    assert results

    sv = results[0]["optimal_variables"]["sv"]
    flat: list[float] = []

    def _flat(v):
        if isinstance(v, list):
            for x in v:
                _flat(x)
        else:
            flat.append(v)

    _flat(sv)
    for i in included:
        assert flat[i - 1] == pytest.approx(1.0, abs=1e-3), f"site sv_{i} expected 1 after inclusion, got {flat[i - 1]}"


@pytest.mark.slow
def test_reoptimize_max_total_sites_binds(client: TestClient, clinic: dict):
    """A tightened max_total_sites caps the total number of selected sites in the optimum."""
    token = login(client)
    problem_id = clinic["problem_db"].id
    pareto = json.loads((CLINIC_DIR / "clinic_pareto.json").read_text())

    cap = 4
    var_resp = _post_constrained_variant(client, problem_id, [], token, max_total_sites=cap)
    assert var_resp.status_code == 200, var_resp.text
    variant_id = var_resp.json()["problem_id"]

    solve_resp = _solve_rpm(client, variant_id, _reference_point(pareto), token)
    assert solve_resp.status_code == 200, solve_resp.text
    results = solve_resp.json()["solver_results"]
    assert results

    sv = results[0]["optimal_variables"]["sv"]
    flat: list[float] = []

    def _flat(v):
        if isinstance(v, list):
            for x in v:
                _flat(x)
        else:
            flat.append(v)

    _flat(sv)
    total = sum(round(x) for x in flat)
    assert total <= cap, f"expected at most {cap} sites selected, got {total}"


@pytest.mark.slow
def test_reoptimize_preserves_objective_finiteness(client: TestClient, clinic: dict):
    """After re-optimization the solver returns finite objective values."""
    token = login(client)
    problem_id = clinic["problem_db"].id
    pareto = json.loads((CLINIC_DIR / "clinic_pareto.json").read_text())

    payload = [{"variable_symbol": "sv_2", "fixed_value": 0.0}]
    var_resp = _post_constrained_variant(client, problem_id, payload, token)
    assert var_resp.status_code == 200, var_resp.text
    variant_id = var_resp.json()["problem_id"]

    solve_resp = _solve_rpm(client, variant_id, _reference_point(pareto), token)
    assert solve_resp.status_code == 200, solve_resp.text
    results = solve_resp.json()["solver_results"]
    assert results

    objs = results[0]["optimal_objectives"]
    for k, v in objs.items():
        scalar = _flatten_var(v)
        assert scalar is not None and float("-inf") < float(scalar) < float("inf"), f"objective {k} not finite: {v}"
