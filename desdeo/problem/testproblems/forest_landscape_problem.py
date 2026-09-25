"""A synthetic forest landscape with several owners, for studying cross-owner consequences.

The landscape consists of four lots (owners) arranged as a 2x2 block. Each lot is a grid of stands,
and each stand is managed with exactly one of a small set of management regimes. The per-hectare
values of each regime are synthetic, but their magnitudes are loosely calibrated against a
MELA-simulated Finnish holding. The numbers are not evidence about any real forest.
"""

import math

import numpy as np
from pydantic import BaseModel, Field

from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    ExtraFunction,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    TensorConstant,
    TensorVariable,
    VariableTypeEnum,
)

REGIMES: tuple[str, ...] = ("set_aside", "selection_cut", "thinning", "clearfell")
"""The management regimes available on every stand, in the order used by all regime-indexed values."""

LOTS: tuple[str, ...] = ("A", "B", "C", "D")
"""The lots of the landscape, placed as A B on the top row and C D on the bottom row."""

# Median per-hectare values of each regime, in the order of REGIMES.
_REGIME_MEDIANS: dict[str, tuple[float, ...]] = {
    "npv": (0.0, 4300.0, 4300.0, 7900.0),  # EUR/ha, discounted harvest revenue
    "carbon": (156.0, 119.0, 116.0, 107.0),  # tCO2/ha, stored at the end of the planning horizon
    "habitat": (0.55, 0.75, 0.58, 0.20),  # hazel grouse habitat suitability index, [0, 1]
    "bilberry": (6.4, 8.1, 5.9, 2.0),  # bilberry yield index
    "mushroom": (0.68, 0.46, 0.58, 0.21),  # marketed mushroom yield index
    "scenic": (6.6, 6.5, 5.9, 5.1),  # scenic value index
    "deadwood": (25.0, 12.0, 8.0, 4.0),  # m3/ha, deadwood from natural mortality
}

# The objectives of an owner: (quantity, name, aggregation over the lot's stands). A "sum" is a total
# over the lot, a "mean" is an area-weighted mean over the lot.
_OBJECTIVES: tuple[tuple[str, str, str], ...] = (
    ("npv", "Net present value", "sum"),
    ("habitat", "Hazel grouse habitat suitability index", "mean"),
    ("carbon", "Carbon stock", "sum"),
)

# The properties of an owner's lot: derived from a solution, but not optimized.
_PROPERTIES: tuple[tuple[str, str, str], ...] = (
    ("bilberry", "Bilberry yield", "mean"),
    ("mushroom", "Mushroom yield", "mean"),
    ("scenic", "Scenic value", "mean"),
    ("deadwood", "Deadwood volume", "mean"),
    ("wind_damage", "Wind damage probability", "mean"),
    ("deadwood_hotspot", "Deadwood hotspot share", "mean"),
)


class Stand(BaseModel):
    """A single stand of the landscape."""

    id: int = Field(description="Index of the stand in the landscape.")
    lot: str = Field(description="The lot (owner) the stand belongs to.")
    row: int = Field(description="Row of the stand in the landscape grid.")
    col: int = Field(description="Column of the stand in the landscape grid.")
    area: float = Field(description="Area of the stand in hectares.")
    values: dict[str, list[float]] = Field(
        description="Per-hectare values of each quantity, one value per regime in the order of `REGIMES`."
    )


class CouplingParameters(BaseModel):
    """Parameters of the effects that the management of a stand has on the stands of other owners."""

    habitat_neighbourhood_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the neighbourhood in the habitat suitability of a stand bordering other owners. The stand-wise"
            " suitability is multiplied by `1 - w + w * N`, where `N` is the area-weighted mean stand-wise suitability"
            " of its neighbours across the ownership boundary."
        ),
    )
    wind_base_damage_probability: float = Field(
        default=0.026,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that a stand is damaged by wind in a 5-year period. The default is the share of damaged plots"
            " in the Finnish national forest inventory data of Suvanto et al. (2019)."
        ),
    )
    wind_open_border_log_odds: float = Field(
        default=0.310,
        description=(
            "Increase in the log-odds of wind damage when a neighbouring stand is open (clear-felled). The default is"
            " the estimate for an open stand border in the GLM of Suvanto et al. (2019)."
        ),
    )
    wind_severity: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplier of the base damage probability, e.g., for a storm-prone landscape.",
    )
    wind_periods: int = Field(default=3, ge=1, description="Number of 5-year periods in the planning horizon.")
    wind_susceptibility: dict[str, float] = Field(
        default={"set_aside": 1.0, "selection_cut": 0.8, "thinning": 1.2, "clearfell": 0.2},
        description=(
            "Multiplier of the base damage probability for each regime of the stand itself: tall, old stands and"
            " recently thinned stands are more susceptible, stands regenerated after clear-felling much less."
        ),
    )
    wind_salvage_share: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Share of the net present value of damaged timber that is recovered by salvage logging.",
    )
    deadwood_hotspot_threshold: float = Field(
        default=20.0,
        ge=0.0,
        description=(
            "Deadwood volume (m3/ha) at which a stand is a deadwood hotspot. The default is the volume supporting"
            " near-threatened wood-inhabiting fungi used by Mazziotta et al. (2023), after Penttilä et al. (2004)."
        ),
    )
    deadwood_spillover_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the neighbourhood in the value of a deadwood hotspot bordering other owners. The hotspot is"
            " multiplied by `1 - b + b * H`, where `H` is the area share of its neighbours across the ownership"
            " boundary that are hotspots too."
        ),
    )


class ForestLandscape(BaseModel):
    """A forest landscape of several lots, each consisting of stands on a grid."""

    regimes: list[str] = Field(description="The management regimes available on every stand.")
    lots: list[str] = Field(description="The lots (owners) of the landscape.")
    stands: list[Stand] = Field(description="All stands of the landscape.")
    neighbours: dict[int, list[int]] = Field(
        description="Ids of the stands sharing an edge with each stand, regardless of the owner."
    )
    baseline: dict[int, str] = Field(
        description="The baseline regime of each stand, which the other lots hold unless stated otherwise."
    )
    coupling: CouplingParameters = Field(description="Parameters of the effects between stands of different owners.")

    def lot_stands(self, lot: str) -> list[Stand]:
        """Return the stands belonging to `lot`."""
        return [stand for stand in self.stands if stand.lot == lot]

    def cross_owner_neighbours(self, stand_id: int) -> list[int]:
        """Return the ids of the neighbours of a stand that belong to another lot."""
        lot = self.stands[stand_id].lot
        return [n for n in self.neighbours[stand_id] if self.stands[n].lot != lot]


def _lot_grid_shape(stands_per_lot: int) -> tuple[int, int]:
    """Return the most square (rows, cols) grid holding exactly `stands_per_lot` stands."""
    rows = math.isqrt(stands_per_lot)
    while stands_per_lot % rows != 0:
        rows -= 1
    return rows, stands_per_lot // rows


def forest_landscape_data(
    stands_per_lot: int = 4,
    seed: int = 0,
    baseline_regime: str = "thinning",
    coupling: CouplingParameters | None = None,
) -> ForestLandscape:
    """Generate a synthetic forest landscape of four lots.

    Each lot is a grid of `stands_per_lot` stands, and the lots are placed as a 2x2 block, so
    that every lot shares a boundary with two other lots. Two stands are neighbours when they share
    an edge in the resulting landscape grid.

    The per-hectare value of each quantity on a stand is the median of its regime, scaled by a
    stand-specific factor and by a small regime-specific perturbation. The stand factor models
    site productivity for NPV and carbon, and site quality for the remaining quantities.

    Args:
        stands_per_lot (int, optional): the number of stands in each lot. Defaults to 4.
        seed (int, optional): seed for generating the areas and values. The same seed always
            gives the same landscape. Defaults to 0.
        baseline_regime (str, optional): the regime every stand holds in the baseline
            configuration. Defaults to "thinning", i.e., business as usual.
        coupling (CouplingParameters | None, optional): parameters of the effects between stands
            of different owners. If `None`, the defaults of `CouplingParameters` are used.
            Defaults to None.

    Returns:
        ForestLandscape: the generated landscape.
    """
    if stands_per_lot < 2:  # noqa: PLR2004
        msg = f"A lot must have at least two stands, got {stands_per_lot}."
        raise ValueError(msg)

    if baseline_regime not in REGIMES:
        msg = f"Unknown baseline regime '{baseline_regime}'; the regimes are {REGIMES}."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    lot_rows, lot_cols = _lot_grid_shape(stands_per_lot)

    stands: list[Stand] = []
    for lot_index, lot in enumerate(LOTS):
        lot_row, lot_col = divmod(lot_index, 2)
        for i in range(stands_per_lot):
            row, col = divmod(i, lot_cols)
            productivity = rng.uniform(0.7, 1.3)
            quality = rng.uniform(0.8, 1.2)
            values = {
                quantity: [
                    float(median * (productivity if quantity in ("npv", "carbon") else quality) * rng.uniform(0.9, 1.1))
                    for median in medians
                ]
                for quantity, medians in _REGIME_MEDIANS.items()
            }
            values["habitat"] = [min(value, 1.0) for value in values["habitat"]]
            stands.append(
                Stand(
                    id=len(stands),
                    lot=lot,
                    row=lot_row * lot_rows + row,
                    col=lot_col * lot_cols + col,
                    area=float(np.clip(rng.lognormal(math.log(1.2), 0.6), 0.3, 4.8)),
                    values=values,
                )
            )

    position = {(stand.row, stand.col): stand.id for stand in stands}
    neighbours = {
        stand.id: [
            position[(stand.row + dr, stand.col + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (stand.row + dr, stand.col + dc) in position
        ]
        for stand in stands
    }

    return ForestLandscape(
        regimes=list(REGIMES),
        lots=list(LOTS),
        stands=stands,
        neighbours=neighbours,
        baseline={stand.id: baseline_regime for stand in stands},
        coupling=coupling if coupling is not None else CouplingParameters(),
    )


def realized_values(
    landscape: ForestLandscape, lot: str, others: dict[int, str] | None = None
) -> dict[int, dict[str, list[float]]]:
    """Compute the per-hectare values of each regime on the stands of a lot, given the management of the other lots.

    The stand-wise values are adjusted by the effects of the stands of other owners:

    - Habitat neighbourhood: following Öhman et al. (2011), habitat suitability depends on both the
      stand and its neighbourhood. Here the spatial condition is graded rather than a threshold,
      and it counts only the neighbours across an ownership boundary: the stand-wise suitability
      is multiplied by `1 - w + w * N`, where `N` is the area-weighted mean stand-wise suitability
      of those neighbours under their regimes, and `w` is `habitat_neighbourhood_weight`.
    - Wind exposure: following Suvanto et al. (2019), a stand with an open (clear-felled)
      neighbour has higher log-odds of wind damage. The stand-wise values are taken to already
      include the ordinary risk of damage, so only the excess probability caused by an open
      neighbour across an ownership boundary reduces them: carbon by the excess probability, and
      net present value by the unsalvaged share of it.
    - Deadwood hotspots: following the spillover argument of Mazziotta et al. (2023), a stand with
      deadwood above `deadwood_hotspot_threshold` is a hotspot, and a hotspot is worth more for
      deadwood-dependent species when its neighbours are hotspots too. A hotspot bordering other
      owners is multiplied by `1 - b + b * H`, where `H` is the area share of its neighbours across
      the ownership boundary that are hotspots under their regimes, and `b` is
      `deadwood_spillover_weight`.

    Stands without neighbours in other lots keep their stand-wise values. In addition, the
    probability of wind damage over the planning horizon is reported for every stand as
    `wind_damage`, and its deadwood hotspot value as `deadwood_hotspot`.

    Öhman, K., Edenius, L., & Mikusiński, G. (2011). Optimizing spatial habitat suitability and
    timber revenue in long-term forest planning. Canadian Journal of Forest Research, 41(3),
    543-551. https://doi.org/10.1139/X10-232

    Mazziotta, A., Borges, P., Kangas, A., Halme, P., & Eyvindson, K. (2023). Spatial trade-offs
    between ecological and economical sustainability in the boreal production forest. Journal of
    Environmental Management, 330, 117144. https://doi.org/10.1016/j.jenvman.2022.117144

    Suvanto, S., Peltoniemi, M., Tuominen, S., Strandström, M., & Lehtonen, A. (2019).
    High-resolution mapping of forest vulnerability to wind for disturbance-aware forestry. Forest
    Ecology and Management, 453, 117619. https://doi.org/10.1016/j.foreco.2019.117619

    Args:
        landscape (ForestLandscape): the landscape.
        lot (str): the lot whose stands' values are computed.
        others (dict[int, str] | None, optional): the regimes of stands in other lots, by stand id.
            Stands not listed hold their baseline regime. If `None`, all other lots hold the
            baseline. Defaults to None.

    Returns:
        dict[int, dict[str, list[float]]]: for each stand of the lot, the per-hectare value of each
            quantity, one value per regime in the order of `landscape.regimes`.
    """
    configuration = {**landscape.baseline, **(others or {})}
    for stand_id, regime in (others or {}).items():
        if stand_id not in landscape.baseline or landscape.stands[stand_id].lot == lot:
            msg = f"Stand {stand_id} is not a stand of another lot than '{lot}'."
            raise ValueError(msg)
        if regime not in landscape.regimes:
            msg = f"Unknown regime '{regime}' for stand {stand_id}; the regimes are {landscape.regimes}."
            raise ValueError(msg)

    def stand_wise(stand_id: int, quantity: str) -> float:
        return landscape.stands[stand_id].values[quantity][landscape.regimes.index(configuration[stand_id])]

    coupling = landscape.coupling
    w = coupling.habitat_neighbourhood_weight

    def damage_over_horizon(regime: str, open_border: bool) -> float:
        """Probability of wind damage over the planning horizon, for a stand under `regime`."""
        p = min(
            coupling.wind_severity * coupling.wind_base_damage_probability * coupling.wind_susceptibility[regime], 1.0
        )
        if open_border and 0.0 < p < 1.0:
            odds = p / (1 - p) * math.exp(coupling.wind_open_border_log_odds)
            p = odds / (1 + odds)
        return 1 - (1 - p) ** coupling.wind_periods

    def is_hotspot(deadwood: float) -> float:
        """Whether a stand with `deadwood` volume per hectare is a deadwood hotspot, as 1.0 or 0.0."""
        return 1.0 if deadwood >= coupling.deadwood_hotspot_threshold else 0.0

    values = {}
    for stand in landscape.lot_stands(lot):
        stand_values = {quantity: list(regime_values) for quantity, regime_values in stand.values.items()}

        across = landscape.cross_owner_neighbours(stand.id)
        if across:
            neighbourhood = sum(landscape.stands[n].area * stand_wise(n, "habitat") for n in across) / sum(
                landscape.stands[n].area for n in across
            )
            stand_values["habitat"] = [(1 - w + w * neighbourhood) * value for value in stand_values["habitat"]]

        open_border = any(configuration[n] == "clearfell" for n in across)
        closed = [damage_over_horizon(regime, open_border=False) for regime in landscape.regimes]
        realized = [damage_over_horizon(regime, open_border=open_border) for regime in landscape.regimes]
        excess = [r - c for r, c in zip(realized, closed, strict=True)]
        stand_values["carbon"] = [value * (1 - e) for value, e in zip(stand_values["carbon"], excess, strict=True)]
        stand_values["npv"] = [
            value * (1 - (1 - coupling.wind_salvage_share) * e)
            for value, e in zip(stand_values["npv"], excess, strict=True)
        ]
        stand_values["wind_damage"] = realized

        spillover = 1.0
        if across:
            hotspot_share = sum(landscape.stands[n].area * is_hotspot(stand_wise(n, "deadwood")) for n in across) / sum(
                landscape.stands[n].area for n in across
            )
            spillover = 1 - coupling.deadwood_spillover_weight + coupling.deadwood_spillover_weight * hotspot_share
        stand_values["deadwood_hotspot"] = [spillover * is_hotspot(value) for value in stand.values["deadwood"]]

        values[stand.id] = stand_values

    return values


def forest_landscape_problem(
    landscape: ForestLandscape | None = None, lot: str = "A", others: dict[int, str] | None = None
) -> Problem:
    r"""Defines the forest planning problem of the owner of one lot in a forest landscape.

    The owner chooses one management regime for each stand of their lot, following the
    plan-selection structure of `forest_problem`. The problem is to

    \begin{align}
        \max_{\mathbf{x}} & \quad \sum_{j \in J} a_j \mathbf{v}_j^\top \mathbf{x}_j & \\
        & \quad \sum_{j \in J} \frac{a_j}{A} \mathbf{h}_j^\top \mathbf{x}_j & \\
        & \quad \sum_{j \in J} a_j \mathbf{c}_j^\top \mathbf{x}_j & \\
        \text{s.t.} & \quad \sum_{i} x_{ji} = 1, & \forall j \in J \\
        & \quad x_{ji} \in \{0,1\}, & \forall j \in J, ~\forall i,
    \end{align}

    where $J$ is the set of stands in the lot, $a_j$ is the area of stand $j$, $A$ is the area of the
    lot, and $\mathbf{v}_j$, $\mathbf{h}_j$, and $\mathbf{c}_j$ are the per-hectare net present value,
    hazel grouse habitat suitability index, and carbon stock of each regime on stand $j$. The binary variable
    $x_{ji}$ is one when regime $i$ is chosen for stand $j$.

    The values are realized values: they depend on how the other lots are managed, as computed by
    `realized_values`. The other lots are fixed, so the problem stays linear.

    The properties of the lot (berry and mushroom yield, scenic value, deadwood volume, wind damage
    probability, and deadwood hotspot share) are defined as extra functions: they are evaluated for
    every solution, but they are not optimized.

    The ideal point is exact. The nadir point is estimated from the payoff table, which is also exact
    here: the problem is separable over stands, so each objective is optimized by choosing its best
    regime on every stand, with no solver needed. Both are always computed with the other lots in
    their baseline configuration, whatever `others` is, so that the ideal and nadir of a lot never
    change.

    Args:
        landscape (ForestLandscape | None, optional): the landscape the lot belongs to. If `None`,
            the default landscape of `forest_landscape_data` is used. Defaults to None.
        lot (str, optional): the lot of the owner. Defaults to "A".
        others (dict[int, str] | None, optional): the regimes of stands in other lots, by stand id.
            Stands not listed hold their baseline regime. Defaults to None.

    Returns:
        Problem: the forest planning problem of the owner of `lot`.
    """
    landscape = forest_landscape_data() if landscape is None else landscape
    if lot not in landscape.lots:
        msg = f"Lot '{lot}' is not in the landscape; the lots are {landscape.lots}."
        raise ValueError(msg)

    stands = landscape.lot_stands(lot)
    lot_area = sum(stand.area for stand in stands)
    n_regimes = len(landscape.regimes)

    values = realized_values(landscape, lot, others)
    baseline_values = realized_values(landscape, lot)

    def weights(
        stand: Stand, quantity: str, aggregation: str, source: dict[int, dict[str, list[float]]] = values
    ) -> list[float]:
        weight = stand.area if aggregation == "sum" else stand.area / lot_area
        return [weight * value for value in source[stand.id][quantity]]

    constants = []
    variables = []
    constraints = []
    for stand in stands:
        variables.append(
            TensorVariable(
                name=f"Regime of stand {stand.id}",
                symbol=f"X_{stand.id}",
                variable_type=VariableTypeEnum.binary,
                shape=[n_regimes],
                lowerbounds=n_regimes * [0],
                upperbounds=n_regimes * [1],
                initial_values=[1] + (n_regimes - 1) * [0],
            )
        )
        constraints.append(
            Constraint(
                name=f"One regime on stand {stand.id}",
                symbol=f"x_con_{stand.id}",
                cons_type=ConstraintTypeEnum.EQ,
                func=f"Sum(X_{stand.id}) - 1",
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )
        constants.extend(
            TensorConstant(
                name=f"{name} of stand {stand.id}",
                symbol=f"{quantity.upper()}_{stand.id}",
                shape=[n_regimes],
                values=weights(stand, quantity, aggregation),
            )
            for quantity, name, aggregation in _OBJECTIVES + _PROPERTIES
        )

    def lot_sum(quantity: str) -> str:
        return " + ".join(f"{quantity.upper()}_{stand.id}@X_{stand.id}" for stand in stands)

    # The problem is separable over stands, so each row of the payoff table is attained by choosing,
    # on every stand, the regime that is best for that row's objective.
    payoff = {
        row: {
            quantity: sum(
                weights(stand, quantity, aggregation, baseline_values)[
                    int(np.argmax(weights(stand, row, row_aggregation, baseline_values)))
                ]
                for stand in stands
            )
            for quantity, _, aggregation in _OBJECTIVES
        }
        for row, _, row_aggregation in _OBJECTIVES
    }

    objectives = [
        Objective(
            name=name,
            symbol=quantity,
            func=lot_sum(quantity),
            maximize=True,
            ideal=payoff[quantity][quantity],
            nadir=min(payoff[row][quantity] for row in payoff),
            objective_type=ObjectiveTypeEnum.analytical,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )
        for quantity, name, _ in _OBJECTIVES
    ]

    extra_funcs = [
        ExtraFunction(
            name=name,
            symbol=quantity,
            func=lot_sum(quantity),
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )
        for quantity, name, _ in _PROPERTIES
    ]

    return Problem(
        name=f"Forest landscape problem, lot {lot}",
        description=(
            f"The forest planning problem of the owner of lot {lot} in a synthetic forest landscape of "
            f"{len(landscape.lots)} lots with {len(stands)} stands each."
        ),
        constants=constants,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        extra_funcs=extra_funcs,
    )


def forest_training_problem(stands_per_lot: int = 4, seed: int = 1, lot: str = "A") -> Problem:
    """Defines the forest planning problem of a training lot, for learning the interface before the study.

    The training lot belongs to a landscape of its own, generated with a different seed than the
    study landscape. It is in the same domain and has the same structure, but shares no stands or
    values with the study landscape, so nothing done on it carries over to the study problem.

    Args:
        stands_per_lot (int, optional): the number of stands in each lot. Defaults to 4.
        seed (int, optional): seed of the training landscape. Must differ from the seed of the
            study landscape. Defaults to 1, while the study landscape defaults to 0.
        lot (str, optional): the lot of the owner in the training landscape. Defaults to "A".

    Returns:
        Problem: the forest planning problem of the training lot.
    """
    landscape = forest_landscape_data(stands_per_lot=stands_per_lot, seed=seed)
    problem = forest_landscape_problem(landscape, lot)

    return problem.model_copy(
        update={
            "name": f"Forest training problem, lot {lot}",
            "description": f"A training lot for learning the interface. {problem.description}",
        }
    )
