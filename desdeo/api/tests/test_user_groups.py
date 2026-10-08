"""Tests for generic user groups and the problem access they grant."""

from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from desdeo.api.models import (
    ProblemDB,
    ReferencePoint,
    RPMSolveRequest,
    User,
    UserGroup,
    UserGroupMember,
    UserGroupProblem,
    UserRole,
)
from desdeo.api.routers.user_authentication import get_password_hash
from desdeo.api.routers.utils import fetch_problem_with_role_check, user_group_problem_ids

from .conftest import get_json, login, post_json


def _add_dm(db_session: Session, username: str) -> User:
    """Helper: add a DM user directly to the database."""
    user = User(username=username, password_hash=get_password_hash(username), role=UserRole.dm)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _analyst_problem(db_session: Session) -> ProblemDB:
    """Helper: the first problem of the analyst created by the fixture."""
    return db_session.exec(select(ProblemDB).order_by(ProblemDB.id)).first()


def test_membership_is_queryable_both_ways(session_and_user: dict):
    """Test that members can be added to and removed from a group, and found from either side."""
    db_session = session_and_user["session"]
    alice, bob = _add_dm(db_session, "alice"), _add_dm(db_session, "bob")

    group = UserGroup(name="pilot", owner_id=session_and_user["user"].id, members=[alice, bob])
    db_session.add(group)
    db_session.commit()
    db_session.refresh(group)

    assert {member.username for member in group.members} == {"alice", "bob"}
    alices_groups = db_session.exec(
        select(UserGroup).join(UserGroupMember).where(UserGroupMember.user_id == alice.id)
    ).all()
    assert [g.name for g in alices_groups] == ["pilot"]

    group.members.remove(bob)
    db_session.commit()
    db_session.refresh(group)
    assert [member.username for member in group.members] == ["alice"]


def test_group_grants_problem_access_to_members_only(session_and_user: dict):
    """Test that a group's problem is accessible to its members, and to nobody else through the group."""
    db_session = session_and_user["session"]
    member, outsider = _add_dm(db_session, "member"), _add_dm(db_session, "outsider")
    problem = _analyst_problem(db_session)

    assert fetch_problem_with_role_check(member, problem.id, db_session) is None

    group = UserGroup(name="live", members=[member], problems=[problem])
    db_session.add(group)
    db_session.commit()

    assert user_group_problem_ids(member, db_session) == {problem.id}
    assert fetch_problem_with_role_check(member, problem.id, db_session).id == problem.id
    assert user_group_problem_ids(outsider, db_session) == set()
    assert fetch_problem_with_role_check(outsider, problem.id, db_session) is None

    # revoking the grant revokes the access
    group.problems.remove(problem)
    db_session.commit()
    assert fetch_problem_with_role_check(member, problem.id, db_session) is None


def test_member_can_list_and_solve_group_problem(client: TestClient, session_and_user: dict):
    """Test that a member sees a group's problem in the problem listings and can solve it."""
    db_session = session_and_user["session"]
    member = _add_dm(db_session, "solver")
    problem = _analyst_problem(db_session)
    db_session.add(UserGroup(name="run", members=[member], problems=[problem]))
    db_session.commit()

    token = login(client, username="solver", password="solver")  # noqa: S106

    assert problem.id in [p["id"] for p in get_json(client, "/problem/all", token).json()]
    assert problem.id in [p["id"] for p in get_json(client, "/problem/all_info", token).json()]

    request = RPMSolveRequest(
        problem_id=problem.id,
        preference=ReferencePoint(aspiration_levels={"f_1": 0.5, "f_2": 0.3, "f_3": 0.4}),
        include_perturbed=False,
    )
    response = post_json(client, "/method/rpm/solve", request.model_dump(), token)
    assert response.status_code == status.HTTP_200_OK


def test_deleting_a_group_keeps_users_and_problems(session_and_user: dict):
    """Test that deleting a group removes its memberships and grants, but not the users or problems."""
    db_session = session_and_user["session"]
    member = _add_dm(db_session, "leaver")
    problem = _analyst_problem(db_session)
    group = UserGroup(name="closed", members=[member], problems=[problem])
    db_session.add(group)
    db_session.commit()

    db_session.delete(group)
    db_session.commit()

    assert db_session.exec(select(UserGroupMember)).all() == []
    assert db_session.exec(select(UserGroupProblem)).all() == []
    assert db_session.get(User, member.id) is not None
    assert db_session.get(ProblemDB, problem.id) is not None
