"""Defines generic groups of users.

A group is a set of users that can be granted access to problems. Groups know nothing about how
they are used; e.g., an experiment run is a group of the subjects enrolled in it.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from .problem import ProblemDB
    from .user import User


class UserGroupMember(SQLModel, table=True):
    """Link table: a user is a member of a group."""

    group_id: int = Field(foreign_key="usergroup.id", primary_key=True, ondelete="CASCADE")
    user_id: int = Field(foreign_key="user.id", primary_key=True, ondelete="CASCADE")


class UserGroupProblem(SQLModel, table=True):
    """Link table: a group grants its members access to a problem."""

    group_id: int = Field(foreign_key="usergroup.id", primary_key=True, ondelete="CASCADE")
    problem_id: int = Field(foreign_key="problemdb.id", primary_key=True, ondelete="CASCADE")


class UserGroup(SQLModel, table=True):
    """The table model of a group of users."""

    id: int | None = Field(primary_key=True, default=None)
    name: str = Field(index=True, unique=True)
    description: str | None = Field(default=None)
    owner_id: int | None = Field(foreign_key="user.id", default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    members: list["User"] = Relationship(link_model=UserGroupMember)
    problems: list["ProblemDB"] = Relationship(link_model=UserGroupProblem)
