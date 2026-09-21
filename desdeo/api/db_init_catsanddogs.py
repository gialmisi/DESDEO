"""Initialize the database with the cat and dog breed problems.

Includes everything from db_init.py and adds:
- cat_breed_problem and dog_breed_problem
- SolverSelectionMetadata pinning the proximal solver for both

Both problems back the cats and dogs demo in the web UI, which is reached at
`/demos/cats-and-dogs`. The demo looks the problems up by name, so the names
given to the problems here must match the ones the demo expects.

Run from desdeo/api/, which is where `run_fullstack.py` starts the API server
from, and therefore where the API server looks for `test.db`:
    python db_init_catsanddogs.py
"""
# ruff: noqa: T201

import warnings

from sqlalchemy_utils import database_exists
from sqlmodel import Session, SQLModel

from desdeo.api.config import ServerConfig, SettingsConfig
from desdeo.api.db import engine
from desdeo.api.models import ProblemDB, User, UserRole
from desdeo.api.models.problem import ProblemMetaDataDB, SolverSelectionMetadata
from desdeo.api.routers.user_authentication import get_password_hash
from desdeo.problem.testproblems import (
    cat_breed_problem,
    dog_breed_problem,
    dtlz2,
    river_pollution_problem,
    simple_knapsack,
)

# The problems are fully data based and come with a discrete representation, so
# the proximal solver is the one to use. Pinning it here keeps the automatic
# solver selection from being consulted at all.
_SOLVER = "proximal"


def add_problem_with_proximal_solver(session: Session, problem, user: User) -> ProblemDB:
    """Add a problem to the database and pin the proximal solver for it.

    Args:
        session (Session): the database session to use.
        problem: the problem to add.
        user (User): the user the problem belongs to.

    Returns:
        ProblemDB: the database entry created for the problem.
    """
    problem_db = ProblemDB.from_problem(problem, user)
    session.add(problem_db)
    session.commit()
    session.refresh(problem_db)

    metadata_db = ProblemMetaDataDB(problem_id=problem_db.id, problem=problem_db)
    session.add(metadata_db)
    session.commit()
    session.refresh(metadata_db)

    session.add(
        SolverSelectionMetadata(
            metadata_id=metadata_db.id,
            solver_string_representation=_SOLVER,
        )
    )
    session.commit()

    return problem_db


if __name__ == "__main__":
    if SettingsConfig.debug:
        print("Creating database tables.")
        if not database_exists(engine.url):
            SQLModel.metadata.create_all(engine)
        else:
            warnings.warn("Database already exists. Clearing it.", stacklevel=1)
            SQLModel.metadata.reflect(bind=engine)
            SQLModel.metadata.drop_all(bind=engine)
            SQLModel.metadata.create_all(engine)
        print("Database tables created.")

        with Session(engine) as session:
            user_analyst = User(
                username=ServerConfig.test_user_analyst_name,
                password_hash=get_password_hash(ServerConfig.test_user_analyst_password),
                role=UserRole.analyst,
                group="test",
            )
            session.add(user_analyst)
            session.commit()
            session.refresh(user_analyst)

            # Standard test problems from db_init.py
            for problem in [dtlz2(10, 3), simple_knapsack(), river_pollution_problem()]:
                session.add(ProblemDB.from_problem(problem, user_analyst))
            session.commit()
            print("Standard test problems added.")

            cat_db = add_problem_with_proximal_solver(session, cat_breed_problem(), user_analyst)
            print(f"Cat breed problem added (id={cat_db.id}).")

            dog_db = add_problem_with_proximal_solver(session, dog_breed_problem(), user_analyst)
            print(f"Dog breed problem added (id={dog_db.id}).")

        print("Done.")

    else:
        pass
