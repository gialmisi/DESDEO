"""Database initialization, user creation, and problem seeding."""

from __future__ import annotations

import importlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import typer

from desdeo.cli.styles import (
    BENCHMARK_PROBLEMS,
    DEFAULT_PROBLEMS,
    REAL_WORLD_PROBLEMS,
    TEST_PROBLEM_CATALOG,
    console,
    fail,
    get_problem_entry,
    get_problems_by_category,
    info,
    step_header,
    success,
    warn,
)

db_app = typer.Typer(help="Set up the database, users, and problems.")


API_DIR = Path(__file__).resolve().parent.parent / "api"
DEFAULT_DB_URL = f"sqlite:///{API_DIR / 'test.db'}"


def _get_db_path_from_config() -> str:
    """Return the default database URL, anchored to desdeo/api/."""
    return DEFAULT_DB_URL


def _init_database(db_url: str) -> None:
    """Create database tables."""
    from sqlalchemy_utils import database_exists
    from sqlmodel import SQLModel, create_engine

    if db_url.startswith("sqlite"):
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(db_url)

    # Import all models so SQLModel knows about them
    import desdeo.api.models  # noqa: F401

    if database_exists(engine.url):
        return engine, True
    SQLModel.metadata.create_all(engine)
    return engine, False


def _handle_existing_db(db_url: str):
    """Handle an existing database: keep, clear, or backup+recreate."""
    from sqlalchemy_utils import database_exists
    from sqlmodel import SQLModel, create_engine

    if db_url.startswith("sqlite"):
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(db_url)

    import desdeo.api.models  # noqa: F401

    if not database_exists(engine.url):
        SQLModel.metadata.create_all(engine)
        success("Database created.")
        return engine

    warn("Existing database found.")
    console.print("    1) Clear and recreate")
    console.print("    2) Keep existing")
    console.print("    3) Backup and recreate\n")

    choice = typer.prompt("  Choice", default="2")

    if choice == "1":
        SQLModel.metadata.reflect(bind=engine)
        SQLModel.metadata.drop_all(bind=engine)
        SQLModel.metadata.create_all(engine)
        success("Database cleared and recreated.")
    elif choice == "3":
        # For SQLite, backup the file
        if db_url.startswith("sqlite"):
            db_file = db_url.replace("sqlite:///", "")
            db_path = Path(db_file)
            if db_path.exists():
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                backup_path = db_path.with_suffix(f".{timestamp}.bak")
                shutil.copy2(db_path, backup_path)
                success(f"Backed up to {backup_path}")
        SQLModel.metadata.reflect(bind=engine)
        SQLModel.metadata.drop_all(bind=engine)
        SQLModel.metadata.create_all(engine)
        success("Database recreated.")
    else:
        info("Keeping existing database.")

    return engine


def _create_user(session, username: str, password: str, role_str: str, group: str):
    """Create a user in the database."""
    from desdeo.api.models import User, UserRole
    from desdeo.api.routers.user_authentication import get_password_hash

    role = UserRole(role_str)
    user = User(
        username=username,
        password_hash=get_password_hash(password),
        role=role,
        group=group,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _setup_users(engine) -> tuple:
    """Guide user creation. Returns (analyst_user, dm_users)."""
    from sqlmodel import Session

    step_header(2, 3, "User Creation")

    analyst_user = None
    dm_users = []

    with Session(engine) as session:
        # Analyst user
        console.print("\n  Create analyst (admin) user:")
        username = typer.prompt("    Username", default="analyst")
        password = typer.prompt("    Password", default="analyst", hide_input=True)
        group = typer.prompt("    Group", default="test")

        analyst_user = _create_user(session, username, password, "analyst", group)
        success(f"Analyst '{username}' created")

        # DM users
        console.print("\n  Create decision maker (DM) users?")
        console.print("    1) Create interactively")
        console.print('    2) Import from JSON file {{"user": "pass", ...}}')
        console.print("    3) Skip\n")

        choice = typer.prompt("  Choice", default="3")

        if choice == "1":
            while True:
                dm_username = typer.prompt("    DM username")
                dm_password = typer.prompt("    DM password", hide_input=True)
                dm_group = typer.prompt("    DM group", default=group)
                dm_user = _create_user(session, dm_username, dm_password, "dm", dm_group)
                dm_users.append(dm_user)
                success(f"DM '{dm_username}' created")
                if not typer.confirm("    Add another?", default=False):
                    break

        elif choice == "2":
            json_path = typer.prompt("    Path to JSON file")
            path = Path(json_path)
            if path.exists():
                with path.open() as f:
                    users_data = json.load(f)
                dm_group = typer.prompt("    Group for imported users", default=group)
                for uname, pwd in users_data.items():
                    dm_user = _create_user(session, uname, pwd, "dm", dm_group)
                    dm_users.append(dm_user)
                    success(f"DM '{uname}' created")
            else:
                fail(f"File not found: {json_path}")

    return analyst_user, dm_users


def _instantiate_problem(entry) -> object:
    """Import and instantiate a test problem from its catalog entry."""
    mod = importlib.import_module(entry.module)
    factory = getattr(mod, entry.name)

    if entry.params:
        kwargs = {}
        for param_name, default_val in entry.params:
            val = typer.prompt(f"      {param_name}", default=str(default_val))
            kwargs[param_name] = int(val)
        return factory(**kwargs)
    return factory()


def _display_problem_catalog() -> dict[int, str]:
    """Display the problem catalog and return number->name mapping."""
    categories = get_problems_by_category()
    number = 1
    mapping: dict[int, str] = {}

    for cat_name, entries in categories.items():
        console.print(f"\n  [bold]{cat_name}:[/bold]")
        for entry in entries:
            params_str = ""
            if entry.params:
                params_str = "(" + ", ".join(f"{p[0]}" for p in entry.params) + ")"
            console.print(f"    [{number:>2}] {entry.name}{params_str}  [dim]— {entry.description}[/dim]")
            mapping[number] = entry.name
            number += 1

    return mapping


def _setup_problems(engine, analyst_user) -> int:
    """Guide problem seeding. Returns number of problems added."""
    from sqlmodel import Session

    from desdeo.api.models import ProblemDB

    step_header(3, 3, "Problem Seeding")

    console.print("\n  Add optimization problems to the database?\n")
    console.print("  Quick selections:")
    console.print("    1) Default (dtlz2, knapsack, river_pollution) — same as db_init.py")
    console.print(f"    2) All benchmark problems ({len(BENCHMARK_PROBLEMS)})")
    console.print(f"    3) All real-world problems ({len(REAL_WORLD_PROBLEMS)})")
    console.print(f"    4) Everything ({len(TEST_PROBLEM_CATALOG)} problems)")
    console.print("    5) Custom selection")
    console.print("    6) Import from JSON file")
    console.print("    7) Skip\n")

    choice = typer.prompt("  Choice", default="1")

    selected_names: list[str] = []

    if choice == "1":
        selected_names = list(DEFAULT_PROBLEMS)
    elif choice == "2":
        selected_names = list(BENCHMARK_PROBLEMS)
    elif choice == "3":
        selected_names = list(REAL_WORLD_PROBLEMS)
    elif choice == "4":
        selected_names = [e.name for e in TEST_PROBLEM_CATALOG]
    elif choice == "5":
        mapping = _display_problem_catalog()
        nums_str = typer.prompt("\n  Enter numbers (comma-separated)")
        try:
            nums = [int(n.strip()) for n in nums_str.split(",")]
            selected_names = [mapping[n] for n in nums if n in mapping]
        except ValueError:
            fail("Invalid input. Skipping problem seeding.")
            return 0
    elif choice == "6":
        json_path = typer.prompt("  Path to problem JSON file")
        path = Path(json_path)
        if path.exists():
            from desdeo.problem.Problem import Problem

            with Session(engine) as session:
                problem = Problem.load_json(str(path))
                problem_db = ProblemDB.from_problem(problem, analyst_user)
                session.add(problem_db)
                session.commit()
                success(f"Imported problem from {json_path}")
                return 1
        else:
            fail(f"File not found: {json_path}")
            return 0
    else:
        info("Skipping problem seeding.")
        return 0

    if not selected_names:
        return 0

    # Instantiate and add selected problems
    count = 0
    with Session(engine) as session:
        # Re-fetch analyst user in this session
        from sqlmodel import select

        from desdeo.api.models import User

        stmt = select(User).where(User.id == analyst_user.id)
        user = session.exec(stmt).first()

        console.print(f"\n  Adding {len(selected_names)} problems...\n")
        for name in selected_names:
            entry = get_problem_entry(name)
            if not entry:
                warn(f"Unknown problem: {name}")
                continue
            try:
                if entry.params:
                    console.print(f"  Configure [bold]{name}[/bold]:")
                problem = _instantiate_problem(entry)
                problem_db = ProblemDB.from_problem(problem, user)
                session.add(problem_db)
                session.commit()
                success(f"{name}")
                count += 1
            except Exception as e:
                fail(f"{name}: {e}")

    return count


@db_app.callback(invoke_without_command=True)
def db() -> None:
    """Set up the database, create users, and seed problems."""
    step_header(1, 3, "Database Configuration")

    console.print("\n  Database mode:")
    console.print("    1) SQLite (local development) — recommended")
    console.print("    2) PostgreSQL (production) — coming soon\n")

    mode = typer.prompt("  Choice", default="1")

    if mode == "2":
        console.print("\n  [dim]PostgreSQL setup coming soon.[/dim]")
        console.print("  [dim]Use config.toml [database-deploy] and environment variables for now.[/dim]\n")
        return

    # SQLite mode
    default_url = _get_db_path_from_config()
    db_url = typer.prompt("  Database URL", default=default_url)

    console.print("\n  [dim]Initializing database (this may take a moment)...[/dim]")
    engine = _handle_existing_db(db_url)

    # Users
    analyst_user, dm_users = _setup_users(engine)

    # Problems
    problem_count = _setup_problems(engine, analyst_user)

    # Summary
    console.print()
    all_users = [analyst_user.username] + [u.username for u in dm_users]
    success(f"Database ready: {len(all_users)} users, {problem_count} problems")
