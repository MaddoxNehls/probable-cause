"""Connection helper and a minimal forward-only SQL migration runner."""

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def connect(database_url: str) -> psycopg.Connection[tuple[object, ...]]:
    return psycopg.connect(database_url)


def migrate(
    conn: psycopg.Connection[tuple[object, ...]], migrations_dir: Path = MIGRATIONS_DIR
) -> list[str]:
    """Apply every migrations/*.sql not yet recorded in schema_migrations, in name order."""
    with conn.transaction():
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
    newly_applied: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        if path.name in applied:
            continue
        with conn.transaction():
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
        newly_applied.append(path.name)
    return newly_applied
