from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from probable_cause import db

FIXTURES = Path(__file__).parent / "fixtures"
AVALL_FIXTURE = FIXTURES / "avall"

type Conn = psycopg.Connection[tuple[object, ...]]


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A throwaway pgvector/Postgres 17 container shared by the whole test session."""
    testcontainers = pytest.importorskip("testcontainers.postgres")
    try:
        container = testcontainers.PostgresContainer("pgvector/pgvector:pg17", driver=None)
        container.start()
    except Exception as exc:  # Docker not running or image unavailable
        pytest.skip(f"Postgres testcontainer unavailable: {exc}")
    try:
        yield str(container.get_connection_url())
    finally:
        container.stop()


@pytest.fixture
def conn(database_url: str) -> Iterator[Conn]:
    """A connection to a freshly migrated, empty schema."""
    with psycopg.connect(database_url, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE")
        c.execute("CREATE SCHEMA public")
    with psycopg.connect(database_url) as c:
        db.migrate(c)
        c.commit()
        yield c
