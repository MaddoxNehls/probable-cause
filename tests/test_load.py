import pytest

from probable_cause.ingest import load, stats
from tests.conftest import AVALL_FIXTURE, Conn

pytestmark = pytest.mark.db

SNAPSHOT = load.Snapshot("avall.zip", "2026-10-01T07:36:33", 123, "a" * 64)


def count(conn: Conn, query: str) -> int:
    row = conn.execute(query).fetchone()
    assert row is not None
    assert isinstance(row[0], int)
    return row[0]


def test_load_fixture_snapshot(conn: Conn) -> None:
    report = load.load_snapshot(conn, AVALL_FIXTURE, SNAPSHOT)
    assert report.loaded["events"] == 5
    assert report.loaded["aircraft"] == 6
    assert report.orphans_dropped == {
        "injuries": 0, "occurrences": 0, "findings": 0, "narratives": 0
    }  # fmt: skip
    assert count(conn, "SELECT count(*) FROM events") == 5

    midair = conn.execute(
        "SELECT count(*), bool_and(phase IS NOT NULL) FROM occurrences "
        "WHERE ev_id = '20080107X00026' AND event = 'Midair collision'"
    ).fetchone()
    assert midair == (2, True)
    category = conn.execute(
        "SELECT category FROM aircraft WHERE ev_id = '20080710X01009'"
    ).fetchone()
    assert category == ("Weight-Shift",)


def test_documents_have_provenance_and_skip_duplicate_analysis(conn: Conn) -> None:
    load.load_snapshot(conn, AVALL_FIXTURE, SNAPSHOT)
    sections = {
        str(row[0])
        for row in conn.execute("SELECT section FROM documents WHERE ev_id = '20080210X00166'")
    }
    assert sections == {"factual", "probable_cause"}  # analysis repeated the factual text
    urls = conn.execute(
        "SELECT DISTINCT source_url FROM documents WHERE ev_id = '20210102102480'"
    ).fetchall()
    assert len(urls) == 1
    assert str(urls[0][0]).endswith("/GenerateNewestReport/102480/pdf")
    assert count(conn, "SELECT count(*) FROM documents WHERE ev_id = '20231009193207'") == 0


def test_reload_is_idempotent_and_keeps_pdf_documents(conn: Conn) -> None:
    first = load.load_snapshot(conn, AVALL_FIXTURE, SNAPSHOT)
    conn.execute(
        "INSERT INTO documents (ev_id, section, source, source_url, body, extraction_method) "
        "VALUES ('20080107X00026', 'docket_pdf', 'docket', 'https://example/doc', 'text', 'ocr')"
    )
    conn.commit()

    assert load.load_snapshot(conn, AVALL_FIXTURE, SNAPSHOT).skipped
    again = load.load_snapshot(conn, AVALL_FIXTURE, SNAPSHOT, force=True)
    assert again.loaded == first.loaded
    assert count(conn, "SELECT count(*) FROM documents WHERE source = 'docket'") == 1
    assert count(conn, "SELECT count(*) FROM source_snapshots") == 2


def test_stats_render_on_loaded_fixture(conn: Conn) -> None:
    load.load_snapshot(conn, AVALL_FIXTURE, SNAPSHOT)
    markdown = stats.render_markdown(stats.collect(conn))
    assert "| events | 5 |" in markdown
    assert "## Coverage by year" in markdown
