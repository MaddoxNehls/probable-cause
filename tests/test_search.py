"""Ranking sanity checks on a small synthetic corpus with known right answers."""

import pytest

from probable_cause.search import search
from tests.conftest import Conn

pytestmark = pytest.mark.db

EVENTS = [
    ("E1", "2012-01-10"),
    ("E2", "2015-06-01"),
    ("E3", "2009-02-01"),
    ("E4", "2019-03-03"),
]
DOCUMENTS = [
    # Probable cause squarely about airframe icing.
    ("E1", "probable_cause", "The pilot's continued flight into known icing conditions, which "
                             "resulted in airframe structural icing and a loss of control."),
    # Long factual narrative that mentions icing once, in passing.
    ("E2", "factual", "The pilot departed for a cross-country flight. " * 30
                      + "An AIRMET for icing was in effect north of the route. "
                      + "The engine lost power over a field. " * 30),
    ("E2", "probable_cause", "A total loss of engine power due to fuel exhaustion."),
    # Icing accident before the year filter.
    ("E3", "probable_cause", "Airframe icing and the pilot's failure to maintain airspeed."),
    ("E4", "probable_cause", "The pilot's improper flare, which resulted in a hard landing."),
]  # fmt: skip


@pytest.fixture
def corpus(conn: Conn) -> Conn:
    for ev_id, ev_date in EVENTS:
        conn.execute(
            "INSERT INTO events (ev_id, ntsb_no, event_type, ev_date, docket_url) "
            "VALUES (%s, %s, 'Accident', %s, 'https://example/docket')",
            (ev_id, f"NTSB-{ev_id}", ev_date),
        )
    for ev_id, section, body in DOCUMENTS:
        conn.execute(
            "INSERT INTO documents (ev_id, section, source, source_url, body, extraction_method) "
            "VALUES (%s, %s, 'avall', 'https://example/docket', %s, 'database')",
            (ev_id, section, body),
        )
    conn.commit()
    return conn


def test_probable_cause_match_outranks_passing_mention(corpus: Conn) -> None:
    hits = search(corpus, "icing")
    assert [h.ev_id for h in hits] == ["E1", "E3", "E2"]
    assert hits[0].section == "probable_cause"
    assert "**icing**" in hits[0].snippet


def test_one_hit_per_event_and_year_filter(corpus: Conn) -> None:
    hits = search(corpus, "icing", year_from=2010)
    assert [h.ev_id for h in hits] == ["E1", "E2"]
    assert search(corpus, "icing", year_to=2010)[0].ev_id == "E3"


def test_all_terms_by_default_any_on_request(corpus: Conn) -> None:
    assert search(corpus, "icing flare") == []
    assert {h.ev_id for h in search(corpus, "icing flare", match_any=True)} == {
        "E1", "E2", "E3", "E4"
    }  # fmt: skip


def test_no_match_returns_empty(corpus: Conn) -> None:
    assert search(corpus, "volcanic ash") == []
