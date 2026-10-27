import csv
from datetime import date, time

import pytest

from probable_cause.ingest import normalize as n
from probable_cause.ingest.codes import CodeBook
from tests.conftest import AVALL_FIXTURE


def rows(name: str) -> list[dict[str, str]]:
    with (AVALL_FIXTURE / f"{name}.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def codes() -> CodeBook:
    return CodeBook.from_csv(AVALL_FIXTURE / "eADMSPUB_DataDictionary.csv")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1907", time(19, 7)), ("5", time(0, 5)), ("2400", None), ("1275", None), ("", None)],
)
def test_parse_hhmm(raw: str, expected: time | None) -> None:
    assert n.parse_hhmm(raw) == expected


def test_clean_narrative_normalizes_whitespace_and_controls() -> None:
    raw = "HISTORY\r\n\r\n\r\n\r\nThe pilot\x0b said  \t\r\nhello\u00a0world  "
    assert n.clean_narrative(raw) == "HISTORY\n\nThe pilot said\nhello world"
    assert n.clean_narrative(" \r\n ") is None


@pytest.mark.parametrize(
    ("ev_id", "mkey"), [("20210102102480", 102480), ("20080107X00026", None), ("2021", None)]
)
def test_carol_mkey(ev_id: str, mkey: int | None) -> None:
    assert n.carol_mkey(ev_id) == mkey


def test_normalize_event_decodes_and_builds_provenance(codes: CodeBook) -> None:
    row = next(r for r in rows("events") if r["ev_id"] == "20210102102480")
    event = n.normalize_event(row, codes)
    assert event.ev_date == date(2021, 1, 2)
    assert event.event_type == "Accident"
    assert event.carol_mkey == 102480
    assert event.report_url is not None
    assert event.report_url.endswith("/102480/pdf")
    assert event.docket_url.endswith("NTSBNumber=CEN21LA104")
    assert event.highest_injury in {"Fatal", "Serious", "Minor", "None"}


def test_normalize_event_without_carol_key(codes: CodeBook) -> None:
    row = next(r for r in rows("events") if r["ev_id"] == "20080107X00026")
    event = n.normalize_event(row, codes)
    assert event.report_url is None
    assert event.weather_condition == "Visual Meteorological Cond"


def test_normalize_aircraft_uses_supplemental_category(codes: CodeBook) -> None:
    row = next(r for r in rows("aircraft") if r["ev_id"] == "20080710X01009")
    aircraft = n.normalize_aircraft(row, codes)
    assert aircraft.category == "Weight-Shift"
    assert aircraft.registration == "N003PC"


def test_normalize_occurrence_splits_phase_and_event(codes: CodeBook) -> None:
    occurrence = n.normalize_occurrence(rows("Events_Sequence")[0], codes)
    assert occurrence.phase_code is not None
    assert occurrence.event_code is not None
    assert occurrence.occurrence_code == occurrence.phase_code + occurrence.event_code
    assert occurrence.phase is not None
    assert not occurrence.phase.isdigit()


def test_normalize_finding_maps_cause_factor() -> None:
    findings = [n.normalize_finding(r) for r in rows("Findings")]
    assert {f.cause_factor for f in findings} <= {"Cause", "Factor", None}
    assert any(f.in_probable_cause for f in findings)


def test_normalize_narrative_maps_sections() -> None:
    row = next(r for r in rows("narratives") if r["ev_id"] == "20080107X00026")
    narrative = n.normalize_narrative(row)
    assert narrative.factual is not None
    assert narrative.factual.startswith("On January 1, 2008")
    assert narrative.probable_cause is not None
