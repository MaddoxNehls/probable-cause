import pytest

from probable_cause.ingest.codes import CodeBook
from tests.conftest import AVALL_FIXTURE


@pytest.fixture(scope="module")
def codes() -> CodeBook:
    return CodeBook.from_csv(AVALL_FIXTURE / "eADMSPUB_DataDictionary.csv")


def test_decodes_dictionary_codes_case_insensitively(codes: CodeBook) -> None:
    assert codes.decode("events", "ev_highest_injury", "FATL") == "Fatal"
    assert codes.decode("Events", "EV_HIGHEST_INJURY", " fatl ") == "Fatal"
    assert codes.decode("aircraft", "far_part", "091") == "Part 91: General Aviation"


def test_blank_is_none_and_unknown_passes_through(codes: CodeBook) -> None:
    assert codes.decode("events", "light_cond", "  ") is None
    assert codes.decode("events", "light_cond", None) is None
    assert codes.decode("events", "light_cond", "ZZZZ") == "ZZZZ"


def test_supplemental_codes_fill_dictionary_gaps(codes: CodeBook) -> None:
    assert codes.decode("aircraft", "acft_category", "WSFT") == "Weight-Shift"
    assert codes.decode("injury", "inj_person_category", "Flig") == "Flight crew"


def test_occurrence_codes_split_into_phase_and_event(codes: CodeBook) -> None:
    assert codes.decode("events_sequence", "phase", "550") == "Landing"
    assert codes.decode("events_sequence", "event", "250") == "Midair collision"
    assert codes.decode("events_sequence", "phase", "601") == "Autorotation"
