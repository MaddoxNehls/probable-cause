"""Turn raw mdb-export CSV rows into typed, decoded records for the relational schema."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time

from probable_cause.ingest.codes import CodeBook

type Row = Mapping[str, str]

DOCKET_URL = "https://data.ntsb.gov/Docket/?NTSBNumber={ntsb_no}"
CAROL_REPORT_URL = (
    "https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/{mkey}/pdf"
)
# Since the 2020 move to CAROL, ev_ids are 8-digit date + 6-digit CAROL key ("20210102102480").
# Older ids look like "20080211X00175" and carry no key.
_NEW_STYLE_EV_ID = re.compile(r"^\d{8}(\d{6})$")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def text_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_int(value: str | None) -> int | None:
    text = text_or_none(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    text = text_or_none(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value: str | None) -> date | None:
    text = text_or_none(value)
    if text is None:
        return None
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def parse_timestamp(value: str | None) -> datetime | None:
    text = text_or_none(value)
    if text is None:
        return None
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def parse_hhmm(value: str | None) -> time | None:
    """NTSB stores local time as an HHMM integer (1907 -> 19:07). Invalid values become None."""
    number = parse_int(value)
    if number is None:
        return None
    hours, minutes = divmod(number, 100)
    if 0 <= hours < 24 and 0 <= minutes < 60:
        return time(hours, minutes)
    return None


def parse_flag(value: str | None) -> bool | None:
    text = text_or_none(value)
    if text is None:
        return None
    return {"Y": True, "1": True, "T": True, "N": False, "0": False, "F": False}.get(text.upper())


def clean_narrative(value: str | None) -> str | None:
    """Normalize line endings and whitespace; drop stray control characters."""
    if value is None:
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text).replace("\u00a0", " ")
    text = _TRAILING_SPACE.sub("\n", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip() or None


def carol_mkey(ev_id: str) -> int | None:
    match = _NEW_STYLE_EV_ID.match(ev_id)
    return int(match.group(1)) if match else None


def _coordinate(value: str | None, limit: float) -> float | None:
    number = parse_float(value)
    if number is None or number == 0 or abs(number) > limit:
        return None
    return number


@dataclass(frozen=True, slots=True)
class Event:
    ev_id: str
    ntsb_no: str | None
    event_type: str
    ev_date: date
    ev_time_local: time | None
    timezone: str | None
    city: str | None
    state: str | None
    country: str | None
    latitude: float | None
    longitude: float | None
    airport_id: str | None
    airport_name: str | None
    highest_injury: str | None
    fatalities: int
    serious_injuries: int
    minor_injuries: int
    uninjured: int
    light_condition: str | None
    weather_condition: str | None
    carol_mkey: int | None
    docket_url: str
    report_url: str | None
    last_changed: datetime | None


@dataclass(frozen=True, slots=True)
class Aircraft:
    ev_id: str
    aircraft_key: int
    registration: str | None
    make: str | None
    model: str | None
    series: str | None
    category: str | None
    far_part: str | None
    far_part_desc: str | None
    operation_type: str | None
    damage: str | None
    amateur_built: bool | None
    num_engines: int | None


@dataclass(frozen=True, slots=True)
class Injury:
    ev_id: str
    aircraft_key: int
    person_category: str
    injury_level: str
    person_count: int


@dataclass(frozen=True, slots=True)
class Occurrence:
    ev_id: str
    aircraft_key: int
    occurrence_no: int
    occurrence_code: str | None
    phase_code: str | None
    phase: str | None
    event_code: str | None
    event: str | None
    description: str | None
    is_defining: bool


@dataclass(frozen=True, slots=True)
class Finding:
    ev_id: str
    aircraft_key: int
    finding_no: int
    finding_code: str | None
    description: str | None
    cause_factor: str | None
    in_probable_cause: bool


@dataclass(frozen=True, slots=True)
class Narrative:
    ev_id: str
    aircraft_key: int
    factual: str | None
    analysis: str | None
    probable_cause: str | None


def normalize_event(row: Row, codes: CodeBook) -> Event:
    ev_id = row["ev_id"].strip()
    ev_date = parse_date(row["ev_date"])
    if ev_date is None:
        raise ValueError(f"event {ev_id} has no ev_date")
    ntsb_no = text_or_none(row.get("ntsb_no"))
    mkey = carol_mkey(ev_id)
    return Event(
        ev_id=ev_id,
        ntsb_no=ntsb_no,
        event_type=codes.decode("events", "ev_type", row["ev_type"]) or "Unknown",
        ev_date=ev_date,
        ev_time_local=parse_hhmm(row.get("ev_time")),
        timezone=text_or_none(row.get("ev_tmzn")),
        city=text_or_none(row.get("ev_city")),
        state=text_or_none(row.get("ev_state")),
        country=text_or_none(row.get("ev_country")),
        latitude=_coordinate(row.get("dec_latitude"), 90),
        longitude=_coordinate(row.get("dec_longitude"), 180),
        airport_id=text_or_none(row.get("ev_nr_apt_id")),
        airport_name=text_or_none(row.get("apt_name")),
        highest_injury=codes.decode("events", "ev_highest_injury", row.get("ev_highest_injury")),
        fatalities=parse_int(row.get("inj_tot_f")) or 0,
        serious_injuries=parse_int(row.get("inj_tot_s")) or 0,
        minor_injuries=parse_int(row.get("inj_tot_m")) or 0,
        uninjured=parse_int(row.get("inj_tot_n")) or 0,
        light_condition=codes.decode("events", "light_cond", row.get("light_cond")),
        weather_condition=codes.decode("events", "wx_cond_basic", row.get("wx_cond_basic")),
        carol_mkey=mkey,
        docket_url=DOCKET_URL.format(ntsb_no=ntsb_no or ev_id),
        report_url=CAROL_REPORT_URL.format(mkey=mkey) if mkey is not None else None,
        last_changed=parse_timestamp(row.get("lchg_date")),
    )


def normalize_aircraft(row: Row, codes: CodeBook) -> Aircraft:
    far_part = text_or_none(row.get("far_part"))
    return Aircraft(
        ev_id=row["ev_id"].strip(),
        aircraft_key=int(row["Aircraft_Key"]),
        registration=text_or_none(row.get("regis_no")),
        make=text_or_none(row.get("acft_make")),
        model=text_or_none(row.get("acft_model")),
        series=text_or_none(row.get("acft_series")),
        category=codes.decode("aircraft", "acft_category", row.get("acft_category")),
        far_part=far_part,
        far_part_desc=codes.decode("aircraft", "far_part", far_part),
        operation_type=codes.decode("aircraft", "type_fly", row.get("type_fly")),
        damage=codes.decode("aircraft", "damage", row.get("damage")),
        amateur_built=parse_flag(row.get("homebuilt")),
        num_engines=parse_int(row.get("num_eng")),
    )


def normalize_injury(row: Row, codes: CodeBook) -> Injury:
    category = row["inj_person_category"]
    level = row["injury_level"]
    return Injury(
        ev_id=row["ev_id"].strip(),
        aircraft_key=int(row["Aircraft_Key"]),
        person_category=codes.decode("injury", "inj_person_category", category) or category,
        injury_level=codes.decode("injury", "injury_level", level) or level,
        person_count=parse_int(row.get("inj_person_count")) or 0,
    )


def normalize_occurrence(row: Row, codes: CodeBook) -> Occurrence:
    phase_code = text_or_none(row.get("phase_no"))
    event_code = text_or_none(row.get("eventsoe_no"))
    return Occurrence(
        ev_id=row["ev_id"].strip(),
        aircraft_key=int(row["Aircraft_Key"]),
        occurrence_no=int(row["Occurrence_No"]),
        occurrence_code=text_or_none(row.get("Occurrence_Code")),
        phase_code=phase_code,
        phase=codes.decode("events_sequence", "phase", phase_code),
        event_code=event_code,
        event=codes.decode("events_sequence", "event", event_code),
        description=text_or_none(row.get("Occurrence_Description")),
        is_defining=parse_flag(row.get("Defining_ev")) is True,
    )


def normalize_finding(row: Row) -> Finding:
    cause_factor = {"C": "Cause", "F": "Factor"}.get((row.get("Cause_Factor") or "").strip())
    return Finding(
        ev_id=row["ev_id"].strip(),
        aircraft_key=int(row["Aircraft_Key"]),
        finding_no=int(row["finding_no"]),
        finding_code=text_or_none(row.get("finding_code")),
        description=text_or_none(row.get("finding_description")),
        cause_factor=cause_factor,
        in_probable_cause=parse_flag(row.get("cm_inPc")) is True,
    )


def normalize_narrative(row: Row) -> Narrative:
    # Column semantics verified against final reports: narr_accp is the factual narrative,
    # narr_accf the analysis ("summary released with the final report"), narr_cause the
    # probable cause statement. The dictionary's short descriptions are misleading here.
    return Narrative(
        ev_id=row["ev_id"].strip(),
        aircraft_key=int(row["Aircraft_Key"]),
        factual=clean_narrative(row.get("narr_accp")),
        analysis=clean_narrative(row.get("narr_accf")),
        probable_cause=clean_narrative(row.get("narr_cause")),
    )
