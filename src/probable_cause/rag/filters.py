"""Structured filters, the query plan an LLM rewrite produces, and the SQL that applies them.

The vocabulary mirrors the decoded values in the database, so a validated plan can be
applied without any further mapping. Free-form values the model is likely to produce
("Part 91", "fatal", "ohio", "night") are normalized before validation.
"""

import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AircraftCategory = Literal[
    "Airplane", "Helicopter", "Glider", "Balloon", "Gyrocraft", "Weight-Shift",
    "Powered Parachute", "Ultralight", "Powered-Lift",
]  # fmt: skip
FarPart = Literal["091", "091K", "103", "107", "121", "125", "129", "133", "135", "137"]
# Coarse phase of flight of the *defining* occurrence (NTSB phase codes rounded to 50).
Phase = Literal[
    "Standing", "Taxi", "Takeoff", "Initial Climb", "Enroute", "Maneuvering", "Approach",
    "Landing", "Emergency Descent", "Uncontrolled Descent",
]  # fmt: skip
PHASE_CODES: dict[str, int] = {
    "Standing": 150, "Taxi": 250, "Takeoff": 300, "Initial Climb": 350, "Enroute": 400,
    "Maneuvering": 450, "Approach": 500, "Landing": 550, "Emergency Descent": 600,
    "Uncontrolled Descent": 650,
}  # fmt: skip
Severity = Literal["Fatal", "Serious", "Minor", "None"]

US_STATES = frozenset(
    [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "DC",
        "PR",
        "GU",
        "VI",
        "AS",
        "MP",
    ]
)
STATE_CODES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "puerto rico": "PR", "district of columbia": "DC",
}  # fmt: skip


def _as_list(value: Any) -> Any:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year_from: int | None = Field(None, ge=1982, le=2100)
    year_to: int | None = Field(None, ge=1982, le=2100)
    aircraft_categories: list[AircraftCategory] = Field(default_factory=list)
    far_parts: list[FarPart] = Field(default_factory=list)
    phases: list[Phase] = Field(default_factory=list)
    injury_severities: list[Severity] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    weather: Literal["VMC", "IMC"] | None = None
    night: bool | None = None
    aircraft_make: str | None = None

    @field_validator("far_parts", mode="before")
    @classmethod
    def _parts(cls, value: Any) -> Any:
        out = []
        for item in _as_list(value):
            text = str(item).upper().replace("PART", "").replace("CFR", "").strip()
            match = re.fullmatch(r"(\d{2,3})(K?)", text.replace(" ", ""))
            out.append(f"{int(match[1]):03d}{match[2]}" if match else item)
        return out

    @field_validator("injury_severities", mode="before")
    @classmethod
    def _severities(cls, value: Any) -> Any:
        aliases = {"fatal": "Fatal", "fatality": "Fatal", "fatalities": "Fatal",
                   "serious": "Serious", "minor": "Minor", "none": "None"}  # fmt: skip
        return [aliases.get(str(v).strip().lower(), v) for v in _as_list(value)]

    @field_validator("aircraft_categories", "phases", mode="before")
    @classmethod
    def _title_case(cls, value: Any) -> Any:
        fixed = {"weight-shift": "Weight-Shift", "powered-lift": "Powered-Lift",
                 "airplanes": "Airplane", "helicopters": "Helicopter"}  # fmt: skip
        out = []
        for v in _as_list(value):
            key = str(v).strip().lower()
            out.append(fixed.get(key, key.title().replace("Initial climb", "Initial Climb")))
        return out

    @field_validator("states", mode="before")
    @classmethod
    def _states(cls, value: Any) -> list[str]:
        out = []
        for v in _as_list(value):
            text = str(v).strip()
            code = STATE_CODES.get(text.lower(), text.upper())
            if code not in US_STATES:
                raise ValueError(f"unknown US state {v!r}; use two-letter codes")
            out.append(code)
        return out

    @field_validator("aircraft_make", mode="before")
    @classmethod
    def _make(cls, value: Any) -> Any:
        return str(value).strip() or None if value is not None else None

    @model_validator(mode="after")
    def _ordered_years(self) -> Self:
        if self.year_from and self.year_to and self.year_from > self.year_to:
            self.year_from, self.year_to = self.year_to, self.year_from
        return self

    def is_empty(self) -> bool:
        return self == Filters()


class QueryPlan(BaseModel):
    """What a rewrite produces: filters, plus the text that should be matched semantically."""

    model_config = ConfigDict(extra="forbid")

    text_query: str = Field(description="The topical part of the question, without the filters")
    filters: Filters


def filter_sql(filters: Filters) -> tuple[str, dict[str, Any]]:
    """A query returning the ev_ids that satisfy every filter."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if filters.year_from:
        clauses.append("e.ev_date >= make_date(%(year_from)s, 1, 1)")
        params["year_from"] = filters.year_from
    if filters.year_to:
        clauses.append("e.ev_date <= make_date(%(year_to)s, 12, 31)")
        params["year_to"] = filters.year_to
    if filters.injury_severities:
        clauses.append("e.highest_injury = ANY(%(severities)s)")
        params["severities"] = list(filters.injury_severities)
    if filters.states:
        clauses.append("e.state = ANY(%(states)s) AND e.country = 'USA'")
        params["states"] = filters.states
    if filters.weather:
        clauses.append("e.weather_condition LIKE %(weather)s")
        params["weather"] = ("Visual" if filters.weather == "VMC" else "Instrument") + "%"
    if filters.night is not None:
        op = "" if filters.night else "NOT "
        clauses.append(f"{op}coalesce(e.light_condition LIKE 'Night%%', false)")
    aircraft: list[str] = []
    if filters.aircraft_categories:
        aircraft.append("a.category = ANY(%(categories)s)")
        params["categories"] = list(filters.aircraft_categories)
    if filters.far_parts:
        aircraft.append("a.far_part = ANY(%(parts)s)")
        params["parts"] = list(filters.far_parts)
    if filters.aircraft_make:
        aircraft.append("a.make ILIKE %(make)s")
        params["make"] = filters.aircraft_make.replace("%", "") + "%"
    if aircraft:
        clauses.append(
            "EXISTS (SELECT 1 FROM aircraft a WHERE a.ev_id = e.ev_id AND "
            + " AND ".join(aircraft)
            + ")"
        )
    if filters.phases:
        clauses.append(
            "EXISTS (SELECT 1 FROM occurrences o WHERE o.ev_id = e.ev_id AND o.is_defining "
            "AND o.phase_code ~ '^[0-9]+$' AND (o.phase_code::int / 50) * 50 = ANY(%(phases)s))"
        )
        params["phases"] = [PHASE_CODES[p] for p in filters.phases]
    where = " AND ".join(clauses) or "true"
    return f"SELECT e.ev_id FROM events e WHERE {where}", params
