"""Benchmark construction. Three item types, one JSONL file (benchmark/questions.jsonl).

* ``lookup``: an LLM writes a question about one event from its narrative, without the
  answer; the gold answer is that event's probable-cause statement. Questions that leak the
  event's identifiers, date, registration or the cause itself are filtered out.
* ``structured``: templated count/list questions with SQL ground truth, kept only when the
  answer set has 1-10 events that all have narrative text (so retrieval can find them).
* ``analytical``: hand-written questions in benchmark/analytical.yaml whose gold events
  come from SQL predicates written alongside them; includes unanswerable questions.
"""

import json
import random
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import psycopg
import yaml
from pydantic import BaseModel, ConfigDict

from probable_cause.llm import LLM, Message
from probable_cause.rag.filters import STATE_CODES
from probable_cause.rag.verify import content_terms

type Conn = psycopg.Connection[tuple[object, ...]]
type ItemType = Literal["lookup", "structured", "analytical"]

STATE_NAMES = {code: name.title() for name, code in STATE_CODES.items()}
FOREIGN_ASSIST = "e.ntsb_no ~ 'WA[0-9]+$'"


@dataclass
class Item:
    id: str
    type: ItemType
    question: str
    gold_events: list[str]
    gold_answer: str
    answerable: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


def write_items(path: Path, items: Iterable[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def read_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as fh:
        return [Item(**json.loads(line)) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# (a) lookup questions
# ---------------------------------------------------------------------------

LOOKUP_SYSTEM = """\
You write evaluation questions for a search system over NTSB aviation accident reports.

For each accident below, write ONE question that a pilot, student or safety analyst might
ask in order to learn why that specific accident happened. The question must:
- describe the accident by its distinguishing circumstances: aircraft type, phase of flight,
  what happened, and optionally the US state and the year;
- be answerable by the probable cause, but NOT state or hint at the cause, the findings, or
  the conclusion (do not say "due to fuel exhaustion" if fuel exhaustion was the cause);
- NOT contain the NTSB number, the event id, the registration (N-number), the exact date,
  or names of people;
- read naturally, in one sentence of 12-35 words."""


class _LookupQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str


class _LookupBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[_LookupQuestion]


_LOOKUP_EVENTS = f"""
SELECT e.ev_id, e.ntsb_no, e.ev_date, concat_ws(', ', e.city, e.state), e.highest_injury,
       a.make, a.model, a.registration, o.phase, o.event,
       n.analysis, n.factual, n.probable_cause
FROM events e
JOIN narratives n ON n.ev_id = e.ev_id AND n.aircraft_key = 1
JOIN aircraft a ON a.ev_id = e.ev_id AND a.aircraft_key = 1
LEFT JOIN occurrences o ON o.ev_id = e.ev_id AND o.aircraft_key = 1 AND o.is_defining
WHERE n.probable_cause IS NOT NULL AND length(n.probable_cause) >= 40
  AND coalesce(n.analysis, n.factual) IS NOT NULL
  AND NOT {FOREIGN_ASSIST} AND e.country = 'USA'
  AND (SELECT count(*) FROM aircraft x WHERE x.ev_id = e.ev_id) = 1
  AND extract(year FROM e.ev_date) = %s
ORDER BY md5(e.ev_id) LIMIT %s
"""


def _words(text: str | None, n: int) -> str:
    return " ".join((text or "").split()[:n])


def lookup_candidates(conn: Conn, per_year: int, years: range) -> list[dict[str, Any]]:
    out = []
    for year in years:
        for r in conn.execute(_LOOKUP_EVENTS, (year, per_year)):
            out.append(
                {
                    "ev_id": r[0], "ntsb_no": r[1], "date": r[2], "location": r[3],
                    "injury": r[4], "aircraft": f"{r[5]} {r[6]}", "registration": r[7],
                    "phase": r[8], "event": r[9], "analysis": r[10], "factual": r[11],
                    "probable_cause": r[12],
                }
            )  # fmt: skip
    return out


def leak_reasons(question: str, event: dict[str, Any]) -> list[str]:
    """Why a generated question gives away its answer or identity (empty list = clean)."""
    reasons = []
    q = question.lower()
    for key in ("ev_id", "ntsb_no", "registration"):
        value = event.get(key)
        if value and str(value).lower() in q:
            reasons.append(key)
    if re.search(r"\b[A-Z]{3}\d{2}[A-Z]{2}\d{3}\b", question):
        reasons.append("ntsb_number_pattern")
    if re.search(r"\bN\d{1,5}[A-Z]{0,2}\b", question):
        reasons.append("registration_pattern")
    d = event["date"]
    month_day = f"{d:%B} {d.day}".lower()
    if month_day in q or d.isoformat() in q or f"{d.month}/{d.day}/" in q:
        reasons.append("exact_date")
    cause = content_terms(event["probable_cause"])
    if cause:
        overlap = len(content_terms(question) & cause) / len(cause)
        if overlap >= 0.5:
            reasons.append(f"cause_overlap={overlap:.2f}")
    return reasons


def generate_lookup(
    llm: LLM, conn: Conn, *, per_year: int, years: range, batch_size: int = 20
) -> tuple[list[Item], list[dict[str, Any]]]:
    events = lookup_candidates(conn, per_year, years)
    items: list[Item] = []
    rejected: list[dict[str, Any]] = []
    for start in range(0, len(events), batch_size):
        batch = events[start : start + batch_size]
        blocks = [
            f"### Accident {i}\nDate: {e['date']:%B %Y}; location: {e['location']}; aircraft: "
            f"{e['aircraft']}; phase: {e['phase']}; defining event: {e['event']}; "
            f"highest injury: {e['injury']}\n"
            f"Narrative: {_words(e['analysis'] or e['factual'], 220)}"
            f"\nProbable cause (do not reveal): {e['probable_cause']}"
            for i, e in enumerate(batch)
        ]
        prompt = (
            "Write one question per accident; use the accident number as id.\n\n"
            + "\n\n".join(blocks)
        )
        result = llm.structured(
            f"lookup/{start}", LOOKUP_SYSTEM, [Message("user", prompt)], _LookupBatch
        )
        for q in result.items:
            if not q.id.strip().isdigit() or int(q.id) >= len(batch):
                continue
            event = batch[int(q.id)]
            reasons = leak_reasons(q.question, event)
            if reasons:
                rejected.append(
                    {"ev_id": event["ev_id"], "question": q.question, "reasons": reasons}
                )
                continue
            items.append(
                Item(
                    id="",
                    type="lookup",
                    question=q.question.strip(),
                    gold_events=[event["ev_id"]],
                    gold_answer=event["probable_cause"],
                    meta={"ntsb_no": event["ntsb_no"], "year": event["date"].year},
                )
            )
    return items, rejected


# ---------------------------------------------------------------------------
# (b) structured questions
# ---------------------------------------------------------------------------

PART_NAMES = {
    "121": "Part 121 (airline)",
    "135": "Part 135 (charter or commuter)",
    "137": "Part 137 (agricultural)",
}
CATEGORY_NAMES = {"Helicopter": "helicopter", "Glider": "glider", "Balloon": "balloon",
                  "Gyrocraft": "gyroplane", "Weight-Shift": "weight-shift trike"}  # fmt: skip
PHASE_NAMES = {300: "takeoff", 350: "initial climb", 400: "the enroute phase", 450: "maneuvering",
               500: "approach", 550: "landing"}  # fmt: skip
DEFINING_EVENTS = {
    "Fuel exhaustion": "ran out of fuel",
    "Fuel starvation": "suffered fuel starvation",
    "Midair collision": "were involved in a midair collision",
    "Aerodynamic stall/spin": "stalled or entered a spin",
    "Control flight into terr/obj": "flew into terrain under control (CFIT)",
    "Loss of engine power (total)": "lost all engine power",
    "Landing gear not configured": "landed with the landing gear not extended",
    "Fire/smoke (non-impact)": "had an in-flight fire",
}

_HAS_TEXT = "EXISTS (SELECT 1 FROM documents d WHERE d.ev_id = e.ev_id)"


@dataclass(frozen=True)
class Template:
    name: str
    kind: Literal["count", "list"]
    question: str
    where: str  # SQL over events e, parameters by name


TEMPLATES = (
    Template(
        "fatal_part", "count",
        "How many fatal {part_name} accidents were there in {year}?",
        "e.highest_injury = 'Fatal' AND e.event_type = 'Accident' "
        "AND EXISTS (SELECT 1 FROM aircraft a WHERE a.ev_id = e.ev_id AND a.far_part = %(part)s) "
        "AND extract(year FROM e.ev_date) = %(year)s",
    ),
    Template(
        "fatal_category_state", "list",
        "Which fatal {category_name} accidents happened in {state_name} in {year}?",
        "e.highest_injury = 'Fatal' AND e.state = %(state)s AND e.country = 'USA' "
        "AND EXISTS (SELECT 1 FROM aircraft a "
        "  WHERE a.ev_id = e.ev_id AND a.category = %(category)s) "
        "AND extract(year FROM e.ev_date) = %(year)s",
    ),
    Template(
        "night_imc_phase", "count",
        "How many airplane accidents in {year} happened during {phase_name} at night in "
        "instrument meteorological conditions?",
        "e.light_condition LIKE 'Night%%' AND e.weather_condition LIKE 'Instrument%%' "
        "AND EXISTS (SELECT 1 FROM aircraft a WHERE a.ev_id = e.ev_id AND a.category = 'Airplane') "
        "AND EXISTS (SELECT 1 FROM occurrences o WHERE o.ev_id = e.ev_id AND o.is_defining "
        "  AND o.phase_code ~ '^[0-9]+$' AND (o.phase_code::int / 50) * 50 = %(phase)s) "
        "AND extract(year FROM e.ev_date) = %(year)s",
    ),
    Template(
        "defining_event_state", "list",
        "Which aircraft that {event_phrase} crashed in {state_name} in {year}?",
        "e.state = %(state)s AND e.country = 'USA' "
        "AND EXISTS (SELECT 1 FROM occurrences o WHERE o.ev_id = e.ev_id AND o.is_defining "
        "  AND o.event = %(event)s) AND extract(year FROM e.ev_date) = %(year)s",
    ),
    Template(
        "part_phase_injury", "count",
        "How many {part_name} accidents in {year} with serious or fatal injuries occurred "
        "during {phase_name}?",
        "e.highest_injury IN ('Fatal', 'Serious') "
        "AND EXISTS (SELECT 1 FROM aircraft a WHERE a.ev_id = e.ev_id AND a.far_part = %(part)s) "
        "AND EXISTS (SELECT 1 FROM occurrences o WHERE o.ev_id = e.ev_id AND o.is_defining "
        "  AND o.phase_code ~ '^[0-9]+$' AND (o.phase_code::int / 50) * 50 = %(phase)s) "
        "AND extract(year FROM e.ev_date) = %(year)s",
    ),
)  # fmt: skip

_BIG_STATES = ("CA", "TX", "FL", "AK", "AZ", "CO", "WA", "GA", "NC", "MI", "NY", "OR", "ID",
               "PA", "OH", "UT", "NV", "MN", "WI", "TN")  # fmt: skip


def _sample_params(template: Template, rng: random.Random) -> dict[str, Any]:
    year = rng.randint(2018, 2024)
    if template.name == "fatal_part":
        part = rng.choice(list(PART_NAMES))
        return {"year": year, "part": part, "part_name": PART_NAMES[part]}
    if template.name == "fatal_category_state":
        category = rng.choice(list(CATEGORY_NAMES))
        state = rng.choice(_BIG_STATES)
        return {"year": year, "category": category, "category_name": CATEGORY_NAMES[category],
                "state": state, "state_name": STATE_NAMES[state]}  # fmt: skip
    if template.name == "night_imc_phase":
        phase = rng.choice([400, 450, 500, 550])
        return {"year": year, "phase": phase, "phase_name": PHASE_NAMES[phase]}
    if template.name == "defining_event_state":
        event = rng.choice(list(DEFINING_EVENTS))
        state = rng.choice(_BIG_STATES)
        return {"year": year, "event": event, "event_phrase": DEFINING_EVENTS[event],
                "state": state, "state_name": STATE_NAMES[state]}  # fmt: skip
    part = rng.choice(["121", "135", "137"])
    phase = rng.choice(list(PHASE_NAMES))
    return {"year": year, "part": part, "part_name": PART_NAMES[part], "phase": phase,
            "phase_name": PHASE_NAMES[phase]}  # fmt: skip


def generate_structured(conn: Conn, per_template: int, seed: int = 7) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []
    for template in TEMPLATES:
        seen: set[str] = set()
        kept = 0
        for _ in range(per_template * 40):
            if kept == per_template:
                break
            params = _sample_params(template, rng)
            question = template.question.format(**params)
            if question in seen:
                continue
            seen.add(question)
            rows = conn.execute(
                f"SELECT e.ev_id, e.ntsb_no, {_HAS_TEXT} FROM events e WHERE {template.where} "
                "ORDER BY e.ev_date",
                params,
            ).fetchall()
            if not 1 <= len(rows) <= 10 or not all(r[2] for r in rows):
                continue
            ids = [str(r[0]) for r in rows]
            numbers = ", ".join(str(r[1]) for r in rows)
            answer = f"{len(rows)} ({numbers})" if template.kind == "count" else numbers
            items.append(
                Item(id="", type="structured", question=question, gold_events=ids,
                     gold_answer=answer, meta={"template": template.name, "kind": template.kind,
                     "params": {k: v for k, v in params.items() if not k.endswith("_name")}})
            )  # fmt: skip
            kept += 1
    return items


# ---------------------------------------------------------------------------
# (c) analytical questions
# ---------------------------------------------------------------------------


def load_analytical(conn: Conn, path: Path) -> list[Item]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = []
    for entry in spec["questions"]:
        gold: list[str] = []
        if entry.get("gold_where"):
            rows = conn.execute(
                "SELECT e.ev_id FROM events e JOIN narratives n ON n.ev_id = e.ev_id "
                f"WHERE {entry['gold_where']} GROUP BY e.ev_id ORDER BY min(e.ev_date)"
            ).fetchall()
            gold = [str(r[0]) for r in rows]
        items.append(
            Item(id="", type="analytical", question=entry["question"].strip(),
                 gold_events=gold, gold_answer=entry["gold_answer"].strip(),
                 answerable=entry.get("answerable", True),
                 meta={"gold_where": entry.get("gold_where"), "gold_size": len(gold)})
        )  # fmt: skip
    return items


def number(items: list[Item]) -> list[Item]:
    prefix = {"lookup": "L", "structured": "S", "analytical": "A"}
    counters: dict[str, int] = {}
    for item in items:
        counters[item.type] = counters.get(item.type, 0) + 1
        item.id = f"{prefix[item.type]}{counters[item.type]:03d}"
    return items
