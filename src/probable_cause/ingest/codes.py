"""Decode NTSB coded fields using the data dictionary shipped inside avall.mdb.

The eADMSPUB_DataDictionary table maps (table, column, code) to a meaning. It is not
complete: a handful of codes that appear in the data are missing from it, and the
injury table uses truncated person categories ("Flig", "Pass") that the dictionary
never lists. Those gaps are filled from SUPPLEMENTAL_CODES, each entry checked by hand
against Occurrence_Description text or the NTSB codman.pdf.
"""

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

type FieldKey = tuple[str, str]

SUPPLEMENTAL_CODES: dict[FieldKey, dict[str, str]] = {
    ("aircraft", "acft_category"): {
        "WSFT": "Weight-Shift",
        "PPAR": "Powered Parachute",
        "RCKT": "Rocket",
    },
    ("aircraft", "far_part"): {
        "107": "Part 107: Small Unmanned Aircraft",
        "437": "Part 437: Commercial Space Experimental Permit",
    },
    ("aircraft", "type_fly"): {
        "PUBF": "Public Aircraft - Federal",
        "PUBL": "Public Aircraft - Local",
        "PUBS": "Public Aircraft - State",
    },
    ("injury", "inj_person_category"): {
        "FLIG": "Flight crew",
        "PASS": "Passengers",
        "CABI": "Cabin crew",
        "LAPC": "Lap child",
    },
    ("events_sequence", "phase"): {
        "553": "Landing-Aborted after touchdown",
        "601": "Autorotation",
    },
    ("events_sequence", "event"): {
        "281": "Course deviation",
        "282": "Altitude deviation",
        "283": "Airspace incursion",
        "284": "Wrong surface or wrong airport",
        "850": "Medical event",
    },
}


@dataclass
class CodeBook:
    """Lookup of code meanings keyed by lower-cased (table, column)."""

    tables: dict[FieldKey, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dictionary_rows(cls, rows: Iterable[Mapping[str, str]]) -> Self:
        tables: defaultdict[FieldKey, dict[str, str]] = defaultdict(dict)
        for row in rows:
            code = (row.get("code_iaids") or "").strip().upper()
            meaning = (row.get("meaning") or "").strip()
            if not code or not meaning:
                continue
            key = (
                (row.get("Table") or "").strip().lower(),
                (row.get("Column") or "").strip().lower(),
            )
            if key == ("events_sequence", "occurrence_code"):
                # Occurrence codes are documented as two 3-digit halves: "PPPxxx" / "xxxEEE".
                if code.endswith("XXX"):
                    tables[("events_sequence", "phase")][code[:3]] = meaning
                elif code.startswith("XXX"):
                    tables[("events_sequence", "event")][code[3:]] = meaning
                continue
            tables[key][code] = meaning
        for key, extra in SUPPLEMENTAL_CODES.items():
            for code, meaning in extra.items():
                tables[key].setdefault(code, meaning)
        return cls(dict(tables))

    @classmethod
    def from_csv(cls, path: Path) -> Self:
        with path.open(encoding="utf-8", newline="") as fh:
            return cls.from_dictionary_rows(csv.DictReader(fh))

    def decode(self, table: str, column: str, code: str | None) -> str | None:
        """Return the meaning of `code`, the raw code if it is unknown, or None if blank."""
        if code is None or not code.strip():
            return None
        normalized = code.strip().upper()
        return self.tables.get((table.lower(), column.lower()), {}).get(normalized, code.strip())

    def is_known(self, table: str, column: str, code: str) -> bool:
        return code.strip().upper() in self.tables.get((table.lower(), column.lower()), {})
