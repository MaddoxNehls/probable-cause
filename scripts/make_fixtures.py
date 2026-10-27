"""Cut a tiny, anonymized fixture set out of a full mdb-export CSV directory.

Usage: uv run python scripts/make_fixtures.py data/raw/csv/avall-<sha> tests/fixtures/avall
"""

import csv
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)

# Chosen to cover: a two-aircraft midair (old-style id), a post-2020 id with a CAROL key,
# an aircraft category missing from the data dictionary (WSFT), a short-form report whose
# analysis repeats the factual text, and a foreign-assist event with no narrative.
EVENT_IDS = (
    "20080107X00026",
    "20210102102480",
    "20080710X01009",
    "20080210X00166",
    "20231009193207",
)
# Free-text columns that can identify private individuals are blanked.
PII_COLUMNS = {
    "owner_acft",
    "owner_street",
    "owner_city",
    "owner_zip",
    "oper_name",
    "oper_dba",
    "oper_street",
    "oper_city",
    "oper_zip",
    "oper_cert_num",
    "acft_serial_no",
    "regis_no",
    "lchg_userid",
    "elt_reason_other",
}
NARRATIVE_LIMIT = 600
DICTIONARY_FIELDS = {
    ("events", "ev_type"),
    ("events", "ev_highest_injury"),
    ("events", "light_cond"),
    ("events", "wx_cond_basic"),
    ("aircraft", "acft_category"),
    ("aircraft", "far_part"),
    ("aircraft", "type_fly"),
    ("aircraft", "damage"),
    ("aircraft", "homebuilt"),
    ("injury", "inj_person_category"),
    ("injury", "injury_level"),
    ("events_sequence", "occurrence_code"),
}


def main(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("events", "aircraft", "injury", "Events_Sequence", "Findings", "narratives"):
        with (src / f"{name}.csv").open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            rows = [r for r in reader if r["ev_id"] in EVENT_IDS]
        for row in rows:
            for col in PII_COLUMNS & row.keys():
                row[col] = ""
            if name == "aircraft":
                row["regis_no"] = f"N{EVENT_IDS.index(row['ev_id']) + 1:03d}PC"
            if name == "narratives":
                for col in ("narr_accp", "narr_accf", "narr_cause"):
                    row[col] = row[col][:NARRATIVE_LIMIT]
        write(dest / f"{name}.csv", fieldnames, rows)

    with (src / "eADMSPUB_DataDictionary.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if (r["Table"].lower(), r["Column"].lower()) in DICTIONARY_FIELDS]
    keep = ("Table", "Column", "code_iaids", "meaning")
    write(dest / "eADMSPUB_DataDictionary.csv", [c for c in fieldnames if c in keep], rows)


def write(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
