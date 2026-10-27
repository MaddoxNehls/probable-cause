"""Export Access tables to CSV with mdbtools, run inside a network-less container.

mdbtools is the reference open-source Jet reader. The pure-Python alternative
(access-parser) silently mangles Jet4 compressed Unicode: smart quotes come out as
control characters in about 17% of narratives. See docs/data-source.md.
"""

import subprocess
from pathlib import Path

TOOLS_IMAGE = "probable-cause-tools:1"
TOOLS_DOCKERFILE_DIR = Path(__file__).resolve().parents[3] / "docker" / "tools"

# Tables the loader consumes. Names are case-sensitive in the .mdb.
EXPORT_TABLES = (
    "events",
    "aircraft",
    "injury",
    "Events_Sequence",
    "Findings",
    "narratives",
    "eADMSPUB_DataDictionary",
)


def ensure_tools_image() -> None:
    probe = subprocess.run(
        ["docker", "image", "inspect", TOOLS_IMAGE], capture_output=True, check=False
    )
    if probe.returncode != 0:
        subprocess.run(
            ["docker", "build", "-t", TOOLS_IMAGE, str(TOOLS_DOCKERFILE_DIR)], check=True
        )


def sandboxed_run(mounts: dict[Path, tuple[str, bool]], script: str) -> None:
    """Run a shell script in the tools container with no network access.

    `mounts` maps host paths to (container path, read_only).
    """
    args = ["docker", "run", "--rm", "--network", "none"]
    for host, (container, read_only) in mounts.items():
        spec = f"{host.resolve()}:{container}" + (":ro" if read_only else "")
        args += ["-v", spec]
    args += [TOOLS_IMAGE, "sh", "-c", script]
    subprocess.run(args, check=True)


def export_tables(
    mdb_path: Path, out_dir: Path, tables: tuple[str, ...] = EXPORT_TABLES
) -> list[Path]:
    """Export each table to out_dir/<table>.csv with ISO dates. Skips tables already exported."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tables if not (out_dir / f"{t}.csv").exists()]
    if todo:
        ensure_tools_image()
        # Write to .part and rename so an interrupted export is never mistaken for a complete one.
        commands = [
            f"mdb-export -D '%Y-%m-%d' -T '%Y-%m-%d %H:%M:%S' /in/{mdb_path.name} {t}"
            f" > /out/{t}.csv.part && mv /out/{t}.csv.part /out/{t}.csv"
            for t in todo
        ]
        sandboxed_run(
            {mdb_path.parent: ("/in", True), out_dir: ("/out", False)},
            "set -e; " + "; ".join(commands),
        )
    return [out_dir / f"{t}.csv" for t in tables]
