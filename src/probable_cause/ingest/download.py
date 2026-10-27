"""Idempotent download of the NTSB avdata bulk files.

The avdata server sends neither ETag nor a meaningful Last-Modified, so the cache key
is the (published date, size) pair from the directory listing, and every file on
disk is verified against the SHA-256 recorded in a local manifest.
"""

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

import httpx

AVDATA_URL = "https://data.ntsb.gov/avdata"
DOWNLOAD_URL = AVDATA_URL + "/FileDirectory/DownloadFile?fileID={name}"
DEFAULT_FILES = ("avall.zip", "eadmspub.pdf", "MDB_Release_Notes.pdf")
USER_AGENT = "probable-cause/0.1 (+https://github.com/; research ingest)"
MANIFEST_NAME = "manifest.json"

_ROW_RE = re.compile(
    r'<td id="fileName">(?P<name>[^<]+)</td>\s*'
    r'<td id="fileDate">(?P<date>[^<]+)</td>\s*'
    r'<td id="fileSize">(?P<size>\d+)</td>'
)


@dataclass(frozen=True)
class ListedFile:
    name: str
    published: datetime
    size: int

    @property
    def url(self) -> str:
        return DOWNLOAD_URL.format(name=self.name)


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    url: str
    published: str
    size: int
    sha256: str
    downloaded_at: str


def parse_listing(html: str) -> dict[str, ListedFile]:
    """Parse the avdata directory page into {file name: ListedFile}."""
    files: dict[str, ListedFile] = {}
    for m in _ROW_RE.finditer(html):
        name = unescape(m["name"]).strip()
        published = datetime.strptime(m["date"].strip(), "%m/%d/%Y %I:%M:%S %p")
        files[name] = ListedFile(name=name, published=published, size=int(m["size"]))
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(downloads_dir: Path) -> dict[str, ManifestEntry]:
    path = downloads_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: ManifestEntry(**entry) for name, entry in raw.items()}


def write_manifest(downloads_dir: Path, manifest: dict[str, ManifestEntry]) -> None:
    path = downloads_dir / MANIFEST_NAME
    payload = {name: asdict(entry) for name, entry in sorted(manifest.items())}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def is_current(listed: ListedFile, entry: ManifestEntry | None, local: Path) -> bool:
    """True when the local copy matches the listing and its recorded checksum."""
    return (
        entry is not None
        and entry.published == listed.published.isoformat()
        and entry.size == listed.size
        and local.exists()
        and local.stat().st_size == listed.size
        and sha256_file(local) == entry.sha256
    )


def download_file(client: httpx.Client, listed: ListedFile, dest: Path) -> str:
    """Stream to a .part file, verify the size, then atomically move into place."""
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with client.stream("GET", listed.url) as response, part.open("wb") as fh:
        response.raise_for_status()
        for chunk in response.iter_bytes(1 << 20):
            fh.write(chunk)
            digest.update(chunk)
    actual = part.stat().st_size
    if actual != listed.size:
        part.unlink()
        raise RuntimeError(f"{listed.name}: expected {listed.size} bytes, got {actual}")
    part.replace(dest)
    return digest.hexdigest()


def sync(downloads_dir: Path, names: tuple[str, ...] = DEFAULT_FILES) -> list[tuple[str, str]]:
    """Bring the requested files up to date. Returns (name, 'cached' | 'downloaded') pairs."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(downloads_dir)
    results: list[tuple[str, str]] = []
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(60, read=300)) as client:
        listing_response = client.get(AVDATA_URL)
        listing_response.raise_for_status()
        listing = parse_listing(listing_response.text)
        for name in names:
            if name not in listing:
                raise KeyError(f"{name} is not in the avdata listing")
            listed = listing[name]
            local = downloads_dir / name
            if is_current(listed, manifest.get(name), local):
                results.append((name, "cached"))
                continue
            sha = download_file(client, listed, local)
            manifest[name] = ManifestEntry(
                name=name,
                url=listed.url,
                published=listed.published.isoformat(),
                size=listed.size,
                sha256=sha,
                downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            write_manifest(downloads_dir, manifest)
            results.append((name, "downloaded"))
    return results


def extract_mdb(archive: Path, extracted_root: Path, sha256: str) -> Path:
    """Extract the single .mdb from an avdata zip into a fresh, checksum-named directory.

    The archive is untrusted: only plain .mdb members are extracted, by basename, so
    path traversal entries are impossible.
    """
    target_dir = extracted_root / f"{archive.stem}-{sha256[:12]}"
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        mdbs = [m for m in members if m.filename.lower().endswith(".mdb")]
        if len(mdbs) != 1:
            raise ValueError(f"expected exactly one .mdb in {archive.name}, found {len(mdbs)}")
        member = mdbs[0]
        target = target_dir / Path(member.filename).name
        if target.exists() and target.stat().st_size == member.file_size:
            return target
        target_dir.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".part")
        with zf.open(member) as src, part.open("wb") as dst:
            while block := src.read(1 << 20):
                dst.write(block)
        part.replace(target)
    return target
