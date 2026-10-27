import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from probable_cause.ingest import download

LISTING = """
<tr>
    <td id="fileName">avall.zip</td>
    <td id="fileDate">10/1/2026 7:36:33 AM</td>
    <td id="fileSize">96583215</td>
    <td><a href="/avdata/FileDirectory/DownloadFile?fileID=avall.zip">avall.zip</a></td>
</tr>
<tr>
    <td id="fileName">up15OCT.zip</td>
    <td id="fileDate">10/15/2026 8:49:18 PM</td>
    <td id="fileSize">1234</td>
</tr>
"""


def test_parse_listing() -> None:
    files = download.parse_listing(LISTING)
    assert set(files) == {"avall.zip", "up15OCT.zip"}
    avall = files["avall.zip"]
    assert avall.published == datetime(2026, 10, 1, 7, 36, 33)
    assert avall.size == 96583215
    assert avall.url.endswith("fileID=avall.zip")
    assert files["up15OCT.zip"].published.hour == 20


def test_is_current_requires_matching_listing_and_checksum(tmp_path: Path) -> None:
    local = tmp_path / "avall.zip"
    local.write_bytes(b"abc")
    listed = download.ListedFile("avall.zip", datetime(2026, 10, 1), 3)
    entry = download.ManifestEntry(
        name="avall.zip",
        url=listed.url,
        published=listed.published.isoformat(),
        size=3,
        sha256=download.sha256_file(local),
        downloaded_at="2026-10-27T00:00:00+00:00",
    )
    assert download.is_current(listed, entry, local)
    assert not download.is_current(listed, None, local)
    newer = download.ListedFile("avall.zip", datetime(2026, 11, 1), 3)
    assert not download.is_current(newer, entry, local)
    local.write_bytes(b"abd")  # same size, different content
    assert not download.is_current(listed, entry, local)


def test_manifest_round_trip(tmp_path: Path) -> None:
    entry = download.ManifestEntry("a", "u", "p", 1, "s", "d")
    download.write_manifest(tmp_path, {"a": entry})
    assert download.read_manifest(tmp_path) == {"a": entry}


def test_extract_mdb_ignores_archive_paths(tmp_path: Path) -> None:
    archive = tmp_path / "avall.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escape/avall.mdb", b"jet")
        zf.writestr("readme.txt", b"hi")
    target = download.extract_mdb(archive, tmp_path / "extracted", "f" * 64)
    assert target == tmp_path / "extracted" / "avall-ffffffffffff" / "avall.mdb"
    assert target.read_bytes() == b"jet"
    assert not (tmp_path / "escape").exists()


def test_extract_mdb_rejects_unexpected_contents(tmp_path: Path) -> None:
    archive = tmp_path / "avall.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.mdb", b"1")
        zf.writestr("b.mdb", b"2")
    with pytest.raises(ValueError, match="exactly one"):
        download.extract_mdb(archive, tmp_path / "extracted", "0" * 64)
