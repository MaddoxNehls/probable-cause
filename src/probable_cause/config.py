"""Runtime settings, read from environment variables with local-dev defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATABASE_URL = "postgresql://pc:pc@localhost:5436/probable_cause"


@dataclass(frozen=True)
class Settings:
    database_url: str
    data_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def downloads_dir(self) -> Path:
        return self.raw_dir / "downloads"

    @property
    def extracted_dir(self) -> Path:
        return self.raw_dir / "extracted"

    @property
    def csv_dir(self) -> Path:
        return self.raw_dir / "csv"

    @property
    def pdf_dir(self) -> Path:
        return self.raw_dir / "pdf"


def load_settings() -> Settings:
    return Settings(
        database_url=os.environ.get("PC_DATABASE_URL", DEFAULT_DATABASE_URL),
        data_dir=Path(os.environ.get("PC_DATA_DIR", "data")),
    )
