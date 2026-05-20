"""Dataset path resolution and raw loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, KAGGLE_DATA_PATH, PROJECT_ROOT, RAW_DATA_REL


def resolve_data_path() -> Path:
    """Prefer local data dir, then Kaggle mount path."""
    local = DATA_DIR / RAW_DATA_REL
    if local.exists():
        return local
    kaggle = Path(KAGGLE_DATA_PATH)
    if kaggle.exists():
        return kaggle
    raise FileNotFoundError(
        f"Dataset not found. Expected local path: {local} or Kaggle: {KAGGLE_DATA_PATH}"
    )


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw BI chatbot interactions CSV."""
    csv_path = path or resolve_data_path()
    return pd.read_csv(csv_path, low_memory=False)
