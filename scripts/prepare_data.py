"""One-time script to clean and export processed dataset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.clean import clean, export_processed  # noqa: E402


def main() -> None:
    df = clean()
    path = export_processed(df)
    print(f"Exported {len(df)} rows to {path}")
    print(f"Intent distribution:\n{df['intent'].value_counts()}")


if __name__ == "__main__":
    main()
