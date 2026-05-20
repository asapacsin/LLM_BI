"""Gold intent labels and stratified splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, QUERY_TO_INTENT
from src.data.clean import load_processed


def add_intent_labels(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ensure intent column exists and save labeled CSV."""
    if df is None:
        df = load_processed()
    df = df.copy()
    df["intent"] = df["user_query"].map(QUERY_TO_INTENT)
    if df["intent"].isna().any():
        unknown = df.loc[df["intent"].isna(), "user_query"].unique()
        raise ValueError(f"Unmapped queries: {unknown}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "interactions_labeled.csv"
    df.to_csv(out, index=False)
    return df


def train_holdout_split(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by intent."""
    if df is None:
        df = add_intent_labels()
    train, holdout = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["intent"],
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    holdout.to_csv(PROCESSED_DIR / "holdout.csv", index=False)
    return train, holdout
