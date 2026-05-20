"""Data cleaning and processed export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, QUERY_TO_INTENT
from src.data.load import load_raw


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean and normalize the interaction dataset."""
    if df is None:
        df = load_raw().copy()
    else:
        df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["department"] = df["department"].str.strip()
    df["user_role"] = df["user_role"].str.strip()
    df["user_query"] = df["user_query"].str.strip()
    df["query_category"] = df["query_category"].str.strip()
    df["metrics_requested"] = df["metrics_requested"].str.strip()
    df["analysis_type"] = df["analysis_type"].str.strip()

    df["estimated_business_impact"] = (
        df["estimated_business_impact"]
        .astype(str)
        .str.strip()
        .replace({"nan": pd.NA, "": pd.NA})
    )
    df.loc[df["estimated_business_impact"].isin(["nan", "None"]), "estimated_business_impact"] = (
        pd.NA
    )

    df["user_feedback_rating"] = pd.to_numeric(df["user_feedback_rating"], errors="coerce")
    df["bot_response_confidence"] = pd.to_numeric(
        df["bot_response_confidence"], errors="coerce"
    )
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")

    df["intent"] = df["user_query"].map(QUERY_TO_INTENT)
    missing_intent = df["intent"].isna().sum()
    if missing_intent:
        raise ValueError(f"Unmapped user_query values: {missing_intent} rows")

    df["feedback_missing"] = df["user_feedback_rating"].isna()
    df["impact_missing"] = df["estimated_business_impact"].isna()

    return df


def export_processed(df: pd.DataFrame | None = None) -> Path:
    """Write cleaned data to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = clean()
    parquet_path = PROCESSED_DIR / "interactions_clean.parquet"
    csv_path = PROCESSED_DIR / "interactions_clean.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path


def load_processed() -> pd.DataFrame:
    """Load cleaned dataset from processed dir."""
    parquet_path = PROCESSED_DIR / "interactions_clean.parquet"
    csv_path = PROCESSED_DIR / "interactions_clean.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    export_processed()
    return pd.read_parquet(parquet_path)
