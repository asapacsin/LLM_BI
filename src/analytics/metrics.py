"""Pandas-based metric retrieval for LLM grounding."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.clean import load_processed
from src.models import ParsedQuery, QueryFilters


def build_metric_context(
    df: pd.DataFrame,
    parsed: ParsedQuery,
    filters: QueryFilters | None = None,
) -> dict[str, Any]:
    """Compute aggregates for the filtered subset."""
    filters = filters or QueryFilters()
    subset = df.copy()

    if filters.department:
        subset = subset[subset["department"] == filters.department]
    elif parsed.department:
        subset = subset[subset["department"] == parsed.department]

    if filters.metric:
        subset = subset[subset["metrics_requested"] == filters.metric]
    elif parsed.metrics:
        subset = subset[subset["metrics_requested"].isin(parsed.metrics)]

    if filters.date_start:
        subset = subset[subset["timestamp"] >= pd.to_datetime(filters.date_start)]
    if filters.date_end:
        subset = subset[subset["timestamp"] <= pd.to_datetime(filters.date_end)]

    if subset.empty:
        return {
            "row_count": 0,
            "warning": "No rows match filters; broaden department or date range.",
        }

    ctx: dict[str, Any] = {
        "row_count": int(len(subset)),
        "avg_confidence": round(float(subset["bot_response_confidence"].mean()), 3),
        "avg_latency_ms": round(float(subset["response_time_ms"].mean()), 1),
        "avg_feedback": round(float(subset["user_feedback_rating"].mean(skipna=True)), 2)
        if subset["user_feedback_rating"].notna().any()
        else None,
        "departments": subset["department"].value_counts().head(5).to_dict(),
        "metrics_breakdown": subset["metrics_requested"].value_counts().to_dict(),
        "analysis_types": subset["analysis_type"].value_counts().to_dict(),
        "impact_distribution": (
            subset["estimated_business_impact"].dropna().value_counts().to_dict()
            if subset["estimated_business_impact"].notna().any()
            else {}
        ),
        "roles": subset["user_role"].value_counts().to_dict(),
        "query_intent": parsed.intent,
        "time_range_hint": parsed.time_range,
    }

    if "timestamp" in subset.columns and subset["timestamp"].notna().any():
        monthly = (
            subset.set_index("timestamp")
            .resample("ME")["bot_response_confidence"]
            .mean()
            .dropna()
        )
        if len(monthly) > 0:
            ctx["monthly_avg_confidence"] = {
                str(k.date()): round(v, 3) for k, v in monthly.tail(6).items()
            }

    dept_conf = subset.groupby("department")["bot_response_confidence"].mean()
    ctx["confidence_by_department"] = {
        k: round(v, 3) for k, v in dept_conf.to_dict().items()
    }

    return ctx


class MetricsEngine:
    """Loads processed data and builds metric context."""

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df if df is not None else load_processed()
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

    def get_context(
        self, parsed: ParsedQuery, filters: QueryFilters | None = None
    ) -> dict[str, Any]:
        return build_metric_context(self.df, parsed, filters)
