"""Orchestrate parse → classify → metrics → insight → persist."""

from __future__ import annotations

import time
from typing import Any

from src.analytics.metrics import MetricsEngine
from src.models import InsightResponse, QueryFilters
from src.nlp.parser import QueryParser
from src.reasoning.engine import ReasoningEngine
from src.state.store import StateStore


class Pipeline:
    """End-to-end BI decision support pipeline."""

    def __init__(
        self,
        store: StateStore | None = None,
        parser: QueryParser | None = None,
        metrics: MetricsEngine | None = None,
        reasoning: ReasoningEngine | None = None,
    ) -> None:
        self.store = store or StateStore()
        self.parser = parser or QueryParser()
        self.metrics = metrics or MetricsEngine()
        self.reasoning = reasoning or ReasoningEngine()

    def run_query(
        self,
        text: str,
        session_id: str | None = None,
        filters: QueryFilters | None = None,
    ) -> dict[str, Any]:
        if not session_id:
            session_id = self.store.create_session(title=text[:40])

        start = time.perf_counter()
        self.store.append_message(session_id, "user", text)

        parsed = self.parser.parse(
            text,
            department_hint=filters.department if filters else None,
        )
        metric_context = self.metrics.get_context(parsed, filters)
        insight = self.reasoning.generate(parsed, metric_context)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.store.log_run(
            session_id,
            text,
            parsed.intent,
            elapsed_ms,
            metadata={
                "confidence": parsed.confidence,
                "row_count": metric_context.get("row_count"),
                "fallback": insight.fallback,
            },
        )
        self.store.append_message(
            session_id,
            "assistant",
            insight.summary,
            metadata=insight.to_dict(),
        )

        return {
            "session_id": session_id,
            "parsed": parsed,
            "metric_context": metric_context,
            "insight": insight,
            "latency_ms": elapsed_ms,
        }


def run_query(
    text: str,
    session_id: str | None = None,
    filters: QueryFilters | None = None,
) -> dict[str, Any]:
    """Convenience wrapper."""
    return Pipeline().run_query(text, session_id, filters)
