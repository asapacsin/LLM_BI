"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedQuery:
    raw_text: str
    intent: str
    metrics: list[str] = field(default_factory=list)
    department: str | None = None
    time_range: str | None = None
    aggregation: str | None = None
    confidence: float = 0.0


@dataclass
class InsightResponse:
    title: str
    summary: str
    bullets: list[str]
    confidence: float
    recommended_actions: list[str]
    intent: str
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "bullets": self.bullets,
            "confidence": self.confidence,
            "recommended_actions": self.recommended_actions,
            "intent": self.intent,
            "fallback": self.fallback,
            "metadata": self.metadata,
        }


@dataclass
class QueryFilters:
    department: str | None = None
    metric: str | None = None
    date_start: str | None = None
    date_end: str | None = None
