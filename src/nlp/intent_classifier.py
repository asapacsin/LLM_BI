"""Intent classification via rules baseline and OpenAI few-shot (sync + async batch)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.config import (
    EVAL_CONCURRENCY,
    INTENT_CATEGORIES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    QUERY_TO_INTENT,
)
from src.llm_client import create_async_llm_client, create_llm_client

CLASSIFY_SYSTEM = """You classify BI chatbot user queries into exactly one intent.
Respond with JSON only: {"intent": "<snake_case_intent>", "confidence": 0.0-1.0}

Valid intents:
- performance_monitoring
- anomaly_detection
- forecasting
- comparative_analysis
- operational_optimization
"""


def classify_intent_rules(query: str) -> tuple[str, float]:
    """Deterministic classifier using template map and keywords."""
    q = query.strip()
    if q in QUERY_TO_INTENT:
        return QUERY_TO_INTENT[q], 1.0

    lower = q.lower()
    if any(w in lower for w in ("forecast", "predict", "next quarter", "projection")):
        return "forecasting", 0.85
    if any(w in lower for w in ("compare", "versus", "vs", "by region", "benchmark")):
        return "comparative_analysis", 0.85
    if any(w in lower for w in ("roi", "campaign", "optimize", "efficiency", "cost")):
        return "operational_optimization", 0.85
    if any(w in lower for w in ("anomaly", "outlier", "unusual", "spike", "drop")):
        return "anomaly_detection", 0.85
    if any(w in lower for w in ("trend", "churn", "kpi", "monitor", "rate", "monthly")):
        return "performance_monitoring", 0.8
    return "performance_monitoring", 0.5


def _build_classify_messages(query: str) -> list[dict[str, str]]:
    examples = "\n".join(
        f'- "{q}" -> {intent}' for q, intent in QUERY_TO_INTENT.items()
    )
    user_msg = f"Examples:\n{examples}\n\nQuery: {query}"
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _parse_classify_json(content: str) -> tuple[str, float]:
    data = json.loads(content or "{}")
    intent = str(data.get("intent", "performance_monitoring")).lower().replace(" ", "_")
    confidence = float(data.get("confidence", 0.7))
    return intent, min(max(confidence, 0.0), 1.0)


def _finalize_result(
    query: str,
    intent: str,
    confidence: float,
    method: str,
    latency_ms: float,
) -> dict[str, Any]:
    if intent not in INTENT_CATEGORIES:
        intent, confidence = classify_intent_rules(query)
        method = "rules_corrected"
    return {
        "intent": intent,
        "confidence": confidence,
        "method": method,
        "latency_ms": latency_ms,
    }


class IntentClassifier:
    """OpenAI-backed intent classifier with rules fallback."""

    def __init__(self, use_openai: bool = True) -> None:
        self.use_openai = use_openai and bool(OPENAI_API_KEY)
        self._client = create_llm_client() if self.use_openai else None

    def classify(self, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        if self.use_openai and self._client:
            try:
                intent, confidence = self._classify_openai_sync(query)
                method = "openai"
            except Exception:
                intent, confidence = classify_intent_rules(query)
                method = "rules_fallback"
        else:
            intent, confidence = classify_intent_rules(query)
            method = "rules"

        latency_ms = (time.perf_counter() - start) * 1000
        return _finalize_result(query, intent, confidence, method, latency_ms)

    def _classify_openai_sync(self, query: str) -> tuple[str, float]:
        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=_build_classify_messages(query),
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return _parse_classify_json(content)


async def _classify_openai_async(client: Any, query: str) -> tuple[str, float, str]:
    """Returns intent, confidence, method."""
    try:
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=_build_classify_messages(query),
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        intent, confidence = _parse_classify_json(content)
        return intent, confidence, "openai"
    except Exception:
        intent, confidence = classify_intent_rules(query)
        return intent, confidence, "rules_fallback"


async def classify_batch_async(
    queries: list[str],
    concurrency: int | None = None,
    dedupe: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Classify many queries concurrently.

    Returns (results in input order, stats dict with api_calls, wall_ms, dedupe).
    """
    if not queries:
        return [], {"api_calls": 0, "wall_ms": 0.0, "dedupe": dedupe}

    if not OPENAI_API_KEY:
        results = []
        for q in queries:
            intent, conf = classify_intent_rules(q)
            results.append(_finalize_result(q, intent, conf, "rules", 0.0))
        return results, {"api_calls": 0, "wall_ms": 0.0, "dedupe": dedupe}

    client = create_async_llm_client()
    if client is None:
        raise RuntimeError("Async LLM client could not be created")

    concurrency = concurrency or EVAL_CONCURRENCY
    sem = asyncio.Semaphore(concurrency)

    if dedupe:
        unique_queries = list(dict.fromkeys(queries))
    else:
        unique_queries = list(queries)

    async def one(q: str) -> dict[str, Any]:
        async with sem:
            t0 = time.perf_counter()
            intent, confidence, method = await _classify_openai_async(client, q)
            latency_ms = (time.perf_counter() - t0) * 1000
            return _finalize_result(q, intent, confidence, method, latency_ms)

    wall_start = time.perf_counter()
    unique_results = await asyncio.gather(*[one(q) for q in unique_queries])
    wall_ms = (time.perf_counter() - wall_start) * 1000

    cache_by_query = {q: r for q, r in zip(unique_queries, unique_results)}
    ordered = [cache_by_query[q] for q in queries]

    stats = {
        "api_calls": len(unique_queries),
        "wall_ms": wall_ms,
        "dedupe": dedupe,
        "concurrency": concurrency,
        "total_rows": len(queries),
    }
    return ordered, stats


def classify_batch_sync(
    queries: list[str],
    concurrency: int | None = None,
    dedupe: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sync entrypoint for async batch classification."""
    return asyncio.run(classify_batch_async(queries, concurrency, dedupe))
