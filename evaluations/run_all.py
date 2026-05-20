"""Run all evaluation suites and write summary report."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import OPENAI_API_KEY, EVAL_CONCURRENCY  # noqa: E402
from src.models import QueryFilters  # noqa: E402
from src.nlp.intent_classifier import classify_batch_sync, classify_intent_rules  # noqa: E402
from src.pipeline.orchestrator import Pipeline  # noqa: E402

from evaluations.intent_eval import run_evaluation  # noqa: E402


async def _run_pipeline_queries_async(queries: list[str]) -> list[float]:
    """Run pipeline queries concurrently (each query = classify + insight)."""
    pipeline = Pipeline()
    sem = asyncio.Semaphore(EVAL_CONCURRENCY)

    async def one(q: str) -> float:
        async with sem:
            t0 = time.perf_counter()
            await asyncio.to_thread(
                pipeline.run_query,
                q,
                None,
                QueryFilters(),
            )
            return (time.perf_counter() - t0) * 1000

    return list(await asyncio.gather(*[one(q) for q in queries]))


def latency_eval() -> dict:
    samples = json.loads((ROOT / "evaluations" / "sample_queries.json").read_text())
    queries = [item["query"] for item in samples[:10]]
    t0 = time.perf_counter()
    if OPENAI_API_KEY:
        latencies = asyncio.run(_run_pipeline_queries_async(queries))
    else:
        pipeline = Pipeline()
        latencies = []
        for q in queries:
            t1 = time.perf_counter()
            pipeline.run_query(q, filters=QueryFilters())
            latencies.append((time.perf_counter() - t1) * 1000)
    total_wall = (time.perf_counter() - t0) * 1000
    latencies.sort()
    n = len(latencies)
    return {
        "count": n,
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)] if n > 1 else latencies[0],
        "mean_ms": sum(latencies) / n,
        "wall_ms": total_wall,
        "async": bool(OPENAI_API_KEY),
    }


def paraphrase_intent_eval() -> dict:
    samples = json.loads((ROOT / "evaluations" / "sample_queries.json").read_text())
    queries = [item["query"] for item in samples]
    expected = [item["expected_intent"] for item in samples]

    if OPENAI_API_KEY:
        results, stats = classify_batch_sync(queries, dedupe=True)
        preds = [r["intent"] for r in results]
        method = "openai_async"
    else:
        preds = [classify_intent_rules(q)[0] for q in queries]
        stats = {}
        method = "rules"

    correct = sum(1 for p, e in zip(preds, expected) if p == e)
    out = {
        "total": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples),
        "method": method,
        "stats": stats,
    }
    return out


def main() -> None:
    report_dir = ROOT / "evaluations" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    intent_rules = run_evaluation(use_openai=False)
    paraphrase = paraphrase_intent_eval()
    latency = latency_eval()

    lines = [
        "# Evaluation Summary",
        "",
        "## Intent classification (holdout, rules)",
        f"- Accuracy: {intent_rules['accuracy']:.4f}",
        f"- Macro F1: {intent_rules['macro_f1']:.4f}",
        f"- Report: {intent_rules['report']}",
        "",
        f"## Paraphrase intent ({paraphrase.get('method', 'rules')})",
        f"- Accuracy: {paraphrase['accuracy']:.4f} ({paraphrase['correct']}/{paraphrase['total']})",
    ]
    if paraphrase.get("stats"):
        s = paraphrase["stats"]
        lines.append(
            f"- API calls (deduped): {s.get('api_calls')}, wall: {s.get('wall_ms', 0):.0f} ms"
        )
    lines.extend([
        "",
        "## End-to-end latency",
        f"- Samples: {latency['count']}",
        f"- Mean per query (ms): {latency['mean_ms']:.1f}",
        f"- P50: {latency['p50_ms']:.1f} ms",
        f"- P95: {latency['p95_ms']:.1f} ms",
        f"- Wall-clock total (ms): {latency.get('wall_ms', 0):.1f}",
        f"- Async pipeline: {latency.get('async', False)}",
        "",
        "## Insight quality",
        "Use docs/EVAL_RUBRIC.md for manual scoring of evaluations/reports/sample_insights.json",
    ])

    if OPENAI_API_KEY:
        lines.extend([
            "",
            "## LLM holdout eval",
            "Run: python evaluations/intent_eval.py (includes async deduped OpenAI pass)",
        ])

    out = report_dir / "evaluation_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
