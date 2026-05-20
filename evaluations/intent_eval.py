"""Evaluate intent classification on holdout set."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import OPENAI_API_KEY  # noqa: E402
from src.labels.intent_gold import add_intent_labels, train_holdout_split  # noqa: E402
from src.nlp.intent_classifier import (  # noqa: E402
    classify_batch_sync,
    classify_intent_rules,
)


def run_evaluation(use_openai: bool = False) -> dict:
    """Run eval with rules (default) or async OpenAI batch if key present."""
    add_intent_labels()
    _, holdout = train_holdout_split()

    y_true = holdout["intent"].tolist()
    queries = holdout["user_query"].tolist()
    latencies: list[float] = []
    stats: dict = {}

    if use_openai:
        results, stats = classify_batch_sync(queries, dedupe=True)
        y_pred = [r["intent"] for r in results]
        latencies = [r["latency_ms"] for r in results]
        method = "openai_async"
    else:
        y_pred = []
        for q in queries:
            intent, _ = classify_intent_rules(q)
            y_pred.append(intent)
        method = "rules"

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))

    report_dir = ROOT / "evaluations" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Intent Classification Evaluation",
        "",
        f"- Method: **{method}**",
        f"- Holdout size: {len(holdout)}",
        f"- Accuracy: **{acc:.4f}**",
        f"- Macro F1: **{macro_f1:.4f}**",
        "",
        "## Classification Report",
        "```",
        classification_report(y_true, y_pred),
        "```",
        "",
        "## Confusion Matrix",
        f"Labels: {sorted(set(y_true))}",
        "```",
        str(cm),
        "```",
    ]
    if latencies:
        lines.extend([
            "",
            f"- Mean per-row latency (ms): {sum(latencies) / len(latencies):.1f}",
        ])
    if stats:
        lines.extend([
            "",
            "## Async batch stats",
            f"- Unique API calls (deduped): {stats.get('api_calls', 'n/a')}",
            f"- Total rows: {stats.get('total_rows', len(holdout))}",
            f"- Concurrency: {stats.get('concurrency', 'n/a')}",
            f"- Wall-clock (ms): {stats.get('wall_ms', 0):.1f}",
        ])

    out_path = report_dir / f"intent_eval_{method}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    # Also write canonical copy for openai runs
    if use_openai:
        (report_dir / "intent_eval_openai.md").write_text(
            out_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(f"Report written to {out_path}")
    print(f"Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")
    if stats:
        print(
            f"API calls: {stats.get('api_calls')} (deduped), "
            f"wall: {stats.get('wall_ms', 0):.0f} ms"
        )
    return {"accuracy": acc, "macro_f1": macro_f1, "report": str(out_path), "stats": stats}


if __name__ == "__main__":
    run_evaluation(use_openai=False)
    if OPENAI_API_KEY:
        print("Running async OpenAI/OpenRouter evaluation (deduped unique queries)...")
        t0 = time.perf_counter()
        run_evaluation(use_openai=True)
        print(f"Total elapsed: {(time.perf_counter() - t0):.1f} s")
