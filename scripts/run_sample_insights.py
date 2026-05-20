"""Generate insights for sample queries (template or OpenAI)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import QueryFilters  # noqa: E402
from src.pipeline.orchestrator import Pipeline  # noqa: E402


def main() -> None:
    samples = json.loads((ROOT / "evaluations" / "sample_queries.json").read_text())
    pipeline = Pipeline()
    out_dir = ROOT / "evaluations" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in samples[:5]:
        r = pipeline.run_query(item["query"], filters=QueryFilters())
        insight = r["insight"]
        results.append(
            {
                "query": item["query"],
                "expected_intent": item["expected_intent"],
                "predicted_intent": r["parsed"].intent,
                "title": insight.title,
                "summary": insight.summary[:200],
            }
        )
    path = out_dir / "sample_insights.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} samples to {path}")


if __name__ == "__main__":
    main()
