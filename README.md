# LLM Business Decision Support System

An LLM-powered BI decision-support pipeline that classifies business questions, computes grounded metrics from 3,200 chatbot interaction records, and generates executive-style insights. Supports **OpenRouter** (OpenAI-compatible API), **async batch evaluation**, and a **Streamlit** demo with local session persistence.

## Evaluation results

Benchmarks are generated locally under `evaluations/reports/`. Re-run the commands in [Run evaluation](#run-evaluation) to refresh.

### Intent classification (primary benchmark)

The dataset contains only **five canonical query templates**; holdout rows repeat those strings. We report **paraphrase accuracy** on reworded business questions as the main intent metric.

| Benchmark | Size | Result | Notes |
|-----------|------|--------|--------|
| Paraphrase intent (OpenRouter, async) | 23 queries | **91.3%** (21/23) | [`evaluations/sample_queries.json`](evaluations/sample_queries.json) |
| Template label alignment (rules) | 640 holdout rows | Consistency check only | Not a generalization claim |

Details: [`evaluations/reports/evaluation_summary.md`](evaluations/reports/evaluation_summary.md)

### Pipeline performance

| Metric | Value (10-query async batch) |
|--------|------------------------------|
| Mean end-to-end latency | ~6.0 s per query |
| P50 / P95 | ~5.3 s / ~10.0 s |
| Holdout LLM eval (deduped) | 5 API calls, ~2.2 s wall-clock |

Each end-to-end run includes intent classification and grounded insight generation. Without an API key, the system uses rule-based intent and template insights (much lower latency).

### Insight outputs

Sample LLM responses for five canonical BI questions are in [`evaluations/reports/sample_insights.json`](evaluations/reports/sample_insights.json). Qualitative quality should be assessed with [`docs/EVAL_RUBRIC.md`](docs/EVAL_RUBRIC.md).

### Scope and limitations

- **Not open-domain NLU** — evaluation reflects a controlled template set plus a small paraphrase suite.
- **Ground truth** — intent labels are derived from query templates, not independent human annotation at scale.
- **`query_category`** in the raw CSV is noisy relative to `user_query` and is not used as classification ground truth.

---

## How it works

```
User query + filters → Parse & classify intent → Pandas metrics (filtered CSV)
                    → LLM insight (JSON) or template fallback → Insight card + SQLite session
```

The LLM receives a **small aggregate `metric_context`**, not the full dataset — reducing hallucination and API cost vs. pasting raw CSV into chat.

---

## Quick start

```bash
pip install -r requirements.txt
copy .env.example .env
python scripts/prepare_data.py
streamlit run app/streamlit_app.py
```

Configure `.env` for OpenRouter:

```env
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
EVAL_CONCURRENCY=10
```

Full setup: [docs/TUTORIAL.md](docs/TUTORIAL.md)

---

## Run evaluation

```bash
python evaluations/intent_eval.py
python evaluations/run_all.py
python scripts/run_sample_insights.py
```

**Outputs (saved locally):**

| File | Content |
|------|---------|
| `evaluations/reports/evaluation_summary.md` | Overview metrics |
| `evaluations/reports/intent_eval_rules.md` | Template label-alignment (rules) |
| `evaluations/reports/intent_eval_openai_async.md` | Template label-alignment (LLM) + async stats |
| `evaluations/reports/sample_insights.json` | Sample query → insight log |

With an API key, `intent_eval.py` dedupes identical holdout queries (**640 rows → 5 API calls**) and runs them concurrently (`EVAL_CONCURRENCY`).

---

## Features

- **Five intent types** — performance monitoring, anomaly detection, forecasting, comparative analysis, operational optimization
- **Grounded analytics** — department / metric / date filters over processed parquet/CSV
- **OpenRouter or direct OpenAI** — via `OPENAI_BASE_URL` in `.env`
- **Async batch evaluation** — fast LLM intent eval without 640 sequential calls
- **Template fallback** — runs without an API key for demos
- **Local state** — `state/agent.db` + optional `state/sessions/*.json`

---

## Project structure

```
data/                      Raw + processed datasets
src/
  data/                    Load, clean, path resolution
  labels/                  Gold intent labels and train/holdout splits
  nlp/                     Parser, sync + async intent classifier
  analytics/               Metric context builder
  reasoning/               LLM insight engine
  pipeline/                End-to-end orchestrator
  state/                   SQLite session store
app/streamlit_app.py       Interactive UI
prompts/                   Per-intent YAML prompts
evaluations/               Eval scripts + reports/
docs/                      Schema, tutorial, demo, technical report
notebooks/                 EDA notebook
```

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/SCHEMA.md](docs/SCHEMA.md) | Dataset columns and intent mapping |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | Setup and usage |
| [docs/DEMO.md](docs/DEMO.md) | Demo query script |
| [docs/REPORT.md](docs/REPORT.md) | Architecture and design |
| [docs/EVAL_RUBRIC.md](docs/EVAL_RUBRIC.md) | Manual insight quality scoring |

---

## Dataset

[BI Chatbot Interactions](https://www.kaggle.com/datasets/jawadaahmed/business-intelligence-chatbot-interaction-dataset) — `data/business-intelligence-chatbot-interaction-dataset/BI_Chatbot_Interactions.csv` (3,200 rows, 12 columns).

---

## License

Academic / course project — comply with dataset and API provider terms of use.
