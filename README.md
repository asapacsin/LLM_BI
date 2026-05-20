# LLM Business Decision Support System

An LLM-powered BI decision-support pipeline that classifies business questions, computes grounded metrics from 3,200 chatbot interaction records, and generates executive-style insights. Supports **OpenRouter** (OpenAI-compatible API), **async batch evaluation**, and a **Streamlit** demo with local session persistence.

## Performance summary

Results from the latest local evaluation (`evaluations/reports/`). Re-run commands below to refresh.

### Intent classification (holdout, n=640)

| Method | Accuracy | Macro F1 | Notes |
|--------|----------|----------|--------|
| Rules baseline | **1.00** | **1.00** | Exact match on 5 canonical query templates |
| OpenRouter (`openai_async`) | **1.00** | **1.00** | Async + deduped to **5 API calls** (~2.2 s wall-clock) |

Per-class F1 is 1.00 for: comparative analysis, forecasting, operational optimization, performance monitoring. See [`evaluations/reports/intent_eval_openai_async.md`](evaluations/reports/intent_eval_openai_async.md).

### Paraphrase robustness (n=23)

| Method | Accuracy |
|--------|----------|
| OpenRouter async | **91.3%** (21/23) |

23 reworded queries from [`evaluations/sample_queries.json`](evaluations/sample_queries.json); ~4.8 s wall-clock with deduped API calls.

### End-to-end pipeline latency (10 sample queries, async)

| Metric | Value |
|--------|-------|
| Mean per query | ~6.0 s |
| P50 | ~5.3 s |
| P95 | ~10.0 s |
| Total wall-clock (10 parallel) | ~10.1 s |

Each query = intent classification + grounded insight generation (OpenRouter). Template-only mode (no API key) is much faster (~tens of ms per query).

### Sample insights (5 canonical queries)

All 5 template queries matched expected intent; LLM-generated titles and summaries are logged in [`evaluations/reports/sample_insights.json`](evaluations/reports/sample_insights.json).

### Limitations

- Training data contains only **5 unique `user_query` templates** — holdout accuracy is a controlled benchmark, not open-domain NLU.
- **`query_category`** in raw CSV is often misaligned with query text; not used as ground truth.
- Insight quality should be scored manually with [`docs/EVAL_RUBRIC.md`](docs/EVAL_RUBRIC.md) (target average ≥ 3.5/5).

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
| `evaluations/reports/intent_eval_rules.md` | Rules-only holdout eval |
| `evaluations/reports/intent_eval_openai_async.md` | LLM holdout eval + async stats |
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
