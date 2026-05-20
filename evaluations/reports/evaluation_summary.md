# Evaluation Summary

## Intent classification (holdout, rules)
- Accuracy: 1.0000
- Macro F1: 1.0000
- Report: E:\LLM_BI\evaluations\reports\intent_eval_rules.md

## Paraphrase intent (openai_async)
- Accuracy: 0.9130 (21/23)
- API calls (deduped): 23, wall: 4800 ms

## End-to-end latency
- Samples: 10
- Mean per query (ms): 5961.7
- P50: 5331.9 ms
- P95: 10008.6 ms
- Wall-clock total (ms): 10073.0
- Async pipeline: True

## Insight quality
Use docs/EVAL_RUBRIC.md for manual scoring of evaluations/reports/sample_insights.json

## LLM holdout eval
Run: python evaluations/intent_eval.py (includes async deduped OpenAI pass)