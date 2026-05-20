"""Project configuration loaded from environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
EVAL_CONCURRENCY = max(1, int(os.getenv("EVAL_CONCURRENCY", "10")))
STATE_DB_PATH = Path(os.getenv("STATE_DB_PATH", "state/agent.db"))
if not STATE_DB_PATH.is_absolute():
    STATE_DB_PATH = PROJECT_ROOT / STATE_DB_PATH

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_REL = (
    "business-intelligence-chatbot-interaction-dataset/BI_Chatbot_Interactions.csv"
)
PROCESSED_DIR = DATA_DIR / "processed"
KAGGLE_DATA_PATH = (
    "/kaggle/input/datasets/jawadaahmed/"
    "business-intelligence-chatbot-interaction-dataset/BI_Chatbot_Interactions.csv"
)

INTENT_CATEGORIES = [
    "performance_monitoring",
    "anomaly_detection",
    "forecasting",
    "comparative_analysis",
    "operational_optimization",
]

QUERY_TO_INTENT = {
    "Forecast next quarter revenue": "forecasting",
    "What is customer churn rate?": "performance_monitoring",
    "Compare sales performance by region": "comparative_analysis",
    "Which campaign had highest ROI?": "operational_optimization",
    "Show monthly revenue trend": "performance_monitoring",
}
