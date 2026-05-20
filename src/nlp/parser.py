"""Query parser extracting structured slots."""

from __future__ import annotations

import re

from src.config import OPENAI_API_KEY
from src.models import ParsedQuery
from src.nlp.intent_classifier import IntentClassifier, classify_intent_rules

METRIC_KEYWORDS = {
    "revenue": "Revenue",
    "churn": "Churn Rate",
    "roi": "ROI",
    "conversion": "Conversion Rate",
    "profit": "Profit Margin",
    "margin": "Profit Margin",
    "sales": "Revenue",
}

DEPARTMENT_KEYWORDS = {
    "finance": "Finance",
    "hr": "HR",
    "human resources": "HR",
    "marketing": "Marketing",
    "operations": "Operations",
    "sales": "Marketing",
}


class QueryParser:
    """Parse user queries into structured slots."""

    def __init__(self, classifier: IntentClassifier | None = None) -> None:
        self.classifier = classifier or IntentClassifier(use_openai=bool(OPENAI_API_KEY))

    def parse(self, text: str, department_hint: str | None = None) -> ParsedQuery:
        result = self.classifier.classify(text)
        intent = result["intent"]
        confidence = result["confidence"]

        metrics = self._extract_metrics(text)
        department = department_hint or self._extract_department(text)
        time_range = self._extract_time_range(text)
        aggregation = self._extract_aggregation(text, intent)

        return ParsedQuery(
            raw_text=text.strip(),
            intent=intent,
            metrics=metrics,
            department=department,
            time_range=time_range,
            aggregation=aggregation,
            confidence=confidence,
        )

    def _extract_metrics(self, text: str) -> list[str]:
        lower = text.lower()
        found = []
        for key, label in METRIC_KEYWORDS.items():
            if key in lower and label not in found:
                found.append(label)
        if not found:
            rules_intent = classify_intent_rules(text)[0]
            if rules_intent == "forecasting":
                found.append("Revenue")
            elif "churn" in lower:
                found.append("Churn Rate")
        return found

    def _extract_department(self, text: str) -> str | None:
        lower = text.lower()
        for key, dept in DEPARTMENT_KEYWORDS.items():
            if key in lower:
                return dept
        return None

    def _extract_time_range(self, text: str) -> str | None:
        lower = text.lower()
        if "next quarter" in lower or "next q" in lower:
            return "next_quarter"
        if "monthly" in lower or "month" in lower:
            return "monthly"
        if "year" in lower or "annual" in lower:
            return "annual"
        if re.search(r"q[1-4]", lower):
            return "quarterly"
        return None

    def _extract_aggregation(self, text: str, intent: str) -> str | None:
        lower = text.lower()
        if "compare" in lower or intent == "comparative_analysis":
            return "comparison"
        if "forecast" in lower or intent == "forecasting":
            return "predictive"
        if "trend" in lower:
            return "trend"
        if "highest" in lower or "top" in lower:
            return "ranking"
        return "descriptive"
