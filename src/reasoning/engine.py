"""OpenAI insight generation with grounding and fallbacks."""

from __future__ import annotations

import json
from typing import Any

from src.config import CONFIDENCE_THRESHOLD, OPENAI_API_KEY, OPENAI_MODEL
from src.llm_client import create_llm_client
from src.models import InsightResponse, ParsedQuery
from src.reasoning.prompts import load_prompt_for_intent


class ReasoningEngine:
    """Generate executive insights from parsed query and metric context."""

    def __init__(self) -> None:
        self.use_openai = bool(OPENAI_API_KEY)
        self._client = create_llm_client() if self.use_openai else None

    def generate(
        self,
        parsed: ParsedQuery,
        metric_context: dict[str, Any],
    ) -> InsightResponse:
        if metric_context.get("row_count", 0) == 0:
            return self._fallback(
                parsed,
                "No data matches your filters. Try a different department or date range.",
            )

        if parsed.confidence < CONFIDENCE_THRESHOLD:
            return self._fallback(
                parsed,
                "Low confidence in query understanding. Please rephrase with metric and department.",
            )

        if self.use_openai and self._client:
            try:
                return self._generate_openai(parsed, metric_context)
            except Exception as exc:
                return self._generate_template(parsed, metric_context, error=str(exc))

        return self._generate_template(parsed, metric_context)

    def _generate_openai(
        self, parsed: ParsedQuery, metric_context: dict[str, Any]
    ) -> InsightResponse:
        assert self._client is not None
        system = load_prompt_for_intent(parsed.intent)
        user = json.dumps(
            {
                "query": parsed.raw_text,
                "parsed": {
                    "intent": parsed.intent,
                    "metrics": parsed.metrics,
                    "department": parsed.department,
                    "time_range": parsed.time_range,
                },
                "metric_context": metric_context,
            },
            default=str,
        )
        resp = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return InsightResponse(
            title=data.get("title", "Business Insight"),
            summary=data.get("summary", ""),
            bullets=list(data.get("bullets", [])),
            confidence=float(data.get("confidence", parsed.confidence)),
            recommended_actions=list(data.get("recommended_actions", [])),
            intent=parsed.intent,
            fallback=False,
            metadata={"method": "openai", "row_count": metric_context.get("row_count")},
        )

    def _generate_template(
        self,
        parsed: ParsedQuery,
        metric_context: dict[str, Any],
        error: str | None = None,
    ) -> InsightResponse:
        rc = metric_context.get("row_count", 0)
        conf = metric_context.get("avg_confidence", 0)
        lat = metric_context.get("avg_latency_ms", 0)
        fb = metric_context.get("avg_feedback")
        dept_top = list(metric_context.get("departments", {}).keys())[:2]

        title_map = {
            "forecasting": "Revenue Forecast Outlook",
            "comparative_analysis": "Cross-Segment Comparison",
            "operational_optimization": "Operational Efficiency Review",
            "anomaly_detection": "Anomaly Watch Summary",
            "performance_monitoring": "KPI Performance Summary",
        }
        title = title_map.get(parsed.intent, "Business Insight")
        summary = (
            f"Based on {rc} interactions"
            + (f" in {', '.join(dept_top)}" if dept_top else "")
            + f", average bot confidence is {conf:.0%} with {lat:.0f}ms latency."
        )
        if fb is not None:
            summary += f" Average user feedback is {fb:.1f}/5."

        bullets = [
            f"Sample size: {rc} records",
            f"Mean confidence: {conf:.2f}",
            f"Mean latency: {lat:.0f} ms",
        ]
        if metric_context.get("confidence_by_department"):
            best = max(
                metric_context["confidence_by_department"].items(),
                key=lambda x: x[1],
            )
            bullets.append(f"Highest confidence department: {best[0]} ({best[1]:.2f})")

        actions = [
            "Review slow-response cohorts and optimize model routing",
            "Validate metric definitions with business stakeholders",
        ]
        if parsed.intent == "forecasting":
            actions.insert(0, "Refresh quarterly forecast assumptions with Finance")

        return InsightResponse(
            title=title,
            summary=summary,
            bullets=bullets,
            confidence=min(parsed.confidence, 0.85),
            recommended_actions=actions,
            intent=parsed.intent,
            fallback=error is not None,
            metadata={"method": "template", "error": error},
        )

    def _fallback(self, parsed: ParsedQuery, message: str) -> InsightResponse:
        return InsightResponse(
            title="Unable to Generate Insight",
            summary=message,
            bullets=[],
            confidence=0.0,
            recommended_actions=["Rephrase your query", "Adjust filters"],
            intent=parsed.intent,
            fallback=True,
            metadata={"method": "fallback"},
        )
