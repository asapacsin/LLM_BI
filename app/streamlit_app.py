"""Streamlit UI for LLM Business Decision Support System."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL  # noqa: E402
from src.data.clean import load_processed  # noqa: E402
from src.models import QueryFilters  # noqa: E402
from src.pipeline.orchestrator import Pipeline  # noqa: E402
from src.state.store import StateStore  # noqa: E402


def render_insight_card(insight) -> None:
    conf_pct = f"{insight.confidence:.0%}"
    badge = "🟢" if insight.confidence >= 0.7 else "🟡" if insight.confidence >= 0.4 else "🔴"
    st.markdown(f"### {insight.title} {badge} `{conf_pct}`")
    if insight.fallback:
        st.warning("Fallback response — check filters or API configuration.")
    st.markdown(insight.summary)
    if insight.bullets:
        st.markdown("**Key points**")
        for b in insight.bullets:
            st.markdown(f"- {b}")
    if insight.recommended_actions:
        st.markdown("**Recommended actions**")
        for a in insight.recommended_actions:
            st.markdown(f"- {a}")


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline(store=StateStore())


@st.cache_data
def get_filter_options():
    df = load_processed()
    return {
        "departments": ["All"] + sorted(df["department"].dropna().unique().tolist()),
        "metrics": ["All"] + sorted(df["metrics_requested"].dropna().unique().tolist()),
        "min_date": df["timestamp"].min().date(),
        "max_date": df["timestamp"].max().date(),
    }


def main() -> None:
    st.set_page_config(
        page_title="LLM BI Decision Support",
        page_icon="📊",
        layout="wide",
    )
    st.title("LLM Business Decision Support System")
    st.caption("Query → Parse → Classify → Analyze → Executive insight")

    opts = get_filter_options()
    pipeline = get_pipeline()
    store = pipeline.store

    with st.sidebar:
        st.header("Session")
        if "session_id" not in st.session_state:
            st.session_state.session_id = store.create_session()

        sessions = store.list_sessions()
        if sessions:
            labels = [f"{s['title'][:24]} ({s['id']})" for s in sessions]
            ids = [s["id"] for s in sessions]
            idx = ids.index(st.session_state.session_id) if st.session_state.session_id in ids else 0
            picked = st.selectbox("Load session", range(len(labels)), format_func=lambda i: labels[i], index=idx)
            st.session_state.session_id = ids[picked]

        if st.button("New conversation"):
            st.session_state.session_id = store.create_session()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("Filters")
        department = st.selectbox("Department", opts["departments"])
        metric = st.selectbox("Metric", opts["metrics"])
        date_range = st.date_input(
            "Date range",
            value=(opts["min_date"], opts["max_date"]),
            min_value=opts["min_date"],
            max_value=opts["max_date"],
        )

        st.divider()
        if OPENAI_API_KEY:
            provider = "OpenRouter" if OPENAI_BASE_URL and "openrouter" in OPENAI_BASE_URL else "OpenAI-compatible"
            api_status = f"Connected ({provider})"
        else:
            api_status = "Template mode (no API key)"
        st.text(f"LLM: {api_status}")
        st.text(f"Model: {OPENAI_MODEL}")

    filters = QueryFilters(
        department=None if department == "All" else department,
        metric=None if metric == "All" else metric,
        date_start=str(date_range[0]) if len(date_range) == 2 else None,
        date_end=str(date_range[1]) if len(date_range) == 2 else None,
    )

    history = store.get_history(st.session_state.session_id)
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("metadata"):
                meta = msg["metadata"]
                if isinstance(meta, dict) and meta.get("title"):
                    with st.expander("Insight details"):
                        st.json(meta)

    examples = [
        "Forecast next quarter revenue",
        "Compare sales performance by region",
        "Which campaign had highest ROI?",
        "Show monthly revenue trend",
        "What is customer churn rate?",
    ]
    st.markdown("**Example queries**")
    cols = st.columns(len(examples))
    example_query = None
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            example_query = ex

    query = st.chat_input("Ask a business intelligence question...")
    if example_query:
        query = example_query

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.spinner("Analyzing..."):
            result = pipeline.run_query(
                query,
                session_id=st.session_state.session_id,
                filters=filters,
            )
        insight = result["insight"]
        with st.chat_message("assistant"):
            render_insight_card(insight)
        with st.expander("Parser & metrics (debug)"):
            st.json(
                {
                    "intent": result["parsed"].intent,
                    "confidence": result["parsed"].confidence,
                    "metrics": result["parsed"].metrics,
                    "department": result["parsed"].department,
                    "metric_context": result["metric_context"],
                    "latency_ms": result["latency_ms"],
                }
            )


if __name__ == "__main__":
    main()
