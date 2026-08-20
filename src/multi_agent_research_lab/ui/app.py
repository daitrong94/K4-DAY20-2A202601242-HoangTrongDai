"""Streamlit demo UI for the multi-agent research lab.

Run with:

    streamlit run src/multi_agent_research_lab/ui/app.py

or `make ui`. Requires the `ui` extra: `pip install -e ".[dev,llm,ui]"`.
"""

from typing import Any

import pandas as pd
import streamlit as st

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.runners import run_baseline, run_multi_agent

st.set_page_config(page_title="Multi-Agent Research Lab", page_icon="🔎", layout="wide")

settings = get_settings()
configure_logging(settings.log_level)


def _trace_rows(state: ResearchState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in state.trace:
        name = event.get("name", "")
        payload = event.get("payload", {})

        if name == "supervisor.route":
            rows.append(
                {
                    "step": "supervisor",
                    "detail": f"-> {payload.get('decision')} (iter {payload.get('iteration')})",
                    "duration_s": None,
                    "tokens_in": None,
                    "tokens_out": None,
                    "cost_usd": None,
                }
            )
            continue

        attributes = payload.get("attributes", {})
        detail = attributes.get("error", "")
        if not detail and "source_count" in attributes:
            detail = f"{attributes['source_count']} source(s) found"
        if not detail and "citation_coverage" in attributes:
            coverage = attributes["citation_coverage"]
            detail = "" if coverage is None else f"citation coverage {coverage:.0%}"
        rows.append(
            {
                "step": name,
                "detail": detail,
                "duration_s": payload.get("duration_seconds"),
                "tokens_in": attributes.get("input_tokens"),
                "tokens_out": attributes.get("output_tokens"),
                "cost_usd": attributes.get("cost_usd"),
            }
        )
    return rows


def _sources_rows(state: ResearchState) -> list[dict[str, Any]]:
    return [
        {
            "title": doc.title,
            "id": doc.metadata.get("document_id") or doc.metadata.get("article_id") or "-",
            "provider": doc.metadata.get("provider", "-"),
            "url": doc.url or "",
        }
        for doc in state.sources
    ]


def _render_state(state: ResearchState, label: str) -> None:
    st.subheader(label)
    st.markdown(state.final_answer or "_(no final answer)_")

    if state.sources:
        with st.expander(f"Sources ({len(state.sources)})"):
            st.dataframe(pd.DataFrame(_sources_rows(state)), hide_index=True, width="stretch")

    if state.trace:
        with st.expander("Pipeline trace", expanded=True):
            st.dataframe(pd.DataFrame(_trace_rows(state)), hide_index=True, width="stretch")

    if state.errors:
        st.warning("\n".join(state.errors))


st.title("Multi-Agent Research Lab — Demo")
st.caption(
    "Supervisor routes Researcher -> Analyst -> Writer -> Critic. "
    "See `docs/lab_guide.md` for the full design."
)

with st.sidebar:
    st.header("Settings")
    st.write(f"Model: `{settings.openai_model}`")
    st.write(f"Max iterations: {settings.max_iterations}")
    st.write("OpenAI key: " + ("configured" if settings.openai_api_key else "missing"))
    st.write(
        "Search source: "
        + ("Tavily" if settings.tavily_api_key else "offline corpus (no key needed)")
    )

query = st.text_area(
    "Research query",
    value="Compare centralized orchestrators vs decentralized agent coordination",
    height=100,
)
mode = st.radio("Mode", ["Multi-agent", "Baseline", "Compare both"], horizontal=True)
run_clicked = st.button("Run", type="primary")

if run_clicked:
    if len(query.strip()) < 5:
        st.error("Query must be at least 5 characters.")
    else:
        try:
            with st.spinner("Running..."):
                if mode == "Baseline":
                    st.session_state["baseline_state"] = run_baseline(query)
                    st.session_state.pop("multi_state", None)
                elif mode == "Multi-agent":
                    st.session_state["multi_state"] = run_multi_agent(query)
                    st.session_state.pop("baseline_state", None)
                else:
                    st.session_state["baseline_state"] = run_baseline(query)
                    st.session_state["multi_state"] = run_multi_agent(query)
        except AgentExecutionError as exc:
            st.error(str(exc))

baseline_state = st.session_state.get("baseline_state")
multi_state = st.session_state.get("multi_state")

if baseline_state is not None and multi_state is not None:
    col1, col2 = st.columns(2)
    with col1:
        _render_state(baseline_state, "Baseline (single-agent)")
    with col2:
        _render_state(multi_state, "Multi-Agent")
elif multi_state is not None:
    _render_state(multi_state, "Multi-Agent")
elif baseline_state is not None:
    _render_state(baseline_state, "Baseline (single-agent)")
else:
    st.info("Enter a query and click Run to see results.")
