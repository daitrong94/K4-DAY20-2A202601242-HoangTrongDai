"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a critical analyst. Given research notes, extract the key claims, compare "
    "viewpoints across sources, and flag any claim backed by weak or single-source "
    "evidence. Preserve the bracketed [source_id] citations from the notes."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        TODO(student): Extract key claims, compare viewpoints, and flag weak evidence.
        """

        with trace_span("analyst.run", {}) as span:
            try:
                if not state.research_notes:
                    raise AgentExecutionError(
                        "analyst requires research_notes from the researcher step"
                    )
                response = self._llm_client.complete(
                    _SYSTEM_PROMPT,
                    f"Research query: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes}\n\n"
                    "Extract the key claims, note where sources agree or disagree, and "
                    "flag any claim with weak or single-source evidence.",
                )
                state.analysis_notes = response.content
                span["attributes"]["input_tokens"] = response.input_tokens
                span["attributes"]["output_tokens"] = response.output_tokens
                span["attributes"]["cost_usd"] = response.cost_usd
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, let supervisor retry/fallback
                state.errors.append(f"analyst: {exc}")
                span["attributes"]["error"] = str(exc)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes or "",
                metadata=dict(span["attributes"]),
            )
        )
        state.add_trace_event(span["name"], dict(span))
        return state
