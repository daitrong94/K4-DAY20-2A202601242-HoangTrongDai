"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a technical writer producing a clear, well-organized final answer for the "
    "specified audience. Preserve the bracketed [source_id] citations from the notes and "
    "do not fabricate facts beyond what the notes support."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        TODO(student): Synthesize a clear response with citations or source references.
        """

        with trace_span("writer.run", {}) as span:
            try:
                if not state.research_notes:
                    raise AgentExecutionError(
                        "writer requires research_notes before drafting an answer"
                    )
                user_prompt = (
                    f"Research query: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n\n"
                    f"Research notes:\n{state.research_notes}\n\n"
                    f"Analysis notes:\n{state.analysis_notes or '(analysis step was skipped)'}\n\n"
                    "Write a clear, well-organized answer for the audience above, roughly "
                    "400-600 words. Keep the bracketed [source_id] citations wherever you "
                    "use a claim from the notes."
                )
                response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)
                state.final_answer = response.content
                span["attributes"]["input_tokens"] = response.input_tokens
                span["attributes"]["output_tokens"] = response.output_tokens
                span["attributes"]["cost_usd"] = response.cost_usd
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, let supervisor retry/fallback
                state.errors.append(f"writer: {exc}")
                span["attributes"]["error"] = str(exc)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer or "",
                metadata=dict(span["attributes"]),
            )
        )
        state.add_trace_event(span["name"], dict(span))
        return state
