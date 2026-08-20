"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a rigorous reviewer. Check a draft research answer for unsupported claims, "
    "missing citations, and overconfident language. Reply with a short bullet list of "
    "issues, or the single line 'No major issues found.' if the answer looks solid."
)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        TODO(student): Add fact-check, citation coverage, or hallucination checks.
        """

        review = ""
        with trace_span("critic.run", {}) as span:
            try:
                if not state.final_answer:
                    raise AgentExecutionError("critic requires a final_answer to review")

                coverage = self._citation_coverage(state)
                span["attributes"]["citation_coverage"] = coverage
                if coverage is not None and coverage < 0.5:
                    state.errors.append(
                        f"critic: low citation coverage ({coverage:.0%}); "
                        "answer may be under-sourced"
                    )

                response = self._llm_client.complete(
                    _SYSTEM_PROMPT,
                    f"Final answer to review:\n{state.final_answer}\n\n"
                    "Available sources:\n" + "\n".join(f"- {doc.title}" for doc in state.sources),
                )
                review = response.content
                span["attributes"]["input_tokens"] = response.input_tokens
                span["attributes"]["output_tokens"] = response.output_tokens
                span["attributes"]["cost_usd"] = response.cost_usd
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, let supervisor retry/fallback
                state.errors.append(f"critic: {exc}")
                span["attributes"]["error"] = str(exc)

        state.agent_results.append(
            AgentResult(agent=AgentName.CRITIC, content=review, metadata=dict(span["attributes"]))
        )
        state.add_trace_event(span["name"], dict(span))
        return state

    @staticmethod
    def _citation_coverage(state: ResearchState) -> float | None:
        known_ids = {
            str(doc.metadata.get("document_id") or doc.metadata.get("article_id"))
            for doc in state.sources
            if doc.metadata.get("document_id") or doc.metadata.get("article_id")
        }
        if not known_ids or not state.final_answer:
            return None
        cited = {source_id for source_id in known_ids if f"[{source_id}]" in state.final_answer}
        return len(cited) / len(known_ids)
