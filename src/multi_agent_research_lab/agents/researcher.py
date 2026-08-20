"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a meticulous research assistant. Given retrieved sources, write concise "
    "research notes as bullet points. Every factual claim must cite the source it came "
    "from using its bracketed id, e.g. [autogen]. Never invent sources or facts that are "
    "not present in the provided material."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._search_client = search_client or SearchClient()
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        TODO(student): Implement search, source filtering, citation capture, and notes.
        """

        with trace_span("researcher.run", {"query": state.request.query}) as span:
            try:
                sources = self._search_client.search(
                    state.request.query, max_results=state.request.max_sources
                )
                state.sources = sources
                span["attributes"]["source_count"] = len(sources)

                if not sources:
                    state.research_notes = (
                        "No sources were found for this query. Consider broadening the search."
                    )
                else:
                    response = self._summarize(state.request.query, sources)
                    state.research_notes = response.content
                    span["attributes"]["input_tokens"] = response.input_tokens
                    span["attributes"]["output_tokens"] = response.output_tokens
                    span["attributes"]["cost_usd"] = response.cost_usd
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, let supervisor retry/fallback
                state.errors.append(f"researcher: {exc}")
                span["attributes"]["error"] = str(exc)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes or "",
                metadata=dict(span["attributes"]),
            )
        )
        state.add_trace_event(span["name"], dict(span))
        return state

    def _summarize(self, query: str, sources: list[SourceDocument]) -> LLMResponse:
        catalog = "\n\n".join(
            f"[{doc.metadata.get('document_id') or doc.metadata.get('article_id') or index}] "
            f"{doc.title}\n{doc.snippet}"
            for index, doc in enumerate(sources, start=1)
        )
        user_prompt = (
            f"Research query: {query}\n\nSources:\n{catalog}\n\n"
            "Write 5-8 bullet point research notes summarizing what these sources say, "
            "citing the bracketed id after every claim."
        )
        return self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)
