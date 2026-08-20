"""Shared single-query runners used by both the CLI and the Streamlit demo UI."""

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer directly and concisely from your own "
    "knowledge; you have no search tool in this baseline mode."
)


def run_baseline(query: str) -> ResearchState:
    """Single-agent baseline: one direct LLM call, no tools, no other agents."""

    state = ResearchState(request=ResearchQuery(query=query))
    response = LLMClient().complete(BASELINE_SYSTEM_PROMPT, query)
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    return state


def run_multi_agent(query: str) -> ResearchState:
    """Full Supervisor -> Researcher/Analyst/Writer/Critic pipeline for one query."""

    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)
