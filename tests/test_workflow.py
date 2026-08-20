"""End-to-end smoke test for MultiAgentWorkflow using fake LLM/search clients.

No network access and no API keys required: this only exercises the LangGraph wiring
(nodes, conditional routing, stop condition) implemented in graph/workflow.py.
"""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse


class _FakeLLMClient:
    def __init__(self, content: str) -> None:
        self._content = content

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(content=self._content, input_tokens=10, output_tokens=10, cost_usd=0.001)


class _FakeSearchClient:
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title="Fake Source",
                url="https://example.com",
                snippet="A fake source for testing.",
                metadata={"document_id": "fake1"},
            )
        ]


def test_workflow_reaches_a_final_answer_without_network() -> None:
    workflow = MultiAgentWorkflow()
    workflow._researcher = ResearcherAgent(
        search_client=_FakeSearchClient(), llm_client=_FakeLLMClient("notes [fake1]")
    )
    workflow._analyst = AnalystAgent(llm_client=_FakeLLMClient("analysis [fake1]"))
    workflow._writer = WriterAgent(llm_client=_FakeLLMClient("final answer [fake1]"))
    workflow._critic = CriticAgent(llm_client=_FakeLLMClient("No major issues found."))

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent research systems"))
    result = workflow.run(state)

    assert result.final_answer == "final answer [fake1]"
    assert result.route_history[-1] == "done"
    assert "critic" in result.route_history
    assert not result.errors
