"""LangGraph workflow skeleton."""

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

_WORKERS = ("researcher", "analyst", "writer", "critic")


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent()

    def build(self) -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
        """Create a LangGraph graph.

        TODO(student): Implement nodes, edges, conditional routing, and stop condition.
        Suggested nodes: supervisor, researcher, analyst, writer, optional critic.
        """

        graph: StateGraph[ResearchState, None, ResearchState, ResearchState] = StateGraph(
            ResearchState
        )
        graph.add_node("supervisor", self._supervisor.run)
        graph.add_node("researcher", self._researcher.run)
        graph.add_node("analyst", self._analyst.run)
        graph.add_node("writer", self._writer.run)
        graph.add_node("critic", self._critic.run)

        graph.set_entry_point("supervisor")

        def route(state: ResearchState) -> str:
            return state.route_history[-1] if state.route_history else DONE

        graph.add_conditional_edges(
            "supervisor",
            route,
            {**{worker: worker for worker in _WORKERS}, DONE: END},
        )
        for worker in _WORKERS:
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        TODO(student): Compile graph, invoke it, and convert result back to ResearchState.
        """

        settings = get_settings()
        compiled = self.build()
        # Each supervisor decision plus each worker call counts as a graph step, so give the
        # recursion limit enough headroom for a full researcher/analyst/writer/critic pass.
        result: Any = compiled.invoke(
            state,
            config={"recursion_limit": settings.max_iterations * len(_WORKERS) + 10},
        )
        return ResearchState.model_validate(result)
