"""Unit tests for SupervisorAgent's routing policy.

NOTE(student): this file used to only guard that the skeleton raised StudentTodoError
(see git history). Now that SupervisorAgent.run is implemented in agents/supervisor.py,
it tests the actual routing/guardrail behavior instead, per the instruction that used to
live in this docstring.
"""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_to_researcher_first() -> None:
    state = SupervisorAgent().run(_state())
    assert state.route_history == ["researcher"]
    assert state.iteration == 1


def test_supervisor_progresses_through_the_pipeline() -> None:
    supervisor = SupervisorAgent()
    state = _state()

    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.research_notes = "notes [doc1]"
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "analysis [doc1]"
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "answer [doc1]"
    state = supervisor.run(state)
    assert state.route_history[-1] == "critic"

    state = supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_stops_at_max_iterations() -> None:
    state = _state()
    for _ in range(6):
        state.record_route("researcher")
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_gives_up_after_repeated_researcher_failures() -> None:
    state = _state()
    state.errors = ["researcher: boom", "researcher: boom again"]
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "done"
    assert any("giving up" in error for error in state.errors)
