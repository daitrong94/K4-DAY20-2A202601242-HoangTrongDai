"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

DONE = "done"
_MAX_ATTEMPTS_PER_STEP = 2


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        TODO(student): Implement routing policy. Suggested steps:
        - Inspect request, current notes, and missing fields.
        - Choose one of: researcher, analyst, writer, done.
        - Enforce max iterations and failure fallback.
        """

        settings = get_settings()
        decision = self._decide(state, settings.max_iterations)
        state.record_route(decision)
        state.add_trace_event(
            "supervisor.route", {"decision": decision, "iteration": state.iteration}
        )
        return state

    def _decide(self, state: ResearchState, max_iterations: int) -> str:
        # Hard guardrail: never let the graph loop forever regardless of what else happens.
        if state.iteration >= max_iterations:
            return DONE

        # 1. No research yet -> Researcher. Retry a bounded number of times on failure,
        #    then give up entirely since nothing downstream can work without sources.
        if state.research_notes is None:
            if self._attempts(state, "researcher") >= _MAX_ATTEMPTS_PER_STEP:
                state.errors.append("supervisor: giving up after repeated researcher failures")
                return DONE
            return "researcher"

        # 2. Research done but no analysis yet -> Analyst. Analysis is a quality
        #    enhancement, not a hard requirement, so fall back to Writer (or stop, if the
        #    writer already produced an answer) instead of retrying forever.
        if state.analysis_notes is None:
            if self._attempts(state, "analyst") >= _MAX_ATTEMPTS_PER_STEP:
                state.errors.append("supervisor: skipping analyst after repeated failures")
                return DONE if state.final_answer is not None else "writer"
            return "analyst"

        # 3. Notes ready but no final answer yet -> Writer. Retry, then give up.
        if state.final_answer is None:
            if self._attempts(state, "writer") >= _MAX_ATTEMPTS_PER_STEP:
                state.errors.append("supervisor: giving up after repeated writer failures")
                return DONE
            return "writer"

        # 4. We have a final answer: run the Critic exactly once as a quality gate.
        if "critic" not in state.route_history:
            return "critic"

        return DONE

    @staticmethod
    def _attempts(state: ResearchState, step: str) -> int:
        """Count prior failures recorded by a worker (see `agents/*.py` error prefixes)."""

        return sum(1 for error in state.errors if error.startswith(f"{step}:"))
