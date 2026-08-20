"""Benchmark heuristics tests using synthetic runners (no LLM/network calls)."""

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def _fake_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    state.sources = [
        SourceDocument(title="A", snippet="a", metadata={"document_id": "doc1"}),
        SourceDocument(title="B", snippet="b", metadata={"document_id": "doc2"}),
    ]
    state.final_answer = "Some answer citing [doc1] but not the other source."
    state.agent_results = [
        AgentResult(agent=AgentName.WRITER, content=state.final_answer, metadata={"cost_usd": 0.01})
    ]
    return state


def test_run_benchmark_computes_citation_coverage_and_cost() -> None:
    _, metrics = run_benchmark("test-run", "some query text", _fake_runner)
    assert metrics.citation_coverage == 0.5
    assert metrics.estimated_cost_usd == 0.01
    assert metrics.failure_rate == 0.0
    assert metrics.quality_score is not None and metrics.quality_score > 0


def test_run_benchmark_marks_failed_runs() -> None:
    def _boom(query: str) -> ResearchState:
        raise RuntimeError("boom")

    _, metrics = run_benchmark("test-run", "some query text", _boom)
    assert metrics.failure_rate == 1.0
    assert metrics.quality_score == 0.0
    assert "boom" in metrics.notes
