"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a placeholder metric object.

    TODO(student): Add quality scoring, estimated token cost, citation coverage, and error rate.
    """

    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a failed run should still produce a benchmark row
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(f"benchmark: run failed: {exc}")
        failed = True
    latency = perf_counter() - started

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_total_cost(state),
        quality_score=_quality_score(state, failed),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if failed or not state.final_answer else 0.0,
        notes="; ".join(state.errors),
    )
    return state, metrics


def _total_cost(state: ResearchState) -> float | None:
    costs: list[float] = []
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, int | float):
            costs.append(float(cost))
    return sum(costs) if costs else None


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


def _quality_score(state: ResearchState, failed: bool) -> float | None:
    """Cheap automated proxy only.

    This is NOT a substitute for the human rubric in `docs/peer_review_rubric.md` - treat
    it as a rough signal for regression-spotting between runs, not a ground-truth grade.
    """

    if failed or not state.final_answer:
        return 0.0

    score = 4.0  # produced a non-empty final answer at all
    coverage = _citation_coverage(state)
    if coverage is not None:
        score += coverage * 4.0
    word_count = len(state.final_answer.split())
    if 150 <= word_count <= 800:
        score += 1.0
    if not state.errors:
        score += 1.0
    return min(score, 10.0)
