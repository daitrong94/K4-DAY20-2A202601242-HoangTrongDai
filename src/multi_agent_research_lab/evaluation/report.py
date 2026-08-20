"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown.

    TODO(student): Add richer analysis, examples, screenshots, and trace links.
    """

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.extend(["", "## Summary", ""])
    if not metrics:
        lines.append("No runs recorded.")
    else:
        latencies = [item.latency_seconds for item in metrics]
        costs = [item.estimated_cost_usd for item in metrics if item.estimated_cost_usd is not None]
        qualities = [item.quality_score for item in metrics if item.quality_score is not None]
        failures = [item.failure_rate for item in metrics if item.failure_rate is not None]
        lines.append(f"- Runs: {len(metrics)}")
        lines.append(f"- Avg latency: {sum(latencies) / len(latencies):.2f}s")
        if costs:
            lines.append(f"- Total estimated cost: ${sum(costs):.4f}")
        if qualities:
            avg_quality = sum(qualities) / len(qualities)
            lines.append(f"- Avg quality (automated proxy): {avg_quality:.1f}/10")
        if failures:
            lines.append(f"- Failure rate: {sum(failures) / len(failures):.0%}")
        failed_runs = [item.run_name for item in metrics if item.failure_rate]
        if failed_runs:
            lines.append(f"- Failed runs: {', '.join(failed_runs)}")

    lines.extend(
        [
            "",
            "> Quality above is a cheap automated proxy (see `evaluation/benchmark.py`). "
            "Pair it with the human rubric in `docs/peer_review_rubric.md`, and attach a "
            "trace link or screenshot per run when submitting this report.",
        ]
    )
    return "\n".join(lines) + "\n"
