"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.runners import run_baseline as _run_baseline
from multi_agent_research_lab.runners import run_multi_agent as _run_multi_agent

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _render_pipeline_table(state: ResearchState) -> Table:
    """Human-readable view of `state.trace` - handy for live demos and screenshots."""

    table = Table(title="Pipeline Trace")
    table.add_column("Step")
    table.add_column("Detail")
    table.add_column("Duration", justify="right")
    table.add_column("Tokens (in/out)", justify="right")
    table.add_column("Cost (USD)", justify="right")

    for event in state.trace:
        name = event.get("name", "")
        payload = event.get("payload", {})

        if name == "supervisor.route":
            table.add_row(
                "[bold cyan]supervisor[/bold cyan]",
                f"-> {payload.get('decision')} (iteration {payload.get('iteration')})",
                "",
                "",
                "",
            )
            continue

        attributes = payload.get("attributes", {})
        duration = payload.get("duration_seconds")
        tokens = ""
        if attributes.get("input_tokens") is not None:
            tokens = f"{attributes['input_tokens']}/{attributes.get('output_tokens', '-')}"
        cost = "" if attributes.get("cost_usd") is None else f"${attributes['cost_usd']:.4f}"
        detail = attributes.get("error", "")
        if not detail and "source_count" in attributes:
            detail = f"{attributes['source_count']} source(s) found"
        if not detail and "citation_coverage" in attributes:
            coverage = attributes["citation_coverage"]
            detail = "" if coverage is None else f"citation coverage {coverage:.0%}"
        table.add_row(
            name,
            detail,
            "" if duration is None else f"{duration:.2f}s",
            tokens,
            cost,
        )
    return table


def _render_sources_table(state: ResearchState) -> Table:
    table = Table(title="Sources")
    table.add_column("#")
    table.add_column("Title")
    table.add_column("ID")
    table.add_column("Provider")
    for index, doc in enumerate(state.sources, start=1):
        doc_id = doc.metadata.get("document_id") or doc.metadata.get("article_id") or "-"
        table.add_row(str(index), doc.title, str(doc_id), str(doc.metadata.get("provider", "-")))
    return table


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline: one direct LLM call, no tools."""

    _init()
    request = _parse_query(query)
    try:
        state = _run_baseline(request.query)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=2) from exc
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the raw ResearchState JSON instead")
    ] = False,
) -> None:
    """Run the multi-agent workflow: Supervisor -> Researcher/Analyst/Writer/Critic."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=2) from exc

    if as_json:
        console.print(result.model_dump_json(indent=2))
        return

    console.print(Panel.fit(result.final_answer or "(no final answer)", title="Final Answer"))
    console.print(_render_pipeline_table(result))
    if result.sources:
        console.print(_render_sources_table(result))
    if result.errors:
        console.print(
            Panel.fit("\n".join(result.errors), title="Warnings / Errors", style="yellow")
        )


@app.command()
def benchmark(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to the benchmark config YAML")
    ] = "configs/lab_default.yaml",
    output: Annotated[
        str, typer.Option("--output", "-o", help="Where to write the markdown report")
    ] = "reports/benchmark_report.md",
) -> None:
    """Run baseline vs multi-agent for every configured query and render a report."""

    _init()
    queries = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["benchmark"]["queries"]

    all_metrics = []
    for query in queries:
        label = query if len(query) <= 40 else f"{query[:37]}..."
        for run_name, runner in (("baseline", _run_baseline), ("multi-agent", _run_multi_agent)):
            _, metrics = run_benchmark(f"{run_name}: {label}", query, runner)
            all_metrics.append(metrics)
            console.print(f"[dim]completed[/dim] {run_name}: {label}")

    report = render_markdown_report(all_metrics)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    console.print(Panel.fit(f"Wrote {len(all_metrics)} rows to {out_path}", title="Benchmark"))


if __name__ == "__main__":
    app()
