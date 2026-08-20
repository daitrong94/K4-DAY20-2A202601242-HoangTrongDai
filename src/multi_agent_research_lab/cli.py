"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

_BASELINE_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer directly and concisely from your own "
    "knowledge; you have no search tool in this baseline mode."
)


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


def _run_baseline(query: str) -> ResearchState:
    """Single-agent baseline: one direct LLM call, no tools, no other agents."""

    state = ResearchState(request=ResearchQuery(query=query))
    response = LLMClient().complete(_BASELINE_SYSTEM_PROMPT, query)
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


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


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
) -> None:
    """Run the multi-agent workflow skeleton."""

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
    console.print(result.model_dump_json(indent=2))


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
