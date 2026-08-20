"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger("multi_agent_research_lab.trace")

# LangSmith orders/nests runs using a "dotted_order" string: a root run's is
# "<timestamp><run_id>"; a child's is "<parent's dotted_order>.<timestamp><run_id>".
# See Client.batch_ingest_runs()'s docstring example for the exact format.
_DOTTED_ORDER_TIME_FORMAT = "%Y%m%dT%H%M%S%fZ"


@dataclass(frozen=True)
class _TraceContext:
    trace_id: UUID
    run_id: UUID
    dotted_order: str


# Whatever trace_span() is currently open, so a nested trace_span() call (e.g. an agent's
# span opened while the workflow's root span is open) is linked as its child in LangSmith
# instead of showing up as an unrelated, unlinked run.
_current_trace: ContextVar[_TraceContext | None] = ContextVar("_current_trace", default=None)


@lru_cache(maxsize=1)
def _langsmith_client() -> tuple[Any, str] | tuple[None, None]:
    """Best-effort LangSmith client, built once and reused for every span.

    Returns (None, None) unless configured/installed.
    """

    settings = get_settings()
    if not settings.langsmith_api_key:
        return None, None
    try:
        from langsmith import Client
    except ImportError:
        return None, None
    try:
        return Client(api_key=settings.langsmith_api_key), settings.langsmith_project
    except Exception:  # noqa: BLE001 - tracing must never break the pipeline
        logger.debug("failed to construct LangSmith client", exc_info=True)
        return None, None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.

    Nests under whatever `trace_span()` is currently open (if any): wrap one call
    (e.g. `MultiAgentWorkflow.run`) as the outer span and every span opened underneath it
    (the Researcher/Analyst/Writer/Critic calls) shows up as a child in the same LangSmith
    trace tree, instead of four separate, unlinked runs.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    client, project = _langsmith_client()
    run_id: UUID | None = uuid4() if client is not None else None
    parent = _current_trace.get()
    token = None

    if client is not None and run_id is not None:
        timestamp = datetime.now(UTC).strftime(_DOTTED_ORDER_TIME_FORMAT)
        if parent is None:
            trace_id, dotted_order, parent_run_id = run_id, f"{timestamp}{run_id}", None
        else:
            trace_id = parent.trace_id
            dotted_order = f"{parent.dotted_order}.{timestamp}{run_id}"
            parent_run_id = parent.run_id
        token = _current_trace.set(_TraceContext(trace_id, run_id, dotted_order))
        try:
            client.create_run(
                name=name,
                inputs=dict(span["attributes"]),
                run_type="chain",
                project_name=project,
                id=run_id,
                trace_id=trace_id,
                dotted_order=dotted_order,
                parent_run_id=parent_run_id,
                start_time=datetime.now(UTC),
            )
        except Exception:  # noqa: BLE001 - tracing must never break the pipeline
            logger.debug("langsmith create_run failed", exc_info=True)

    error: BaseException | None = None
    try:
        yield span
    except BaseException as exc:
        error = exc
        raise
    finally:
        if token is not None:
            _current_trace.reset(token)
        span["duration_seconds"] = perf_counter() - started
        logger.debug("trace_span %s", span)
        if client is not None and run_id is not None:
            try:
                client.update_run(
                    run_id,
                    end_time=datetime.now(UTC),
                    outputs={
                        "attributes": span["attributes"],
                        "duration_seconds": span["duration_seconds"],
                    },
                    error=str(error) if error else None,
                )
            except Exception:  # noqa: BLE001 - tracing must never break the pipeline
                logger.debug("langsmith update_run failed", exc_info=True)
