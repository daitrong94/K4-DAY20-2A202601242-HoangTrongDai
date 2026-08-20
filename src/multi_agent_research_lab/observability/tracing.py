"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger("multi_agent_research_lab.trace")


def _langsmith_client() -> tuple[Any, str] | tuple[None, None]:
    """Best-effort LangSmith client. Returns (None, None) unless configured/installed."""

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
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    client, project = _langsmith_client()
    run_id: UUID | None = uuid4() if client is not None else None
    if client is not None and run_id is not None:
        try:
            client.create_run(
                name=name,
                inputs=dict(span["attributes"]),
                run_type="chain",
                project_name=project,
                id=run_id,
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
