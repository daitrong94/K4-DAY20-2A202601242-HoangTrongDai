"""LLMClient tests that avoid any real network calls."""

import pytest

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.llm_client import LLMClient


def test_complete_without_api_key_raises_agent_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    try:
        client = LLMClient()
        with pytest.raises(AgentExecutionError):
            client.complete("system prompt", "user prompt")
    finally:
        get_settings.cache_clear()
