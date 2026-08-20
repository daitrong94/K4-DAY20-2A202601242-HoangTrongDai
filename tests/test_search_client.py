"""Offline corpus fallback tests for SearchClient (no network, no API key)."""

from multi_agent_research_lab.services.search_client import SearchClient


def test_offline_search_returns_relevant_sources() -> None:
    client = SearchClient()
    results = client.search("multi-agent research architecture", max_results=3)
    assert 0 < len(results) <= 3
    assert all(doc.snippet for doc in results)


def test_offline_search_respects_max_results() -> None:
    client = SearchClient()
    results = client.search("agent memory architecture retrieval", max_results=2)
    assert len(results) <= 2


def test_offline_search_returns_empty_for_unrelated_query() -> None:
    client = SearchClient()
    results = client.search("zzqzxk wwqzpl vvbnmq yyklmp", max_results=5)
    assert results == []
