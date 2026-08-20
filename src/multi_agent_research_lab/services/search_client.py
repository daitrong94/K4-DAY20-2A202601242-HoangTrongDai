"""Search client abstraction for ResearcherAgent."""

import json
import re
import ssl
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

# src/multi_agent_research_lab/services/search_client.py -> repo root
_DEFAULT_CORPUS_DIR = (
    Path(__file__).resolve().parents[3] / "ai_agent_offline_research_corpus_v2" / "topics"
)
_TAVILY_URL = "https://api.tavily.com/search"

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "with",
        "from",
        "into",
        "this",
        "are",
        "was",
        "were",
        "can",
        "but",
        "not",
        "all",
        "has",
        "have",
        "had",
        "its",
        "about",
        "your",
        "you",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "who",
        "does",
        "than",
        "then",
        "over",
        "under",
        "such",
        "these",
        "those",
        "will",
        "would",
        "should",
        "could",
        "also",
    }
)


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {word for word in words if word not in _STOPWORDS}


@lru_cache(maxsize=4)
def _load_corpus(corpus_dir: Path) -> tuple[dict[str, Any], ...]:
    if not corpus_dir.is_dir():
        return ()
    return tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(corpus_dir.glob("*.json"))
    )


class SearchClient:
    """Provider-agnostic search client skeleton.

    Uses Tavily when `TAVILY_API_KEY` is configured. Otherwise it falls back to the bundled
    offline corpus (`ai_agent_offline_research_corpus_v2/`) so the lab works end-to-end
    without any search API key.
    """

    def __init__(self, corpus_dir: Path = _DEFAULT_CORPUS_DIR) -> None:
        self._settings = get_settings()
        self._corpus_dir = corpus_dir

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        TODO(student): Implement with Tavily, Bing, SerpAPI, internal docs, or a local mock.
        """

        if self._settings.tavily_api_key:
            return self._search_tavily(query, max_results)
        return self._search_offline_corpus(query, max_results)

    # -- Tavily -------------------------------------------------------------

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self._settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
            }
        ).encode("utf-8")
        request = Request(
            _TAVILY_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_context = ssl.create_default_context()

        try:
            with urlopen(
                request, timeout=self._settings.timeout_seconds, context=ssl_context
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - normalize provider errors for callers
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc

        return [
            SourceDocument(
                title=item.get("title") or "Untitled",
                url=item.get("url"),
                snippet=(item.get("content") or "")[:1000],
                metadata={"provider": "tavily", "score": item.get("score")},
            )
            for item in body.get("results", [])[:max_results]
        ]

    # -- Offline corpus fallback ---------------------------------------------

    def _search_offline_corpus(self, query: str, max_results: int) -> list[SourceDocument]:
        query_terms = _terms(query)
        scored: list[tuple[int, SourceDocument]] = []

        for topic in _load_corpus(self._corpus_dir):
            topic_meta = topic.get("topic", {})
            topic_text = f"{topic_meta.get('name', '')} {' '.join(topic_meta.get('tags', []))}"
            topic_score = len(query_terms & _terms(topic_text))
            kb = topic.get("knowledge_base", {})

            for doc in kb.get("source_documents", []):
                text = f"{doc.get('title', '')} {doc.get('full_text', '')[:2000]}"
                score = topic_score + len(query_terms & _terms(text))
                if score <= 0:
                    continue
                scored.append(
                    (
                        score,
                        SourceDocument(
                            title=doc.get("title") or "Untitled",
                            url=doc.get("provenance_url"),
                            snippet=(doc.get("full_text") or "")[:600],
                            metadata={
                                "provider": "offline_corpus",
                                "document_id": doc.get("document_id"),
                                "topic": topic_meta.get("name"),
                                "is_synthetic": doc.get("is_synthetic", False),
                            },
                        ),
                    )
                )

            for article in kb.get("knowledge_articles", []):
                text = f"{article.get('title', '')} {article.get('content', '')[:2000]}"
                score = topic_score + len(query_terms & _terms(text))
                if score <= 0:
                    continue
                scored.append(
                    (
                        score,
                        SourceDocument(
                            title=article.get("title") or "Untitled",
                            url=None,
                            snippet=(article.get("content") or "")[:600],
                            metadata={
                                "provider": "offline_corpus",
                                "article_id": article.get("article_id"),
                                "topic": topic_meta.get("name"),
                            },
                        ),
                    )
                )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[:max_results]]
