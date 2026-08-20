"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import APIError, OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

# Rough USD price per 1M tokens (input, output). Used only to produce an order-of-magnitude
# cost estimate for the benchmark report - update this table if pricing or the model changes.
_PRICE_PER_MILLION_TOKENS_USD: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.openai_model
        self._timeout = float(settings.timeout_seconds)
        self._client: OpenAI | None = (
            OpenAI(api_key=settings.openai_api_key, timeout=self._timeout)
            if settings.openai_api_key
            else None
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        TODO(student): Connect OpenAI, Azure OpenAI, or another provider.
        Keep retry, timeout, and token logging here rather than inside agents.
        """

        if self._client is None:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Add it to .env to enable real LLM calls."
            )

        try:
            response = self._call(self._client, system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - normalize provider errors for callers
            raise AgentExecutionError(f"LLM completion failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(APIError),
    )
    def _call(self, client: OpenAI, system_prompt: str, user_prompt: str) -> ChatCompletion:
        return client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        prices = _PRICE_PER_MILLION_TOKENS_USD.get(self._model)
        if prices is None or input_tokens is None or output_tokens is None:
            return None
        input_price, output_price = prices
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
