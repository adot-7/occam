"""Provider-independent completion client with retries, cache, and accounting."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

from occam.llm.cache import DiskCache
from occam.llm.config import (
    ConfigurationError,
    MissingCredentialsError,
    ModelConfig,
    load_model_configs,
)
from occam.llm.cost import calculate_cost
from occam.llm.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderResponse,
)
from occam.llm.rate_limit import PerModelRateLimiter


class LLMError(RuntimeError):
    """Base class for errors surfaced by the LLM client."""


class ProviderError(LLMError):
    """A provider call failed or returned an unusable response."""


class CompletionError(LLMError):
    """A completion could not be produced after the configured attempts."""


class TruncatedCompletionError(CompletionError):
    """The model spent its output budget on reasoning before visible content."""


@dataclass(slots=True)
class Completion:
    """Stable result returned by :func:`complete`.

    ``tokens_out`` is the provider's completion-token usage, not a count of
    visible characters or words.  For GLM-4.7-Flash this includes hidden
    reasoning tokens by design.
    """

    text: str
    tool_calls: list[dict[str, Any]]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float
    cached: bool
    cost_label: str = "metered"
    billed_cost_usd: float = 0.0
    finish_reason: str | None = None
    reasoning: str | None = None
    usage_estimated: bool = False
    model_key: str | None = None

    @property
    def cost_basis(self) -> str:
        """Alias used by metrics/TUI consumers."""

        return self.cost_label

    @property
    def list_rate_equivalent(self) -> bool:
        return self.cost_label == "list-rate-equivalent"

    @property
    def cost_display_label(self) -> str:
        """Human-facing label used by the eventual TUI cost strip."""

        if self.list_rate_equivalent:
            return "cost (list-rate eq.)"
        return self.cost_label

    @property
    def actual_cost_usd(self) -> float:
        """Alias for nominal provider billing before grant equivalence."""

        return self.billed_cost_usd

    def cache_payload(self) -> dict[str, Any]:
        """Serialize only deterministic response fields, never raw SDK objects."""

        return {
            "text": self.text,
            "tool_calls": self.tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "finish_reason": self.finish_reason,
            "reasoning": self.reasoning,
            "usage_estimated": self.usage_estimated,
        }


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff for transient provider failures."""

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("retry delays cannot be negative")

    def delay(self, failed_attempt: int) -> float:
        """Return delay after attempt number 1, 2, ... has failed."""

        return min(self.max_delay_s, self.base_delay_s * (2 ** max(0, failed_attempt - 1)))


_TRANSIENT_TEXT = re.compile(r"\b(?:429|500|502|503|504)\b|rate.?limit|temporar", re.I)


def _status_code(exc: BaseException) -> int | None:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "code", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status == 429 or 500 <= status <= 599
    return bool(_TRANSIENT_TEXT.search(str(exc)))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _estimate_tokens(
    messages: Sequence[Mapping[str, Any]], text: str, tools: Sequence[Any] | None
) -> tuple[int, int]:
    """Use tiktoken when available, with a deterministic dependency-free fallback."""

    prompt = json.dumps(_jsonable({"messages": messages, "tools": tools}), ensure_ascii=False)
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return max(1, len(encoder.encode(prompt))), max(1, len(encoder.encode(text)))
    except (ImportError, LookupError, RuntimeError):
        return max(1, (len(prompt) + 3) // 4), max(1, (len(text) + 3) // 4)


def _coerce_response(value: Any) -> ProviderResponse:
    if isinstance(value, ProviderResponse):
        return value
    if isinstance(value, Completion):
        return ProviderResponse(
            text=value.text,
            tool_calls=value.tool_calls,
            tokens_in=value.tokens_in,
            tokens_out=value.tokens_out,
            finish_reason=value.finish_reason,
            reasoning=value.reasoning,
        )
    if isinstance(value, Mapping):
        usage = value.get("usage")
        usage_map = usage if isinstance(usage, Mapping) else {}
        tokens_in = value.get(
            "tokens_in", usage_map.get("prompt_tokens", usage_map.get("input_tokens"))
        )
        tokens_out = value.get(
            "tokens_out", usage_map.get("completion_tokens", usage_map.get("output_tokens"))
        )
        return ProviderResponse(
            text=str(value.get("text") or ""),
            tool_calls=list(value.get("tool_calls") or []),
            tokens_in=_optional_int(tokens_in),
            tokens_out=_optional_int(tokens_out),
            finish_reason=value.get("finish_reason"),
            reasoning=value.get("reasoning"),
            usage=value.get("usage"),
            raw=value,
        )
    raise ProviderError(f"provider returned unsupported response type {type(value).__name__}")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _valid_cache_payload(value: Mapping[str, Any]) -> bool:
    """Reject truncated/corrupt entries before they can masquerade as answers."""

    required = {"text", "tool_calls", "tokens_in", "tokens_out"}
    if not required <= value.keys() or not isinstance(value["tool_calls"], list):
        return False
    try:
        int(value["tokens_in"])
        int(value["tokens_out"])
    except (TypeError, ValueError):
        return False
    return True


class LLMClient:
    """Synchronous LLM facade shared by the architect and workers."""

    def __init__(
        self,
        configs: Mapping[str, ModelConfig] | None = None,
        *,
        config_path: str | Path | None = None,
        providers: Mapping[str, Provider] | None = None,
        cache: DiskCache | None = None,
        cache_dir: str | Path | None = None,
        global_cache_dir: str | Path | None = None,
        rate_limiter: PerModelRateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        raw_configs = load_model_configs(config_path) if configs is None else configs
        self.configs = {}
        for key, value in raw_configs.items():
            config = (
                value if isinstance(value, ModelConfig) else ModelConfig.from_mapping(key, value)
            )
            self.configs[key] = config if config.key == key else replace(config, key=key)
        self.providers = dict(providers or {})
        if cache is not None:
            self.cache = cache
        else:
            directory = Path.home() / ".occam" / "cache" if cache_dir is None else Path(cache_dir)
            self.cache = DiskCache(directory, global_cache_dir)
        self.rate_limiter = rate_limiter or PerModelRateLimiter(clock=clock, sleeper=sleeper)
        self.retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper
        self._provider_instances: dict[str, Provider] = {}

    def _provider_for(self, config: ModelConfig) -> Provider:
        provider = self.providers.get(config.key) or self.providers.get(config.provider)
        if provider is not None:
            return provider
        if config.key in self._provider_instances:
            return self._provider_instances[config.key]
        if config.provider == "openai_compat":
            provider = OpenAICompatibleProvider()
        elif config.provider == "anthropic":
            provider = AnthropicProvider()
        else:
            raise ConfigurationError(f"unsupported provider {config.provider!r} for {config.key}")
        self._provider_instances[config.key] = provider
        return provider

    def _request_payload(
        self,
        config: ModelConfig,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Any] | None,
        response_schema: Mapping[str, Any] | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "config": config.cache_identity(),
            "messages": _jsonable(messages),
            "tools": _jsonable(tools),
            "response_schema": _jsonable(response_schema),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def complete(
        self,
        model_key: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Any] | None = None,
        response_schema: Mapping[str, Any] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Completion:
        """Complete one request, applying the cache, limiter, retry, and cost rules."""

        try:
            config = self.configs[model_key]
        except KeyError as exc:
            raise ConfigurationError(f"unknown model key {model_key!r}") from exc
        budget = config.max_tokens if max_tokens is None else max(1024, int(max_tokens))
        request = self._request_payload(
            config, messages, tools, response_schema, budget, temperature
        )
        address = self.cache.address(request)
        cached_payload = self.cache.get(address)
        if cached_payload is not None and _valid_cache_payload(cached_payload):
            return Completion(
                text=str(cached_payload.get("text") or ""),
                tool_calls=list(cached_payload.get("tool_calls") or []),
                tokens_in=int(cached_payload.get("tokens_in", 0)),
                tokens_out=int(cached_payload.get("tokens_out", 0)),
                cost_usd=0.0,
                latency_s=0.0,
                cached=True,
                cost_label="cache-hit",
                billed_cost_usd=0.0,
                finish_reason=cached_payload.get("finish_reason"),
                reasoning=cached_payload.get("reasoning"),
                usage_estimated=bool(cached_payload.get("usage_estimated", False)),
                model_key=model_key,
            )

        provider = self._provider_for(config)
        response: ProviderResponse | None = None
        attempt = 0
        truncation_retry = False
        started = time.perf_counter()
        current_budget = budget
        while response is None:
            attempt += 1
            try:
                self.rate_limiter.acquire(config.key, config.rpm)
                provider_kwargs: dict[str, Any] = {
                    "tools": tools,
                    "response_schema": response_schema,
                    "max_tokens": current_budget,
                }
                if temperature != 0.0:
                    provider_kwargs["temperature"] = temperature
                response = _coerce_response(provider.complete(config, messages, **provider_kwargs))
            except Exception as exc:
                if not _retryable(exc) or attempt >= self.retry_policy.max_attempts:
                    if isinstance(exc, (LLMError, ConfigurationError)):
                        raise
                    raise CompletionError(
                        f"LLM completion failed after {attempt} attempt(s): {type(exc).__name__}"
                    ) from exc
                self._sleeper(self.retry_policy.delay(attempt))
                continue

            if (
                not response.text.strip()
                and response.reasoning
                and response.finish_reason == "length"
            ):
                if truncation_retry:
                    raise TruncatedCompletionError(
                        "truncated completion: output budget was consumed by reasoning "
                        f"after retry at max_tokens={current_budget}"
                    )
                truncation_retry = True
                current_budget *= 2
                response = None
                continue
            break

        assert response is not None
        if not response.text.strip() and not response.tool_calls:
            raise CompletionError("provider returned empty content without native tool calls")
        elapsed = time.perf_counter() - started
        estimated_in, estimated_out = _estimate_tokens(messages, response.text, tools)
        usage_estimated = response.tokens_in is None or response.tokens_out is None
        tokens_in = estimated_in if response.tokens_in is None else response.tokens_in
        tokens_out = estimated_out if response.tokens_out is None else response.tokens_out
        breakdown = calculate_cost(config, tokens_in, tokens_out, approximate=usage_estimated)
        completion = Completion(
            text=response.text,
            tool_calls=response.tool_calls or [],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=breakdown.cost_usd,
            latency_s=elapsed,
            cached=False,
            cost_label=breakdown.label,
            billed_cost_usd=breakdown.billed_cost_usd,
            finish_reason=response.finish_reason,
            reasoning=response.reasoning,
            usage_estimated=usage_estimated,
            model_key=model_key,
        )
        self.cache.put(address, completion.cache_payload())
        return completion


_default_client: LLMClient | None = None


def get_default_client() -> LLMClient:
    """Lazily create the process-wide client used by the module function."""

    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def complete(
    model_key: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Any] | None = None,
    response_schema: Mapping[str, Any] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> Completion:
    """Module-level convenience wrapper for the shared completion interface."""

    return get_default_client().complete(
        model_key,
        messages,
        tools,
        response_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )


__all__ = [
    "Completion",
    "CompletionError",
    "LLMClient",
    "LLMError",
    "MissingCredentialsError",
    "ProviderError",
    "RetryPolicy",
    "TruncatedCompletionError",
    "complete",
    "get_default_client",
]
