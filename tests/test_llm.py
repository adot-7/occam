"""Deterministic coverage for the WP-02 LLM layer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from occam.core import ToolSpec
from occam.llm import (
    AnthropicProvider,
    CompletionError,
    LLMClient,
    ModelConfig,
    OpenAICompatibleProvider,
    ProviderResponse,
    RetryPolicy,
    TokenBucket,
    TruncatedCompletionError,
    load_environment,
    load_model_configs,
)
from occam.llm.config import ConfigurationError, MissingCredentialsError, interpolate_env


def _config(
    key: str = "worker_fast",
    *,
    provider: str = "openai_compat",
    in_per_m: float = 0.0,
    out_per_m: float = 0.0,
    grant_equiv_in_per_m: float | None = 0.06,
    grant_equiv_out_per_m: float | None = 0.40,
    rpm: float | None = None,
    supports_json_schema: bool = False,
) -> ModelConfig:
    return ModelConfig(
        key=key,
        provider=provider,
        model="test-model",
        base_url="https://example.test/v1",
        api_key="${TEST_API_KEY}",
        in_per_m=in_per_m,
        out_per_m=out_per_m,
        grant_equiv_in_per_m=grant_equiv_in_per_m,
        grant_equiv_out_per_m=grant_equiv_out_per_m,
        rpm=rpm,
        supports_json_schema=supports_json_schema,
        tool_choice_modes=("auto",),
    )


class FakeProvider:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, config: ModelConfig, messages: Any, **kwargs: Any) -> Any:
        self.calls.append({"config": config, "messages": messages, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(str(status_code))
        self.status_code = status_code


def test_v3_model_table_has_exact_lanes_and_lazy_credentials() -> None:
    configs = load_model_configs(environ={})

    assert set(configs) == {"worker_fast", "worker_alt", "architect"}
    assert configs["worker_fast"].model == "glm-4-7-flash"
    assert configs["worker_fast"].base_url == "https://api.tensormux.com/v1"
    assert configs["worker_fast"].supports_json_schema is False
    assert configs["worker_fast"].tool_choice_modes == ("auto",)
    assert configs["worker_fast"].grant_equiv_in_per_m == 0.06
    assert configs["worker_alt"].model == "gpt-5-nano"
    assert configs["architect"].model == "claude-sonnet-5"
    with pytest.raises(MissingCredentialsError, match="TENSORMUX_API_KEY"):
        configs["worker_fast"].resolve_api_key({})


def test_environment_interpolation_is_recursive_and_strict_mode_is_clear() -> None:
    value = {"url": "https://${HOST}/v1", "nested": ["${PORT:-443}"]}
    assert interpolate_env(value, {"HOST": "api.test"}) == {
        "url": "https://api.test/v1",
        "nested": ["443"],
    }
    with pytest.raises(ConfigurationError, match="MISSING"):
        interpolate_env("${MISSING}", {}, strict=True)


def test_openai_provider_sends_native_tools_and_never_json_schema_for_worker_fast() -> None:
    class Completions:
        def __init__(self) -> None:
            self.request: dict[str, Any] | None = None

        def create(self, **request: Any) -> Any:
            self.request = request
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            content=None,
                            reasoning="hidden",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="fx_rate", arguments='{"date":"x"}'
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=152),
            )

    completions = Completions()
    provider = OpenAICompatibleProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    result = provider.complete(
        _config(),
        [{"role": "user", "content": "fetch"}],
        tools=[ToolSpec(name="fx_rate", description="rate", parameters={"type": "object"})],
        response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        max_tokens=2048,
    )

    assert completions.request is not None
    assert completions.request["temperature"] == 0.0
    assert completions.request["tool_choice"] == "auto"
    assert completions.request["tools"][0]["type"] == "function"
    assert "json_schema" not in json.dumps(completions.request)
    assert result.tokens_out == 152
    assert result.tool_calls[0]["function"]["arguments"] == '{"date":"x"}'


def test_openai_provider_uses_json_schema_only_when_config_allows_it() -> None:
    calls: list[dict[str, Any]] = []

    class Completions:
        def create(self, **request: Any) -> Any:
            calls.append(request)
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    provider = OpenAICompatibleProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    )
    provider.complete(
        _config(supports_json_schema=True),
        [{"role": "user", "content": "answer"}],
        tools=None,
        response_schema={"type": "object"},
        max_tokens=2048,
    )
    assert calls[0]["response_format"]["type"] == "json_schema"


def test_client_accounts_grant_equivalent_cost_and_cache_hit_is_free(tmp_path: Path) -> None:
    provider = FakeProvider([ProviderResponse(text="ok", tokens_in=1_000, tokens_out=2_000)])
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": provider},
        cache_dir=tmp_path,
    )

    first = client.complete("worker_fast", [{"role": "user", "content": "hello"}])
    second = client.complete("worker_fast", [{"role": "user", "content": "hello"}])

    assert provider.calls[0]["temperature"] == 0.0
    assert first.cost_usd == pytest.approx(0.00086)
    assert first.billed_cost_usd == 0
    assert first.cost_label == "list-rate-equivalent"
    assert first.list_rate_equivalent is True
    assert second.cached is True
    assert second.cost_usd == 0
    assert second.latency_s == 0
    assert len(provider.calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_bypass_fetches_fresh_response_without_replacing_canonical_cache(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ProviderResponse(text="canonical", tokens_in=1, tokens_out=1),
            ProviderResponse(text="fresh-repeat", tokens_in=2, tokens_out=2),
        ]
    )
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": provider},
        cache_dir=tmp_path,
    )
    messages = [{"role": "user", "content": "same request"}]

    canonical = client.complete("worker_fast", messages)
    repeat = client.complete("worker_fast", messages, use_cache=False)
    cached_again = client.complete("worker_fast", messages)

    assert canonical.text == "canonical"
    assert repeat.text == "fresh-repeat"
    assert repeat.cached is False
    assert cached_again.text == "canonical"
    assert cached_again.cached is True
    assert len(provider.calls) == 2


def test_corrupt_cache_entry_is_a_miss_and_gets_repaired(tmp_path: Path) -> None:
    provider = FakeProvider([ProviderResponse(text="repaired", tokens_in=1, tokens_out=1)])
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": provider},
        cache_dir=tmp_path,
    )
    request = client._request_payload(
        client.configs["worker_fast"],
        [{"role": "user", "content": "corrupt"}],
        None,
        None,
        2048,
        0.0,
    )
    address = client.cache.address(request)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{address}.json").write_text("{}", encoding="utf-8")

    result = client.complete("worker_fast", [{"role": "user", "content": "corrupt"}])

    assert result.text == "repaired"
    assert len(provider.calls) == 1


def test_client_retries_only_bounded_transient_failures(tmp_path: Path) -> None:
    provider = FakeProvider(
        [StatusError(429), StatusError(503), ProviderResponse(text="ok", tokens_in=1, tokens_out=1)]
    )
    delays: list[float] = []
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": provider},
        cache_dir=tmp_path,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.1, max_delay_s=1),
        sleeper=delays.append,
    )

    result = client.complete("worker_fast", [{"role": "user", "content": "retry"}])

    assert result.text == "ok"
    assert delays == [0.1, 0.2]
    assert len(provider.calls) == 3


def test_client_surfaces_non_transient_and_exhausted_failures(tmp_path: Path) -> None:
    bad_request = FakeProvider([StatusError(400)])
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": bad_request},
        cache_dir=tmp_path / "bad",
    )
    with pytest.raises(CompletionError):
        client.complete("worker_fast", [{"role": "user", "content": "bad"}])
    assert len(bad_request.calls) == 1

    exhausted = FakeProvider([StatusError(500), StatusError(500)])
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": exhausted},
        cache_dir=tmp_path / "exhausted",
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0),
        sleeper=lambda _: None,
    )
    with pytest.raises(CompletionError, match="2 attempt"):
        client.complete("worker_fast", [{"role": "user", "content": "down"}])


def test_reasoning_only_length_response_gets_one_doubled_budget_retry(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ProviderResponse(text="", reasoning="thinking", finish_reason="length"),
            ProviderResponse(text="answer", tokens_in=4, tokens_out=5),
        ]
    )
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": provider},
        cache_dir=tmp_path,
    )

    result = client.complete("worker_fast", [{"role": "user", "content": "think"}], max_tokens=1)

    assert result.text == "answer"
    assert [call["max_tokens"] for call in provider.calls] == [1024, 2048]


def test_reasoning_only_length_response_fails_after_the_single_retry(tmp_path: Path) -> None:
    response = ProviderResponse(text="", reasoning="thinking", finish_reason="length")
    provider = FakeProvider([response, response])
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": provider},
        cache_dir=tmp_path,
    )

    with pytest.raises(TruncatedCompletionError, match="truncated completion"):
        client.complete("worker_fast", [{"role": "user", "content": "think"}])
    assert [call["max_tokens"] for call in provider.calls] == [2048, 4096]
    assert list(tmp_path.glob("*.json")) == []


def test_missing_usage_is_approximate_and_empty_answer_is_a_failure(tmp_path: Path) -> None:
    provider = FakeProvider([ProviderResponse(text="hello")])
    client = LLMClient(
        {
            "metered": _config(
                in_per_m=1,
                out_per_m=2,
                grant_equiv_in_per_m=None,
                grant_equiv_out_per_m=None,
            )
        },
        providers={"metered": provider},
        cache_dir=tmp_path,
    )
    result = client.complete("metered", [{"role": "user", "content": "hello"}])
    assert result.usage_estimated is True
    assert result.cost_label == "approximate"
    assert result.tokens_in > 0 and result.tokens_out > 0

    granted_estimated = FakeProvider([ProviderResponse(text="hello")])
    granted_client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": granted_estimated},
        cache_dir=tmp_path / "granted",
    )
    granted_result = granted_client.complete("worker_fast", [{"role": "user", "content": "hello"}])
    assert granted_result.usage_estimated is True
    assert granted_result.cost_label == "list-rate-equivalent (approximate)"
    assert granted_result.list_rate_equivalent is True
    assert "approximate" in granted_result.cost_display_label

    empty = FakeProvider([ProviderResponse(text="")])
    empty_client = LLMClient(
        {"empty": _config()}, providers={"empty": empty}, cache_dir=tmp_path / "empty"
    )
    with pytest.raises(CompletionError, match="empty content"):
        empty_client.complete("empty", [{"role": "user", "content": "hello"}])


def test_truncated_cache_entry_is_a_miss_and_is_replaced(tmp_path: Path) -> None:
    provider = FakeProvider([ProviderResponse(text="fresh", tokens_in=2, tokens_out=3)])
    client = LLMClient(
        {"worker_fast": _config()},
        providers={"worker_fast": provider},
        cache_dir=tmp_path,
    )
    messages = [{"role": "user", "content": "cached truncation"}]
    request = client._request_payload(
        client.configs["worker_fast"], messages, None, None, 2048, 0.0
    )
    address = client.cache.address(request)
    client.cache.put(
        address,
        {
            "text": "",
            "tool_calls": [],
            "tokens_in": 2,
            "tokens_out": 2,
            "finish_reason": "length",
            "reasoning": "hidden reasoning only",
        },
    )

    result = client.complete("worker_fast", messages)

    assert result.cached is False
    assert result.text == "fresh"
    assert len(provider.calls) == 1
    assert client.cache.get(address)["text"] == "fresh"


def test_dotenv_values_are_optional_and_process_environment_wins(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "TEST_API_KEY=from-file\nFILE_ONLY=file-value\n",
        encoding="utf-8",
    )

    merged = load_environment(
        dotenv_path,
        {"TEST_API_KEY": "from-process", "PROCESS_ONLY": "process-value"},
    )
    assert merged["TEST_API_KEY"] == "from-process"
    assert merged["FILE_ONLY"] == "file-value"
    assert merged["PROCESS_ONLY"] == "process-value"
    assert load_environment(tmp_path / "missing.env", {}) == {}

    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "test:\n  provider: openai_compat\n  model: test-model\n  api_key: ${TEST_API_KEY}\n",
        encoding="utf-8",
    )
    configs = load_model_configs(
        config_path,
        {"TEST_API_KEY": "from-process"},
        dotenv_path=dotenv_path,
    )
    assert configs["test"].api_key == "from-process"
    missing_file_configs = load_model_configs(
        config_path,
        {"TEST_API_KEY": "from-process"},
        dotenv_path=tmp_path / "missing.env",
    )
    assert missing_file_configs["test"].api_key == "from-process"


def test_anthropic_adapter_maps_system_tools_and_native_tool_use() -> None:
    requests: list[dict[str, Any]] = []

    class Messages:
        def create(self, **request: Any) -> Any:
            requests.append(request)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="tool_use", id="tool_1", name="fx_rate", input={"x": 1})
                ],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=3, output_tokens=4),
            )

    provider = AnthropicProvider(client=SimpleNamespace(messages=Messages()))
    result = provider.complete(
        _config(provider="anthropic"),
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "use tool"},
        ],
        tools=[{"name": "fx_rate", "description": "rate", "parameters": {"type": "object"}}],
        response_schema=None,
        max_tokens=2048,
    )

    assert requests[0]["system"] == "system"
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["tools"][0]["input_schema"] == {"type": "object"}
    assert requests[0]["tool_choice"] == {"type": "auto"}
    assert result.tool_calls[0]["function"]["name"] == "fx_rate"
    assert result.tokens_out == 4


def test_anthropic_adapter_preserves_multi_turn_text_and_correlated_tool_blocks() -> None:
    requests: list[dict[str, Any]] = []

    class Messages:
        def create(self, **request: Any) -> Any:
            requests.append(request)
            return {
                "content": [{"type": "text", "text": "continued"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 6},
            }

    provider = AnthropicProvider(client=SimpleNamespace(messages=Messages()))
    provider.complete(
        _config(provider="anthropic", supports_json_schema=True),
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fetch both"},
            {
                "role": "assistant",
                "content": "I will check both sources.",
                "tool_calls": [
                    {
                        "id": "call_first",
                        "type": "function",
                        "function": {
                            "name": "fx_rate",
                            "arguments": '{"date":"2024-01-02","base":"USD"}',
                        },
                    },
                    {
                        "id": "call_second",
                        "type": "function",
                        "function": {
                            "name": "fx_series",
                            "arguments": '{"start":"2024-01-01","end":"2024-01-02"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_second", "content": "second result"},
            {"role": "tool", "tool_call_id": "call_first", "content": "first result"},
            {"role": "assistant", "content": "I received both results."},
            {"role": "user", "content": "now summarize"},
        ],
        tools=None,
        response_schema=None,
        max_tokens=2048,
    )

    converted = requests[0]["messages"]
    assert converted == [
        {"role": "user", "content": "fetch both"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will check both sources."},
                {
                    "type": "tool_use",
                    "id": "call_first",
                    "name": "fx_rate",
                    "input": {"date": "2024-01-02", "base": "USD"},
                },
                {
                    "type": "tool_use",
                    "id": "call_second",
                    "name": "fx_series",
                    "input": {"start": "2024-01-01", "end": "2024-01-02"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_second",
                    "content": "second result",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_first",
                    "content": "first result",
                },
            ],
        },
        {"role": "assistant", "content": "I received both results."},
        {"role": "user", "content": "now summarize"},
    ]


def test_anthropic_provider_sends_native_json_schema_payload() -> None:
    requests: list[dict[str, Any]] = []

    class Messages:
        def create(self, **request: Any) -> Any:
            requests.append(request)
            return {
                "content": [{"type": "text", "text": '{"answer":"ok"}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    provider = AnthropicProvider(client=SimpleNamespace(messages=Messages()))
    provider.complete(
        _config(provider="anthropic", supports_json_schema=True),
        [{"role": "user", "content": "answer"}],
        tools=None,
        response_schema=schema,
        max_tokens=2048,
    )

    assert requests[0]["temperature"] == 0.0
    assert requests[0]["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    assert "response_format" not in requests[0]


def test_anthropic_provider_fails_explicitly_without_native_schema_sdk_support() -> None:
    class OldMessages:
        def create(self, model: str, max_tokens: int, messages: Any, temperature: float) -> Any:
            del model, max_tokens, messages, temperature
            return {}

    provider = AnthropicProvider(client=SimpleNamespace(messages=OldMessages()))
    with pytest.raises(ConfigurationError, match="output_config"):
        provider.complete(
            _config(provider="anthropic", supports_json_schema=True),
            [{"role": "user", "content": "answer"}],
            tools=None,
            response_schema={"type": "object"},
            max_tokens=2048,
        )


def test_token_bucket_refills_deterministically() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    bucket = TokenBucket(2, 1, clock=lambda: now[0], sleeper=sleep)
    assert bucket.acquire() == 0
    assert bucket.acquire() == 0
    assert bucket.acquire() == pytest.approx(1)
    assert sleeps == [1.0]
