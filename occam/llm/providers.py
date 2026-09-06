"""OpenAI-compatible and Anthropic provider adapters.

The third-party SDK imports intentionally live inside ``_client_for``.  The
CLI can therefore initialise Neatlogs before the first provider is constructed
without this module importing ``openai`` too early.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

from occam.llm.config import ConfigurationError, ModelConfig


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns no usable completion choice."""


@dataclass(slots=True)
class ProviderResponse:
    """Provider-neutral response before client-side accounting and caching."""

    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    finish_reason: str | None = None
    reasoning: str | None = None
    usage: dict[str, Any] | None = None
    raw: Any = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


class Provider(Protocol):
    """Structural interface accepted by :class:`occam.llm.LLMClient`."""

    def complete(
        self,
        config: ModelConfig,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Any] | None,
        response_schema: Mapping[str, Any] | None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ProviderResponse: ...


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if is_dataclass(value):
        dumped = asdict(value)
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"tool must be a mapping or model-dumpable object, got {type(value).__name__}")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pieces: list[str] = []
        for part in value:
            part_type = _get(part, "type")
            if part_type in {None, "text", "output_text"}:
                text = _get(part, "text", _get(part, "value", ""))
                if text:
                    pieces.append(str(text))
        return "".join(pieces)
    return str(value)


def _usage_mapping(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    result: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
        value = getattr(usage, name, None)
        if value is not None:
            result[name] = value
    return result


def _usage_count(usage: Mapping[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in usage and usage[name] is not None:
            return int(usage[name])
    return None


def normalize_openai_tools(tools: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    """Convert Occam ``ToolSpec``/short mappings to native OpenAI tools."""

    if tools is None:
        return None
    normalized: list[dict[str, Any]] = []
    for raw_tool in tools:
        tool = dict(_mapping(raw_tool))
        if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
            function = dict(tool["function"])
            normalized.append({**tool, "function": function})
            continue
        if "name" not in tool:
            raise ConfigurationError("each tool must contain a function name")
        function: dict[str, Any] = {
            "name": str(tool["name"]),
            "description": str(tool.get("description", "")),
            "parameters": tool.get(
                "parameters",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
        }
        normalized.append({"type": "function", "function": function})
    return normalized


def _json_schema_response(response_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a plain JSON schema or an OpenAI named schema."""

    if "json_schema" in response_schema:
        return {"type": "json_schema", "json_schema": dict(response_schema["json_schema"])}
    if "name" in response_schema and "schema" in response_schema:
        payload = dict(response_schema)
    else:
        payload = {"name": "response", "schema": dict(response_schema), "strict": True}
    return {"type": "json_schema", "json_schema": payload}


def _tool_call_mapping(tool_call: Any) -> dict[str, Any]:
    raw = dict(_mapping(tool_call))
    function = _get(raw, "function", {})
    function_mapping = dict(_mapping(function)) if function else {}
    arguments = function_mapping.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return {
        "id": raw.get("id"),
        "type": raw.get("type", "function"),
        "function": {
            "name": function_mapping.get("name", ""),
            "arguments": arguments,
        },
    }


class OpenAICompatibleProvider:
    """Adapter for OpenAI, TensorMux, and other OpenAI-shaped gateways."""

    def __init__(self, client: Any = None, client_factory: Any = None) -> None:
        self._client = client
        self._client_factory = client_factory
        self._clients: dict[tuple[str | None, str], Any] = {}

    def _client_for(self, config: ModelConfig) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            return self._client_factory(config)
        api_key = config.resolve_api_key()
        cache_key = (config.base_url, api_key)
        if cache_key in self._clients:
            return self._clients[cache_key]
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ConfigurationError("openai is required for openai_compat models") from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        client = OpenAI(**kwargs)
        self._clients[cache_key] = client
        return client

    def complete(
        self,
        config: ModelConfig,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Any] | None,
        response_schema: Mapping[str, Any] | None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        request: dict[str, Any] = {
            "model": config.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
        }
        if temperature != 0.0:
            request["temperature"] = temperature
        normalized_tools = normalize_openai_tools(tools)
        if normalized_tools:
            request["tools"] = normalized_tools
            # GLM-4.7-Flash supports auto only.  Never pass required/none.
            if "auto" not in config.tool_choice_modes:
                raise ConfigurationError(f"{config.key} does not permit native auto tool choice")
            request["tool_choice"] = "auto"
        if response_schema is not None:
            if config.supports_json_schema:
                request["response_format"] = _json_schema_response(response_schema)
            else:
                # GLM supports json_object, not json_schema.  The schema itself
                # is intentionally not sent because the gateway rejects it.
                request["response_format"] = {"type": "json_object"}

        response = self._client_for(config).chat.completions.create(**request)
        choices = _get(response, "choices", [])
        if not choices:
            raise ProviderResponseError("provider returned no completion choices")
        choice = choices[0]
        message = _get(choice, "message", {})
        usage = _usage_mapping(_get(response, "usage"))
        tool_calls = [_tool_call_mapping(item) for item in (_get(message, "tool_calls", []) or [])]
        return ProviderResponse(
            text=_text(_get(message, "content")),
            tool_calls=tool_calls,
            tokens_in=_usage_count(usage, ("prompt_tokens", "input_tokens")),
            # completion_tokens is intentionally first: GLM includes hidden
            # reasoning tokens in this field and it must be used verbatim.
            tokens_out=_usage_count(usage, ("completion_tokens", "output_tokens")),
            finish_reason=_get(choice, "finish_reason"),
            reasoning=_text(_get(message, "reasoning", _get(message, "reasoning_content"))) or None,
            usage=usage,
            raw=response,
        )


def normalize_anthropic_tools(tools: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    """Convert the shared short/OpenAI tool shape to Anthropic's shape."""

    if tools is None:
        return None
    normalized: list[dict[str, Any]] = []
    for raw_tool in tools:
        tool = dict(_mapping(raw_tool))
        if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
            function = dict(tool["function"])
            normalized.append(
                {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {}),
                }
            )
        else:
            normalized.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {}),
                }
            )
    return normalized


def _anthropic_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    systems: list[str] = []
    converted: list[dict[str, Any]] = []
    for raw_message in messages:
        message = dict(raw_message)
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            systems.append(_text(content))
        elif role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": _text(content),
                        }
                    ],
                }
            )
        else:
            converted.append({"role": role, "content": content})
    return ("\n\n".join(systems) or None), converted


class AnthropicProvider:
    """Adapter for Anthropic's Messages API."""

    def __init__(self, client: Any = None, client_factory: Any = None) -> None:
        self._client = client
        self._client_factory = client_factory

    def _client_for(self, config: ModelConfig) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            return self._client_factory(config)
        api_key = config.resolve_api_key()
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ConfigurationError("anthropic is required for anthropic models") from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return anthropic.Anthropic(**kwargs)

    def complete(
        self,
        config: ModelConfig,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Any] | None,
        response_schema: Mapping[str, Any] | None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        system, converted_messages = _anthropic_messages(messages)
        request: dict[str, Any] = {
            "model": config.model,
            "max_tokens": max_tokens,
            "messages": converted_messages,
        }
        if temperature != 0.0:
            request["temperature"] = temperature
        if system:
            request["system"] = system
        normalized_tools = normalize_anthropic_tools(tools)
        if normalized_tools:
            request["tools"] = normalized_tools
            request["tool_choice"] = {"type": "auto"}
        # Anthropic's Messages API has no OpenAI response_format parameter.
        # The shared response_schema remains provider-neutral and is handled by
        # the caller's prompt/checker when using this provider.
        del response_schema
        response = self._client_for(config).messages.create(**request)
        blocks = _get(response, "content", []) or []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        for block in blocks:
            block_type = _get(block, "type")
            if block_type == "text":
                text_parts.append(_text(_get(block, "text")))
            elif block_type == "tool_use":
                input_value = _get(block, "input", {})
                tool_calls.append(
                    {
                        "id": _get(block, "id"),
                        "type": "function",
                        "function": {
                            "name": _get(block, "name", ""),
                            "arguments": json.dumps(
                                input_value, sort_keys=True, separators=(",", ":")
                            ),
                        },
                    }
                )
            elif block_type in {"thinking", "reasoning"}:
                reasoning_parts.append(_text(_get(block, "thinking", _get(block, "text", ""))))
        usage = _usage_mapping(_get(response, "usage"))
        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            tokens_in=_usage_count(usage, ("input_tokens", "prompt_tokens")),
            tokens_out=_usage_count(usage, ("output_tokens", "completion_tokens")),
            finish_reason=_get(response, "stop_reason"),
            reasoning="".join(reasoning_parts) or None,
            usage=usage,
            raw=response,
        )


OpenAICompatProvider = OpenAICompatibleProvider
OpenAIProvider = OpenAICompatibleProvider


__all__ = [
    "AnthropicProvider",
    "OpenAICompatProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderResponse",
    "ProviderResponseError",
    "normalize_anthropic_tools",
    "normalize_openai_tools",
]
