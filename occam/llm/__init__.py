"""The provider-independent LLM interface used by Occam's engine.

The SDKs are imported lazily by the provider implementations.  This is
intentional: WP-15 initialises Neatlogs before importing ``openai`` so that its
automatic instrumentation can see the SDK import.
"""

# This import must stay ahead of the provider imports below.  It initializes
# Neatlogs (when configured) before the provider's lazy ``openai`` import.
from occam.llm.tracing import (  # noqa: I001
    flush,
    initialize,
    is_enabled,
    set_span_attributes,
    shutdown,
    span,
    trace_context,
)
from occam.llm.cache import DiskCache
from occam.llm.client import (
    Completion,
    CompletionError,
    LLMClient,
    LLMError,
    ProviderError,
    RetryPolicy,
    TruncatedCompletionError,
    complete,
    get_default_client,
    safe_provider_error_metadata,
)
from occam.llm.config import (
    ConfigurationError,
    MissingCredentialsError,
    ModelConfig,
    interpolate_env,
    load_config,
    load_environment,
    load_model_configs,
    load_models,
)
from occam.llm.cost import CostBreakdown, calculate_cost
from occam.llm.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    OpenAICompatProvider,
    OpenAIProvider,
    ProviderResponse,
)
from occam.llm.rate_limit import PerModelRateLimiter, TokenBucket

__all__ = [
    "AnthropicProvider",
    "Completion",
    "CompletionError",
    "ConfigurationError",
    "CostBreakdown",
    "DiskCache",
    "flush",
    "initialize",
    "is_enabled",
    "LLMClient",
    "LLMError",
    "MissingCredentialsError",
    "ModelConfig",
    "OpenAICompatibleProvider",
    "OpenAICompatProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderResponse",
    "RetryPolicy",
    "safe_provider_error_metadata",
    "set_span_attributes",
    "shutdown",
    "span",
    "trace_context",
    "TruncatedCompletionError",
    "complete",
    "calculate_cost",
    "get_default_client",
    "interpolate_env",
    "load_environment",
    "load_model_configs",
    "load_models",
    "load_config",
    "PerModelRateLimiter",
    "TokenBucket",
]
