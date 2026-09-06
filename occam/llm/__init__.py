"""The provider-independent LLM interface used by Occam's engine.

The SDKs are imported lazily by the provider implementations.  This is
intentional: WP-15 initialises Neatlogs before importing ``openai`` so that its
automatic instrumentation can see the SDK import.
"""

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
