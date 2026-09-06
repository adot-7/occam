"""Model configuration and environment interpolation for the LLM layer."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from occam.config.settings import project_root

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigurationError(ValueError):
    """Raised when the model configuration is invalid or incomplete."""


class MissingCredentialsError(ConfigurationError):
    """A live provider call was requested without its API key."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """The resolved settings needed for one provider/model lane.

    Prices are USD per million tokens.  ``grant_equiv_*`` are display-only
    rates for granted models; the nominal rates remain zero so callers can
    still distinguish granted usage from billed usage.
    """

    key: str
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    in_per_m: float = 0.0
    out_per_m: float = 0.0
    grant_equiv_in_per_m: float | None = None
    grant_equiv_out_per_m: float | None = None
    rpm: float | None = None
    tpm: float | None = None
    supports_json_schema: bool = True
    tool_choice_modes: tuple[str, ...] = ("auto",)
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if not self.key or not self.model or not self.provider:
            raise ConfigurationError("model config requires key, provider, and model")
        for name in (
            "in_per_m",
            "out_per_m",
            "grant_equiv_in_per_m",
            "grant_equiv_out_per_m",
            "rpm",
            "tpm",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ConfigurationError(f"{self.key}.{name} cannot be negative")
        if self.rpm == 0:
            raise ConfigurationError(f"{self.key}.rpm must be positive when provided")
        if self.tpm == 0:
            raise ConfigurationError(f"{self.key}.tpm must be positive when provided")
        if self.max_tokens < 1024:
            raise ConfigurationError(f"{self.key}.max_tokens must be at least 1024")
        if not self.tool_choice_modes:
            raise ConfigurationError(f"{self.key}.tool_choice_modes cannot be empty")
        if any(mode not in {"auto", "required", "none"} for mode in self.tool_choice_modes):
            raise ConfigurationError(f"{self.key}.tool_choice_modes contains an invalid mode")

    @classmethod
    def from_mapping(cls, key: str, values: Mapping[str, Any]) -> ModelConfig:
        """Build a validated config from one YAML mapping."""

        if not isinstance(values, Mapping):
            raise ConfigurationError(f"model {key!r} must be a mapping")
        modes = values.get("tool_choice_modes", ("auto",))
        if isinstance(modes, str):
            modes = (modes,)
        else:
            modes = tuple(modes)

        def optional_float(name: str) -> float | None:
            value = values.get(name)
            return None if value is None else float(value)

        return cls(
            key=key,
            provider=str(values.get("provider", "")),
            model=str(values.get("model", "")),
            base_url=_optional_string(values.get("base_url")),
            api_key=_optional_string(values.get("api_key")),
            api_key_env=_optional_string(values.get("api_key_env")),
            in_per_m=float(values.get("in_per_m", 0.0)),
            out_per_m=float(values.get("out_per_m", 0.0)),
            grant_equiv_in_per_m=optional_float("grant_equiv_in_per_m"),
            grant_equiv_out_per_m=optional_float("grant_equiv_out_per_m"),
            rpm=optional_float("rpm"),
            tpm=optional_float("tpm"),
            supports_json_schema=bool(values.get("supports_json_schema", True)),
            tool_choice_modes=modes,
            max_tokens=int(values.get("max_tokens", 2048)),
        )

    def resolve_api_key(self, environ: Mapping[str, str] | None = None) -> str:
        """Resolve the configured key without including it in error messages."""

        env = os.environ if environ is None else environ
        value = self.api_key
        if value and (match := _ENV_PATTERN.fullmatch(value)):
            variable, default = match.groups()
            value = env.get(variable, default)
        if not value and self.api_key_env:
            value = env.get(self.api_key_env)
        if not value:
            env_name = self.api_key_env or _placeholder_name(self.api_key) or "the configured key"
            raise MissingCredentialsError(f"credentials unavailable for {self.key}; set {env_name}")
        if _ENV_PATTERN.search(value):
            raise MissingCredentialsError(
                f"credentials unavailable for {self.key}; resolve its environment settings"
            )
        return value

    @property
    def is_granted(self) -> bool:
        """Whether at least one nominal rate is replaced by an equivalent rate."""

        return (self.in_per_m == 0 and self.grant_equiv_in_per_m is not None) or (
            self.out_per_m == 0 and self.grant_equiv_out_per_m is not None
        )

    def display_rate(self, direction: str) -> float:
        """Return the rate used in the displayed cost for input or output."""

        if direction == "input":
            nominal, equivalent = self.in_per_m, self.grant_equiv_in_per_m
        elif direction == "output":
            nominal, equivalent = self.out_per_m, self.grant_equiv_out_per_m
        else:
            raise ValueError("direction must be 'input' or 'output'")
        if nominal == 0 and equivalent is not None:
            return equivalent
        return nominal

    def cache_identity(self) -> dict[str, Any]:
        """Return non-secret config fields that affect a completion."""

        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "supports_json_schema": self.supports_json_schema,
            "tool_choice_modes": list(self.tool_choice_modes),
        }


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _placeholder_name(value: str | None) -> str | None:
    if value and (match := _ENV_PATTERN.fullmatch(value)):
        return match.group(1)
    return None


def interpolate_env(
    value: Any,
    environ: Mapping[str, str] | None = None,
    *,
    strict: bool = False,
) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in config data.

    Missing variables are left as placeholders by default.  This lets offline
    commands load the model table and produce a useful credential error only
    when a network completion is actually requested.  ``strict=True`` is
    available to CI and callers that want configuration validation up front.
    """

    env = os.environ if environ is None else environ
    if isinstance(value, str):
        missing: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name, default = match.groups()
            if name in env:
                return env[name]
            if default is not None:
                return default
            missing.append(name)
            return match.group(0)

        result = _ENV_PATTERN.sub(replace, value)
        if strict and missing:
            names = ", ".join(sorted(set(missing)))
            raise ConfigurationError(f"environment variable(s) not set: {names}")
        return result
    if isinstance(value, Mapping):
        return {key: interpolate_env(item, env, strict=strict) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate_env(item, env, strict=strict) for item in value]
    if isinstance(value, tuple):
        return tuple(interpolate_env(item, env, strict=strict) for item in value)
    return value


def default_models_path() -> Path:
    """Locate the checked-in model table."""

    return project_root() / "occam" / "config" / "models.yaml"


def load_model_configs(
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    strict_env: bool = False,
) -> dict[str, ModelConfig]:
    """Load and validate all model lanes from a YAML file."""

    config_path = default_models_path() if path is None else Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read model config {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in model config {config_path}") from exc

    expanded = interpolate_env(raw, environ, strict=strict_env)
    if not isinstance(expanded, Mapping) or not expanded:
        raise ConfigurationError("model config must contain at least one model")
    return {
        str(key): ModelConfig.from_mapping(str(key), values) for key, values in expanded.items()
    }


def load_models(
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    strict_env: bool = False,
) -> dict[str, ModelConfig]:
    """Backward-compatible short alias for :func:`load_model_configs`."""

    return load_model_configs(path, environ, strict_env=strict_env)


load_config = load_model_configs


__all__ = [
    "ConfigurationError",
    "MissingCredentialsError",
    "ModelConfig",
    "default_models_path",
    "interpolate_env",
    "load_config",
    "load_model_configs",
    "load_models",
]
