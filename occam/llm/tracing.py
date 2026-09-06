"""Optional Neatlogs lifecycle and manual span helpers.

Neatlogs is deliberately loaded only when ``NEATLOGS_API_KEY`` is present.  In
particular, importing Occam without a key must not make the optional provider
SDKs less lazy or make offline commands depend on the Neatlogs package being
installed.  When enabled, this module is imported before the provider adapters
and initializes Neatlogs before the first lazy ``openai`` import.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from sys import exc_info
from typing import Any

_ATTRIBUTE_PREFIX = "occam."
_TRACE_ATTRIBUTES: ContextVar[dict[str, Any] | None] = ContextVar(
    "occam_trace_attributes", default=None
)

# Keep the SDK module out of ``sys.modules`` on the no-key path.  Besides being
# cheaper for offline commands, this preserves the provider SDK import order
# that Neatlogs' OpenAI instrumentation requires.
neatlogs: Any | None = None
ENABLED = False
_initialization_attempted = False
_shutdown_complete = False


def initialize() -> bool:
    """Initialize Neatlogs once when a non-empty API key is configured.

    The exact initialization call is kept here, before any provider can import
    ``openai``.  A missing optional SDK is treated like disabled tracing so a
    wheel installed without runtime extras still supports offline commands.
    SDK initialization failures are also isolated from the application: tracing
    must never take down an otherwise valid run.
    """

    global ENABLED, _initialization_attempted, _shutdown_complete, neatlogs
    if ENABLED:
        return True
    api_key = os.environ.get("NEATLOGS_API_KEY")
    if not api_key:
        return False
    if _initialization_attempted:
        return False
    _initialization_attempted = True
    try:
        import neatlogs as sdk
    except ImportError:
        return False
    try:
        sdk.init(
            api_key=api_key,
            workflow_name="occam",
            instrumentations=["openai"],
        )
    except Exception:  # noqa: BLE001 - observability must not break a run
        return False
    neatlogs = sdk
    ENABLED = True
    _shutdown_complete = False
    return True


def is_enabled() -> bool:
    """Return whether manual spans can currently be emitted."""

    return ENABLED and not _shutdown_complete and neatlogs is not None


def _attribute_value(value: Any) -> Any:
    """Coerce an attribute into a value accepted by OpenTelemetry spans."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def set_span_attributes(span_object: Any, attributes: Mapping[str, Any]) -> None:
    """Set Occam attributes on a Neatlogs/OTel span without leaking failures."""

    if span_object is None:
        return
    setter = getattr(span_object, "set_attribute", None)
    if not callable(setter):
        return
    for key, value in attributes.items():
        value = _attribute_value(value)
        if value is None:
            continue
        name = str(key)
        if not name.startswith(_ATTRIBUTE_PREFIX):
            name = f"{_ATTRIBUTE_PREFIX}{name}"
        try:
            setter(name, value)
        except Exception:  # noqa: BLE001 - an attribute cannot break the run
            continue


@contextmanager
def trace_context(
    attributes: Mapping[str, Any] | None = None,
    **extra_attributes: Any,
) -> Iterator[None]:
    """Make run/case/role metadata available to nested spans.

    Context variables are copied by ``asyncio`` tasks and ``asyncio.to_thread``
    workers, which keeps concurrent cases and roles associated with the right
    trace without changing the engine's event contract.
    """

    if not is_enabled():
        yield None
        return
    merged = dict(_TRACE_ATTRIBUTES.get() or {})
    if attributes:
        merged.update(attributes)
    merged.update(extra_attributes)
    token = _TRACE_ATTRIBUTES.set(merged)
    try:
        yield None
    finally:
        _TRACE_ATTRIBUTES.reset(token)


@contextmanager
def span(
    name: str,
    *,
    kind: str = "CHAIN",
    attributes: Mapping[str, Any] | None = None,
    **extra_attributes: Any,
) -> Iterator[Any | None]:
    """Create one manual Neatlogs span, or yield ``None`` when disabled."""

    if not is_enabled():
        yield None
        return

    merged = dict(_TRACE_ATTRIBUTES.get() or {})
    if attributes:
        merged.update(attributes)
    merged.update(extra_attributes)
    sdk_trace = getattr(neatlogs, "trace", None)
    if not callable(sdk_trace):
        yield None
        return

    try:
        context_manager = sdk_trace(name, kind=kind)
        span_object = context_manager.__enter__()
    except Exception:  # noqa: BLE001 - tracing is best effort
        yield None
        return

    try:
        set_span_attributes(span_object, merged)
        yield span_object
    except BaseException:
        try:
            context_manager.__exit__(*exc_info())
        except Exception:  # noqa: BLE001 - tracing is best effort
            pass
        raise
    else:
        try:
            context_manager.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - tracing is best effort
            pass


def flush() -> bool:
    """Flush buffered spans, returning ``False`` only when the SDK reports failure."""

    if not is_enabled():
        return False
    try:
        result = neatlogs.flush()
    except Exception:  # noqa: BLE001 - tracing is best effort
        return False
    return result is not False


def shutdown() -> bool:
    """Flush and stop the Neatlogs exporter at the end of a run."""

    global ENABLED, _shutdown_complete
    if not is_enabled():
        return False
    success = flush()
    try:
        neatlogs.shutdown()
    except Exception:  # noqa: BLE001 - tracing is best effort
        success = False
    _shutdown_complete = True
    ENABLED = False
    return success


# Importing ``occam.llm`` is the earliest common point for the provider
# adapters.  The provider also calls initialize immediately before its lazy
# OpenAI import to cover applications that set the environment later.
initialize()


__all__ = [
    "ENABLED",
    "flush",
    "initialize",
    "is_enabled",
    "set_span_attributes",
    "shutdown",
    "span",
    "trace_context",
]
