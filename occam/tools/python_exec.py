"""Run agent-written arithmetic in a bounded child process.

``python_exec`` is intentionally a *capability-limited calculation surface*,
not a general Python sandbox.  The child parses source and interprets only an
allow-listed AST; it never compiles or executes the submitted Python.  It gets
an explicit environment allow-list and capability wrappers for Decimal, JSON,
date/time, math, and ordinary arithmetic.  Timeout, bounded output, a
temporary working directory, and strict value/step limits remain defense in
depth.  There are no user-visible modules, builtins, Python frames, or
callable objects that could reach the filesystem, a process, or the network.

The boundary is not perfect host isolation.  This module must not be described
as safe for arbitrary hostile Python: unsupported syntax is rejected, but a
bug in this evaluator or resource exhaustion still needs OS-level controls in
production (for example a container, separate user, seccomp/job controls, or
a dedicated sandbox service).  The temporary directory is a convenience, not
a filesystem-security claim; the security property here is that submitted
source is never executed as Python and receives no file/process/network
capability.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000

#: The only variables the child inherits: what a Python interpreter needs to
#: start, and nothing else.  ``-I`` ignores ``PYTHON*`` variables but does not
#: scrub the environment generally, so the allow-list does it explicitly.
ENV_PASSTHROUGH: tuple[str, ...] = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR")

# When a POSIX child starts with no locale variables, CPython may synthesize
# this key while initialising its locale/filesystem encoding.  It is not passed
# through by us and must not be added to ``ENV_PASSTHROUGH``.
CPYTHON_SYNTHESIZED_ENV: frozenset[str] = frozenset({"LC_CTYPE"})

#: Module names recognized by the calculator.  They are capability wrappers,
#: not imports performed by submitted code; everything outside this set is
#: rejected before evaluation.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "datetime",
        "decimal",
        "json",
        "math",
    }
)

NETWORK_BLOCKED_MESSAGE = "python_exec: network access is disabled"
TIMEOUT_MESSAGE = "python_exec: timed out after {timeout:g}s"

# Runs inside the child.  It reads the agent's program from stdin so the code
# never has to survive a round trip through shell quoting.
_BOOTSTRAP = '''
import ast
import datetime as _datetime
import decimal as _decimal
import json as _json
import math as _math
import sys

ALLOWED = frozenset(__ALLOWED__)
NETWORK_BLOCKED_MESSAGE = __NETWORK_MESSAGE__
MAX_OUTPUT_CHARS = __MAX_OUTPUT__
MAX_SOURCE_CHARS = 1_000_000

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class _BoundedTextIO:
    """Keep untrusted writes bounded before they reach the parent process."""

    def __init__(self, stream, limit):
        self._stream = stream
        self._limit = limit
        self._written = 0
        self._truncated = False
        self.encoding = stream.encoding
        self.errors = stream.errors

    def write(self, value):
        if not isinstance(value, str):
            value = str(value)
        if self._limit <= 0 or self._truncated:
            return len(value)
        remaining = self._limit - self._written
        if len(value) <= remaining:
            self._stream.write(value)
            self._written += len(value)
            return len(value)
        marker = "\\n... [truncated at %d characters]" % self._limit
        keep = max(0, remaining - len(marker))
        if keep:
            self._stream.write(value[:keep])
        self._stream.write(marker)
        self._stream.flush()
        self._written = self._limit
        self._truncated = True
        return len(value)

    def flush(self):
        self._stream.flush()


if MAX_OUTPUT_CHARS > 0:
    sys.stdout = _BoundedTextIO(sys.stdout, MAX_OUTPUT_CHARS)
    sys.stderr = _BoundedTextIO(sys.stderr, MAX_OUTPUT_CHARS)

source = sys.stdin.read()
if len(source) > MAX_SOURCE_CHARS:
    sys.stderr.write(
        "python_exec: source exceeds %d characters\\n" % MAX_SOURCE_CHARS
    )
    raise SystemExit(1)


def _reject(name):
    raise RuntimeError("python_exec: import of %r is not allowed" % name)


_FORBIDDEN_NAMES = frozenset(
    {
        "breakpoint",
        "compile",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "hasattr",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "object",
        "setattr",
        "delattr",
        "type",
        "vars",
    }
)
_FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "attrgetter",
        "convert_field",
        "format",
        "format_field",
        "format_map",
        "get_field",
        "get_value",
        "methodcaller",
        "vformat",
        # Generator/coroutine/traceback frames expose f_globals, f_builtins,
        # and f_back even though those names do not start with an underscore.
        "ag_await",
        "ag_code",
        "ag_frame",
        "cr_await",
        "cr_code",
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "gi_code",
        "gi_frame",
        "gi_yieldfrom",
        "tb_frame",
        "tb_lasti",
        "tb_lineno",
        "tb_next",
    }
)


def _reject_policy(kind, name):
    raise RuntimeError("python_exec: %s %r is not allowed" % (kind, name))


class _Policy(ast.NodeVisitor):
    """Reject syntax that can recover interpreter capabilities indirectly."""

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in ALLOWED:
                _reject(alias.name)
            if alias.asname and alias.asname.startswith("_"):
                _reject_policy("import alias", alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level or node.module not in ALLOWED:
            _reject(node.module or "<relative>")
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_"):
                _reject_policy("import name", alias.name)
            if alias.asname and alias.asname.startswith("_"):
                _reject_policy("import alias", alias.asname)
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("__"):
            _reject_policy("dunder identifier", node.id)
        if node.id in _FORBIDDEN_NAMES:
            _reject_policy("builtin", node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Reject all private/dunder attributes, not just the common examples:
        # this covers __globals__, __subclasses__, traceback frames, and module
        # loaders without trying to maintain a complete object-model catalogue.
        if node.attr.startswith("_"):
            _reject_policy("private attribute", node.attr)
        if node.attr in _FORBIDDEN_ATTRIBUTES:
            _reject_policy("attribute", node.attr)
        self.generic_visit(node)

    def visit_Global(self, node):
        for name in node.names:
            if name.startswith("_"):
                _reject_policy("private global", name)
        self.generic_visit(node)

    def visit_Nonlocal(self, node):
        for name in node.names:
            if name.startswith("_"):
                _reject_policy("private nonlocal", name)
        self.generic_visit(node)


class _EvalError(RuntimeError):
    pass


class _UserRaised(Exception):
    def __init__(self, name, message):
        super().__init__(message)
        self.name = name
        self.message = message


class _ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class _SafeException:
    __slots__ = ("name", "message")

    def __init__(self, name, message=""):
        self.name = name
        self.message = str(message)


class _Capability:
    __slots__ = ("label", "function", "attributes", "allow_wrappers")

    def __init__(self, label, function, attributes=None, allow_wrappers=False):
        self.label = label
        self.function = function
        self.attributes = attributes or {}
        self.allow_wrappers = allow_wrappers


class _SafeModule:
    __slots__ = ("name", "values")

    def __init__(self, name, values):
        self.name = name
        self.values = values


class _Environment:
    __slots__ = ("values", "parent")

    def __init__(self, parent=None):
        self.values = {}
        self.parent = parent

    def get(self, name):
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise _EvalError("python_exec: name %r is not available" % name)

    def set(self, name, value):
        self.values[name] = value


class _UserFunction:
    __slots__ = ("evaluator", "name", "arguments", "body", "closure", "expression")

    def __init__(self, evaluator, name, arguments, body, closure, expression=False):
        self.evaluator = evaluator
        self.name = name
        self.arguments = arguments
        self.body = body
        self.closure = closure
        self.expression = expression


def _is_wrapper(value):
    return isinstance(value, (_Capability, _SafeModule, _UserFunction))


def _validate_value(value, depth=0):
    if depth > 64:
        raise _EvalError("python_exec: value nesting limit exceeded")
    if _is_wrapper(value) or isinstance(value, _SafeException):
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, (str, bytes)):
        if len(value) > _Evaluator.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string value limit exceeded")
        return
    if isinstance(
        value, (_decimal.Decimal, _datetime.date, _datetime.datetime, _datetime.timedelta)
    ):
        return
    if isinstance(value, (list, tuple, set, frozenset, range)):
        if len(value) > _Evaluator.MAX_ITEMS:
            raise _EvalError("python_exec: collection item limit exceeded")
        for item in value:
            _validate_value(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _Evaluator.MAX_ITEMS:
            raise _EvalError("python_exec: collection item limit exceeded")
        for key, item in value.items():
            _validate_value(key, depth + 1)
            _validate_value(item, depth + 1)
        return
    raise _EvalError("python_exec: value type %r is not available" % type(value).__name__)


class _Evaluator:
    MAX_STEPS = 10_000_000
    MAX_ITEMS = 100_000
    MAX_STRING_CHARS = 1_000_000

    def __init__(self, tree):
        self.tree = tree
        self.steps = 0
        self.global_env = _Environment()
        self.modules = self._make_modules()
        self.global_env.values.update(self._make_builtins())

    def _tick(self):
        self.steps += 1
        if self.steps > self.MAX_STEPS:
            raise _EvalError("python_exec: calculation step limit exceeded")

    def _capability(self, label, function, attributes=None, allow_wrappers=False):
        return _Capability(label, function, attributes, allow_wrappers)

    def _make_modules(self):
        decimal_cap = self._capability("decimal.Decimal", _decimal.Decimal)
        decimal_cap.attributes.update(
            {
                "from_float": self._capability(
                    "decimal.Decimal.from_float", _decimal.Decimal.from_float
                ),
            }
        )

        date_cap = self._capability("datetime.date", _datetime.date)
        date_cap.attributes.update(
            {
                "fromisoformat": self._capability(
                    "datetime.date.fromisoformat", _datetime.date.fromisoformat
                ),
            }
        )
        datetime_cap = self._capability("datetime.datetime", _datetime.datetime)
        datetime_cap.attributes.update(
            {
                "fromisoformat": self._capability(
                    "datetime.datetime.fromisoformat", _datetime.datetime.fromisoformat
                ),
                "strptime": self._capability(
                    "datetime.datetime.strptime", _datetime.datetime.strptime
                ),
            }
        )

        return {
            "decimal": _SafeModule(
                "decimal",
                {
                    "Decimal": decimal_cap,
                    "ROUND_DOWN": _decimal.ROUND_DOWN,
                    "ROUND_HALF_EVEN": _decimal.ROUND_HALF_EVEN,
                    "ROUND_HALF_UP": _decimal.ROUND_HALF_UP,
                    "ROUND_UP": _decimal.ROUND_UP,
                },
            ),
            "datetime": _SafeModule(
                "datetime",
                {
                    "date": date_cap,
                    "datetime": datetime_cap,
                    "timedelta": self._capability("datetime.timedelta", _datetime.timedelta),
                },
            ),
            "json": _SafeModule(
                "json",
                {
                    "dumps": self._capability(
                        "json.dumps", self._json_dumps, allow_wrappers=True
                    ),
                    "loads": self._capability("json.loads", self._json_loads),
                },
            ),
            "math": _SafeModule(
                "math",
                {
                    "e": _math.e,
                    "pi": _math.pi,
                    "ceil": self._capability("math.ceil", _math.ceil),
                    "fabs": self._capability("math.fabs", _math.fabs),
                    "floor": self._capability("math.floor", _math.floor),
                    "isclose": self._capability("math.isclose", _math.isclose),
                    "isfinite": self._capability("math.isfinite", _math.isfinite),
                    "isinf": self._capability("math.isinf", _math.isinf),
                    "isnan": self._capability("math.isnan", _math.isnan),
                    "sqrt": self._capability("math.sqrt", _math.sqrt),
                },
            ),
        }

    def _make_builtins(self):
        values = {
            "abs": self._capability("abs", lambda value: abs(self._plain(value))),
            "all": self._capability("all", self._all),
            "any": self._capability("any", self._any),
            "ascii": self._capability("ascii", self._safe_ascii, allow_wrappers=True),
            "bin": self._capability("bin", lambda value: bin(self._plain(value))),
            "bool": self._capability("bool", lambda value=False: bool(self._plain(value))),
            "bytes": self._capability("bytes", self._safe_bytes),
            "callable": self._capability("callable", lambda value: _is_wrapper(value)),
            "chr": self._capability("chr", lambda value: chr(self._plain(value))),
            "dict": self._capability("dict", self._make_dict),
            "divmod": self._capability(
                "divmod", lambda left, right: divmod(self._plain(left), self._plain(right))
            ),
            "enumerate": self._capability("enumerate", self._enumerate),
            "filter": self._capability("filter", self._filter),
            "float": self._capability("float", lambda value=0: float(self._plain(value))),
            "frozenset": self._capability("frozenset", self._make_frozenset),
            "hash": self._capability("hash", lambda value: hash(self._plain(value))),
            "hex": self._capability("hex", lambda value: hex(self._plain(value))),
            "int": self._capability("int", self._safe_int),
            "len": self._capability("len", lambda value: len(self._plain(value))),
            "list": self._capability("list", self._make_list),
            "map": self._capability("map", self._map),
            "max": self._capability("max", self._maximum),
            "min": self._capability("min", self._minimum),
            "oct": self._capability("oct", lambda value: oct(self._plain(value))),
            "ord": self._capability("ord", lambda value: ord(self._plain(value))),
            "pow": self._capability(
                "pow", lambda left, right, modulo=None: self._safe_pow(left, right, modulo)
            ),
            "print": self._capability("print", self._print, allow_wrappers=True),
            "range": self._capability("range", self._range),
            "repr": self._capability("repr", self._safe_repr, allow_wrappers=True),
            "reversed": self._capability("reversed", self._reversed),
            "round": self._capability("round", self._round),
            "set": self._capability("set", self._make_set),
            "sorted": self._capability("sorted", self._sorted),
            "str": self._capability("str", self._safe_str, allow_wrappers=True),
            "sum": self._capability("sum", self._sum),
            "tuple": self._capability("tuple", self._make_tuple),
            "zip": self._capability("zip", self._zip),
        }
        for name in (
            "ArithmeticError",
            "AssertionError",
            "AttributeError",
            "Exception",
            "IndexError",
            "KeyError",
            "LookupError",
            "OSError",
            "OverflowError",
            "RuntimeError",
            "StopIteration",
            "TypeError",
            "ValueError",
            "ZeroDivisionError",
        ):
            values[name] = self._capability(
                name, self._exception_factory(name), allow_wrappers=True
            )
        return values

    def _exception_factory(self, name):
        def make(*args):
            if len(args) > 1:
                raise _EvalError("python_exec: exception constructors accept one message")
            return _SafeException(name, "" if not args else self._safe_str(args[0]))

        return make

    def _plain(self, value):
        if _is_wrapper(value):
            raise _EvalError("python_exec: capability values cannot be used as data")
        _validate_value(value)
        return value

    def _safe_str(self, value):
        if _is_wrapper(value):
            return self._display(value)
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            return self._display(value)
        return str(value)

    def _safe_repr(self, value):
        if _is_wrapper(value):
            return self._display(value)
        return self._repr_value(value)

    def _safe_ascii(self, value):
        return ascii(self._plain(value))

    def _safe_bytes(self, value=b"", encoding=None, errors="strict"):
        if encoding is None:
            return bytes(self._plain(value))
        return bytes(self._plain(value), self._plain(encoding), self._plain(errors))

    def _safe_int(self, value=0, base=10):
        value = self._plain(value)
        if isinstance(value, str):
            return int(value, self._plain(base))
        return int(value)

    def _safe_pow(self, left, right, modulo=None):
        left = self._plain(left)
        right = self._plain(right)
        if isinstance(right, int) and abs(right) > 10_000:
            raise _EvalError("python_exec: exponent limit exceeded")
        if modulo is None:
            return pow(left, right)
        return pow(left, right, self._plain(modulo))

    def _range(self, *args):
        values = tuple(self._plain(value) for value in args)
        result = range(*values)
        try:
            size = len(result)
        except OverflowError as exc:
            raise _EvalError("python_exec: range limit exceeded") from exc
        if size > self.MAX_ITEMS:
            raise _EvalError("python_exec: range limit exceeded")
        return result

    def _iter_values(self, value):
        self._plain(value)
        if isinstance(value, dict):
            return list(value)
        if isinstance(value, (list, tuple, set, frozenset, str, bytes, range)):
            return value
        raise _EvalError("python_exec: value is not iterable")

    def _make_list(self, value=()):
        return list(self._iter_values(value))

    def _make_tuple(self, value=()):
        return tuple(self._iter_values(value))

    def _make_set(self, value=()):
        return set(self._iter_values(value))

    def _make_frozenset(self, value=()):
        return frozenset(self._iter_values(value))

    def _make_dict(self, value=(), **kwargs):
        if value == ():
            result = {}
        elif isinstance(value, dict):
            result = dict(value)
        else:
            result = dict(self._iter_values(value))
        result.update(kwargs)
        _validate_value(result)
        return result

    def _all(self, value):
        return all(self._iter_values(value))

    def _any(self, value):
        return any(self._iter_values(value))

    def _enumerate(self, value, start=0):
        return list(enumerate(self._iter_values(value), self._plain(start)))

    def _filter(self, function, value):
        items = self._iter_values(value)
        return [item for item in items if self._truth(self._invoke(function, (item,), {}))]

    def _map(self, function, *values):
        items = [self._iter_values(value) for value in values]
        return [self._invoke(function, tuple(row), {}) for row in zip(*items)]

    def _zip(self, *values):
        return list(zip(*(self._iter_values(value) for value in values)))

    def _reversed(self, value):
        return list(reversed(self._iter_values(value)))

    def _sorted(self, value, reverse=False):
        items = list(self._iter_values(value))
        reverse = bool(self._plain(reverse))
        return sorted(items, reverse=reverse)

    def _minimum(self, *values, **kwargs):
        return self._extreme(min, values, kwargs)

    def _maximum(self, *values, **kwargs):
        return self._extreme(max, values, kwargs)

    def _extreme(self, function, values, kwargs):
        default = kwargs.pop("default", None)
        if kwargs:
            raise _EvalError("python_exec: unsupported min/max keyword")
        if len(values) == 1:
            values = tuple(self._iter_values(values[0]))
        if not values:
            if default is not None:
                return default
            raise _EvalError("python_exec: min/max needs a value")
        return function(*(self._plain(value) for value in values))

    def _round(self, value, ndigits=None):
        value = self._plain(value)
        if ndigits is None:
            return round(value)
        return round(value, self._plain(ndigits))

    def _sum(self, value, start=0):
        total = self._plain(start)
        for item in self._iter_values(value):
            total = self._binary(ast.Add(), total, self._plain(item))
        return total

    def _print(self, *values, sep=" ", end="\\n"):
        sep = self._plain(sep)
        end = self._plain(end)
        if not isinstance(sep, str) or not isinstance(end, str):
            raise _EvalError("python_exec: print separators must be strings")
        sys.stdout.write(sep.join(self._display(value) for value in values) + end)
        sys.stdout.flush()

    def _display(self, value):
        if isinstance(value, _Capability):
            return "<capability %s>" % value.label
        if isinstance(value, _SafeModule):
            return "<module %s>" % value.name
        if isinstance(value, _UserFunction):
            return "<function %s>" % value.name
        if isinstance(value, _SafeException):
            return "%s: %s" % (value.name, value.message)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set, frozenset)):
            left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
            if isinstance(value, (set, frozenset)):
                left, right = ("{", "}")
            return left + ", ".join(self._repr_value(item) for item in value) + right
        if isinstance(value, dict):
            return "{" + ", ".join(
                "%s: %s" % (self._repr_value(key), self._repr_value(item))
                for key, item in value.items()
            ) + "}"
        return str(value)

    def _repr_value(self, value):
        if _is_wrapper(value):
            return self._display(value)
        if isinstance(value, _SafeException):
            return self._display(value)
        if isinstance(value, dict):
            return self._display(value)
        if isinstance(value, (list, tuple, set, frozenset)):
            return self._display(value)
        return repr(value)

    def _json_plain(self, value, depth=0):
        if depth > 64:
            raise _EvalError("python_exec: JSON nesting limit exceeded")
        if _is_wrapper(value) or isinstance(value, _SafeException):
            raise _EvalError("python_exec: JSON cannot contain capabilities")
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, (_decimal.Decimal, _datetime.date, _datetime.datetime)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._json_plain(item, depth + 1) for item in value]
        if isinstance(value, dict):
            return {
                self._json_plain(key, depth + 1): self._json_plain(item, depth + 1)
                for key, item in value.items()
            }
        raise _EvalError("python_exec: JSON value is not available")

    def _json_dumps(self, value, **kwargs):
        allowed = {
            "allow_nan",
            "check_circular",
            "default",
            "ensure_ascii",
            "indent",
            "separators",
            "skipkeys",
            "sort_keys",
        }
        if set(kwargs) - allowed:
            raise _EvalError("python_exec: unsupported json.dumps option")
        default = kwargs.get("default")
        if isinstance(default, (_Capability, _UserFunction)):
            kwargs["default"] = lambda item: self._invoke(default, (item,), {})
        elif default is not None:
            raise _EvalError("python_exec: json default must be an allowed function")
        return _json.dumps(self._json_plain(value), **kwargs)

    def _json_loads(self, value):
        return _json.loads(self._plain(value))

    def _bound_method(self, value, name):
        def call(*args, **kwargs):
            raw_args = tuple(self._plain(arg) for arg in args)
            raw_kwargs = {key: self._plain(item) for key, item in kwargs.items()}
            return getattr(value, name)(*raw_args, **raw_kwargs)

        return self._capability("%s.%s" % (type(value).__name__, name), call)

    def _attribute(self, value, name):
        if name.startswith("_") or name in {"format", "format_map"}:
            raise _EvalError("python_exec: attribute %r is not available" % name)
        if isinstance(value, _SafeModule):
            if name not in value.values:
                raise _EvalError("python_exec: module attribute %r is not available" % name)
            return value.values[name]
        if isinstance(value, _Capability):
            if name not in value.attributes:
                raise _EvalError("python_exec: capability attribute %r is not available" % name)
            return value.attributes[name]
        if isinstance(value, _decimal.Decimal):
            if name in {
                "adjusted",
                "copy_abs",
                "copy_negate",
                "is_finite",
                "is_infinite",
                "is_nan",
                "is_normal",
                "is_zero",
                "normalize",
                "quantize",
                "to_integral_value",
            }:
                return self._bound_method(value, name)
        if isinstance(value, _datetime.datetime):
            if name == "date":
                return self._capability("datetime.datetime.date", value.date)
            if name in {"day", "hour", "microsecond", "minute", "month", "second", "year"}:
                return getattr(value, name)
            if name in {"isoformat", "strftime", "timestamp", "weekday", "isoweekday"}:
                return self._bound_method(value, name)
        if isinstance(value, _datetime.date):
            if name in {"day", "month", "year"}:
                return getattr(value, name)
            if name in {"isoformat", "strftime", "toordinal", "weekday", "isoweekday"}:
                return self._bound_method(value, name)
        if isinstance(value, _datetime.timedelta):
            if name in {"days", "seconds", "microseconds"}:
                return getattr(value, name)
            if name == "total_seconds":
                return self._bound_method(value, name)
        if isinstance(value, dict):
            if name in {"clear", "copy", "get", "items", "keys", "pop", "setdefault", "values"}:
                if name == "items":
                    return self._capability("dict.items", lambda: list(value.items()))
                if name == "keys":
                    return self._capability("dict.keys", lambda: list(value.keys()))
                if name == "values":
                    return self._capability("dict.values", lambda: list(value.values()))
                return self._bound_method(value, name)
        if isinstance(value, list):
            if name in {
                "append",
                "clear",
                "copy",
                "count",
                "extend",
                "index",
                "insert",
                "pop",
                "remove",
                "reverse",
            }:
                return self._bound_method(value, name)
        if isinstance(value, tuple):
            if name in {"count", "index"}:
                return self._bound_method(value, name)
        if isinstance(value, str):
            if name in {
                "capitalize",
                "casefold",
                "center",
                "count",
                "encode",
                "endswith",
                "find",
                "index",
                "isalnum",
                "isalpha",
                "isdigit",
                "islower",
                "isspace",
                "isupper",
                "join",
                "lower",
                "lstrip",
                "partition",
                "replace",
                "rfind",
                "rindex",
                "rsplit",
                "rstrip",
                "split",
                "splitlines",
                "startswith",
                "strip",
                "swapcase",
                "title",
                "upper",
                "zfill",
            }:
                return self._bound_method(value, name)
        if isinstance(value, bytes):
            if name in {"decode", "endswith", "hex", "startswith"}:
                return self._bound_method(value, name)
        if isinstance(value, range):
            if name in {"start", "stop", "step"}:
                return getattr(value, name)
        raise _EvalError("python_exec: attribute %r is not available" % name)

    def _getitem(self, value, key):
        self._plain(key)
        if not isinstance(value, (dict, list, tuple, str, bytes, range)):
            raise _EvalError("python_exec: subscription is not available")
        try:
            return value[key]
        except (IndexError, KeyError, TypeError) as exc:
            raise _EvalError("python_exec: subscription failed: %s" % exc) from exc

    def _setitem(self, value, key, item):
        self._plain(key)
        _validate_value(item)
        if isinstance(value, (dict, list)):
            try:
                value[key] = item
            except (IndexError, KeyError, TypeError) as exc:
                raise _EvalError("python_exec: assignment failed: %s" % exc) from exc
            return
        raise _EvalError("python_exec: subscription assignment is not available")

    def _binary(self, operator, left, right):
        left = self._plain(left)
        right = self._plain(right)
        if isinstance(operator, ast.Mult):
            for sequence, multiplier in ((left, right), (right, left)):
                if isinstance(sequence, (str, bytes, list, tuple)) and isinstance(
                    multiplier, int
                ):
                    limit = (
                        self.MAX_STRING_CHARS
                        if isinstance(sequence, (str, bytes))
                        else self.MAX_ITEMS
                    )
                    if len(sequence) * abs(multiplier) > limit:
                        raise _EvalError("python_exec: multiplication result limit exceeded")
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.Pow: lambda: left**right,
            ast.BitOr: lambda: left | right,
            ast.BitAnd: lambda: left & right,
            ast.BitXor: lambda: left ^ right,
            ast.LShift: lambda: left << right,
            ast.RShift: lambda: left >> right,
        }
        operation = operations.get(type(operator))
        if operation is None:
            raise _EvalError("python_exec: binary operator is not available")
        try:
            result = operation()
        except Exception as exc:
            raise _EvalError("python_exec: calculation failed: %s" % exc) from exc
        _validate_value(result)
        return result

    def _unary(self, operator, value):
        value = self._plain(value)
        operations = {
            ast.UAdd: lambda: +value,
            ast.USub: lambda: -value,
            ast.Invert: lambda: ~value,
            ast.Not: lambda: not value,
        }
        operation = operations.get(type(operator))
        if operation is None:
            raise _EvalError("python_exec: unary operator is not available")
        result = operation()
        _validate_value(result)
        return result

    def _truth(self, value):
        if _is_wrapper(value):
            raise _EvalError("python_exec: capability values have no truth value")
        return bool(value)

    def _compare(self, operator, left, right):
        left = self._plain(left)
        right = self._plain(right)
        operations = {
            ast.Eq: lambda: left == right,
            ast.NotEq: lambda: left != right,
            ast.Lt: lambda: left < right,
            ast.LtE: lambda: left <= right,
            ast.Gt: lambda: left > right,
            ast.GtE: lambda: left >= right,
            ast.In: lambda: left in right,
            ast.NotIn: lambda: left not in right,
            ast.Is: lambda: left is right,
            ast.IsNot: lambda: left is not right,
        }
        operation = operations.get(type(operator))
        if operation is None:
            raise _EvalError("python_exec: comparison operator is not available")
        return operation()

    def _invoke(self, function, args, kwargs):
        if isinstance(function, _UserFunction):
            return self._invoke_user(function, args, kwargs)
        if not isinstance(function, _Capability):
            raise _EvalError("python_exec: call target is not an allowed capability")
        if not function.allow_wrappers:
            if any(_is_wrapper(value) for value in args) or any(
                _is_wrapper(value) for value in kwargs.values()
            ):
                raise _EvalError("python_exec: capability argument is not allowed")
        try:
            result = function.function(*args, **kwargs)
        except _EvalError:
            raise
        except Exception as exc:
            raise _EvalError("%s failed: %s" % (function.label, exc)) from exc
        _validate_value(result)
        return result

    def _invoke_user(self, function, args, kwargs):
        arguments = function.arguments
        positional = list(arguments.posonlyargs) + list(arguments.args)
        defaults = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
        local = _Environment(function.closure)
        if len(args) > len(positional) and arguments.vararg is None:
            raise _EvalError("python_exec: too many function arguments")
        for index, argument in enumerate(positional):
            if index < len(args):
                local.set(argument.arg, args[index])
            elif argument.arg in kwargs:
                local.set(argument.arg, kwargs.pop(argument.arg))
            elif defaults[index] is not None:
                local.set(argument.arg, defaults[index])
            else:
                raise _EvalError("python_exec: missing function argument %r" % argument.arg)
        if arguments.vararg is not None:
            local.set(arguments.vararg.arg, tuple(args[len(positional) :]))
        elif len(args) > len(positional):
            raise _EvalError("python_exec: too many function arguments")
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            if argument.arg in kwargs:
                local.set(argument.arg, kwargs.pop(argument.arg))
            elif default is not None:
                local.set(argument.arg, default)
            else:
                raise _EvalError("python_exec: missing keyword-only argument %r" % argument.arg)
        if arguments.kwarg is not None:
            local.set(arguments.kwarg.arg, dict(kwargs))
            kwargs.clear()
        if kwargs:
            raise _EvalError("python_exec: unexpected function keyword")
        try:
            if function.expression:
                return self._expression(function.body, local)
            self._block(function.body, local)
        except _ReturnSignal as signal:
            return signal.value
        return None

    def _expression(self, node, environment):
        return self._eval(node, environment)

    def _store(self, target, value, environment):
        _validate_value(value)
        if isinstance(target, ast.Name):
            if target.id.startswith("__"):
                raise _EvalError("python_exec: dunder identifier is not available")
            environment.set(target.id, value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            items = list(self._iter_values(value))
            if len(items) != len(target.elts):
                raise _EvalError("python_exec: unpacking assignment failed")
            for item_target, item in zip(target.elts, items):
                self._store(item_target, item, environment)
            return
        if isinstance(target, ast.Subscript):
            self._setitem(
                self._eval(target.value, environment),
                self._eval(target.slice, environment),
                value,
            )
            return
        raise _EvalError("python_exec: assignment target is not available")

    def _target_value(self, target, environment):
        if isinstance(target, ast.Name):
            return environment.get(target.id)
        if isinstance(target, ast.Subscript):
            return self._getitem(
                self._eval(target.value, environment),
                self._eval(target.slice, environment),
            )
        raise _EvalError("python_exec: augmented assignment target is not available")

    def _comprehension(self, generators, emit, environment):
        result = []

        def visit(index, current):
            if index == len(generators):
                result.append(emit(current))
                return
            generator = generators[index]
            values = self._iter_values(self._eval(generator.iter, current))
            for value in values:
                child = _Environment(current)
                self._store(generator.target, value, child)
                if all(self._truth(self._eval(condition, child)) for condition in generator.ifs):
                    visit(index + 1, child)

        visit(0, _Environment(environment))
        return result

    def _eval(self, node, environment):
        self._tick()
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (bool, int, float, str, bytes)):
                return node.value
            raise _EvalError("python_exec: literal type is not available")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise _EvalError("python_exec: dunder identifier %r is not available" % node.id)
            return environment.get(node.id)
        if isinstance(node, ast.List):
            return [self._eval(item, environment) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(item, environment) for item in node.elts)
        if isinstance(node, ast.Set):
            return set(self._eval(item, environment) for item in node.elts)
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    result.update(self._plain(self._eval(value, environment)))
                else:
                    result[self._plain(self._eval(key, environment))] = self._eval(
                        value, environment
                    )
            _validate_value(result)
            return result
        if isinstance(node, ast.Starred):
            return self._eval(node.value, environment)
        if isinstance(node, ast.Attribute):
            return self._attribute(self._eval(node.value, environment), node.attr)
        if isinstance(node, ast.Subscript):
            return self._getitem(
                self._eval(node.value, environment),
                self._eval(node.slice, environment),
            )
        if isinstance(node, ast.Slice):
            return slice(
                self._eval(node.lower, environment) if node.lower else None,
                self._eval(node.upper, environment) if node.upper else None,
                self._eval(node.step, environment) if node.step else None,
            )
        if isinstance(node, ast.Call):
            function = self._eval(node.func, environment)
            args = []
            for argument in node.args:
                value = (
                    self._eval(argument.value, environment)
                    if isinstance(argument, ast.Starred)
                    else self._eval(argument, environment)
                )
                if isinstance(argument, ast.Starred):
                    args.extend(self._iter_values(value))
                else:
                    args.append(value)
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise _EvalError("python_exec: keyword unpacking is not available")
                kwargs[keyword.arg] = self._eval(keyword.value, environment)
            return self._invoke(function, tuple(args), kwargs)
        if isinstance(node, ast.BinOp):
            return self._binary(
                node.op, self._eval(node.left, environment), self._eval(node.right, environment)
            )
        if isinstance(node, ast.UnaryOp):
            return self._unary(node.op, self._eval(node.operand, environment))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                value = True
                for item in node.values:
                    value = self._eval(item, environment)
                    if not self._truth(value):
                        return value
                return value
            value = False
            for item in node.values:
                value = self._eval(item, environment)
                if self._truth(value):
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, environment)
            for operator, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, environment)
                if not self._compare(operator, left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            branch = node.body if self._truth(self._eval(node.test, environment)) else node.orelse
            return self._eval(branch, environment)
        if isinstance(node, ast.Lambda):
            return _UserFunction(
                self, "<lambda>", node.args, node.body, environment, expression=True
            )
        if isinstance(node, ast.JoinedStr):
            return "".join(self._eval(value, environment) for value in node.values)
        if isinstance(node, ast.FormattedValue):
            value = self._eval(node.value, environment)
            if node.conversion == 114:
                value = self._safe_repr(value)
            elif node.conversion == 115:
                value = self._safe_str(value)
            elif node.conversion == 97:
                value = self._safe_ascii(value)
            else:
                value = self._safe_str(value)
            if node.format_spec is not None:
                spec = self._eval(node.format_spec, environment)
                value = format(self._plain(value), self._plain(spec))
            return value
        if isinstance(node, ast.ListComp):
            return self._comprehension(
                node.generators, lambda current: self._eval(node.elt, current), environment
            )
        if isinstance(node, ast.SetComp):
            return set(
                self._comprehension(
                    node.generators, lambda current: self._eval(node.elt, current), environment
                )
            )
        if isinstance(node, ast.DictComp):
            pairs = self._comprehension(
                node.generators,
                lambda current: (
                    self._eval(node.key, current),
                    self._eval(node.value, current),
                ),
                environment,
            )
            return dict(pairs)
        if isinstance(node, ast.GeneratorExp):
            return self._comprehension(
                node.generators, lambda current: self._eval(node.elt, current), environment
            )
        if isinstance(node, ast.NamedExpr):
            value = self._eval(node.value, environment)
            self._store(node.target, value, environment)
            return value
        raise _EvalError("python_exec: syntax %s is not available" % type(node).__name__)

    def _block(self, statements, environment):
        for statement in statements:
            self._execute(statement, environment)

    def _execute(self, node, environment):
        self._tick()
        if isinstance(node, ast.Expr):
            self._eval(node.value, environment)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in self.modules or alias.name not in ALLOWED:
                    _reject(alias.name)
                environment.set(alias.asname or alias.name, self.modules[alias.name])
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in self.modules or node.module not in ALLOWED:
                _reject(node.module or "<relative>")
            module = self.modules[node.module]
            for alias in node.names:
                if alias.name == "*" or alias.name not in module.values:
                    raise _EvalError("python_exec: import name %r is not available" % alias.name)
                environment.set(alias.asname or alias.name, module.values[alias.name])
        elif isinstance(node, ast.Assign):
            value = self._eval(node.value, environment)
            for target in node.targets:
                self._store(target, value, environment)
        elif isinstance(node, ast.AnnAssign):
            self._store(node.target, self._eval(node.value, environment), environment)
        elif isinstance(node, ast.AugAssign):
            self._store(
                node.target,
                self._binary(
                    node.op,
                    self._target_value(node.target, environment),
                    self._eval(node.value, environment),
                ),
                environment,
            )
        elif isinstance(node, ast.If):
            self._block(
                node.body if self._truth(self._eval(node.test, environment)) else node.orelse,
                environment,
            )
        elif isinstance(node, ast.For):
            broken = False
            for value in self._iter_values(self._eval(node.iter, environment)):
                try:
                    self._store(node.target, value, environment)
                    self._block(node.body, environment)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    broken = True
                    break
            if not broken:
                self._block(node.orelse, environment)
        elif isinstance(node, ast.While):
            broken = False
            while self._truth(self._eval(node.test, environment)):
                try:
                    self._block(node.body, environment)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    broken = True
                    break
            if not broken:
                self._block(node.orelse, environment)
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise _EvalError("python_exec: decorators are not available")
            defaults = tuple(self._eval(value, environment) for value in node.args.defaults)
            kw_defaults = tuple(
                self._eval(value, environment) if value is not None else None
                for value in node.args.kw_defaults
            )
            arguments = ast.arguments(
                posonlyargs=node.args.posonlyargs,
                args=node.args.args,
                vararg=node.args.vararg,
                kwonlyargs=node.args.kwonlyargs,
                kw_defaults=list(kw_defaults),
                kwarg=node.args.kwarg,
                defaults=list(defaults),
            )
            environment.set(
                node.name,
                _UserFunction(self, node.name, arguments, node.body, environment),
            )
        elif isinstance(node, ast.Return):
            raise _ReturnSignal(self._eval(node.value, environment) if node.value else None)
        elif isinstance(node, ast.Raise):
            value = self._eval(node.exc, environment) if node.exc else _SafeException("Exception")
            if not isinstance(value, _SafeException):
                raise _EvalError("python_exec: raise requires an allowed exception")
            raise _UserRaised(value.name, value.message)
        elif isinstance(node, ast.Assert):
            if not self._truth(self._eval(node.test, environment)):
                raise _EvalError("python_exec: assertion failed")
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.Break):
            raise _BreakSignal()
        elif isinstance(node, ast.Continue):
            raise _ContinueSignal()
        else:
            raise _EvalError("python_exec: syntax %s is not available" % type(node).__name__)

    def run(self):
        self._block(self.tree.body, self.global_env)


try:
    tree = ast.parse(source, filename="<python_exec>", mode="exec")
    _Policy().visit(tree)
    _Evaluator(tree).run()
except _UserRaised as exc:
    sys.stderr.write("%s: %s\\n" % (exc.name, exc.message))
    raise SystemExit(1)
except Exception as exc:
    sys.stderr.write("%s\\n" % exc)
    raise SystemExit(1)
'''


class PythonExecError(RuntimeError):
    """Raised when the sandbox itself could not be started."""


@dataclass(frozen=True)
class PythonExecResult:
    """Outcome of one sandboxed run, before it is flattened to a string."""

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool

    @property
    def ok(self) -> bool:
        """Whether the program ran to completion without raising."""

        return not self.timed_out and self.returncode == 0

    def as_text(self) -> str:
        """Flatten to the single string the agent sees.

        Failures come back as text rather than as an exception so that a role's
        tool loop can read the traceback and try again.
        """

        if self.timed_out:
            return _join(self.stdout, self.stderr)
        if self.ok:
            return self.stdout
        return _join(self.stdout, self.stderr or f"python_exec: exited with {self.returncode}")


def run(
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> PythonExecResult:
    """Execute ``code`` in the sandbox and return the structured result."""

    if not isinstance(code, str):
        raise TypeError("python_exec: code must be a string")
    if timeout_s <= 0:
        raise ValueError("python_exec: timeout_s must be positive")

    bootstrap = (
        _BOOTSTRAP.replace("__ALLOWED__", repr(sorted(ALLOWED_IMPORTS)))
        .replace("__NETWORK_MESSAGE__", repr(NETWORK_BLOCKED_MESSAGE))
        .replace("__MAX_OUTPUT__", repr(max_output_chars))
        .strip()
    )
    # -I isolates the child from PYTHON* variables and the user site directory;
    # the temporary cwd keeps a stray open() away from the repository.
    argv = [sys.executable, "-I", "-B", "-c", bootstrap]

    with tempfile.TemporaryDirectory(prefix="occam-python-exec-") as workdir:
        try:
            # Fixed argv, no shell: the agent's code travels on stdin.
            completed = subprocess.run(
                argv,
                input=code,
                cwd=workdir,
                env=child_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as expired:
            return PythonExecResult(
                stdout=_clip(_decode(expired.stdout), max_output_chars),
                stderr=TIMEOUT_MESSAGE.format(timeout=timeout_s),
                returncode=None,
                timed_out=True,
            )
        except OSError as exc:  # pragma: no cover - interpreter is missing
            raise PythonExecError(f"python_exec: could not start the sandbox: {exc}") from exc

    return PythonExecResult(
        stdout=_clip(completed.stdout, max_output_chars),
        stderr=_clip(completed.stderr, max_output_chars),
        returncode=completed.returncode,
        timed_out=False,
    )


def python_exec(
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """Run a short Python program and return what it wrote to stdout."""

    return run(code, timeout_s=timeout_s, max_output_chars=max_output_chars).as_text()


def child_env() -> dict[str, str]:
    """Build the child's environment: interpreter essentials, no secrets.

    Every API key the engine loads from ``.env`` is absent by construction, so
    no introspection escape inside the child can surface one.  CPython may add
    ``LC_CTYPE`` while initialising a POSIX child; that implementation detail
    is not part of this explicit allow-list.
    """

    return {name: os.environ[name] for name in ENV_PASSTHROUGH if name in os.environ}


def _join(stdout: str, stderr: str) -> str:
    parts = [part for part in (stdout.rstrip("\n"), stderr.rstrip("\n")) if part]
    return "\n".join(parts)


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _clip(value: str, limit: int) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated at {limit} characters]"


__all__ = [
    "ALLOWED_IMPORTS",
    "CPYTHON_SYNTHESIZED_ENV",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_TIMEOUT_S",
    "ENV_PASSTHROUGH",
    "NETWORK_BLOCKED_MESSAGE",
    "child_env",
    "PythonExecError",
    "PythonExecResult",
    "python_exec",
    "run",
]
