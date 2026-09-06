"""Run agent-written arithmetic in a bounded child process.

``python_exec`` is intentionally a *capability-limited calculation surface*,
not a general Python sandbox.  The child parses source and interprets only an
allow-listed AST; it never compiles or executes the submitted Python.  It gets
an explicit environment allow-list and capability wrappers for Decimal, JSON,
date/time, math, and ordinary arithmetic.  Timeout, bounded output, a
temporary working directory, and strict value/step limits remain defense in
depth.  The user-visible modules, builtins, Python frames, and callable
wrappers are capability-limited; none can reach the filesystem, a process, or
the network.

The boundary is not perfect host isolation.  This module must not be described
as safe for arbitrary hostile Python: unsupported syntax is rejected, but a
bug in this evaluator or resource exhaustion still needs OS-level controls in
production (for example a container, separate user, seccomp/job controls, or
a dedicated sandbox service).  The temporary directory is a convenience, not
a filesystem-security claim; the security property here is that submitted
source is never executed as Python and receives no file/process/network
capability.

Collection construction, sorting, conversion, and JSON encoding use bounded
transient copies where the underlying CPython operation allocates one.  The
limits bound those copies' item/display budgets; they are not a zero-copy or
perfect-memory-isolation guarantee.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000
MAX_OUTPUT_CHARS_LIMIT = 1_000_000
MAX_INTEGER_BITS = 100_000

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

# Every callable attribute exposed by the child evaluator is classified here.
# ``_attribute`` is table-driven: adding a bound method therefore requires an
# explicit resource policy instead of silently falling back to raw getattr().
BOUND_METHOD_POLICIES: dict[str, dict[str, str]] = {
    "Decimal": {
        "adjusted": "pure",
        "copy_abs": "decimal_alloc",
        "copy_negate": "decimal_alloc",
        "is_finite": "pure",
        "is_infinite": "pure",
        "is_nan": "pure",
        "is_normal": "pure",
        "is_zero": "pure",
        "normalize": "decimal_alloc",
        "quantize": "decimal_alloc",
        "to_integral_value": "decimal_alloc",
    },
    "datetime": {
        "date": "date_alloc",
        "isoformat": "string_alloc",
        "isoweekday": "pure",
        "strftime": "string_alloc",
        "timestamp": "pure",
        "weekday": "pure",
    },
    "date": {
        "isoformat": "string_alloc",
        "isoweekday": "pure",
        "strftime": "string_alloc",
        "toordinal": "pure",
        "weekday": "pure",
    },
    "timedelta": {"total_seconds": "pure"},
    "dict": {
        "clear": "mutator",
        "copy": "collection_alloc",
        "get": "pure",
        "items": "collection_alloc",
        "keys": "collection_alloc",
        "pop": "mutator",
        "setdefault": "mutator",
        "values": "collection_alloc",
    },
    "list": {
        "append": "mutator",
        "clear": "mutator",
        "copy": "collection_alloc",
        "count": "pure",
        "extend": "mutator",
        "index": "pure",
        "insert": "mutator",
        "pop": "mutator",
        "remove": "mutator",
        "reverse": "mutator",
    },
    # Set mutators are deliberately not part of the exposed surface; set and
    # frozenset operators remain preflighted in _check_sequence_operator.
    "set": {},
    "tuple": {"count": "pure", "index": "pure"},
    "str": {
        "capitalize": "string_expand",
        "casefold": "string_expand",
        "center": "string_aggregate",
        "count": "pure",
        "encode": "bytes_alloc",
        "endswith": "pure",
        "find": "pure",
        "index": "pure",
        "isalnum": "pure",
        "isalpha": "pure",
        "isdigit": "pure",
        "islower": "pure",
        "isspace": "pure",
        "isupper": "pure",
        "join": "string_aggregate",
        "ljust": "string_aggregate",
        "lower": "string_expand",
        "lstrip": "string_alloc",
        "partition": "collection_alloc",
        "replace": "string_aggregate",
        "rfind": "pure",
        "rindex": "pure",
        "rsplit": "collection_alloc",
        "rstrip": "string_alloc",
        "split": "collection_alloc",
        "splitlines": "collection_alloc",
        "startswith": "pure",
        "strip": "string_alloc",
        "swapcase": "string_expand",
        "title": "string_expand",
        "upper": "string_expand",
        "zfill": "string_aggregate",
        "rjust": "string_aggregate",
    },
    "bytes": {
        "decode": "string_alloc",
        "endswith": "pure",
        "hex": "string_aggregate",
        "startswith": "pure",
    },
}

BOUND_METHOD_GUARDS: frozenset[str] = frozenset(
    {
        "pure",
        "mutator",
        "string_alloc",
        "string_expand",
        "string_aggregate",
        "bytes_alloc",
        "collection_alloc",
        "decimal_alloc",
        "date_alloc",
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
BOUND_METHOD_POLICIES = __BOUND_METHOD_POLICIES__
NETWORK_BLOCKED_MESSAGE = __NETWORK_MESSAGE__
MAX_OUTPUT_CHARS = __MAX_OUTPUT__
MAX_INTEGER_BITS = __MAX_INTEGER_BITS__
MAX_SOURCE_CHARS = 1_000_000

if hasattr(sys, "set_int_max_str_digits"):
    # The evaluator applies the actual bit-length guard after a bounded,
    # base-aware parse.  CPython's much smaller default would reject valid
    # binary/base-36 values before our contract could be applied.
    sys.set_int_max_str_digits(MAX_SOURCE_CHARS)

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
        if len(marker) >= remaining:
            clipped = value[:remaining]
        else:
            clipped = value[: remaining - len(marker)] + marker
        self._stream.write(clipped)
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
        "bytearray",
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


def _unsupported_syntax(name):
    raise _EvalError(
        "python_exec: syntax %r is unsupported by the restricted calculation subset" % name
    )


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
        raise _EvalError(
            "python_exec: name %r is unavailable in the restricted calculation subset" % name
        )

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


def _validate_value(value, depth=0, active=None):
    if depth > 64:
        raise _EvalError("python_exec: value nesting limit exceeded")
    if active is None:
        active = set()
    if _is_wrapper(value) or isinstance(value, _SafeException):
        return
    if value is None or isinstance(value, (bool, float)):
        return
    if isinstance(value, int):
        if value.bit_length() > _Evaluator.MAX_INTEGER_BITS:
            raise _EvalError("python_exec: integer magnitude limit exceeded")
        return
    if isinstance(value, (str, bytes)):
        if len(value) > _Evaluator.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string value limit exceeded")
        return
    if isinstance(value, slice):
        _validate_value(value.start, depth + 1, active)
        _validate_value(value.stop, depth + 1, active)
        _validate_value(value.step, depth + 1, active)
        return
    if isinstance(
        value, (_decimal.Decimal, _datetime.date, _datetime.datetime, _datetime.timedelta)
    ):
        return
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        marker = id(value)
        if marker in active:
            raise _EvalError("python_exec: value cycle detected")
        active.add(marker)
        try:
            if len(value) > _Evaluator.MAX_ITEMS:
                raise _EvalError("python_exec: collection item limit exceeded")
            if isinstance(value, dict):
                for key, item in value.items():
                    _validate_value(key, depth + 1, active)
                    _validate_value(item, depth + 1, active)
            else:
                for item in value:
                    _validate_value(item, depth + 1, active)
        finally:
            active.remove(marker)
        return
    if isinstance(value, range):
        if len(value) > _Evaluator.MAX_ITEMS:
            raise _EvalError("python_exec: collection item limit exceeded")
        return
    raise _EvalError("python_exec: value type %r is not available" % type(value).__name__)


class _Evaluator:
    MAX_STEPS = 10_000_000
    MAX_ITEMS = 100_000
    MAX_STRING_CHARS = 1_000_000
    MAX_INTEGER_BITS = MAX_INTEGER_BITS

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

    def _new_list(self):
        return []

    def _new_dict(self):
        return {}

    def _new_set(self):
        return set()

    def _collection_cost(self, value):
        return self._display_size(value)

    def _contains_mutable_container(self, value, active=None):
        """Detect mutable descendants without retaining state on the value."""

        if isinstance(value, (list, dict, set)):
            return True
        if not isinstance(value, (tuple, frozenset)):
            return False
        if active is None:
            active = set()
        marker = id(value)
        if marker in active:
            return True
        active.add(marker)
        try:
            return any(self._contains_mutable_container(item, active) for item in value)
        finally:
            active.remove(marker)

    def _collection_item_cost(self, value):
        return self._display_size(value, nested=True) + 2

    def _check_collection_add(self, value, item, label, cost=None):
        if len(value) >= self.MAX_ITEMS:
            raise _EvalError("python_exec: %s item limit exceeded" % label)
        if cost is None:
            cost = self._collection_cost(value)
        cost += self._collection_item_cost(item)
        if cost > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: %s display size limit exceeded" % label)
        return cost

    def _append_collection_item(self, value, item, label, cost=None, has_mutable=False):
        if cost is None or has_mutable:
            cost = self._collection_cost(value)
        cost = self._check_collection_add(value, item, label, cost)
        value.append(item)
        return cost, has_mutable or self._contains_mutable_container(item)

    def _add_set_item(self, value, item, label, cost=None, has_mutable=False):
        if item in value:
            return cost, has_mutable
        if cost is None or has_mutable:
            cost = self._collection_cost(value)
        cost = self._check_collection_add(value, item, label, cost)
        value.add(item)
        return cost, has_mutable or self._contains_mutable_container(item)

    def _dict_entry_cost(self, key, item):
        return (
            self._display_size(key, nested=True)
            + self._display_size(item, nested=True)
            + 4
        )

    def _set_dict_item(self, value, key, item, label, cost=None, has_mutable=False):
        try:
            present = key in value
        except TypeError as exc:
            raise _EvalError("python_exec: %s failed: %s" % (label, exc)) from exc
        old_cost = 0
        if present:
            old_cost = self._dict_entry_cost(key, value[key])
        elif len(value) >= self.MAX_ITEMS:
            raise _EvalError("python_exec: %s item limit exceeded" % label)
        if cost is None or has_mutable:
            cost = self._collection_cost(value)
        cost = cost - old_cost + self._dict_entry_cost(key, item)
        if cost > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: %s display size limit exceeded" % label)
        value[key] = item
        return (
            cost,
            has_mutable
            or self._contains_mutable_container(key)
            or self._contains_mutable_container(item),
        )

    def _validate_mutated(self, value):
        _validate_value(value)
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            if len(value) > self.MAX_ITEMS:
                raise _EvalError("python_exec: collection item limit exceeded")
            if self._display_size(value) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: collection display size limit exceeded")

    def _validate_collection_result(self, value, label):
        self._validate_mutated(value)
        return value

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
                    "ceil": self._capability("math.ceil", self._ceil),
                    "fabs": self._capability("math.fabs", _math.fabs),
                    "floor": self._capability("math.floor", self._floor),
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
        _validate_value(value)
        if isinstance(value, int) and not isinstance(value, bool):
            if value.bit_length() > self.MAX_STRING_CHARS * 4:
                raise _EvalError("python_exec: string constructor size limit exceeded")
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            return self._display(value)
        if isinstance(value, bytes) and self._display_size(value) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string conversion size limit exceeded")
        result = str(value)
        if len(result) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string conversion size limit exceeded")
        return result

    def _safe_repr(self, value):
        if _is_wrapper(value):
            return self._display(value)
        _validate_value(value)
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            return self._display(value)
        if isinstance(value, str):
            if self._ascii_string_size(value) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: string conversion size limit exceeded")
            result = repr(value)
            if len(result) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: string conversion size limit exceeded")
            return result
        if isinstance(value, bytes) and self._display_size(value) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string conversion size limit exceeded")
        result = repr(value)
        if len(result) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string conversion size limit exceeded")
        return result

    def _ascii_string_size(self, value):
        total = 2
        for character in value:
            code = ord(character)
            if code < 128:
                if character == chr(92) or character == "'" or character == '"':
                    total += 2
                elif code < 32 or code == 127:
                    total += 4
                else:
                    total += 1
            elif code <= 255:
                total += 4
            elif code <= 0xFFFF:
                total += 6
            else:
                total += 10
            if total > self.MAX_STRING_CHARS:
                return total
        return total

    def _safe_ascii(self, value):
        value = self._plain(value)
        if isinstance(value, str):
            if self._ascii_string_size(value) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: string conversion size limit exceeded")
            return ascii(value)
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            # _display_size recursively preflights the representation.  Its
            # nested string estimate is deliberately conservative enough for
            # ascii/repr escaping, so no aggregate is built before the check.
            if self._display_size(value) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: string conversion size limit exceeded")
            result = self._ascii_render(value)
            if len(result) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: string conversion size limit exceeded")
            return result
        if isinstance(value, bytes) and self._display_size(value) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string conversion size limit exceeded")
        result = ascii(value)
        if len(result) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string conversion size limit exceeded")
        return result

    def _safe_bytes(self, value=b"", encoding=None, errors="strict"):
        value = self._plain(value)
        if isinstance(value, int):
            self._check_constructor_size(value, "bytes", self.MAX_STRING_CHARS)
        if encoding is None:
            return self._check_bytes_result(bytes(value), "bytes constructor")
        encoding = self._plain(encoding)
        errors = self._plain(errors)
        if not isinstance(encoding, str):
            return self._check_bytes_result(bytes(value, encoding, errors), "bytes constructor")
        normalized = encoding.lower().replace("_", "-")
        factor = {
            "ascii": 1,
            "latin-1": 1,
            "iso-8859-1": 1,
            "cp1252": 1,
            "utf-8": 4,
            "utf8": 4,
            "utf-8-sig": 4,
            "utf-16": 4,
            "utf-16-le": 4,
            "utf-16-be": 4,
            "utf-32": 4,
            "utf-32-le": 4,
            "utf-32-be": 4,
            "utf-7": 8,
            "unicode-escape": 6,
            "raw-unicode-escape": 10,
        }.get(normalized)
        if factor is None:
            raise _EvalError("python_exec: encoding %r is not available" % encoding)
        if errors != "strict":
            factor = max(factor, 10)
        if isinstance(value, str) and len(value) * factor > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: encoded bytes constructor size limit exceeded")
        return self._check_bytes_result(bytes(value, encoding, errors), "bytes constructor")

    def _check_decimal_integer_magnitude(self, value, label):
        if not isinstance(value, _decimal.Decimal) or not value.is_finite():
            return
        sign, digits, exponent = value.as_tuple()
        if not digits or value.is_zero():
            return
        adjusted = value.adjusted()
        if adjusted < 0:
            return
        if exponent >= 0:
            decimal_digits = len(digits) + exponent
        else:
            decimal_digits = adjusted + 1
        # A decimal digit count is a safe pre-conversion resource estimate;
        # conversion is only attempted while the result can be near the
        # bounded integer contract.  The final bit-length check handles the
        # small leading-digit slack at the boundary.
        max_decimal_digits = int(self.MAX_INTEGER_BITS * _math.log10(2)) + 1
        if decimal_digits > max_decimal_digits:
            raise _EvalError("python_exec: %s integer magnitude limit exceeded" % label)

    def _safe_int(self, value=0, base=10):
        value = self._plain(value)
        base = self._plain(base)
        if isinstance(value, bytes):
            value = value.decode("ascii")
        if isinstance(value, str):
            digits = value.strip().lstrip("+-").replace("_", "")
            effective_base = base
            lowered = digits.lower()
            if base == 0:
                if lowered.startswith("0b"):
                    effective_base, digits = 2, digits[2:]
                elif lowered.startswith("0o"):
                    effective_base, digits = 8, digits[2:]
                elif lowered.startswith("0x"):
                    effective_base, digits = 16, digits[2:]
                else:
                    effective_base = 10
            if isinstance(effective_base, int) and 2 <= effective_base <= 36:
                significant = digits.lstrip("0") or "0"
                projected_bits = _math.ceil(
                    len(significant) * _math.log2(effective_base)
                )
                if projected_bits > self.MAX_INTEGER_BITS:
                    raise _EvalError("python_exec: integer magnitude limit exceeded")
            elif not isinstance(effective_base, int) or effective_base not in {
                0,
                *range(2, 37),
            }:
                return int(value, base)
            result = int(value, base)
            _validate_value(result)
            return result
        self._check_decimal_integer_magnitude(value, "int")
        if base != 10:
            return int(value, base)
        result = int(value)
        _validate_value(result)
        return result

    def _check_pow(self, left, right):
        if not isinstance(right, int):
            raise _EvalError("python_exec: power exponent must be a bounded integer")
        if abs(right) > self.MAX_INTEGER_BITS:
            raise _EvalError("python_exec: integer magnitude limit exceeded")
        if right < 0 or not isinstance(left, int) or isinstance(left, bool):
            return
        magnitude = abs(left)
        if magnitude <= 1:
            return
        if magnitude == 2:
            projected_bits = right + 1
        else:
            projected_bits = left.bit_length() * right
        if projected_bits > self.MAX_INTEGER_BITS:
            raise _EvalError("python_exec: integer magnitude limit exceeded")

    def _check_integer_binary(self, operator, left, right):
        if isinstance(operator, ast.Pow):
            self._check_pow(left, right)
            return
        integer_operands = isinstance(left, int) and isinstance(right, int)
        if isinstance(operator, (ast.LShift, ast.RShift)):
            if isinstance(right, int):
                if abs(right) > self.MAX_INTEGER_BITS:
                    raise _EvalError("python_exec: integer magnitude limit exceeded")
            if not integer_operands:
                return
            if right < 0:
                raise _EvalError("python_exec: negative shift count")
            if isinstance(operator, ast.LShift):
                projected_bits = 0 if left == 0 else abs(left).bit_length() + right
                if projected_bits > self.MAX_INTEGER_BITS:
                    raise _EvalError("python_exec: integer magnitude limit exceeded")
            return
        if not integer_operands:
            return
        if not isinstance(operator, ast.Mult):
            return
        if left == 0 or right == 0 or abs(left) == 1 or abs(right) == 1:
            return
        bits_sum = left.bit_length() + right.bit_length()
        if bits_sum > self.MAX_INTEGER_BITS + 1:
            raise _EvalError("python_exec: integer magnitude limit exceeded")
        if (
            bits_sum == self.MAX_INTEGER_BITS + 1
            and (left * right).bit_length() > self.MAX_INTEGER_BITS
        ):
            raise _EvalError("python_exec: integer magnitude limit exceeded")

    def _safe_pow(self, left, right, modulo=None):
        left = self._plain(left)
        right = self._plain(right)
        self._check_pow(left, right)
        if modulo is None:
            return pow(left, right)
        return pow(left, right, self._plain(modulo))

    def _check_constructor_size(self, size, label, limit):
        if isinstance(size, int) and size > limit:
            raise _EvalError(
                "python_exec: %s constructor size exceeds its bounded limit" % label
            )

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
            return value
        if isinstance(value, (list, tuple, set, frozenset, str, bytes, range)):
            return value
        raise _EvalError("python_exec: value is not iterable")

    def _materialize(self, values, label):
        result = self._new_list()
        cost = 2
        has_mutable = False
        for value in values:
            cost, has_mutable = self._append_collection_item(
                result, value, label, cost, has_mutable
            )
        self._validate_mutated(result)
        return result

    def _make_list(self, value=()):
        return self._materialize(self._iter_values(value), "list constructor")

    def _make_tuple(self, value=()):
        result = tuple(self._materialize(self._iter_values(value), "tuple constructor"))
        self._validate_mutated(result)
        return result

    def _make_set(self, value=()):
        result = self._new_set()
        cost = 2
        has_mutable = False
        for item in self._iter_values(value):
            cost, has_mutable = self._add_set_item(
                result, item, "set constructor", cost, has_mutable
            )
        self._validate_mutated(result)
        return result

    def _make_frozenset(self, value=()):
        result = self._new_set()
        cost = 2
        has_mutable = False
        for item in self._iter_values(value):
            cost, has_mutable = self._add_set_item(
                result, item, "frozenset constructor", cost, has_mutable
            )
        result = frozenset(result)
        self._validate_mutated(result)
        return result

    def _make_dict(self, value=(), **kwargs):
        result = self._new_dict()
        cost = 2
        has_mutable = False
        if value == ():
            pass
        elif isinstance(value, dict):
            for key, item in value.items():
                cost, has_mutable = self._set_dict_item(
                    result, key, item, "dict constructor", cost, has_mutable
                )
        else:
            for pair in self._iter_values(value):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise _EvalError("python_exec: dict constructor needs key/value pairs")
                cost, has_mutable = self._set_dict_item(
                    result, pair[0], pair[1], "dict constructor", cost, has_mutable
                )
        for key, item in kwargs.items():
            cost, has_mutable = self._set_dict_item(
                result, key, item, "dict constructor", cost, has_mutable
            )
        self._validate_mutated(result)
        return result

    def _all(self, value):
        return all(self._iter_values(value))

    def _any(self, value):
        return any(self._iter_values(value))

    def _enumerate(self, value, start=0):
        return self._materialize(
            enumerate(self._iter_values(value), self._plain(start)), "enumerate"
        )

    def _filter(self, function, value):
        items = self._iter_values(value)
        result = self._new_list()
        cost = 2
        has_mutable = False
        for item in items:
            if self._truth(self._invoke(function, (item,), {})):
                cost, has_mutable = self._append_collection_item(
                    result, item, "filter result", cost, has_mutable
                )
        self._validate_mutated(result)
        return result

    def _map(self, function, *values):
        items = [self._iter_values(value) for value in values]
        return self._materialize(
            (self._invoke(function, tuple(row), {}) for row in zip(*items)), "map result"
        )

    def _zip(self, *values):
        return self._materialize(
            zip(*(self._iter_values(value) for value in values)), "zip result"
        )

    def _reversed(self, value):
        return self._materialize(reversed(self._iter_values(value)), "reversed result")

    def _sorted(self, value, reverse=False):
        items = self._materialize(self._iter_values(value), "sorted result")
        reverse = bool(self._plain(reverse))
        # CPython's sort allocates a second list.  Both copies are bounded by
        # the preflighted item/display contract before the sort starts.
        self._validate_mutated(items)
        result = sorted(items, reverse=reverse)
        return self._validate_collection_result(result, "sorted result")

    def _minimum(self, *values, **kwargs):
        return self._extreme(min, values, kwargs)

    def _maximum(self, *values, **kwargs):
        return self._extreme(max, values, kwargs)

    def _extreme(self, function, values, kwargs):
        default_provided = "default" in kwargs
        default = kwargs.pop("default", None)
        if kwargs:
            raise _EvalError("python_exec: unsupported min/max keyword")
        if len(values) == 1:
            values = self._iter_values(values[0])
            if default_provided:
                return function(values, default=self._plain(default))
            return function(values)
        if not values:
            if default_provided:
                return default
            raise _EvalError("python_exec: min/max needs a value")
        return function(*(self._plain(value) for value in values))

    def _round(self, value, ndigits=None):
        value = self._plain(value)
        if ndigits is None:
            self._check_decimal_integer_magnitude(value, "round")
            return round(value)
        return round(value, self._plain(ndigits))

    def _ceil(self, value):
        value = self._plain(value)
        self._check_decimal_integer_magnitude(value, "math.ceil")
        return _math.ceil(value)

    def _floor(self, value):
        value = self._plain(value)
        self._check_decimal_integer_magnitude(value, "math.floor")
        return _math.floor(value)

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
        for index, value in enumerate(values):
            if index:
                sys.stdout.write(sep)
            sys.stdout.write(self._display(value))
        sys.stdout.write(end)
        sys.stdout.flush()

    def _display_size(self, value, nested=False, depth=0, active=None):
        if depth > 64:
            raise _EvalError("python_exec: display nesting limit exceeded")
        if active is None:
            active = set()
        container = isinstance(value, (list, tuple, set, frozenset, dict))
        marker = id(value)
        if container:
            if marker in active:
                raise _EvalError("python_exec: display cycle detected")
            active.add(marker)
        try:
            if isinstance(value, _Capability):
                return len(value.label) + 13
            if isinstance(value, _SafeModule):
                return len(value.name) + 9
            if isinstance(value, _UserFunction):
                return len(value.name) + 11
            if isinstance(value, _SafeException):
                return len(value.name) + len(value.message) + 2
            if isinstance(value, str):
                return len(value) if not nested else self._ascii_string_size(value) + 2
            if isinstance(value, bytes):
                return len(value) * 4 + 4
            if isinstance(value, int) and not isinstance(value, bool):
                bits = value.bit_length()
                if bits <= 4096:
                    return len(str(value))
                return bits * 4 + 1
            if isinstance(value, (list, tuple, set, frozenset)):
                total = 2
                for item in value:
                    total += (
                        self._display_size(
                            item, nested=True, depth=depth + 1, active=active
                        )
                        + 2
                    )
                    if total > self.MAX_STRING_CHARS:
                        return total
                return total
            if isinstance(value, dict):
                total = 2
                for key, item in value.items():
                    total += (
                        self._display_size(
                            key, nested=True, depth=depth + 1, active=active
                        )
                        + self._display_size(
                            item, nested=True, depth=depth + 1, active=active
                        )
                        + 4
                    )
                    if total > self.MAX_STRING_CHARS:
                        return total
                return total
            return len(str(value)) + 16
        finally:
            if container:
                active.remove(marker)

    def _bounded_join(self, parts, separator, label, overhead=0):
        chunks = []
        total = overhead
        for part in parts:
            if len(chunks) >= self.MAX_ITEMS:
                raise _EvalError("python_exec: %s item limit exceeded" % label)
            if not isinstance(part, str):
                raise _EvalError("python_exec: %s requires text parts" % label)
            if chunks:
                total += len(separator)
            total += len(part)
            if total > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: %s size limit exceeded" % label)
            chunks.append(part)
        return separator.join(chunks)

    def _display(self, value):
        if self._display_size(value) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: string conversion size limit exceeded")
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
            return left + self._bounded_join(
                (self._render_value(item) for item in value),
                ", ",
                "display",
                overhead=len(left) + len(right),
            ) + right
        if isinstance(value, dict):
            return "{" + self._bounded_join(
                (
                    self._bounded_join(
                        (self._render_value(key), self._render_value(item)),
                        ": ",
                        "display entry",
                    )
                    for key, item in value.items()
                ),
                ", ",
                "display",
                overhead=2,
            ) + "}"
        return str(value)

    def _ascii_render(self, value):
        if isinstance(value, (list, tuple, set, frozenset)):
            left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
            if isinstance(value, (set, frozenset)):
                left, right = ("{", "}")
            return left + self._bounded_join(
                (self._ascii_render(item) for item in value),
                ", ",
                "ascii display",
                overhead=len(left) + len(right),
            ) + right
        if isinstance(value, dict):
            return "{" + self._bounded_join(
                (
                    self._bounded_join(
                        (self._ascii_render(key), self._ascii_render(item)),
                        ": ",
                        "ascii display entry",
                    )
                    for key, item in value.items()
                ),
                ", ",
                "ascii display",
                overhead=2,
            ) + "}"
        return ascii(value)

    def _render_value(self, value):
        if _is_wrapper(value):
            return self._display(value)
        if isinstance(value, _SafeException):
            return self._display(value)
        if isinstance(value, dict):
            return self._display(value)
        if isinstance(value, (list, tuple, set, frozenset)):
            return self._display(value)
        return repr(value)

    def _json_plain(self, value, depth=0, active=None):
        if depth > 64:
            raise _EvalError("python_exec: JSON nesting limit exceeded")
        if active is None:
            active = set()
        if _is_wrapper(value) or isinstance(value, _SafeException):
            raise _EvalError("python_exec: JSON cannot contain capabilities")
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, (_decimal.Decimal, _datetime.date, _datetime.datetime)):
            return value
        if isinstance(value, (list, tuple, dict)):
            marker = id(value)
            if marker in active:
                raise _EvalError("python_exec: JSON cycle detected")
            active.add(marker)
            try:
                if isinstance(value, dict):
                    for key, item in value.items():
                        self._json_plain(key, depth + 1, active)
                        self._json_plain(item, depth + 1, active)
                else:
                    for item in value:
                        self._json_plain(item, depth + 1, active)
            finally:
                active.remove(marker)
            return value
        raise _EvalError("python_exec: JSON value is not available")

    def _json_string_size(self, value, ensure_ascii):
        total = 2
        for character in value:
            code = ord(character)
            if character == chr(92) or character == '"':
                total += 2
            elif code < 32:
                total += 2 if character in "\\b\\t\\n\\f\\r" else 6
            elif ensure_ascii and code > 127:
                total += 6 if code <= 0xFFFF else 12
            else:
                total += 1
            if total > self.MAX_STRING_CHARS:
                return total
        return total

    def _json_size(self, value, depth=0, ensure_ascii=True, active=None):
        if depth > 64:
            raise _EvalError("python_exec: JSON nesting limit exceeded")
        if active is None:
            active = set()
        if _is_wrapper(value) or isinstance(value, _SafeException):
            raise _EvalError("python_exec: JSON cannot contain capabilities")
        if value is None:
            return 4
        if isinstance(value, bool):
            return 5
        if isinstance(value, int):
            if value.bit_length() > self.MAX_INTEGER_BITS:
                raise _EvalError("python_exec: integer magnitude limit exceeded")
            return value.bit_length() * 4 + 2
        if isinstance(value, float):
            return 32
        if isinstance(value, str):
            return self._json_string_size(value, ensure_ascii)
        if isinstance(value, bytes):
            return len(value) * 4 + 2
        if isinstance(value, (_decimal.Decimal, _datetime.date, _datetime.datetime)):
            return len(str(value)) * 12 + 2
        if isinstance(value, (list, tuple, dict)):
            marker = id(value)
            if marker in active:
                raise _EvalError("python_exec: JSON cycle detected")
            active.add(marker)
            try:
                if len(value) > self.MAX_ITEMS:
                    raise _EvalError("python_exec: collection item limit exceeded")
                if isinstance(value, dict):
                    total = 2
                    for key, item in value.items():
                        total += self._json_size(key, depth + 1, ensure_ascii, active)
                        total += self._json_size(item, depth + 1, ensure_ascii, active) + 2
                        if total > self.MAX_STRING_CHARS:
                            return total
                else:
                    total = 2
                    for item in value:
                        total += self._json_size(item, depth + 1, ensure_ascii, active) + 1
                        if total > self.MAX_STRING_CHARS:
                            return total
                return total
            finally:
                active.remove(marker)
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
            ensure_ascii = bool(kwargs.get("ensure_ascii", True))

            def bounded_default(item):
                result = self._invoke(default, (item,), {})
                if self._json_size(result, ensure_ascii=ensure_ascii) > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: JSON output size limit exceeded")
                return result

            kwargs["default"] = bounded_default
        elif default is not None:
            raise _EvalError("python_exec: json default must be an allowed function")
        ensure_ascii = bool(kwargs.get("ensure_ascii", True))
        indent = kwargs.get("indent")
        if isinstance(indent, str):
            indent_size = len(indent)
        elif isinstance(indent, int) and not isinstance(indent, bool):
            indent_size = abs(indent)
        else:
            indent_size = 0
        if indent_size * 64 > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: JSON output size limit exceeded")
        if self._json_size(value, ensure_ascii=ensure_ascii) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: JSON output size limit exceeded")
        encoder = _json.JSONEncoder(**kwargs)
        pieces = []
        length = 0
        for piece in encoder.iterencode(self._json_plain(value)):
            if length + len(piece) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: JSON output size limit exceeded")
            if len(pieces) >= self.MAX_ITEMS:
                raise _EvalError("python_exec: JSON output item limit exceeded")
            pieces.append(piece)
            length += len(piece)
        return "".join(pieces)

    def _json_loads(self, value):
        value = self._plain(value)
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        elif isinstance(value, str):
            text = value
        else:
            raise _EvalError("python_exec: json.loads needs text or bytes")
        depth = 0
        in_string = False
        escaped = False
        estimated_items = 0
        for character in text:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "[{":
                depth += 1
                if depth > 64:
                    raise _EvalError("python_exec: JSON nesting limit exceeded")
                estimated_items += 1
            elif character == ",":
                estimated_items += 1
            if estimated_items > self.MAX_ITEMS:
                raise _EvalError("python_exec: collection item limit exceeded")

        def bounded_pairs(pairs):
            result = self._new_dict()
            cost = 2
            has_mutable = False
            for key, item in pairs:
                cost, has_mutable = self._set_dict_item(
                    result, key, item, "json.loads", cost, has_mutable
                )
            self._validate_mutated(result)
            return result

        result = _json.loads(
            text,
            parse_int=self._safe_int,
            object_pairs_hook=bounded_pairs,
        )
        return self._validate_collection_result(result, "json.loads")

    def _check_text_result(self, result, label="string result"):
        if not isinstance(result, str) or len(result) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: %s size limit exceeded" % label)
        return result

    def _check_bytes_result(self, result, label="bytes result"):
        if not isinstance(result, bytes) or len(result) > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: %s size limit exceeded" % label)
        return result

    def _width_from_call(self, args, kwargs, default=None):
        if args:
            return args[0]
        return kwargs.get("width", default)

    def _check_width(self, width, label):
        if isinstance(width, int) and width > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: %s size limit exceeded" % label)

    def _preflight_string_method(self, value, name, args, kwargs):
        if name in {"center", "ljust", "rjust", "zfill"}:
            self._check_width(self._width_from_call(args, kwargs, len(value)), "string result")
            return
        if name == "join":
            if len(args) != 1 or kwargs:
                return
            pieces = []
            total = 0
            for item in self._iter_values(args[0]):
                if len(pieces) >= self.MAX_ITEMS:
                    raise _EvalError("python_exec: string join item limit exceeded")
                if not isinstance(item, str):
                    raise _EvalError("python_exec: string join requires strings")
                total += len(item)
                if pieces:
                    total += len(value)
                if total > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: string join size limit exceeded")
                pieces.append(item)
            return pieces
        if name == "replace":
            old = args[0] if args else kwargs.get("old")
            new = args[1] if len(args) > 1 else kwargs.get("new")
            count = args[2] if len(args) > 2 else kwargs.get("count", -1)
            if isinstance(old, str) and isinstance(new, str) and isinstance(count, int):
                occurrences = len(value) + 1 if old == "" else value.count(old)
                replacements = occurrences if count < 0 else min(count, occurrences)
                size = len(value) + replacements * (len(new) - len(old))
                if size > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: string replace size limit exceeded")
            return
        if name in {"capitalize", "casefold", "lower", "swapcase", "title", "upper"}:
            if len(value) * 4 > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: string result size limit exceeded")

    def _preflight_strftime(self, format_string):
        if not isinstance(format_string, str):
            return
        directive_sizes = {
            "%": 1,
            "a": 16,
            "A": 32,
            "b": 16,
            "B": 32,
            "c": 128,
            "C": 2,
            "d": 2,
            "D": 8,
            "e": 2,
            "F": 10,
            "g": 2,
            "G": 4,
            "H": 2,
            "I": 2,
            "j": 3,
            "k": 2,
            "l": 2,
            "m": 2,
            "M": 2,
            "n": 1,
            "p": 16,
            "r": 16,
            "R": 5,
            "S": 2,
            "t": 1,
            "T": 8,
            "u": 1,
            "U": 2,
            "V": 2,
            "w": 1,
            "W": 2,
            "x": 32,
            "X": 32,
            "y": 2,
            "Y": 4,
            "z": 16,
            "Z": 128,
        }
        total = 0
        index = 0
        while index < len(format_string):
            if format_string[index] != "%":
                total += 1
                index += 1
            else:
                index += 1
                if index >= len(format_string):
                    total += 1
                else:
                    if format_string[index] == ":":
                        index += 1
                        if index < len(format_string) and format_string[index] == ":":
                            index += 1
                    while index < len(format_string) and format_string[index] in "-_0^#":
                        index += 1
                    if index < len(format_string) and format_string[index] in "EO":
                        index += 1
                    if index < len(format_string):
                        directive = format_string[index]
                        total += directive_sizes.get(directive, 256)
                        index += 1
                    else:
                        total += 1
            if total > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: formatted string size limit exceeded")

    def _safe_string_method(self, value, name, args, kwargs):
        if name == "strftime":
            format_string = args[0] if args else kwargs.get("format", "")
            self._preflight_strftime(format_string)
        preflight = self._preflight_string_method(value, name, args, kwargs)
        if name == "join" and preflight is not None:
            return self._check_text_result(value.join(preflight))
        result = getattr(value, name)(*args, **kwargs)
        return self._check_text_result(result)

    def _safe_encode(self, value, args, kwargs):
        encoding = args[0] if args else kwargs.get("encoding", "utf-8")
        if encoding is None:
            encoding = "utf-8"
        if not isinstance(encoding, str):
            return value.encode(*args, **kwargs)
        normalized = encoding.lower().replace("_", "-")
        factors = {
            "ascii": 1,
            "latin-1": 1,
            "iso-8859-1": 1,
            "cp1252": 1,
            "utf-8": 4,
            "utf8": 4,
            "utf-16": 4,
            "utf-16-le": 4,
            "utf-16-be": 4,
            "utf-32": 4,
            "utf-32-le": 4,
            "utf-32-be": 4,
            "utf-7": 8,
            "unicode-escape": 6,
            "raw-unicode-escape": 10,
        }
        factor = factors.get(normalized)
        if factor is None:
            raise _EvalError("python_exec: encoding %r is not available" % encoding)
        errors = args[1] if len(args) > 1 else kwargs.get("errors", "strict")
        if errors != "strict":
            factor = max(factor, 10)
        if len(value) * factor > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: encoded bytes result size limit exceeded")
        return self._check_bytes_result(value.encode(*args, **kwargs), "encoded bytes result")

    def _safe_bytes_decode(self, value, args, kwargs):
        encoding = args[0] if args else kwargs.get("encoding", "utf-8")
        if encoding is None:
            encoding = "utf-8"
        if not isinstance(encoding, str):
            raise _EvalError("python_exec: decoding encoding must be text")
        normalized = encoding.lower().replace("_", "-")
        if normalized not in {
            "ascii",
            "latin-1",
            "iso-8859-1",
            "cp1252",
            "utf-8",
            "utf8",
            "utf-8-sig",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
            "utf-32",
            "utf-32-le",
            "utf-32-be",
            "utf-7",
            "unicode-escape",
            "raw-unicode-escape",
        }:
            raise _EvalError("python_exec: encoding %r is not available" % encoding)
        if len(value) * 4 > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: decoded string result size limit exceeded")
        return self._check_text_result(value.decode(*args, **kwargs), "decoded string result")

    def _safe_bytes_hex(self, value, args, kwargs):
        separator = args[0] if args else kwargs.get("sep", "")
        bytes_per_sep = args[1] if len(args) > 1 else kwargs.get("bytes_per_sep", 1)
        if isinstance(separator, str) and isinstance(bytes_per_sep, int):
            groups = (
                (len(value) - 1) // abs(bytes_per_sep)
                if value and bytes_per_sep
                else 0
            )
            size = len(value) * 2 + groups * len(separator)
            if size > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: bytes.hex result size limit exceeded")
        result = value.hex(*args, **kwargs)
        return self._check_text_result(result, "bytes.hex result")

    def _preflight_split(self, value, args, kwargs):
        separator = args[0] if args else kwargs.get("sep", None)
        maxsplit = args[1] if len(args) > 1 else kwargs.get("maxsplit", -1)
        if not isinstance(maxsplit, int):
            return
        if separator is None:
            words = 0
            in_word = False
            for character in value:
                if character.isspace():
                    in_word = False
                elif not in_word:
                    words += 1
                    in_word = True
            count = words if maxsplit < 0 else min(words, maxsplit + 1)
        elif isinstance(separator, str) and separator:
            occurrences = value.count(separator)
            splits = occurrences if maxsplit < 0 else min(occurrences, maxsplit)
            count = splits + 1
        else:
            return
        if count > self.MAX_ITEMS:
            raise _EvalError("python_exec: split result item limit exceeded")
        if len(value) * 12 + count * 2 + 2 > self.MAX_STRING_CHARS:
            raise _EvalError("python_exec: split result display size limit exceeded")

    def _safe_collection_method(self, value, name, args, kwargs):
        if isinstance(value, str):
            if name in {"split", "rsplit"}:
                self._preflight_split(value, args, kwargs)
            elif name == "splitlines":
                count = value.count("\\n") + value.count("\\r") + 1
                if count > self.MAX_ITEMS:
                    raise _EvalError("python_exec: split result item limit exceeded")
                if len(value) * 12 + count * 2 + 2 > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: split result display size limit exceeded")
            elif name == "partition":
                separator = args[0] if args else kwargs.get("sep", "")
                if isinstance(separator, str) and len(value) + len(separator) > 0:
                    if (len(value) + len(separator)) * 12 + 8 > self.MAX_STRING_CHARS:
                        raise _EvalError(
                            "python_exec: partition result display size limit exceeded"
                        )
        elif isinstance(value, dict):
            if len(value) > self.MAX_ITEMS:
                raise _EvalError("python_exec: collection item limit exceeded")
            if self._display_size(value) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: collection display size limit exceeded")
            if name == "copy":
                result = self._new_dict()
                cost = 2
                has_mutable = False
                for key, item in value.items():
                    cost, has_mutable = self._set_dict_item(
                        result, key, item, "dict.copy", cost, has_mutable
                    )
                self._validate_mutated(result)
                return result
            if name in {"items", "keys", "values"}:
                result = self._new_list()
                cost = 2
                has_mutable = False
                for key, item in value.items():
                    output = (key, item) if name == "items" else key if name == "keys" else item
                    cost, has_mutable = self._append_collection_item(
                        result, output, "dict.%s" % name, cost, has_mutable
                    )
                self._validate_mutated(result)
                return result
        elif isinstance(value, list):
            if self._display_size(value) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: collection display size limit exceeded")
            if name == "copy":
                result = self._new_list()
                cost = 2
                has_mutable = False
                for item in value:
                    cost, has_mutable = self._append_collection_item(
                        result, item, "list.copy", cost, has_mutable
                    )
                self._validate_mutated(result)
                return result
        result = getattr(value, name)(*args, **kwargs)
        return self._validate_collection_result(result, "%s.%s" % (type(value).__name__, name))

    def _concat_strings(self, parts, label):
        chunks = []
        total = 0
        for part in parts:
            if len(chunks) >= self.MAX_ITEMS:
                raise _EvalError("python_exec: %s item limit exceeded" % label)
            if not isinstance(part, str):
                raise _EvalError("python_exec: %s requires string parts" % label)
            total += len(part)
            if total > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: %s size limit exceeded" % label)
            chunks.append(part)
        return "".join(chunks)

    def _safe_format(self, value, spec):
        if not isinstance(value, str) or not isinstance(spec, str):
            return format(value, spec)
        number = 0
        in_number = False
        for character in spec:
            if character.isdigit():
                number = number * 10 + int(character)
                in_number = True
                if number > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: formatted string size limit exceeded")
            elif in_number:
                number = 0
                in_number = False
        return self._check_text_result(format(value, spec), "formatted string")

    def _list_extend(self, value, source):
        source = self._iter_values(source)
        if source is value:
            if len(value) > self.MAX_ITEMS - len(value):
                raise _EvalError("python_exec: list.extend item limit exceeded")
            source = tuple(value)
        if hasattr(source, "__len__") and len(source) > self.MAX_ITEMS - len(value):
            raise _EvalError("python_exec: list.extend item limit exceeded")
        pending = []
        cost = self._collection_cost(value)
        for item in source:
            if len(value) + len(pending) >= self.MAX_ITEMS:
                raise _EvalError("python_exec: list.extend item limit exceeded")
            cost += self._collection_item_cost(item)
            if cost > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: list.extend display size limit exceeded")
            pending.append(item)
        value.extend(pending)
        self._validate_mutated(value)
        return None

    def _safe_mutator(self, value, name, args, kwargs):
        if isinstance(value, list):
            if name == "append" and len(args) == 1 and not kwargs:
                self._append_collection_item(value, args[0], "list.append")
                self._validate_mutated(value)
                return None
            if name == "extend" and len(args) == 1 and not kwargs:
                result = self._list_extend(value, args[0])
                self._validate_mutated(value)
                return result
            if name == "insert" and len(args) == 2 and not kwargs:
                self._check_collection_add(value, args[1], "list.insert")
                value.insert(args[0], args[1])
                self._validate_mutated(value)
                return None
            result = getattr(value, name)(*args, **kwargs)
            self._validate_mutated(value)
            return result
        if isinstance(value, dict):
            if name == "setdefault" and len(args) in {1, 2} and not kwargs:
                key = args[0]
                default = args[1] if len(args) == 2 else None
                if key not in value:
                    self._set_dict_item(value, key, default, "dict.setdefault")
                self._validate_mutated(value)
                return value[key]
            result = getattr(value, name)(*args, **kwargs)
            self._validate_mutated(value)
            return result
        result = getattr(value, name)(*args, **kwargs)
        self._validate_mutated(value)
        return result

    def _bound_method(self, value, name, policy):
        def call(*args, **kwargs):
            raw_args = tuple(self._plain(arg) for arg in args)
            raw_kwargs = {key: self._plain(item) for key, item in kwargs.items()}
            if policy == "mutator":
                return self._safe_mutator(value, name, raw_args, raw_kwargs)
            if isinstance(value, bytes) and name == "hex":
                return self._safe_bytes_hex(value, raw_args, raw_kwargs)
            if isinstance(value, bytes) and name == "decode":
                return self._safe_bytes_decode(value, raw_args, raw_kwargs)
            if isinstance(value, str) and name == "encode":
                return self._safe_encode(value, raw_args, raw_kwargs)
            if policy in {"string_alloc", "string_expand", "string_aggregate"}:
                return self._safe_string_method(value, name, raw_args, raw_kwargs)
            if policy == "collection_alloc":
                return self._safe_collection_method(value, name, raw_args, raw_kwargs)
            result = getattr(value, name)(*raw_args, **raw_kwargs)
            if policy == "decimal_alloc":
                _validate_value(result)
            elif policy == "date_alloc":
                _validate_value(result)
            return result

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
            method_type = "Decimal"
        elif isinstance(value, _datetime.datetime):
            method_type = "datetime"
            if name in {"day", "hour", "microsecond", "minute", "month", "second", "year"}:
                return getattr(value, name)
        elif isinstance(value, _datetime.date):
            method_type = "date"
            if name in {"day", "month", "year"}:
                return getattr(value, name)
        elif isinstance(value, _datetime.timedelta):
            method_type = "timedelta"
            if name in {"days", "seconds", "microseconds"}:
                return getattr(value, name)
        elif isinstance(value, dict):
            method_type = "dict"
        elif isinstance(value, list):
            method_type = "list"
        elif isinstance(value, set):
            method_type = "set"
        elif isinstance(value, tuple):
            method_type = "tuple"
        elif isinstance(value, str):
            method_type = "str"
        elif isinstance(value, bytes):
            method_type = "bytes"
        else:
            method_type = None
        if method_type is not None:
            policy = BOUND_METHOD_POLICIES.get(method_type, {}).get(name)
            if policy is not None:
                return self._bound_method(value, name, policy)
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

    def _set_list_slice(self, value, key, item):
        start, stop, step = key.indices(len(value))
        removed = range(start, stop, step)
        removed_cost = sum(self._collection_item_cost(value[index]) for index in removed)
        remaining_items = self.MAX_ITEMS - (len(value) - len(removed))
        if remaining_items < 0:
            raise _EvalError("python_exec: list assignment item limit exceeded")
        source = self._iter_values(item)
        if source is value:
            if len(source) > remaining_items:
                raise _EvalError("python_exec: list assignment item limit exceeded")
            source = tuple(source)
        pending = []
        cost = self._collection_cost(value) - removed_cost
        for new_item in source:
            if len(pending) >= remaining_items:
                raise _EvalError("python_exec: list assignment item limit exceeded")
            cost += self._collection_item_cost(new_item)
            if cost > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: list assignment display size limit exceeded")
            pending.append(new_item)
        if step != 1 and len(pending) != len(removed):
            raise _EvalError("python_exec: list assignment size mismatch")
        value[key] = pending
        self._validate_mutated(value)

    def _setitem(self, value, key, item):
        self._plain(key)
        _validate_value(item)
        if isinstance(value, dict):
            self._set_dict_item(value, key, item, "dict assignment")
            return
        if isinstance(value, list):
            if isinstance(key, slice):
                self._set_list_slice(value, key, item)
                return
            try:
                old = value[key]
                cost = (
                    self._collection_cost(value)
                    - self._collection_item_cost(old)
                    + self._collection_item_cost(item)
                )
                if cost > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: list assignment display size limit exceeded")
                value[key] = item
                self._validate_mutated(value)
            except _EvalError:
                raise
            except (IndexError, KeyError, TypeError) as exc:
                raise _EvalError("python_exec: assignment failed: %s" % exc) from exc
            return
        raise _EvalError("python_exec: subscription assignment is not available")

    def _binary(self, operator, left, right):
        left = self._plain(left)
        right = self._plain(right)
        self._check_integer_binary(operator, left, right)
        self._check_sequence_operator(operator, left, right)
        if isinstance(operator, ast.Mod) and isinstance(left, (str, bytes)):
            raise _EvalError("python_exec: string formatting operator is not available")
        if isinstance(operator, ast.Mult):
            for sequence, multiplier in ((left, right), (right, left)):
                if isinstance(sequence, (str, bytes, list, tuple)) and isinstance(
                    multiplier, int
                ):
                    self._check_sequence_multiplication(sequence, multiplier)
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
        if isinstance(result, (list, dict, set)):
            return self._validate_collection_result(result, "operator result")
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
            values = self._iter_values(value)
            if len(values) != len(target.elts):
                raise _EvalError("python_exec: unpacking assignment failed")
            for item_target, item in zip(target.elts, values):
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

    def _check_sequence_operator(self, operator, left, right):
        if isinstance(operator, ast.Add):
            if type(left) is type(right) and isinstance(left, (str, bytes)):
                if len(left) + len(right) > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: concatenation result size limit exceeded")
            elif type(left) is type(right) and isinstance(left, (list, tuple)):
                if len(left) + len(right) > self.MAX_ITEMS:
                    raise _EvalError("python_exec: concatenation result item limit exceeded")
                if self._display_size(left) + self._display_size(right) > self.MAX_STRING_CHARS:
                    raise _EvalError(
                        "python_exec: concatenation result display size limit exceeded"
                    )
            elif type(left) is type(right) and isinstance(left, (set, frozenset, dict)):
                if len(left) + len(right) > self.MAX_ITEMS:
                    raise _EvalError("python_exec: collection operator item limit exceeded")
                if self._display_size(left) + self._display_size(right) > self.MAX_STRING_CHARS:
                    raise _EvalError("python_exec: collection operator display size limit exceeded")
        if isinstance(operator, ast.BitOr) and type(left) is type(right) and isinstance(
            left, (set, frozenset, dict)
        ):
            if len(left) + len(right) > self.MAX_ITEMS:
                raise _EvalError("python_exec: collection operator item limit exceeded")
            if self._display_size(left) + self._display_size(right) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: collection operator display size limit exceeded")
        if (
            isinstance(operator, (ast.BitAnd, ast.BitXor))
            and type(left) is type(right)
            and isinstance(left, (set, frozenset))
        ):
            if len(left) + len(right) > self.MAX_ITEMS:
                raise _EvalError("python_exec: collection operator item limit exceeded")
            if self._display_size(left) + self._display_size(right) > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: collection operator display size limit exceeded")

    def _check_sequence_multiplication(self, sequence, multiplier):
        if not isinstance(multiplier, int) or multiplier <= 1:
            return
        limit = self.MAX_STRING_CHARS if isinstance(sequence, (str, bytes)) else self.MAX_ITEMS
        if len(sequence) and multiplier > limit // len(sequence):
            if isinstance(sequence, (str, bytes)):
                raise _EvalError("python_exec: multiplication result limit exceeded")
            raise _EvalError("python_exec: multiplication result item limit exceeded")
        projected = len(sequence) * multiplier
        if isinstance(sequence, (str, bytes)):
            if projected > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: multiplication result limit exceeded")
        elif isinstance(sequence, (list, tuple)):
            base = self._display_size(sequence)
            if base > 2 and base * multiplier > self.MAX_STRING_CHARS:
                raise _EvalError("python_exec: string conversion size limit exceeded")

    def _comprehension(self, generators, emit, environment, result=None, add=None):
        if result is None:
            result = self._new_list()
            cost = 2
            has_mutable = False

            def add(value):
                nonlocal cost, has_mutable
                cost, has_mutable = self._append_collection_item(
                    result, value, "comprehension result", cost, has_mutable
                )

        elif add is None:
            raise _EvalError("python_exec: internal comprehension collector is missing")

        def visit(index, current):
            if index == len(generators):
                add(emit(current))
                return
            generator = generators[index]
            values = self._iter_values(self._eval(generator.iter, current))
            for value in values:
                child = _Environment(current)
                self._store(generator.target, value, child)
                if all(self._truth(self._eval(condition, child)) for condition in generator.ifs):
                    visit(index + 1, child)

        visit(0, _Environment(environment))
        self._validate_mutated(result)
        return result

    def _eval(self, node, environment):
        self._tick()
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (bool, int, float, str, bytes)):
                _validate_value(node.value)
                return node.value
            raise _EvalError("python_exec: literal type is not available")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise _EvalError("python_exec: dunder identifier %r is not available" % node.id)
            return environment.get(node.id)
        if isinstance(node, ast.List):
            result = self._new_list()
            cost = 2
            has_mutable = False
            for item in node.elts:
                cost, has_mutable = self._append_collection_item(
                    result,
                    self._eval(item, environment),
                    "list literal",
                    cost,
                    has_mutable,
                )
            self._validate_mutated(result)
            return result
        if isinstance(node, ast.Tuple):
            result = self._new_list()
            cost = 2
            has_mutable = False
            for item in node.elts:
                cost, has_mutable = self._append_collection_item(
                    result,
                    self._eval(item, environment),
                    "tuple literal",
                    cost,
                    has_mutable,
                )
            result = tuple(result)
            self._validate_mutated(result)
            return result
        if isinstance(node, ast.Set):
            result = self._new_set()
            cost = 2
            has_mutable = False
            for item in node.elts:
                value = self._eval(item, environment)
                cost, has_mutable = self._add_set_item(
                    result, value, "set literal", cost, has_mutable
                )
            self._validate_mutated(result)
            return result
        if isinstance(node, ast.Dict):
            result = self._new_dict()
            cost = 2
            has_mutable = False
            for key, value in zip(node.keys, node.values):
                if key is None:
                    unpacked = self._plain(self._eval(value, environment))
                    for unpacked_key, unpacked_value in unpacked.items():
                        cost, has_mutable = self._set_dict_item(
                            result,
                            unpacked_key,
                            unpacked_value,
                            "dict literal",
                            cost,
                            has_mutable,
                        )
                else:
                    evaluated_key = self._plain(self._eval(key, environment))
                    evaluated_value = self._eval(value, environment)
                    cost, has_mutable = self._set_dict_item(
                        result,
                        evaluated_key,
                        evaluated_value,
                        "dict literal",
                        cost,
                        has_mutable,
                    )
            self._validate_mutated(result)
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
                    for item in self._iter_values(value):
                        if len(args) >= self.MAX_ITEMS:
                            raise _EvalError("python_exec: call argument item limit exceeded")
                        args.append(item)
                else:
                    if len(args) >= self.MAX_ITEMS:
                        raise _EvalError("python_exec: call argument item limit exceeded")
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
            return self._concat_strings(
                (self._eval(value, environment) for value in node.values), "f-string"
            )
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
                value = self._safe_format(self._plain(value), self._plain(spec))
            return value
        if isinstance(node, ast.ListComp):
            return self._comprehension(
                node.generators, lambda current: self._eval(node.elt, current), environment
            )
        if isinstance(node, ast.SetComp):
            result = self._new_set()
            cost = 2
            has_mutable = False

            def add(value):
                nonlocal cost, has_mutable
                cost, has_mutable = self._add_set_item(
                    result, value, "set comprehension", cost, has_mutable
                )

            return self._comprehension(
                node.generators,
                lambda current: self._eval(node.elt, current),
                environment,
                result=result,
                add=add,
            )
        if isinstance(node, ast.DictComp):
            result = self._new_dict()
            cost = 2
            has_mutable = False

            def add(pair):
                nonlocal cost, has_mutable
                key, value = pair
                cost, has_mutable = self._set_dict_item(
                    result, key, value, "dict comprehension", cost, has_mutable
                )

            return self._comprehension(
                node.generators,
                lambda current: (
                    self._eval(node.key, current),
                    self._eval(node.value, current),
                ),
                environment,
                result=result,
                add=add,
            )
        if isinstance(node, ast.GeneratorExp):
            return self._comprehension(
                node.generators, lambda current: self._eval(node.elt, current), environment
            )
        if isinstance(node, ast.NamedExpr):
            value = self._eval(node.value, environment)
            self._store(node.target, value, environment)
            return value
        _unsupported_syntax(type(node).__name__)

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
            _unsupported_syntax(type(node).__name__)

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
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS

    def __post_init__(self) -> None:
        _validate_max_output_chars(self.max_output_chars)
        object.__setattr__(self, "stdout", _clip(self.stdout, self.max_output_chars))
        object.__setattr__(self, "stderr", _clip(self.stderr, self.max_output_chars))

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
            return _clip(_join(self.stdout, self.stderr), self.max_output_chars)
        if self.ok:
            return _clip(self.stdout, self.max_output_chars)
        return _clip(
            _join(self.stdout, self.stderr or f"python_exec: exited with {self.returncode}"),
            self.max_output_chars,
        )


def run(
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> PythonExecResult:
    """Evaluate a restricted calculation subset and return its structured result."""

    if not isinstance(code, str):
        raise TypeError("python_exec: code must be a string")
    if timeout_s <= 0:
        raise ValueError("python_exec: timeout_s must be positive")
    _validate_max_output_chars(max_output_chars)

    bootstrap = (
        _BOOTSTRAP.replace("__ALLOWED__", repr(sorted(ALLOWED_IMPORTS)))
        .replace("__BOUND_METHOD_POLICIES__", repr(BOUND_METHOD_POLICIES))
        .replace("__NETWORK_MESSAGE__", repr(NETWORK_BLOCKED_MESSAGE))
        .replace("__MAX_OUTPUT__", repr(max_output_chars))
        .replace("__MAX_INTEGER_BITS__", repr(MAX_INTEGER_BITS))
        .strip()
    )
    # -I isolates the child from PYTHON* variables and the user site directory;
    # the temporary cwd keeps a stray open() away from the repository.  The
    # bootstrap is written to a file instead of passed via ``-c``: Windows
    # caps a child process's command line at 32,767 characters and the
    # capability evaluator is well past that, while the agent's own code
    # still only ever travels on stdin.
    with tempfile.TemporaryDirectory(prefix="occam-python-exec-") as workdir:
        bootstrap_path = os.path.join(workdir, "_bootstrap.py")
        with open(bootstrap_path, "w", encoding="utf-8") as bootstrap_file:
            bootstrap_file.write(bootstrap)
        argv = [sys.executable, "-I", "-B", bootstrap_path]
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
                stderr=_clip(TIMEOUT_MESSAGE.format(timeout=timeout_s), max_output_chars),
                returncode=None,
                timed_out=True,
                max_output_chars=max_output_chars,
            )
        except OSError as exc:  # pragma: no cover - interpreter is missing
            raise PythonExecError(f"python_exec: could not start the sandbox: {exc}") from exc

    return PythonExecResult(
        stdout=_clip(completed.stdout, max_output_chars),
        stderr=_clip(completed.stderr, max_output_chars),
        returncode=completed.returncode,
        timed_out=False,
        max_output_chars=max_output_chars,
    )


def python_exec(
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """Evaluate a restricted calculation subset; arbitrary Python is unsupported."""

    _validate_max_output_chars(max_output_chars)
    return run(code, timeout_s=timeout_s, max_output_chars=max_output_chars).as_text()


def _validate_max_output_chars(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("python_exec: max_output_chars must be a positive integer")
    if not 0 < value <= MAX_OUTPUT_CHARS_LIMIT:
        raise ValueError(
            f"python_exec: max_output_chars must be between 1 and {MAX_OUTPUT_CHARS_LIMIT}"
        )


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
    if len(value) <= limit:
        return value
    marker = f"\n... [truncated at {limit} characters]"
    if len(marker) >= limit:
        return value[:limit]
    return value[: limit - len(marker)] + marker


__all__ = [
    "ALLOWED_IMPORTS",
    "BOUND_METHOD_GUARDS",
    "BOUND_METHOD_POLICIES",
    "CPYTHON_SYNTHESIZED_ENV",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_TIMEOUT_S",
    "ENV_PASSTHROUGH",
    "MAX_OUTPUT_CHARS_LIMIT",
    "MAX_INTEGER_BITS",
    "NETWORK_BLOCKED_MESSAGE",
    "child_env",
    "PythonExecError",
    "PythonExecResult",
    "python_exec",
    "run",
]
