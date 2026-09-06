"""Run agent-written arithmetic in a bounded child process.

``python_exec`` is intentionally a *capability-limited runner*, not a general
Python sandbox.  The child gets an explicit environment allow-list, a small
safe-builtin namespace, a strict AST check, and an exact standard-library
module allow-list.  Timeout, bounded output, a temporary working directory,
and a second socket guard remain defense in depth.  These layers cover the
known filesystem, process, introspection, secret, and network escape routes
needed by the FX calculation task while retaining ``Decimal``, JSON, date/time,
and ordinary arithmetic.

The boundary is not perfect isolation.  This module must not be described as
safe for arbitrary hostile Python: Python's object model and implementation
details are too large for a subprocess wrapper to prove secure.  Production
deployment that needs a security boundary still needs OS-level isolation (for
example a container, separate user, seccomp/job controls, or a dedicated
sandbox service).  In particular, the temporary directory is a convenience,
not a claim that arbitrary filesystem access is impossible; the AST and
capability checks reject the known file/process routes before execution.
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

#: Top-level modules agent code may import.  Deliberately arithmetic- and
#: text-shaped: everything needed to add up a ledger, nothing that opens a
#: socket, spawns a process or directly opens the engine's files.  ``io`` is
#: intentionally absent: even ``io.open`` would be a filesystem capability.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "array",
        "base64",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "copy",
        "csv",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "itertools",
        "json",
        "math",
        "numbers",
        "pprint",
        "random",
        "re",
        "statistics",
        "textwrap",
        "time",
        "unicodedata",
    }
)

NETWORK_BLOCKED_MESSAGE = "python_exec: network access is disabled"
TIMEOUT_MESSAGE = "python_exec: timed out after {timeout:g}s"

# Runs inside the child.  It reads the agent's program from stdin so the code
# never has to survive a round trip through shell quoting.
_BOOTSTRAP = '''
import ast
import builtins
import importlib
import sys

ALLOWED = frozenset(__ALLOWED__)
NETWORK_BLOCKED_MESSAGE = __NETWORK_MESSAGE__
MAX_OUTPUT_CHARS = __MAX_OUTPUT__

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

# Import the allow-list up front: afterwards the guards below reject everything,
# including the lazy internal imports an allowed module might otherwise make.
for name in sorted(ALLOWED):
    try:
        importlib.import_module(name)
    except ImportError:
        pass


def _no_network(*_args, **_kwargs):
    raise OSError(NETWORK_BLOCKED_MESSAGE)


# socket underlies every stdlib network client, so neutering it closes urllib,
# http.client and anything an agent might have vendored in.
import socket

for _attr in (
    "socket",
    "socketpair",
    "create_connection",
    "create_server",
    "getaddrinfo",
    "gethostbyname",
):
    setattr(socket, _attr, _no_network)


def _reject(root):
    raise ImportError("python_exec: import of %r is not allowed" % root)


class _ImportGuard:
    """Reject fresh imports of anything outside the allow-list."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ALLOWED:
            _reject(fullname)
        return None


_real_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    # sys.modules is consulted inside __import__, so wrapping it is what stops
    # `import os` succeeding just because the interpreter loaded os at startup.
    if name not in ALLOWED:
        _reject(name)
    return _real_import(name, globals, locals, fromlist, level)


sys.meta_path.insert(0, _ImportGuard())
builtins.__import__ = _guarded_import


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


tree = ast.parse(source, filename="<python_exec>", mode="exec")
_Policy().visit(tree)

_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytes",
    "callable",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
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
)
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
_SAFE_BUILTINS["__import__"] = _guarded_import

exec(
    compile(tree, "<python_exec>", "exec"),
    {"__name__": "__main__", "__builtins__": _SAFE_BUILTINS},
)
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
