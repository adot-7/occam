"""``python_exec`` — run agent-written Python in a throwaway subprocess.

Three properties matter, in this order: the call always terminates (wall-clock
timeout), it cannot reach the network, and it can only import from a small
allow-list.  The child is a fresh isolated interpreter, so nothing an agent does
can touch the engine's own state.

The guards are for a confused LLM, not a hostile one: they turn "the model tried
to `pip install requests`" into a legible error instead of a hung run.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000

#: Top-level modules agent code may import.  Deliberately arithmetic- and
#: text-shaped: everything needed to add up a ledger, nothing that opens a
#: socket, spawns a process or reads the engine's files.
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
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "io",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "pprint",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "time",
        "types",
        "typing",
        "unicodedata",
    }
)

NETWORK_BLOCKED_MESSAGE = "python_exec: network access is disabled"
TIMEOUT_MESSAGE = "python_exec: timed out after {timeout:g}s"

# Runs inside the child.  It reads the agent's program from stdin so the code
# never has to survive a round trip through shell quoting.
_BOOTSTRAP = '''
import builtins
import importlib
import sys

ALLOWED = frozenset(__ALLOWED__)
NETWORK_BLOCKED_MESSAGE = __NETWORK_MESSAGE__

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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
        root = fullname.partition(".")[0]
        if root not in ALLOWED:
            _reject(root)
        return None


_real_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    # sys.modules is consulted inside __import__, so wrapping it is what stops
    # `import os` succeeding just because the interpreter loaded os at startup.
    root = name.partition(".")[0]
    if root and root not in ALLOWED:
        _reject(root)
    return _real_import(name, globals, locals, fromlist, level)


sys.meta_path.insert(0, _ImportGuard())
builtins.__import__ = _guarded_import

exec(compile(source, "<python_exec>", "exec"), {"__name__": "__main__"})
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
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_TIMEOUT_S",
    "NETWORK_BLOCKED_MESSAGE",
    "PythonExecError",
    "PythonExecResult",
    "python_exec",
    "run",
]
