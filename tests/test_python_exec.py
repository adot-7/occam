"""Sandbox guarantees for ``python_exec``: termination, no network, no imports."""

from __future__ import annotations

import ast
import time

import pytest

from occam.tools.python_exec import (
    ALLOWED_IMPORTS,
    ENV_PASSTHROUGH,
    NETWORK_BLOCKED_MESSAGE,
    PythonExecResult,
    python_exec,
    run,
)


def test_returns_stdout_of_a_successful_program() -> None:
    assert python_exec("print(2 + 2)") == "4\n"


def test_allowed_imports_cover_the_arithmetic_an_fx_ledger_needs() -> None:
    code = (
        "from decimal import Decimal\n"
        "import datetime, json, math\n"
        "print(json.dumps({'d': str(Decimal('1.10') * 3), "
        "'day': datetime.date(2026, 4, 4).isoformat(), 'r': round(math.pi, 2)}, sort_keys=True))"
    )
    assert python_exec(code) == '{"d": "3.30", "day": "2026-04-04", "r": 3.14}\n'


@pytest.mark.parametrize("module", ["socket", "urllib", "http", "httpx", "requests", "ftplib"])
def test_network_capable_modules_cannot_be_imported(module: str) -> None:
    result = run(f"import {module}\nprint('imported')")

    assert not result.ok
    assert "imported" not in result.stdout
    assert f"import of '{module}' is not allowed" in result.stderr


def test_a_socket_reached_around_the_import_guard_still_cannot_connect() -> None:
    # Second layer: reaching sys.modules through the guard's own closure skips
    # the import check entirely, so the socket module itself is neutered too.
    code = (
        "sock = __builtins__['__import__'].__globals__['sys'].modules['socket']\n"
        "try:\n"
        "    sock.create_connection(('example.com', 80), timeout=2)\n"
        "except OSError as exc:\n"
        "    print(exc)\n"
    )
    assert python_exec(code).strip() == NETWORK_BLOCKED_MESSAGE


def test_the_child_inherits_none_of_the_engines_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # os is blocked by both import guards but sits in sys.modules regardless
    # (random pulls in urandom during the allow-list preload), so introspection
    # can still reach os.environ. Anything printed there would land in
    # RoleTrace.tool_calls, events.jsonl and the committed fixtures.
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TENSORMUX_API_KEY"):
        monkeypatch.setenv(name, "sk-must-not-leak")

    code = (
        "environ = __builtins__['__import__'].__globals__['sys'].modules['os'].environ\n"
        "print(sorted(environ))\n"
    )
    text = python_exec(code)

    assert "sk-must-not-leak" not in text
    inherited = ast.literal_eval(text.strip())
    leaked = sorted(set(inherited) - set(ENV_PASSTHROUGH))
    assert leaked == [], f"child inherited unexpected variables: {leaked}"


def test_the_environment_allow_list_carries_no_credentials() -> None:
    assert not any(
        marker in name
        for name in ENV_PASSTHROUGH
        for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")
    )


def test_process_and_filesystem_escapes_are_blocked() -> None:
    for module in ("os", "subprocess", "shutil", "importlib", "ctypes", "pathlib"):
        result = run(f"import {module}")
        assert not result.ok, module
        assert f"import of '{module}' is not allowed" in result.stderr, module


def test_import_guard_survives_an_importlib_style_bypass() -> None:
    result = run("__import__('socket').create_connection(('example.com', 80))")

    assert not result.ok
    assert "import of 'socket' is not allowed" in result.stderr


def test_an_endless_program_is_killed_at_the_timeout() -> None:
    started = time.perf_counter()
    result = run("while True:\n    pass", timeout_s=1.0)
    elapsed = time.perf_counter() - started

    assert result.timed_out
    assert result.returncode is None
    assert "timed out after 1s" in result.as_text()
    assert elapsed < 20.0


def test_a_raising_program_returns_its_traceback_instead_of_raising() -> None:
    text = python_exec("print('before')\nraise ValueError('boom')")

    assert text.startswith("before")
    assert "ValueError: boom" in text


def test_output_is_clipped_so_one_call_cannot_flood_a_transcript() -> None:
    text = python_exec("print('x' * 5000)", max_output_chars=100)

    assert len(text) < 200
    assert "truncated at 100 characters" in text


def test_the_child_cannot_see_the_engines_own_package() -> None:
    result = run("import occam\nprint(occam.__file__)")

    assert not result.ok
    assert "import of 'occam' is not allowed" in result.stderr


def test_allow_list_excludes_every_network_and_process_module() -> None:
    forbidden = {
        "asyncio",
        "ctypes",
        "ftplib",
        "http",
        "httpx",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "urllib",
    }
    assert not (ALLOWED_IMPORTS & forbidden)


def test_result_ok_is_false_for_a_timeout() -> None:
    assert not PythonExecResult(stdout="", stderr="", returncode=None, timed_out=True).ok


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_a_nonpositive_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        run("print(1)", timeout_s=timeout)
