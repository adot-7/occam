"""Capability-surface guarantees for ``python_exec``: limits and no escapes."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time

import pytest

from occam.tools.python_exec import (
    ALLOWED_IMPORTS,
    CPYTHON_SYNTHESIZED_ENV,
    ENV_PASSTHROUGH,
    MAX_OUTPUT_CHARS_LIMIT,
    PythonExecResult,
    child_env,
    python_exec,
    run,
)


def test_returns_stdout_of_a_successful_program() -> None:
    assert python_exec("print(2 + 2)") == "4\n"


def test_public_docstring_describes_the_restricted_calculation_surface() -> None:
    assert "restricted calculation subset" in (python_exec.__doc__ or "")
    assert "arbitrary Python is unsupported" in (python_exec.__doc__ or "")


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


def test_child_env_is_explicit_and_allows_cpython_locale_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TENSORMUX_API_KEY"):
        monkeypatch.setenv(name, "sk-must-not-leak")
    monkeypatch.setenv("LC_CTYPE", "secret-locale-value")

    expected = {name: os.environ[name] for name in ENV_PASSTHROUGH if name in os.environ}
    actual = child_env()

    assert actual == expected
    assert set(actual) <= set(ENV_PASSTHROUGH)
    assert "LC_CTYPE" not in ENV_PASSTHROUGH
    assert CPYTHON_SYNTHESIZED_ENV == {"LC_CTYPE"}
    assert all("must-not-leak" not in value for value in actual.values())

    # CPython may add LC_CTYPE while starting with an otherwise empty POSIX
    # environment.  That implementation detail is allowed in the child, but
    # it is not a value passed through by child_env().
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", "import os; print(sorted(os.environ.items()))"],
        env=actual,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = dict(ast.literal_eval(completed.stdout))
    assert set(observed) <= set(ENV_PASSTHROUGH) | set(CPYTHON_SYNTHESIZED_ENV)
    assert all(
        name not in observed
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TENSORMUX_API_KEY")
    )
    assert "secret-locale-value" not in completed.stdout


def test_absolute_env_file_read_is_blocked(tmp_path) -> None:
    secret_file = tmp_path / ".env"
    secret_file.write_text("OCCAM_FILE_SENTINEL=repo-secret\n", encoding="utf-8")

    result = run(f"print(open({str(secret_file)!r}, encoding='utf-8').read())")

    assert not result.ok
    assert "OCCAM_FILE_SENTINEL=repo-secret" not in result.as_text()
    assert "builtin 'open' is not allowed" in result.stderr


def test_process_and_introspection_routes_cannot_print_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCCAM_SECRET_SENTINEL", "environment-secret")
    attempts = (
        "__builtins__['__import__'].__globals__['sys'].modules['os'].system("
        "'echo PROCESS_SENTINEL')",
        "__builtins__['__import__'].__globals__['sys'].modules['os'].environ["
        "'OCCAM_SECRET_SENTINEL']",
        "def f():\n    pass\nprint('{0.__globals__}'.format(f))",
        "try:\n    1 / 0\nexcept Exception as exc:\n"
        "    print(exc.__traceback__.tb_frame.f_globals)",
        "def g():\n    yield 1\nprint(g().gi_frame.f_back.f_globals)",
    )

    for code in attempts:
        result = run(code)
        assert not result.ok
        output = result.as_text()
        assert "PROCESS_SENTINEL" not in output
        assert "environment-secret" not in output


@pytest.mark.parametrize(
    ("label", "code"),
    (
        (
            "process launch",
            "import subprocess\n"
            "subprocess.Popen(['python', '-c', \"print('PROCESS_LAUNCH_SENTINEL')\"])",
        ),
        (
            "child network",
            "import subprocess\n"
            "subprocess.run(['python', '-c', \"import socket; "
            "socket.create_connection(('example.com', 80), 1)\"])",
        ),
    ),
    ids=lambda case: case[0],
)
def test_subprocess_and_child_network_escapes_are_rejected(label: str, code: str) -> None:
    result = run(code, timeout_s=1.0)

    assert not result.ok, label
    assert "PROCESS_LAUNCH_SENTINEL" not in result.as_text()
    assert "import of 'subprocess' is not allowed" in result.stderr


def test_the_environment_allow_list_carries_no_credentials() -> None:
    assert not any(
        marker in name
        for name in ENV_PASSTHROUGH
        for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")
    )


def test_process_and_filesystem_escapes_are_blocked() -> None:
    for module in (
        "os",
        "subprocess",
        "shutil",
        "importlib",
        "ctypes",
        "pathlib",
        "io",
        "operator",
    ):
        result = run(f"import {module}")
        assert not result.ok, module
        assert f"import of '{module}' is not allowed" in result.stderr, module


def test_ast_guard_rejects_dynamic_import_and_network_bypass() -> None:
    result = run("__import__('socket').create_connection(('example.com', 80))")

    assert not result.ok
    assert "dunder identifier '__import__' is not allowed" in result.stderr


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


@pytest.mark.parametrize(
    "limit",
    [0, -1, True, 1.5, MAX_OUTPUT_CHARS_LIMIT + 1],
)
def test_output_limit_must_be_a_positive_bounded_integer(limit) -> None:
    with pytest.raises((TypeError, ValueError), match="max_output_chars"):
        run("print('ok')", max_output_chars=limit)


def test_python_exec_public_entry_also_rejects_disabled_output_clipping() -> None:
    with pytest.raises(ValueError, match="max_output_chars"):
        python_exec("print('ok')", max_output_chars=0)


def test_unsupported_syntax_reports_the_restricted_surface() -> None:
    result = run("class NotAvailable:\n    pass")

    assert not result.ok
    assert "unsupported by the restricted calculation subset" in result.stderr


@pytest.mark.parametrize(
    ("label", "code", "message"),
    (
        ("bytes", "bytes(2 ** 60)", "bytes constructor size exceeds"),
        ("bytearray", "bytearray(2 ** 60)", "builtin 'bytearray' is not allowed"),
        ("string", "str('x' * (2 ** 60))", "multiplication result limit exceeded"),
        (
            "string conversion",
            "str(['x' * 100_000] * 100_000)",
            "string conversion size limit exceeded",
        ),
        ("list", "list(range(2 ** 60))", "range limit exceeded"),
        ("tuple", "tuple(range(2 ** 60))", "range limit exceeded"),
    ),
    ids=lambda case: case[0],
)
def test_oversized_constructors_fail_before_large_allocation(
    label: str, code: str, message: str
) -> None:
    result = run(code, timeout_s=1.0)

    assert not result.ok, label
    assert not result.timed_out, label
    assert message in result.stderr, label


@pytest.mark.parametrize(
    ("label", "code"),
    (
        (
            "list comprehension",
            "[x for group in range(2) for x in range(50_001)]",
        ),
        (
            "set comprehension",
            "{group * 50_001 + x for group in range(2) for x in range(50_001)}",
        ),
        (
            "dict comprehension",
            "{group * 50_001 + x: x for group in range(2) for x in range(50_001)}",
        ),
        (
            "generator expression",
            "sum(x for group in range(2) for x in range(50_001))",
        ),
    ),
    ids=lambda case: case[0],
)
def test_comprehension_results_are_limited_incrementally(label: str, code: str) -> None:
    result = run(code, timeout_s=2.0)

    assert not result.ok, label
    assert not result.timed_out, label
    assert "item limit exceeded" in result.stderr, label


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
