"""Capability-surface guarantees for ``python_exec``: limits and no escapes."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time

import pytest

from occam.tools.python_exec import (
    _BOOTSTRAP,
    ALLOWED_IMPORTS,
    BOUND_METHOD_GUARDS,
    BOUND_METHOD_POLICIES,
    CPYTHON_SYNTHESIZED_ENV,
    DEFAULT_MAX_OUTPUT_CHARS,
    ENV_PASSTHROUGH,
    MAX_INTEGER_BITS,
    MAX_OUTPUT_CHARS_LIMIT,
    NETWORK_BLOCKED_MESSAGE,
    PythonExecResult,
    child_env,
    python_exec,
    run,
)

# Windows's CreateProcess refuses a command line longer than this many
# characters ([WinError 206]).  The capability evaluator inlined into the
# bootstrap is well past it, which is why the bootstrap must reach the child
# through a file rather than a ``-c`` argument.
_WINDOWS_ARGV_LIMIT = 32_767


def _rendered_bootstrap() -> str:
    return (
        _BOOTSTRAP.replace("__ALLOWED__", repr(sorted(ALLOWED_IMPORTS)))
        .replace("__BOUND_METHOD_POLICIES__", repr(BOUND_METHOD_POLICIES))
        .replace("__NETWORK_MESSAGE__", repr(NETWORK_BLOCKED_MESSAGE))
        .replace("__MAX_OUTPUT__", repr(DEFAULT_MAX_OUTPUT_CHARS))
        .replace("__MAX_INTEGER_BITS__", repr(MAX_INTEGER_BITS))
        .strip()
    )


def test_returns_stdout_of_a_successful_program() -> None:
    assert python_exec("print(2 + 2)") == "4\n"


def test_public_docstring_describes_the_restricted_calculation_surface() -> None:
    assert "restricted calculation subset" in (python_exec.__doc__ or "")
    assert "arbitrary Python is unsupported" in (python_exec.__doc__ or "")


@pytest.mark.parametrize(
    ("type_name", "method_name", "policy"),
    tuple(
        (type_name, method_name, policy)
        for type_name, methods in BOUND_METHOD_POLICIES.items()
        for method_name, policy in methods.items()
    ),
)
def test_every_exposed_bound_method_has_an_explicit_resource_policy(
    type_name: str, method_name: str, policy: str
) -> None:
    assert type_name
    assert method_name
    assert policy in BOUND_METHOD_GUARDS


def test_set_growth_methods_are_not_exposed() -> None:
    result = run("values = set()\nvalues.add('sentinel')")

    assert not result.ok
    assert "attribute 'add' is not available" in result.stderr


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


def test_full_size_bootstrap_exceeds_the_windows_argv_limit() -> None:
    """The real bootstrap is far past what ``-c`` could carry on Windows.

    A test built from a short, hand-written bootstrap would not reproduce the
    launch failure: the AST capability evaluator inlined into ``_BOOTSTRAP``
    is what pushes the command line over Windows's limit.
    """
    assert len(_rendered_bootstrap()) > _WINDOWS_ARGV_LIMIT


def test_sandbox_starts_with_the_real_full_size_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: python_exec.py used to pass ``_BOOTSTRAP`` via ``-c``.

    On Windows, ``CreateProcess`` rejects a command line over 32,767
    characters, so the ~100K-character bootstrap could never launch the
    sandbox at all ([WinError 206]).  The fix writes the bootstrap to a file
    inside the run's temporary directory and launches that file instead, so
    every argv element must stay short and no element may be the bootstrap
    itself.
    """
    captured_argv: list[list[str]] = []
    real_run = subprocess.run

    def spy(argv, *args, **kwargs):
        captured_argv.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)

    result = run("print(1 + 1)")

    assert result.ok
    assert result.stdout == "2\n"
    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert "-c" not in argv
    bootstrap = _rendered_bootstrap()
    assert bootstrap not in argv
    for arg in argv:
        assert len(arg) < _WINDOWS_ARGV_LIMIT


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


def test_thirty_six_adversarial_escape_attempts_cannot_exfiltrate_sentinels(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_file = tmp_path / ".env-sentinel"
    secret_file.write_text("OCCAM_FILE_SENTINEL=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("OCCAM_SECRET_SENTINEL", "environment-secret")
    path = str(secret_file)
    attempts = (
        f"open({path!r}).read()",
        "open('relative-sentinel').read()",
        f"import pathlib\npathlib.Path({path!r}).read_text()",
        f"import io\nio.open({path!r}).read()",
        f"import os\nos.open({path!r}, os.O_RDONLY)",
        "import os\nos.system('echo PROCESS_SENTINEL')",
        "import subprocess\nsubprocess.run(['echo', 'PROCESS_SENTINEL'])",
        "import shutil\nshutil.which('python')",
        "import ctypes\nctypes.CDLL('libc.so.6')",
        "import socket\nsocket.create_connection(('example.com', 80), 1)",
        "import urllib\nprint(urllib)",
        "import http\nprint(http)",
        "import requests\nprint(requests)",
        "__import__('os')",
        "__import__.__globals__['sys'].modules['os']",
        "def f():\n    pass\nprint(f.__globals__)",
        "def f():\n    pass\nprint(f.__code__)",
        "try:\n    1 / 0\nexcept Exception as exc:\n    print(exc.__traceback__)",
        "try:\n    1 / 0\nexcept Exception as exc:\n"
        "    print(exc.__traceback__.tb_frame.f_globals)",
        "def g():\n    yield 1\nprint(g().gi_frame)",
        "().__class__.__mro__",
        "object.__subclasses__()",
        "type(1)",
        "globals()",
        "locals()",
        "eval('1')",
        "exec('print('PROCESS_SENTINEL')')",
        "compile('print('PROCESS_SENTINEL')', '<x>', 'exec')",
        "memoryview(b'x')",
        "bytearray(1)",
        "import importlib.machinery\nprint(importlib.machinery)",
        "import sys\nprint(sys.modules)",
        "import multiprocessing\nprint(multiprocessing)",
        "__builtins__['open']('relative-sentinel')",
        "__builtins__['__import__']('os')",
        "import subprocess\n"
        "subprocess.run(['python', '-c', \"import socket; "
        "socket.create_connection(('example.com', 80), 1)\"])",
    )

    assert len(attempts) == 36
    for code in attempts:
        result = run(code, timeout_s=2.0)
        output = result.as_text()
        assert not result.ok
        assert not result.timed_out
        assert "OCCAM_FILE_SENTINEL" not in output
        assert "file-secret" not in output
        assert "environment-secret" not in output
        assert "PROCESS_SENTINEL" not in output


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


@pytest.mark.parametrize("limit", [1, 2, 32, 100, DEFAULT_MAX_OUTPUT_CHARS])
def test_output_limit_includes_markers_at_every_positive_bound(limit: int) -> None:
    result = run("print('x' * 100_000)", max_output_chars=limit)
    marker = f"\n... [truncated at {limit} characters]"

    assert len(result.stdout) == limit
    assert len(result.stderr) <= limit
    assert len(result.as_text()) <= limit
    if len(marker) < limit:
        assert result.stdout.endswith(marker)
    else:
        assert result.stdout == "x" * limit


@pytest.mark.parametrize("limit", [1, 2, 32, 100, DEFAULT_MAX_OUTPUT_CHARS])
def test_parent_result_clipping_is_also_a_strict_total_bound(limit: int) -> None:
    result = PythonExecResult(
        stdout="stdout" * 1000,
        stderr="stderr" * 1000,
        returncode=1,
        timed_out=False,
        max_output_chars=limit,
    )

    assert len(result.stdout) <= limit
    assert len(result.stderr) <= limit
    assert len(result.as_text()) <= limit


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
        (
            "repr",
            "repr('x' * 999_999)",
            "string conversion size limit exceeded",
        ),
        (
            "ascii",
            "ascii('x' * 999_999)",
            "string conversion size limit exceeded",
        ),
        (
            "json",
            "import json\njson.dumps(['x'] * 100_000, indent=100)",
            "JSON output size limit exceeded",
        ),
    ),
    ids=lambda case: case[0],
)
def test_aggregate_renderers_reject_before_building_large_representations(
    label: str, code: str, message: str
) -> None:
    result = run(code, timeout_s=2.0)

    assert not result.ok, label
    assert not result.timed_out, label
    assert message in result.stderr, label


def test_mutable_aliases_are_recounted_after_child_mutation() -> None:
    result = run(
        "inner = ['x' * 300_000]\n"
        "outer = [inner]\n"
        "inner.append('y' * 300_000)\n"
        "outer.append('z' * 500_000)\n"
        "print('COMPLETED')",
        timeout_s=3.0,
    )

    assert not result.ok
    assert not result.timed_out
    assert "list.append display size limit exceeded" in result.stderr
    assert "COMPLETED" not in result.as_text()


def test_aliases_count_as_repeated_display_and_cycles_use_separate_detection() -> None:
    repeated = run(
        "inner = ['x' * 500_000]\nouter = [inner, inner]\nprint('COMPLETED')",
        timeout_s=3.0,
    )
    cycle = run("value = []\nvalue.append(value)\nprint(value)", timeout_s=3.0)

    assert not repeated.ok
    assert not repeated.timed_out
    assert "display size limit exceeded" in repeated.stderr
    assert "COMPLETED" not in repeated.as_text()
    assert not cycle.ok
    assert not cycle.timed_out
    assert "cycle detected" in cycle.stderr


@pytest.mark.parametrize(
    "code",
    (
        "from decimal import Decimal\nprint(int(Decimal('1e1000000')))\nprint('COMPLETED')",
        "from decimal import Decimal\nprint(round(Decimal('1e1000000')))\nprint('COMPLETED')",
        "from decimal import Decimal\nimport math\n"
        "print(math.ceil(Decimal('1e1000000')))\nprint('COMPLETED')",
        "from decimal import Decimal\nimport math\n"
        "print(math.floor(Decimal('1e1000000')))\nprint('COMPLETED')",
    ),
)
def test_decimal_to_integer_paths_reject_huge_exponents_before_conversion(code: str) -> None:
    result = run(code, timeout_s=2.0)

    assert not result.ok
    assert not result.timed_out
    assert "integer magnitude limit exceeded" in result.stderr
    assert "COMPLETED" not in result.as_text()


def test_decimal_integer_paths_preserve_normal_fx_sized_values() -> None:
    result = run(
        "from decimal import Decimal\n"
        "import math\n"
        "print(int(Decimal('12.9')), round(Decimal('1.25')), "
        "math.ceil(Decimal('1.1')), math.floor(Decimal('1.9')))",
    )

    assert result.ok
    assert result.stdout == "12 1 2 1\n"


def test_bytes_integer_parsing_honors_the_explicit_base() -> None:
    result = run("print(int(b'ff', 16), int(b'101', 2))")

    assert result.ok
    assert result.stdout == "255 5\n"


def test_strftime_directives_are_preflighted_before_formatting() -> None:
    result = run(
        "import datetime\n"
        "print(datetime.date(2026, 1, 2).strftime('%Y' * 300_000))\n"
        "print('COMPLETED')",
        timeout_s=2.0,
    )

    assert not result.ok
    assert not result.timed_out
    assert "formatted string size limit exceeded" in result.stderr
    assert "COMPLETED" not in result.as_text()


def test_safe_list_slice_assignment_is_reachable_and_bounded() -> None:
    result = run("values = [1, 2]\nvalues[1:1] = [3, 4]\nprint(values)")
    oversized = run(
        "values = [''] * 100_000\nvalues[0:0] = ['']\nprint('COMPLETED')",
        timeout_s=2.0,
    )

    assert result.ok
    assert result.stdout == "[1, 3, 4, 2]\n"
    assert not oversized.ok
    assert not oversized.timed_out
    assert "item limit exceeded" in oversized.stderr
    assert "COMPLETED" not in oversized.as_text()


def test_print_streams_each_argument_and_separator_into_the_bounded_sink() -> None:
    result = run(
        "print('x' * 800_000, 'y' * 800_000, sep='s' * 800_000, end='e' * 800_000)",
        max_output_chars=128,
    )

    assert result.ok
    assert len(result.stdout) == 128
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("label", "code"),
    (
        (
            "repeated strings",
            "values = []\n"
            "values.append('x' * 600_000)\n"
            "values.append('y' * 600_000)\n"
            "print('COMPLETED')",
        ),
        (
            "1.1M append attempt",
            "values = []\n"
            "for index in range(1_100_000):\n"
            "    values.append('')\n"
            "print('COMPLETED')",
        ),
        (
            "1.1M extend attempt",
            "values = []\n"
            "for index in range(1_100_000):\n"
            "    values.extend([''])\n"
            "print('COMPLETED')",
        ),
        (
            "append item bound",
            "values = [''] * 100_000\nvalues.append('')\nprint('COMPLETED')",
        ),
        (
            "insert item bound",
            "values = [''] * 100_000\nvalues.insert(0, '')\nprint('COMPLETED')",
        ),
        (
            "extend item bound",
            "values = [''] * 100_000\nvalues.extend([''])\nprint('COMPLETED')",
        ),
        (
            "f-string",
            "print(f\"{'x' * 600_000}{'y' * 600_000}\")\nprint('COMPLETED')",
        ),
        (
            "bytes",
            "print((b'x' * 600_000).hex())\nprint('COMPLETED')",
        ),
    ),
    ids=lambda case: case[0],
)
def test_growth_probes_reject_before_completion(label: str, code: str) -> None:
    result = run(code, timeout_s=3.0)

    assert not result.ok, label
    assert not result.timed_out, label
    assert "COMPLETED" not in result.as_text(), label


@pytest.mark.parametrize(
    ("label", "code"),
    (
        ("join", "print(('x' * 500_000).join(['y' * 300_000, 'z' * 300_000]))"),
        ("center", "print('x'.center(1_000_001))"),
        ("ljust", "print('x'.ljust(1_000_001))"),
        ("rjust", "print('x'.rjust(1_000_001))"),
        ("zfill", "print('1'.zfill(1_000_001))"),
        ("replace", "print(('x' * 500_001).replace('x', 'xx'))"),
        ("encode", "print(('x' * 300_000).encode('utf-32'))"),
        ("bytes.hex", "print((b'x' * 600_000).hex())"),
    ),
    ids=lambda case: case[0],
)
def test_exposed_string_and_bytes_aggregators_are_preflighted(label: str, code: str) -> None:
    result = run(code, timeout_s=3.0)

    assert not result.ok, label
    assert not result.timed_out, label
    assert not result.stdout, label


@pytest.mark.parametrize(
    ("label", "code"),
    (
        (
            "dict assignment",
            "values = {index: 0 for index in range(100_000)}\n"
            "values[100_000] = 0\n"
            "print('COMPLETED')",
        ),
        (
            "dict setdefault",
            "values = {index: 0 for index in range(100_000)}\n"
            "values.setdefault(100_000, 0)\n"
            "print('COMPLETED')",
        ),
    ),
    ids=lambda case: case[0],
)
def test_dict_growth_is_checked_before_mutation(label: str, code: str) -> None:
    result = run(code, timeout_s=3.0)

    assert not result.ok, label
    assert not result.timed_out, label
    assert "COMPLETED" not in result.as_text(), label


def test_json_loads_checks_aggregate_items_before_parser_allocation() -> None:
    result = run(
        "import json\n"
        "payload = '[' + ('0,' * 100_001) + '0]'\n"
        "json.loads(payload)\n"
        "print('COMPLETED')",
        timeout_s=3.0,
    )

    assert not result.ok
    assert not result.timed_out
    assert "collection item limit exceeded" in result.stderr
    assert "COMPLETED" not in result.as_text()


@pytest.mark.parametrize(
    "code",
    (
        f"print(2 ** {MAX_INTEGER_BITS})",
        "print(2 ** 1_000_000)",
        f"print(1 << {MAX_INTEGER_BITS})",
        "print(1 << 1_000_000)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) * 2)",
    ),
)
def test_explosive_integer_operations_fail_before_growth(code: str) -> None:
    result = run(code, timeout_s=2.0)

    assert not result.ok
    assert not result.timed_out
    assert "integer magnitude limit exceeded" in result.stderr


def test_integer_bit_boundary_allows_normal_sized_results() -> None:
    power = run(f"print(2 ** {MAX_INTEGER_BITS - 1} > 0)", timeout_s=2.0)
    shift = run(f"print(1 << {MAX_INTEGER_BITS - 1} > 0)", timeout_s=2.0)

    assert power.ok and power.stdout == "True\n"
    assert shift.ok and shift.stdout == "True\n"


@pytest.mark.parametrize(
    "code",
    (
        f"print(2 ** {MAX_INTEGER_BITS - 1} > 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) + (2 ** {MAX_INTEGER_BITS - 1} - 1) > 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) - (-(2 ** {MAX_INTEGER_BITS - 1} - 1)) > 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 2}) * 2 > 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) * 0 == 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) * -1 < 0)",
        f"print(1 << {MAX_INTEGER_BITS - 1} > 0)",
        f"print(True << {MAX_INTEGER_BITS - 1} > 0)",
        f"print(1 >> {MAX_INTEGER_BITS} == 0)",
        f"print(int('1' * {MAX_INTEGER_BITS}, 2) > 0)",
        f"print(int('-' + '1' * {MAX_INTEGER_BITS}, 2) < 0)",
        "print(int('z' * 19_342, 36) > 0)",
    ),
)
def test_integer_operations_allow_exact_bounded_results(code: str) -> None:
    result = run(code, timeout_s=4.0)

    assert result.ok
    assert result.stdout == "True\n"


@pytest.mark.parametrize(
    "code",
    (
        f"print(2 ** {MAX_INTEGER_BITS} > 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) + (2 ** {MAX_INTEGER_BITS - 1}) > 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) - (-(2 ** {MAX_INTEGER_BITS - 1})) > 0)",
        f"print((2 ** {MAX_INTEGER_BITS - 1}) * 2 > 0)",
        f"print(1 << {MAX_INTEGER_BITS} > 0)",
        f"print(True << {MAX_INTEGER_BITS} > 0)",
        f"print(1 >> {MAX_INTEGER_BITS + 1} == 0)",
        f"print(int('1' * {MAX_INTEGER_BITS + 1}, 2) > 0)",
        "print(int('z' * 19_343, 36) > 0)",
        "print(2 ** 1_000_000)",
    ),
)
def test_integer_operations_reject_one_over_before_completion(code: str) -> None:
    result = run(code, timeout_s=4.0)

    assert not result.ok
    assert not result.timed_out
    assert "integer magnitude limit exceeded" in result.stderr


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
    assert (
        "item limit exceeded" in result.stderr or "display size limit exceeded" in result.stderr
    ), label


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
