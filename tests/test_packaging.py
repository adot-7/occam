"""Regression coverage for schemas shipped in the installable package."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from importlib.resources import files
from pathlib import Path

from occam.store.schema import load_schema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAMES = ("events.schema.json", "state.schema.json", "task.schema.json")


def test_schema_resources_are_available_as_package_data() -> None:
    package_schemas = files("occam.schemas")
    for name in SCHEMA_NAMES:
        packaged = package_schemas.joinpath(name)
        source = ROOT / "schemas" / name
        assert packaged.is_file()
        assert packaged.read_bytes() == source.read_bytes()
        assert load_schema(name)["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_wheel_contains_schema_resources_and_loads_them_outside_checkout(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_path = next(wheel_dir.glob("occam-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert {f"occam/schemas/{name}" for name in SCHEMA_NAMES} <= names

    installed = tmp_path / "installed"
    installed.mkdir()
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(installed)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from occam.store.schema import load_schema; "
            "assert all(load_schema(name) for name in "
            "('events.schema.json', 'state.schema.json', 'task.schema.json'))",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
