"""Content-addressed, best-effort disk caching for LLM responses."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a request deterministically for hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class DiskCache:
    """Read/write JSON values by SHA-256 content address.

    Cache failures are deliberately non-fatal.  A bad or partially written
    entry is treated as a miss and a provider call can repair it.
    """

    def __init__(self, directory: str | Path, global_directory: str | Path | None = None) -> None:
        directories = [Path(directory)]
        if global_directory is not None:
            global_path = Path(global_directory)
            if global_path not in directories:
                directories.append(global_path)
        self.directories = tuple(directories)
        self._lock = RLock()

    @staticmethod
    def address(request: Any) -> str:
        """Return the content address for a request payload."""

        return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()

    def _path(self, directory: Path, address: str) -> Path:
        return directory / f"{address}.json"

    def get(self, address: str) -> dict[str, Any] | None:
        """Return the first valid cached mapping, if present."""

        with self._lock:
            for directory in self.directories:
                path = self._path(directory, address)
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    continue
                if isinstance(value, dict):
                    return value
        return None

    def put(self, address: str, value: Mapping[str, Any]) -> None:
        """Atomically write a response to each configured cache root."""

        payload = canonical_json(dict(value)).encode("utf-8")
        with self._lock:
            for directory in self.directories:
                temporary_path: Path | None = None
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=directory,
                        prefix=f".{address}.",
                        suffix=".tmp",
                        delete=False,
                    ) as temporary:
                        temporary.write(payload)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                        temporary_path = Path(temporary.name)
                    os.replace(temporary_path, self._path(directory, address))
                except OSError:
                    if temporary_path is not None:
                        try:
                            temporary_path.unlink(missing_ok=True)
                        except OSError:
                            pass


__all__ = ["DiskCache", "canonical_json"]
