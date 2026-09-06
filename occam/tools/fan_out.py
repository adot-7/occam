"""``fan_out`` — the one multi-agent primitive available *inside* a role.

The calling role's own prompt is re-run once per subtask, concurrently and
bounded.  The architect can also express parallelism structurally as separate
DAG roles; ``fan_out`` is the version a single role can reach for at run time.

Results come back in subtask order regardless of completion order, because the
executor's traces and the grader both have to be reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_SUBTASKS = 16

#: A role runner takes one subtask string and returns that role's answer.
RoleRunner = Callable[[str], str]


class FanOutUnavailableError(RuntimeError):
    """Raised when ``fan_out`` is called without a role to fan out to."""


def fan_out(
    subtasks: Sequence[str],
    *,
    runner: RoleRunner | None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_subtasks: int = DEFAULT_MAX_SUBTASKS,
) -> list[str]:
    """Run ``runner`` on each subtask concurrently and return one result each.

    A subtask that raises yields an error string in its slot rather than failing
    the whole batch: one bad branch should cost the role one branch, not the
    case.
    """

    if runner is None:
        raise FanOutUnavailableError(
            "fan_out is only available to a role; no calling role prompt is bound"
        )
    if isinstance(subtasks, str) or not isinstance(subtasks, Sequence):
        raise TypeError("fan_out: subtasks must be a list of strings")
    items = list(subtasks)
    if not all(isinstance(item, str) for item in items):
        raise TypeError("fan_out: subtasks must be a list of strings")
    if max_workers < 1:
        raise ValueError("fan_out: max_workers must be at least 1")
    if len(items) > max_subtasks:
        raise ValueError(f"fan_out: at most {max_subtasks} subtasks per call, got {len(items)}")
    if not items:
        return []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = [pool.submit(runner, item) for item in items]
        return [_settle(future.result, index) for index, future in enumerate(futures)]


def _settle(resolve: Callable[[], str], index: int) -> str:
    try:
        result = resolve()
    except Exception as exc:
        # One branch failing must not fail the batch.
        return f"[fan_out error on subtask {index + 1}: {exc}]"
    return result if isinstance(result, str) else str(result)


__all__ = [
    "DEFAULT_MAX_SUBTASKS",
    "DEFAULT_MAX_WORKERS",
    "FanOutUnavailableError",
    "RoleRunner",
    "fan_out",
]
