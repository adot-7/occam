# occam

## `python_exec` containment

The FX calculator's `python_exec` tool runs in a short-lived child with a
sanitized environment, bounded timeout/output, and an AST capability evaluator.
Submitted source is parsed and interpreted; it is never compiled or executed as
Python. The evaluator exposes only the arithmetic, `Decimal`, date/time, JSON,
and small standard-library capabilities used by the task. Filesystem-, process-,
and network-capable modules or builtins, arbitrary callable dispatch, and
interpreter introspection are absent, so unsupported routes fail closed before
they can run. CPython may
synthesize `LC_CTYPE` while starting a POSIX child; that key is not passed
through by Occam and is not an application secret.

This is a calculation surface, not a perfect security sandbox for arbitrary
hostile Python. It is not a claim of arbitrary-code host isolation: evaluator
bugs, parser/runtime denial of service, and the child interpreter itself still
require OS-level controls for a hard boundary (for example a separate user,
container with no repository mounts, seccomp/job controls, or a dedicated
sandbox service).
