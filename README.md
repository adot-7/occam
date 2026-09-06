# occam

## `python_exec` containment

The FX calculator's `python_exec` tool runs in a short-lived child with a
sanitized environment, bounded timeout/output, no network-capable imports, and
an AST plus safe-builtin capability allow-list. It preserves the arithmetic,
`Decimal`, date/time, and JSON operations used by the task while rejecting
known absolute-file, process, and interpreter-introspection routes. CPython may
synthesize `LC_CTYPE` while starting a POSIX child; that key is not passed
through by Occam and is not an application secret.

This is containment for model-written calculations, not a perfect security
sandbox for arbitrary hostile Python. A deployment that needs a hard security
boundary must add OS-level isolation such as a separate user, container,
seccomp/job controls, or a dedicated sandbox service.
