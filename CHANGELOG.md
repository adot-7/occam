# Changelog

All notable changes to Occam are documented here.

## [Unreleased]

### Added

- WP-01b v3 contract fields: lessons, lesson-written and pass³ reliability events,
  run memory metadata, per-case sub-results, and tool-call/reliability metrics.
- Hand-authored FX revaluation replay fixtures for run 1 and run 2.
- Replay fixtures now record the explicit 10-case stratified ablation subset,
  measure repeat noise on that subset, and attach run 1 pass³ to best generation g3.
- WP-02 LLM layer with OpenAI-compatible and Anthropic providers, model config
  interpolation, bounded retries, per-model rate limiting, content-addressed cache,
  native tools, and labelled list-rate-equivalent grant accounting.
- WP-02 review hardening: explicit zero-temperature forwarding, Anthropic native
  structured outputs and multi-turn tool adaptation, approximate grant labels,
  poisoned-cache rejection, and non-overriding project `.env` loading.
- WP-03 FX packs, Frankfurter cache-backed client, formula reference implementation,
  and tolerance-aware `fx_total` grader.
- WP-04 tool registry mapping `fx_rate`, `fx_series`, `python_exec` and `fan_out` to
  `(ToolSpec, callable)` pairs, with deliberately plain base descriptions, the
  architect-time `tool_note` append hook, and per-call latency/bytes/status/cached
  accounting for successful and failed calls alike.
- WP-04 `python_exec` calculation surface: isolated subprocess, wall-clock
  timeout, a strict AST capability evaluator, and an explicit environment
  allow-list so submitted calculations cannot read the engine's API keys into a
  trace, the event log, or a fixture.
- WP-04 `fan_out`, the in-role multi-agent primitive: bounded concurrency, results in
  subtask order, one failing branch does not fail the batch.

- WP-01c contract patch: `ablation.started` now carries the generation's
  `noise_rate` (required), so the ablation table's noise floor is readable while
  the table fills instead of only after `metrics.snapshot`; `RoleTrace` gains
  `billed_cost_usd` and `cost_label` so granted spend is shown at its list-rate
  equivalent and labelled as such. WP-05 now carries both cost bases from each
  completion into the role trace, including cache-hit labels. Both fixtures
  updated; tests cover reducer retention and schema-copy parity.

### Fixed

- Replaced `python_exec`'s arbitrary `exec` path with a fail-closed AST
  capability evaluator that preserves the FX arithmetic/Decimal/JSON surface,
  bounds source, values, steps, timeout, and output, and rejects filesystem,
  process, network, and introspection escapes before execution. Added sentinel
  regressions for absolute files, recovered `os.system`, subprocess launch, and
  a subprocess-based network escape. The documentation explicitly treats this
  as model-calculation containment rather than a perfect arbitrary-Python
  sandbox.
- Added bounded public output-limit validation, pre-allocation guards for large
  constructors, and incremental collection limits for comprehensions and
  generator expressions. The plain tool description and callable docstring now
  identify the restricted calculation subset and its actionable unsupported-
  construct errors.
- Closed the remaining calculation-surface resource gaps: child and parent
  output clipping now includes truncation markers within the requested bound;
  repr/ascii/JSON preflight cumulative display size and print streams through a
  bounded sink; and power, shifts, and integer results enforce a pre-operation
  bit-length limit. This remains capability containment, not a perfect hostile
  Python sandbox or OS isolation boundary.
- `EventWriter` now locks with `msvcrt` on Windows and `fcntl` elsewhere. The
  store imported `fcntl` at module level, so `occam.store` was unimportable on
  Windows and three test modules failed at collection. Covered by a fresh-
  interpreter import test and a cross-process concurrent append test.

## [0.1.0] - 2026-09-05

### Added

- WP-01 repository skeleton and Python package layout.
- Versioned JSON Schemas for event logs, state snapshots, and task manifests.
- Complete metrics snapshots with pass-rate CIs, token/reliability/speed measures,
  and p50/p90 latency quantiles.
- Pydantic v2 core models for architectures, roles, cases, traces, results, and events.
- Deterministic event reader/writer and pure event reducer.
- `occam validate` for fixture schema validation and reducer determinism checks.
- Hand-authored four-generation SMFR replay fixture.
- Schema resources packaged in the installable wheel.
- Strict empty-run and derived-state validation, append-safe writers, and live-tail retry handling.
