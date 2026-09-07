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
- WP-06 architect, immutable mutation lineage, and validated JSONL/Markdown lessons
  memory, including `occam lessons show|reset`.
- WP-07 read-only Textual shell (`occam/tui/`): header, lineage tree, diagnosis
  feed, layout skeleton, keymap, palette, and `update_view` panel seams for WP-09.
- WP-07 replay driver: `occam replay <run_dir> [--speed] [--to-gen] [--at] [--pause]`
  with silent fast-forward, capped inter-event gaps, pause/step/speed transport,
  and `occam tui --run <run_dir>` for live or finished runs.
- `store.reducer.Reduction`, the incremental form of `reduce()`, so a follower can
  fold new events without re-reducing the whole log. Both paths run the same
  transition function; `reduce()` is now defined in terms of it.
- WP-15 optional Neatlogs tracing: import-order-safe OpenAI initialization, contextual
  completion/case/tool spans, honest cache attributes, and run-end flush/shutdown.

- WP-01c contract patch: `ablation.started` now carries the generation's
  `noise_rate` (required), so the ablation table's noise floor is readable while
  the table fills instead of only after `metrics.snapshot`; `RoleTrace` gains
  `billed_cost_usd` and `cost_label` so granted spend is shown at its list-rate
  equivalent and labelled as such. WP-05 now carries both cost bases from each
  completion into the role trace, including cache-hit labels. Both fixtures
  updated; tests cover reducer retention and schema-copy parity.

- WP-08 ablation engine: noise floor from a `full_repeat` pass, paired divergence
  and influence, bootstrap influence CIs, the coded verdict rule, structural
  fidelity, knockout DAG helpers, and the stratified ablation subset.
- WP-08 metrics package: Wilson pass-rate intervals, cost/token/latency
  aggregation with p50/p90, tool calls per case, cost shares, and the
  `metrics.snapshot` builder.

### Fixed

- Execution case failures now retain checker errors and expose bounded answer
  prefixes plus checker/role reasons in the validated event projection, without
  copying raw provider traces into events.
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
- Added regression-tested resource guards for positive child/parent output
  bounds, recursive repr/ascii/JSON serialization, bounded print streaming,
  string/bytes aggregators, collection mutation, and integer power/shift and
  magnitude boundaries. This remains capability containment, not a perfect
  hostile-Python sandbox or OS isolation boundary; the residual OS caveat in
  the README still applies.
- Removed mutable display-cost state in `python_exec`; current-graph display
  validation now recounts aliases and detects cycles separately. Decimal-to-
  integer conversions, `round`, `math.ceil`, and `math.floor` preflight
  metadata before conversion, while byte-base parsing and `strftime`
  directives retain their bounded contracts. Collection operations may make
  bounded transient copies; this is not a claim of zero-copy execution.
- `EventWriter` now locks with `msvcrt` on Windows and `fcntl` elsewhere. The
  store imported `fcntl` at module level, so `occam.store` was unimportable on
  Windows and three test modules failed at collection. Covered by a fresh-
  interpreter import test and a cross-process concurrent append test.
- `test_full_repeat_is_a_fresh_nonzero_accounted_run` compares the fixture's
  floating-point total within tolerance instead of requiring bit-for-bit equality.

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
