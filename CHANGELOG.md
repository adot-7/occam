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
- WP-04 `python_exec` sandbox: isolated subprocess, wall-clock timeout, neutered
  sockets, an allow-list import guard, and an environment allow-list so the child
  cannot read the engine's API keys into a trace, the event log, or a fixture.
- WP-04 `fan_out`, the in-role multi-agent primitive: bounded concurrency, results in
  subtask order, one failing branch does not fail the batch.

### Fixed

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
