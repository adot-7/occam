# Changelog

All notable changes to Occam are documented here.

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
