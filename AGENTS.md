# AGENTS.md — read this first

You are an AI coding agent (orchestrator or worker) building **Occam**, a Track 1 submission for the Syndicate by Maximor hackathon. This file tells you how to work in this repo. The product is specified in `prd/`. **This is v3 (final). Earlier drafts mentioned SMFR/BFCL/MGSM benchmarks or a memory-only "v2" — both are gone; if you see them in old branches or PR descriptions, ignore them.**

## What Occam is, in four sentences

Occam is an automated agent-engineering system: given a goal, real tools and an eval set, it architects a multi-agent team, runs it with live HTTP calls to a third-party API, and iteratively improves it. After each run it **ablates every role** — re-runs with that role knocked out — and prunes roles that exert no causal influence on the answers ("expensive witnesses"), so teams get smaller and cheaper while accuracy holds. It also **writes lessons** — typed, evidence-backed rules about the tool's behaviour and the domain's conventions — and the **next run's architect reads them first**, so a new team for a new ledger starts where the last one finished. The task is month-end FX revaluation of a multi-currency receivables ledger using the ECB rates API (Frankfurter); the TUI shows lineage, the ablation table, cases, metrics vs a cost-matched CoT-SC baseline, the lessons growing, and the run-over-run comparison.

## Read order

1. `prd/00-PRD.md` — what, why, success criteria
2. `prd/02-DATA-AND-TASKS.md` — the workflow, the formula, the tool, the two packs, what "run 2" means
3. `prd/01-ARCHITECTURE.md` — modules, data model incl. `Lesson`, the event log that joins engine and TUI, loop, CLI
4. `prd/03-ABLATION-AND-METRICS.md` — the algorithm, cost, noise, verdicts, structural fidelity, baseline, pass³
5. `prd/08-END-TO-END-WALKTHROUGH.md` — two runs traced with every number worked; if the TUI shows a number you can't derive this way, it's a bug
6. `prd/04-TUI-SPEC.md` — Textual screens (§1b lists the v3 additions)
7. `prd/05-WORK-PACKAGES.md` — ordered work packages with acceptance criteria and hour budgets
8. `prd/09-DEMO-SHOTLIST.md` — what `occam replay` must support, shot by shot
9. `prd/06-SPONSOR-INTEGRATIONS.md`, `prd/07-CITATIONS.md`, `prd/OPEN-QUESTIONS.md`

If docs disagree: `01` wins on structure, `02` on task/formula/tool, `03` on algorithm.

## Non-negotiables

- **Python 3.11+, single package `occam/`.** TUI is Textual. No second language.
- **Engine and TUI never import each other.** They communicate only through `runs/<run_id>/events.jsonl` + `state.json` (`01 §1`). `occam replay` must be indistinguishable from live.
- **Every event validates against `schemas/events.schema.json`.** Schema changes update the schema, both fixtures, and `CHANGELOG.md` in the same PR.
- **The formula in `02 §1.2` is the only definition of the answer.** Grader, reference implementation and README all cite it. Change it in one place or not at all.
- **Tool base descriptions stay plain.** Nothing about holidays or the range endpoint. Those are *lessons* the system must learn; hardcoding them defeats the product.
- **Lessons pass the leak guard** (`01 §4.4`). A lesson containing a date, an invoice id, or a case value is a bug.
- **Deterministic where possible:** temperature 0, seeds, sorted iteration; remaining nondeterminism is measured (noise floor, pass³), never hidden.
- **Model IDs and rates live in `occam/config/models.yaml`.** Never hardcode a model string.
- **Bulk rollouts use metered/granted API keys (TensorMux, OpenAI), never Claude Code or Codex subscriptions.**
- **No secrets in the repo.** `.env` gitignored; `.env.example` lists every variable. `data/fx_cache/` IS committed (public historical rates).
- **Tests:** `pytest` passes; ablation, leak guard, grader and reference implementation have unit/property tests.
- **Fixture replay always works offline:** `occam replay fixtures/demo_run1` and `fixtures/demo_run2`.

## Working conventions for AO sessions

- One work package = one AO worker session = one branch `wp/<nn>-<slug>` = one PR. Paste the WP's acceptance criteria into the PR with evidence (command + output).
- Do not widen scope. If a PRD is wrong or blocks you, append to `prd/OPEN-QUESTIONS.md` under a dated heading and ask the orchestrator.
- PRs reviewable in ten minutes. Two medium PRs beat one large one.
- `ruff check . && ruff format --check . && pytest -q` before opening a PR.
- Commit messages: imperative one-liner, blank line, *why*.

## Repo layout (target)

```
occam/
  AGENTS.md  README.md  CHANGELOG.md  pyproject.toml  .env.example
  prd/                       specs
  schemas/                   events / state / task JSON Schemas
  design/                    Figma reference (visual target; PRD wins on information content)
  occam/
    cli.py                   occam run | tui | replay | compare | lessons | validate | llm ping
    config/models.yaml, settings.py
    core/                    pydantic models: ToolSpec, Role, Architecture, Case, CaseResult, RunResult, Lesson, Event
    llm/                     providers, rate limiting, cost, cache, tracing (Neatlogs)
    tasks/                   pack loader, checkers (fx_total), fx_reference.py
    tools/                   registry, fx.py (fx_rate, fx_series + disk cache), python_exec, fan_out
    engine/                  architect, executor, ablation, diagnose (+lesson writer, leak guard), mutate, baseline, loop, compare
    metrics/                 pass rate, cost, latency, calls/case, SF, CIs, pass³
    memory/                  lessons store (jsonl + md render)
    store/                   run dir I/O, event writer/reader, reducer
    tui/                     Textual app (read-only)
  tasks/fx_recon_a/  tasks/fx_recon_b/          committed packs
  data/fx_cache/                                 committed API cache
  memory/fx_recon/lessons.jsonl                  written by run 1, read by run 2
  fixtures/demo_run1/  fixtures/demo_run2/        replay fixtures
  scripts/gen_fx_cases.py  scripts/record_demo.sh
  tests/
```

## Vocabulary
**Task pack** — `task.yaml` + `cases.jsonl` (`02`). **Case** — one ledger snapshot → total FX gain/loss + per-invoice breakdown. **Architecture** — DAG of **roles**. **Generation** — one architecture version within a run. **Run** — one full loop on one pack; run 2 reads run 1's lessons. **Ablation** — re-run with one role knocked out. **Influence / divergence / witness / structural fidelity** — `03`. **Lesson** — a typed rule (tool_note | domain_rule) with evidence, written by diagnose, read by the next architect. **Baseline** — cost-matched CoT-SC. **pass³** — reliability on the final generation.
