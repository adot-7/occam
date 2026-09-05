# AGENTS.md — read this first

You are an AI coding agent (orchestrator or worker) building **Occam**, a Track 1 submission for the Syndicate by Maximor hackathon. This file tells you how to work in this repo. The product itself is specified in `prd/`.

## What Occam is, in three sentences

Occam is an automated agent-engineering system: given a goal, a set of tools and an eval set, it generates a multi-agent architecture, runs it, and iteratively improves it. Its distinguishing move is **causal ablation**: after each run it knocks out every role one at a time and measures whether the final answers change, so it can delete roles that cost tokens but exert no causal influence ("expensive witnesses"). The demo shows an architecture getting *smaller* while accuracy holds and cost falls.

## Read order

1. `prd/00-PRD.md` — what we're building, why, what "done" means
2. `prd/01-ARCHITECTURE.md` — modules, state model, the event log that joins engine and TUI
3. `prd/02-DATA-AND-TASKS.md` — task packs, the three domains, exact file formats, live-task flow
4. `prd/03-ABLATION-AND-METRICS.md` — the algorithm, its cost, noise handling, metric definitions
5. `prd/04-TUI-SPEC.md` — Textual screens and widgets
6. `prd/05-WORK-PACKAGES.md` — ordered, independently-shippable work packages with acceptance criteria
7. `prd/06-SPONSOR-INTEGRATIONS.md` — TensorMux (primary compute), GPT-5 nano, Neatlogs tracing; where each plugs in
8. `prd/07-CITATIONS.md` — verified references; README/pitch only, never the TUI
9. `prd/08-END-TO-END-WALKTHROUGH.md` — one run traced with every number worked; if the TUI shows a number you can't derive this way, it's a bug
10. `prd/09-DEMO-SHOTLIST.md` — what `occam replay` must support, shot by shot

If a PRD and this file disagree, the PRD wins. If two PRDs disagree, `01-ARCHITECTURE.md` wins on structure and `03-ABLATION-AND-METRICS.md` wins on algorithm.

## Non-negotiables

- **Python 3.11+, single package `occam/`.** No second language. TUI is [Textual](https://textual.textualize.io/).
- **Engine and TUI never import each other.** They communicate only through the run directory (`runs/<run_id>/events.jsonl` + `state.json`) defined in `prd/01-ARCHITECTURE.md`. This is what lets two humans and many workers build in parallel.
- **Every event written to `events.jsonl` must validate against `schemas/events.schema.json`.** Schema changes require updating the schema file, the fixture replay, and a note in `CHANGELOG.md` in the same PR.
- **Deterministic where possible.** temperature=0, fixed seeds, sorted iteration. Nondeterminism that remains must be measured, not hidden (see `03`).
- **No secrets in the repo.** Keys via `.env` (gitignored); `.env.example` lists every variable.
- **Model access is config-driven.** All model IDs live in `occam/config/models.yaml`. Never hardcode a model string in code — free-tier rosters change weekly.
- **Bulk rollouts go through metered/free API keys (Groq, Gemini, TensorMux), never through Claude Code or Codex subscriptions.** Those are for you, the coding agent, not for the product.
- **Tests:** `pytest` must pass. Every module gets at least smoke tests. Ablation and metrics get property tests (see `03`).
- **Fixture replay must always work:** `occam replay fixtures/demo_run` renders the TUI with no network. This is also the demo backup.

## Working conventions for AO sessions

- One work package (`prd/05`) = one AO worker session = one branch = one PR. Name branches `wp/<nn>-<slug>`.
- Read the work package's **acceptance criteria** before starting; your PR description must show each one met (command + output).
- Do not widen scope. If you discover the PRD is wrong or incomplete, stop, write your finding to `prd/OPEN-QUESTIONS.md` under a dated heading, and ask the orchestrator.
- Keep PRs small enough to review in ten minutes. Two medium PRs beat one large one.
- Commit messages: imperative, one line, then a blank line, then *why*.
- Run `ruff check . && ruff format --check . && pytest -q` before opening a PR.

## Repo layout (target)

```
occam/
  AGENTS.md                 ← you are here
  README.md                 ← submission-facing; written in WP-12
  CHANGELOG.md
  pyproject.toml
  .env.example
  prd/                      ← specs (this folder)
  schemas/
    events.schema.json
    state.schema.json
    task.schema.json
  occam/
    __init__.py
    cli.py                  ← `occam` entrypoint (typer)
    config/
      models.yaml
      settings.py
    core/                   ← data types (pydantic): Architecture, Role, Case, RunResult, Event…
    llm/                    ← provider clients, cost accounting, caching
    tasks/                  ← task pack loader + domain adapters (smfr, bfcl, mgsm) + live-task generator
    tools/                  ← tool implementations exposed to candidate agents
    engine/
      architect.py          ← propose architecture
      executor.py           ← run architecture on cases (DAG executor)
      ablation.py           ← causal ablation
      diagnose.py           ← failure + ablation → diagnosis + mutation choice
      mutate.py             ← the closed mutation menu
      baseline.py           ← cost-matched CoT-SC
      loop.py               ← orchestrates one full run, emits events
    metrics/                ← pass rate, cost, latency, structural fidelity, CIs
    store/                  ← run directory I/O, event writer/reader, state snapshots
    tui/                    ← Textual app; reads store only
  fixtures/
    demo_run/events.jsonl   ← hand-authored replay for TUI dev + demo backup
  tests/
```

## Definitions you'll see everywhere

- **Task pack** — a directory with `task.yaml` + `cases.jsonl`. The unit of "a domain". See `02`.
- **Architecture** — a DAG of **roles**. Each role = system prompt + model + tool bindings + inputs/outputs. See `01`.
- **Generation** — one architecture version in the improvement loop. Gen 0 is the architect's first proposal.
- **Run** — one complete loop on one task pack: many generations, one `runs/<run_id>/`.
- **Ablation** — re-running the eval with one role knocked out. See `03`.
- **Influence** — paired change in outcomes when a role is removed. **Witness** — a role with ≈0 influence and >0 cost.
- **Structural fidelity** — share of spend that is causally load-bearing. See `03`.
- **Baseline** — cost-matched CoT-SC (single agent, sampled k times, majority vote). See `03`.
