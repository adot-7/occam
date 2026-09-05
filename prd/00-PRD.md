# 00 — Product Requirements: Occam

> Working name **Occam** (as in the razor: delete what isn't doing work). Rename freely; the CLI is `occam`.

## 1. Problem

Automated agent-design systems (ADAS, AFlow, MaAS, DyLAN, MAS-Zero…) produce architectures with more agents than the task needs. *The Illusion of Multi-Agent Advantage* (Jwalapuram et al., arXiv:2606.13003, June 2026) audited six such frameworks and found: agents reach unanimous consensus in >90% of GPT-5 cases; an "all-assistant" configuration beat task-specific experts (54.4% vs 53.4%); verifiers pick the first answer >45% of the time; later agents are "expensive witnesses that incur full inference costs while exerting near-zero causal influence." Their explicit call: evaluate architectures on **structural fidelity** — "the degree to which assigned agentic roles exert measurable causal influence on the final decision."

Nobody has built that into the design loop. Occam does.

## 2. What Occam does

Given **a goal, a tool manifest, and an eval set** (together: a *task pack*), Occam:

1. **Architects** a candidate multi-agent architecture (roles, prompts, tool bindings, DAG wiring).
2. **Executes** it against the eval set, recording per-case pass/fail, tokens, dollars, latency.
3. **Ablates** every role — re-runs with the role knocked out — and computes each role's causal influence and cost share.
4. **Diagnoses** failures and witnesses in plain language and selects **one** mutation from a closed menu.
5. **Mutates** and loops until plateau or budget.
6. Throughout, compares against a **cost-matched CoT-SC baseline** and reports **structural fidelity**.

A Textual TUI renders the lineage of generations, the ablation table, metrics vs baseline, and the diagnosis stream, live from an append-only event log.

## 3. Why this wins the track

The brief asks for: generate → run → analyze failures → improve prompts/tools/memory/**orchestration** → measurable gains in accuracy, reliability, **cost, speed** → across multiple domains. Occam is dead-centre: pruning a role *is* an orchestration mutation, and cost/speed are our headline axes, not afterthoughts.

Differentiation: every other Track 1 demo shows a diagram growing and a line going up. Occam shows a diagram **shrinking** while accuracy holds and cost drops — and can conclude "one agent was enough," which is a visible act of judgment.

## 4. Users

- **Primary (demo):** hackathon judges — AO team, Maximor engineers (they build with agent harnesses themselves), Rayed Chowdhury (22× hackathon winner). Technical, will spot fakery, will appreciate acknowledged limitations.
- **Secondary:** anyone building agent systems who wants to know which roles are earning their tokens.

## 5. Scope

### In scope (MVP — must ship)
- Task pack format + loaders for three domains: **SMFR** (financial multi-hop), **BFCL-simple** (tool calling), **MGSM-en** (math word problems).
- Architect → Execute → Ablate → Diagnose → Mutate loop with the closed mutation menu.
- Cost-matched CoT-SC baseline.
- Metrics: pass rate, $ cost, latency, tokens, per-role influence, structural fidelity, with confidence intervals.
- Event log + state snapshot per run; `occam run`, `occam replay`, `occam task new`.
- Textual TUI: lineage, ablation table, metrics strip, diagnosis feed.
- Fixture replay for no-network TUI development and demo backup.
- README meeting submission requirements.

### Stretch (only after MVP is demo-ready)
- Live task creation with LLM-drafted eval cases and human approval in the TUI.
- Second-order ablation (pairs) for small architectures.
- `textual serve` web view for large-font recording.
- Neatlogs tracing; TensorMux routing.

### Out of scope
- Open-ended code-generation search (ADAS-style). Mutations come from a closed menu.
- Population/evolutionary search. Single lineage with branching on revert only.
- Embedding AO in the product. AO is our *build* tool, not a runtime.
- Web dashboard. TUI only.
- Any domain requiring Docker sandboxes (SWE-bench etc.).

## 6. Success criteria

| # | Criterion | How verified |
|---|---|---|
| S1 | On SMFR, Occam produces ≥1 generation where a role is pruned with pass-rate change within the noise band and cost reduced ≥20% | `occam run --task smfr` output + events |
| S2 | On each of 3 domains, the final generation beats generation 0 on cost at equal-or-better pass rate | metrics in `state.json` |
| S3 | Ablation table shows influence with confidence intervals, and at least one witness is flagged in the demo run | TUI + events |
| S4 | Structural fidelity reported per generation and trends upward across the run | metrics |
| S5 | Cost-matched CoT-SC baseline reported alongside on every generation | metrics |
| S6 | A revert event occurs at least once in the demo run (mutation tried, ablation says witness, reverted) | events |
| S7 | `occam replay fixtures/demo_run` renders the full TUI with no network in <2s | manual + test |
| S8 | Legibility: TUI screen-recorded at 1080p is readable at 100% zoom | recorded test at hour 12 |
| S9 | README: what it does, how to run, track, workflow, what improved across iterations, demo link, AO usage | checklist |

## 7. Demo (3 minutes) — what the product must make possible

| Beat | Product capability required |
|---|---|
| Cold open: ablation table with two red WITNESS rows | Replay of a completed run to a chosen generation |
| Live task from judges | `occam task new --goal "..."` (stretch) **or** a pre-built pack chosen live from a menu |
| Architect proposes 5 roles; diagram draws | `architecture.proposed` event → lineage node + role list |
| First run fails 2 of N in red | per-case results in `run.completed` event → case grid |
| Ablation runs; two roles → influence ≈ 0 | `ablation.role` events streaming into the table |
| Prune; diagram shrinks; pass holds; cost −40% | `mutation.applied{type:prune}` then `run.completed` with lower cost |
| Try split → ablation says witness → **revert** | `mutation.applied{type:split}`, `ablation.*`, `mutation.reverted` |
| Second domain, pre-recorded | second run directory, replayed |
| Close: final vs CoT-SC | `baseline.completed` + metrics strip |

## 8. Honesty requirements (these go in the README and the pitch)

- Leave-one-out influence is a first-order approximation of Shapley value. We say so.
- Ablation is noisy at small N; we report confidence intervals and a natural-variance band, and only flag a witness when the CI excludes meaningful influence.
- The demo eval set is a fixed slice; the live task (if shown) uses LLM-drafted cases with human approval.
- The comparator is cost-matched CoT-SC because that is what the field's most recent audit used; the brief defines no baseline.

## 9. Prior art we must be able to speak to

ADAS (2408.08435), DGM (2505.22954), AgentSquare, AFlow, GPTSwarm, DyLAN, MaAS, MAS-Zero, Meta-Harness (2603.28052), Self-Harness (2606.09498), Co-Coder (2606.00953), OpenEvolve, GEPA/DSPy. **Our delta:** they optimise *for a score*; we optimise *against redundancy*, using causal ablation as the signal inside the loop. Full verified citation list: `prd/07-CITATIONS.md`. Citations appear in the README and pitch only — **never in the TUI**.
