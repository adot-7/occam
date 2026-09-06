# 00 — Product Requirements: Occam (v3, final)

> Working name **Occam** (the razor). CLI `occam`.

## 1. Problem

Automated agent-design systems produce architectures with more agents than the task needs. *The Illusion of Multi-Agent Advantage* (arXiv:2606.13003, June 2026) audited six such frameworks: roles are redundant, agents reach unanimous consensus in >90% of GPT-5 cases, later agents are "expensive witnesses that incur full inference costs while exerting near-zero causal influence." Their call: evaluate architectures on **structural fidelity** — whether each role exerts measurable causal influence on the final decision. Separately, agents given real third-party tools fail on those tools' quirks and re-fail every time a new agent is built for the same tool.

## 2. What Occam does

Given a **goal, real tools, and an eval set**, Occam:
1. **Architects** a multi-agent system (roles, prompts, tool bindings, DAG).
2. **Executes** it on the eval set with **live HTTP calls to a third-party API**, recording per-case pass/fail, per-role tokens/cost, latency, tool calls.
3. **Ablates** every role — re-runs with it knocked out — and prunes roles with no causal influence.
4. **Diagnoses** failures in plain language, picks one mutation (prune / rewrite prompt / rebind tools / split / merge / collapse), and **writes lessons**: typed, evidence-backed facts about the tool's behaviour and the domain's conventions.
5. Loops to plateau. **On the next run — a different ledger — the architect reads the lessons first**, so the new team starts with what the last one learned.

A Textual TUI renders lineage, the ablation table, cases, metrics vs a cost-matched CoT-SC baseline, the lessons file growing, and (in run 2) the run-1-vs-run-2 comparison.

## 3. The workflow (see `02`)
**Month-end FX revaluation** of a multi-currency receivables ledger for an Indian exporter, reporting in INR, using the ECB rates API (Frankfurter, live HTTP). Every case is a ledger of 6–10 invoices → one total gain/loss + per-invoice breakdown. Real accountant work; parallel across invoices; deterministic ground truth; a tool with real, verified quirks (holiday date resolution, a range endpoint) and a company convention (bank-fee-net settlements) that must be learned.

## 4. Why this wins Track 1

The brief: generate an architecture, run it, analyze failures, iteratively improve **prompts, tools, memory, orchestration**, show gains in accuracy, reliability, cost, speed. The clarification: an agent with third-party app access that learns how to use tools over time, learns contextual logic from tool data and applies it in later runs, balances cost and speed. Occam answers both literally:

| judges ask | Occam shows |
|---|---|
| gets better over time? | run 1 g0 → g_final within a run; **run 1 g0 → run 2 g0 across runs on unseen data** |
| self-reflection and memory growing? | diagnosis → `lessons.jsonl` growing, each with evidence; architect reads it |
| learns contextual logic from third-party data, applies later? | holiday-date resolution and the range endpoint, discovered by failing, applied in run 2 |
| cost/speed balance? | ablation prunes witnesses; range endpoint halves calls; cost-matched CoT-SC baseline on every generation |

Differentiation: everyone else's diagram grows. Ours **shrinks** while accuracy holds — and the second run starts smaller and smarter than the first.

## 5. Scope

**MVP (must ship):** two fx_recon packs + reference implementation + cached tool · architect/execute/ablate/diagnose/mutate loop · lessons write/read with leak guard · cost-matched CoT-SC baseline · pass³ reliability on final gen · `occam run`, `occam replay`, `occam compare` · TUI: lineage, ablation table, cases, metrics, lessons pane, diagnosis feed · fixture replay · README.
**Stretch:** lesson ablation at run 2 (which lessons causally mattered) · TUI compare tab · Neatlogs shareable traces · `textual serve` recording rig polish.
**Out:** more domains · open-ended code mutation · population search · embedding AO · web UI · auth flows.

## 6. Success criteria

| # | criterion | evidence |
|---|---|---|
| S1 | Run 1 g0 fails ≥4 of the holiday/fee cases; a later generation passes them after a `rewrite_prompt` | `runs/run1` events |
| S2 | ≥1 role pruned as WITNESS in run 1 with pass rate within noise band and cost −20% or better | `ablation.*`, `metrics.snapshot` |
| S3 | ≥2 lessons written in run 1 (one tool_note, one domain_rule), each with evidence pointers, none containing case-specific values | `lesson.written` |
| S4 | Run 2 g0 pass rate ≥ run 1 g0 + 0.25, with fewer tool calls per case and fewer generations to plateau | `occam compare run1 run2` |
| S5 | Cost-matched CoT-SC reported on every generation of both runs | `baseline.completed` |
| S6 | pass³ reliability reported on final gen of both runs | `metrics.snapshot.reliability_pass3` |
| S7 | `occam replay fixtures/demo_run1` and `demo_run2` render with no network in <2s | test |
| S8 | TUI legible at 1080p/100% | recorded check at +10h |
| S9 | README covers what/how/track/workflow/what-improved/demo/AO usage + honesty notes | checklist |

## 7. Honesty (README + pitch)
- LOO influence ≈ first-order Shapley; CIs and noise floor reported; `uncertain` never pruned.
- Ground truth from a reference implementation using the same API and the formula in `02 §1.2`.
- Lessons are rules about tool/domain behaviour; a code guard rejects case-specific values.
- Held-out pack is a different seed; both `cases.jsonl` are committed.
- Costs on granted tokens shown at list-rate equivalent.
- The demo is a replay of recorded run logs; reproducible with `occam replay`.

## 8. Prior art → `07-CITATIONS.md`. Never in the TUI.
