# 08 — End-to-End Walkthrough, With the Numbers Worked

One SMFR run, traced from `occam run` to the ablation table, with every calculation shown. If a number in the TUI can't be derived by the method here, the implementation is wrong. Figures are illustrative but internally consistent.

## 0. Setup

```
occam run --task smfr_2inv --max-gens 6 --cases 20 --ablate-cases 10 --tui
```
- Loads `tasks/smfr_2inv/` → 20 `Case`s. Each `input` ≈ 3,500 tokens of price tables + transactions + question. `expected` e.g. `["Rachel"]`.
- Ablation subset: 10 cases, stratified (5 `reverse_target_sell` / 5 `_buy`, mixed earliest/latest). Fixed by seed → same 10 every generation.
- Emits `run.started`. Creates `runs/2026…_smfr_2inv/`.

## 1. Architect → g0

One call to `architect` (Sonnet 5) with: goal, tool manifest (`python_exec`, `lookup_price`, `list_transactions`), answer format, 3 example cases. Response is JSON constrained to the `Architecture` schema. It proposes:

| id | name | justification | tools | inputs | model |
|---|---|---|---|---|---|
| r_extract | Transaction Extractor | context_isolation | list_transactions | task | worker_fast |
| r_pnl | P&L Calculator | parallel | lookup_price, python_exec | r_extract | worker_fast |
| r_critic | Critic | verification | — | r_pnl | worker_fast |
| r_second | Second Opinion | ensemble | lookup_price, python_exec | r_extract | worker_fast |
| r_synth | Synthesizer | control | — | r_pnl, r_critic, r_second | worker_fast |

`final_role = r_synth`, `control = deterministic`. Emits `architecture.proposed{generation:0}`. TUI: lineage gets node `g0 · 5 roles`; architecture panel draws `task → Extract → {P&L, Second} → Critic → Synth`.

## 2. Execute g0 — one case in detail (`smfr2_0007`)

Topological order: r_extract → (r_pnl ‖ r_second) → r_critic → r_synth. The two middle roles run concurrently.

| role | what it does | calls | tok in | tok out | latency |
|---|---|---|---|---|---|
| r_extract | calls `list_transactions("Rachel")`, `("Patricia")`; returns structured JSON of holdings | 3 | 4,100 | 420 | 3.1s |
| r_pnl | for each investor calls `lookup_price` ×~20, then `python_exec` to compute P&L per date, returns per-investor valid dates + optimal date | 6 | 5,900 | 900 | 6.4s |
| r_second | independently does the same thing, differently phrased | 5 | 5,600 | 850 | 5.9s |
| r_critic | reads r_pnl output, writes "looks consistent" | 1 | 1,300 | 90 | 1.2s |
| r_synth | reads all three, emits `["Rachel"]` | 1 | 2,600 | 40 | 1.4s |

Cost per role = `tok_in × in_rate/1e6 + tok_out × out_rate/1e6`. With a list-rate-equivalent of $0.10/$0.40 per M (GLM-4.7-Flash placeholder — fill real):
```
r_extract: 4100×0.10e-6 + 420×0.40e-6  = $0.000578
r_pnl:     5900×0.10e-6 + 900×0.40e-6  = $0.000950
r_second:  5600×0.10e-6 + 850×0.40e-6  = $0.000900
r_critic:  1300×0.10e-6 +  90×0.40e-6  = $0.000166
r_synth:   2600×0.10e-6 +  40×0.40e-6  = $0.000276
case total                              = $0.002870
```
Latency for the case = critical path = 3.1 + max(6.4, 5.9) + 1.2 + 1.4 = **12.1s** (not the sum — parallel roles overlap).

Checker `json_set_equal(answer='["Rachel"]', expected=["Rachel"])` → **pass**. Emits `execution.case`.

After all 20 cases: `pass_rate = 13/20 = 0.65`, `cost = $0.0574`, `latency_mean = 11.8s`. Wilson 95% CI for 13/20 ≈ [0.43, 0.82]. Emits `execution.completed{variant:"full"}`. TUI: cases grid shows ✓✓✗✓…, metrics strip gets its first point.

Per-role cost shares over the 20 cases (sum of each role's cost / total):
```
r_extract 20%   r_pnl 33%   r_second 31%   r_critic 6%   r_synth 10%
```

## 3. Noise floor

Re-run `full` on the 10 ablation cases → `variant:"full_repeat"`. Compare answers case by case with the first run: 9 agree, 1 differs (case 0012 flipped `["Patricia"]` → `[]`). **noise_rate = 0.10.** Also gives `reliability = 0.90`. Cost: another ~$0.029 (10 cases). All upstream roles hit cache? No — this is a genuine re-run, temperature 0, but hosted models drift; that's the point.

## 4. Ablate g0 — knock out r_critic

Run the 10 cases with r_critic removed. r_synth's prompt template renders `[no input from Critic]` in that slot.

- r_extract, r_pnl, r_second: **cache hits** (identical inputs) → $0, 0 added latency, `cached=true`.
- r_synth: recomputes (its input changed).

Paired comparison over 10 cases:

| case | full answer | −critic answer | full pass | −critic pass |
|---|---|---|---|---|
| 0003 | ["Rachel"] | ["Rachel"] | ✓ | ✓ |
| 0007 | ["Rachel"] | ["Rachel"] | ✓ | ✓ |
| 0012 | ["Patricia"] | [] | ✗ | ✗ |
| … 7 more identical … |

```
divergence(r_critic) = 1/10 = 0.10      (only case 0012 changed — and it was already wrong)
influence(r_critic)  = (0 cases ✓→✗  −  0 cases ✗→✓) / 10 = 0.00
bootstrap CI (2000 resamples of the 10 paired outcomes) = [−0.10, +0.10]
cost_share(r_critic) = 0.06
```
Verdict rule: `divergence 0.10 ≤ noise_rate 0.10 + eps 0.05` → **WITNESS**. Emits `ablation.role{role_id:"r_critic", verdict:"witness"}`. TUI row turns red.

## 5. Ablate g0 — knock out r_second

r_extract cached. r_pnl cached. r_critic recomputes? No — r_critic reads only r_pnl, unchanged → cached. r_synth recomputes.

Paired: 9/10 answers identical; case 0012 changes again (this case is just unstable).
```
divergence = 0.10 → ≤ noise floor → WITNESS.   influence 0.00, CI [−0.10,+0.10].   cost_share 0.31
```
**This is the money row**: a role eating 31% of spend that changed nothing beyond the noise floor. The "ensemble" tag next to a red verdict is the Illusion paper's finding on screen.

## 6. Ablate g0 — knock out r_pnl

r_synth now sees `[no input from P&L Calculator]` and only has r_second's numbers and r_critic's (now empty-input) comment. Answers change on 8/10 cases; 4 cases that passed now fail, 0 flip the other way.
```
divergence = 0.80
influence  = (4 − 0)/10 = +0.40, CI [+0.20, +0.60]  → lo > 0 → LOAD-BEARING
```
Similarly r_extract → influence +0.30, CI [+0.10, +0.50] → LOAD-BEARING. r_synth → divergence 0.60, influence +0.20, CI [0.00, +0.40] → **UNCERTAIN** (lo == 0). Amber; never pruned.

## 7. Structural fidelity of g0

```
SF(g0) = Σ cost_share over LOAD-BEARING roles = r_extract 0.20 + r_pnl 0.33 = 0.53
```
Emits `ablation.completed{structural_fidelity:0.53, witnesses:["r_critic","r_second"]}`. Footer: `ablated 10/20 · noise 0.10 · SF 0.53 · 2 witnesses`.

## 8. Baseline for g0 — cost-matched CoT-SC

Single role, worker_fast, CoT prompt, temperature 0.7. One sample on a 3.5k-token input ≈ 3,600 in / 400 out ≈ $0.00052. Generation cost per case = $0.00287.
```
k = clamp(floor(0.00287 / 0.00052), 1, 9) = floor(5.5) = 5
```
Run 5 samples per case on all 20 cases, majority vote → 11/20 = 0.55, cost $0.052, latency 4.0s (samples run in parallel). Emits `baseline.completed{k:5, pass_rate:0.55, cost_usd:0.052, matched_to_cost_usd:0.0574}`.

Metrics strip: `vs CoT-SC(k=5) · Δpass +0.10 · cost ×1.10`. We're slightly more accurate and slightly more expensive at g0. Fine — that's the starting point.

## 9. Diagnose → mutate → g1

`diagnose` gets: 7 failed cases with per-role outputs, the ablation table, history. **Hard rule fires before the LLM is even asked:** witnesses exist → mutation must be `prune` of the highest-cost witness. The LLM writes the rationale:

> "Second Opinion (31% of spend) and Critic (6%) each changed answers on 1/10 cases, at the noise floor. Second Opinion duplicates P&L Calculator's work; Critic's output is never acted on. Pruning both. Expected: pass rate unchanged (±0.10), cost −35–40%."

Both are pruned in one mutation (rule: prune *all* witnesses whose combined removal was individually justified — cheaper than two generations). `mutate.prune` removes the roles and rewires r_synth's inputs to `[r_pnl]`. Emits `mutation.applied{type:"prune", target_role:["r_second","r_critic"], diff:"−Second Opinion, −Critic; Synthesizer.inputs: [P&L Calculator]"}`. TUI: architecture panel strikes through two boxes for 2s, then they vanish. Lineage: `g1 · 3 roles`.

## 10. Execute g1

Cache does a lot here: r_extract and r_pnl outputs for all 20 cases are byte-identical → cache hits. Only r_synth (changed inputs) recomputes. 20 calls total.
```
pass_rate 13/20 = 0.65 (unchanged)   cost $0.0353 (−39%)   latency_mean 9.9s (−16%)
```
Baseline re-matched: k = floor(0.001765/0.00052) = 3 → 0.50, $0.031. Now `Δpass +0.15, cost ×1.14`.

Ablation g1 (3 roles): r_synth now LOAD-BEARING (influence +0.30 — without a control role nothing aggregates). **SF(g1) = 0.20 + 0.50 + 0.30 = 1.00.** Everything left is load-bearing.

## 11. g2 — try a split, verify, revert

No witnesses → hard rule doesn't fire → LLM picks from the menu. Failures show r_pnl's `python_exec` sometimes mixing up the two investors' holdings. It proposes `split r_pnl → r_pnl_A, r_pnl_B` (one per investor), justification `parallel`.

Execute g2': pass 14/20, cost +8% (two smaller roles ≈ one bigger one, plus overhead). Ablate the two new roles: knock out r_pnl_A → divergence 0.5, influence +0.20 CI [0.0, +0.40] → UNCERTAIN. Knock out r_pnl_B → same. Neither is a witness, so **no revert**; but pass improved by 1 case — inside noise. Plateau rule: `(pass, −cost)` improved on pass → keep, continue.

*(In the fixture we make the split fail instead, to show the revert beat: the new roles come back WITNESS because the architect wired them so r_synth still reads the old combined key. Emits `mutation.reverted{restored_to:"g1", reason:"new roles r_pnl_A, r_pnl_B: divergence 0.0 — not wired into final answer"}`. Lineage shows `○ g2' split → reverted`.)*

## 12. g3 — rewrite prompt, plateau, done

Diagnosis targets the remaining failures (investor-holding confusion) with `rewrite_prompt` on r_pnl adding an explicit "process one investor at a time; label every intermediate by investor name." Execute: 16/20 = 0.80, cost $0.037. Ablation: all three LOAD-BEARING, SF 1.00. Baseline k=3: 0.50. `Δpass +0.30, cost ×1.19`.

g4 tries `set_memory scratchpad` on r_extract: 16/20, cost +3% → no improvement in `(pass, −cost)`. g5 tries `rebind_tools` (remove `lookup_price` from r_pnl, rely on `python_exec` over the parsed table): 16/20, cost −4% → improvement on cost. g6: no improvement → **plateau after 2 non-improving generations** → `run.completed{best_generation:"g5"}`.

## 13. The summary the README table shows

| gen | roles | pass | cost (list-equiv) | latency | SF | vs CoT-SC Δpass | cost ratio | mutation |
|---|---|---|---|---|---|---|---|---|
| g0 | 5 | 0.65 | $0.0574 | 11.8s | 0.53 | +0.10 | 1.10 | — |
| g1 | 3 | 0.65 | $0.0353 | 9.9s | 1.00 | +0.15 | 1.14 | prune ×2 |
| g2' | 4 | — | — | — | — | — | — | split → **reverted** |
| g3 | 3 | 0.80 | $0.0370 | 10.1s | 1.00 | +0.30 | 1.19 | rewrite_prompt |
| g5 | 3 | 0.80 | $0.0355 | 9.6s | 1.00 | +0.30 | 1.15 | rebind_tools |

Read across: **accuracy +15pp, cost −38%, latency −19%, structural fidelity 0.53 → 1.00, two agents deleted, one attempted addition reverted.** Four of the brief's axes, one table, all derived from the events.

## 14. How every number reaches the TUI

```
engine emits event → store.writer appends line to events.jsonl (fsync)
                  → store.reducer folds it into State → state.json rewritten
TUI: EventReader.tail() yields the same line → reducer (same code) → StateChanged → widgets re-render
```
The reducer is the only shared code and it is pure. `occam validate <run>` replays events through it and diffs against `state.json`; any divergence is a bug.

## 15. Where the numbers can lie, and the guard for each

| risk | guard |
|---|---|
| Cache hits make cost look lower than a cold run | `cached` flag recorded; README reports both "run cost" and "cold-equivalent cost" |
| One unstable case dominates influence at n=10 | noise floor + CI + `uncertain` verdict; escalate n before acting |
| Prune looks free because pass didn't move — but it's inside noise either way | say so: "pass unchanged within ±0.10 noise; cost −39% is outside any noise" |
| Baseline k rounds down, flattering us | report `matched_to_cost_usd` next to actual baseline cost; ratio shown |
| Granted tokens → "$0" | list-rate-equivalent, labelled |
