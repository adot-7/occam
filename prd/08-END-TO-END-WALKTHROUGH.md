# 08 — End-to-End Walkthrough (v3), With the Numbers Worked

Two runs on the FX revaluation task. Figures are illustrative but internally consistent; every one is derivable from the events by the methods in `03`.

## 0. Run 1 setup

```
occam lessons reset --memory memory/fx_recon
occam run --task fx_recon_a --run-name run1 --memory memory/fx_recon --max-gens 6 --cases 20 --ablate-cases 10 --pass3
```
20 cases, 6–10 invoices each. `lessons_loaded: []`.

## 1. Architect → g0 (no lessons)

| id | name | justification | tools | inputs |
|---|---|---|---|---|
| r_parse | Ledger Parser | context_isolation | — | task |
| r_rates | Rate Fetcher | parallel | fx_rate, fx_series | r_parse |
| r_calc | FX Calculator | control | python_exec | r_parse, r_rates |
| r_verify | Verifier | verification | — | r_calc |
| r_report | Reporter | control | — | r_calc, r_verify |

Tool description the architect saw for `fx_rate`: *"Get the exchange rate between two currencies on a date."* Nothing about holidays.

## 2. Execute g0 — case `fxa_007` (8 invoices, valuation 2026-04-30)

| role | behaviour | calls | tok in | tok out | latency |
|---|---|---|---|---|---|
| r_parse | structures 8 invoices as JSON | 1 | 1,900 | 620 | 2.2s |
| r_rates | **16 single-date `fx_rate` calls** (issue + settle/valuation per invoice), one at a time | 17 | 9,400 | 1,900 | 14.8s |
| r_calc | `python_exec` over parsed ledger + rates | 2 | 3,100 | 700 | 4.1s |
| r_verify | "totals look consistent" | 1 | 1,600 | 80 | 1.1s |
| r_report | emits final JSON | 1 | 1,400 | 260 | 1.3s |

Cost at list-rate-equivalent $0.10/$0.40 per M: `r_parse $0.000438 · r_rates $0.001700 · r_calc $0.000590 · r_verify $0.000192 · r_report $0.000244 = $0.003164/case`. Critical path 2.2+14.8+4.1+1.1+1.3 = **23.5s**.

**Where it goes wrong.** INV-2291 settled 2026-04-04 (Saturday; Friday was Good Friday). `fx_rate("2026-04-04","USD","INR")` returns `{"rate_date":"2026-04-02","rate":85.91}`. The fetcher passes 85.91 on as "the April 4 rate" — correct number by luck? No: the *calculator* separately decided "weekend → use Friday" and asked the fetcher for `2026-04-03`, got back `rate_date 2026-04-02` again, and then *averaged* the two identical values with a stale cached Thursday figure it had labelled wrongly. Net: INV-2291 off by ₹1,340. INV-2310 has "net of bank fee INR 1,250": calculator treats the fee as FX loss → off by exactly ₹1,250. Total off by ₹2,590 > tolerance (max(5, 0.1%·41,872) = ₹41.87). **Fail.** `sub_results`: 6/8 invoices ✓, INV-2291 ✗, INV-2310 ✗.

After 20 cases: **pass 11/20 = 0.55**, cost $0.0633, latency 22.9s, **calls/case 17.4**. Failures: 6 holiday/weekend cases, 5 fee cases (some overlap) → 9 distinct failures.

## 3. Noise floor, ablation g0

`full_repeat` on 10 cases: 1 disagreement → noise 0.10.

| role | divergence | influence | CI | cost share | verdict |
|---|---|---|---|---|---|
| r_parse | 0.90 | +0.40 | [+.20,+.60] | 14% | LOAD-BEARING |
| r_rates | 1.00 | +0.50 | [+.30,+.70] | 54% | LOAD-BEARING |
| r_calc | 1.00 | +0.50 | [+.30,+.70] | 19% | LOAD-BEARING |
| **r_verify** | **0.10** | **0.00** | [−.10,+.10] | **6%** | **WITNESS** |
| r_report | 0.30 | +0.10 | [−.10,+.30] | 8% | UNCERTAIN |

SF(g0) = 0.14 + 0.54 + 0.19 = **0.87**. Baseline CoT-SC: single sample ≈ $0.0009 → k = floor(0.003164/0.0009) = 3 → 8/20 = 0.40, $0.054. Δpass +0.15, cost ×1.17.

## 4. Diagnose g0 → prune + lessons

Hard rule: witness exists → `prune r_verify`. Diagnosis text also reads the failing traces' raw tool responses and writes **two lessons** (both pass the leak guard — no dates, no ids, no ≥4-digit numbers):

```json
{"kind":"tool_note","tool":"fx_rate","text":"The response's rate_date is the actual ECB business day used and may be earlier than the requested date (weekends and ECB holidays). Always use rate_date and the returned rate as-is; never adjust the date yourself or average adjacent days.","evidence":{"run_id":"run1","generation":0,"case_ids":["fxa_003","fxa_007","fxa_012","fxa_015"]}}
{"kind":"domain_rule","text":"When a settlement is marked net of a bank fee in the reporting currency, the fee is a bank charge, not FX: add it back to the received value before computing the FX gain or loss.","evidence":{"run_id":"run1","generation":0,"case_ids":["fxa_002","fxa_007","fxa_011"]}}
```
Both → `lesson.written`. TUI lessons pane: 2 rows.

Mutation for g1 is `prune r_verify` (hard rule). The prompt fixes come next generation.

## 5. g1 → g2

g1 (4 roles): r_parse, r_rates cached for all cases; r_calc, r_report recompute. pass 0.55 (unchanged), cost $0.0595 (−6%), SF 1.00 after re-ablation (r_report becomes LOAD-BEARING without a verifier in between).

Diagnose g1: no witnesses → LLM picks `rewrite_prompt` on **r_calc and r_rates** (one mutation, two roles allowed when the diagnosis is identical for both): fetcher told to pass through `rate_date`, calculator told to trust it and to add back bank fees. g2: **pass 18/20 = 0.90**, cost $0.0601, calls/case 17.2. Two remaining failures are JPY-heavy batches at the tolerance edge (uncertain — see OPEN-QUESTIONS).

## 6. g3 — the efficiency lesson

Diagnose g2 sees 17 calls/case and the unused `fx_series` tool in the manifest. Mutation `rewrite_prompt r_rates`: "fetch each currency's rates for the whole date range in one `fx_series` call, then look up dates locally." g3: pass 0.90, **calls/case 17.2 → 5.1**, cost $0.0388 (−35%), latency 22.9s → 11.6s. Third lesson written:

```json
{"kind":"tool_note","tool":"fx_series","text":"For several dates in the same currency pair, one fx_series call over the date range is cheaper and faster than many fx_rate calls; look up individual dates from the returned map and treat missing days as non-business days.","evidence":{"run_id":"run1","generation":2,"case_ids":["fxa_001","fxa_004"]}}
```

g4, g5: no improvement → plateau. **pass³ on g3:** 17/20 cases pass 3/3 → `rel³ = 0.85`.

## 7. Run 1 summary

| gen | roles | pass | cost | calls/case | latency | SF | vs CoT-SC Δpass | mutation |
|---|---|---|---|---|---|---|---|---|
| g0 | 5 | 0.55 | $0.0633 | 17.4 | 22.9s | 0.87 | +0.15 | — |
| g1 | 4 | 0.55 | $0.0595 | 17.4 | 22.0s | 1.00 | +0.15 | prune Verifier |
| g2 | 4 | 0.90 | $0.0601 | 17.2 | 22.4s | 1.00 | +0.45 | rewrite Fetcher+Calculator |
| **g3** | 4 | **0.90** | **$0.0388** | **5.1** | **11.6s** | 1.00 | +0.50 | rewrite Fetcher (range endpoint) |

## 8. Run 2 — held-out ledger, lessons loaded

```
occam run --task fx_recon_b --run-name run2 --memory memory/fx_recon --max-gens 6 --cases 20 --ablate-cases 10 --pass3
```
`run.started.lessons_loaded` = 3. The architect sees `fx_rate`'s description **with L1 appended**, `fx_series` **with L2 appended**, and D1 under "Known conventions." It proposes **4 roles** (no verifier — the architect is also told the prior run's ablation history) with the fetcher already using `fx_series`.

**Run 2 g0: pass 17/20 = 0.85, cost $0.0371, calls/case 4.9, latency 11.1s.** Ablation: all four LOAD-BEARING, SF 1.00. Diagnose g0 → `rewrite_prompt r_calc` (a JPY rounding fix) → g1: 0.90. g2, g3 no improvement → plateau at g1. pass³ = 0.90.

## 9. `occam compare run1 run2`

```
                        run1        run2       Δ
g0 pass rate            0.55        0.85      +0.30
final pass rate         0.90        0.90       0.00
g0 calls/case           17.4        4.9      −72%
g0 cost/case            $0.0032     $0.0019  −41%
generations to plateau  3           1         −2
roles at g0 → final     5 → 4       4 → 4
pass³ (final)           0.85        0.90     +0.05
lessons loaded/written  0 / 3       3 / 0
```

That table is the answer to "does it get better over time." The second team started where the first one finished.

## 10. Where the numbers can lie, and the guard

| risk | guard |
|---|---|
| Lessons leak case answers | code guard (`01 §4.4`); lessons are printed in README |
| Run 2 is easier | same generator, different seed, same mix constraints; both `cases.jsonl` committed |
| Cache flatters cost | `cached` flag; README shows cold-equivalent |
| One unstable case at n=10 | noise floor, CI, `uncertain` never pruned |
| pass³ vs noise floor disagree | expected; both reported, both defined |
| Granted tokens → $0 | list-rate equivalent, labelled |
