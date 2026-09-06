# 03 — Ablation and Metrics

This is the core of the product. Get this right and everything else is presentation.

## 1. What ablation is, concretely

An architecture is a DAG of roles. **Ablating role r** = run the same eval cases with r removed: its output slot is filled with a sentinel `[no input from <r.name>]`, consumers see that marker, everything downstream recomputes, everything upstream is a cache hit.

For each case *c* we now have two final answers: `a_full(c)` and `a_{-r}(c)`. Comparing them **per case, paired**, is the whole idea.

## 2. The three per-role numbers

For role *r*, over the ablation case set *C* (|C| = n):

```
divergence(r)  = |{c : a_{-r}(c) ≠ a_full(c)}| / n
                 "how often does removing r change the answer at all"

influence(r)   = pass_rate_full − pass_rate_{-r}
               = ( |{c: full passed, −r failed}| − |{c: full failed, −r passed}| ) / n
                 "how much correctness do we lose when r is gone" (paired difference; can be negative)

cost_share(r)  = Σ_c cost(r, c) / Σ_c cost(all roles, c)     from the FULL run's per-role traces
```

**Why divergence and influence are both needed.** Aggregate pass rate can stay flat while individual answers flip both ways. Divergence catches "this role changes outputs" even when the net effect on correctness is zero — that's *churn*, not value, and it's a different diagnosis (unstable role) from *witness* (inert role). Influence alone would call both "zero."

## 3. Why ablation is slow, with numbers, and what we do about it

Let R = roles, M = eval cases, A = ablation cases (A ≤ M), K = mean LLM calls per case per role (tool loops make K > 1), G = generations.

| quantity | calls |
|---|---|
| Full run per generation | M × R × K |
| Naive ablation per generation | R × A × R × K (each knockout re-runs all R−1 surviving roles on A cases) |
| Total naive, G generations | G × [ M·R·K + R²·A·K ] |

Plug in R=5, M=20, A=10, K=2, G=6: full = 200/gen, naive ablation = **500/gen**, total ≈ **4,200 LLM calls** — plus, in g0 before the range-endpoint lesson, ~17 HTTP tool calls per case. On a gateway with a modest per-minute cap that's **hours** if serialised. With the TensorMux grant the token *budget* isn't the constraint; **wall-clock and per-minute limits are**. That's why "ablation is slow" is a real risk, not a vibe.

**Mitigations, in the order the executor applies them:**

1. **Upstream cache hits.** Knocking out r only invalidates r and its *descendants*. In a 5-role pipeline where r is role 4, roles 1–3 are byte-identical cache hits. Real recomputation per knockout averages ~R/2 roles, halving ablation cost. Requires the content-addressed cache in `01 §5` — this is why it's mandatory, not nice-to-have.
2. **Ablation subset.** A = 10 of M = 20, stratified by `meta` (holiday / fee / cross-currency / plain). Documented; the TUI shows "ablated on 10/20."
3. **Concurrency within rate limits.** Independent (case, knockout) pairs run concurrently under the per-model token bucket. `worker_fast` (TensorMux) is the main lane; `worker_alt` (GPT-5 nano) is a second lane for knockouts. **Tool calls are disk-cached** (`02 §2`), so ablation never touches Frankfurter after the first full run.
4. **Skip unchanged roles.** After a `rewrite_prompt` on role 2, roles whose prompt *and* upstream inputs are unchanged don't need re-ablation — their knockout result from the previous generation is still valid. Only re-ablate roles whose config or ancestry changed. Big win in later generations.
5. **Precompute the demo run.** The live demo is a `replay`. Ablation running for 8 minutes is fine in development; it never happens on camera.

Revised estimate with 1–4: **~1,200–1,600 LLM calls per 5–6-generation run.** At ~60–120 RPM across two lanes that's **~15–30 minutes per run**, all offline for tool calls after g0. Two runs plus pass³ fit comfortably in a working session. Acceptable — and the demo replays a recorded run regardless.

## 4. Why ablation is noisy, with numbers, and what we do about it

Hosted models are not deterministic even at temperature 0. Run the *same* architecture twice on the same 10 cases and you might get 7/10 then 8/10. So if `full` = 8/10 and `−r` = 7/10, **is that r's influence or the dice?** With n=10, one case = 10 percentage points. A naive threshold would flag noise as signal and signal as noise, and a judge who knows this will ask.

**What we do:**

1. **Measure the noise floor first.** Before ablating, re-run the *full* architecture once more on the ablation subset (`variant="full_repeat"`). `noise(c)` = 1 if the two full runs disagree on case c. `noise_rate` = mean. This costs one extra pass and gives us an empirical variance band per generation. Report it.
2. **Paired statistics, not aggregate.** Because the same cases are used, influence is a *paired* difference. Compute a **95% CI via bootstrap over cases** (resample the n paired outcomes 2,000 times — microseconds). Emit `influence_ci{lo,hi}`.
3. **Verdict rule (code, not prompt):**
   ```
   if divergence(r) <= noise_rate + eps        → "witness"        (removing it changes nothing beyond noise)
   elif influence_ci.lo > 0                    → "load_bearing"   (removing it clearly hurts)
   elif influence_ci.hi < 0                    → "harmful"        (removing it clearly helps)
   else                                        → "uncertain"      (changes answers, but effect on correctness not resolved at this n)
   ```
   `eps` = 0.05 default. `uncertain` is a legitimate verdict shown in the TUI as amber. **We never prune `uncertain`.** We prune `witness` and `harmful`.
4. **Escalate n only where it matters.** If a role is `uncertain` and the diagnosis wants to act on it, re-run its knockout on the *remaining* M−A cases (so n→M) before deciding. Spend tokens where the decision is close, not everywhere.
5. **Determinism hygiene.** temperature=0 for all roles in full/ablation runs, `seed` where the provider supports it, sorted tool results, normalised answers. Baseline CoT-SC uses temperature 0.7 by design (it's sampling) — that's separate.
6. **Say it on camera.** "Two roles are witnesses — removing them changed the answer on 0 of 10 cases, and our noise floor this generation was 1 of 10." One sentence, and the noise objection is pre-empted.

**Property tests (`tests/test_ablation_props.py`):**
- A role whose output is never referenced by any downstream prompt must get divergence = 0 → `witness`.
- A role that is the sole producer of the final answer must get divergence = 1 and influence = pass_rate_full → `load_bearing`.
- Adding pure-noise flips to a synthetic result set must widen the CI, not shift the point estimate's sign systematically.
- `reduce(events)` is deterministic: same events → identical `state.json`.

## 5. Structural fidelity

```
SF(A) = Σ_r max(influence(r), 0) · 1[verdict(r) ∈ {load_bearing}]
        ───────────────────────────────────────────────────────────    ∈ [0, 1] after normalisation below
        Σ_r cost_share(r)                       (= 1)
```
Simpler, and what we actually display: **the fraction of total spend that goes to load-bearing roles.**
```
SF(A) = Σ_{r load_bearing} cost_share(r)
```
A single-role architecture that works has SF = 1.0. A 5-role system where two witnesses eat 40% of tokens has SF = 0.6. This is deliberately the simplest defensible number; the paper defines the *concept* and leaves the operationalisation open, so we say "we operationalise structural fidelity as load-bearing spend share" and move on.

Shown per generation in the metrics strip. Expected to rise across the run as witnesses are pruned.

## 6. Cost-matched CoT-SC baseline

- Single role, `worker_fast`, prompt: "Think step by step, then give the final answer in the required format."
- Sample k times at temperature 0.7; majority vote on normalised answer (ties → first).
- `k = clamp(floor(cost_full_gen / cost_single_sample), 1, 9)` — spend the same dollars as the current generation.
- Report `pass_rate`, `cost_usd`, `latency_s_mean`, `k`, `matched_to_cost_usd`.
- Metrics strip shows `Δpass = ours − baseline` and `cost_ratio = ours / baseline`. The honest target on the FX task: **Δpass clearly > 0** (a single CoT pass mishandles holidays/fees and can't parallelise 16 lookups well) with **cost_ratio trending to ≤ 1** after the witness prune and the range-endpoint lesson.

## 7. Other metrics (all in `metrics.snapshot`)

| metric | definition |
|---|---|
| `pass_rate` | passed / M, with Wilson 95% CI |
| `cost_usd` | Σ per-role cost (cache hits count $0 and are flagged) |
| `latency_s_mean` | wall-clock per case, mean; also p50/p90 |
| `tokens` | in + out |
| `reliability` | 1 − noise_rate (agreement between the two full runs on the ablation subset) — this is our operationalisation of the brief's "reliability" axis, and it's honest: it measures whether the system gives the same answer twice |
| `speed` | 1 / latency_s_mean, shown as cases/min |

## 8. What the ablation table looks like (feeds `04-TUI-SPEC.md`)

```
 role                    justification      influence   95% CI          divergence  cost   verdict
 ─────────────────────────────────────────────────────────────────────────────────────────────────
 Ledger Parser           context_isolation   +0.40     [+0.20, +0.60]     0.90      14%   LOAD-BEARING
 Rate Fetcher            parallel            +0.50     [+0.30, +0.70]     1.00      54%   LOAD-BEARING
 FX Calculator           control             +0.50     [+0.30, +0.70]     1.00      19%   LOAD-BEARING
 Verifier                verification        +0.00     [−0.10, +0.10]     0.10       6%   WITNESS
 Reporter                control             +0.10     [−0.10, +0.30]     0.30       8%   UNCERTAIN
 ─────────────────────────────────────────────────────────────────────────────────────────────────
 ablated on 10/20 cases · noise floor 0.10 · structural fidelity 0.87 · 1 witness
```

The correlation you want the audience to notice without being told: **the `verification` row is the red one.** That is the Illusion paper's finding, reproduced live on a real task. (Numbers here match `08 §3`.)

## 9. Lesson ablation (stretch — WP-14; schema already supports it)

At run 2, each loaded lesson is a component too. Knock out lesson *l* = run g0 of run 2 with *l* removed from the architect's input (fresh architect call, then execute on the ablation subset). Same divergence/influence/CI/verdict machinery as roles. Reported in a second table under the roles table: `lesson · kind · influence · CI · verdict`. Expected: L1 (holiday resolution) and D1 (bank fee) LOAD-BEARING; anything else UNCERTAIN or WITNESS → `status: retired`. Cost: one architect call + one ablation-subset run per lesson (~3–5 lessons). Build only if run 2 is working end to end by +16h.

## 10. pass³ reliability
Final generation only: 3 independent runs per case at temperature 0. `reliability_pass3 = |{c : all 3 pass}| / N`. Emitted as `reliability.completed`; shown in the metrics strip as `rel³`. Noise floor from §4 stays as the *internal* signal for ablation verdicts; pass³ is the *reported* reliability. They will disagree slightly; that's expected — say so if asked.
