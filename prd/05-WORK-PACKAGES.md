# 05 — Work Packages (v3, final)

Two humans (H1 engine, H2 data + TUI) plus AO workers. One WP = one AO session = one branch `wp/<nn>-<slug>` = one PR. Acceptance criteria are the definition of done; paste them into the PR with evidence.

**Clock:** ~23h from 2026-09-06 04:00 IST to submission. Hour budgets below sum to ~20h across two lanes. **WP-01 is shipped**; WP-01b is a small additive patch.

---

### WP-01b — Schema patch for v3  **[H1, ≤1h, first]**
Add `Lesson` model; events `lesson.written`, `reliability.completed`; fields `run.started.{run_name,memory_ns,lessons_loaded}`, `metrics.snapshot.{tool_calls_per_case,reliability_pass3}`, `CaseResult.sub_results`. Extend `fixtures/demo_run` → **two fixtures**: `fixtures/demo_run1` (5 gens, holiday failures at g0, witness prune, 2 lessons written, pass³) and `fixtures/demo_run2` (lessons loaded, g0 mostly passing, plateau at g1, compare.json). Hand-authored; realistic numbers from `08`.
- [ ] Both fixtures validate; reducer deterministic
- [ ] `occam replay fixtures/demo_run2` header shows `lessons loaded 3`

### WP-02 — LLM layer  **[H1, ≤1.5h]** (unchanged from v1)
`models.yaml` with `worker_fast` (TensorMux GLM-4.7-Flash), `worker_alt` (GPT-5 nano), `architect` (Sonnet 5). Cache, rate limiter, cost accounting incl. list-rate-equivalent.
- [ ] `occam llm ping worker_fast` works; native `tool_calls` verified or JSON-in-text fallback flag set

### WP-03 — FX packs + reference implementation + grader  **[H2, ≤3h]**
`occam/tools/fx.py` (client + disk cache, used by the reference impl too), `occam/tasks/fx_reference.py` (formula `02 §1.2`), `scripts/gen_fx_cases.py`, `tasks/fx_recon_a`, `tasks/fx_recon_b`, checker `fx_total` with `sub_results`. Commit `data/fx_cache/`.
- [ ] 3 hand-computed cases match the reference to the paisa
- [ ] Both packs: ≥6 holiday/weekend, ≥5 bank-fee, ≥5 cross-ccy, ≥4 JPY; printed by the generator
- [ ] Tolerance sanity report (min/max |expected|, tightest relative tolerance) printed and reviewed
- [ ] Second run of the generator is byte-identical (seeded, cached)

### WP-04 — Tool registry  **[H2, ≤1.5h, after WP-03]**
`fx_rate`, `fx_series` (plain base descriptions!), `python_exec`, `fan_out`. Per-call accounting: latency, bytes, status, cached. `tool_note` append hook.
- [ ] `fx_rate("2026-04-04","EUR","USD")` → `rate_date == "2026-04-02"`, `cached` true on second call
- [ ] `python_exec` sandbox tests (no network, timeout)

### WP-05 — Executor  **[H1, ≤3h, after WP-02; needs WP-04 for integration test]**
DAG executor per `01 §4.2`; sentinel knockouts; per-role traces incl. **raw tool responses**; `sub_results` from grader; emits `execution.*`.
- [ ] 3-role hand-written arch on 5 fx cases end to end with real HTTP on first run, cache on second
- [ ] Knockout of role 2: role 1 cached, role 3 recomputed

### WP-06 — Architect + Mutations + Lessons I/O  **[H1, ≤2.5h, after WP-05]**
Architect reads `memory/<ns>/lessons.jsonl`, appends `tool_note`s to tool descriptions, passes `domain_rule`s. Mutation menu. `occam lessons show|reset`.
- [ ] With 2 lessons present, the proposed g0 fetcher prompt/tool description contains them; with 0, it doesn't
- [ ] Each mutation type unit-tested

### WP-07 — TUI shell + replay  **[H2, ≤3h, parallel]** (unchanged; drive from fixtures)
- [ ] Replays both fixtures; `--to-gen`, `--at`, `--pause` work

### WP-08 — Ablation + metrics  **[H1, ≤2h, after WP-05]** (unchanged math)
- [ ] Property tests pass; inert role → WITNESS; SF computed

### WP-09 — TUI panels  **[H2, ≤3.5h, after WP-07]**
Ablation table, architecture DAG w/ prune strikethrough, cases grid with sub-results, Inspector with **tool-response highlight**, Lessons pane, compare strip, metrics incl. `calls/case`, `rel³`.
- [ ] Replay `demo_run1` to g0 ablation: verdicts render; Inspector on a holiday case highlights `requested_date → rate_date`
- [ ] Replay `demo_run2`: lessons pane shows 3 loaded; compare strip populated

### WP-10 — Diagnose (with lesson writer + leak guard) + Baseline + pass³ + Loop + compare  **[H1, ≤3h, after WP-06, WP-08]**
- [ ] Leak guard unit tests: rejects ISO dates, ≥4-digit numbers, invoice ids, near-expected values; accepts the two canonical lessons L1/D1
- [ ] Witness present ⇒ mutation is `prune` (stubbed LLM)
- [ ] `occam run --task fx_recon_a --run-name t1 --max-gens 3 --cases 8 --pass3` completes; `reliability.completed` emitted
- [ ] `occam compare` prints the table and writes `compare.json`

### WP-11 — Real runs + curation  **[both, ≤3h]**
`occam lessons reset` → run1 on `fx_recon_a` → run2 on `fx_recon_b`. Tune architect/diagnose prompts until S1–S6 (`00 §6`) hold. Copy to `fixtures/demo_run1`, `demo_run2` (replacing hand-authored). Record noise floors, CIs, calls/case, costs for README.
- [ ] S1–S6 evidenced; `occam compare` shows run2.g0 ≥ run1.g0 + 0.25
- [ ] Lessons file contains ≥2 lessons, none case-specific (eyeball + guard log)

### WP-12 — README  **[H2, ≤1.5h, after WP-11]**
What/how/track/workflow/what improved (table from `compare`)/demo link/AO usage/honesty (`00 §7`)/sponsors (`06 §5`)/citations (`07`).
- [ ] Fresh clone → `pip install -e . && occam replay fixtures/demo_run1` in <5 min from README alone

### WP-13 — Recording rig  **[H2, ≤1h; first legibility pass by +10h]**
`textual serve` wrapper, `scripts/record_demo.sh` per `09`.
- [ ] 1080p capture of the ablation table readable at 100%

### WP-15 — Neatlogs tracing  **[H1, ≤1h, right after WP-05]**
Exact API per `prd/10-SETUP-VERIFICATION.md §2`: `neatlogs.init(api_key=..., workflow_name="occam", instrumentations=["openai"])` **called before `openai` is imported** (or `neatlogs.wrap(OpenAI())` per client); manual spans via `with neatlogs.trace(name, kind=...) as s: s.set_attribute(...)` around each `complete()` and each `fx_rate`/`fx_series` HTTP call (httpx is **not** auto-instrumented); `neatlogs.flush(); neatlogs.shutdown()` at run end. No-op when `NEATLOGS_API_KEY` unset.
- [ ] 5-case run → traces visible with role/generation/variant attributes; tool spans show `requested_date`/`rate_date`
- [ ] With key unset: identical events, timing within ±5%
- [ ] Last case's spans present after a short run (flush works)

### WP-14 (stretch) — Lesson ablation at run 2  (`03 §9`)
Only if WP-11 is done by +16h.

---

## Parallel schedule

| block (h from 04:00) | H1 | H2 |
|---|---|---|
| 0–1 | WP-01b | WP-03 starts |
| 1–3 | WP-02 | WP-03 |
| 3–6 | WP-05 (+WP-15) | WP-04 → WP-07 |
| 6–9 | WP-06 → WP-08 | WP-09 |
| **9** | **integration: real 8-case run replayed in TUI** | |
| 9–12 | WP-10 | WP-09 finish, WP-13 first pass |
| 12–15 | WP-11 (both) | WP-11 (both) |
| 15–17 | WP-11 tuning / WP-14 if green | WP-12 |
| 17–20 | recording (both) | recording (both) |
| 20–23 | README final, submission post, buffer | |

**Hour-12 rule:** if a real run hasn't produced an ablation table *and* at least one lesson by +12h, cut pass³ and the compare strip; ship run1 alone with lessons visible. **Hour-16 rule:** if run2 isn't showing improvement over run1.g0, the demo becomes within-run improvement + `cat lessons.md`; still a complete answer.
