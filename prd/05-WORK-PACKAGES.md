# 05 — Work Packages

Ordered so that the two humans (H1: engine lead, H2: TUI + data lead) and their AO workers are never blocked on each other after WP-01. Each WP = one AO session = one branch `wp/<nn>-<slug>` = one PR. **Acceptance criteria are the definition of done; copy them into the PR.**

Dependencies are listed; anything not listed is independent. Parallel lanes are marked **[H1]** / **[H2]**; a WP with no lane can go to whoever is free.

---

### WP-01 — Repo skeleton + schemas + fixture  **[both, first 60–90 min]**
Create `pyproject.toml` (ruff, pytest, pydantic, typer, textual, httpx, anthropic, openai, datasets/pyarrow), package layout from `AGENTS.md`, `.env.example`, `CHANGELOG.md`.
Write `schemas/events.schema.json`, `schemas/state.schema.json`, `schemas/task.schema.json` per `01 §3` and `02 §1`.
Write `occam/core/` models. Write `occam/store/reducer.py` (`reduce(events) -> State`) and `occam/store/writer.py` / `reader.py`.
**Hand-author `fixtures/demo_run/events.jsonl`**: a complete, realistic 4-generation SMFR run including a prune, a split+revert, ablation rows with mixed verdicts, and baselines — ~120 events. This is the TUI's development dataset and the demo backup; make it plausible.
- [ ] `pytest tests/test_schemas.py` validates every fixture event against the schema
- [ ] `occam validate fixtures/demo_run` passes; `reduce()` is deterministic (run twice, byte-identical state)
- [ ] `ruff check` clean

> Everything below forks from here. Do not start WP-02+ until WP-01 is merged.

---

### WP-02 — LLM layer  **[H1]**
`occam/llm/`: `complete()` interface; OpenAI-compatible + Anthropic providers; `models.yaml` loader with `${ENV}` interpolation; per-model token-bucket rate limiter; retries; cost computation; content-addressed disk cache; `cached` flag in results.
- [ ] `occam llm ping worker_fast` returns a completion with tokens, cost, latency
- [ ] Same call twice → second is `cached=True`, cost 0
- [ ] Rate limiter test: 40 calls at rpm=30 take ≥ 20s wall-clock (mock provider)
- [ ] Unknown model key → clear error naming `models.yaml`

### WP-03 — Task packs + adapters + checkers  **[H2]**
`occam/tasks/`: pack loader (`task.yaml` + `cases.jsonl`, schema-validated); checkers per `02 §2`; adapters `smfr.py`, `bfcl.py`, `mgsm.py`; `scripts/prepare_tasks.py`. Commit generated packs `tasks/smfr_2inv`, `tasks/bfcl_simple`, `tasks/mgsm_en` plus two held-out packs (`smfr_3inv`, `mgsm_de` or `bfcl_multiple`).
- [ ] `python scripts/prepare_tasks.py --all` regenerates identical packs (fixed seeds)
- [ ] `occam task list` shows 5 packs with case counts and mean input tokens
- [ ] Checker unit tests: `json_set_equal` (order/case), `numeric_exact` (last-number extraction), `bfcl_ast` (required/optional params, allowed-value lists) — ≥ 15 cases
- [ ] SMFR haystack parsed into `meta["_haystack"]`; `lookup_price("Airbnb","2026-01-05","Close")` returns 135.87 for the first record

### WP-04 — Tools  **[H2, after WP-03]**
`occam/tools/registry.py` + `python_exec` (subprocess, 5s timeout, no network, whitelist imports), `calculator`, `lookup_price`, `list_transactions`, virtual BFCL tools from per-case manifests.
- [ ] `python_exec("print(2**10)")` → `"1024"`; `import socket` → rejected; infinite loop → timeout error
- [ ] BFCL virtual tools expose the record's `function` list as tool specs; a model tool-call is captured, not executed

### WP-05 — Executor  **[H1, after WP-02; needs WP-03 packs for tests]**
`engine/executor.py`: DAG executor per `01 §4.2`; per-role traces; concurrency; sentinel handling for knocked-out roles; emits `execution.*`.
- [ ] Executes a hand-written 3-role architecture on 5 MGSM cases end to end; results in `generations/g000/results.jsonl`
- [ ] Per-role cost sums equal case cost
- [ ] Knockout of role 2 with sentinel: roles 1 is a cache hit (asserted via `cached=True`), role 3 recomputes
- [ ] A provider failure on one case marks it failed and the run completes

### WP-06 — Architect + Mutations  **[H1, after WP-05]**
`engine/architect.py` (JSON-schema-constrained proposal, 3–6 roles, justification tags, tools only from manifest) and `engine/mutate.py` (closed menu, `parent_id`, `diff`).
- [ ] `occam architect --task smfr_2inv` prints a valid `Architecture`; roles reference only manifest tools; DAG acyclic
- [ ] Each mutation type has a unit test producing a valid child architecture; `prune` rewires consumers correctly
- [ ] Architect prompt includes 3 example cases and the answer format

### WP-07 — TUI shell + replay  **[H2, after WP-01; parallel to WP-02..06]**
`occam/tui/`: app, layout, Header, Lineage tree, Metrics strip, Diagnosis feed, replay reader with speed/pause/step, `state.json` instant paint. Driven entirely by `fixtures/demo_run`.
- [ ] `occam replay fixtures/demo_run --speed 8` plays to completion; lineage shows 4 gens incl. a reverted node
- [ ] `--to-gen 3` fast-forwards then plays
- [ ] Pilot test: after replay, lineage has 4 nodes; metrics strip shows 4 points
- [ ] Renders correctly at 100×30 and 120×40

### WP-08 — Ablation + metrics  **[H1, after WP-05]** ★ the core
`engine/ablation.py`, `metrics/`: noise-floor repeat run, per-role knockouts with upstream cache reuse, divergence/influence/cost_share, bootstrap CI, verdict rule, structural fidelity, skip-unchanged-roles optimisation; emits `ablation.*`.
- [ ] Property tests from `03 §4` pass
- [ ] On a synthetic architecture with a deliberately inert role (output never referenced), verdict is `witness` with divergence 0
- [ ] On MGSM 15 cases with a 3-role arch, full ablation completes and emits 3 `ablation.role` events with CIs
- [ ] `structural_fidelity` = sum of load-bearing cost shares, asserted on fixture data

### WP-09 — Ablation table + Architecture panel + Case Inspector  **[H2, after WP-07]**
Hero `DataTable` with verdict styling and live row updates; ASCII DAG with prune strikethrough animation; Cases grid; Inspector modal with per-role outputs and full-vs-ablated side-by-side.
- [ ] Replay to gen 3 shows 5 rows with verdicts matching the fixture
- [ ] A `mutation.applied{prune}` event strikes through then removes the role box
- [ ] Inspector opens on a failed case and shows every role's output

### WP-10 — Baseline + Diagnose + Loop  **[H1, after WP-06, WP-08]**
`engine/baseline.py` (cost-matched CoT-SC), `engine/diagnose.py` (with the hard rule: witnesses → prune), `engine/loop.py` (plateau, budget, split-then-verify-then-revert), `occam run`.
- [ ] `occam run --task mgsm_en --max-gens 3 --cases 15` completes, writes a valid run dir, `occam validate` passes
- [ ] `baseline.completed` has `k` such that cost is within ±30% of the generation's cost
- [ ] If a witness exists, `diagnosis.emitted.chosen_mutation.type == "prune"` (unit test with a stubbed LLM)
- [ ] A `split` whose new roles are witnesses produces `mutation.reverted`

### WP-11 — Full runs on three domains + demo run curation  **[both, after WP-10]**
Run `smfr_2inv`, `bfcl_simple`, `mgsm_en` with real models. Tune prompts/thresholds until S1–S6 in `00 §6` hold. Pick the best SMFR run as `runs/demo_smfr` and copy it to `fixtures/demo_run` (replacing the hand-authored one). Record the noise floor, CIs and costs for the README.
- [ ] S1–S6 evidenced with run dirs committed under `runs/` (cache dirs excluded)
- [ ] `occam replay runs/demo_smfr --to-gen 2` lands on a populated ablation table with ≥1 witness
- [ ] Total spend across all runs logged in `CHANGELOG.md`

### WP-12 — README + submission assets  **[H2, after WP-11]**
README per submission rules: what it does, how to run, track, agent workflow, what improved across iterations (table from real runs), demo link, **how AO was used** (sessions, worktrees, which WPs ran in parallel). Honesty section from `00 §8`. Citations from the research pack.
- [ ] Fresh clone → `pip install -e . && occam replay fixtures/demo_run` works in < 5 min following README only
- [ ] Every claim in README traces to a run dir or a citation

### WP-13 — Legibility + `textual serve` + recording rig  **[H2, any time after WP-09; do a first pass by hour 12]**
Large-font palette, `textual serve` wrapper, a `scripts/record_demo.sh` that launches replay with the right `--to-gen`/speed for each demo beat.
- [ ] A 1080p screen recording of the ablation table is readable at 100% zoom (human check)

### WP-14 (stretch) — Live task creation
`occam task new --goal ...`: LLM drafts cases; TUI approval screen; writes `tasks/live_<slug>/`.
- [ ] Draft → approve 8/10 → run proceeds using only approved cases

### WP-15 (stretch) — Neatlogs + TensorMux
Optional OTel export of LLM spans to Neatlogs; `worker_fast` routed via TensorMux hosted `base_url` for dollar-denominated tracking.
- [ ] One run visible as traces in Neatlogs; TensorMux dashboard shows spend matching our computed cost within 10%

---

## Suggested parallel schedule

| Block | H1 lane | H2 lane |
|---|---|---|
| 0 | WP-01 (together) | WP-01 (together) |
| 1 | WP-02 → WP-05 | WP-03 → WP-04 → WP-07 |
| 2 | WP-06 → WP-08 | WP-09 → WP-13 (first pass) |
| **integration** | **WP-08 engine output replayed in WP-09 TUI — first real end-to-end** | |
| 3 | WP-10 | WP-13 polish, README skeleton |
| 4 | WP-11 (together) | WP-11 (together) |
| 5 | WP-14/15 if time | WP-12 |

**Hour-20 rule:** if WP-08's ablation table isn't populating from a real run by hour 20, stop everything else and make it work with fewer cases, fewer roles, one domain. The ablation table alone is a shippable submission; nothing else is.
