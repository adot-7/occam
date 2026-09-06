# Open Questions

Workers: append under a dated heading when the PRD is wrong, ambiguous, or blocks you. Orchestrator resolves and moves items to "Resolved".

## Unresolved (as of 2026-09-06 04:00 IST)

- **TensorMux rate limit** — still unpublished. Start at rpm 60, raise if no 429s.
- **List-rate equivalent for GLM-4.7-Flash.** TensorMux publishes no rate card (`/pricing` 404). Using Cloudflare Workers AI's resale price ($0.06/$0.40 per MTok) as the labelled equivalent. If TensorMux gives a number, replace it in one place (`models.yaml`).
- **Frankfurter etiquette.** No documented rate limit; be polite: ≤5 concurrent, and the disk cache means generation ≥1 and all ablations are offline anyway. Commit `data/fx_cache/` after the first full run.
- **Neatlogs shareable trace URL** — trace viewed fine while logged in; open the same URL in a private window to see if it's public. Decides README link only; not blocking.
- **GLM thinking toggle.** Test whether `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (vLLM convention) or `extra_body={"thinking": {"type": "disabled"}}` (Z.ai convention) suppresses the `reasoning` field through TensorMux. If either works, expose it as a per-role flag `thinking: on|off` — a cheap, honest cost/latency lever (and optionally a mutation type `set_thinking`).

## Unresolved — appended 2026-09-06 by WP-05 (executor)

- **`control="llm"` has no defined router.** `01 §4.2` says "a router role decides"
  but no role type, tag, or wiring convention for a router is specified anywhere.
  The executor implements the weakest defensible reading: under `control="llm"`
  every role sees the whole context (task plus every produced `output_key`) rather
  than only its declared `inputs`, so the model decides relevance instead of the
  DAG. Deterministic control is unaffected and remains the default. Confirm before
  any architect is allowed to emit `control="llm"`.
- **WP-05 acceptance "real HTTP on the first run" is evidenced for the tool leg
  only.** No `.env` and no `TENSORMUX_API_KEY`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`
  exist in this workspace, so the LLM leg cannot make a real call here. The
  Frankfurter leg is evidenced live (`OCCAM_LIVE_HTTP=1`, real ECB rates, and
  `2026-04-03 → rate_date 2026-04-02` observed in a trace). The LLM leg is
  evidenced with a scripted provider that counts provider hits: run 1 makes 20,
  run 2 makes 0. `OCCAM_LIVE_LLM=1` runs the same test against `worker_fast` and
  should be run once a key is available.
### 2026-09-06 — WP-03 (FX packs, reference implementation, grader)

- **Tolerance sanity — answers the standing "Tolerance sanity" item above.** Reviewed on both generated packs: `fx_recon_a` |expected| ₹17,762.30–₹246,238.85, `fx_recon_b` ₹8,930.52–₹309,275.60; the tightest relative tolerance is the 0.1% floor (no case is impossibly tight), and the loosest absolute tolerance is ₹309.28 (no case is trivially passable — a whole invoice's gain is orders of magnitude larger). **The constraint the PRD does not state:** the D1 bank fee must exceed the grader's tolerance, or an agent that ignores the fee still passes the total and the fee cases teach nothing. The first pack draft had fees of ₹125–₹1,250 against tolerances up to ₹637, so 5 of 10 fee cases were undetectable at the total level. Fees are now ₹1,500–₹4,500 and the generator asserts `min_bank_fee_headroom > 1` (currently 6.09x and 10.63x). If `02 §1.2`'s tolerance ever changes, this invariant must be rechecked.

### 2026-09-06 — WP-03 hardening

- **FX cache coordination scope.** The client now coordinates cache-key locks and
  the five-request Frankfurter budget across all `FXClient` instances in one
  process, and uses same-directory atomic replacement so readers never observe a
  partial file. The PRDs require one engine process and do not say whether
  multiple processes may share `data/fx_cache`; process-local locks therefore do
  not claim to enforce a cross-process request budget or deduplicate live calls.
  Decide whether a cross-process lock/rate limiter is needed before introducing
  file locking or another platform-specific mechanism.

### 2026-09-06 — WP-04 (tool registry)

- **`ruff format --check .` fails on 38 pre-existing files in an AO Windows worktree.** The committed blobs are LF and format-clean (verified by running ruff against a `git archive` export), but the worktree checkout has CRLF while `pyproject.toml` sets `line-ending = "lf"`. Not a repo content bug, but it makes the standard verification command unusable in an AO worktree; a `.gitattributes` with `*.py text eol=lf` would settle it.
- **`wasted_calls` — resolved, see Resolved below.** Dropped from the PRD rather than defined.

## 2026-09-06 — WP-07 (TUI shell + replay): PRD notes for the orchestrator

Two PRD notes, neither blocking:

- **`04 §2` header shows `gen 1/6`; `design/figma.png` shows `g0/g5`.** Implemented the Figma form
  (`g{selected}/g{config.max_generations}`), since the PRD's example is a mock with no matching run.
- **`04 §1b` wants the task label `fx_recon_a · FX revaluation`.** Nothing in the event log carries
  the short human string, so it is derived from `task.goal` up to the first `:` — the header reads
  `fx_recon_a · Month-end FX revaluation`. If the exact wording matters, `run.started`'s `task`
  needs a `label` field (a schema change, so not taken here).

## 2026-09-06 — WP-10 live acceptance blockers

- **OpenAI worker request shape is not current-model compatible.** The configured
  `worker_alt` probe reaches OpenAI but returns a redacted HTTP 400 with
  `unsupported_parameter: max_tokens` for `gpt-5-nano`. The WP-10 checkpoint uses
  `worker_fast`; resolve the provider parameter contract before relying on the
  alternate worker lane.

## 2026-09-06 — global `~/.occam/cache` hazard for cold-start acceptance evidence

- **`~/.occam/cache` is global and shared across every worktree, checkout, and session for the OS user — a cold run can silently be served from a previous session's cache.** `01 §5` names both caches: "Content-addressed disk cache under `runs/<run_id>/cache/` (and a global `~/.occam/cache/` for cross-run reuse)." Discovered empirically while producing the live acceptance evidence for the `claude-sonnet-5` sampling-params fix (PR #19/#21): the first `python -m occam llm ping architect` came back `cached=True` even though it was the first invocation from that worktree, because an identical ping request had already been cached at `~/.occam/cache` by an earlier session on the same machine. Moving `~/.occam/cache` aside and rerunning produced a genuine `cached=False, cost_basis=metered` call; the original entries were then merged back.
  - **Where this bites `occam llm ping`, `Architect`, `Executor`, and `diagnose()` when used standalone:** all four fall back to `LLMClient()` with no arguments when not given an explicit shared client (`occam/cli.py` `llm_ping`; `occam/engine/architect.py:195`; `occam/engine/executor.py`; `occam/engine/diagnose.py:397`), and `LLMClient.__init__` defaults an unset `cache_dir` straight to `Path.home() / ".occam" / "cache"` (`occam/llm/client.py:267`) with no per-run isolation at all — every call through one of these bare constructors reads and writes the one shared, cross-session cache.
  - **Where the real `occam run` path currently differs:** `RunLoop.run()` (`occam/engine/loop.py:152`) constructs `LLMClient(cache_dir=run_dir / "cache")` — no `global_cache_dir` — and threads that single client into `Executor`, `Architect`, and `diagnose()` (`loop.py:188-350`), so `DiskCache.directories` for a full `occam run` currently contains only `runs/<run_id>/cache/`, not the global directory (`DiskCache.__init__` only appends `global_directory` when it is explicitly passed — `occam/llm/cache.py:28-34`). Read this as **current wiring, not a guarantee**: it depends on `self.llm` being threaded through every sub-component on every path, and on nobody adding a bare `LLMClient()` call anywhere reachable from a real run. It is exactly the kind of thing that would be invisible in the event log if it regressed — a cache hit looks identical whether it came from this run, a prior run, or the global directory, and no event field currently distinguishes them.
  - **Action for whoever runs WP-11:** before any run intended as cold-start evidence, explicitly clear or move aside `~/.occam/cache` (or otherwise confirm it is empty / not reachable from the run's `DiskCache.directories`), and state in the run report which cache state the run started from (e.g. "global cache cleared/absent; run cache empty at start"). Do not rely on `runs/<run_id>/cache/` looking empty as sole proof of a cold start — verify the global directory too. This is unresolved: no code change is proposed here, only the operational discipline needed until/unless the event schema is extended to record which cache tier served each hit.

## Resolved
- **2026-09-06 — WP-10 architect lane `BadRequestError` was misdiagnosed as a credential problem; it was a sampling-parameter regression — CONFIRMED LIVE against the real API.** The lane's model is `claude-sonnet-5`, which has removed sampling controls entirely. Direct A/B against the live API with a real key (once `occam/.env` existed) reproduced it exactly: sending `temperature` returns `400` with body `{"type":"invalid_request_error","message":"`temperature` is deprecated for this model."}` (`request_id req_011CenR1qTQqWbJyfiMdp6Ek`); the identical request with `temperature` omitted succeeds (`usage in=16 out=4`). The server names the parameter directly — this is not a credential, model-id, or structured-output-shape problem. `AnthropicProvider.complete` (`occam/llm/providers.py`) unconditionally sent `"temperature": temperature` on every request regardless of model, which is what produced the redacted `BadRequestError` right after `run.started`. Fix: the Anthropic request builder no longer puts `temperature`/`top_p`/`top_k` in the payload; `temperature` stays in the method signature so callers (including cache-key derivation) are unchanged. The OpenAI-compatible path (`OpenAICompatibleProvider`, used by the worker lane — TensorMux/GLM and `gpt-5-nano`) is untouched and still sends `temperature: 0` on every call — confirmed live via `occam llm ping worker_fast` (succeeds against TensorMux `glm-4-7-flash`, native `tool_calls`, `cost_basis: list-rate-equivalent`) — since that path's model family still accepts it and the worker lane's determinism depends on it. End-to-end acceptance: `python -m occam llm ping architect` run from `C:\Users\Sarthak\workspaces\occam` (the checkout with `.env`) after the fix — see PR #19 for pasted output. Regression coverage added in `tests/test_llm.py`: `test_anthropic_provider_never_sends_sampling_params` asserts none of the three keys appear in the Anthropic payload, and `test_openai_compatible_provider_still_sends_temperature_zero` asserts the worker payload still carries `temperature: 0`. **Honest consequence:** `AGENTS.md`'s non-negotiable "Deterministic where possible: temperature 0" is now literally unexpressible for the architect and diagnose lanes on Sonnet 5 — there is no request field left to pin. The worker lane, which makes far more calls per case (~20 per case vs. a couple of architect calls per run), still runs at temperature 0 and stays fully deterministic there. The residual architect/diagnose-lane nondeterminism is not assumed away: it is exactly the kind of variation the noise-floor and pass³ machinery (`03 §10`, WP-01c) already exist to measure, so it is caught and quantified rather than silently ignored.
- **2026-09-06 — Sonnet 5 price confirmed: $2.00 in / $10.00 out per MTok.** Resolves the standing "Sonnet 5 price" item above (the pages contradicting $2/$10 vs. $3/$15 from Sept 1 2026). $2.00/$10.00 is the correct current rate per the Anthropic API reference; `occam/config/models.yaml` already carries `in_per_m: 2.00` / `out_per_m: 10.00` for `claude-sonnet-5` and needs no change.
- **2026-09-06 — WP-03 answer format: the canonical answer is the LAST FENCED JSON BLOCK.** `02` contradicted itself — §1.2 extracted the answer from the last fenced JSON block while the `task.yaml` template in §3 instructed `answer_format: 'Final line: a JSON object {...}'`, so an agent obeying the pack's own instruction was ungradeable. **§1.2 wins and §3 was changed**, because §1.2 is the grading contract and `AGENTS.md` treats `02` as authoritative for the task; because a fenced block survives trailing prose whereas "final line" breaks the moment a model adds a closing sentence, and GLM-4.7-Flash emits reasoning and prose freely; and because the grader already implemented fence extraction. Both packs' `task.yaml` were regenerated to match. `fx_total` keeps a bare-JSON-line fallback when no fence is present — defensive salvage so a dropped fence cannot crash or fail a run, explicitly **not** part of the contract.
- **2026-09-06 — `wasted_calls` is dropped, not defined.** `02 §2` named it but nothing ever specified the aggregation. Ruling: remove the phrase; `02 §2` now reads "Feeds `n_tool_calls`." Reasoning: the disk cache is committed and permanent, so a repeat call costs approximately nothing and "wasted" is close to meaningless as a cost signal; and the metric appears in no success criterion (`00 §6`), no field in `events.schema.json`, and no TUI panel. `tool_calls_per_case` is already in `metrics.snapshot` and carries the whole cost/speed story, including the L2 range-endpoint halving. Defining a new metric under this deadline is scope we do not need. WP-04 keeps its raw per-call fields unchanged (`name`, `arguments`, `status`, `latency_s`, `bytes`, `cached`, `http_status`, `error`), so the metric can be reconstructed later if it ever earns its place.
- **2026-09-06 — WP-03 tolerance sanity is complete.** Both generated packs were checked; the 0.1% floor is the tightest relative tolerance, no case is impossibly tight or trivially passable, and bank-fee headroom exceeds the tolerance in every fee case. See the WP-03 evidence above.
- **2026-09-06 — Windows event locking is complete on `main`.** PR #7 is merged at `fde72e3`; `EventWriter` now selects `msvcrt` on Windows and `fcntl` elsewhere, with import and concurrent-append regression coverage. The duplicate WP-03/WP-04 blockers are removed from Unresolved.
- **2026-09-06 — WP-08 executor result paths are confirmed.** A full run writes
  `generations/gNNN/results.jsonl`; `full_repeat` and knockouts write
  `results.<slugged variant>.jsonl`, including the raw per-role traces needed by
  ablation. `tests/test_ablation_executor_integration.py` validates these paths
  through `Executor` and `EventReader`.
- **2026-09-06 — WP-08 full-repeat cache bypass is confirmed.** The executor
  passes `use_cache=False` through the LLM client; fresh responses are neither
  read from nor written to the content cache, and the integration test verifies
  the canonical full cache remains unchanged.
- **2026-09-06 — WP-01c noise-floor and cost contracts are on `main`.** PR #8
  merged at `597f349`; WP-08 consumes the shared contract without duplicating its
  schema, model, fixture, executor-cost, or documentation changes.
  `noise_rate` is required on `ablation.started`; `RoleTrace.cost_usd` is the
  displayed cost, while `billed_cost_usd` and `cost_label` preserve the nominal
  billing basis. Cost shares reject a zero displayed-cost denominator rather
  than invent token or uniform shares.
- **2026-09-06 — WP-01b fixture and design-reference decisions:**
  - **Run-2 lesson count:** resolved to 3, matching `prd/08-END-TO-END-WALKTHROUGH.md` and lessons L1, D1, and L2. The WP-01b acceptance wording of 2 is corrected; the canonical fixture and acceptance evidence use 3.
  - **Design reference filename:** resolved to `design/figma.png`; it is the intended visual target and there is no missing `figma-v2` asset. No design file was changed.
  - **Benchmark citation discrepancy:** SMFR, BFCL, and MGSM entries in `prd/07-CITATIONS.md` intentionally remain as research foundation, not active benchmarks. The citations file was left unchanged.

- **Neatlogs end-to-end: CONFIRMED 2026-09-06 05:32** on 1.4.21 — `case.smoke` (WORKFLOW) → `tool.fx_rate` (TOOL, manual, 25µs) → auto-captured `glm-4-7-flash` LLM span with token count. Manual spans nest under auto-instrumented ones with no extra wiring. `flush()` returned True.
- **GLM-4.7-Flash reasons before answering.** With `max_tokens=5` the response `content` was `None` — the budget went to the hidden `reasoning` field. Rules for the executor: (1) never set `max_tokens` below ~1024 for worker roles (use 2048 default); (2) if `content` is None/empty and `reasoning` is present and `finish_reason == "length"`, treat as TRUNCATED, not as an empty answer — retry once with a larger budget, then fail the case; (3) cost = `usage.completion_tokens` verbatim (includes reasoning).
- **TensorMux native tool calling: CONFIRMED 2026-09-06 05:20** with the real key — vLLM backend (`vllm-0.25.1`), OpenAI-shaped `tools`, response has `tool_calls[0].function.{name,arguments}` (arguments is a JSON string), `finish_reason: "tool_calls"`, full `usage` block. Executor uses native tools. **Note:** the model also returns a `reasoning` field; its tokens are billed inside `completion_tokens` (152 tokens for a one-line tool call) — cost accounting must use `usage.completion_tokens` as-is, not count visible output.
- **GPT-5 nano key works** (paid tier confirmed by a successful call).
- **Neatlogs SDK version:** the API in `06 §3` (`workflow_name`, `trace`, `span`, `wrap`, `flush`, `shutdown`) is **1.4.21**, which requires **Python ≥3.10**. On Python 3.9 pip silently installs 1.1.x with a different API (`init(api_key, tags, debug)` only). Pin `neatlogs>=1.4.21` in `pyproject.toml`; the project is 3.11+ anyway.
- Domain: one workflow, FX revaluation, two packs (`02`). Benchmarks dropped.
- Language: Python + Textual (`04 §0`). Engine/TUI seam: run directory + events (`01 §1`).
- Baseline: cost-matched CoT-SC (`03 §6`). Reliability: pass³ (`03 §10`).
- Lessons: typed, evidence-backed, leak-guarded, read by the architect on the next run (`01 §4.1, §4.4`).
- Lesson ablation: stretch (`03 §9`).
- Tool: Frankfurter via `fx_rate`/`fx_series`, disk-cached and committed (`02 §2`).
