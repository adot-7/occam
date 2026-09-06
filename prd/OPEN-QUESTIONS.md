# Open Questions

Workers: append under a dated heading when the PRD is wrong, ambiguous, or blocks you. Orchestrator resolves and moves items to "Resolved".

## Unresolved (as of 2026-09-06 04:00 IST)

- **TensorMux rate limit** — still unpublished. Start at rpm 60, raise if no 429s.
- **List-rate equivalent for GLM-4.7-Flash.** TensorMux publishes no rate card (`/pricing` 404). Using Cloudflare Workers AI's resale price ($0.06/$0.40 per MTok) as the labelled equivalent. If TensorMux gives a number, replace it in one place (`models.yaml`).
- **Sonnet 5 price** — Anthropic pages contradict ($2/$10 vs $3/$15 from Sept 1 2026). Immaterial to the demo (few architect calls); recheck before quoting publicly.
- **Frankfurter etiquette.** No documented rate limit; be polite: ≤5 concurrent, and the disk cache means generation ≥1 and all ablations are offline anyway. Commit `data/fx_cache/` after the first full run.
- **Tolerance sanity.** After generating packs, check that `max(5, 0.1%)` doesn't make any case trivially passable or impossibly tight (JPY-heavy batches). Adjust in `02 §1.2` if needed — one place.
- **Neatlogs shareable trace URL** — trace viewed fine while logged in; open the same URL in a private window to see if it's public. Decides README link only; not blocking.
- **GLM thinking toggle.** Test whether `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (vLLM convention) or `extra_body={"thinking": {"type": "disabled"}}` (Z.ai convention) suppresses the `reasoning` field through TensorMux. If either works, expose it as a per-role flag `thinking: on|off` — a cheap, honest cost/latency lever (and optionally a mutation type `set_thinking`).

## Resolved
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
