# 06 — Sponsor Integrations (v3)

What we have, what each is for, exactly where it plugs in. None of these change product logic; they set `models.yaml`, add one tracing hook, and give us truthful lines for the README and X. **Setup specifics (endpoints, param names, prices) must come from `prd/10-SETUP-VERIFICATION.md`, which was checked against the vendors' own docs — not from memory.**

## 1. TensorMux — primary worker compute

**What we have:** GLM-4.7-Flash via TensorMux's OpenAI-compatible gateway, **50M granted tokens**, more on request. Their ask: tag them on X when we share.

**Quickstart (from TensorMux, verified by the team):**
```bash
curl -sS "https://api.tensormux.com/v1/chat/completions" \
  -H "Authorization: Bearer $TENSORMUX_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm-4-7-flash","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```
So: `base_url = https://api.tensormux.com/v1`, `model = glm-4-7-flash` (hyphens, not dots).

**Why it matters:** the ablation cost analysis in `03 §3` was originally written against a free tier with tight per-minute limits. 50M tokens removes the token budget as a constraint. A full two-run FX demo (two 5–6-generation runs with ablation and pass³) is roughly 2–4M tokens. We can iterate freely.

**`models.yaml`:**
```yaml
worker_fast:                          # all candidate-agent roles, ablation knockouts, CoT-SC baseline
  provider: openai_compat
  base_url: https://api.tensormux.com/v1
  api_key: ${TENSORMUX_API_KEY}
  model: glm-4-7-flash
  in_per_m: 0.0                       # granted
  out_per_m: 0.0
  grant_equiv_in_per_m: 0.06          # TensorMux publishes no rate card; this is Cloudflare Workers AI's resale price for glm-4.7-flash — label it as such
  grant_equiv_out_per_m: 0.40
  rpm: 60                             # UNKNOWN; start conservative, raise if no 429s
  supports_json_schema: false         # model supports response_format text|json_object only
  tool_choice_modes: [auto]           # "required"/"none" not supported by the model
```
**Verified (prd/10 §1):** `GET https://api.tensormux.com/v1/models` works without a key and lists exactly one model, `glm-4-7-flash`. **Unverified on this host:** `tools`/`tool_calls`, `usage` block, rate limits, price. The model itself (Z.ai) supports `tools` with `tool_choice="auto"` **only** and `response_format` `text`/`json_object` **only — no `json_schema`**. Consequences: (a) the executor must never rely on `json_schema` for worker roles — final answers are parsed from the last fenced JSON block, which is already the answer format; (b) never pass `tool_choice="required"`; (c) confirm the `usage` block on the very first response — cost accounting depends on it. Fallback if `usage` is missing: count tokens client-side with `tiktoken` and label the cost as approximate.
**Cost display rule:** granted tokens would show `$0.00` and kill the cost story. Compute cost at list rate, label it `cost (list-rate eq.)`, say so in the README. Ratios between generations are rate-independent, so nothing is misrepresented.

**Day-one check — DONE 2026-09-06 05:20, PASSED.** Native `tool_calls` work (vLLM backend, OpenAI-shaped), `finish_reason: "tool_calls"`, full `usage` block present. **Executor uses native tool calling.** Three facts from real responses: (a) the model returns a `reasoning` field whose tokens are billed inside `completion_tokens` — use `usage.completion_tokens` verbatim for cost, never count visible text; (b) `tools` must be an array of `{"type":"function","function":{...}}` objects — vLLM rejects bare names with a 400; (c) **a small `max_tokens` starves the visible answer**: with `max_tokens=5` `content` came back `None` because the budget went to reasoning. Worker roles default to `max_tokens=2048`; `content is None` + `finish_reason == "length"` ⇒ TRUNCATED → one retry with double budget, then fail the case. Optional: test the thinking-off flag (`OPEN-QUESTIONS.md`) as a per-role cost/latency lever.

## 2. AI Grants India — GPT-5 nano + Smallest.ai voice credits

**GPT-5 nano** (OpenAI, granted credits; model id and list price per `prd/10-SETUP-VERIFICATION.md`):
```yaml
worker_alt:                           # second concurrency lane for ablation knockouts; noise-floor repeat run
  provider: openai_compat
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-5-nano                   # confirmed id (prd/10 §3); Chat Completions supported; 400K context
  in_per_m: 0.05                      # $0.05 in / $0.005 cached / $0.40 out per MTok — column labels lost in crawl; re-check the pricing page before quoting publicly
  out_per_m: 0.40
  rpm: 500                            # Tier 1: 500 RPM / 200K TPM. NOT available on the Free tier — the AI Grants key must be on a paid tier; verify with one call
```
- Strong native tool calling → also the **fallback worker** if GLM tool calls prove flaky.
- Never the architect; that needs a frontier model.

**Smallest.ai voice credits:** not part of the product. Use for the **demo voiceover** (script in `09`); credit them in the README. Nothing else.

## 3. Neatlogs — traces for the inspector and the README

**Use, one thing done well:** a span around every `llm.complete()` **and every HTTP call in `tools/fx.py`**. Attributes: `run_id, run_name, generation, variant (full | full_repeat | ablate:<role> | baseline | pass3), case_id, role_id, role_name, justification, model, tokens_in, tokens_out, cost_usd, cached, passed`; for tool spans: `tool, requested_date, rate_date, status, bytes, cached`. One trace per `(run, generation, variant, case)`.

What it buys: a hosted trace for any demo case — including the `requested_date → rate_date` mismatch on the failing holiday case — plus a trace-count-over-generations view, plus a truthful README line: "every model call and every API call in our runs is traced."

**Implementation (WP-15, ≤1h, right after WP-05)** — exact API confirmed in `prd/10 §2` and by reading the 1.4.21 wheel source. **Pin `neatlogs>=1.4.21` (requires Python ≥3.10); older 1.1.x has a different API and no `trace`/`span`/`flush`.**
```python
# occam/llm/tracing.py — imported FIRST in cli.py, before anything imports openai
import os, neatlogs
ENABLED = bool(os.getenv("NEATLOGS_API_KEY"))
if ENABLED:
    neatlogs.init(api_key=os.environ["NEATLOGS_API_KEY"], workflow_name="occam", instrumentations=["openai"])

from contextlib import contextmanager
@contextmanager
def span(name: str, kind: str = "CHAIN", **attrs):
    # valid kinds (from the 1.4.21 source): WORKFLOW, AGENT, CHAIN, TOOL, RETRIEVER, EMBEDDING, MCP_TOOL
    if not ENABLED:
        yield None; return
    # neatlogs.trace is a @contextmanager yielding an OTel span; extra kwargs become span attributes
    with neatlogs.trace(name, kind=kind, **{f"occam.{k}": v for k, v in attrs.items()}) as s:
        yield s

def shutdown():
    if ENABLED: neatlogs.flush(); neatlogs.shutdown()
```
Use `span("llm.complete", kind="CHAIN", model=..., role_id=..., generation=..., variant=..., case_id=...)` in `complete()` and `span("tool.fx_rate", kind="TOOL", requested_date=..., rate_date=..., cached=...)` in `fx.py`; wrap each case in `span(f"case.{case_id}", kind="WORKFLOW", ...)` — **httpx is not auto-instrumented**, so tool spans are manual. Call `shutdown()` at the end of `occam run`. Footguns: `init()` before importing `openai`; flush or lose the last spans; record latency before export so tracing never changes metrics. Free-tier limits and shareable trace URLs are **UNKNOWN** — smoke-test on day one; if capped, trace `full` variants only.

## 4. AO — build tool

Orchestrator + workers on parallel worktrees per `05`. Screen-record the Kanban with 3–4 live sessions and `git log --graph --oneline --all`. That footage is the AO requirement. AO is never in the runtime.

## 5. Frankfurter — the third-party tool (not a sponsor)

Free, keyless ECB reference rates at `https://api.frankfurter.dev/v1`. It is the "third-party app access" in our submission. Not a sponsor; that's fine — the judge said the app and domain don't matter. Verified behaviour in `02 §2`.

## 6. What we say, where

| Party | README line | Demo/video | X |
|---|---|---|---|
| AO | "Built with AO: N sessions across M work packages, engine and TUI on parallel worktrees." | 45s Kanban + git graph | tag @aoagents |
| TensorMux | "All candidate-agent inference ran on GLM-4.7-Flash via TensorMux (~X M tokens)." | cost strip footnote | tag them (their ask) |
| AI Grants India | "GPT-5 nano via AI Grants India credits as second lane; Smallest.ai for narration." | voiceover | tag |
| Neatlogs | "Every model and API call traced; example: <link>." | inspector screenshot | tag |
| Maximor | "The workflow is month-end FX revaluation — real Office-of-the-CFO work — chosen so Track 1's agent-engineering story runs on a finance task." | task badge | — |
| Frankfurter/ECB | "Rates: ECB reference rates via the open Frankfurter API." | — | — |
