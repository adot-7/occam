# 06 — Sponsor Integrations

What we have, what each is for, and exactly where it plugs in. None of these change the product's logic; they change `models.yaml`, add one tracing decorator, and give us things to say in the README and on X.

## 1. TensorMux — primary worker compute

**What we have:** GLM-4.7-Flash via TensorMux's OpenAI-compatible gateway, **50M tokens**, more on request ("burn our GPUs"). Ask: mention them on X when sharing.

**Why this matters more than it looks:** the whole ablation cost analysis in `03 §3` was written around Groq's free tier (30 RPM, 6,000 TPM). 50M tokens at a hosted gateway removes the token budget as a constraint entirely — a 6-generation SMFR run is roughly 1.5–3M tokens including ablation. We can run all three domains several times over and still have most of the grant left.

**Role in `models.yaml`:**
```yaml
worker_fast:                       # candidate-agent roles, ablation knockouts, CoT-SC baseline
  provider: openai_compat
  base_url: ${TENSORMUX_BASE_URL}  # https://<endpoint>/v1
  api_key: ${TENSORMUX_API_KEY}
  model: glm-4.7-flash             # confirm exact id via GET /v1/models on day one
  in_per_m: 0.0                    # granted; still record token counts — cost shown as "grant-equivalent" using a published rate
  out_per_m: 0.0
  grant_equiv_in_per_m: <fill from TensorMux pricing page>
  grant_equiv_out_per_m: <fill>
  rpm: <ask TensorMux>             # unknown; assume generous, set 120, back off on 429
```
**Cost display rule:** the TUI's `$` column must show a *real* number or the demo's cost story dies. Since the tokens are granted, compute cost at TensorMux's list rate and label the metrics strip `cost (list-rate equiv.)`. State this in the README. It's honest and it keeps "cost −40%" meaningful — the *ratio* between generations is what matters, and it's rate-independent.

**Open check (day one, 10 minutes):** does GLM-4.7-Flash do native OpenAI-style tool calls through TensorMux? SMFR roles need it. If flaky → executor falls back to the JSON-in-text tool protocol (`OPEN-QUESTIONS.md`). Test with one `lookup_price` call before anything else.

**Observability bonus:** TensorMux's gateway exposes per-request latency and token counts (Prometheus `/metrics` on self-hosted; dashboard on hosted). We don't depend on it — our own per-role accounting is authoritative — but a screenshot of their dashboard showing our run's token curve is a free README image and a genuine "we used the sponsor's product" signal.

## 2. AI Grants India — GPT-5 nano + Smallest.ai voice credits

**GPT-5 nano** (OpenAI, metered via granted credits; $0.05/$0.40 per M list):
```yaml
worker_alt:                        # second lane for ablation concurrency; BFCL primary worker
  provider: openai_compat
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}       # the AI Grants key
  model: gpt-5-nano
  in_per_m: 0.05
  out_per_m: 0.40
  rpm: 500
```
- Strong native tool calling → make it the **default worker for `bfcl_simple`** (tool-call fidelity is the whole task there) and the **second concurrency lane** for SMFR ablation knockouts.
- Because it's cheap and deterministic-ish, use it for the **noise-floor repeat run** too.
- Do *not* use it as the architect; that needs a frontier model (Sonnet 5 / GPT-5).

**Smallest.ai voice credits:** not part of the product. Two legitimate uses, both optional:
1. **Demo video voiceover.** Script the 3-minute narration, generate with Smallest.ai, lay it over the screen recording. Consistent pacing, no room noise, retakes are free. Mention in README under sponsors.
2. (Gimmick, skip unless bored) `occam run --speak`: read each `diagnosis.emitted` aloud. Cute in a live room, useless in a video. Not MVP.

## 3. Neatlogs — traces for the failure inspector and the README

**What it is:** LLM/agent observability, OTel/OpenInference-native, Python SDK, one-line init, "free to start, no card." The hackathon page explicitly says it can "show how your system improved over time."

**What we use it for — one thing, done well:** every LLM call in the executor already goes through `llm.complete()`. Wrap that in a span. Attributes: `run_id, generation, variant (full | ablate:<role> | baseline), case_id, role_id, role_name, justification, model, tokens_in, tokens_out, cost_usd, cached, passed`. One trace per `(generation, variant, case)`.

That gives us, for free:
- A hosted, clickable trace for any case in the demo — the "failure inspector" the judges can open themselves if a link is shareable.
- A trace-count-over-generations view that literally is "how the system improved over time."
- A screenshot for the README and a truthful sentence: "every one of the N,NNN model calls in our runs is traced in Neatlogs."

**Implementation (WP-15, ~90 minutes, do it right after WP-05 so all later runs are traced):**
```python
# occam/llm/tracing.py
import neatlogs; neatlogs.init(api_key=os.environ["NEATLOGS_API_KEY"])   # no-op if key missing
@contextmanager
def llm_span(attrs: dict): ...    # opens span, sets gen_ai.* + occam.* attributes, records tokens/cost on exit
```
- Wrap `complete()`; add `occam.*` attributes from a contextvar set by the executor per (generation, variant, case, role).
- **Two documented footguns:** flush spans before process exit (short runs drop the last spans otherwise) and don't use generic LangChain auto-instrumentation (we don't use LangChain, so irrelevant — but don't add it).
- Tracing must be **off by default when no key is present** and must never affect cost accounting or timing (record latency before span export).
- Pricing tiers beyond "free to start" are undocumented. Do a 5-minute smoke test on day one; if volume is capped, trace only `full` variants, not ablation knockouts.

## 4. AO — build tool (unchanged)

Orchestrator + workers on parallel worktrees per `05-WORK-PACKAGES.md`. Screen-record the Kanban board with 3–4 sessions live and `git log --graph --oneline` showing the WP branches merging. That footage is the AO requirement. AO is never in the runtime.

## 5. What we say, where

| Sponsor | README line | Demo/video | X post |
|---|---|---|---|
| AO | "Built entirely with AO: N sessions across M work packages, engine and TUI on parallel worktrees." | 45s Kanban + git graph | tag @aoagents |
| TensorMux | "All candidate-agent inference ran on GLM-4.7-Flash via TensorMux (~X M tokens)." | cost strip footnote | tag them, per their ask |
| AI Grants India | "GPT-5 nano via AI Grants India credits served as second lane and BFCL worker; Smallest.ai for narration." | voiceover | tag |
| Neatlogs | "Every model call traced; example trace: <link>." | one inspector screenshot | tag |
| Maximor | "Primary domain is SMFR, the financial multi-hop reasoning benchmark from Jwalapuram et al. 2026." | domain badge | — |
| Dodo | not used; don't force it | — | — |
