# Setup Verification Report — Primary-Source Check

Methodology: every claim below was checked live against the vendor's own site/docs
(via `curl`, or `ExaContents` with `livecrawl: always` against the vendor domain) or the
official PyPI JSON API. Anything I could not confirm this way is marked **UNKNOWN** —
no invented parameter names, model ids, or prices appear anywhere in this document.
Checked 2026-09-06.

---

## 1. TensorMux (api.tensormux.com, glm-4-7-flash)

**What TensorMux actually is:** TensorMux is a real product with two faces:
1. An open-source, self-hosted "Tensormux Gateway" (config-file-driven OpenAI-compatible
   reverse proxy in front of your own inference backends — GitHub `KrxGu/Tensormux`).
2. A managed multi-tenant cloud offering (`www.tensormux.com` homepage: "One OpenAI
   endpoint. Many inference backends," with Gateway/Shared/Dedicated/On-Prem tiers).
   The `api.tensormux.com` host the team's quickstart points at is this managed side.

**(a) Docs URL** — `https://docs.tensormux.com` returns **HTTP 502 Bad Gateway**
(confirmed live via `curl`; the docs subdomain is currently broken). `https://tensormux.com/docs`
307-redirects to `https://www.tensormux.com/docs`, which **is live** and returns content —
but that page documents the **self-hosted OSS gateway** (YAML config, Docker Compose,
`/v1/chat/completions`, `/v1/models`, `/tensormux/status`, `/tensormux/requests`, `/metrics`,
`/ui` dashboard), not the hosted `api.tensormux.com` service specifically. There is no
separate hosted-API reference doc I could locate.

**(b) Does `GET https://api.tensormux.com/v1/models` exist without a key?**
Yes — confirmed live, no `Authorization` header sent:
```
$ curl https://api.tensormux.com/v1/models
{"object":"list","data":[{"id":"glm-4-7-flash","created":0,"object":"model","owned_by":"aibrix"}]}
```
HTTP 200. So this endpoint requires **no auth** to list models (auth is only enforced on
`/v1/chat/completions`, confirmed separately — see below).

**(c) Model id** — Confirmed **exactly `glm-4-7-flash`** (hyphenated), and it is the
**only** model this deployment lists. `owned_by` is `"aibrix"`, indicating the backend
serving engine is AIBrix.

**(d) `tools`/`tool_choice`/`tool_calls`** — Not verifiable end-to-end without a real key
(a request with a fake key returned `{"error":{"code":"invalid_api_key",...}}` before
reaching the model). TensorMux's own docs page doesn't document the chat-completions
payload schema. However, the underlying model (GLM-4.7-Flash) is documented by its
vendor Z.ai as fully supporting `tools`/`tool_choice` (see §GLM below), and TensorMux
markets itself as a byte-for-byte OpenAI-compatible passthrough — so this is **very
likely supported but not directly confirmed against api.tensormux.com**. Mark as
**UNKNOWN (not directly confirmed on this host)**.

**(e) `response_format`** — Same caveat as (d): **UNKNOWN**, not directly confirmed on
`api.tensormux.com`. (Z.ai's own docs say the underlying model only supports
`response_format` of `text` or `json_object` — **not** `json_schema`. If TensorMux passes
requests straight through, assume the same ceiling.)

**(f) Documented rate/concurrency limits** — **UNKNOWN**. Neither `www.tensormux.com/docs`
nor the homepage publishes numeric rate limits for the hosted `api.tensormux.com` service.
(The OSS gateway's own docs only describe backend health-check intervals, not client-facing
rate limits.)

**(g) List price for glm-4-7-flash on TensorMux** — **UNKNOWN / not publicly published.**
`https://tensormux.com/pricing` and `https://www.tensormux.com/pricing` both 404. The
homepage names pricing *models* per tier ("Free" for self-hosted Gateway, "Per-token" for
Shared cloud, "Reserved" for Dedicated, "Custom" for On-Prem) but shows **no dollar
figures**. I could not find a TensorMux-published rate card for this model. If you need a
reference number for slides, the model's own vendor (Z.ai) publishes list pricing — see
the GLM section below — but that is Z.ai's price, not necessarily what TensorMux charges.

**(h) Usage/metrics dashboard URL** — `https://app.tensormux.com/` returns **HTTP 302**
(a real, live, login-gated app — almost certainly the account/usage dashboard implied by
the homepage's "Usage dashboard" feature for paid tiers). `https://dashboard.tensormux.com/`
returns 502 (not the right host). I did **not** log in, so I can't confirm exactly what's
on `app.tensormux.com` — treat "app.tensormux.com is the dashboard" as a strong inference,
not a fully confirmed fact.

**(i) `usage.prompt_tokens`/`completion_tokens` in responses** — **UNKNOWN**, not directly
confirmed (no working key), but required by OpenAI-compatibility and near-certain.

### GLM-4.7-Flash (the model itself, per Z.ai's own docs)
- Vendor: Z.ai (Zhipu). Z.ai's own hosted endpoint is `https://api.z.ai/api/paas/v4`
  (**different host from TensorMux** — TensorMux is a separate operator proxying to some
  backend, per its `owned_by: "aibrix"` tag, not necessarily Z.ai's own infra).
- Context window: confirmed **131,072 tokens** on Cloudflare Workers AI's model card and
  **200K tokens (with up to 131,072 output tokens)** on a third-party aggregator
  (EmpirioLabs); Microsoft Foundry's catalog shows "202.752k." These three numbers
  disagree slightly by source — treat context window as **~131K-200K depending on
  provider/quantization**, not a single confirmed number.
- Function calling: **confirmed** by Z.ai's own docs (`docs.z.ai/guides/capabilities/function-calling`,
  `docs.z.ai/guides/tools/stream-tool`) — the GLM family supports `tools` (function type),
  `tool_choice` (**only `"auto"` is supported** — not `"required"`/`"none"` per Z.ai's own
  page), and returns `tool_calls` with `function.name`/`function.arguments`/`id`, exactly
  OpenAI-shaped. Z.ai's `response_format` only supports `text` or `json_object` (confirmed
  earlier in this research) — **no `json_schema` support** per Z.ai's own docs.
- Z.ai's own list price for GLM-4.7-Flash: **UNKNOWN** — I could not get Z.ai's pricing
  page to render actual figures (it returned only a stub/JS shell on livecrawl). A
  third-party reseller (Cloudflare Workers AI) lists **$0.06 / M input, $0.40 / M output**
  for `@cf/zai-org/glm-4.7-flash` — that is **Cloudflare's resale price, not Z.ai's own
  list price**, and almost certainly not TensorMux's price either. Do not quote this as
  "the" price in slides without citing Cloudflare specifically.

### Copy-paste setup (TensorMux)
```bash
export TENSORMUX_API_KEY="sk-..."       # from your TensorMux account
```
```python
from openai import OpenAI

client = OpenAI(
    api_key="TENSORMUX_API_KEY_HERE",   # os.environ["TENSORMUX_API_KEY"]
    base_url="https://api.tensormux.com/v1",
)

resp = client.chat.completions.create(
    model="glm-4-7-flash",              # confirmed exact id via GET /v1/models
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```
Rate limits, `usage` field shape, dashboard URL, and TensorMux's own price: **UNKNOWN** —
budget conservatively and watch your first few responses for a `usage` block before relying
on it.

---

## 2. Neatlogs (docs.neatlogs.com, PyPI `neatlogs`)

**(a) pip install / current version** — `pip install neatlogs`. Current PyPI version
confirmed via the PyPI JSON API directly (more reliable than cached `pip index`):
**1.4.21** (`requires_python: <3.14,>=3.10`).

**(b) Exact init call + env vars** (from `docs.neatlogs.com/sdk/python`, quoted):
```python
import os
import neatlogs

neatlogs.init(api_key=os.environ["NEATLOGS_API_KEY"], workflow_name="my-first-app")
```
Env vars confirmed from docs/GitHub README: `NEATLOGS_API_KEY` (required for export —
"if unset, spans are created but not exported"), and optionally `NEATLOGS_ENDPOINT`
(default `https://ingest.neatlogs.com`, but the GitHub README's own example shows the
default as `https://staging-cloud.neatlogs.com` — the two docs pages disagree slightly on
the literal default string; treat the ingest host as either of those unless you set
`endpoint=` explicitly).

**(c) Manual span/trace creation with custom attributes** — confirmed exact API, two forms:
```python
# Decorator form — wraps a whole function as a step
@neatlogs.span(kind="TOOL", name="lookup_price")
def lookup_price(ticker: str) -> float:
    ...

# Context-manager form — for a block, not a whole function
with neatlogs.trace("Copilot chat", kind="WORKFLOW") as root:
    root.set_attribute("neatlogs.workflow_name", "Copilot chat")  # custom attribute
    ...
```
`@neatlogs.span(kind=...)` only accepts: `WORKFLOW`, `AGENT`, `CHAIN`, `TOOL`, `RETRIEVER`,
`EMBEDDING`, `GUARDRAIL`, `MCP_TOOL`. `LLM`, `RERANKER`, `VECTOR_STORE` must be created via
`with neatlogs.trace(name, kind="...")` instead (per `docs.neatlogs.com/sdk/span-kinds`).

**(d) Auto-instrumentation of `openai`/`httpx`** — confirmed: pass
`instrumentations=["openai"]` to `neatlogs.init()` (must be called **before** importing
`openai`), or call `client = neatlogs.wrap(OpenAI())` on a specific client instance (no
import-order requirement for `wrap()`). `docs.neatlogs.com/sdk/supported-libraries` lists
valid instrumentation keys; **I did not find `httpx` explicitly listed** among them in the
pages I fetched — **UNKNOWN whether raw `httpx` calls are auto-instrumented** (the
documented libraries are AI-SDK-specific: `openai`, `crewai`, `dspy`, `chromadb`, etc., not
generic HTTP clients).

**(e) Flush before process exit** — confirmed exact calls:
```python
neatlogs.flush()     # forces immediate export of buffered spans
neatlogs.shutdown()  # stops the background export thread cleanly
```
Docs explicitly warn: call both at the end of short scripts/CLI tools; for long-running
servers, call `init()` once and do `flush()`+`shutdown()` only once at server shutdown
(e.g. a FastAPI `lifespan` handler), not per-request.

**(f) Shareable/public trace URLs** — **UNKNOWN**. Not mentioned in any Neatlogs doc page
I fetched (Traces, Analytics, dashboard quickstart, Python/Go/Browser SDK pages). Neatlogs
is described as a "collaborative debugging workspace" with comments/shared traces on the
marketing homepage, but no explicit "public URL" or "share link" feature/API was
documented on the pages I could reach.

**(g) Free tier limits** — **UNKNOWN specific numbers**. `neatlogs.com` homepage states
"free to start, no credit card required." `neatlogs.com/terms` §7 confirms "certain
features require a paid subscription" and pricing can change with 30 days' notice, but
gives no numeric trace/token quota for the free tier.

**(h) OTLP endpoint + raw OpenTelemetry SDK exporter** — confirmed, quoted from
`docs.neatlogs.com/sdk/http-injection`:
> "For an OpenTelemetry exporter, point it at the OTLP route `POST /v1/traces` instead...
> Point any OpenTelemetry gRPC trace exporter at Neatlogs and your spans are ingested — no
> Neatlogs SDK required."

So: base URL `https://ingest.neatlogs.com`, OTLP path `/v1/traces` (gRPC-style OTLP, per
docs — "that path speaks OTLP protobuf (and gRPC)"). **Exact required headers (e.g.
whether the API key goes in an `Authorization` header, a custom header, or as gRPC
metadata) are UNKNOWN** — no page I fetched showed a concrete header example for a raw
OTel SDK exporter (only the Neatlogs-SDK-managed path documents `api_key=`).

### Copy-paste setup (Neatlogs)
```bash
pip install neatlogs   # 1.4.21 confirmed current on PyPI
export NEATLOGS_API_KEY="your-api-key"
```
```python
import os
import neatlogs

neatlogs.init(
    api_key=os.environ["NEATLOGS_API_KEY"],
    workflow_name="hackathon-app",
    instrumentations=["openai"],   # call init() before importing openai
)

from openai import OpenAI
client = neatlogs.wrap(OpenAI())   # or rely on instrumentations=["openai"] above

@neatlogs.span(kind="WORKFLOW", name="main_run")
def main():
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "hello"}],
    )
    return resp.choices[0].message.content

print(main())
neatlogs.flush()
neatlogs.shutdown()
```

---

## 3. OpenAI GPT-5 nano

**Model id** — confirmed exactly **`gpt-5-nano`** (a "snapshot/alias" style id — the docs
also list dated snapshots under it), from `developers.openai.com/api/docs/models/gpt-5-nano`.

**Context window / output** — confirmed: **400,000-token context window**, **up to
128,000 max output tokens**, knowledge cutoff **May 31, 2024**.

**Pricing** — the fetched model page showed pricing figures in a table that rendered as
plain numbers without clear column labels attached in the crawl: `$0.05`, `$0.005`,
`$0.40`. Based on standard OpenAI table layout (input / cached-input / output) this reads
as **$0.05 per 1M input tokens, $0.005 per 1M cached-input tokens, $0.40 per 1M output
tokens** — but because the crawled text lost the column headers, **treat this exact
mapping as reasonably confident, not fully certain**; verify against the live
`platform.openai.com/docs/pricing` page before printing it on a slide as gospel.

**Chat Completions vs. Responses API** — confirmed **both work**:
`developers.openai.com/api/docs/guides/function-calling` shows the identical `tools`
workflow via `openai.chat.completions.create(...)` **and** `openai.responses.create(...)`.
GPT-5-nano is listed with "Chat Completions" support explicitly (`v1/chat/completions`)
on its own model page — so **Chat Completions works, you are not forced onto Responses**.

**`tools`/function calling** — confirmed supported generically across the GPT-5 family via
the function-calling guide (both APIs). Not a nano-specific isolated confirmation, but the
guide makes no exception for nano, and nano's own model page lists Chat Completions as a
supported endpoint.

**`response_format` json_schema** — confirmed the **mechanism** exists on Chat Completions
(`response_format={"type": "json_schema", "json_schema": {...}}` / `.parse()` helper) via
OpenAI's structured-outputs docs — again shown with a general/example model id in the
docs, not nano-specifically isolated, but no documented exception excludes nano.

**Rate limits** (from the gpt-5-nano model page, by usage tier — RPM / TPM / batch queue
tokens):
| Tier | RPM | TPM | Batch queue limit |
|---|---|---|---|
| Free | Not supported | — | — |
| Tier 1 | 500 | 200,000 | 2,000,000 |
| Tier 2 | 5,000 | 2,000,000 | 20,000,000 |
| Tier 3 | 5,000 | 4,000,000 | 40,000,000 |
| Tier 4 | 10,000 | 10,000,000 | 1,000,000,000 |
| Tier 5 | 30,000 | 180,000,000 | 15,000,000,000 |

Note: **gpt-5-nano is not available on the Free usage tier** at all per this table.

### Copy-paste setup (OpenAI GPT-5 nano)
```bash
export OPENAI_API_KEY="sk-..."
pip install openai
```
```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY

resp = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "hello"}],
    tools=[...],            # confirmed supported on Chat Completions
    response_format={"type": "json_schema", "json_schema": {...}},  # mechanism confirmed
)
print(resp.choices[0].message.content)
```

---

## 4. Frankfurter (frankfurter.dev)

Confirmed live against `https://api.frankfurter.dev/v1/` and the docs page content itself.

- **Base URL (v1):** `https://api.frankfurter.dev/v1/` — confirmed working, no API key.
- **Parameter names:** `base` and `symbols` — confirmed **both from docs text and live
  test**: `?base=USD&symbols=EUR` works exactly as documented. `from`/`to` are **not**
  Frankfurter v1 query parameter names (I tested `?from=USD&to=EUR` live — it silently
  ignored both and just returned the default EUR-base rates, i.e. `from`/`to` are **not
  valid params on v1**, contrary to a name pattern some other FX APIs use).
- **Historical / time-series syntax** (from docs, quoted):
  - Specific date: `GET /v1/1999-01-04`
  - Range: `GET /v1/2000-01-01..2000-12-31`
  - Open-ended range to present: `GET /v1/2024-01-01..`
- **Weekend/holiday behavior** — confirmed both by docs text ("Fetch the latest working
  day's rates, updated daily around 16:00 CET") and by a live test: querying a Saturday
  date (`/v1/2026-09-05`) returned data dated the prior Friday (`2026-09-04`) — confirming
  the documented "most recent working day" fallback.
- **`amount` parameter** — **not documented** on the v1 docs page (the docs instead show
  computing the converted amount client-side in JS: `amount * data.rates[to]`). However, I
  **live-tested** `?amount=100&base=USD&symbols=EUR` and the API **did** honor it,
  returning `{"amount":100.0,"base":"USD",...,"rates":{"EUR":86.04}}` — so `amount` **is
  accepted server-side even though it's undocumented on the v1 page** as an input param.
  Treat it as working-but-not-officially-documented for v1.
- **Rate limit** — docs/behavior confirm **no API key required and no published quota**;
  Frankfurter's own docs state usage is "rate-limited to prevent abuse" without giving a
  specific number (no RPM/RPS figure found anywhere on frankfurter.dev).
- Note: v1 is explicitly marked **"superseded by v2"** on the docs ("will continue to
  work"), and v2 covers 201 currencies from 84 sources vs. v1's ECB-only set — if the team
  wants the newer/larger data set they should look at `https://frankfurter.dev/` (v2) docs,
  not `/v1/`.

### Copy-paste setup (Frankfurter)
```bash
# No API key needed, no install beyond a normal HTTP client
```
```python
import requests

resp = requests.get(
    "https://api.frankfurter.dev/v1/latest",
    params={"base": "USD", "symbols": "EUR", "amount": 100},
)
data = resp.json()
print(data)  # {'amount': 100.0, 'base': 'USD', 'date': '...', 'rates': {'EUR': ...}}
```

---

## 5. Textual

Confirmed via direct PyPI JSON API queries (authoritative — more reliable than cached
`pip index` output, which I found to be stale in this same investigation):
- `textual`: current version **8.2.8**
- `textual-serve`: current version **1.1.3**, a **separate PyPI package** — Textual's own
  install/getting-started flow instructs installing it as an explicit additional step
  ("Then install textual-serve from PyPI"), i.e. **`textual serve` is NOT bundled into the
  base `textual` package** — you must `pip install textual-serve` separately to get the
  `textual serve` / `textual-web`-style browser serving capability.

### Copy-paste setup (Textual)
```bash
pip install textual          # 8.2.8
pip install textual-serve    # 1.1.3 — required separately for `textual serve`
```
```python
from textual.app import App, ComposeResult
from textual.widgets import Label

class HelloApp(App):
    def compose(self) -> ComposeResult:
        yield Label("Hello, hackathon!")

if __name__ == "__main__":
    HelloApp().run()
```

---

## 6. Anthropic Claude Sonnet (as of Sept 6, 2026)

**Model id** — confirmed **`claude-sonnet-5`**, a dateless "pinned snapshot" id (per
`platform.claude.com/docs/en/models/sonnet-5/overview` and the models-overview page). Same
literal string is used on the Claude API, Google Cloud, and Microsoft Foundry; Amazon
Bedrock prefixes it: `anthropic.claude-sonnet-5`.

**Context window / output** — confirmed **1M token context window, 128K max output**.

**Pricing — flag a real discrepancy in Anthropic's own docs:**
- The model overview page and Anthropic's marketing page (`anthropic.com/claude/sonnet`)
  both explicitly and currently state: **"Sonnet 5 is available today at $2 per million
  input tokens and $10 per million output tokens."** (`anthropic.com/claude/sonnet`, live
  fetch.)
- However, a full pricing table on `platform.claude.com/docs/en/about-claude/pricing` still
  shows two separate rows: **"Claude Sonnet 5 through August 31, 2026" at $2/$10**, and
  **"Claude Sonnet 5 starting September 1, 2026" at $3/$15** — which, taken literally and
  given today's date (Sept 6, 2026), would imply the price has already stepped up to
  $3/$15.
- These two primary-source pages **directly contradict each other** as of today. The
  overview/marketing pages read as more current/authoritative (they say "available today"
  in the present tense, and the "What's new in Claude Sonnet 5" page also just says "$2 per
  million input tokens and $10 per million output tokens" with no date qualifier at all).
  My working conclusion, pending the team's own re-check right before demo day, is that
  the scheduled Sept 1 increase table row is **stale** and the effective current price is
  **$2 input / $10 output per MTok** — but I cannot fully resolve this contradiction from
  docs alone, since two official pages disagree. **Recommend the team re-fetch
  `platform.claude.com/docs/en/about-claude/pricing` themselves right before finalizing any
  cost slide**, and quote whichever number that page shows at that moment.
- Other confirmed cache/batch pricing (both tables agree on these): 5-minute cache write
  $2.50/MTok, 1-hour cache write $4/MTok, cache read $0.20/MTok, Batch API 50% off input
  and output — at the $2/$10 base rate. (These would presumably also shift if the $3/$15
  row is in fact now live.)

### Copy-paste setup (Anthropic Claude Sonnet 5)
```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```
```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

resp = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.content[0].text)
```

---

## Summary of items marked UNKNOWN (do not present these as fact)

- TensorMux: `tools`/`tool_choice`/`tool_calls` support on `api.tensormux.com` specifically
  (only inferred via the underlying model + OpenAI-compat marketing claim)
- TensorMux: `response_format` support/shape on `api.tensormux.com`
- TensorMux: documented numeric rate/concurrency limits for the hosted API
- TensorMux: exact dashboard URL contents (app.tensormux.com inferred but not logged into)
- TensorMux: its own published list price for glm-4-7-flash (not found; likely not public)
- TensorMux: `usage.prompt_tokens`/`completion_tokens` presence (not directly tested)
- GLM-4.7-Flash exact context window (sources disagree: 131K vs 200K vs ~202K)
- Z.ai's own list price for GLM-4.7-Flash (only a third-party reseller price was found)
- Neatlogs: whether raw `httpx` calls are auto-instrumented (not found in docs)
- Neatlogs: shareable/public trace URLs (feature not documented in pages I reached)
- Neatlogs: exact numeric free-tier limits
- Neatlogs: exact headers required for a raw OpenTelemetry SDK exporter against the OTLP route
- OpenAI gpt-5-nano: exact price-to-column mapping (table headers lost in the crawl; strongly implied but not 100% certain)
- Anthropic Claude Sonnet 5: whether the $2/$10 or $3/$15 rate is actually in effect today — Anthropic's own pages disagree; re-check immediately before quoting a number publicly
