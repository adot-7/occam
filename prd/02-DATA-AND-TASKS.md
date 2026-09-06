# 02 — Data, Task and Tool (v3, final)

One task family. One real third-party tool. Two packs. This replaces every earlier domain plan.

## 1. The workflow: month-end FX revaluation of a multi-currency receivables ledger

**Who does this in real life:** the accountant at any company that invoices abroad. Once a month they take every open and recently-settled foreign-currency invoice, look up exchange rates for the relevant dates, and compute the FX gain or loss in the reporting currency. It's tedious, rate lookups are error-prone (weekends, holidays, wrong date), and it must tie out exactly. It is also exception #3 in the finance domain briefing and squarely inside "Office of the CFO" work — a nod to the sponsor without leaving Track 1.

**Our company:** an Indian software exporter. Reporting currency **INR**. Invoices in EUR, USD, GBP, JPY, AUD, SGD, CHF, CAD.

**One case = one ledger snapshot.** Input: valuation date + 6–10 invoices. Output: **total FX gain/loss in INR** (one number) plus a per-invoice breakdown.

### 1.1 Case format (`cases.jsonl`)
```json
{"id":"fxa_007",
 "input":"Valuation date: 2026-04-30. Reporting currency: INR.\nLedger:\n1. INV-2291 · ACME GmbH · EUR 12,400.00 · issued 2026-03-14 · SETTLED 2026-04-04, received USD 13,610.00\n2. INV-2302 · Kyoto Labs · JPY 1,850,000 · issued 2026-03-20 · OPEN\n3. INV-2310 · Harbor Ltd · GBP 8,000.00 · issued 2026-04-01 · SETTLED 2026-04-27, received GBP 8,000.00, net of bank fee INR 1,250.00\n... \nCompute the total FX gain/(loss) in INR as of the valuation date, and the gain/(loss) per invoice.",
 "expected":{"total_inr":-41872.35,"per_invoice":{"INV-2291":-12030.10,"INV-2302":3115.50,"INV-2310":-9822.75}},
 "meta":{"n_invoices":8,"has_weekend_or_holiday":true,"has_bank_fee":true,"has_cross_ccy":true,"valuation_date":"2026-04-30"}}
```
`expected.total_inr` is graded; `per_invoice` feeds diagnosis (which invoice went wrong tells the diagnoser *why*).

### 1.2 The formula (single source of truth — grader, reference implementation, and README all use this)

For invoice *i* with currency *C*, amount *A*, issue date *d₀*; reporting currency *R* = INR; valuation date *V*:

```
booked_i      = A · rate(C→R, d₀)

if OPEN:      value_i = A · rate(C→R, V)                                 (unrealised)
if SETTLED on d₁, received amount P in currency S, optional bank fee F (in R):
              value_i = P · rate(S→R, d₁) + F                              (realised; fee is a bank charge, not FX)

gain_i        = value_i − booked_i
total         = Σ gain_i                       rounded to 2 dp
rate(X→R, d)  = the rate Frankfurter returns for base=X, symbols=R on date d,
                i.e. the last ECB business day ≤ d (the response's `date` field is authoritative)
```

**Grader** `fx_total`: pass iff `|answer − expected.total_inr| ≤ max(5.00, 0.001·|expected.total_inr|)`. Also records per-invoice correctness (tolerance ₹1 or 0.1%) into `CaseResult.sub_results` for diagnosis. Answer is extracted from the last fenced JSON block: `{"total_inr": -41872.35, "per_invoice": {...}}`.

### 1.3 What the agent has to get right (these are the learnable lessons — all verified real)

| # | kind | the fact | how it shows up as a failure |
|---|---|---|---|
| L1 | tool_note | Frankfurter returns the **last ECB business day ≤ requested date**; the response `date` field is the actual rate date. Weekends *and* ECB holidays (e.g. **Good Friday 2026-04-03 → returns 2026-04-02**). | Agent assumes "Saturday → Friday" or ignores the date field; on holiday-adjacent dates it's silently wrong. Truth uses the API's own resolution, so the fix is "trust the response date," not "compute business days yourself." |
| L2 | tool_note | `fx_series(start, end, base, symbol)` returns every business day in one call. | g0 makes 16–20 single-date calls per case; latency and cost balloon. Learning L2 halves both. |
| D1 | domain_rule | "net of bank fee INR F": the fee is a bank charge; **add F back** before computing gain/loss. | Agent treats the fee as FX loss → per-invoice error exactly F. |
| D2 | domain_rule | Cross-currency settlement: book at invoice currency on issue date; realise at *received* currency on settlement date. | Agent converts the received USD at the EUR rate or at issue date. |
| D3 | domain_rule | Settled invoices are **realised** at settlement date — do not revalue them at the valuation date. Open ones are revalued at V. | Agent revalues everything at V. |

**Not a quirk (do not teach):** cross rates. Frankfurter serves `base=USD&symbols=INR` directly. The API's old host `api.frankfurter.app` 301-redirects to `api.frankfurter.dev/v1` — our tool wrapper handles it, so it's not agent-learnable; mention in README only.

## 2. The tool (`occam/tools/fx.py`)

Real HTTP to `https://api.frankfurter.dev/v1` (free, no key, ECB reference rates, ~30 currencies incl. INR). Two tools exposed to candidate agents, plus `python_exec`:

```python
fx_rate(date: str, base: str, symbol: str) -> {"requested_date": date, "rate_date": str, "base": base, "symbol": symbol, "rate": float}
    # GET /{date}?base={base}&symbols={symbol}

fx_series(start: str, end: str, base: str, symbol: str) -> {"base":..., "symbol":..., "rates": {"YYYY-MM-DD": float, ...}}
    # GET /{start}..{end}?base={base}&symbols={symbol}   — business days only

python_exec(code: str) -> str
```
- **Base tool descriptions are deliberately plain** ("Get the exchange rate for a date"). They say nothing about weekends/holidays or about the range endpoint. Lessons L1/L2, once learned, are appended to these descriptions in later runs — the agent rewrites its own tool docs.
- **Disk cache** keyed on the full request path, under `data/fx_cache/` and **committed to the repo**. Historical ECB rates never change, so the cache is permanent. First run hits the network (visible in Neatlogs); ablations, replays and the demo are offline.
- Records per call: latency, bytes, status, `cached`. Feeds `n_tool_calls` and `wasted_calls`.
- Verified 2026-09-06: `/2026-04-04?base=EUR&symbols=USD` → `{"date":"2026-04-02", ...}` (Sat → Thu, Good Friday skipped). `/2026-03-14?base=USD&symbols=INR` → `{"date":"2026-03-13","rates":{"INR":92.38}}`. `/2026-03-02..2026-03-06?base=EUR&symbols=USD` → 5 daily rates.
- **Param names are `base` and `symbols`. The legacy `from`/`to` names are silently ignored on v1** (live-tested: you get EUR→everything instead of an error). The tool wrapper must use `base`/`symbols`; a test asserts the returned `base` equals the requested one.
- No published rate limit. Be polite (≤5 concurrent); the cache makes this moot after g0.

## 3. Generating the packs (`scripts/gen_fx_cases.py`)

```
python scripts/gen_fx_cases.py --seed 7  --n 20 --out tasks/fx_recon_a
python scripts/gen_fx_cases.py --seed 11 --n 20 --out tasks/fx_recon_b     # held-out
```
- Invoice dates: issued 2026-01-05 … 2026-04-20; settlements up to 2026-04-30; valuation date 2026-03-31 or 2026-04-30 per case.
- Guaranteed mix per pack: ≥6 cases with a weekend/holiday date on a lookup (force some onto 2026-04-03/04/05/06 — Good Friday through Easter Monday), ≥5 with a bank-fee settlement, ≥5 with cross-currency settlement, ≥4 with JPY (large nominal, small rate), batch sizes 6–10.
- **Reference implementation** computes `expected` by calling the same cached `fx_rate` with the formula in §1.2. It is ~60 lines and lives in `occam/tasks/fx_reference.py`. Unit-tested against 3 hand-computed cases.
- `task.yaml`:
  ```yaml
  name: fx_recon_a
  domain: finance_ops
  goal: >
    Month-end FX revaluation: for a ledger of foreign-currency receivables, compute the total FX
    gain/(loss) in INR as of the valuation date and the gain/(loss) per invoice. Use the exchange
    rate tools for rates. Follow the company's ledger conventions as stated in each case.
  answer_format: 'End your reply with a fenced json code block containing {"total_inr": <number>, "per_invoice": {"<id>": <number>, ...}}'
  tools: [fx_rate, fx_series, python_exec]
  checker: fx_total
  examples: 3
  memory: memory/fx_recon          # lessons namespace shared by both packs
  ```
- Both packs committed. `data/fx_cache/` committed after generation so the demo needs no network.

## 4. Two runs — what "run 2" means

```
occam run --task fx_recon_a --run-name run1        # memory/fx_recon/lessons.jsonl is empty
occam run --task fx_recon_b --run-name run2        # architect reads the lessons run1 wrote
```
Run 1 learns the hard way: g0 fails the holiday cases and the fee cases, ablation prunes the witness, diagnosis fixes the fetcher prompt **and writes lessons L1, D1 (and later L2)** to `memory/fx_recon/lessons.jsonl`. Run 2 is a *different* ledger (held-out pack) — the architect's very first proposal already carries L1/L2/D1 in the fetcher's tool descriptions and role prompts. Run 2's g0 passes most of what run 1's g0 failed, uses fewer calls, and plateaus in fewer generations. **That gap — run1.g0 vs run2.g0 on unseen data — is the "gets better over time" evidence.** `occam compare run1 run2` prints it.

## 5. Reliability = pass³
On the **final generation only**, each case is run 3× (temperature 0; hosted models still drift). A case is *reliable* iff 3/3 pass. `reliability_pass3 = reliable / N`. Reported next to pass rate. Noise floor (`03 §4`) is still measured for ablation; pass³ is the judge-facing number.
