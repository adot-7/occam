# 02 — Data, Tasks and Domains

## 1. What "a task" is to Occam

The brief says the system receives "a goal, available tools, and a way to evaluate success." We make that literal. A **task pack** is a directory:

```
tasks/<name>/
  task.yaml
  cases.jsonl
```

`task.yaml` (validated by `schemas/task.schema.json`):
```yaml
name: smfr_2inv
domain: financial_reasoning           # free label, shown in TUI
goal: >
  Given historical stock prices for several companies and each investor's transaction
  history, determine the dates on which each investor could sell their remaining holding
  to reach the target profit, then name the investor(s) who satisfy the aggregation rule.
answer_format: >
  A JSON list of investor names, e.g. ["Rachel"], or [] if none qualify.
tools: [python_exec, lookup_price, list_transactions]     # names from occam/tools/registry.py
checker: json_set_equal                                   # see §2
examples: 3                                               # how many cases the architect sees
source: {kind: huggingface, repo: the-illusion-of-multi-agent-advantages/smfr-dataset, file: balanced_dataset_single_2_fixed.jsonl, license: CC-BY-4.0}
```

`cases.jsonl` — one `Case` per line:
```json
{"id":"smfr2_0007","input":"<full problem text>","expected":["Rachel"],"meta":{"investors":2,"target_pct":1.3,"aggregation":"latest","price_type":"Close"}}
```

**A "domain" is just a task pack.** "Multiple distinct domains" = run Occam on three packs that look nothing alike. The system code does not change between them; only `task.yaml` and `cases.jsonl` do. That's the whole point and it's what we say to judges.

## 2. Checkers (`occam/tasks/checkers.py`)

Pure functions `(answer: str, expected: Any) -> bool`. Registered by name.

| name | behaviour | used by |
|---|---|---|
| `json_set_equal` | parse answer as JSON list; compare as sets, case-insensitive | SMFR |
| `numeric_exact` | extract last number in answer; compare to expected within 1e-6 | MGSM |
| `bfcl_ast` | parse the emitted function call(s); every required param must match one of the allowed values in `possible_answer`; optional params may be absent | BFCL |
| `exact` | normalised string equality | live tasks |
| `llm_judge` | architect model grades against a rubric string (stretch; never used for the three core domains) | live tasks |

Answer **normalisation** happens before checking: strip, lowercase (unless `case_sensitive`), collapse whitespace, extract the last fenced JSON block if present.

## 3. The three domains

### 3.1 SMFR — Synthetic Multi-Hop Financial Reasoning (primary)

**Why:** it's the diagnostic dataset from the paper our thesis rests on; it's *designed* to reward multi-agent structure (parallel across investors, context-heavy haystack); Expert-MAS took GPT-5 from 49.1% → 96.7% on it while automatic MAS didn't beat CoT-SC. So when Occam prunes roles here and still holds accuracy, the result is striking; and when it *keeps* a parallel-per-investor role because ablation shows it matters, that's the paper's Expert-MAS being rediscovered. And it's finance — the sponsor's world.

**Source:** HuggingFace `the-illusion-of-multi-agent-advantages/smfr-dataset`, CC-BY-4.0.
Files: `balanced_dataset_single_{2,3,4,5,6}_fixed.jsonl` (96/96/104/152/140 samples by investor count) + `_validate_fixed.jsonl` (16).
**Use the 2- and 3-investor files.** Problems are ~13k chars (≈3.5k tokens) of price tables + transactions; 5–6-investor ones are far longer and blow the free-tier TPM.

**Record shape (verified from the actual file):**
```json
{
  "problem": "Here is some data on the stock prices of a few companies... <30-day OHLC tables for B companies> ... <dated transaction list> ... Each investor has completed several transactions and holds shares in one remaining common stock. Based on when they could sell these remaining shares to achieve at least 1.3% overall portfolio profit, who has the latest possible sell date ... if all transactions were made at closing prices?",
  "answer": {
    "investor_dates": {"Rachel": ["December 30, 2025", "..."], "Patricia": ["January 05, 2026", "..."]},
    "comparison": {"Rachel": "January 26, 2026", "Patricia": "..."},
    "answer": ["Rachel"]
  },
  "cot": "...",
  "metadata": {"num_instances": 2, "aggregation_op": "latest_date",
               "task_params": {"price_type": "Close", "question_type": "reverse_target_sell", "target_percentage": 1.3, "num_distractors": 2},
               "breadth": 2, "depth": 2, "updatable_data": {"haystack": {...}}}
}
```

**Adapter (`occam/tasks/adapters/smfr.py`):**
- `input` = `problem` verbatim.
- `expected` = `answer.answer` (list of names; `[]` means "None").
- `meta` = investors, target_pct, aggregation, price_type.
- **Tools:** the adapter also parses the haystack into structured data attached to the case (`case.meta["_haystack"]`) so `lookup_price(company, date, price_type)` and `list_transactions(investor)` can answer from it. This turns SMFR into a genuine tool-use task rather than a 3.5k-token reading test, and it gives roles a reason to exist (an extractor role that calls tools vs a reasoner role that computes). `python_exec` lets a role do the P&L arithmetic in code — which is what the paper's Expert-MAS did.
- Pack: 20 cases sampled with fixed seed, balanced across `question_type` and `aggregation`. Ablation subset: 10 of them.
- Expected difficulty: CoT-SC on a fast open model will land ~20–40%. Good — leaves room to show improvement without saturating.

### 3.2 BFCL — Berkeley Function-Calling Leaderboard, `simple_python` (tool calling)

**Why:** a totally different task shape (one-shot structured tool call, no reasoning chain), deterministic grading, no execution needed. Cheap and fast. Shows "distinct domain" unambiguously.

**Source:** `github.com/ShishirPatil/gorilla`, `berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json` + `possible_answer/BFCL_v4_simple_python.json`. Apache-2.0.

**Record shape (verified):**
```json
{"id":"simple_python_0",
 "question":[[{"role":"user","content":"Find the area of a triangle with a base of 10 units and height of 5 units."}]],
 "function":[{"name":"calculate_triangle_area","description":"...","parameters":{"type":"dict","properties":{"base":{"type":"integer",...},"height":{...},"unit":{"type":"string",...}},"required":["base","height"]}}]}
```
possible_answer:
```json
{"id":"simple_python_0","ground_truth":[{"calculate_triangle_area":{"base":[10],"height":[5],"unit":["units",""]}}]}
```

**Adapter (`occam/tasks/adapters/bfcl.py`):**
- `input` = the user message. The `function` list becomes the case's **virtual tool manifest**: the executor exposes these as tools to roles but does **not** execute them — the emitted call is the answer.
- `expected` = `ground_truth`. Checker `bfcl_ast`: each required param's value must be in the allowed list; BFCL's `"type":"dict"` → treat as `object`.
- Pack: 40 cases, fixed seed. Ablation subset 15.
- Expected: strong models score 90%+. That's fine — here the story is **cost**: a 4-role architecture on a one-shot tool call is pure bloat, and Occam should collapse it toward one role. That's the "knows when to stay simple" beat.

### 3.3 MGSM — Multilingual Grade School Math, English split (reasoning)

**Why:** the ADAS/Illusion lineage's standard reasoning benchmark; text-only; numeric exact grading; trivially cheap. Third distinct shape (sequential reasoning, no parallelism to exploit — the paper's territory where CoT-SC wins).

**Source:** HuggingFace `juletxara/mgsm`, `en/test-00000-of-00001.parquet` (250 problems). CC-BY-SA-4.0. Fields: `question`, `answer_number`.

**Adapter (`occam/tasks/adapters/mgsm.py`):** `input` = question + "Give the final numeric answer on the last line as `Answer: <number>`." `expected` = `answer_number`. Checker `numeric_exact`. Tools: `[calculator]`. Pack: 40 cases, ablation subset 15.
Expected: multi-agent structures should show near-zero influence here — exactly the Illusion result — so Occam should prune to one role and match CoT-SC. **This is the domain where "it concluded multi-agent was pointless" is the correct outcome**, and we should say so.

### 3.4 Domain roster summary

| pack | shape | tools | checker | cases / ablate | expected story |
|---|---|---|---|---|---|
| `smfr_2inv` | long-context, parallel-per-investor, multi-hop | python_exec, lookup_price, list_transactions | json_set_equal | 20 / 10 | keeps justified parallel/extract roles, prunes ensemble/critic roles |
| `bfcl_simple` | one-shot structured call | virtual per-case | bfcl_ast | 40 / 15 | collapses toward one role; cost is the win |
| `mgsm_en` | sequential reasoning | calculator | numeric_exact | 40 / 15 | prunes to single agent, matches CoT-SC |

Three different *correct outcomes* from one system. That's a much better "multiple domains" story than "line went up three times."

## 4. Data preparation (`scripts/prepare_tasks.py`, run once before the window)

```
python scripts/prepare_tasks.py --all
  → downloads sources to data/raw/ (gitignored)
  → writes tasks/smfr_2inv/, tasks/bfcl_simple/, tasks/mgsm_en/ (committed: small, licensed)
  → prints per-pack token statistics (mean input tokens) so we can sanity-check TPM budgets
```
Commit the generated packs. The demo must not depend on HuggingFace being up.

## 5. The live "never seen before" task — how it actually works

The brief's "tasks it has never seen before" needs an honest mechanism, because a task pack needs eval cases and a judge shouting a goal doesn't come with any. Three options; we build **Option A as MVP** and **Option B as stretch**.

**Option A — held-out pack chosen live (MVP).**
Prepare 5–6 packs; only three are used during development. In the demo, the judge (or a coin flip on camera) picks one of the untouched packs. Occam has genuinely never run it. This is honest and zero-risk. Candidate extra packs, all cheap to prepare with the same adapters: `smfr_3inv` (harder SMFR), `bfcl_multiple` (pick the right function from several), `mgsm_de` (German split — same checker), a BFCL `parallel` slice.

**Option B — goal → drafted eval set → human approval (stretch).**
```
occam task new --goal "Classify GitHub issues by severity" --n 10
```
1. Architect model drafts 10 input/expected pairs + picks a checker (`exact` or `llm_judge`).
2. TUI shows the draft cases; human approves/edits/rejects each (Textual `DataTable` + edit modal).
3. Approved cases become `tasks/live_<slug>/`. Run proceeds as normal.
State on camera: "the eval set was drafted by the system and approved by us — the architecture is not hand-tuned." That's true and it's how a judge would expect it to work.

**Option C — judge brings cases.** Accept a pasted JSONL. Trivial to support (it's just the pack format); mention as possible.

## 6. Synthetic SMFR generator (fallback / extension, `occam/tasks/generators/smfr_gen.py`)

The paper describes the generator precisely: sample B tickers' 30-day OHLC (yfinance), sample investors with buy/sell pairs, pick target % / price type / aggregation, compute valid dates by brute force, render text. If we ever need more or fresher cases (or a `_hard` pack), this is ~150 lines and doesn't depend on HF. Not MVP.
