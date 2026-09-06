# 01 — Architecture

## 1. The one design decision that makes parallel work possible

**Engine and TUI share nothing but a directory on disk.**

```
runs/<run_id>/
  task.yaml            copy of the task pack manifest (provenance)
  events.jsonl         append-only; one JSON object per line; THE contract
  state.json           latest full snapshot, rewritten after every event batch
  generations/
    g000/architecture.json
    g000/results.jsonl         one line per case
    g000/ablation.json
    g001/...
  baseline/results.jsonl
  cache/                       LLM response cache (content-addressed)
```

- The **engine** only *writes* here. It never renders anything.
- The **TUI** only *reads* here. It tails `events.jsonl` (like `tail -f`) and rebuilds view state; on startup it can load `state.json` for an instant first paint.
- `occam replay <dir>` feeds a recorded `events.jsonl` to the TUI at a chosen speed. **The TUI cannot tell replay from live.** This is deliberate: it's how the TUI gets built before the engine exists, and it's the demo backup.

"Schema JSON" means exactly this: `schemas/events.schema.json` and `schemas/state.schema.json` (JSON Schema draft 2020-12) define every event and the snapshot. Both sides validate against them in tests. If the engine emits an event the schema doesn't know, the test fails. If the TUI expects a field the schema doesn't have, the test fails. That's the integration guarantee.

## 2. Core data model (`occam/core/`) — pydantic v2

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict            # JSON Schema
    # implementation looked up by name in occam/tools/registry.py

class Role(BaseModel):
    id: str                     # short, stable, e.g. "r_extract"
    name: str                   # human label, e.g. "Transaction Extractor"
    justification: Literal["parallel","context_isolation","verification","ensemble","control","unspecified"]
    model: str                  # key into models.yaml, e.g. "worker_fast"
    system_prompt: str
    tools: list[str]            # ToolSpec names this role may call
    inputs: list[str]           # ids of upstream roles (or "task" for the raw input)
    output_key: str             # name under which this role's output is stored
    memory: Literal["none","scratchpad","summary"] = "none"
    max_turns: int = 6          # tool-call loop cap

class Architecture(BaseModel):
    id: str                     # "g003"
    parent_id: str | None
    roles: list[Role]           # topologically sortable by `inputs`
    final_role: str             # role whose output is the answer
    control: Literal["llm","deterministic"] = "deterministic"   # how outputs are routed; see §4
    notes: str = ""             # architect's rationale (shown in TUI)

class Case(BaseModel):
    id: str
    input: str                  # the full prompt text given to the system
    expected: Any               # checker-specific
    meta: dict = {}

class Lesson(BaseModel):
    id: str
    kind: Literal["tool_note", "domain_rule"]
    text: str                   # ≤ 60 words, a general rule; NEVER a case-specific value (guard in diagnose.py)
    tool: str | None            # tool_note: which tool's description this is appended to
    evidence: dict              # {run_id, generation, case_ids: [..], trace_refs: [..]}
    born: dict                  # {run_id, generation}
    status: Literal["active", "retired"] = "active"

class CaseResult(BaseModel):
    case_id: str
    answer: str                 # normalised final answer
    passed: bool
    sub_results: dict[str, bool] = {}   # e.g. per-invoice correctness; feeds diagnosis
    tokens_in: int; tokens_out: int
    cost_usd: float
    latency_s: float
    per_role: dict[str, RoleTrace]     # role_id -> {tokens_in, tokens_out, cost_usd, billed_cost_usd, cost_label, latency_s, output, tool_calls}

class RunResult(BaseModel):
    architecture_id: str
    variant: str                # "full" | f"ablate:{role_id}" | "baseline:cot_sc"
    results: list[CaseResult]
    pass_rate: float; cost_usd: float; latency_s_mean: float; tokens: int
```

**Justification tags matter.** The Illusion paper shows the roles that survive scrutiny are those exploiting *parallelism*, *context isolation*, or *deterministic control*; the ones that collapse are *ensemble/debate/critic*. The architect must tag every role; the TUI shows the tag next to the ablation verdict. Expect `ensemble` and `verification` roles to be the witnesses. That correlation, shown live, is the story.

## 3. Event log — the contract (`schemas/events.schema.json`)

Every line: `{"ts": ISO8601, "run_id": str, "seq": int, "type": str, "data": {...}}`. `seq` is monotonic per run. Types and required `data` fields:

| type | data |
|---|---|
| `run.started` | `task{name,domain,n_cases,goal}`, `config{models,budget_usd,max_generations}`, `run_name`, `memory_ns`, `lessons_loaded: [Lesson]` |
| `lesson.written` | `generation`, `lesson` (full `Lesson`) — appended to `memory/<ns>/lessons.jsonl` at the same moment |
| `reliability.completed` | `generation`, `method:"pass3"`, `reliable_cases`, `n_cases`, `reliability_pass3` |
| `architecture.proposed` | `architecture` (full `Architecture`), `generation` |
| `execution.started` | `generation`, `variant`, `n_cases` |
| `execution.case` | `generation`, `variant`, `case_id`, `passed`, `cost_usd`, `latency_s` |
| `execution.completed` | `generation`, `variant`, `pass_rate`, `cost_usd`, `latency_s_mean`, `tokens`, `ci{lo,hi}` |
| `ablation.started` | `generation`, `roles: [role_id]`, `n_cases`, `case_ids: [case_id]`, `noise_rate` |
| `ablation.role` | `generation`, `role_id`, `influence`, `influence_ci{lo,hi}`, `divergence`, `cost_share`, `verdict: "load_bearing"|"witness"|"harmful"|"uncertain"` |
| `ablation.completed` | `generation`, `structural_fidelity`, `witnesses: [role_id]` |
| `baseline.completed` | `generation`, `method:"cot_sc"`, `k`, `pass_rate`, `cost_usd`, `latency_s_mean`, `matched_to_cost_usd` |
| `diagnosis.emitted` | `generation`, `text` (plain language, ≤ 600 chars), `failure_summary`, `chosen_mutation{type, target_role, rationale}` |
| `mutation.applied` | `generation` (new), `parent`, `type`, `target_role`, `diff` (human-readable) |
| `mutation.reverted` | `generation` (reverted), `restored_to`, `reason` |
| `metrics.snapshot` | `generation`, `pass_rate`, `cost_usd`, `latency_s_mean`, `tool_calls_per_case`, `structural_fidelity`, `reliability_pass3?`, `vs_baseline{pass_delta,cost_ratio}` |
| `run.completed` | `best_generation`, `summary` |
| `log` | `level`, `message` — for the diagnosis feed; never load-bearing |

Rules: events are **never** edited; new info = new event. `state.json` is derived purely from events (a pure function `reduce(events) -> State`, implemented once in `occam/store/reducer.py` and used by both engine and TUI — the only shared code path, and it's read-only).

## 4. Engine (`occam/engine/`)

### 4.1 `architect.py`
Input: task pack manifest (goal, tool manifest, answer format, 3 example cases) **plus the active lessons from `memory/<ns>/lessons.jsonl`**. Output: `Architecture` for g000.
- One LLM call (the `architect` model) with a strict JSON schema response.
- Prompt instructs: 3–6 roles, each with a `justification`, DAG wiring, deterministic control by default.
- **Lessons injection:** `tool_note` lessons are appended verbatim to the matching tool's description *before* the architect sees the manifest (the agent's self-authored tool docs). `domain_rule` lessons are given under a heading "Known conventions and facts about this task (learned in earlier runs)"; the architect is told to place them in the prompts of the roles that need them.
- **Do not** let the architect invent tools. It may only bind tools from the manifest.
- Also produces `notes` explaining why each role exists (shown in TUI).

### 4.2 `executor.py` — DAG executor
- Topologically sort roles. Run roles whose inputs are ready; roles with no mutual dependency run **concurrently** (asyncio, bounded by `models.yaml` rate limits).
- Each role runs a bounded tool-call loop (`max_turns`), using only its bound tools.
- `control="deterministic"`: outputs are passed by `output_key` into downstream prompts via a template — no LLM decides routing. `control="llm"`: a router role decides; allowed but discouraged (the paper's finding).
- Records a `RoleTrace` per role per case. **Cost accounting is per role** — this is what ablation's `cost_share` reads.
- **Cache:** every LLM call is keyed on `(model, messages, tools, temperature)`. Cache hits cost $0 and 0 latency-added but are recorded as cached. This is what makes ablation affordable (see `03` §3).
- Emits `execution.*` events.

### 4.3 `ablation.py` — see `03-ABLATION-AND-METRICS.md`
Knocks out one role at a time. Knock-out semantics: the role is removed from the DAG; its `output_key` is set to an empty sentinel and downstream prompts render a `[no input from <role>]` marker. Upstream roles are **not** re-run (cache hits). Only downstream roles recompute.

### 4.4 `diagnose.py`
Input: failed cases (inputs, answers, expected, **per-invoice `sub_results`**, per-role outputs **including raw tool responses** — this is where the agent can see `requested_date ≠ rate_date`), ablation table, generation history, active lessons. Output: `diagnosis.emitted` and **0–3 `lesson.written`**.
One LLM call (the `architect` model). Must pick exactly one mutation from the menu (`mutate.py`), with a target and a one-paragraph rationale. **Hard rule encoded in code, not prompt:** if any role has verdict `witness`, the chosen mutation must be `prune` of the highest-cost witness. Diagnosis may explain, not override.
**Lesson leak guard (code):** a proposed lesson is rejected if it contains an ISO date, any number with ≥4 digits, any invoice id, or any value within 1% of an expected total/sub-result in the pack. It must be a *rule*, not a *fact about a case*. Deduplicate against active lessons by cosine > 0.9. Max 5 lessons per run.

### 4.5 `mutate.py` — the closed menu
| type | effect | constraint |
|---|---|---|
| `prune` | remove role, rewire its consumers to its inputs | only on `witness`/`harmful` |
| `rewrite_prompt` | replace one role's system prompt | targeted by diagnosis |
| `rebind_tools` | add/remove a tool on one role | tools from manifest only |
| `split` | one role → two roles with `parallel` or `context_isolation` justification | new roles must survive next ablation or get reverted |
| `merge` | two roles → one | |
| `set_memory` | change one role's memory policy | |
| `set_retry` | change max_turns / retry policy on one role | |
| `collapse` | replace everything with a single role | always legal |

Each mutation produces a new `Architecture` with `parent_id` set and a human-readable `diff`.

### 4.6 `baseline.py`
Cost-matched CoT-SC: single role, same worker model, chain-of-thought prompt, sampled `k` times at temperature 0.7, majority vote on normalised answer. `k` chosen so that `cost ≈ current generation's cost` (round down, min 1, max 9). Emits `baseline.completed` with `matched_to_cost_usd`.

### 4.7 `loop.py`
```
run(task_pack, config, run_name, memory_ns):
  L = load_lessons(memory_ns)    ; emit run.started{lessons_loaded: L}
  A = architect(task, L)         ; emit architecture.proposed g000
  for g in 0..max_generations:
    R = execute(A, cases)        ; emit execution.*
    B = baseline(cost=R.cost)    ; emit baseline.completed
    T = ablate(A, cases_subset)  ; emit ablation.*
    emit metrics.snapshot
    if plateau(history) or budget_exhausted(): break
    D = diagnose(R, T, history, L) ; emit diagnosis.emitted; for l in D.lessons: append(memory_ns, l); emit lesson.written
    A' = mutate(A, D.mutation)   ; emit mutation.applied
    if D.mutation.type == "split":
        R' = execute(A'); T' = ablate(A')
        if any new role is witness: emit mutation.reverted; A' = A; continue
    A = A'
  R3 = pass3(best A, cases)     ; emit reliability.completed          # final generation only, 3 runs per case
  emit run.completed
```
Plateau: no improvement in `(pass_rate, -cost)` lexicographic for 2 generations.

### 4.8 `compare.py`
`occam compare run1 run2` → table: g0 pass, final pass, generations to plateau, total cost, tool calls/case at g0, pass³, lessons loaded/written. Also emitted as `compare.json` in run2's dir so the TUI compare tab (stretch) can read it.

## 5. LLM layer (`occam/llm/`)

- One client interface: `complete(model_key, messages, tools=None, response_schema=None) -> Completion{text, tool_calls, tokens_in, tokens_out, cost_usd, latency_s, cached}`.
- Providers behind it: OpenAI-compatible (TensorMux, OpenAI; any other OpenAI-shaped gateway is a `base_url` swap), Anthropic. All via `base_url` + key from `.env`.
- `models.yaml`:
  ```yaml
  worker_fast:   {provider: openai_compat, base_url: https://api.tensormux.com/v1, model: glm-4-7-flash, in_per_m: 0, out_per_m: 0, grant_equiv_in_per_m: 0.06, grant_equiv_out_per_m: 0.40, rpm: 60, supports_json_schema: false, tool_choice_modes: [auto]}   # TensorMux grant; equiv. rate = Cloudflare resale price, labelled
  worker_alt:    {provider: openai_compat, base_url: https://api.openai.com/v1, model: gpt-5-nano, in_per_m: 0.05, out_per_m: 0.40, rpm: 500}   # AI Grants India credits; second ablation lane; fallback worker
  architect:     {provider: anthropic, model: claude-sonnet-5, in_per_m: 2.00, out_per_m: 10.00}   # Anthropic pages disagree ($2/$10 vs $3/$15 from Sept 1); few calls, immaterial to demo — recheck before quoting
  ```
  Exact ids and prices are filled from `prd/10-SETUP-VERIFICATION.md` (checked against vendor docs), never from memory.
  Cost is computed client-side from token counts × rates. For granted tokens (`in_per_m: 0`) the TUI shows **list-rate-equivalent** cost using `grant_equiv_*` and labels it so. See `06-SPONSOR-INTEGRATIONS.md`. **Verify model ids and rates on day one.**
- Rate limiting per model key (token bucket). Retries with backoff on 429/5xx.
- Content-addressed disk cache under `runs/<run_id>/cache/` (and a global `~/.occam/cache/` for cross-run reuse).

## 6. Tools (`occam/tools/`)

Registry: `name -> (ToolSpec, callable)`. Tools available to candidate agents are declared per task pack (see `02`). Built-ins:
- `fx_rate(date, base, symbol)` and `fx_series(start, end, base, symbol)` — live HTTP to Frankfurter (`api.frankfurter.dev/v1`), disk-cached under `data/fx_cache/` (committed). Base descriptions are deliberately plain; learned `tool_note`s are appended at architect time. Records latency/bytes/status/cached per call.
- `python_exec(code) -> stdout` — subprocess with timeout, no network, restricted imports.
- `fan_out(subtasks: list[str]) -> list[str]` — runs the calling role's prompt on each subtask concurrently (bounded). The only multi-agent primitive available *inside* a role; the architect may also express parallelism structurally via DAG roles.

## 7. CLI (`occam/cli.py`, typer)

```
occam run     --task <pack_dir|name> --run-name <name> [--memory memory/<ns>] [--max-gens 6] [--cases 20] [--ablate-cases 10] [--pass3] [--out runs/]
occam compare <run_dir_1> <run_dir_2>          # run-over-run table; writes compare.json into run 2
occam lessons show|reset --memory memory/<ns>  # print lessons.md / clear (for a clean run 1)
occam tui     --run <run_dir>                # attach to live or finished run
occam replay  <run_dir> [--speed 4] [--to-gen 3] [--at <event.type>] [--pause]   # --at: fast-forward to first event of that type within --to-gen; --pause: start paused (static frame for recording)
occam task new --goal "..." [--n 10]         # stretch; drafts cases for review
occam task list / occam task show <name>
occam baseline --task <pack> --cost <usd>    # standalone
occam validate <run_dir>                     # schema-check events + reducer determinism
```

`occam run` prints nothing but a run_id and a one-line summary per generation; the TUI is the view. `occam run --tui` launches both.

## 8. Concurrency & failure

- Engine is `asyncio`; per-model semaphores from `models.yaml`.
- Any LLM failure after retries → the case is marked `passed=false` with `error` in trace; the run continues. Never crash a run for one case.
- Events are flushed to disk after every event (line-buffered). If the engine dies, the TUI shows everything up to the crash and `occam replay` still works.
