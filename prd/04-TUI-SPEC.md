# 04 — TUI Specification (Textual)

## 0. Decision: Textual, not BubbleTea

We considered Go + BubbleTea (proven fluency, one binary) versus Python + Textual. **Textual wins for this build**, for three reasons that outrank familiarity:

1. **One language, one process, zero cross-language seam.** The engine is necessarily Python (LLM SDKs, dataset tooling). A Go TUI would mean the JSON contract is the *only* thing holding two codebases together across a 30-hour clock. With Textual, the contract still exists (engine and TUI still only talk via the run directory — see `01 §1`) but a mismatch is a Python test failure, not a cross-repo debugging session at hour 26.
2. **Textual's widget library is genuinely richer for this content.** `DataTable` (sortable, styled cells), `Tree`, `Sparkline`, `RichLog`, `ProgressBar`, `TabbedContent`, CSS-like styling, and `textual serve` to render the same app in a browser for large-font screen recording. BubbleTea would need Bubbles + lipgloss + hand-rolled tables.
3. **Coding agents are very strong at Textual.** It's declarative, well-documented, Python — an AO worker will produce a working screen from this spec quickly. The design reference from Figma maps onto Textual CSS naturally.

The differentiator was always "it's a TUI, not a web dashboard." That survives intact.

## 1. Principles

- **Read-only.** The TUI never calls the engine or an LLM. It reads `events.jsonl` and `state.json`. Replay and live are indistinguishable.
- **Legible at 1080p at 100% zoom.** Minimum effective font: whatever `textual serve` renders at 18px, or a terminal at ≥16pt. Never more than ~110 columns of content. Test a recording at hour 12.
- **The ablation table is the hero.** Everything else supports it.
- **Colour carries meaning, consistently:** green = load-bearing / pass, red = witness / fail, amber = uncertain, dim = pruned / dead branch, cyan = current generation, magenta = baseline.

## 1b. v3 additions (read with §2–§3)

- **Header** gains `run 2/2 · lessons loaded 3` (or `run 1 · lessons 0`). Task label is the pack name plus a short human string: `fx_recon_b · FX revaluation`.
- **Lessons pane** (new, right column under Metrics or as a tab `l`): one row per lesson — `kind badge · text · born run/gen · evidence ▸`. New rows animate in on `lesson.written`. In run 2, rows loaded at start are marked `● loaded`. Clicking evidence opens the Inspector on the originating case with the tool response highlighted (`requested_date 2026-04-04 → rate_date 2026-04-02`). **This highlight is the single most explanatory frame in the demo.**
- **Cases grid** cells show per-invoice sub-results on hover/focus (`7/8 invoices ✓ · INV-2310 ✗ (fee)`).
- **Metrics strip** adds `calls/case` and `rel³` (final gen only).
- **Compare strip** (run 2 only, reads `compare.json`): `run1 g0 0.55 · run2 g0 0.85 │ calls/case 17 → 6 │ gens to plateau 5 → 2`. Stretch: a full compare tab.
- Ablation table unchanged: it's the hero. Columns per `03 §8`. Verdict vocabulary LOAD-BEARING / WITNESS / HARMFUL / UNCERTAIN.

## 2. Layout (single screen, 4 regions)

```
┌─ OCCAM ─ fx_recon_a · FX revaluation ─ run 1 · lessons 0 ─ gen 1/6 ─ ⏵ REPLAY ──────────┐
│ LINEAGE                      │ ABLATION — g0 · 5 roles · LOO approx.                     │
│ ● g0  5 roles  0.55  $0.063  │ role            just.    infl   95% CI      div  cost verd │
│ └─◉ g1  4 roles 0.55  $0.060 │ Ledger Parser   ctx_iso  +0.40 [+.20,+.60]  .90  14% LOAD │
│    ✗ Verifier (pruned)       │ Rate Fetcher    parallel +0.50 [+.30,+.70] 1.00  54% LOAD │
│                              │ FX Calculator   control  +0.50 [+.30,+.70] 1.00  19% LOAD │
│ ARCHITECTURE g1              │ Verifier        verify   +0.00 [−.10,+.10]  .10   6% WITN │
│ task ─▶ Parser ─▶ Fetcher ─▶ │ Reporter        control  +0.10 [−.10,+.30]  .30   8% UNC  │
│   Calculator ─▶ Reporter     │ ablated 10/20 · noise 0.10 · SF 0.87 · 1 witness          │
│   ~~Verifier~~               ├───────────────────────────────────────────────────────────┤
│                              │ CASES 11/20  ✓✓✗✓✗✓✓✗✓✓ ✗✓✓✗✓✗✓✓✗✓                       │
│ LESSONS · 2                  │  fxa_007 · 6/8 invoices ✓ · INV-2291 ✗ (holiday) · INV-2310 ✗ (fee) │
│ [tool] fx_rate: rate_date is │├───────────────────────────────────────────────────────────┤
│  the actual ECB day …  g0    ││ METRICS pass 0.55 ▁▁  cost $0.063 ▇▇  lat 22.9s ▇▇  calls/case 17.4  SF 0.87 ▅▆ │
│ [rule] bank fee: add back g0 ││ vs CoT-SC(k=3) pass 0.40 · Δpass +0.15 · cost ×1.17                            │
├──────────────────────────────┴───────────────────────────────────────────────────────────┤
│ DIAGNOSIS ▸ g0: 9 failures — 6 holiday/weekend, 5 bank-fee (overlap 2). Rate Fetcher    │
│ passed 2026-04-02 rates as "April 4" without reading rate_date. Verifier: WITNESS.       │
│ ▸ prune Verifier · lesson written [tool] fx_rate · lesson written [rule] bank fee        │
└──────────────────────────────────────────────────────────────────────────────────────────┘
 ←/→ gen · a ablation · c cases · d diagnosis · l lessons · i inspect · space pause · . step · q
```

## 3. Widgets

### 3.1 Header
`Header` with: app name, task name, `gen i/N`, mode badge (`▶ live` / `⏵ replay ×4` / `■ done`), elapsed, spend so far.

### 3.2 Lineage (`Tree`)
- One node per generation. Label: `g{n}  {roles} roles  {pass:.2f}  ${cost:.2f}`.
- Reverted generations render as `○ g3' split → reverted` in dim.
- Current generation `◉` cyan. Best-so-far marked `★`.
- Selecting a node switches the Architecture, Ablation and Cases panels to that generation. `←/→` step generations.

### 3.3 Architecture (`Static` with Rich renderable)
Left-to-right box-drawing of the DAG for the selected generation. Roles as boxes with name and justification tag. Pruned roles (relative to parent) drawn with strikethrough for 2 seconds after the `mutation.applied{prune}` event, then removed — **this is the shrink animation**. Keep it ASCII; no graph library.

### 3.4 Ablation (`DataTable`) — the hero
Columns exactly as in `03 §8`. Rows update in place as `ablation.role` events arrive (so the audience watches verdicts appear one by one). Verdict cell styled: `LOAD-BEARING` green bold, `WITNESS` red bold on dark red, `HARMFUL` red, `UNCERTAIN` amber. Cost column shows a small bar. Footer line: `ablated a/M · noise floor x · SF y · n witnesses`.
While ablation is running: a `ProgressBar` under the table, `ablating role 3/5 · case 7/10`.

### 3.5 Cases (`Static`)
Grid of ✓/✗ per case for the selected generation and variant (default `full`). Click/enter on a case opens the **Case Inspector** modal.

### 3.6 Case Inspector (`ModalScreen`)
Input (scrollable), expected, answer, pass/fail, then one collapsible per role showing its output and tool calls. If an ablation variant is selected in the table, show both `full` and `−r` answers side by side. This is the "failure inspector" from the demo.

### 3.7 Metrics strip
Five `Sparkline`s across generations: pass, cost, latency, SF, reliability; current value left of each. Second line: baseline comparison in magenta.

### 3.8 Diagnosis feed (`RichLog`)
Streams `diagnosis.emitted`, `mutation.applied`, `mutation.reverted`, and `log` events with a `▸` prefix and generation tag. Auto-scroll; `d` focuses it for scrollback.

### 3.9 Baseline panel (`b` toggles a `TabbedContent` tab)
Table of baseline runs per generation: k, pass, cost, matched cost.

## 4. Data flow

```
store.EventReader(run_dir).tail()  →  reducer.reduce(state, event)  →  app.post_message(StateChanged)
```
- On mount: load `state.json` if present (instant paint), then tail from `seq = state.last_seq + 1`.
- Replay mode: `EventReader` yields events from the file with `sleep(dt / speed)` between them, where `dt` is the real inter-event gap (capped at 2s so long ablations don't stall the demo). `--to-gen N` fast-forwards silently to the first event of generation N then plays normally — this is how the cold open lands on a populated ablation table.
- `space` pauses replay; `.` steps one event. Useful for recording.

## 5. Keybindings
`q` quit · `←/→` prev/next generation · `a` focus ablation · `c` cases · `d` diagnosis · `b` baseline tab · `i` inspect selected case · `space` pause/resume replay · `.` step one event · `+/-` replay speed · `?` help.

## 6. Textual specifics
- `App` subclass `OccamApp(run_dir, mode)`; one `Screen`; `ModalScreen` for the inspector.
- Styling in `occam/tui/occam.tcss`. Palette variables at the top so the Figma reference can be applied by editing ~10 lines.
- `textual serve "occam tui --run runs/demo"` for browser rendering during recording; set `--port` and record the browser tab at 1080p with a large font.
- Unit tests with `App.run_test()` + `Pilot`: mount against `fixtures/demo_run1`, assert the ablation table has 5 rows with expected verdicts after replay to g0's `ablation.completed`; mount `fixtures/demo_run2`, assert header shows `lessons loaded` and the compare strip is populated.

## 7. Design reference
A Figma export (PNG + tokens) will be placed at `design/` by the human team. Treat it as the visual target for spacing, palette and typography; treat *this document* as the source of truth for what information appears where. If they conflict on information content, this document wins.
