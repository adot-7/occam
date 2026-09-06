# 09 — Demo Shot List (v3)

Replay of real run logs via `textual serve` at 1080p, large font, edited. Each shot = one `occam replay` command from `scripts/record_demo.sh`. Total 3:00. Voiceover: Smallest.ai or human.

| # | t | shot | command | VO (draft) |
|---|---|---|---|---|
| 1 | 0:00–0:10 | Title: *"Automated agent design builds teams where half the agents do nothing — and every new team relearns the same mistakes. We fixed both."* | — | |
| 2 | 0:10–0:25 | Header: `fx_recon_a · FX revaluation · run 1 · lessons 0`. Architect draws 5 roles. One line of a real case visible (ledger with a Saturday settlement and a "net of bank fee" note). | `replay runs/demo_run1 --to-gen 0 --at architecture.proposed --speed 2` | "The job: month-end FX revaluation for an exporter — real invoices, real exchange rates from the ECB API. Occam designs a team to do it." |
| 3 | 0:25–0:45 | Cases fill: ✓✓✗✓✗… **11/20.** Click a red case → Inspector → tool response highlighted: **`requested_date 2026-04-04 → rate_date 2026-04-02`**. Hold 3s. | `--speed 6`, then `--pause`, `i` on case fxa_007 | "Eleven of twenty. Here's why: it asked for a Saturday rate. The API returned Thursday's — Friday was Good Friday — and the agent didn't read the date field." |
| 4 | 0:45–1:10 | **Ablation, 1×, one take.** Rows resolve: LOAD, LOAD, LOAD, **WITNESS** (Verifier), UNCERTAIN. Footer: noise 0.10, SF 0.87. | `--speed 1` through `ablation.completed` | "Then it removes each agent and re-runs. The Verifier changed the answer on one case in ten — our noise floor was one in ten. It's an expensive witness." |
| 5 | 1:10–1:25 | Diagnosis feed types the rationale. Architecture panel: Verifier strikes through, vanishes. **Lessons pane: two rows animate in** — a tool note about `rate_date`, a rule about bank fees. | `--speed 2` through `lesson.written` ×2 | "It fires the Verifier — and writes down what it learned. Not the answers: the *rules*. One about the tool, one about the company." |
| 6 | 1:25–1:45 | g2: prompts rewritten, cases flip green → **18/20**. g3: calls/case **17 → 5**, cost bar drops 35%, latency halves. Third lesson appears (range endpoint). | `--speed 5` | "Two generations later: ninety percent, and it discovered the API's range endpoint on its own — a third of the calls, half the time." |
| 7 | 1:45–2:00 | Metrics strip vs CoT-SC(k=3): Δpass +0.50. `rel³ 0.85`. | `--to-gen 3 --at reliability.completed` | "Cost-matched single-agent baseline: forty percent. Occam: ninety, three runs out of three on seventeen cases." |
| 8 | 2:00–2:35 | **Hard cut: run 2.** Header: `fx_recon_b · run 2 · lessons loaded 3`. Different ledger. Architect proposes **4 roles**, fetcher description visibly contains the learned note. g0 cases: **17/20 green immediately.** Compare strip: `g0 0.55 → 0.85 · calls 17 → 5 · gens to plateau 3 → 1`. | `replay runs/demo_run2 --to-gen 0 --speed 3` | "New month, new ledger the system has never seen. This time the architect reads the lessons first. No Verifier. Range endpoint from the start. Eighty-five percent on the first try — the new team started where the old one finished." |
| 9 | 2:35–2:50 | `cat memory/fx_recon/lessons.md` in a terminal, three entries, evidence pointers. | terminal capture | "This is the whole memory. Three sentences. Each one traceable to the cases that taught it." |
| 10 | 2:50–3:00 | Title: *"Occam. Smaller teams. Fewer mistakes. It remembers what it can prove mattered."* | — | |

**Separate 45s AO clip**: Kanban with parallel WP sessions → `git log --graph --oneline --all` → one VO line on the engine/TUI worktree split.

## Rules
- Shot 3's Inspector highlight and shot 4's ablation take are the two frames the whole demo rests on. Test both at +10h.
- Never cut inside shot 4.
- Header badge reads `REPLAY`. Leave it. README: "replayed from recorded run logs; reproducible with `occam replay`."
- Show a *fragment* of a real case in shot 2 (large font). Judges must see it's a real ledger, not an abstract benchmark.
- If pass³ is cut (hour-12 rule), drop the `rel³` clause from shot 7's VO. If run 2 underperforms (hour-16 rule), shot 8 becomes "g0 → g3 within run 1" plus shot 9.
