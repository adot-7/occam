# 09 — Demo Shot List

The demo is a **replay of real run logs**, recorded via `textual serve` in a browser at 1080p with a large font, then edited. Every shot below maps to a `occam replay` invocation. `scripts/record_demo.sh` (WP-13) launches each.

Total: 3:00. Voiceover (Smallest.ai or human) written per shot.

| # | t | shot | replay command | VO (draft) |
|---|---|---|---|---|
| 1 | 0:00–0:12 | Black. Two lines of text fade in. | — (title card in edit) | "Automated agent design produces systems where most of the agents do nothing. We measured it." |
| 2 | 0:12–0:20 | Cut straight to the **ablation table at g0**, two rows already red. Hold. | `occam replay runs/demo_smfr --to-gen 0 --at ablation.completed --pause` | "Five agents. Two of them changed the answer on zero of ten cases. They cost 37% of the budget." |
| 3 | 0:20–0:35 | Rewind feel: lineage panel, g0 only. Header shows task + domain. Architecture panel draws 5 boxes. | `--to-gen 0 --at architecture.proposed --speed 2` | "This is Occam. Give it a goal, tools, and an eval set — here, multi-hop financial reasoning over stock data — and it designs an agent system." |
| 4 | 0:35–0:50 | Cases grid fills ✓✓✗… in real time (sped 6×). Metrics strip gets first point. | `--speed 6` through `execution.completed` | "It runs it. Thirteen of twenty. Then it does the thing nobody does." |
| 5 | 0:50–1:25 | **Ablation, row by row.** Slow. Progress bar under table. LOAD-BEARING green, then WITNESS red, WITNESS red, UNCERTAIN amber. Footer appears. | `--speed 1` from `ablation.started` to `ablation.completed` | "It removes each agent, one at a time, and checks whether the answer changes. Extractor: removing it breaks 3 of 10. P&L: breaks 4. Critic: changes nothing. Second Opinion — 31% of spend — changes nothing. Our noise floor this run was one case in ten; these two are inside it. The paper that measured this calls them *expensive witnesses*." |
| 6 | 1:25–1:45 | Diagnosis feed types out the prune rationale. **Architecture panel: two boxes strike through and vanish.** Lineage adds g1 · 3 roles. | `--speed 2` through `mutation.applied` | "So it fires them. Not because a heuristic said so — because the measurement did." |
| 7 | 1:45–2:00 | g1 executes (fast). Metrics strip: pass flat, **cost bar drops**, SF sparkline jumps to 1.0. | `--speed 8` | "Same accuracy. Thirty-nine percent cheaper. Structural fidelity — the share of spend doing real work — from 0.53 to 1.0." |
| 8 | 2:00–2:20 | g2' split appears in lineage → ablation runs on the two new roles → both WITNESS → **`○ g2' REVERTED`**. Diagnosis explains. | `--speed 3` | "Next it tries adding agents back — one per investor. Ablation says the new ones aren't wired into the answer. It reverts its own change. That's the part I care about: it knows when *not* to be a multi-agent system." |
| 9 | 2:20–2:40 | **Split screen, three panes, time-lapse.** Left: SMFR lineage g0→g5. Middle: BFCL run collapsing 4 roles → 1. Right: MGSM run pruning to 1 role, metrics strip showing `Δpass 0.00 · cost ×1.0` vs CoT-SC. | three `textual serve` windows, `--speed 30`, composited in edit | "Three domains, one system, three different right answers. Financial reasoning: keep the parallel workers, cut the critics. Tool calling: one agent was always enough. Math word problems: it converged to a single agent and matched the single-agent baseline exactly — which is the correct answer, and the one most systems can't give." |
| 10 | 2:40–2:55 | Final SMFR metrics strip vs CoT-SC. README table (from `08 §13`) as a card. | `--to-gen 5 --at run.completed` | "Fifteen points more accurate than cost-matched chain-of-thought, 38% cheaper than where it started, two agents deleted, one addition reverted." |
| 11 | 2:55–3:00 | Title card. | — | "Occam. It learned when not to be a multi-agent system." |

**Separate 45s AO clip** (required, appended or as its own video): AO Kanban with 3–4 worker sessions live → `git log --graph --oneline --all` scrolling → one line of VO on the engine/TUI worktree split.

## Rules for recording
- Font ≥ 18px in `textual serve`. Test a 10-second capture at hour 12 and view at 100% on a laptop.
- Never cut inside shot 5. The ablation rows must appear in one continuous take; cuts read as staging.
- `--pause` on the cold open so the frame is static; `space` to resume.
- Header badge reads `REPLAY`. Leave it. "Replayed from recorded run logs" is in the README; it is a strength (reproducible), not a weakness.
- Shot 9 needs three run dirs: `runs/demo_smfr`, `runs/demo_bfcl`, `runs/demo_mgsm`. Curate in WP-11.
- If voiceover is synthetic, say so in the README credits.
