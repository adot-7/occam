# 07 — Citations

**Where citations live:** the README's "Background" section and the pitch. **Never in the TUI.** The interface shows numbers and verdicts; the README explains where the ideas came from. `00-PRD.md §9` refers to this file.

Every entry below was verified against a primary source (arXiv API / GitHub / publisher) on 2026-09-03. Do not add entries without doing the same.

## The paper Occam operationalises
- Jwalapuram, Lin, Li, Jiao, Wang, Ming, Ke, Qin, Carenini, Joty. **The Illusion of Multi-Agent Advantage.** arXiv:2606.13003, June 2026. Salesforce Research / HKUST(GZ) / UBC / NTU. Code+data: https://multi-agent-eval.github.io/ · SMFR: https://huggingface.co/datasets/the-illusion-of-multi-agent-advantages/smfr-dataset (CC-BY-4.0).
  Safe quotes: "consistently underperform CoT-SC despite being up to 10× more expensive" · "expensive witnesses that incur full inference costs while exerting near-zero causal influence on the output" · "MAS must be evaluated on their structural fidelity: the degree to which assigned agentic roles exert measurable causal influence on the final decision." Numbers: unanimous consensus in ~70% (GPT-4o) / >90% (GPT-5) of cases; all-assistant 54.4% vs experts 53.4%; verifier picks first block >45%; Expert-MAS on SMFR: GPT-5 49.1% → 96.7%.

## Automated agent design lineage
- Hu, Lu, Clune. **Automated Design of Agentic Systems (ADAS).** arXiv:2408.08435. https://github.com/ShengranHu/ADAS
- Zhang, Hu, Lu, Lange, Clune. **Darwin Gödel Machine.** arXiv:2505.22954. SWE-bench 20.0→50.0%, Polyglot 14.2→30.7%. https://github.com/jennyzzt/dgm
- Shang et al. **AgentSquare.** arXiv:2410.06153 · Zhang et al. **AFlow.** arXiv:2410.10762 · Zhuge et al. **Language Agents as Optimizable Graphs (GPTSwarm).** arXiv:2402.16823 · Liu et al. **A Dynamic LLM-Powered Agent Network (DyLAN).** arXiv:2310.02170 · Zhang et al. **MaAS.** arXiv:2502.04180 · Ke et al. **MAS-Zero.** arXiv:2505.14996
- Xu, Tai. **Meta-Agent: From Task Descriptions to Verified Multi-Agent Systems.** arXiv:2605.25233
- Wang et al. **AgentConductor.** arXiv:2602.17100

## Harness optimisation (2026 vocabulary)
- Lee, Nair, Zhang, Lee, Khattab, Finn. **Meta-Harness.** arXiv:2603.28052
- Zhang et al. **Self-Harness: Harnesses That Improve Themselves.** arXiv:2606.09498

## Single-agent vs multi-agent, cost-matched
- Tran, Kiela. **Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets.** arXiv:2604.02460 — say "match or outperform."
- Anthropic Engineering. **How we built our multi-agent research system.** https://www.anthropic.com/engineering/multi-agent-research-system — safe quote: "token usage by itself explains 80% of the variance." **Not** a real quote: "decompose by context required, not task type."

## When decomposition *does* pay
- Yang, Nie, Chandra, Gannutin, Lin, Chaudhuri. **When Parallelism Pays Off: Cohesion-Aware Task Partitioning for Multi-Agent Coding (Co-Coder).** arXiv:2606.00953. https://github.com/Flitternie/CoCoder — up to +14% pass, 2.1× speed, **−35% cost** (not accuracy).

## Benchmarks
- BFCL: Patil et al., Berkeley Function-Calling Leaderboard, https://github.com/ShishirPatil/gorilla (Apache-2.0)
- MGSM: Shi et al., arXiv:2210.03057; data https://huggingface.co/datasets/juletxara/mgsm (CC-BY-SA-4.0)

## Open-source neighbours (name the delta)
OpenEvolve (AlphaEvolve reimplementation), GEPA (arXiv:2507.19457) / DSPy. They optimise *for a score*; Occam optimises *against redundancy* using causal ablation inside the loop.

## Do not say
"Berkeley reward-hacked all 8 benchmarks" (unverified) · "DGM ran 80 iterations on SWE-bench *Verified*" (label unconfirmed) · any Maximor product claim.

## Added for v3 (verify IDs before quoting)
- Shinn et al. **Reflexion: Language Agents with Verbal Reinforcement Learning.** arXiv:2303.11366 — verbal self-reflection stored as memory for later attempts; our `lessons.jsonl` is this, made cross-run and typed.
- Zhao et al. **ExpeL: LLM Agents Are Experiential Learners.** arXiv:2308.10144 — extracting reusable insights from trajectories.
- Anthropic Engineering. **Writing tools for agents** and **Demystifying evals for AI agents** (linked by the Track 1 judge). Our self-authored tool descriptions and pass³/noise-floor reporting follow both.
- **Frankfurter** — open ECB reference-rates API, https://frankfurter.dev (base URL `api.frankfurter.dev/v1`). ECB TARGET closing days: https://www.ecb.europa.eu/paym/target/target2/profuse/calendar/html/index.en.html
