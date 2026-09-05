# Open Questions

Workers: append under a dated heading when the PRD is wrong, ambiguous, or blocks you. Orchestrator resolves and moves items to "Resolved".

## Unresolved (as of 2026-09-04)

- **Groq model roster.** `models.yaml` assumes `llama-3.3-70b-versatile` exists on Groq's free tier on Saturday. Check `GET /openai/v1/models` first thing and update the yaml. Do not hardcode elsewhere.
- **Price rates in `models.yaml`.** Verify per-token prices for the chosen worker/architect models on Saturday morning; cost numbers in the demo depend on them.
- **Tool-calling support on the worker model.** SMFR roles need native tool calls. If the Groq model's tool calling is flaky, fall back to a JSON-in-text tool protocol in the executor (parse ```tool blocks). Decide in WP-05.
- **SMFR TPM pressure.** 3.5k-token inputs × concurrency may hit Groq's 6,000 TPM. If so: cap concurrency at 1 for SMFR full runs and use `worker_alt` (Gemini) for ablation lane.
- **BFCL `"type":"dict"`.** BFCL uses `dict` where JSON Schema says `object`; the adapter must rewrite this before handing tool specs to providers.
- **Hackathon rubric.** Unconfirmed; assumed to mirror The Orchestra (AO usage · working demo · creativity · presentation).

## Resolved

- Language: Python + Textual (see `04 §0`).
- Engine/TUI integration: run directory + event log only (see `01 §1`).
- Baseline: cost-matched CoT-SC (see `03 §6`).
