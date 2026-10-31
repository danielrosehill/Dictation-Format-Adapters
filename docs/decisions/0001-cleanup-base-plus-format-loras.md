# 0001: Cleanup model as base, one LoRA per format, single pass from raw

Date: 2026-10-31. Status: accepted.

## Context

Daniel (2026-10-31): one model for general cleanup, then adapters for specific formats such as shopping list,
casual email, business email and blog outline, chosen from a bounded list instead of described in free text.
The cleanup model already exists: `danielrosehill/S1-ASR-Cleanup` v1.0, a merged LoRA on S1-mini
(ASR-Post-Processor-Model). Its taxonomy assigns formatting (layer 3) to a prompted LLM pass; this project
replaces that pass, for a fixed set of formats, with small adapters.

## Decision

- Base = `danielrosehill/S1-ASR-Cleanup` @ `6b38b1877e59b6524465b7206f92102f40744028`. No adapter = cleanup only.
- One LoRA (rank 16, all linear layers) per catalogue format, trained on raw dictation → formatted text.
- Single pass: the format adapter gets the raw transcript, not the cleanup model's output.

## Alternatives

- **Chain (cleanup output → format adapter on stock S1-mini).** Two passes, double the latency, and errors add up.
  The benefit is that a cleanup release doesn't force adapter retraining. Rejected: retraining an adapter on ~150
  pairs takes minutes on an L4.
- **One multi-format adapter with the format tag routing.** Kept as the Phase 3 comparison. The training data is
  the same either way, because every pair carries its `[Format: …]` tag.
- **Stock S1-mini as the base.** Each adapter would have to relearn cleanup. Rejected.

## Consequences

- Adapters are tied to the pinned cleanup weights. A new cleanup release means retraining every adapter
  (CLAUDE.md ground rule).
- The `stock` and `cleanup` baselines in `modal_run.py` show how much the adapter adds over the tag alone.
