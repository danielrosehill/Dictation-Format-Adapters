# 0003: Distilled pairs over real and synthetic dictations, filtered by shape checks

Date: 2026-10-31. Status: accepted.

## Decision

- **Inputs.** Real raw dictations from the public pool `danielrosehill/ASR-Cleanup-Before-After`
  (`pool/pool.jsonl`, 774 items, use cases message/email/ai_prompt/note/calendar/code_switch/tech/other),
  filtered per format by `use_case` and a Haiku fit check. Then synthetic dictations (Haiku: scenarios, then
  Parakeet-style raw text with fillers and self-corrections) to reach ~150 per format. Most pool items are
  short messages and won't fit a shopping list or outline, so synthetic data is expected to make up most of
  those two formats.
- **Targets.** `google/gemini-3.8-flash` (what `~google/gemini-flash-latest` served on 2026-10-31) through the
  OpenRouter Batch API, with `catalogue/prompts/_shared.md` + `<format>.md`. Fit checks and scenario seeds on
  `deepseek/deepseek-v4.1-flash`, reasoning off. Synthetic dictations rotate three writers by item index
  (`deepseek/deepseek-v4.1-flash`, `anthropic/claude-haiku-5.5`, `google/gemini-3.1-flash-lite`), one per
  scenario seed. An 11-model probe on 2026-10-31 found that all models given one bare prompt converge on the same
  content (83% of samples mentioned the dentist), so variety comes from per-item seeds. Rotating writer
  families cut wording overlap about 3× (OpenRouter-Batch-Notes notebook/09). Daniel, 2026-10-31: OpenRouter
  batches on Gemini Flash latest or DeepSeek to bring cost down. The 6-pair pilot used Claude Sonnet (sync).
  Every pair records its `teacher`, so a later teacher change can be traced.
- **Filter.** `run_checks` in `scripts/pairs.py` (bullets only, checkbox lines, email greeting/sign-off,
  outline shape, JSON schema). Failing pairs are kept in `pairs/` with `checks_failed` and left out by `split`.
- **Held out.** Per format, up to 8 real + synthetic to 15, fixed by seed. Real items are preferred for test
  because they are closest to actual use.

## Unlike the cleanup project

The cleanup model's targets are Daniel's hand edits, because light-touch cleanup is a matter of personal
preference. Format targets are more mechanical, so a teacher model is enough to start with. Hand review is a
10-pair read-through per format (plan, Phase 1), not annotation of every pair.

## Pilot 2, 2026-10-31 (sync, rotation)

Shopping list: 6 seeded synthetic dictations (2 per writer) taught by Gemini 3.8 Flash. 6/6 passed the shape
checks. A remark to a driver was correctly dropped, and a self-correction ("no, make that two bags") was resolved.
Cost US$0.0047 in total, about US$0.80 per 1,000 pairs at sync prices.

## Batch latency

A one-request OpenRouter batch (DeepSeek V4.1 Flash) was still `in_progress` after 40+ minutes on
2026-10-31. Pilots use `--sync`. Bulk runs use the Batch API and are left to finish in the background.

## Known teacher habit (pilot, 2026-10-31)

In 1 of 6 blog-outline pairs the teacher added a bracketed meta-comment ("(speaker is unsure and always mixes them
up)"). It is a borderline addition. Check for it in the Phase 1 read-through, and tighten `_shared.md` if it recurs.
