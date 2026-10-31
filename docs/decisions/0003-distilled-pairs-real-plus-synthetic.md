# 0003: Distilled pairs over real and synthetic dictations, filtered by shape checks

Date: 2026-10-31. Status: accepted.

## Decision

- **Inputs.** Real raw dictations from the public pool `danielrosehill/ASR-Cleanup-Before-After`
  (`pool/pool.jsonl`, 774 items, use cases message/email/ai_prompt/note/calendar/code_switch/tech/other),
  filtered per format by `use_case` and a Haiku fit check. Then synthetic dictations (Haiku: scenarios, then
  Parakeet-style raw text with fillers and self-corrections) to reach ~150 per format. Most pool items are
  short messages and won't fit a shopping list or outline, so synthetic data is expected to make up most of
  those two formats.
- **Targets.** Claude Sonnet (`claude-sonnet-5-5`) with `catalogue/prompts/_shared.md` + `<format>.md`.
- **Filter.** `run_checks` in `scripts/pairs.py` (bullets only, checkbox lines, email greeting/sign-off,
  outline shape, JSON schema). Failing pairs are kept in `pairs/` with `checks_failed` and left out by `split`.
- **Held out.** Per format, up to 8 real + synthetic to 15, fixed by seed. Real items are preferred for test
  because they are closest to actual use.

## Unlike the cleanup project

The cleanup model's targets are Daniel's hand edits, because light-touch cleanup is a matter of personal
preference. Format targets are more mechanical, so a teacher model is enough to start with. Hand review is a
10-pair read-through per format (plan, Phase 1), not annotation of every pair.

## Known teacher habit (pilot, 2026-10-31)

In 1 of 6 blog-outline pairs the teacher added a bracketed meta-comment ("(speaker is unsure and always mixes them
up)"). It is a borderline addition. Check for it in the Phase 1 read-through, and tighten `_shared.md` if it recurs.
