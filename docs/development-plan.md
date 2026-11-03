# Development plan

Proof of concept first: the smallest data and compute that answers "does a 0.6B adapter learn this format?".

## Phase 0: setup (done 2026-10-31)

Catalogue v0, teacher prompts, `scripts/pairs.py`, `training/format_lora.py`, `training/modal_run.py`.
**Exit test (passed):** `pairs.py catalogue` validates. A 6-item blog-outline pilot (synth → teach → check) runs
end to end; 5/6 passed the shape checks.

## Phase 1: pairs for the PoC formats

PoC formats: `shopping-list`, `blog-outline`, `business-email` (one list, one structured, one prose-with-layout).
For each: `real`, `synth --n 150`, `teach`, then eyeball 10 pairs per format and fix the teacher prompt before scaling.
Then `split` and `upload`.
**Exit test:** ≥150 passing pairs per PoC format, ≥15 held out each, a 10-pair read-through per format with no
added facts, dataset on HF.

Cost estimate (prices 2026-10-31, OpenRouter batch): ~500 teacher calls on `google/gemini-3.8-flash` at ~600 in +
~1,000 out tokens (reasoning included) ≈ $1.10, plus ~1,000 drafter calls on `deepseek/deepseek-v4.1-flash` ≈ $0.10.
Re-check with `llm-ops:check-api-prices` before running.
Measured in pilot 2 (sync): about US$0.80 per 1,000 pairs.

## Phase 2: first adapters

`modal_run.py --format <F>` for each PoC format. Score stock / cleanup / adapter on the test set (architecture.md,
Evaluation).
**Exit test:** for at least 2 of 3 formats, the adapter's shape pass rate is ≥90% and it has additions on ≤1 in 20
test items. If it fails: more pairs (300) before changing anything else, then rank 32.

## Phase 3: rest of catalogue v0, and one-vs-many

Pairs and adapters for `todo-list`, `casual-email`, `calendar-json`. Then train one multi-format adapter on all six
formats' pairs and compare it with the per-format adapters on the same test sets.
**Exit test:** a written decision (0004) on per-format vs multi-format, backed by the numbers.

## Phase 4: release

Adapters to HF (`danielrosehill/S1-ASR-Format-<format>`, or one repo with subfolders, decided in Phase 3), model
cards naming "S1-mini by Superwhisper", GGUF merges if the target is on-device.

## Later catalogue candidates

From the prompt library, reformat-only: note to self, meeting minutes (internal), main points as bullets, chore
list, WhatsApp message. Not taken: anything that writes new content (cover letter, op-ed, product description).

## Decision log

| # | Date | Decision |
|---|---|---|
| [0001](decisions/0001-cleanup-base-plus-format-loras.md) | 2026-10-31 | Cleanup model as base, one LoRA per format, single pass from raw |
| [0002](decisions/0002-bounded-catalogue-v0.md) | 2026-10-31 | Bounded catalogue v0: six reformat-only formats |
| [0003](decisions/0003-distilled-pairs-real-plus-synthetic.md) | 2026-10-31 | Pairs distilled from a teacher over real pool items and synthetic dictations, filtered by shape checks |
