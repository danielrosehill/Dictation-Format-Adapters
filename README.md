# Dictation Format Adapters

A small dictation post-processing model with a **fixed catalogue of output formats**. Raw speech-to-text goes in.
Out comes either clean text (no adapter) or the same content in one named format: shopping list, to-do list, casual
email, business email, blog outline, calendar entries as JSON. You pick a format from a list. You do not describe
what you want in free text.

**Status (2026-10-31): planning and data setup.** Catalogue v0 (6 formats), teacher prompts, the pair builder and the
trainer are written. A 6-item blog-outline pilot ran end to end (5/6 passed the shape checks; the failure was a
synthetic dictation with too little content, correctly rejected). No adapter has been trained yet. Next step:
[development plan](docs/development-plan.md), Phase 1.

## How it works

```
raw dictation ──► S1-ASR-Cleanup (merged S1-mini by Superwhisper + cleanup LoRA, 0.6B)
                     │ no adapter          → clean text, wording kept           [Format: none]
                     ├ + shopping-list LoRA → "- Milk, 2 litres" …               [Format: shopping-list]
                     ├ + business-email LoRA → "Dear Ms Cohen, …"               [Format: business-email]
                     └ + blog-outline LoRA  → "# Title / ## Introduction / …"   [Format: blog-outline]
```

- **One base, one adapter per format.** The general cleanup model is the base and works on its own. Each format is
  a small LoRA that is loaded on top in a single pass: raw dictation in, formatted text out
  ([decision 0001](docs/decisions/0001-cleanup-base-plus-format-loras.md)).
- **A bounded catalogue.** [`catalogue/formats.json`](catalogue/formats.json) is the whole list. A 0.6B model
  learns a few named output shapes reliably; it does not follow arbitrary instructions reliably
  ([decision 0002](docs/decisions/0002-bounded-catalogue-v0.md)).
- **Distilled data.** Real dictations from the public cleanup pool plus synthetic ones, formatted by a teacher model
  (Claude Sonnet) under a fixed prompt per format, then filtered by deterministic shape checks
  ([decision 0003](docs/decisions/0003-distilled-pairs-real-plus-synthetic.md)).

## Contents

| Path | What |
|---|---|
| [`docs/development-plan.md`](docs/development-plan.md) | Phases with exit tests, decision log |
| [`docs/architecture.md`](docs/architecture.md) | Prompt and control line, adapter stacking, serving options |
| [`catalogue/formats.json`](catalogue/formats.json) | The format catalogue: ids, control axes, source prompts, checks |
| [`catalogue/prompts/`](catalogue/prompts/) | Teacher prompts: `_shared.md` (faithfulness rules) + one file per format |
| [`scripts/pairs.py`](scripts/pairs.py) | Pair builder: `catalogue`, `real`, `synth`, `teach`, `check`, `split`, `upload` |
| [`training/format_lora.py`](training/format_lora.py) | Self-contained LoRA trainer and inference (render / train / infer) |
| [`training/modal_run.py`](training/modal_run.py) | Train one format on Modal, then test stock vs cleanup vs adapter |
| [`docs/decisions/`](docs/decisions/) | Decision records |

## Related

- [ASR-Post-Processor-Model](https://github.com/danielrosehill/ASR-Post-Processor-Model): the cleanup model
  (layers 1–2), released as [`danielrosehill/S1-ASR-Cleanup`](https://huggingface.co/danielrosehill/S1-ASR-Cleanup).
  That repo treats formatting as layer 3; this repo is layer 3.
- [`danielrosehill/ASR-Cleanup-Before-After`](https://huggingface.co/datasets/danielrosehill/ASR-Cleanup-Before-After):
  public pool of raw dictations, the source of the real inputs here.
- [`danielrosehill/Text-Transformation-Prompts-300525`](https://huggingface.co/datasets/danielrosehill/Text-Transformation-Prompts-300525):
  the 206-prompt library the catalogue's teacher prompts are adapted from.
- [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini): the underlying base, Qwen3-0.6B.

## Licence

Code: MIT. Adapters, when released, inherit S1-mini's licence (Apache 2.0 plus its naming clause:
"S1-mini by Superwhisper" wherever the model is used).
