# CLAUDE.md

Public repo: a catalogue of LoRA format adapters on top of the dictation cleanup model. Never add private material.

## Read first, in this order

1. `project.json`: stage and status.
2. `docs/development-plan.md`: the first phase whose exit test hasn't passed is the work.
3. `catalogue/formats.json`: the scope. A format not in it doesn't exist.
4. `docs/architecture.md`: prompt format, control line, stacking.
5. `docs/decisions/`: a new decision gets a new numbered file and a row in the plan's decision log.

## Ground rules

- **The catalogue is bounded.** Adding a format means a `formats.json` entry, a `catalogue/prompts/<id>.md`, a check
  in `scripts/pairs.py` `run_checks` if the shape is new, and a line in the plan. No free-text instruction formats.
- **Reformat, never author.** Formats only rearrange what the speaker said. Formats that need invented content
  (cover letters, op-eds, product copy) are out of scope because a 0.6B model hallucinates them.
- **The base is pinned.** `danielrosehill/S1-ASR-Cleanup@6b38b18…` (v1.0). When the cleanup model gets a new
  release, bump the pin in `catalogue/formats.json` and `training/format_lora.py` together and retrain every adapter.
  Adapters are tied to the exact base weights.
- **S1-mini's system prompt is verbatim.** Do not reword it. Steering is the control line only.
- `training/format_lora.py` stays one self-contained file (Modal and HF Jobs ship only it). Pairs carry their full
  `control` line, so the trainer never needs the catalogue.
- Test pairs (`data/splits/test.jsonl`) are never trained on. `split` is deterministic per format; don't reshuffle.
- `data/` is gitignored. Pairs live in the HF dataset `danielrosehill/Dictation-Format-Pairs` (public, created by
  `pairs.py upload`). Sources are the public pool and synthetic text only.
- Training runs on Modal (`MODAL_PROFILE=danielrosehill`). The laptop has no GPU; 0.6B inference runs on CPU.
- Secrets via `with-secrets` (Infisical): `ANTHROPIC_API_KEY`, `HF_TOKEN`. Python via `uv`.
- Licence: "S1-mini by Superwhisper" must be named wherever an adapter is published or used.
