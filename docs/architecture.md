# Architecture

## Stack

| Layer | Model | Status |
|---|---|---|
| Base | `superwhisper/s1-mini` @ `88f6b15896c73bbb13a3b596e0afe8ea0d5150b4` (Qwen3-0.6B) | upstream |
| Cleanup | `danielrosehill/S1-ASR-Cleanup` @ `6b38b1877e59b6524465b7206f92102f40744028`: S1-mini + cleanup LoRA (405 pairs), merged, v1.0 | released 2026-10-29/30 in ASR-Post-Processor-Model |
| Format | one LoRA per catalogue format, trained on the cleanup model | this repo |

Because the cleanup model is merged, it is the base for the format LoRAs. With no adapter loaded the model only
cleans. With an adapter it cleans and formats in one pass. Each adapter is trained on raw dictation → formatted
target, so it does not depend on the cleanup output being right first.

## Prompt

S1-mini's system prompt, verbatim (its card says rewording it degrades output). The user turn is a control line,
a newline, then the raw transcript:

```
<|im_start|>system
You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>
<|im_start|>user
[Styling: semi-formal] [Structure: lists] [Context: general] [Layers: 1,2] [Format: blog-outline]
<raw transcript><|im_end|>
<|im_start|>assistant
<think>

</think>

<formatted text><|im_end|>
```

- The first three axes are S1-mini's own, set per format in `catalogue/formats.json` (`s1_axes`) to the closest
  thing the stock model already knows: `[Structure: lists]` for lists and outlines, `[Context: email]` for emails,
  `[Styling: formal]` for the business email. The adapter starts close to the target.
- `[Layers: 1,2]` is the cleanup model's axis, always on.
- `[Format: <id>]` is the new axis. It is redundant with the adapter that is loaded, on purpose: if per-format
  adapters are later merged into one multi-format adapter (see below), the same data trains it unchanged.

## Serving

- **Per-format adapters (default).** PEFT `load_adapter` / `set_adapter`; vLLM and llama.cpp (`--lora`) both serve
  several LoRAs over one base, switched per request. Each adapter is a few MB at rank 16.
- **Merged GGUF per format.** For on-device use (Whisperian on Android) where hot-swapping is unavailable, each
  format can ship as its own merged Q4_K_M/Q8_0 GGUF like S1-ASR-Cleanup v1.0 (~400 MB each at Q4_K_M; assumed by
  analogy, not measured).
- **One multi-format adapter (fallback).** Train a single LoRA on all formats' pairs, with the format tag doing the
  routing. Tried only if the per-format adapters show the base can't hold six adapters' worth of shape, or if
  serving needs a single file. Phase 3 of the plan compares the two.

## Evaluation per format

`training/modal_run.py` runs the held-out test pairs three ways: `stock` (S1-mini, own axes only), `cleanup` (the
cleanup model with the full control line, no adapter) and `adapter`. Scoring (Phase 2):

1. **Shape pass rate.** `scripts/pairs.py` `run_checks`: the same deterministic checks that filter training data.
2. **Content fidelity.** A judge model lists items or facts in the transcript missing from the output (omissions)
   and items in the output not in the transcript (additions). Additions are the failure that matters most.
3. **Against the teacher.** The teacher's output is the reference. The pass bar is the adapter matching the teacher
   on shape and having additions on no more than 1 in 20 test items.
