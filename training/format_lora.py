# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.8",
#   "transformers>=5.20,<6",
#   "peft>=0.21,<1",
#   "trl>=1.16,<2",
#   "datasets>=3.0",
#   "accelerate>=1.0",
# ]
# ///
"""One LoRA format adapter on top of the S1-ASR-Cleanup model (decision 0001).

Self-contained on purpose (Modal and `hf jobs uv run` ship only this file): the prompt format lives here.
Each pair carries its full control line in `control`, so the catalogue is not duplicated here.

  render  print the exact prompt for one transcript and control line
  train   LoRA on the pairs of ONE format (--format filters a mixed file)
  infer   greedy decoding over test pairs: --base cleanup (default) with or without --adapter, or --base stock
          (S1-mini by Superwhisper with only its own axes from the control line: the no-fine-tune baseline)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASES = {
    # danielrosehill/S1-ASR-Cleanup v1.0: S1-mini by Superwhisper + the cleanup LoRA, merged. Pinned commit.
    "cleanup": ("danielrosehill/S1-ASR-Cleanup", "6b38b1877e59b6524465b7206f92102f40744028"),
    "stock": ("superwhisper/s1-mini", "88f6b15896c73bbb13a3b596e0afe8ea0d5150b4"),
}
# S1-mini's system prompt, verbatim from its card (rewording degrades it). The cleanup model was trained on it too.
SYSTEM_PROMPT = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
# Qwen3 template with thinking off; S1-mini returns blank output without the empty think block.
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END_OF_TURN = "<|im_end|>"
CONTROL_RE = re.compile(r"^\[Styling: [a-z-]+\] \[Structure: [a-z]+\] \[Context: [a-z]+\] \[Layers: 1,2\] \[Format: [a-z-]+\]$")


def s1_axes(control: str) -> str:
    """Only S1-mini's own three axes: what the stock model understands."""
    return " ".join(re.findall(r"\[(?:Styling|Structure|Context): [a-z-]+\]", control))


def render_prompt(transcript: str, control: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}{END_OF_TURN}\n"
            f"<|im_start|>user\n{control}\n{transcript}{END_OF_TURN}\n{ASSISTANT_PREFIX}")


def load_pairs(path: str, fmt: str | None) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if fmt is None or r["format"] == fmt]
    for r in rows:
        if not CONTROL_RE.match(r["control"]):
            raise ValueError(f"{r['id']}: bad control line {r['control']!r}")
    if not rows:
        raise SystemExit(f"no pairs for format {fmt!r} in {path}")
    return rows


def cmd_render(a: argparse.Namespace) -> None:
    print(render_prompt(a.text, a.control))


def cmd_train(a: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    repo, rev = BASES["cleanup"]
    rows = load_pairs(a.data, a.format)
    ds = Dataset.from_list([{"prompt": render_prompt(r["input"], r["control"]), "completion": r["output"] + END_OF_TURN}
                            for r in rows])
    print(f"{len(ds)} {a.format} pairs; base {repo}@{rev[:8]}", flush=True)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16 if bf16 else torch.float32)
    cfg = SFTConfig(output_dir=a.output_dir, num_train_epochs=a.epochs, max_steps=a.max_steps,
                    per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
                    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.05, max_length=a.max_length,
                    completion_only_loss=True, bf16=bf16, logging_steps=5, save_strategy="no", report_to="none",
                    seed=a.seed)
    lora = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)
    (Path(a.output_dir) / "run.json").write_text(json.dumps(
        {"format": a.format, "base_model": repo, "base_revision": rev, "pairs": len(ds), "control": rows[0]["control"],
         "system_prompt": SYSTEM_PROMPT, "args": vars(a) | {"func": None}}, indent=2))


def cmd_infer(a: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, rev = BASES[a.base]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    out = open(a.out, "w") if a.out else sys.stdout
    for r in load_pairs(a.data, a.format):
        ctl = s1_axes(r["control"]) if a.base == "stock" else r["control"]
        enc = tok(render_prompt(r["input"], ctl), return_tensors="pt", add_special_tokens=False).to(dev)
        n = enc.input_ids.shape[1]
        with torch.no_grad():  # formats can be longer than the input (outline headings, JSON keys): 2x + 64
            gen = model.generate(**enc, max_new_tokens=2 * n + 64, do_sample=False)
        pred = tok.decode(gen[0][n:], skip_special_tokens=True).strip()
        out.write(json.dumps({"id": r["id"], "format": r["format"], "control": ctl, "base": a.base,
                              "adapter": a.adapter, "input": r["input"], "prediction": pred,
                              "reference": r["output"]}, ensure_ascii=False) + "\n")
        out.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    r = sub.add_parser("render")
    r.add_argument("text")
    r.add_argument("--control", default="[Styling: semi-formal] [Structure: lists] [Context: general] [Layers: 1,2] [Format: blog-outline]")
    r.set_defaults(func=cmd_render)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--format", required=True)
    t.add_argument("--output-dir", default="outputs/adapter")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--epochs", type=float, default=4)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--max-length", type=int, default=3072)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)
    i = sub.add_parser("infer")
    i.add_argument("--data", required=True)
    i.add_argument("--format")
    i.add_argument("--base", choices=list(BASES), default="cleanup")
    i.add_argument("--adapter")
    i.add_argument("--out")
    i.set_defaults(func=cmd_infer)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
