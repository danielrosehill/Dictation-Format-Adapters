"""Train one format adapter on Modal and run the three-way test for that format.

  MODAL_PROFILE=danielrosehill uv run --with modal modal run training/modal_run.py --format blog-outline

Reads data/splits/{train,test}.jsonl (scripts/pairs.py split). On one L4/A10G: train the LoRA on the cleanup
model, then infer the format's test pairs three ways:
  stock    S1-mini by Superwhisper, own axes only (e.g. [Structure: lists] [Context: email]): no fine-tune
  cleanup  the cleanup model with the full control line, no adapter: does the tag alone do anything?
  adapter  cleanup model + this format's LoRA
-> outputs/<format>/{stock,cleanup,adapter}.jsonl and outputs/<format>/adapter/.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("dictation-format-adapters")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch>=2.8", "transformers>=5.20,<6", "peft>=0.21,<1", "trl>=1.16,<2", "datasets>=3.0", "accelerate>=1.0")
         .add_local_file(ROOT / "training" / "format_lora.py", "/root/format_lora.py"))


@app.function(image=image, gpu=["L4", "A10G"], timeout=3600)
def run(fmt: str, train: str, test: str, epochs: float) -> dict:
    import subprocess

    Path("/tmp/train.jsonl").write_text(train)
    Path("/tmp/test.jsonl").write_text(test)
    py = ["python", "/root/format_lora.py"]
    subprocess.run(py + ["train", "--data", "/tmp/train.jsonl", "--format", fmt, "--output-dir", "/tmp/adapter",
                         "--epochs", str(epochs)], check=True)
    res = {}
    for name, extra in (("stock", ["--base", "stock"]), ("cleanup", []), ("adapter", ["--adapter", "/tmp/adapter"])):
        subprocess.run(py + ["infer", "--data", "/tmp/test.jsonl", "--format", fmt, "--out", f"/tmp/{name}.jsonl"] + extra, check=True)
        res[name] = Path(f"/tmp/{name}.jsonl").read_text()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add("/tmp/adapter", arcname="adapter")
    res["adapter_tar"] = buf.getvalue()
    return res


@app.local_entrypoint()
def main(format: str, epochs: float = 4):
    splits = ROOT / "data" / "splits"
    res = run.remote(format, (splits / "train.jsonl").read_text(), (splits / "test.jsonl").read_text(), epochs)
    out = ROOT / "outputs" / format
    out.mkdir(parents=True, exist_ok=True)
    for name in ("stock", "cleanup", "adapter"):
        (out / f"{name}.jsonl").write_text(res[name])
    with tarfile.open(fileobj=io.BytesIO(res["adapter_tar"]), mode="r:gz") as t:
        t.extractall(out, filter="data")
    print(f"wrote {out}/{{stock,cleanup,adapter}}.jsonl and {out}/adapter/")
