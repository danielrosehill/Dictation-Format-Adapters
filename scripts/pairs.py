"""Training pairs for the format adapters: raw dictation -> formatted text, one set per catalogue format.

  uv run --with huggingface_hub python scripts/pairs.py catalogue            # validate catalogue, print control lines
  with-secrets uv run --with anthropic --with huggingface_hub python scripts/pairs.py real [--format F]
  with-secrets uv run --with anthropic python scripts/pairs.py synth --format F --n 40
  with-secrets uv run --with anthropic python scripts/pairs.py teach [--format F]
  uv run python scripts/pairs.py check [--format F]                          # re-run format checks, print failures
  uv run python scripts/pairs.py split                                       # -> data/splits/{train,test}.jsonl
  with-secrets uv run --with huggingface_hub python scripts/pairs.py upload  # -> HF dataset (DATASET below)

real   Raw items from the public pool danielrosehill/ASR-Cleanup-Before-After (pool/pool.jsonl) whose use_case is in
       the format's real_use_cases, kept only if a Haiku fit check says the dictation is something a person would
       want in that format. -> data/sources/real-<F>.jsonl
synth  Haiku writes N scenarios for the format, then one raw Parakeet-style dictation per scenario (punctuated,
       cased, one block, fillers and false starts). -> data/sources/synth-<F>.jsonl
teach  The teacher (Sonnet) applies catalogue/prompts/_shared.md + <F>.md to every source item; the format's
       checks run on the output. -> data/pairs/<F>.jsonl (rows failing a check are kept, with checks_failed set;
       split leaves them out).
split  Per format, a fixed held-out test set: up to TEST_REAL real items, then synthetic to TEST_TOTAL. Never trained on.

Pair row: {id, format, control, input, output, source, teacher, checks_failed}. `control` is the full control line
the trainer puts before the transcript (training/format_lora.py reads it verbatim).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = json.loads((ROOT / "catalogue" / "formats.json").read_text())
PROMPTS = ROOT / "catalogue" / "prompts"
DATA = ROOT / "data"
POOL_REPO = "danielrosehill/ASR-Cleanup-Before-After"
PROMPT_LIBRARY = "danielrosehill/Text-Transformation-Prompts-300525"
DATASET = "danielrosehill/Dictation-Format-Pairs"
TEACHER = "claude-sonnet-5-5"
DRAFTER = "claude-haiku-5-5"
TEST_REAL, TEST_TOTAL = 8, 15
WORKERS = 8

FORMATS = {f["id"]: f for f in CATALOGUE["formats"]}


def control(fmt: str) -> str:
    return f"{FORMATS[fmt]['s1_axes']} [Layers: 1,2] [Format: {fmt}]"


def teacher_prompt(fmt: str) -> str:
    return (PROMPTS / "_shared.md").read_text().strip() + "\n\n" + (PROMPTS / f"{fmt}.md").read_text().strip()


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def ask(model: str, system: str, user: str, max_tokens: int = 3000) -> str:
    import anthropic

    global _client
    if "_client" not in globals():
        _client = anthropic.Anthropic()
    m = _client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in m.content if b.type == "text").strip()


def pick(args_format: str | None) -> list[str]:
    if args_format and args_format not in FORMATS:
        sys.exit(f"unknown format {args_format!r}; catalogue has {list(FORMATS)}")
    return [args_format] if args_format else list(FORMATS)


# --- Checks: deterministic shape tests, shared with evaluation ------------------------------------------------
PREAMBLE = re.compile(r"^(here(?:'s| is| are)|sure|certainly|below is)\b", re.I)


def run_checks(fmt: str, text: str) -> list[str]:
    failed = []
    lines = [x for x in text.splitlines() if x.strip()]
    for check in FORMATS[fmt]["checks"]:
        name, _, arg = check.partition(":")
        ok = True
        if name == "no_preamble":
            ok = not PREAMBLE.match(text.strip()) and not text.strip().startswith("```")
        elif name == "bullets_only":
            ok = bool(lines) and all(x.startswith(("- ", "## ", "Note: ")) for x in lines)
        elif name == "checkboxes_only":
            ok = bool(lines) and all(x.startswith(("- [ ] ", "Note: ")) for x in lines)
        elif name == "email_shape":
            ok = len(lines) >= 3 and lines[0].rstrip().endswith(",") and "\n\n" in text
        elif name == "signed":
            ok = len(lines) >= 2 and lines[-1].strip() == CATALOGUE["signature"][arg]
        elif name == "outline_shape":
            h1 = [x for x in lines if x.startswith("# ")]
            h2 = [x for x in lines if x.startswith("## ")]
            ok = (len(h1) == 1 and lines[0].startswith("# ") and len(h2) >= 3
                  and h2[0] == "## Introduction" and h2[-1] == "## Conclusion")
        elif name == "calendar_json":
            try:
                ev = json.loads(text)
                keys = {"title", "date_text", "start_time", "end_time", "location", "attendees", "description"}
                ok = isinstance(ev, list) and bool(ev) and all(isinstance(e, dict) and set(e) == keys for e in ev)
                ok = ok and all(t is None or re.fullmatch(r"\d\d:\d\d", t) for e in ev for t in (e["start_time"], e["end_time"]))
            except (json.JSONDecodeError, TypeError):
                ok = False
        else:
            raise ValueError(f"unknown check {check!r} on {fmt}")
        if not ok:
            failed.append(check)
    return failed


# --- Subcommands -----------------------------------------------------------------------------------------------
def cmd_catalogue(_: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    lib = json.loads(Path(hf_hub_download(PROMPT_LIBRARY, "all_prompts.json", repo_type="dataset")).read_text())
    names = {p["name"] for p in lib}
    bad = 0
    for f in CATALOGUE["formats"]:
        src = f["source_prompt"]
        missing = [p for p in ("_shared", f["id"]) if not (PROMPTS / f"{p}.md").exists()]
        flag = "" if src is None or src in names else f"  !! source_prompt {src!r} not in {PROMPT_LIBRARY}"
        bad += bool(flag or missing)
        print(f"{f['id']:<15} {control(f['id'])}{flag}{'  !! missing prompt ' + str(missing) if missing else ''}")
    sys.exit(1 if bad else 0)


FIT = """You decide whether a dictated transcript is suitable source material for one output format.
Answer with JSON only: {"fits": true|false, "why": "<5-10 words>"}.
It fits only if a person who dictated this would plausibly want it turned into the format, and it holds
enough content for it (for a list: at least two items; for an email: something to tell or ask someone;
for a blog outline: at least two distinct points on a topic; for calendar entries: at least one event)."""


def cmd_real(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    pool = read(Path(hf_hub_download(POOL_REPO, "pool/pool.jsonl", repo_type="dataset")))
    for fmt in pick(args.format):
        f = FORMATS[fmt]
        cands = [r for r in pool if r["use_case"] in f["real_use_cases"] and r["words_in"] >= args.min_words]

        def judge(r):
            try:
                v = json.loads(re.search(r"\{.*\}", ask(DRAFTER, FIT, f"Format: {f['label']}\n\nTranscript:\n{r['input']}", 200), re.S).group())
            except (AttributeError, json.JSONDecodeError):
                return None
            return {"id": f"real-{r['id']}", "format": fmt, "input": r["input"], "source": f"pool:{r['id']}",
                    "use_case": r["use_case"], "fit_why": v.get("why")} if v.get("fits") else None

        with ThreadPoolExecutor(WORKERS) as ex:
            kept = [x for x in ex.map(judge, cands) if x]
        write(DATA / "sources" / f"real-{fmt}.jsonl", kept)
        print(f"{fmt}: {len(kept)}/{len(cands)} pool candidates fit -> data/sources/real-{fmt}.jsonl")


SCENARIOS = """List {n} varied, realistic situations in which one person dictates, by voice into a phone, something they
want turned into: {label}. Vary topic, length (very short to long), setting (home, work, family, travel,
hobby, admin) and how organised the speaker is. Israel-based English speaker is fine for some, not all.
JSON array of strings only, one sentence each."""

DICTATE = """Write the raw speech-to-text transcript of someone dictating, in one take, what is described below. It is
the material they will have turned into: {label}. Write what they would actually SAY, not the finished text.

Make it read like real ASR output (NVIDIA Parakeet): punctuated and cased, one single block with no line breaks,
with natural speech: fillers (um, so, like, you know), a false start or two, the odd repeated word, sometimes a
spoken self-correction ("no, actually make that..."), and items or points in spoken order rather than tidy order.
Length: {words} words. Output only the transcript."""


def cmd_synth(args: argparse.Namespace) -> None:
    fmt, f = args.format, FORMATS[args.format]
    out = DATA / "sources" / f"synth-{fmt}.jsonl"
    have = read(out)
    raw = ask(DRAFTER, "You write test scenarios.", SCENARIOS.format(n=args.n, label=f["label"].lower()), 4000)
    scenarios = json.loads(re.search(r"\[.*\]", raw, re.S).group())[: args.n]
    rng = random.Random(len(have))
    jobs = [(s, rng.choice(["30-60", "60-120", "120-250"])) for s in scenarios]

    def one(job):
        s, words = job
        return ask(DRAFTER, DICTATE.format(label=f["label"].lower(), words=words), s, 1500)

    with ThreadPoolExecutor(WORKERS) as ex:
        texts = list(ex.map(one, jobs))
    start = len(have)
    rows = have + [{"id": f"synth-{fmt}-{start + i:04d}", "format": fmt, "input": t, "source": "synth",
                    "scenario": s, "drafter": DRAFTER} for i, ((s, _), t) in enumerate(zip(jobs, texts)) if t]
    write(out, rows)
    print(f"{fmt}: +{len(rows) - start} synthetic dictations ({len(rows)} total) -> {out.relative_to(ROOT)}")


def cmd_teach(args: argparse.Namespace) -> None:
    for fmt in pick(args.format):
        out = DATA / "pairs" / f"{fmt}.jsonl"
        done = {r["id"] for r in read(out)}
        todo = [r for p in sorted((DATA / "sources").glob(f"*-{fmt}.jsonl")) for r in read(p) if r["id"] not in done]
        system = teacher_prompt(fmt)

        def one(r):
            o = ask(TEACHER, system, r["input"])
            return {"id": r["id"], "format": fmt, "control": control(fmt), "input": r["input"], "output": o,
                    "source": r["source"], "teacher": TEACHER, "checks_failed": run_checks(fmt, o)}

        with ThreadPoolExecutor(WORKERS) as ex:
            new = list(ex.map(one, todo))
        write(out, read(out) + new)
        bad = sum(bool(r["checks_failed"]) for r in new)
        print(f"{fmt}: +{len(new)} pairs, {bad} failing checks -> {out.relative_to(ROOT)}")


def cmd_check(args: argparse.Namespace) -> None:
    for fmt in pick(args.format):
        rows = read(DATA / "pairs" / f"{fmt}.jsonl")
        for r in rows:
            r["checks_failed"] = run_checks(fmt, r["output"])
            if r["checks_failed"]:
                print(f"  {r['id']}: {r['checks_failed']}")
        write(DATA / "pairs" / f"{fmt}.jsonl", rows)
        print(f"{fmt}: {sum(not r['checks_failed'] for r in rows)}/{len(rows)} pass")


def cmd_split(_: argparse.Namespace) -> None:
    train, test = [], []
    for fmt in FORMATS:
        rows = [r for r in read(DATA / "pairs" / f"{fmt}.jsonl") if not r["checks_failed"]]
        rng = random.Random(fmt)
        real = sorted((r for r in rows if r["source"].startswith("pool:")), key=lambda r: r["id"])
        synth = sorted((r for r in rows if r["source"] == "synth"), key=lambda r: r["id"])
        held = rng.sample(real, min(TEST_REAL, len(real) // 4))
        held += rng.sample(synth, min(TEST_TOTAL - len(held), len(synth) // 4))
        ids = {r["id"] for r in held}
        test += held
        train += [r for r in rows if r["id"] not in ids]
        print(f"{fmt}: train {len(rows) - len(held)}, test {len(held)}")
    write(DATA / "splits" / "train.jsonl", train)
    write(DATA / "splits" / "test.jsonl", test)


def cmd_upload(_: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(DATASET, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=str(DATA),
                      allow_patterns=["splits/*.jsonl", "pairs/*.jsonl", "sources/*.jsonl"],
                      commit_message="pairs.py upload")
    print(f"https://huggingface.co/datasets/{DATASET}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sub.add_parser("catalogue").set_defaults(func=cmd_catalogue)
    r = sub.add_parser("real")
    r.add_argument("--format")
    r.add_argument("--min-words", type=int, default=20)
    r.set_defaults(func=cmd_real)
    s = sub.add_parser("synth")
    s.add_argument("--format", required=True, choices=list(FORMATS))
    s.add_argument("--n", type=int, default=40)
    s.set_defaults(func=cmd_synth)
    for name, fn in (("teach", cmd_teach), ("check", cmd_check)):
        t = sub.add_parser(name)
        t.add_argument("--format")
        t.set_defaults(func=fn)
    sub.add_parser("split").set_defaults(func=cmd_split)
    sub.add_parser("upload").set_defaults(func=cmd_upload)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
