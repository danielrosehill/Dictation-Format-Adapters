"""Training pairs for the format adapters: raw dictation -> formatted text, one set per catalogue format.

  uv run --with huggingface_hub python scripts/pairs.py catalogue            # validate catalogue, print control lines
  with-secrets uv run --with huggingface_hub python scripts/pairs.py real [--format F]
  with-secrets uv run python scripts/pairs.py synth --format F --n 40      # add --sync (before the subcommand) for pilots
  with-secrets uv run python scripts/pairs.py teach [--format F]
  uv run python scripts/pairs.py check [--format F]                          # re-run format checks, print failures
  uv run python scripts/pairs.py split                                       # -> data/splits/{train,test}.jsonl
  with-secrets uv run --with huggingface_hub python scripts/pairs.py upload  # -> HF dataset (DATASET below)

LLM calls go through the OpenRouter Batch API (POST /api/v1/batches; async, 24 h window, median minutes), stdlib
only, OPENROUTER_API_KEY via with-secrets. Every submitted batch id is recorded in data/batches.json under a key
built from its stage and inputs, so an interrupted run re-attaches to the same batch instead of paying twice.
Prices checked 2026-10-31 (per M tokens in/out): teacher google/gemini-3.8-flash:batch 0.375/1.875 (what
~google/gemini-flash-latest resolves to; batch needs a concrete id), drafter deepseek/deepseek-v4.1-flash:batch
0.050/0.150. Compare batch vs sync on the day before bulk runs: OpenRouter batch is not always cheaper.

real   Raw items from the public pool danielrosehill/ASR-Cleanup-Before-After (pool/pool.jsonl) whose use_case is in
       the format's real_use_cases, kept only if a Haiku fit check says the dictation is something a person would
       want in that format. -> data/sources/real-<F>.jsonl
synth  Haiku writes N scenarios for the format, then one raw Parakeet-style dictation per scenario (punctuated,
       cased, one block, fillers and false starts). -> data/sources/synth-<F>.jsonl
teach  The teacher (Sonnet) applies catalogue/prompts/_shared.md + <F>.md to every source item; the format's
       checks run on the output. (Pilot 2026-10-31 used Claude Sonnet sync; switched to the batch teacher after.) -> data/pairs/<F>.jsonl (rows failing a check are kept, with checks_failed set;
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = json.loads((ROOT / "catalogue" / "formats.json").read_text())
PROMPTS = ROOT / "catalogue" / "prompts"
DATA = ROOT / "data"
POOL_REPO = "danielrosehill/ASR-Cleanup-Before-After"
PROMPT_LIBRARY = "danielrosehill/Text-Transformation-Prompts-300525"
DATASET = "danielrosehill/Dictation-Format-Pairs"
TEACHER = "google/gemini-3.8-flash"
DRAFTER = "deepseek/deepseek-v4.1-flash"
# Gemini 3.x has no reasoning off switch; its reasoning tokens count against max_tokens, so low effort + high cap
# (ASR-Post-Processor-Model saw 1,500-token caps cut replies, 2026-10-29). DeepSeek: {"enabled": false} or it
# bills ~480 reasoning tokens per call even with exclude.
# Synthetic dictations are written by a rotation of three families (item i -> WRITERS[i % 3]): one model's
# habits would otherwise become the dataset's habits. Probe 2026-10-31 (OpenRouter-Batch-Notes notebook/09,
# skill batch-jobs:synthetic-data-generation): every model given the same prompt converged on the same content;
# mixing families cut wording overlap ~3x. Variety itself comes from the per-item scenario seed.
WRITERS = ["deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5", "google/gemini-3.1-flash-lite"]
REASONING = {TEACHER: {"effort": "low"}, DRAFTER: {"enabled": False}, "anthropic/claude-haiku-5.5": {"enabled": False},
             "google/gemini-3.1-flash-lite": {"effort": "low"}}
SYNC = False  # --sync: same requests as threaded sync calls (pilots: a 1-request batch sat in_progress 40+ min, 2026-10-31)
OR_API = "https://openrouter.ai/api/v1"
TEST_REAL, TEST_TOTAL = 8, 15

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


def _or(method: str, path: str, body: dict | None = None, missing_ok: bool = False) -> dict | None:
    import os
    import urllib.error
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("OPENROUTER_API_KEY not set: run under `with-secrets`")
    req = urllib.request.Request(f"{OR_API}{path}", method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=300).read())
    except urllib.error.HTTPError as e:
        if missing_ok and e.code == 404:
            return None
        sys.exit(f"{method} {path}: HTTP {e.code} {e.read().decode()[:1500]}")


def sync(stage: str, model: str, jobs: list[tuple[str, str, str]], max_tokens: int) -> dict[str, str]:
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        i, sy, u = job
        d = _or("POST", "/chat/completions", {"model": model, "max_tokens": max_tokens, "reasoning": REASONING[model],
                                               "messages": [{"role": "system", "content": sy}, {"role": "user", "content": u}]})
        ch = d["choices"][0]
        return i, ("" if ch.get("finish_reason") == "length" else (ch["message"].get("content") or "").strip()), \
            (d.get("usage") or {}).get("cost") or 0

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, jobs))
    out = {i: t for i, t, _ in res if t}
    cost = sum(c for *_, c in res)
    print(f"{stage}: {len(out)}/{len(jobs)} ok (sync, {model}), US${cost:.4f}", flush=True)
    with open(DATA / "usage.jsonl", "a") as f:
        f.write(json.dumps({"stage": stage, "model": model, "mode": "sync", "ok": len(out), "cost": cost}) + "\n")
    return out


def batch(stage: str, model: str, jobs: list[tuple[str, str, str]], max_tokens: int = 8000) -> dict[str, str]:
    """jobs = [(id, system, user)] -> {id: reply text}. Failed or truncated requests are left out (and reported)."""
    import hashlib
    import time

    if not jobs:
        return {}
    if SYNC:
        return sync(stage, model, jobs, max_tokens)
    ledger_path = DATA / "batches.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    key = f"{stage}:{model}:" + hashlib.sha256(json.dumps(jobs).encode()).hexdigest()[:16]
    if key in ledger:
        bid = ledger[key]["id"]
        print(f"{stage}: re-attaching to batch {bid}", flush=True)
    else:
        # Key order matters: endpoint/model/completion_window before requests, else 400 (batch-jobs runner, 2026-10-22).
        body = {"endpoint": "/v1/chat/completions", "model": model, "completion_window": "24h", "requests": [
            {"custom_id": i, "body": {"max_tokens": max_tokens, "reasoning": REASONING[model],
                                      "messages": [{"role": "system", "content": sy}, {"role": "user", "content": u}]}}
            for i, sy, u in jobs]}
        bid = _or("POST", "/batches", body)["id"]
        ledger[key] = {"id": bid, "stage": stage, "model": model, "n": len(jobs)}
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=1))
        print(f"{stage}: batch {bid}, {len(jobs)} requests to {model}", flush=True)
    t0 = time.time()
    while True:  # a GET right after submit 404s for a while (seen 2026-10-22)
        b = _or("GET", f"/batches/{bid}", missing_ok=time.time() - t0 < 300)
        if b and b["status"] in {"completed", "failed", "expired", "cancelled"}:
            break
        c = (b or {}).get("request_counts") or {}
        print(f"  {bid}: {(b or {}).get('status', 'not visible yet')} {c.get('completed', 0)}/{c.get('total', '?')}", flush=True)
        time.sleep(30)
    if b["status"] != "completed":
        sys.exit(f"{stage}: batch {bid} {b['status']} {b.get('error')}")
    out, bad = {}, 0
    for r in b.get("results") or []:
        resp = r.get("response") or {}
        ch = ((resp.get("body") or {}).get("choices") or [{}])[0]
        if r.get("error") or resp.get("status_code") != 200 or ch.get("finish_reason") == "length":
            bad += 1
            continue
        out[r["custom_id"]] = (ch["message"].get("content") or "").strip()
    u = b.get("usage") or {}
    print(f"{stage}: {len(out)} ok, {bad} failed; {u.get('prompt_tokens')} in / {u.get('completion_tokens')} out, US${u.get('cost')}", flush=True)
    with open(DATA / "usage.jsonl", "a") as f:
        f.write(json.dumps({"stage": stage, "model": model, "batch": bid, "ok": len(out), "failed": bad,
                            "provider": b.get("provider"), "usage": u}) + "\n")
    return out


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

        replies = batch(f"real-{fmt}", DRAFTER, [(r["id"], FIT, f"Format: {f['label']}\n\nTranscript:\n{r['input']}")
                                                 for r in cands], 300)
        kept = []
        for r in cands:
            try:
                v = json.loads(re.search(r"\{.*\}", replies.get(r["id"], ""), re.S).group())
            except (AttributeError, json.JSONDecodeError):
                continue
            if v.get("fits"):
                kept.append({"id": f"real-{r['id']}", "format": fmt, "input": r["input"], "source": f"pool:{r['id']}",
                             "use_case": r["use_case"], "fit_why": v.get("why")})
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
    start = len(have)
    # Scenarios in chunks of 25 (one long list in one reply repeats itself), all chunks in one batch.
    chunks = [min(25, args.n - k) for k in range(0, args.n, 25)]
    sc = batch(f"scenarios-{fmt}-{start}", DRAFTER, [(f"c{k}", "You write test scenarios. Reply with a JSON array only.",
               SCENARIOS.format(n=n, label=f["label"].lower()) + f" (Set {k + 1} of {len(chunks)}: avoid the obvious ones.)")
               for k, n in enumerate(chunks)], 4000)
    scenarios = []
    for k in range(len(chunks)):
        try:
            scenarios += json.loads(re.search(r"\[.*\]", sc.get(f"c{k}", ""), re.S).group())
        except (AttributeError, json.JSONDecodeError):
            print(f"  scenario chunk {k} unparseable, skipped")
    rng = random.Random(start)
    jobs = [(f"synth-{fmt}-{start + i:04d}", s, rng.choice(["30-60", "60-120", "120-250"])) for i, s in enumerate(scenarios[: args.n])]
    writer = {i: WRITERS[n % len(WRITERS)] for n, (i, _, _) in enumerate(jobs)}
    texts = {}
    for w_model in WRITERS:  # one batch per writer model
        texts |= batch(f"synth-{fmt}-{start}-{w_model.split('/')[1]}", w_model,
                       [(i, DICTATE.format(label=f["label"].lower(), words=w), s) for i, s, w in jobs if writer[i] == w_model], 1500)
    rows = have + [{"id": i, "format": fmt, "input": texts[i], "source": "synth", "scenario": s, "drafter": writer[i]}
                   for i, s, _ in jobs if texts.get(i)]
    write(out, rows)
    print(f"{fmt}: +{len(rows) - start} synthetic dictations ({len(rows)} total) -> {out.relative_to(ROOT)}")


def cmd_teach(args: argparse.Namespace) -> None:
    for fmt in pick(args.format):
        out = DATA / "pairs" / f"{fmt}.jsonl"
        done = {r["id"] for r in read(out)}
        todo = [r for p in sorted((DATA / "sources").glob(f"*-{fmt}.jsonl")) for r in read(p) if r["id"] not in done]
        system = teacher_prompt(fmt)

        outs = batch(f"teach-{fmt}", TEACHER, [(r["id"], system, r["input"]) for r in todo])
        new = [{"id": r["id"], "format": fmt, "control": control(fmt), "input": r["input"], "output": outs[r["id"]],
                "source": r["source"], "teacher": TEACHER, "checks_failed": run_checks(fmt, outs[r["id"]])}
               for r in todo if outs.get(r["id"])]
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
    p.add_argument("--sync", action="store_true", help="threaded sync calls instead of the Batch API (pilots)")
    a = p.parse_args()
    global SYNC
    SYNC = a.sync
    a.func(a)


if __name__ == "__main__":
    main()
