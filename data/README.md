Gitignored working data, rebuilt by `scripts/pairs.py`. The published copy is the HF dataset
`danielrosehill/Dictation-Format-Pairs`.

- `sources/real-<format>.jsonl`, `sources/synth-<format>.jsonl`: raw dictations (input side)
- `pairs/<format>.jsonl`: teacher outputs with `checks_failed`
- `splits/{train,test}.jsonl`: all formats, failing pairs left out; test is held out
