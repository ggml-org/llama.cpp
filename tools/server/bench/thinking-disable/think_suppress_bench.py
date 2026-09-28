#!/usr/bin/env python3
"""
Nightly / manual gate C: compare think-block rate for --reasoning on vs off
under --jinja and --no-jinja.

Requires a running llama-server pointed at a reasoning-capable model (e.g. Qwen3-0.6B).
This is NOT part of default CI (model-dependent).

Example:
  # Terminal A: llama-server -m qwen3-0.6b.gguf --jinja --reasoning off -c 4096
  python think_suppress_bench.py --url http://127.0.0.1:8080 --label jinja-off

  # Restart with --no-jinja --reasoning off, then:
  python think_suppress_bench.py --url http://127.0.0.1:8080 --label nojinja-off

Accept: nojinja-off think_hit_rate <= jinja-off + 0.05, and vs on drop >= 50%.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

PROMPTS = [
    "What is 2+2? Answer briefly.",
    "Name the capital of France.",
    "Is water wet? One sentence.",
    "Translate 'hello' to Spanish.",
    "List three colors.",
]


def chat(url: str, prompt: str, n_predict: int) -> str:
    r = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": 0.0,
        },
        timeout=300,
    )
    r.raise_for_status()
    body = r.json()
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return content + "\n" + reasoning


def think_chars(text: str) -> int:
    m = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return sum(len(x) for x in m)
    # unclosed / leaked think
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        return max(0, len(text) // 2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    hits = 0
    chars = []
    for p in PROMPTS:
        text = chat(args.url, p, args.n_predict)
        tc = think_chars(text)
        chars.append(tc)
        hit = tc > 0
        hits += int(hit)
        rows.append({"prompt": p, "think_chars": tc, "hit": hit, "sample": text[:200]})

    summary = {
        "label": args.label,
        "n": len(PROMPTS),
        "think_hit_rate": hits / len(PROMPTS),
        "think_chars_median": sorted(chars)[len(chars) // 2],
        "rows": rows,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
