#!/usr/bin/env python3
"""
Radix / prefix-cache workload harness for llama-server.

Workloads:
  gsp        - Generated Shared Prefix (groups share a long system prompt)
  dualagent  - Two alternating sessions with a shared system prefix (issue #20510 style)
  multiturn  - Monotonic growing context on one session
  unique     - No shared prefixes (regression / overhead check)

Compares radix on vs off when the server supports --radix-cache / --no-radix-cache.
Requires a running llama-server (or use --spawn).

Example:
  python radix_prefix_bench.py --url http://127.0.0.1:8080 --workload gsp --prefix-len 512
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)


def make_prompt(prefix: str, suffix: str) -> str:
    return prefix + "\n" + suffix


def gen_tokens_approx(n: int, seed: str) -> str:
    # Approximate token length with space-separated words (server tokenizer differs).
    words = []
    i = 0
    while len(words) < n:
        words.append(f"{seed}{i}")
        i += 1
    return " ".join(words)


def completion(url: str, prompt: str, n_predict: int, temperature: float = 0.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{url.rstrip('/')}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": True,
        },
        timeout=600,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    timings = body.get("timings") or {}
    return {
        "dt": dt,
        "prompt_n": timings.get("prompt_n", 0),
        "cache_n": timings.get("cache_n", 0),
        "predicted_n": timings.get("predicted_n", 0),
        "content": body.get("content", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    dts = [x["dt"] for x in rows]
    prompt_n = sum(x["prompt_n"] for x in rows)
    cache_n = sum(x["cache_n"] for x in rows)
    return {
        "n_requests": len(rows),
        "req_per_s": len(rows) / max(sum(dts), 1e-9),
        "ttft_proxy_p50_s": statistics.median(dts),
        "ttft_proxy_mean_s": statistics.mean(dts),
        "prompt_tokens_computed": prompt_n,
        "prompt_tokens_cached": cache_n,
        "cache_hit_frac": cache_n / max(prompt_n + cache_n, 1),
    }


def run_gsp(url: str, prefix_len: int, suffix_len: int, n_predict: int,
            num_groups: int, prompts_per_group: int, concurrency: int) -> dict[str, Any]:
    groups = []
    for g in range(num_groups):
        prefix = gen_tokens_approx(prefix_len, f"P{g}_")
        groups.append(prefix)

    jobs = []
    for g in range(num_groups):
        for i in range(prompts_per_group):
            suffix = gen_tokens_approx(suffix_len, f"S{g}_{i}_")
            jobs.append(make_prompt(groups[g], suffix))

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(completion, url, p, n_predict) for p in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    return summarize(rows)


def run_dualagent(url: str, prefix_len: int, turns: int, n_predict: int) -> dict[str, Any]:
    shared = gen_tokens_approx(prefix_len, "SYS_")
    a_ctx = shared
    b_ctx = shared
    rows = []
    for t in range(turns):
        a_ctx = a_ctx + "\n" + gen_tokens_approx(64, f"A{t}_")
        rows.append(completion(url, a_ctx, n_predict))
        b_ctx = b_ctx + "\n" + gen_tokens_approx(64, f"B{t}_")
        rows.append(completion(url, b_ctx, n_predict))
    # Focus on turns after the first pair (cross-eviction region)
    warm = rows[2:] if len(rows) > 2 else rows
    return {"all": summarize(rows), "after_first_pair": summarize(warm)}


def run_multiturn(url: str, prefix_len: int, turns: int, n_predict: int) -> dict[str, Any]:
    ctx = gen_tokens_approx(prefix_len, "M_")
    rows = []
    for t in range(turns):
        ctx = ctx + "\n" + gen_tokens_approx(32, f"T{t}_")
        rows.append(completion(url, ctx, n_predict))
    return summarize(rows)


def run_unique(url: str, prompt_len: int, n_req: int, n_predict: int, concurrency: int) -> dict[str, Any]:
    jobs = [gen_tokens_approx(prompt_len, f"U{i}_") for i in range(n_req)]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(completion, url, p, n_predict) for p in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    return summarize(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--workload", choices=["gsp", "dualagent", "multiturn", "unique", "all"], default="all")
    ap.add_argument("--prefix-len", type=int, default=512)
    ap.add_argument("--suffix-len", type=int, default=64)
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--prompts-per-group", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--n-unique", type=int, default=16)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out: dict[str, Any] = {"label": args.label, "url": args.url, "workloads": {}}

    if args.workload in ("gsp", "all"):
        out["workloads"]["gsp"] = run_gsp(
            args.url, args.prefix_len, args.suffix_len, args.n_predict,
            args.num_groups, args.prompts_per_group, args.concurrency)
    if args.workload in ("dualagent", "all"):
        out["workloads"]["dualagent"] = run_dualagent(args.url, args.prefix_len, args.turns, args.n_predict)
    if args.workload in ("multiturn", "all"):
        out["workloads"]["multiturn"] = run_multiturn(args.url, args.prefix_len, args.turns, args.n_predict)
    if args.workload in ("unique", "all"):
        out["workloads"]["unique"] = run_unique(
            args.url, args.prefix_len, args.n_unique, args.n_predict, args.concurrency)

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
