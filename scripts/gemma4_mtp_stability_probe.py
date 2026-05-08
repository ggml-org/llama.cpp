#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request


PROMPTS = [
    "Write a dense technical explanation of speculative decoding, including acceptance, rejection, and why memory bandwidth matters.",
    "Explain how a GPU inference server should be benchmarked for stable token throughput across several prompts and why short runs can mislead.",
    "Describe practical optimization strategies for transformer decoding on a single RTX 4090, focusing on cache format, attention kernels, and draft models.",
]


def complete(endpoint: str, prompt: str, n_predict: int) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    out["wall_seconds"] = time.time() - start
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma4 MTP throughput stability probes against llama-server.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18191")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--n-predict", type=int, default=256)
    args = parser.parse_args()

    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        out = complete(args.endpoint, prompt, args.n_predict)
        timings = out.get("timings", {})
        item = {
            "run": i,
            "predicted_n": timings.get("predicted_n"),
            "predicted_ms": timings.get("predicted_ms"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n"),
            "draft_n_accepted": timings.get("draft_n_accepted"),
            "wall_seconds": out["wall_seconds"],
            "content_prefix": out.get("content", "")[:120],
        }
        print(json.dumps(item, sort_keys=True), flush=True)
        results.append(item)

    rates = [float(item["predicted_per_second"]) for item in results]
    summary = {
        "results": results,
        "min_predicted_per_second": min(rates),
        "avg_predicted_per_second": sum(rates) / len(rates),
        "all_ge_40": all(rate >= 40.0 for rate in rates),
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
