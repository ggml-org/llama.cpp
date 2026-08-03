#!/usr/bin/env python3
"""Compare preserved base/candidate outputs from test-dsv4-validation.sh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODES = ("reference", "tensor")
REQUESTS = ("first", "continuation", "replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path, help="base DSV4_OUTPUT_DIR")
    parser.add_argument("candidate", type=Path, help="candidate DSV4_OUTPUT_DIR")
    parser.add_argument("--json", type=Path, help="optional summary output")
    return parser.parse_args()


def load(root: Path, mode: str, request: str) -> dict:
    path = root / mode / f"{request}.json"
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def main() -> int:
    args = parse_args()
    base_root = args.base.resolve()
    candidate_root = args.candidate.resolve()
    comparisons: dict[str, dict] = {}

    for mode in MODES:
        for request in REQUESTS:
            base = load(base_root, mode, request)
            candidate = load(candidate_root, mode, request)
            for key in ("content", "tokens", "prompt", "tokens_evaluated", "tokens_predicted"):
                if base.get(key) != candidate.get(key):
                    raise ValueError(f"{mode}/{request}: base and candidate differ for {key}")
            name = f"{mode}/{request}"
            comparisons[name] = {
                "content_equal": True,
                "tokens_equal": True,
                "base_cache_n": base.get("timings", {}).get("cache_n"),
                "candidate_cache_n": candidate.get("timings", {}).get("cache_n"),
            }

    performance: dict[str, dict] = {}
    print("mode\tprompt_n\tbase_tps\tcandidate_tps\tdelta_pct\tbase_ms\tcandidate_ms")
    for mode in MODES:
        base = load(base_root, mode, "first")
        candidate = load(candidate_root, mode, "first")
        base_timings = base.get("timings", {})
        candidate_timings = candidate.get("timings", {})
        base_n = base_timings.get("prompt_n")
        candidate_n = candidate_timings.get("prompt_n")
        if not isinstance(base_n, (int, float)) or base_n <= 0 or base_n != candidate_n:
            raise ValueError(f"{mode}/first: invalid or mismatched prompt_n")
        base_tps = base_timings.get("prompt_per_second")
        candidate_tps = candidate_timings.get("prompt_per_second")
        base_ms = base_timings.get("prompt_ms")
        candidate_ms = candidate_timings.get("prompt_ms")
        if not all(isinstance(value, (int, float)) and value > 0 for value in (base_tps, candidate_tps, base_ms, candidate_ms)):
            raise ValueError(f"{mode}/first: invalid prompt timings")
        delta_pct = 100.0 * (candidate_tps / base_tps - 1.0)
        performance[mode] = {
            "prompt_n": base_n,
            "base_tps": base_tps,
            "candidate_tps": candidate_tps,
            "delta_pct": delta_pct,
            "base_ms": base_ms,
            "candidate_ms": candidate_ms,
        }
        print(f"{mode}\t{base_n}\t{base_tps:.3f}\t{candidate_tps:.3f}\t{delta_pct:+.2f}\t{base_ms:.3f}\t{candidate_ms:.3f}")

    print("base/candidate content, token IDs, prompts, and token counts match for all six responses")
    if args.json:
        output = {
            "base": str(base_root),
            "candidate": str(candidate_root),
            "comparisons": comparisons,
            "performance": performance,
        }
        args.json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)