#!/usr/bin/env python3
"""Compare preserved base/candidate outputs from test-dsv4-validation.sh."""

from __future__ import annotations

import argparse
import json
import math
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


def require_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} is not a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def validate_response(value: dict, name: str) -> dict:
    content = value.get("content")
    prompt = value.get("prompt")
    tokens = value.get("tokens")
    if not isinstance(content, str) or not content:
        raise ValueError(f"{name}: content is missing or empty")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{name}: prompt is missing or empty")
    if not isinstance(tokens, list) or not tokens or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
        raise ValueError(f"{name}: tokens must be a nonempty integer list")
    for key in ("tokens_evaluated", "tokens_predicted"):
        field = value.get(key)
        if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
            raise ValueError(f"{name}: {key} must be a positive integer")

    timings = value.get("timings")
    if not isinstance(timings, dict):
        raise ValueError(f"{name}: timings is not an object")
    prompt_n = require_number(timings.get("prompt_n"), f"{name}: timings.prompt_n", positive=True)
    prompt_ms = require_number(timings.get("prompt_ms"), f"{name}: timings.prompt_ms", positive=True)
    prompt_tps = require_number(timings.get("prompt_per_second"), f"{name}: timings.prompt_per_second", positive=True)
    require_number(timings.get("cache_n"), f"{name}: timings.cache_n")
    expected_tps = prompt_n * 1000.0 / prompt_ms
    if not math.isclose(prompt_tps, expected_tps, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{name}: prompt_per_second is inconsistent with prompt_n/prompt_ms")
    return timings


def main() -> int:
    args = parse_args()
    base_root = args.base.resolve()
    candidate_root = args.candidate.resolve()
    comparisons: dict[str, dict] = {}

    for mode in MODES:
        for request in REQUESTS:
            base = load(base_root, mode, request)
            candidate = load(candidate_root, mode, request)
            name = f"{mode}/{request}"
            base_timings = validate_response(base, f"base/{name}")
            candidate_timings = validate_response(candidate, f"candidate/{name}")
            for key in ("content", "tokens", "prompt", "tokens_evaluated", "tokens_predicted"):
                if base[key] != candidate[key]:
                    raise ValueError(f"{name}: base and candidate differ for {key}")
            comparisons[name] = {
                "content_equal": True,
                "tokens_equal": True,
                "base_cache_n": base_timings["cache_n"],
                "candidate_cache_n": candidate_timings["cache_n"],
            }

    performance: dict[str, dict] = {}
    print("mode\tprompt_n\tbase_tps\tcandidate_tps\tdelta_pct\tbase_ms\tcandidate_ms")
    for mode in MODES:
        base = load(base_root, mode, "first")
        candidate = load(candidate_root, mode, "first")
        base_timings = validate_response(base, f"base/{mode}/first")
        candidate_timings = validate_response(candidate, f"candidate/{mode}/first")
        base_n = base_timings["prompt_n"]
        candidate_n = candidate_timings["prompt_n"]
        if base_n != candidate_n:
            raise ValueError(f"{mode}/first: mismatched prompt_n")
        base_tps = base_timings["prompt_per_second"]
        candidate_tps = candidate_timings["prompt_per_second"]
        base_ms = base_timings["prompt_ms"]
        candidate_ms = candidate_timings["prompt_ms"]
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