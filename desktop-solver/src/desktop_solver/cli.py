"""Headless CLI — useful for batch verification on the same model the GUI uses."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import scanner
from .llm_engine import EngineConfig, LLMEngine
from .solver import solve


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="GCSE math solver — headless CLI")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Path to a GGUF model")
    g.add_argument("--auto", action="store_true",
                   help="Scan the filesystem and auto-select the best GGUF")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument("--n-gpu-layers", type=int, default=0)
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p.add_argument("problem", nargs="*", help="Problem text (defaults to stdin)")
    args = p.parse_args(argv)

    if args.auto:
        models = scanner.scan()
        best = scanner.auto_pick(models)
        if best is None:
            print("error: no GGUF models found on the filesystem", file=sys.stderr)
            return 1
        print(f"[auto] selected {best.path} ({best.size_gb:.2f} GB)", file=sys.stderr)
        model_path = str(best.path)
    else:
        model_path = args.model

    problem = " ".join(args.problem).strip()
    if not problem:
        problem = sys.stdin.read().strip()
    if not problem:
        print("error: no problem provided", file=sys.stderr)
        return 1

    engine = LLMEngine(EngineConfig(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    ))
    result = solve(engine, problem)

    if args.json:
        payload = {
            "problem":      result.problem,
            "answer":       result.answer_repr,
            "latex":        result.latex,
            "code":         result.code,
            "steps":        result.steps,
            "verified":     result.verified,
            "verify_note":  result.verify_reason,
            "error":        result.error,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status = "VERIFIED" if result.verified else (
            f"UNVERIFIED ({result.verify_reason})" if result.success else "ERROR"
        )
        print(f"[PROB]: {result.problem}")
        print(f"[CODE]: {result.code}")
        print(f"[ANS]:  {result.answer_repr}")
        print(f"[STAT]: {status}")
        if result.error:
            print(f"[ERR]:  {result.error}")

    return 0 if result.success and result.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
