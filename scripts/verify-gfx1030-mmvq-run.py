#!/usr/bin/env python3
"""Verify MMVQ correctness/benchmark artifacts without using GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


def manifest(path: Path) -> dict:
    data = json.loads((path / "manifest.json").read_text())
    if data.get("failed"):
        raise RuntimeError(f"failed manifest: {path}")
    if any(x.get("returncode") != 0 for x in data.get("commands", [])):
        raise RuntimeError(f"nonzero command in: {path}")
    return data


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--correctness", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--simple-sweep", type=Path, required=True)
    p.add_argument("--k-sweep", type=Path, required=True)
    args = p.parse_args()
    try:
        correct = manifest(args.correctness)
        tests = sorted(args.correctness.glob("tests/*.stdout.log"))
        if len(tests) != 5:
            raise RuntimeError(f"expected five correctness logs, found {len(tests)}")
        for path in tests:
            text = path.read_text(errors="replace")
            if not re.search(r"414/414 tests passed", text):
                raise RuntimeError(f"correctness count missing: {path}")
        native_envs = [x["env"] for x in correct["commands"] if "native" in x["label"]]
        if {e.get("GGML_HIP_GFX1030_NATIVE") for e in native_envs} != {"1"}:
            raise RuntimeError("native correctness commands did not record the native flag")
        if {e.get("GGML_HIP_MMVQ_Q8_1_BLOCK_SIZE") for e in native_envs} != {"32", "64", "128", "256"}:
            raise RuntimeError("native correctness layout set is incomplete")

        bench = manifest(args.benchmark)
        bench_logs = sorted(args.benchmark.glob("bench/*.stdout.log"))
        if len(bench_logs) != 15:
            raise RuntimeError(f"expected 15 benchmark logs, found {len(bench_logs)}")
        for path in bench_logs:
            text = path.read_text(errors="replace")
            if "Backend ROCm0:" not in text or "MUL_MAT(" not in text:
                raise RuntimeError(f"benchmark result missing: {path}")

        for sweep in (args.simple_sweep, args.k_sweep):
            data = json.loads((sweep / "manifest.json").read_text())
            if not data.get("restored") or len(data.get("results", [])) != 5:
                raise RuntimeError(f"incomplete sweep: {sweep}")
            if any(x.get("build_rc") != 0 or x.get("run_rc") != 0 for x in data["results"]):
                raise RuntimeError(f"failed sweep member: {sweep}")

        source = (args.root / "ggml/src/ggml-cuda/mmvq.cu").read_text()
        if "GGML_HIP_RDNA2_MMVQ_NATIVE_NWARPS_SIMPLE" in source:
            raise RuntimeError("native simple nwarps policy is still present")
        if "GGML_HIP_RDNA2_MMVQ_NATIVE_NWARPS_K" in source:
            raise RuntimeError("native K nwarps policy is still present")
        if "calc_nwarps_gfx1030" in source:
            raise RuntimeError("native nwarps dispatch helper is still present")
        print("verified MMVQ artifacts")
        print("correctness_logs=5 benchmark_logs=15 simple_sweep=5 k_sweep=5")
        print("all commands returned zero; correctness=414/414; final source uses stock calc_nwarps()")
        return 0
    except (OSError, KeyError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())