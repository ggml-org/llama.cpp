#!/usr/bin/env python3
"""Run adaptive graph-resident Tessera MoE calibration rounds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SAMPLER_PATH = Path(__file__).with_name("moe-sampler.py")
SPEC = importlib.util.spec_from_file_location("tessera_moe_sampler", SAMPLER_PATH)
SAMPLER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SAMPLER
SPEC.loader.exec_module(SAMPLER)


def namespace(**values):
    return argparse.Namespace(**values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adaptive Tessera MoE imatrix calibration"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-index", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--llama-imatrix",
        default="/Volumes/Julian T7/llama-cpp-build/bin/llama-imatrix",
    )
    parser.add_argument(
        "--gguf-py",
        default="/Users/user/Developer/GitHub/llama.cpp/gguf-py",
    )
    parser.add_argument("--initial-samples", type=int, default=128)
    parser.add_argument("--step-samples", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--minimum-expert-count", type=int, default=16)
    parser.add_argument("--coverage-percentile", type=float, default=5.0)
    parser.add_argument("--stability-p95", type=float, default=0.02)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("imatrix_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "sampler-state.json"
    batch_path = work / "pending-calibration.txt"
    if not state_path.exists():
        SAMPLER.initialize(namespace(
            index=args.corpus_index,
            state=str(state_path),
            batch_output=str(batch_path),
            seed=640,
            initial_samples=args.initial_samples,
            step_samples=args.step_samples,
            max_samples=args.max_samples,
            minimum_expert_count=args.minimum_expert_count,
            coverage_percentile=args.coverage_percentile,
            stability_p95=args.stability_p95,
            stable_rounds=args.stable_rounds,
        ))
    while True:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["complete"]:
            break
        round_number = state["round"]
        output = work / f"round-{round_number:02d}.imatrix.gguf"
        command = [
            args.llama_imatrix,
            "-m", args.model,
            "-f", str(batch_path),
            "-o", str(output),
            "--output-format", "gguf",
            "--no-ppl",
            "--output-frequency", "0",
            "--ctx-size", str(args.ctx_size),
            "--batch-size", str(args.batch_size),
            "--ubatch-size", str(args.ubatch_size),
            "--gpu-layers", str(args.gpu_layers),
        ]
        if round_number > 0:
            previous = work / f"round-{round_number - 1:02d}.imatrix.gguf"
            command.extend(["--in-file", str(previous)])
        command.extend(args.imatrix_args)
        log_path = work / f"round-{round_number:02d}.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        SAMPLER.advance(namespace(
            state=str(state_path),
            imatrix=str(output),
            batch_output=str(batch_path),
            gguf_py=args.gguf_py,
        ))
    final = json.loads(state_path.read_text(encoding="utf-8"))
    final_imatrix = work / f"round-{final['history'][-1]['round']:02d}.imatrix.gguf"
    (work / "result.json").write_text(
        json.dumps({
            "schema": "llama.tessera.moe-calibration-result.v1",
            "complete": True,
            "stop_reason": final["stop_reason"],
            "samples": len(final["selected_ids"]),
            "rounds": len(final["history"]),
            "imatrix": str(final_imatrix.resolve()),
            "coverage": final["history"][-1],
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
