#!/usr/bin/env python3
"""Correctness-first gfx1030 MMVQ/layout harness.

Dry-run is the default. GPU work requires --run --allow-gpu. The benchmark
phase is intentionally separate from correctness so it cannot run by accident.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

GPU_PROGRAMS = {"llama-server", "llama-cli", "llama-bench", "test-backend-ops"}
RECORDED_ENV = (
    "GGML_HIP_GFX1030_NATIVE",
    "GGML_HIP_MMVQ_Q8_1_BLOCK_SIZE",
    "GGML_CUDA_DISABLE_GRAPHS",
    "GGML_CUDA_ALLREDUCE",
    "GGML_CUDA_P2P",
    "GGML_TP_SHARDED_OUTPUT",
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def active_gpu_processes() -> list[str]:
    try:
        text = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    result = []
    for line in text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) >= 2 and fields[1] in GPU_PROGRAMS:
            result.append(line.strip())
    return result


def parse_layouts(value: str) -> list[int]:
    try:
        values = [int(x) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated layout sizes") from exc
    if not values or any(x not in (32, 64, 128, 256) for x in values):
        raise argparse.ArgumentTypeError("layouts must be 32, 64, 128, or 256")
    return values


def parse_positive_ints(value: str) -> list[int]:
    try:
        values = [int(x) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def env_for(native: bool, layout: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if native:
        env["GGML_HIP_GFX1030_NATIVE"] = "1"
    else:
        env.pop("GGML_HIP_GFX1030_NATIVE", None)
    if layout is None:
        env.pop("GGML_HIP_MMVQ_Q8_1_BLOCK_SIZE", None)
    else:
        env["GGML_HIP_MMVQ_Q8_1_BLOCK_SIZE"] = str(layout)
    return env


def env_record(env: dict[str, str]) -> dict[str, str | None]:
    return {key: env.get(key) for key in RECORDED_ENV}


def run_logged(argv: list[str], root: Path, env: dict[str, str], out: Path,
               label: str, entries: list[dict[str, Any]], dry_run: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "label": label,
        "command": argv,
        "command_text": shlex.join(argv),
        "cwd": str(root),
        "env": env_record(env),
        "started": now(),
    }
    print(f"[{label}] {entry['command_text']}")
    if dry_run:
        entry.update({"dry_run": True, "returncode": None, "finished": now()})
        entries.append(entry)
        return 0
    stdout_path = out / f"{label}.stdout.log"
    stderr_path = out / f"{label}.stderr.log"
    start = time.monotonic()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        try:
            proc = subprocess.run(argv, cwd=root, env=env, stdout=stdout, stderr=stderr, check=False)
            rc = proc.returncode
        except OSError as exc:
            stderr.write(f"{type(exc).__name__}: {exc}\n")
            rc = 127
    entry.update({
        "dry_run": False,
        "returncode": rc,
        "elapsed_sec": time.monotonic() - start,
        "finished": now(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    })
    entries.append(entry)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--phase", choices=("correctness", "benchmark"), default="correctness")
    p.add_argument("--model", type=Path, help="required for llama-bench benchmark phase")
    p.add_argument("--engine", choices=("backend", "llama-bench"), default="backend",
                   help="benchmark engine; backend mode covers synthetic K-quant cases")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--layouts", type=parse_layouts, default=[32, 64, 128, 256])
    p.add_argument("--ops", default="MUL_MAT,MUL_MAT_ID")
    p.add_argument("--test-filter", default="type_a=(q4_0|q8_0|q4_[Kk]|q5_[Kk]|q6_[Kk])")
    p.add_argument("--backend", default="ROCm0")
    p.add_argument("--prompt-sizes", type=parse_positive_ints, default=[512, 4096, 16384])
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--run", action="store_true")
    p.add_argument("--allow-gpu", action="store_true")
    args = p.parse_args()

    args.root = args.root.resolve()
    if not (args.root / ".git").exists():
        p.error(f"not a git worktree: {args.root}")
    if args.run != args.allow_gpu:
        p.error("GPU execution requires both --run and --allow-gpu")
    if args.phase == "benchmark" and args.engine == "llama-bench":
        if args.model is None or not args.model.exists():
            p.error("--model must name an existing GGUF for llama-bench phase")
    elif args.model is not None:
        args.model = args.model.resolve()
    if args.repetitions < 1:
        p.error("--repetitions must be positive")

    bin_dir = args.root / "build" / "bin"
    test_bin = bin_dir / "test-backend-ops"
    bench_bin = bin_dir / "llama-bench"
    if not test_bin.exists():
        p.error(f"missing {test_bin}")
    if args.phase == "benchmark" and args.engine == "llama-bench" and not bench_bin.exists():
        p.error(f"missing {bench_bin}")
    if args.run:
        active = active_gpu_processes()
        if active:
            print("GPU program already active; refusing to run:\n" + "\n".join(active), file=sys.stderr)
            return 2
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = args.root / "benchmark-artifacts" / f"gfx1030-mmvq-{args.phase}-{stamp}"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "started": now(), "phase": args.phase, "engine": args.engine, "root": str(args.root),
        "model": str(args.model) if args.model else None,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "active_gpu_processes_at_start": active_gpu_processes(), "commands": entries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if args.phase == "correctness":
        conditions = [(False, None)] + [(True, layout) for layout in args.layouts]
    else:
        conditions = [(False, None)] + [(True, layout) for layout in args.layouts]
    print(f"Run directory: {args.output_dir}")
    print("Mode:", "GPU RUN" if args.run else "DRY RUN")
    print("Conditions:", [("native" if n else "stock", l) for n, l in conditions])

    failed = False
    for rep in range(args.repetitions):
        order = conditions if rep % 2 == 0 else list(reversed(conditions))
        for native, layout in order:
            label = f"rep{rep+1:02d}-{'native' if native else 'stock'}-{layout or 'default'}"
            env = env_for(native, layout)
            if args.phase == "correctness":
                cmd = [str(test_bin), "test", "-o", args.ops, "-b", args.backend]
                if args.test_filter:
                    cmd += ["-p", args.test_filter]
                rc = run_logged(cmd, args.root, env, args.output_dir / "tests", label, entries, not args.run)
            else:
                if args.engine == "backend":
                    cmd = [str(test_bin), "perf", "-o", args.ops, "-b", args.backend]
                    if args.test_filter:
                        cmd += ["-p", args.test_filter]
                else:
                    cmd = [str(bench_bin), "-m", str(args.model), "-ngl", "999", "-sm", "layer",
                           "-ts", "1/1/1/1", "-fa", "on", "-p", ",".join(map(str, args.prompt_sizes)),
                           "-n", str(args.n_gen), "-b", str(args.batch), "-ub", str(args.ubatch),
                           "-r", "1", "-o", "json"]
                rc = run_logged(cmd, args.root, env, args.output_dir / "bench", label, entries, not args.run)
            failed |= rc != 0
            metadata["commands"] = entries
            (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    metadata["finished"] = now()
    metadata["failed"] = failed
    metadata["commands"] = entries
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.run:
        print("No GPU work performed. Re-run with --run --allow-gpu after approval.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())