#!/usr/bin/env python3
"""Build, correctness-check, and benchmark the Arc A770 MMVQ geometry matrix.

The matrix is the Cartesian product MMV_Y={1,2,4} x
MMVQ_NUM_SUBGROUPS={4,8,16,32}. Builds may run concurrently; every GPU phase
runs sequentially and aborts if the render node has a foreign holder. Source
oneAPI before invoking this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

CELLS = [(y, sg) for y in (1, 2, 4) for sg in (4, 8, 16, 32)]
GPU_FAULT_RE = re.compile(
    r"reset|hang|timeout|GPU HANG|device lost|i915.*error|xe.*error", re.I
)


class SweepError(RuntimeError):
    """A reproducibility or gate failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("build", "correctness", "benchmark", "all"),
        default="all",
    )
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path.home(),
        help="Parent directory for build-p58-yY-sgSG-TAG directories.",
    )
    parser.add_argument(
        "--tag",
        help="Build/output identity; defaults to the source HEAD short SHA.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Benchmark model; repeat for cross-model promotion gates.",
    )
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--parallel-builds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    parser.add_argument("--baseline-y", type=int, default=1)
    parser.add_argument("--baseline-subgroups", type=int, default=16)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    return parser.parse_args()


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    output: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(proc.stdout, encoding="utf-8")
    return proc


def source_tag(source: Path) -> str:
    proc = run(["git", "-C", str(source), "rev-parse", "--short=9", "HEAD"])
    if proc.returncode != 0:
        raise SweepError(f"cannot resolve source HEAD: {proc.stdout.strip()}")
    return proc.stdout.strip()


def build_dir(build_root: Path, tag: str, y: int, sg: int) -> Path:
    return build_root / f"build-p58-y{y}-sg{sg}-{tag}"


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SweepError(f"required executable not found on PATH: {name}")
    return path


def build_one(args: argparse.Namespace, tag: str, y: int, sg: int) -> dict[str, Any]:
    out = build_dir(args.build_root, tag, y, sg)
    out.mkdir(parents=True, exist_ok=True)
    flags = f"-DGGML_SYCL_MMV_Y={y} -DGGML_SYCL_MMVQ_NUM_SUBGROUPS={sg}"
    configure = [
        require_tool("cmake"), "-S", str(args.source), "-B", str(out), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={require_tool('icx')}",
        f"-DCMAKE_CXX_COMPILER={require_tool('icpx')}",
        f"-DCMAKE_CXX_FLAGS={flags}",
        "-DGGML_SYCL=ON", "-DGGML_SYCL_TARGET=INTEL", "-DGGML_SYCL_F16=ON",
        "-DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON", "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TESTS=ON",
    ]
    configured = run(configure, output=out / "p58-configure.log")
    if configured.returncode != 0:
        raise SweepError(f"configure failed for y={y} sg={sg}: {out / 'p58-configure.log'}")
    built = run(
        [require_tool("cmake"), "--build", str(out), f"-j{args.jobs}", "--target", "llama-bench", "test-sycl-turbo-correctness"],
        output=out / "p58-build.log",
    )
    if built.returncode != 0:
        raise SweepError(f"build failed for y={y} sg={sg}: {out / 'p58-build.log'}")
    cache = (out / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    if f"CMAKE_CXX_FLAGS:STRING={flags}" not in cache:
        raise SweepError(f"CMakeCache lost requested flags for y={y} sg={sg}")
    return {"y": y, "subgroups": sg, "build_dir": str(out), "flags": flags}


def build_matrix(args: argparse.Namespace, tag: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_builds) as pool:
        futures = {
            pool.submit(build_one, args, tag, y, sg): (y, sg) for y, sg in CELLS
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"built y={result['y']} sg={result['subgroups']}", flush=True)
    return sorted(results, key=lambda item: (item["y"], item["subgroups"]))


def require_sole_tenancy(render_node: str) -> None:
    proc = run([require_tool("fuser"), "-v", render_node])
    output = proc.stdout.strip()
    if proc.returncode == 1 and not output:
        return
    detail = output or f"fuser exited {proc.returncode} without holder details"
    raise SweepError(
        f"cannot prove render node is idle: {render_node}\n{detail}"
    )


def dmesg_faults() -> list[str]:
    proc = run([require_tool("sudo"), "-n", require_tool("dmesg"), "--ctime"])
    if proc.returncode != 0:
        raise SweepError(f"cannot capture privileged dmesg: {proc.stdout.strip()}")
    return [line for line in proc.stdout.splitlines() if GPU_FAULT_RE.search(line)]


def added_suffix(before: list[str], after: list[str]) -> list[str]:
    max_overlap = min(len(before), len(after))
    for overlap in range(max_overlap, 0, -1):
        if before[-overlap:] == after[:overlap]:
            return after[overlap:]
    return list(after)


def correctness_matrix(args: argparse.Namespace, tag: str) -> list[dict[str, Any]]:
    env = os.environ.copy()
    env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
    results: list[dict[str, Any]] = []
    for y, sg in CELLS:
        require_sole_tenancy(args.render_node)
        out = build_dir(args.build_root, tag, y, sg)
        binary = out / "bin/test-sycl-turbo-correctness"
        if not binary.is_file():
            raise SweepError(f"missing correctness binary: {binary}")
        before = dmesg_faults()
        proc = run([require_tool("timeout"), str(args.timeout), str(binary)], env=env, output=out / "p58-correctness.log")
        after = dmesg_faults()
        new_faults = added_suffix(before, after)
        summary = next(
            (line for line in proc.stdout.splitlines() if line.startswith("== summary:")),
            "",
        )
        valid = proc.returncode == 0 and "0 GATE-FAIL" in summary and not new_faults
        result = {
            "y": y,
            "subgroups": sg,
            "returncode": proc.returncode,
            "summary": summary,
            "new_gpu_faults": new_faults,
            "valid": valid,
        }
        results.append(result)
        print(f"correctness y={y} sg={sg} valid={valid}", flush=True)
        if not valid:
            raise SweepError(f"correctness gate failed for y={y} sg={sg}")
    return results


def parse_models(raw_models: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in raw_models:
        if "=" not in value:
            raise SweepError(f"--model must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or not path.is_file() or path.stat().st_size == 0:
            raise SweepError(f"invalid benchmark model: {value}")
        models[name] = path
    return models


def benchmark_matrix(args: argparse.Namespace, tag: str) -> list[dict[str, Any]]:
    models = parse_models(args.model)
    if not models:
        raise SweepError("benchmark phase requires at least one --model NAME=PATH")
    harness = args.source / "scripts/bench-a770-fork-unique.py"
    if not harness.is_file():
        raise SweepError(f"product benchmark harness not found: {harness}")
    out_root = (args.out_root or Path(f"/tmp/a770-mmvq-geometry-{tag}")).resolve()
    baseline = build_dir(
        args.build_root, tag, args.baseline_y, args.baseline_subgroups
    ) / "bin"
    results: list[dict[str, Any]] = []
    for y, sg in CELLS:
        candidate = build_dir(args.build_root, tag, y, sg) / "bin"
        for model_name, model in models.items():
            out_dir = out_root / f"y{y}-sg{sg}-{model_name}"
            if out_dir.exists():
                raise SweepError(f"refusing to mix with existing result directory: {out_dir}")
            require_sole_tenancy(args.render_node)
            argv = [
                sys.executable, str(harness), "--campaign", "product",
                "--bin-dir", str(baseline), "--candidate-bin-dir", str(candidate),
                "--model", str(model), "--depths", "0", "--kv-types", "q8_0/q8_0",
                "--repetitions", str(args.repetitions),
                "--model-layers", str(args.model_layers),
                "--query-heads", str(args.query_heads), "--head-dim", str(args.head_dim),
                "--baseline-label", f"y{args.baseline_y}-sg{args.baseline_subgroups}",
                "--candidate-label", f"y{y}-sg{sg}", "--env", "GGML_SYCL_DEBUG=0",
                "--out-dir", str(out_dir), "--timeout", str(args.timeout),
            ]
            proc = run(argv, cwd=args.source)
            result = {
                "y": y,
                "subgroups": sg,
                "model": model_name,
                "returncode": proc.returncode,
                "product_json": str(out_dir / "product.json"),
            }
            results.append(result)
            print(f"benchmark y={y} sg={sg} model={model_name} exit={proc.returncode}", flush=True)
            if proc.returncode != 0:
                raise SweepError(f"benchmark failed for y={y} sg={sg} model={model_name}")
    return results




def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.build_root = args.build_root.resolve()
    if args.jobs < 1 or args.parallel_builds < 1 or args.repetitions < 2:
        print("error: jobs/parallel-builds must be positive and repetitions >= 2", file=sys.stderr)
        return 2
    try:
        tag = args.tag or source_tag(args.source)
        manifest_path = (
            args.out_root or Path(f"/tmp/a770-mmvq-geometry-{tag}")
        ).resolve() / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source") != str(args.source)
                or manifest.get("tag") != tag
            ):
                raise SweepError(
                    f"existing manifest identity mismatch: {manifest_path}"
                )
        else:
            manifest = {
                "source": str(args.source),
                "tag": tag,
                "cells": [{"y": y, "subgroups": sg} for y, sg in CELLS],
            }
        if args.phase in ("build", "all"):
            manifest["builds"] = build_matrix(args, tag)
        if args.phase in ("correctness", "all"):
            manifest["correctness"] = correctness_matrix(args, tag)
        if args.phase in ("benchmark", "all"):
            manifest["benchmarks"] = benchmark_matrix(args, tag)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path}")
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
