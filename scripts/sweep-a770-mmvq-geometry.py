#!/usr/bin/env python3
"""Build, correctness-check, and benchmark the Arc A770 MMVQ geometry matrix.

The candidate set starts from MMV_Y={1,2,4} x
MMVQ_NUM_SUBGROUPS={4,8,16,32}, then excludes combinations whose SIMD32
workgroup exceeds the A770 limit. Builds may run concurrently; every GPU
phase runs sequentially and aborts if the render node has a foreign holder.
Source oneAPI before invoking this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

A770_MAX_WORK_ITEMS: int = 1024



def _cell_workgroup_size(y: int, sg: int) -> int | None:
    """Return the SIMD32 workgroup size for a (y,sg) cell, or None on bad input."""
    if isinstance(y, int) and isinstance(sg, int):
        return y * sg * 32
    return None


def _cell_valid(y: int, sg: int) -> bool:
    workgroup_size = _cell_workgroup_size(y, sg)
    return workgroup_size is not None and workgroup_size <= A770_MAX_WORK_ITEMS


_ALL_CELL_CONFIGS = [(y, sg) for y in (1, 2, 4) for sg in (4, 8, 16, 32)]

CELLS = tuple(c for c in _ALL_CELL_CONFIGS if _cell_valid(*c))

GPU_FAULT_RE = re.compile(
    r"reset|hang|timeout|GPU HANG|device lost|i915.*error|xe.*error", re.I
)

RE_GATE_FAIL = re.compile(r"\b(\d+)\s+GATE-FAIL\b")


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
        help="Parent directory for build-p58-yY-sgSG-IDENTITY directories.",
    )
    parser.add_argument(
        "--tag",
        help="Build/output identity; defaults to <sha9>-<fingerprint>. Use --tag to supply a fixed identity (no fingerprint appended).",
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


def source_fingerprint(source: Path) -> str:
    """Hash tracked diffs plus untracked paths and contents deterministically."""
    diff = run(
        ["git", "-C", str(source), "diff", "--binary", "HEAD", "--", "."],
        cwd=source,
    )
    untracked = run(
        [
            "git",
            "-C",
            str(source),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=source,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        return ""

    paths = sorted(path for path in untracked.stdout.split("\0") if path)
    if not diff.stdout and not paths:
        return ""

    digest = hashlib.sha256(diff.stdout.encode("utf-8"))
    for relative in paths:
        path = source / relative
        content = (
            os.readlink(path).encode("utf-8")
            if path.is_symlink()
            else path.read_bytes()
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def source_tag(source: Path) -> str:
    proc = run(["git", "-C", str(source), "rev-parse", "--short=9", "HEAD"])
    if proc.returncode != 0:
        raise SweepError(f"cannot resolve source HEAD: {proc.stdout.strip()}")
    head = proc.stdout.strip()
    fp = source_fingerprint(source)
    return f"{head}-{fp}" if fp else head


def build_dir(build_root: Path, manifest_identity: str, y: int, sg: int) -> Path:
    return build_root / f"build-p58-y{y}-sg{sg}-{manifest_identity}"


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SweepError(f"required executable not found on PATH: {name}")
    return path


def cmake_cache_has_flags(cache: str, requested_flags: str) -> bool:
    prefix = "CMAKE_CXX_FLAGS:STRING="
    configured = next(
        (
            line.removeprefix(prefix)
            for line in cache.splitlines()
            if line.startswith(prefix)
        ),
        "",
    )
    return set(requested_flags.split()).issubset(configured.split())


def build_one(
    args: argparse.Namespace, manifest_identity: str, y: int, sg: int
) -> dict[str, Any]:
    out = build_dir(args.build_root, manifest_identity, y, sg)
    out.mkdir(parents=True, exist_ok=True)
    flags = f"-DGGML_SYCL_MMV_Y={y} -DGGML_SYCL_MMVQ_NUM_SUBGROUPS={sg}"
    configure = [
        require_tool("cmake"),
        "-S",
        str(args.source),
        "-B",
        str(out),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={require_tool('icx')}",
        f"-DCMAKE_CXX_COMPILER={require_tool('icpx')}",
        f"-DCMAKE_CXX_FLAGS={flags}",
        "-DGGML_SYCL=ON",
        "-DGGML_SYCL_TARGET=INTEL",
        "-DGGML_SYCL_F16=ON",
        "-DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TESTS=ON",
    ]
    configured = run(configure, output=out / "p58-configure.log")
    if configured.returncode != 0:
        raise SweepError(
            f"configure failed for y={y} sg={sg}: {out / 'p58-configure.log'}"
        )
    built = run(
        [
            require_tool("cmake"),
            "--build",
            str(out),
            f"-j{args.jobs}",
            "--target",
            "llama-bench",
            "test-sycl-turbo-correctness",
        ],
        output=out / "p58-build.log",
    )
    if built.returncode != 0:
        raise SweepError(f"build failed for y={y} sg={sg}: {out / 'p58-build.log'}")
    cache = (out / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    if not cmake_cache_has_flags(cache, flags):
        raise SweepError(f"CMakeCache lost requested flags for y={y} sg={sg}")
    return {"y": y, "subgroups": sg, "success": True}


def build_matrix(
    args: argparse.Namespace, manifest_identity: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.parallel_builds
    ) as pool:
        futures = {
            pool.submit(build_one, args, manifest_identity, y, sg): (y, sg)
            for y, sg in CELLS
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
    raise SweepError(f"cannot prove render node is idle: {render_node}\n{detail}")


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


def parse_gate_fail(summary: str) -> bool:
    """Return True only when the numeric GATE-FAIL count is exactly 0."""
    if not summary:
        return False
    match = RE_GATE_FAIL.search(summary)
    if not match:
        return False
    fail_count = int(match.group(1))
    return fail_count == 0


def _correctness_env() -> dict[str, str]:
    env = os.environ.copy()
    for knob in (
        "GGML_SYCL_FA_XMX",
        "GGML_SYCL_FA_FORCE_VEC_STANDARD",
        "GGML_SYCL_FA_Q8_GQA_TILE",
        "GGML_SYCL_Q8_KV_QUANTS_FIRST",
        "LLAMA_ENABLE_INNERQ",
        "TURBO_LAYER_ADAPTIVE",
        "TURBO_AUTO_ASYMMETRIC",
    ):
        env.pop(knob, None)
    env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
    return env


def correctness_matrix(
    args: argparse.Namespace, tag: str, manifest_identity: str
) -> list[dict[str, Any]]:
    env = _correctness_env()
    results: list[dict[str, Any]] = []
    for y, sg in CELLS:
        require_sole_tenancy(args.render_node)
        out = build_dir(args.build_root, manifest_identity, y, sg)
        binary = out / "bin/test-sycl-turbo-correctness"
        if not binary.is_file():
            raise SweepError(f"missing correctness binary: {binary}")
        before = dmesg_faults()
        proc = run(
            [require_tool("timeout"), str(args.timeout), str(binary)],
            env=env,
            output=out / "p58-correctness.log",
        )
        after = dmesg_faults()
        new_faults = added_suffix(before, after)
        summary = next(
            (
                line
                for line in proc.stdout.splitlines()
                if line.startswith("== summary:")
            ),
            "",
        )
        valid = proc.returncode == 0 and parse_gate_fail(summary) and not new_faults
        result = {
            "y": y,
            "subgroups": sg,
            "returncode": proc.returncode,
            "summary": summary,
            "new_gpu_faults": new_faults,
            "valid": valid,
            "source": str(args.source),
            "tag": tag,
            "manifest_identity": manifest_identity,
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
        if not name:
            raise SweepError(f"--model name is empty: {value}")
        if path.is_file() and path.stat().st_size > 0 and name in models:
            raise SweepError(f"duplicate model label '{name}' - refusing to overwrite")
        if not path.is_file() or path.stat().st_size == 0:
            raise SweepError(f"invalid benchmark model: {value}")
        models[name] = path
    return models


def _has_matching_correctness(
    manifest: dict[str, Any], target_y: int, target_sg: int, manifest_identity: str
) -> bool:
    """Check that a valid correctness record exists under the same identity."""
    for rec in manifest.get("correctness", []):
        if (
            rec.get("y") == target_y
            and rec.get("subgroups") == target_sg
            and rec.get("manifest_identity") == manifest_identity
            and rec.get("valid") is True
        ):
            return True
    return False


def output_root(args: argparse.Namespace, manifest_identity: str) -> Path:
    return (
        args.out_root or Path(f"/tmp/a770-mmvq-geometry-{manifest_identity}")
    ).resolve()


def benchmark_matrix(
    args: argparse.Namespace, manifest_identity: str
) -> list[dict[str, Any]]:
    models = parse_models(args.model)
    if not models:
        raise SweepError("benchmark phase requires at least one --model NAME=PATH")
    harness = args.source / "scripts/bench-a770-fork-unique.py"
    if not harness.is_file():
        raise SweepError(f"product benchmark harness not found: {harness}")

    out_root = output_root(args, manifest_identity)

    results: list[dict[str, Any]] = []
    manifest_path = out_root / "manifest.json"
    if not manifest_path.is_file():
        raise SweepError("no manifest.json found for benchmark-only correctness check")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Fail closed: every benchmark geometry must have valid correctness evidence
    baseline_manifest_id = manifest.get("manifest_identity")
    if baseline_manifest_id is None:
        raise SweepError(
            "manifest lacks 'manifest_identity'; correctness evidence not found"
        )
    for y, sg in CELLS:
        baseline_key = f"y{y}-sg{sg}"
        if not _has_matching_correctness(manifest, y, sg, baseline_manifest_id):
            raise SweepError(
                f"no valid correctness record for {baseline_key} "
                f"(identity={baseline_manifest_id}) - benchmark cannot proceed"
            )

    baseline = (
        build_dir(
            args.build_root, manifest_identity, args.baseline_y, args.baseline_subgroups
        )
        / "bin"
    )
    for y, sg in CELLS:
        candidate = build_dir(args.build_root, manifest_identity, y, sg) / "bin"
        for model_name, model in models.items():
            out_dir = out_root / f"y{y}-sg{sg}-{model_name}"
            if out_dir.exists():
                raise SweepError(
                    f"refusing to mix with existing result directory: {out_dir}"
                )
            require_sole_tenancy(args.render_node)
            env = os.environ.copy()
            env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
            argv = [
                sys.executable,
                str(harness),
                "--campaign",
                "product",
                "--bin-dir",
                str(baseline),
                "--candidate-bin-dir",
                str(candidate),
                "--model",
                str(model),
                "--depths",
                "0",
                "--kv-types",
                "q8_0/q8_0",
                "--repetitions",
                str(args.repetitions),
                "--model-layers",
                str(args.model_layers),
                "--query-heads",
                str(args.query_heads),
                "--head-dim",
                str(args.head_dim),
                "--baseline-label",
                f"y{args.baseline_y}-sg{args.baseline_subgroups}",
                "--candidate-label",
                f"y{y}-sg{sg}",
                "--env",
                "GGML_SYCL_DEBUG=0",
                "--out-dir",
                str(out_dir),
                "--timeout",
                str(args.timeout),
            ]
            proc = run(argv, cwd=args.source, env=env)
            result = {
                "y": y,
                "subgroups": sg,
                "model": model_name,
                "returncode": proc.returncode,
                "product_json": str(out_dir / "product.json"),
                "oneapi_device_selector": "level_zero:0",
            }
            results.append(result)
            print(
                f"benchmark y={y} sg={sg} model={model_name} exit={proc.returncode}",
                flush=True,
            )
            if proc.returncode != 0:
                raise SweepError(
                    f"benchmark failed for y={y} sg={sg} model={model_name}"
                )
    return results


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist completed phases for later fail-closed reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.build_root = args.build_root.resolve()
    if args.jobs < 1 or args.parallel_builds < 1 or args.repetitions < 3:
        print(
            "error: jobs/parallel-builds must be positive and repetitions >= 3",
            file=sys.stderr,
        )
        return 2
    try:
        if args.tag:
            tag = args.tag
            manifest_identity = tag
        else:
            tag = source_tag(args.source)
            manifest_identity = tag
        manifest_path = output_root(args, manifest_identity) / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source") != str(args.source)
                or manifest.get("manifest_identity") != manifest_identity
            ):
                raise SweepError(
                    f"existing manifest identity mismatch: {manifest_path}"
                )
        else:
            manifest = {
                "source": str(args.source),
                "tag": tag,
                "manifest_identity": manifest_identity,
                "cells": [{"y": y, "subgroups": sg} for y, sg in CELLS],
            }
        if args.phase in ("build", "all"):
            manifest["builds"] = build_matrix(args, manifest_identity)
            write_manifest(manifest_path, manifest)
        if args.phase in ("correctness", "all"):
            manifest["correctness"] = correctness_matrix(args, tag, manifest_identity)
            write_manifest(manifest_path, manifest)
        if args.phase in ("benchmark", "all"):
            manifest["benchmarks"] = benchmark_matrix(args, manifest_identity)
            write_manifest(manifest_path, manifest)
        print(f"wrote {manifest_path}")
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
