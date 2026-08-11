#!/usr/bin/env python3
"""Guarded stock/native gfx1030 FlashAttention harness.

The default action is a dry run. GPU work requires both --run and --allow-gpu.
The harness records commands, environments, git/build metadata, test logs,
llama-bench JSON, profiler output, and before/after ROCm snapshots in one run
 directory. It intentionally does not build the source tree.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


GPU_PROGRAMS = {"llama-server", "llama-cli", "llama-bench", "test-backend-ops"}
RECORDED_ENV = (
    "GGML_HIP_GFX1030_NATIVE",
    "GGML_CUDA_DISABLE_GRAPHS",
    "GGML_CUDA_ALLREDUCE",
    "GGML_CUDA_P2P",
    "GGML_TP_SHARDED_OUTPUT",
    "NCCL_P2P_LEVEL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HSA_NO_SCRATCH_RECLAIM",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command_text(argv: list[str]) -> str:
    return shlex.join(argv)


def run_capture(argv: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=False)
        return p.returncode, p.stdout
    except OSError as exc:
        return 127, f"{type(exc).__name__}: {exc}\n"


def git_info(root: Path) -> dict[str, Any]:
    def one(*args: str) -> str:
        rc, out = run_capture(["git", *args], root)
        return out.strip() if rc == 0 else f"<error {rc}: {out.strip()}>"
    return {
        "root": str(root),
        "commit": one("rev-parse", "HEAD"),
        "branch": one("branch", "--show-current"),
        "status": one("status", "--short"),
        "diff_stat": one("diff", "--stat"),
    }


def active_gpu_processes() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    found = []
    for line in out.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        comm = fields[1]
        if pid == os.getpid() or comm not in GPU_PROGRAMS:
            continue
        found.append(line.strip())
    return found


def parse_int_list(value: str) -> list[int]:
    result = [int(x) for x in value.split(",") if x.strip()]
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return result


def env_for(native: bool, graphs: str) -> dict[str, str]:
    env = os.environ.copy()
    if native:
        env["GGML_HIP_GFX1030_NATIVE"] = "1"
    else:
        env.pop("GGML_HIP_GFX1030_NATIVE", None)
    if graphs == "off":
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    else:
        env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    return env


def recorded_env(env: dict[str, str]) -> dict[str, str | None]:
    return {key: env.get(key) for key in RECORDED_ENV}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def snapshot(root: Path, out: Path, label: str, env: dict[str, str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {
        "time": utc_now(),
        "label": label,
        "env": recorded_env(env),
        "platform": platform.platform(),
    }
    for name, argv in {
        "rocm_smi": ["rocm-smi", "--showtemp", "--showuse"],
        "rocminfo": ["rocminfo"],
    }.items():
        rc, text = run_capture(argv, root, env)
        write_text(out / f"{label}-{name}.txt", text)
        records[name] = {"returncode": rc, "path": str(out / f"{label}-{name}.txt")}
    write_text(out / f"{label}-metadata.json", json.dumps(records, indent=2) + "\n")


def run_logged(argv: list[str], root: Path, env: dict[str, str], out: Path,
               label: str, manifest: list[dict[str, Any]], dry_run: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "label": label,
        "command": argv,
        "command_text": command_text(argv),
        "cwd": str(root),
        "env": recorded_env(env),
        "started": utc_now(),
    }
    print(f"[{label}] {entry['command_text']}")
    if dry_run:
        entry.update({"dry_run": True, "returncode": None, "finished": utc_now()})
        manifest.append(entry)
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
        "finished": utc_now(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    })
    manifest.append(entry)
    return rc


def bench_command(args: argparse.Namespace, bench: Path) -> list[str]:
    return [
        str(bench), "-m", str(args.model),
        "-ngl", str(args.gpu_layers), "-sm", args.split_mode,
        "-ts", args.tensor_split, "-fa", "on",
        "-p", ",".join(str(x) for x in args.prompt_sizes),
        "-n", str(args.n_gen), "-b", str(args.batch), "-ub", str(args.ubatch),
        "-r", "1", "-o", "json",
    ]


def backend_test_command(args: argparse.Namespace, test_bin: Path,
                         filter_override: str | None = None) -> list[str]:
    cmd = [str(test_bin), "test", "-o", "FLASH_ATTN_EXT", "-b", args.backend]
    test_filter = args.test_filter if filter_override is None else filter_override
    if test_filter:
        cmd += ["-p", test_filter]
    return cmd


def profile_command(args: argparse.Namespace, test_bin: Path, profile_dir: Path) -> list[str]:
    cmd = [
        str(args.rocprof), "--runtime-trace", "--kernel-trace", "--stats", "--summary",
        "--output-directory", str(profile_dir), "--output-format", "csv",
    ]
    cmd += ["--", *backend_test_command(args, test_bin, args.profile_filter)]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="execute GPU work; otherwise only print the plan")
    parser.add_argument("--allow-gpu", action="store_true", help="required together with --run")
    parser.add_argument("--profile", action="store_true", help="collect rocprofv3 traces for focused FA tests")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--graphs", choices=("on", "off", "both"), default="both")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prompt-sizes", type=parse_int_list, default=[512, 4096, 16384])
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--split-mode", default="layer")
    parser.add_argument("--tensor-split", default="1/1/1/1")
    parser.add_argument("--backend", default="ROCm0")
    parser.add_argument("--test-filter", default="", help="optional test-backend-ops parameter regex")
    parser.add_argument("--profile-filter", default="hsk=256.*kv=4096.*nb=512",
                        help="focused parameter regex used by rocprofv3")
    parser.add_argument("--rocprof", type=Path, default=Path("/opt/rocm/core-7.14/bin/rocprofv3"))
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.model = args.model.resolve()
    if not (args.root / ".git").exists():
        parser.error(f"not a git worktree: {args.root}")
    if not args.model.exists():
        parser.error(f"model does not exist: {args.model}")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.run != args.allow_gpu:
        parser.error("GPU execution requires both --run and --allow-gpu")
    if args.profile and not args.rocprof.exists():
        parser.error(f"rocprofv3 not found: {args.rocprof}")

    build_bin = args.root / "build" / "bin"
    bench = build_bin / "llama-bench"
    test_bin = build_bin / "test-backend-ops"
    for binary in (bench, test_bin):
        if not binary.exists():
            parser.error(f"missing build artifact: {binary}")

    active = active_gpu_processes()
    if args.run and active:
        print("Refusing to start while GPU programs are active:", file=sys.stderr)
        print("\n".join(active), file=sys.stderr)
        return 2

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = args.root / "benchmark-artifacts" / f"gfx1030-native-fa-{stamp}"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph_modes = ["on", "off"] if args.graphs == "both" else [args.graphs]
    conditions = [(native, graphs) for graphs in graph_modes for native in (False, True)]
    manifest: list[dict[str, Any]] = []
    metadata = {
        "started": utc_now(),
        "root": str(args.root),
        "model": str(args.model),
        "model_size": args.model.stat().st_size,
        "build_bin": str(build_bin),
        "git": git_info(args.root),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "profiler": shutil.which(str(args.rocprof)) or str(args.rocprof),
        "active_gpu_processes_at_start": active,
        "commands": manifest,
    }
    write_text(args.output_dir / "manifest.json", json.dumps(metadata, indent=2) + "\n")
    write_text(args.output_dir / "commands.txt", "")

    print(f"Run directory: {args.output_dir}")
    print("Execution mode:", "GPU RUN" if args.run else "DRY RUN (no GPU work)")
    print("Alternating conditions:", [("native" if n else "stock", g) for n, g in conditions])
    if not args.run:
        for rep in range(args.repetitions):
            order = conditions if rep % 2 == 0 else list(reversed(conditions))
            for native, graphs in order:
                env = env_for(native, graphs)
                tag = f"rep{rep+1:02d}-{graphs}-{'native' if native else 'stock'}"
                if not args.skip_tests:
                    run_logged(backend_test_command(args, test_bin), args.root, env,
                               args.output_dir / "tests", tag, manifest, True)
                run_logged(bench_command(args, bench), args.root, env,
                           args.output_dir / "bench" / tag, tag, manifest, True)
                if args.profile and graphs == "on":
                    run_logged(profile_command(args, test_bin, args.output_dir / "profiles" / tag),
                               args.root, env, args.output_dir / "profiles", f"{tag}-rocprof", manifest, True)
        write_text(args.output_dir / "commands.txt",
                   "\n".join(entry["command_text"] for entry in manifest) + "\n")
        metadata["commands"] = manifest
        metadata["dry_run"] = True
        write_text(args.output_dir / "manifest.json", json.dumps(metadata, indent=2) + "\n")
        print("Re-run with --run --allow-gpu after stopping any active server to execute this plan.")
        return 0

    base_env = os.environ.copy()
    snapshot(args.root, args.output_dir / "system", "before", base_env)
    failed = False
    for rep in range(args.repetitions):
        order = conditions if rep % 2 == 0 else list(reversed(conditions))
        for native, graphs in order:
            env = env_for(native, graphs)
            tag = f"rep{rep+1:02d}-{graphs}-{'native' if native else 'stock'}"
            if not args.skip_tests:
                rc = run_logged(backend_test_command(args, test_bin), args.root, env,
                                args.output_dir / "tests", tag, manifest, False)
                failed |= rc != 0
            bench_out = args.output_dir / "bench" / tag
            rc = run_logged(bench_command(args, bench), args.root, env, bench_out, tag, manifest, False)
            failed |= rc != 0
            if args.profile and graphs == "on" and rep == 0:
                profile_out = args.output_dir / "profiles" / tag
                rc = run_logged(profile_command(args, test_bin, profile_out), args.root, env,
                                args.output_dir / "profiles", f"{tag}-rocprof", manifest, False)
                failed |= rc != 0
            write_text(args.output_dir / "commands.txt",
                       "\n".join(entry["command_text"] for entry in manifest) + "\n")
            metadata["commands"] = manifest
            write_text(args.output_dir / "manifest.json", json.dumps(metadata, indent=2) + "\n")
    snapshot(args.root, args.output_dir / "system", "after", os.environ.copy())
    metadata["finished"] = utc_now()
    metadata["failed"] = failed
    metadata["commands"] = manifest
    write_text(args.output_dir / "manifest.json", json.dumps(metadata, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())