#!/usr/bin/env python3
"""Run a multi-candidate llama-bench sweep against existing RPC endpoints."""

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import time
from pathlib import Path

from bench_rpc_remote import (
    DEFAULT_RECORDED_ENV,
    build_env,
    captured_environment,
    dedupe,
    load_jsonl,
    parse_bench_extra_args,
    parse_rpc_arg,
    redact_env_value,
    rpc_arg_from_endpoints,
    summarize_rows,
    utc_now,
    wait_for_endpoints,
    write_summary,
)


def parse_candidate(value):
    name, sep, tensor_split = value.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(
            f"invalid candidate {value!r}; expected NAME=TENSOR_SPLIT or NAME=auto"
        )
    name = name.strip()
    tensor_split = tensor_split.strip()
    if not name:
        raise argparse.ArgumentTypeError("candidate name is empty")
    if "/" in name or "\\" in name:
        raise argparse.ArgumentTypeError("candidate name must not contain path separators")
    if not tensor_split:
        raise argparse.ArgumentTypeError(f"candidate {name!r} has an empty tensor split")
    if tensor_split != "auto":
        if "," in tensor_split:
            raise argparse.ArgumentTypeError(
                f"candidate {name!r}: tensor split must describe one case; commas create llama-bench grids"
            )
        for part in tensor_split.replace(";", "/").split("/"):
            if not part:
                continue
            try:
                if float(part) < 0:
                    raise ValueError
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"candidate {name!r}: invalid non-negative split value {part!r}"
                ) from exc
    return {
        "name": name,
        "tensor_split": None if tensor_split == "auto" else tensor_split,
        "tensor_split_display": tensor_split,
    }


def build_bench_cmd(args, rpc_arg, candidate):
    cmd = [
        str(args.llama_bench),
        "-m",
        str(args.model),
        "--rpc",
        rpc_arg,
        "-ngl",
        str(args.ngl),
        "-p",
        str(args.prompt),
        "-n",
        str(args.gen),
        "-r",
        str(args.repetitions),
    ]
    if args.split_mode is not None:
        cmd.extend(["-sm", args.split_mode])
    if args.flash_attn is not None:
        cmd.extend(["-fa", args.flash_attn])
    if candidate["tensor_split"] is not None:
        cmd.extend(["-ts", candidate["tensor_split"]])
    if args.threads is not None:
        cmd.extend(["-t", str(args.threads)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    cmd.extend(args.bench_extra)
    cmd.extend(["-o", "jsonl"])
    return cmd


def make_case(args, rpc_arg, endpoints, candidate):
    jsonl_path = args.out_dir / f"{candidate['name']}.jsonl"
    stderr_path = args.out_dir / f"{candidate['name']}.stderr.txt"
    command = build_bench_cmd(args, rpc_arg, candidate)
    return {
        "status": "planned",
        "name": candidate["name"],
        "rpc": rpc_arg,
        "endpoints": endpoints,
        "device": args.device,
        "split_mode": args.split_mode,
        "flash_attn": args.flash_attn,
        "tensor_split": candidate["tensor_split"],
        "tensor_split_display": candidate["tensor_split_display"],
        "command": command,
        "command_display": shlex.join(command),
        "jsonl_path": str(jsonl_path),
        "stderr_path": str(stderr_path),
    }


def run_case(args, case, env):
    if args.dry_run:
        case["status"] = "dry-run"
        return

    jsonl_path = Path(case["jsonl_path"])
    stderr_path = Path(case["stderr_path"])
    started = time.monotonic()
    case["started_utc"] = utc_now()
    try:
        with jsonl_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
            completed = subprocess.run(
                case["command"],
                check=False,
                env=env,
                stderr=stderr_fh,
                stdout=stdout_fh,
                text=True,
                timeout=args.timeout,
            )
    except subprocess.TimeoutExpired as exc:
        case["finished_utc"] = utc_now()
        case["elapsed_sec"] = time.monotonic() - started
        case["status"] = "timeout"
        case["error"] = f"llama-bench timed out after {args.timeout} seconds"
        raise RuntimeError(f"{case['name']} {case['error']}") from exc

    case["finished_utc"] = utc_now()
    case["elapsed_sec"] = time.monotonic() - started
    case["returncode"] = completed.returncode

    if completed.returncode != 0:
        case["status"] = "failed"
        case["error"] = f"llama-bench exited with status {completed.returncode}"
        raise RuntimeError(f"{case['name']} {case['error']}; see {stderr_path}")

    metrics = summarize_rows(load_jsonl(jsonl_path))
    if "prompt" not in metrics or "decode" not in metrics:
        case["status"] = "failed"
        case["metrics"] = metrics
        case["error"] = f"missing prompt or decode metrics in {jsonl_path}"
        raise RuntimeError(f"{case['name']} {case['error']}")

    case["status"] = "ok"
    case["metrics"] = metrics


def metric(case, rank_by):
    return (
        case.get("metrics", {})
        .get(rank_by, {})
        .get("avg_tokens_per_second")
    )


def rank_cases(summary, rank_by):
    rows = [
        case for case in summary["cases"]
        if case.get("status") == "ok" and metric(case, rank_by) is not None
    ]
    rows.sort(key=lambda case: metric(case, rank_by), reverse=True)
    for rank, case in enumerate(rows, start=1):
        case["rank"] = rank
    summary["rank_by"] = rank_by
    summary["ranked_cases"] = [
        {
            "rank": case["rank"],
            "name": case["name"],
            "tensor_split": case["tensor_split_display"],
            "prompt_avg_tokens_per_second": metric(case, "prompt"),
            "decode_avg_tokens_per_second": metric(case, "decode"),
            "elapsed_sec": case.get("elapsed_sec"),
            "jsonl_path": case["jsonl_path"],
            "stderr_path": case["stderr_path"],
        }
        for case in rows
    ]


def write_ranked_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_ranked_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def validate_inputs(args):
    if args.dry_run:
        return
    if not args.llama_bench.is_file():
        raise FileNotFoundError(f"llama-bench not found: {args.llama_bench}")
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run llama-bench placement candidates against existing RPC endpoints."
    )
    parser.add_argument("--llama-bench", required=True, type=Path, help="local llama-bench executable")
    parser.add_argument("--model", required=True, type=Path, help="local GGUF model path")
    parser.add_argument("--rpc", required=True, help="RPC host:port list")
    parser.add_argument("--candidate", action="append", required=True, type=parse_candidate, metavar="NAME=TENSOR_SPLIT", help="candidate tensor split; use NAME=auto to omit -ts")
    parser.add_argument("--device", default=None, help="llama-bench device selector, for example RPC0/RPC1")
    parser.add_argument("--split-mode", choices=("none", "layer", "row", "tensor"), default=None, help="optional llama-bench split mode for -sm")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default=None, help="optional llama-bench flash attention setting for -fa")
    parser.add_argument("--prompt", default=32, type=int, help="prompt tokens for llama-bench -p")
    parser.add_argument("--gen", default=32, type=int, help="generated tokens for llama-bench -n")
    parser.add_argument("--repetitions", default=7, type=int, help="llama-bench repetitions for -r")
    parser.add_argument("--ngl", default=99, type=int, help="GPU layers for llama-bench -ngl")
    parser.add_argument("--threads", default=None, type=int, help="optional llama-bench CPU thread count for -t")
    parser.add_argument("--rank-by", choices=("prompt", "decode"), default="decode", help="metric used for ranks")
    parser.add_argument("--out-dir", default=Path("rpc-placement-sweep"), type=Path, help="directory for outputs")
    parser.add_argument("--summary", default=None, type=Path, help="summary JSON path; defaults to OUT_DIR/summary.json")
    parser.add_argument("--timeout", default=1800, type=float, help="per-candidate timeout in seconds")
    parser.add_argument("--connect-timeout", default=10.0, type=float, help="per-endpoint TCP connect timeout")
    parser.add_argument("--skip-port-check", action="store_true", help="skip pre-run TCP checks")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment override; repeatable")
    parser.add_argument("--record-env", action="append", default=[], metavar="KEY", help="extra environment variable to include in summary")
    parser.add_argument("--bench-extra", action="append", default=[], metavar="ARGS", help="extra shell-style llama-bench args for all candidates; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="write planned commands without running llama-bench")
    args = parser.parse_args()

    try:
        args.endpoints = parse_rpc_arg(args.rpc)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        args.bench_extra = parse_bench_extra_args(args.bench_extra, "--bench-extra")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if args.prompt < 1:
        parser.error("--prompt must be >= 1")
    if args.gen < 1:
        parser.error("--gen must be >= 1")
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be > 0")

    names = [candidate["name"] for candidate in args.candidate]
    if len(names) != len(set(names)):
        parser.error("candidate names must be unique")

    return args


def main():
    args = parse_args()
    args.summary = args.summary or args.out_dir / "summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env, env_overrides = build_env(args)
    record_env_keys = dedupe([
        *DEFAULT_RECORDED_ENV,
        *args.record_env,
        *env_overrides.keys(),
    ])
    rpc_arg = rpc_arg_from_endpoints(args.endpoints)
    summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "dry_run": args.dry_run,
        "cwd": os.getcwd(),
        "llama_bench": str(args.llama_bench),
        "model": str(args.model),
        "rpc": rpc_arg,
        "endpoints": args.endpoints,
        "prompt": args.prompt,
        "gen": args.gen,
        "repetitions": args.repetitions,
        "ngl": args.ngl,
        "device": args.device,
        "split_mode": args.split_mode,
        "flash_attn": args.flash_attn,
        "threads": args.threads,
        "timeout_sec": args.timeout,
        "connect_timeout_sec": args.connect_timeout,
        "skip_port_check": args.skip_port_check,
        "out_dir": str(args.out_dir),
        "summary_path": str(args.summary),
        "record_env_keys": record_env_keys,
        "environment": captured_environment(env, record_env_keys),
        "env_overrides": {key: redact_env_value(key, value) for key, value in env_overrides.items()},
        "bench_extra": args.bench_extra,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "cases": [
            make_case(args, rpc_arg, args.endpoints, candidate)
            for candidate in args.candidate
        ],
    }

    try:
        validate_inputs(args)
        if not args.skip_port_check and not args.dry_run:
            wait_for_endpoints(args.endpoints, args.connect_timeout)
        for case in summary["cases"]:
            case["environment"] = captured_environment(env, record_env_keys)
            run_case(args, case, env)
        if not args.dry_run:
            rank_cases(summary, args.rank_by)
            write_ranked_csv(args.out_dir / "ranked.csv", summary["ranked_cases"])
            write_ranked_jsonl(args.out_dir / "ranked.jsonl", summary["ranked_cases"])
    except Exception as exc:
        summary["error"] = str(exc)
        write_summary(args.summary, summary)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 1

    write_summary(args.summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
