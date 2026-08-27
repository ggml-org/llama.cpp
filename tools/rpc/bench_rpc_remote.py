#!/usr/bin/env python3
"""Compare two already-running RPC endpoints with local llama-bench runs."""

import argparse
import json
import os
import platform
import shlex
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RECORDED_ENV = (
    "GGML_RPC_TCP_BUFFER_SIZE",
    "GGML_RPC_CACHE_MIN_SIZE",
    "GGML_RPC_DEBUG",
    "GGML_RPC_TRACE",
    "GGML_RPC_TIMEOUT",
    "LLAMA_CACHE",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
    "GGML_CUDA_FORCE_MMQ",
    "GGML_CUDA_FORCE_CUBLAS",
    "GGML_CUDA_NO_VMM",
    "GGML_VK_VISIBLE_DEVICES",
)

SECRET_ENV_MARKERS = (
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_env_assignment(value):
    key, sep, env_value = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"environment override must be KEY=VALUE: {value!r}")
    return key, env_value


def parse_env_assignments(values):
    overrides = {}
    for assignment in values:
        key, value = parse_env_assignment(assignment)
        overrides[key] = value
    return overrides


def parse_bench_extra_args(values, option):
    args = []
    for value in values:
        try:
            args.extend(shlex.split(value))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{option}: invalid argument string {value!r}: {exc}") from exc
    return args


def format_endpoint(host, port):
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def parse_endpoint(value):
    text = value.strip()
    if not text:
        raise ValueError("empty RPC endpoint")

    if text.startswith("["):
        end = text.find("]")
        if end < 0 or end + 1 >= len(text) or text[end + 1] != ":":
            raise ValueError(f"invalid RPC endpoint {value!r}; use [host]:port for IPv6")
        host = text[1:end]
        port_text = text[end + 2 :]
    else:
        host, sep, port_text = text.rpartition(":")
        if not sep:
            raise ValueError(f"invalid RPC endpoint {value!r}; expected host:port")
        if ":" in host:
            raise ValueError(f"invalid RPC endpoint {value!r}; use [host]:port for IPv6")

    if not host:
        raise ValueError(f"invalid RPC endpoint {value!r}; host is empty")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"invalid RPC endpoint {value!r}; port is not an integer") from exc

    if not 1 <= port <= 65535:
        raise ValueError(f"invalid RPC endpoint {value!r}; port must be 1..65535")

    return {
        "host": host,
        "port": port,
        "endpoint": format_endpoint(host, port),
    }


def parse_rpc_arg(value):
    endpoints = [parse_endpoint(part) for part in value.split(",")]
    if not endpoints:
        raise ValueError("RPC endpoint list is empty")
    return endpoints


def rpc_arg_from_endpoints(endpoints):
    return ",".join(endpoint["endpoint"] for endpoint in endpoints)


def wait_for_port(host, port, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=min(1.0, timeout)):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"timed out waiting for {host}:{port}{detail}")


def wait_for_endpoints(endpoints, timeout):
    for endpoint in endpoints:
        wait_for_port(endpoint["host"], endpoint["port"], timeout)


def dedupe(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def redact_env_value(key, value):
    upper_key = key.upper()
    if any(marker in upper_key for marker in SECRET_ENV_MARKERS):
        return "<redacted>"
    return value


def captured_environment(env, keys):
    captured = {}
    for key in dedupe(keys):
        if key in env:
            captured[key] = redact_env_value(key, env[key])
    return captured


def build_env(args):
    env = os.environ.copy()
    overrides = parse_env_assignments(args.env)
    env.update(overrides)
    return env, overrides


def build_case_env(env, assignments):
    case_env = env.copy()
    overrides = parse_env_assignments(assignments)
    case_env.update(overrides)
    return case_env, overrides


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def numeric_samples(row):
    samples = row.get("samples_ts")
    if isinstance(samples, list):
        return [float(sample) for sample in samples]
    if "avg_ts" in row:
        return [float(row["avg_ts"])]
    return []


def classify_row(row):
    test_name = str(row.get("test", "")).lower()
    n_prompt = int(row.get("n_prompt") or 0)
    n_gen = int(row.get("n_gen") or 0)

    if "pp" in test_name:
        return "prompt"
    if "tg" in test_name:
        return "decode"
    if n_prompt > 0 and n_gen == 0:
        return "prompt"
    if n_gen > 0 or n_prompt == 0:
        return "decode"
    return None


def summarize_rows(rows):
    grouped = {
        "prompt": [],
        "decode": [],
    }
    unclassified = 0

    for row in rows:
        kind = classify_row(row)
        samples = numeric_samples(row)
        if kind is None or not samples:
            unclassified += 1
            continue
        grouped[kind].extend(samples)

    metrics = {}
    for kind, samples in grouped.items():
        if not samples:
            continue
        avg_ts = statistics.fmean(samples)
        median_ts = statistics.median(samples)
        stdev_ts = statistics.stdev(samples) if len(samples) > 1 else 0.0
        metrics[kind] = {
            "avg_tokens_per_second": avg_ts,
            "median_tokens_per_second": median_ts,
            "stdev_tokens_per_second": stdev_ts,
            "avg_ts": avg_ts,
            "median_ts": median_ts,
            "stdev_ts": stdev_ts,
            "min_tokens_per_second": min(samples),
            "max_tokens_per_second": max(samples),
            "samples_tokens_per_second": samples,
            "sample_count": len(samples),
        }

    if unclassified:
        metrics["unclassified_row_count"] = unclassified

    return metrics


def required_metric_kinds(args):
    kinds = []
    if args.prompt > 0:
        kinds.append("prompt")
    if args.gen > 0:
        kinds.append("decode")
    return kinds


def case_tensor_split(args, name):
    if name == "base" and args.base_tensor_split is not None:
        return args.base_tensor_split
    if name == "patch" and args.patch_tensor_split is not None:
        return args.patch_tensor_split
    return args.tensor_split


def case_bench_extra_args(args, name):
    extra = list(args.bench_extra)
    if name == "base":
        extra.extend(args.base_bench_extra)
    elif name == "patch":
        extra.extend(args.patch_bench_extra)
    return extra


def build_bench_cmd(args, name, llama_bench, rpc_arg):
    cmd = [
        str(llama_bench),
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
    tensor_split = case_tensor_split(args, name)
    if tensor_split is not None:
        cmd.extend(["-ts", tensor_split])
    if args.threads is not None:
        cmd.extend(["-t", str(args.threads)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    cmd.extend(case_bench_extra_args(args, name))
    cmd.extend(["-o", "jsonl"])
    return cmd


def make_case(args, name, llama_bench, endpoints):
    rpc_arg = rpc_arg_from_endpoints(endpoints)
    jsonl_path = args.out_dir / f"{name}-llama-bench.jsonl"
    stderr_path = args.out_dir / f"{name}-llama-bench.stderr.txt"
    tensor_split = case_tensor_split(args, name)
    bench_extra = case_bench_extra_args(args, name)
    command = build_bench_cmd(args, name, llama_bench, rpc_arg)
    return {
        "status": "planned",
        "llama_bench": str(llama_bench),
        "rpc": rpc_arg,
        "endpoints": endpoints,
        "split_mode": args.split_mode,
        "flash_attn": args.flash_attn,
        "tensor_split": tensor_split,
        "bench_extra": bench_extra,
        "command": command,
        "command_display": shlex.join(command),
        "jsonl_path": str(jsonl_path),
        "stderr_path": str(stderr_path),
    }


def run_case(args, name, case, env):
    if args.dry_run:
        case["status"] = "dry-run"
        return

    if not args.skip_port_check:
        wait_for_endpoints(case["endpoints"], args.connect_timeout)

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
        raise RuntimeError(f"{name} {case['error']}") from exc

    case["finished_utc"] = utc_now()
    case["elapsed_sec"] = time.monotonic() - started
    case["returncode"] = completed.returncode

    if completed.returncode != 0:
        case["status"] = "failed"
        case["error"] = f"llama-bench exited with status {completed.returncode}"
        raise RuntimeError(f"{name} {case['error']}; see {stderr_path}")

    metrics = summarize_rows(load_jsonl(jsonl_path))
    missing_metrics = [kind for kind in required_metric_kinds(args) if kind not in metrics]
    if missing_metrics:
        case["status"] = "failed"
        case["metrics"] = metrics
        case["error"] = f"missing {', '.join(missing_metrics)} metrics in {jsonl_path}"
        raise RuntimeError(f"{name} {case['error']}")

    case["status"] = "ok"
    case["metrics"] = metrics


def metric_avg(summary, case_name, kind):
    return (
        summary.get("cases", {})
        .get(case_name, {})
        .get("metrics", {})
        .get(kind, {})
        .get("avg_tokens_per_second")
    )


def case_elapsed(summary, case_name):
    return summary.get("cases", {}).get(case_name, {}).get("elapsed_sec")


def percent_delta(patch_value, base_value):
    if base_value is None or patch_value is None or base_value == 0:
        return None
    return (patch_value / base_value - 1.0) * 100.0


def measured_metric_kinds(summary):
    kinds = []
    for kind in ("prompt", "decode"):
        if any(
            kind in case.get("metrics", {})
            for case in summary.get("cases", {}).values()
        ):
            kinds.append(kind)
    return kinds


def build_comparison(summary):
    if "base" not in summary.get("cases", {}) or "patch" not in summary.get("cases", {}):
        return

    comparison = {}
    metric_kinds = measured_metric_kinds(summary)
    for kind in metric_kinds:
        base_avg = metric_avg(summary, "base", kind)
        patch_avg = metric_avg(summary, "patch", kind)
        comparison[kind] = {
            "base_avg_tokens_per_second": base_avg,
            "patch_avg_tokens_per_second": patch_avg,
            "delta_pct": percent_delta(patch_avg, base_avg),
        }
    base_elapsed = case_elapsed(summary, "base")
    patch_elapsed = case_elapsed(summary, "patch")
    comparison["elapsed"] = {
        "base_elapsed_sec": base_elapsed,
        "patch_elapsed_sec": patch_elapsed,
        "delta_pct": percent_delta(patch_elapsed, base_elapsed),
    }
    summary["measured_metric_kinds"] = metric_kinds
    if "prompt" in comparison:
        summary["prompt_avg_ts_delta_pct"] = comparison["prompt"]["delta_pct"]
    if "decode" in comparison:
        summary["decode_avg_ts_delta_pct"] = comparison["decode"]["delta_pct"]
    summary["elapsed_delta_pct"] = comparison["elapsed"]["delta_pct"]
    summary["comparison"] = comparison


def validate_inputs(args):
    if args.dry_run:
        return
    if args.only in ("base", "both") and not args.base_llama_bench.is_file():
        raise FileNotFoundError(f"base llama-bench not found: {args.base_llama_bench}")
    if args.only in ("patch", "both") and not args.patch_llama_bench.is_file():
        raise FileNotFoundError(f"patch llama-bench not found: {args.patch_llama_bench}")
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def path_or_none(path):
    return str(path) if path is not None else None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a local llama-bench client against two already-running remote "
            "RPC endpoint sets and write a comparison summary."
        )
    )
    parser.add_argument("--llama-bench", default=None, type=Path, help="local llama-bench executable used for both cases")
    parser.add_argument("--base-llama-bench", default=None, type=Path, help="baseline llama-bench executable; defaults to --llama-bench")
    parser.add_argument("--patch-llama-bench", default=None, type=Path, help="patched llama-bench executable; defaults to --llama-bench")
    parser.add_argument("--model", required=True, type=Path, help="local GGUF model path")
    parser.add_argument("--base-rpc", default=None, help="baseline RPC host:port, or comma-separated host:port list")
    parser.add_argument("--patch-rpc", default=None, help="patched RPC host:port, or comma-separated host:port list")
    parser.add_argument("--only", choices=("base", "patch", "both"), default="both", help="run one case or both cases")
    parser.add_argument("--prompt", default=32, type=int, help="prompt tokens for llama-bench -p; use 0 for generation-only runs")
    parser.add_argument("--gen", default=32, type=int, help="generated tokens for llama-bench -n")
    parser.add_argument("--repetitions", default=7, type=int, help="llama-bench repetitions for -r")
    parser.add_argument("--ngl", default=99, type=int, help="GPU layers for llama-bench -ngl")
    parser.add_argument("--device", default=None, help="llama-bench device selector, for example RPC0")
    parser.add_argument("--split-mode", choices=("none", "layer", "row", "tensor"), default=None, help="optional llama-bench split mode for -sm")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default=None, help="optional llama-bench flash attention setting for -fa")
    parser.add_argument("--tensor-split", default=None, help="common llama-bench tensor split for -ts")
    parser.add_argument("--base-tensor-split", default=None, help="baseline-only llama-bench tensor split for -ts; overrides --tensor-split")
    parser.add_argument("--patch-tensor-split", default=None, help="patch-only llama-bench tensor split for -ts; overrides --tensor-split")
    parser.add_argument("--threads", default=None, type=int, help="optional llama-bench CPU thread count for -t")
    parser.add_argument("--out-dir", default=Path("rpc-remote-bench-results"), type=Path, help="directory for jsonl and summary outputs")
    parser.add_argument("--summary", default=None, type=Path, help="summary JSON path; defaults to OUT_DIR/summary.json")
    parser.add_argument("--timeout", default=1800, type=float, help="per-case llama-bench timeout in seconds")
    parser.add_argument("--connect-timeout", default=10.0, type=float, help="per-endpoint TCP connect timeout in seconds")
    parser.add_argument("--skip-port-check", action="store_true", help="skip pre-run TCP checks for the RPC endpoints")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment override for llama-bench; repeatable")
    parser.add_argument("--base-env", action="append", default=[], metavar="KEY=VALUE", help="baseline-only environment override for llama-bench; repeatable")
    parser.add_argument("--patch-env", action="append", default=[], metavar="KEY=VALUE", help="patch-only environment override for llama-bench; repeatable")
    parser.add_argument("--record-env", action="append", default=[], metavar="KEY", help="extra environment variable to include in the summary")
    parser.add_argument("--bench-extra", action="append", default=[], metavar="ARGS", help="extra shell-style llama-bench args for both cases; repeatable")
    parser.add_argument("--base-bench-extra", action="append", default=[], metavar="ARGS", help="baseline-only extra shell-style llama-bench args; repeatable")
    parser.add_argument("--patch-bench-extra", action="append", default=[], metavar="ARGS", help="patch-only extra shell-style llama-bench args; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="write the planned commands without connecting or running llama-bench")
    args = parser.parse_args()

    for attr, option in (
        ("bench_extra", "--bench-extra"),
        ("base_bench_extra", "--base-bench-extra"),
        ("patch_bench_extra", "--patch-bench-extra"),
    ):
        try:
            setattr(args, attr, parse_bench_extra_args(getattr(args, attr), option))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    if args.prompt < 0:
        parser.error("--prompt must be >= 0")
    if args.gen < 1:
        parser.error("--gen must be >= 1")
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be > 0")

    if args.only in ("base", "both") and args.base_rpc is None:
        parser.error("--base-rpc is required when --only is base or both")
    if args.only in ("patch", "both") and args.patch_rpc is None:
        parser.error("--patch-rpc is required when --only is patch or both")

    try:
        args.base_endpoints = parse_rpc_arg(args.base_rpc) if args.base_rpc is not None else None
        args.patch_endpoints = parse_rpc_arg(args.patch_rpc) if args.patch_rpc is not None else None
    except ValueError as exc:
        parser.error(str(exc))

    if args.base_llama_bench is None:
        args.base_llama_bench = args.llama_bench
    if args.patch_llama_bench is None:
        args.patch_llama_bench = args.llama_bench
    if args.only in ("base", "both") and args.base_llama_bench is None:
        parser.error("provide --llama-bench or --base-llama-bench")
    if args.only in ("patch", "both") and args.patch_llama_bench is None:
        parser.error("provide --llama-bench or --patch-llama-bench")

    return args


def main():
    args = parse_args()
    args.summary = args.summary or args.out_dir / "summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env, env_overrides = build_env(args)
    base_env, base_env_overrides = build_case_env(env, args.base_env)
    patch_env, patch_env_overrides = build_case_env(env, args.patch_env)
    case_envs = {
        "base": base_env,
        "patch": patch_env,
    }
    case_env_overrides = {
        "base": base_env_overrides,
        "patch": patch_env_overrides,
    }
    record_env_keys = dedupe([
        *DEFAULT_RECORDED_ENV,
        *args.record_env,
        *env_overrides.keys(),
        *base_env_overrides.keys(),
        *patch_env_overrides.keys(),
    ])
    summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "dry_run": args.dry_run,
        "cwd": os.getcwd(),
        "llama_bench": str(args.llama_bench) if args.llama_bench is not None else None,
        "base_llama_bench": path_or_none(args.base_llama_bench),
        "patch_llama_bench": path_or_none(args.patch_llama_bench),
        "model": str(args.model),
        "prompt": args.prompt,
        "gen": args.gen,
        "repetitions": args.repetitions,
        "ngl": args.ngl,
        "device": args.device,
        "split_mode": args.split_mode,
        "flash_attn": args.flash_attn,
        "tensor_split": args.tensor_split,
        "base_tensor_split": args.base_tensor_split,
        "patch_tensor_split": args.patch_tensor_split,
        "bench_extra": args.bench_extra,
        "base_bench_extra": args.base_bench_extra,
        "patch_bench_extra": args.patch_bench_extra,
        "threads": args.threads,
        "only": args.only,
        "timeout_sec": args.timeout,
        "connect_timeout_sec": args.connect_timeout,
        "skip_port_check": args.skip_port_check,
        "out_dir": str(args.out_dir),
        "summary_path": str(args.summary),
        "record_env_keys": record_env_keys,
        "environment": captured_environment(env, record_env_keys),
        "env_overrides": {key: redact_env_value(key, value) for key, value in env_overrides.items()},
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "cases": {},
    }

    if args.only in ("base", "both"):
        summary["cases"]["base"] = make_case(args, "base", args.base_llama_bench, args.base_endpoints)
    if args.only in ("patch", "both"):
        summary["cases"]["patch"] = make_case(args, "patch", args.patch_llama_bench, args.patch_endpoints)

    try:
        validate_inputs(args)
        for name in ("base", "patch"):
            if name not in summary["cases"]:
                continue
            summary["cases"][name]["environment"] = captured_environment(case_envs[name], record_env_keys)
            summary["cases"][name]["env_overrides"] = {
                key: redact_env_value(key, value)
                for key, value in case_env_overrides[name].items()
            }
            run_case(args, name, summary["cases"][name], case_envs[name])
        if not args.dry_run:
            build_comparison(summary)
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
