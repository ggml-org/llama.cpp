#!/usr/bin/env python3
"""Run a context-matched adaptive-KV sweep and plot the collected results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/server-ab"))
from server_ab import request_json, stream_completion  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--server", type=Path, default=ROOT / "build-device-memory-infra-cuda-release/bin/llama-server")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/fixed-span-8k-192k")
    parser.add_argument("--min-context", type=int, default=8192)
    parser.add_argument("--max-context", type=int, default=196608)
    parser.add_argument("--context-step", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--arena-mib", type=int, default=2688)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--port", type=int, default=1246)
    parser.add_argument("--production-container", default="llm-llmster")
    parser.add_argument("--no-manage-production", action="store_true")
    parser.add_argument("--uvm", action="store_true")
    return parser.parse_args()


def wait_ready(url: str, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}; see {log_path}")
        try:
            if request_json(url + "/health", timeout=2).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server readiness timed out; see {log_path}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def decode_layout(log_path: Path) -> tuple[float, int, int, int, float, int]:
    text = log_path.read_text(errors="replace")
    phase = re.findall(
        r"memory_phase: phase=decode .*?kv_pool=(\d+).*?resident_pages=(\d+).*?"
        r"ring_slots=(\d+).*?active_pages=(\d+)",
        text,
    )
    transfers = re.findall(r"attention: accepted KV layout copied ([0-9.]+) MiB in (\d+) H2D calls", text)
    if not phase:
        raise RuntimeError(f"missing decode layout telemetry in {log_path}")
    pool, resident, ring, active = map(int, phase[-1])
    h2d_mib, h2d_calls = (float(transfers[-1][0]), int(transfers[-1][1])) if transfers else (0.0, 0)
    return pool / 1048576.0, resident, ring, active, h2d_mib, h2d_calls


def write_outputs(rows: list[dict], output: Path) -> None:
    fields = list(rows[0])
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    x = [row["context_capacity"] / 1024 for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), sharex=True)
    panels = [
        ("decode_tps", "Decode throughput", "tokens/s", "-"),
        ("prefill_tps", "Prefill throughput", "tokens/s", "--"),
        ("decode_kv_pool_mib", "Effective decode KV pool", "MiB", "-."),
        ("h2d_util_pct", "Estimated decode H2D utilization", "% of measured 50 GB/s", ":"),
    ]
    for axis, (field, title, ylabel, style) in zip(axes.flat, panels):
        y = [row[field] for row in rows]
        axis.plot(x, y, color="#7A3E9D", linestyle=style, linewidth=2.4, marker="o", markersize=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.28)
    for axis in axes[1]:
        axis.set_xlabel("Configured context capacity (Ki tokens)")
    for axis in axes.flat:
        axis.set_xticks(x[::2])
    axes[1, 1].axhline(100, color="black", linewidth=1, alpha=0.35)
    fig.suptitle("Qwen3.8-27B IQ4_XS — fixed-span adaptive KV sweep", fontsize=15)
    fig.text(
        0.5,
        0.015,
        "Context-matched capacity · Q8_0 K / Q4_0 V · batch/ubatch 256 · 256 decoded tokens · UVM off",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(output / "fixed-span-sweep.png", dpi=180)
    fig.savefig(output / "fixed-span-sweep.svg")


def main() -> int:
    args = arguments()
    args.model = args.model.resolve()
    args.server = args.server.resolve()
    args.output = args.output.resolve()
    if not args.model.is_file() or not args.server.is_file():
        raise SystemExit("model or server executable does not exist")
    if args.min_context <= args.decode_tokens or args.context_step <= 0 or args.max_context < args.min_context:
        raise SystemExit("invalid context range")

    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    managed = False
    if not args.no_manage_production:
        running = subprocess.run(
            ["podman", "inspect", "-f", "{{.State.Running}}", args.production_container],
            text=True,
            capture_output=True,
        )
        if running.returncode == 0 and running.stdout.strip() == "true":
            subprocess.run(["podman", "stop", "-t", "30", args.production_container], check=True)
            managed = True

    rows: list[dict] = []
    try:
        for context in range(args.min_context, args.max_context + 1, args.context_step):
            prompt_tokens = context - args.decode_tokens
            log_path = logs / f"context-{context}-arena-{args.arena_mib}.log"
            env = os.environ.copy()
            if not args.uvm:
                for key in (
                    "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
                    "GGML_CUDA_PREFER_MODEL_WEIGHTS",
                    "GGML_CUDA_PREFER_KV_HOST",
                    "GGML_CUDA_KV_ACCESSED_BY_GPU",
                ):
                    env.pop(key, None)
            command = [
                str(args.server),
                "--model", str(args.model),
                "--ctx-size", str(context),
                "--batch-size", str(args.batch_size),
                "--ubatch-size", str(args.ubatch_size),
                "--parallel", "1",
                "--n-gpu-layers", "999",
                "--flash-attn", "on",
                "--cache-type-k", "q8_0",
                "--cache-type-v", "q4_0",
                "--kv-stream-arena-mib", str(args.arena_mib),
                "--fit", "off",
                "--no-mmproj",
                "--host", "127.0.0.1",
                "--port", str(args.port),
                "--threads", "8",
                "--threads-batch", "8",
                "-lv", "3",
            ]
            started = time.monotonic()
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    wait_ready(url, process, log_path)
                    tokens = request_json(
                        url + "/tokenize",
                        {"content": "Adaptive KV streaming fixed-span benchmark. ", "add_special": False},
                        timeout=10,
                    )["tokens"]
                    warmup = stream_completion(
                        url + "/completion",
                        {
                            "prompt": (tokens * ((256 + len(tokens) - 1) // len(tokens)))[:256],
                            "n_predict": 4,
                            "temperature": 0,
                            "seed": 123,
                            "ignore_eos": True,
                            "cache_prompt": False,
                            "stream": True,
                            "return_tokens": True,
                        },
                        timeout=120,
                    )
                    if warmup["timings"].get("predicted_n") != 4:
                        raise RuntimeError("warmup did not complete")
                    prompt = (tokens * ((prompt_tokens + len(tokens) - 1) // len(tokens)))[:prompt_tokens]
                    result = stream_completion(
                        url + "/completion",
                        {
                            "prompt": prompt,
                            "n_predict": args.decode_tokens,
                            "temperature": 0,
                            "seed": 123,
                            "ignore_eos": True,
                            "cache_prompt": False,
                            "stream": True,
                            "return_tokens": True,
                        },
                        timeout=1800,
                    )
                    timing = result["timings"]
                    if timing.get("prompt_n") != prompt_tokens or timing.get("predicted_n") != args.decode_tokens:
                        raise RuntimeError("incomplete benchmark response")
                finally:
                    stop_process(process)

            pool, resident, ring, active, h2d_mib, h2d_calls = decode_layout(log_path)
            decode_tps = float(timing["predicted_per_second"])
            h2d_gbs = h2d_mib * 1048576 * decode_tps / 1e9
            row = {
                "context_capacity": context,
                "prompt_tokens": prompt_tokens,
                "decode_tokens": args.decode_tokens,
                "arena_mib": args.arena_mib,
                "prefill_tps": float(timing["prompt_per_second"]),
                "decode_tps": decode_tps,
                "prompt_ms": float(timing["prompt_ms"]),
                "predicted_ms": float(timing["predicted_ms"]),
                "wall_seconds": time.monotonic() - started,
                "decode_kv_pool_mib": pool,
                "resident_pages": resident,
                "ring_slots": ring,
                "active_pages": active,
                "h2d_mib_per_token": h2d_mib,
                "h2d_calls_per_token": h2d_calls,
                "h2d_util_pct": h2d_gbs / 50.0 * 100.0,
                "log": str(log_path),
            }
            rows.append(row)
            write_outputs(rows, args.output)
            print(json.dumps(row, sort_keys=True), flush=True)
        try:
            plot(rows, args.output)
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            print("matplotlib is unavailable; CSV/JSONL/log results were preserved without a plot", file=sys.stderr)
    finally:
        if managed:
            subprocess.run(["podman", "start", args.production_container], check=False)

    print(args.output / "results.csv")
    if (args.output / "fixed-span-sweep.png").exists():
        print(args.output / "fixed-span-sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
