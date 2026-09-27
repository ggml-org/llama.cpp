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


def parse_mtp_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("MTP lengths must be comma-separated integers from 0 to 4") from error
    if not lengths or len(set(lengths)) != len(lengths) or any(length < 0 or length > 4 for length in lengths):
        raise argparse.ArgumentTypeError("MTP lengths must be unique integers from 0 to 4; 0 disables MTP")
    return lengths


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--prefill-text", type=Path, default=ROOT.parent / "online-articles-262144-words.txt",
        help="UTF-8 article text to tokenize once and use as the prefill prefix",
    )
    parser.add_argument("--server", type=Path, default=ROOT / "build-device-memory-infra-cuda-release/bin/llama-server")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/fixed-span-8k-192k")
    parser.add_argument("--min-context", type=int, default=8192)
    parser.add_argument("--max-context", type=int, default=196608)
    parser.add_argument("--context-step", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--arena-mib", type=int, default=2368)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--port", type=int, default=1246)
    parser.add_argument("--production-container", default="llm-llmster")
    parser.add_argument("--no-manage-production", action="store_true")
    parser.add_argument("--uvm", action="store_true")
    parser.add_argument(
        "--mtp-lengths", type=parse_mtp_lengths, default=(0,), metavar="N[,N...]",
        help="maximum MTP draft lengths to sweep (0=target-only baseline; supported: 1-4; default: 0)",
    )
    return parser.parse_args()


def server_command(args: argparse.Namespace, context: int, mtp_length: int) -> list[str]:
    if mtp_length < 0 or mtp_length > 4:
        raise ValueError("MTP draft length must be between 0 and 4")
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
    if mtp_length:
        command += [
            "--kv-stream-auxiliary-layers", "1",
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(mtp_length),
        ]
    return command


def log_name(context: int, arena_mib: int, mtp_length: int) -> str:
    return f"context-{context}-arena-{arena_mib}-mtp-{mtp_length}.log"


def series(rows: list[dict], mtp_length: int) -> list[dict]:
    return sorted(
        (row for row in rows if row.get("mtp_length", 0) == mtp_length),
        key=lambda row: row["context_capacity"],
    )


def prompt_tokens_for_context(context: int, decode_tokens: int, mtp_lengths: tuple[int, ...]) -> int:
    # Draft verification needs room beyond the requested output; keep the prompt
    # identical across MTP settings so the sweep remains comparable.
    draft_headroom = max(mtp_lengths) + 1 if max(mtp_lengths) else 0
    return context - decode_tokens - draft_headroom


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


def parse_draft_acceptance(log_text: str) -> tuple[int | None, int | None]:
    last_timing = log_text.rfind("prompt eval time =")
    if last_timing >= 0:
        log_text = log_text[last_timing:]
    matches = re.findall(
        r"draft acceptance\s*=\s*[0-9.]+\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\)",
        log_text,
    )
    return tuple(map(int, matches[-1])) if matches else (None, None)


def write_outputs(rows: list[dict], output: Path) -> None:
    fields = list(rows[0])
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def plot(rows: list[dict], output: Path, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    contexts = sorted({row["context_capacity"] / 1024 for row in rows})
    lengths = sorted({row.get("mtp_length", 0) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), sharex=True)
    panels = [
        ("decode_tps", "Decode throughput", "tokens/s", "-"),
        ("prefill_tps", "Prefill throughput", "tokens/s", "--"),
        ("decode_kv_pool_mib", "Effective decode KV pool", "MiB", "-."),
        ("h2d_util_pct", "Estimated decode H2D utilization", "% of measured 50 GB/s", ":"),
    ]
    colors = {0: "#7A3E9D", 1: "#E69F00", 2: "#0072B2", 3: "#009E73", 4: "#D55E00"}
    for axis, (field, title, ylabel, style) in zip(axes.flat, panels):
        for length in lengths:
            points = series(rows, length)
            x = [row["context_capacity"] / 1024 for row in points]
            y = [row[field] for row in points]
            if all(value is None for value in y):
                continue
            label = "No MTP" if length == 0 else f"MTP max {length}"
            axis.plot(x, y, color=colors[length], linestyle=style, linewidth=2.4, marker="o", markersize=4, label=label)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.28)
    for axis in axes[1]:
        axis.set_xlabel("Configured context capacity (Ki tokens)")
    for axis in axes.flat:
        axis.set_xticks(contexts[::2])
    axes[0, 0].legend(title="Max draft length", fontsize=8)
    axes[1, 1].axhline(100, color="black", linewidth=1, alpha=0.35)
    if any(length > 0 for length in lengths):
        axes[1, 1].text(
            0.98, 0.97, "MTP H2D rate unavailable without per-evaluation accounting",
            ha="right", va="top", transform=axes[1, 1].transAxes, fontsize=8,
        )
    fig.suptitle(f"{args.model.stem} - adaptive KV / MTP draft length sweep", fontsize=15)
    fig.text(
        0.5,
        0.015,
        f"Context-matched capacity | Q8_0 K / Q4_0 V | batch/ubatch {args.batch_size}/{args.ubatch_size} | "
        f"{args.decode_tokens} decoded tokens | UVM {'on' if args.uvm else 'off'}",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(output / "fixed-span-sweep.png", dpi=180)
    fig.savefig(output / "fixed-span-sweep.svg")
    plt.close(fig)


def main() -> int:
    args = arguments()
    args.model = args.model.resolve()
    args.prefill_text = args.prefill_text.resolve()
    args.server = args.server.resolve()
    args.output = args.output.resolve()
    if not args.model.is_file() or not args.server.is_file():
        raise SystemExit("model or server executable does not exist")
    if not args.prefill_text.is_file():
        raise SystemExit(f"prefill text does not exist: {args.prefill_text}")
    if prompt_tokens_for_context(args.min_context, args.decode_tokens, args.mtp_lengths) <= 0 or args.context_step <= 0 or args.max_context < args.min_context:
        raise SystemExit("invalid context range")

    article_text = args.prefill_text.read_text(encoding="utf-8")
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
    tokens: list[int] | None = None
    try:
        for context, mtp_length in (
            (context, length)
            for context in range(args.min_context, args.max_context + 1, args.context_step)
            for length in args.mtp_lengths
        ):
            prompt_tokens = prompt_tokens_for_context(context, args.decode_tokens, args.mtp_lengths)
            log_path = logs / log_name(context, args.arena_mib, mtp_length)
            env = os.environ.copy()
            if not args.uvm:
                for key in (
                    "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
                    "GGML_CUDA_PREFER_MODEL_WEIGHTS",
                    "GGML_CUDA_PREFER_KV_HOST",
                    "GGML_CUDA_KV_ACCESSED_BY_GPU",
                ):
                    env.pop(key, None)
            command = server_command(args, context, mtp_length)
            started = time.monotonic()
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    wait_ready(url, process, log_path)
                    if tokens is None:
                        tokens = request_json(
                            url + "/tokenize",
                            {"content": article_text, "add_special": False},
                            timeout=180,
                        )["tokens"]
                        needed = max(256, prompt_tokens_for_context(
                            args.max_context, args.decode_tokens, args.mtp_lengths,
                        ))
                        if len(tokens) < needed:
                            raise RuntimeError(
                                f"prefill text has only {len(tokens)} tokens; need {needed}: {args.prefill_text}"
                            )
                    warmup = stream_completion(
                        url + "/completion",
                        {
                            "prompt": tokens[:256],
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
                    prompt = tokens[:prompt_tokens]
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
                    decoded_tokens = result["tokens"]
                    if not decoded_tokens:
                        raise RuntimeError("server returned no generated tokens despite return_tokens=true")
                    decoded_first_10_text = request_json(
                        url + "/detokenize", {"tokens": decoded_tokens[:10]}, timeout=10,
                    )["content"]
                    decoded_last_10_text = request_json(
                        url + "/detokenize", {"tokens": decoded_tokens[-10:]}, timeout=10,
                    )["content"]
                finally:
                    stop_process(process)

            pool, resident, ring, active, h2d_mib, h2d_calls = decode_layout(log_path)
            draft_accepted, draft_generated = parse_draft_acceptance(log_path.read_text(errors="replace"))
            if mtp_length and not draft_generated:
                print(f"warning: no MTP drafts recorded for context {context}, length {mtp_length}; see {log_path}", file=sys.stderr)
            decode_tps = float(timing["predicted_per_second"])
            h2d_gbs = h2d_mib * 1048576 * decode_tps / 1e9 if mtp_length == 0 else None
            row = {
                "context_capacity": context,
                "mtp_length": mtp_length,
                "mtp_draft_accepted": draft_accepted,
                "mtp_draft_generated": draft_generated,
                "mtp_acceptance_pct": 100.0 * draft_accepted / draft_generated if draft_generated else None,
                "prompt_tokens": prompt_tokens,
                "prefill_source": str(args.prefill_text),
                "decode_tokens": args.decode_tokens,
                "decoded_first_10_text": decoded_first_10_text,
                "decoded_last_10_text": decoded_last_10_text,
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
                "h2d_mib_per_token": h2d_mib if mtp_length == 0 else None,
                "h2d_calls_per_token": h2d_calls if mtp_length == 0 else None,
                "h2d_util_pct": h2d_gbs / 50.0 * 100.0 if h2d_gbs is not None else None,
                "log": str(log_path),
            }
            rows.append(row)
            write_outputs(rows, args.output)
            print(json.dumps(row, sort_keys=True), flush=True)
        try:
            plot(rows, args.output, args)
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
