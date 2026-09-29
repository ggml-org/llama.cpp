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
import urllib.request
from pathlib import Path
from typing import Callable

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
        "--prefill-text", type=Path, default=ROOT / "benchmarks/data/online-articles-262144-words.txt",
        help="UTF-8 article text to tokenize once and use as the prefill prefix (default: benchmarks/data/online-articles-262144-words.txt)",
    )
    parser.add_argument("--server", type=Path, default=ROOT / "build-device-memory-infra-cuda-release/bin/llama-server")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/fixed-span-8k-256k")
    parser.add_argument("--min-context", type=int, default=8192)
    parser.add_argument("--max-context", type=int, default=262144,
                        help="inclusive context endpoint in tokens (default: 262144 / 256 Ki)")
    parser.add_argument("--context-step", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--arena-mib", type=int, default=2368,
                        help="fixed arena size in MiB, or starting probe when --auto-max-arena is set")
    parser.add_argument("--auto-max-arena", action="store_true",
                        help="find the largest full-workload arena in 1 MiB steps for each context and MTP mode; no reserve; can be slow")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--port", type=int, default=1246)
    parser.add_argument("--production-container", default="llm-llmster")
    parser.add_argument("--no-manage-production", action="store_true")
    parser.add_argument("--uvm", action="store_true")
    parser.add_argument("--no-kv-stream-rs-rollback", action="store_true",
                        help="use the old full-checkpoint path instead of default MTP recurrent rollback")
    parser.add_argument(
        "--mtp-lengths", type=parse_mtp_lengths, default=(0,), metavar="N[,N...]",
        help="maximum MTP draft lengths to sweep (0=target-only baseline; supported: 1-4; default: 0)",
    )
    return parser.parse_args()


def server_command(args: argparse.Namespace, context: int, mtp_length: int, arena_mib: int | None = None) -> list[str]:
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
        "--kv-stream-arena-mib", str(args.arena_mib if arena_mib is None else arena_mib),
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
        if getattr(args, "no_kv_stream_rs_rollback", False):
            command.append("--no-kv-stream-rs-rollback")
    return command


def server_environment(uvm: bool) -> dict[str, str]:
    env = os.environ.copy()
    if uvm:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    else:
        for key in (
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
            "GGML_CUDA_PREFER_MODEL_WEIGHTS",
            "GGML_CUDA_PREFER_KV_HOST",
            "GGML_CUDA_KV_ACCESSED_BY_GPU",
        ):
            env.pop(key, None)
    return env


def is_arena_capacity_failure(log_text: str) -> bool:
    if re.search(r"out of memory|cudaMalloc failed|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED", log_text, re.IGNORECASE):
        return True
    return "invalid resource handle" in log_text and "copy_queue::~copy_queue" in log_text and "acquire_mtp_layer" in log_text


def find_max_arena_mib(start: int, probe: Callable[[int], bool]) -> int:
    if start < 1:
        raise ValueError("arena search start must be positive")
    if probe(start):
        low = start
        step = 16
        for _ in range(32):
            high = low + step
            if high > 1048576:
                raise RuntimeError("arena search could not find an upper bound below 1 TiB")
            if not probe(high):
                break
            low = high
            step *= 2
        else:
            raise RuntimeError("arena search could not find an upper bound")
    else:
        high = start
        step = 16
        for _ in range(32):
            low = max(1, high - step)
            if probe(low):
                break
            if low == 1:
                raise RuntimeError("no allocatable arena found")
            high = low
            step *= 2
        else:
            raise RuntimeError("no allocatable arena found")
    while high - low > 1:
        middle = (low + high) // 2
        if probe(middle):
            low = middle
        else:
            high = middle
    return low


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


def probe_prompt_prefix(url: str, prompt: list[int], decode_tokens: int, processed_min: int) -> None:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": decode_tokens,
        "temperature": 0,
        "seed": 123,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": True,
        "return_tokens": True,
        "return_progress": True,
    }).encode("utf-8")
    request = urllib.request.Request(url + "/completion", data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    processed = 0
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data:"):
                continue
            value = raw_line[5:].strip()
            if not value or value == b"[DONE]":
                continue
            event = json.loads(value)
            if "error" in event:
                raise RuntimeError(f"prompt probe returned an error: {event['error']}")
            progress = event.get("prompt_progress")
            if isinstance(progress, dict):
                processed = int(progress.get("processed", 0))
                if processed >= processed_min:
                    return
            if event.get("stop") is True:
                raise RuntimeError(f"prompt probe ended before reaching {processed_min} tokens")
    raise RuntimeError(f"stream ended before prompt probe reached {processed_min} tokens (processed={processed})")


def probe_arena(args: argparse.Namespace, context: int, mtp_length: int, arena_mib: int, logs: Path, url: str,
                article_text: str | None = None, token_cache: dict[str, list[int] | None] | None = None,
                full_workload: bool = False, result_cache: dict[int, dict] | None = None) -> bool:
    probe_logs = logs / "arena-probes"
    probe_logs.mkdir(exist_ok=True)
    log_path = probe_logs / (("full-" if full_workload else "fast-") + log_name(context, arena_mib, mtp_length))
    command = server_command(args, context, mtp_length, arena_mib)
    failure = None
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=server_environment(args.uvm))
        try:
            wait_ready(url, process, log_path)
            if article_text is None:
                warmups = [("The moon reflects sunlight. " * 52, 4),
                           ("The moon reflects sunlight. " * 128, args.decode_tokens)]
            else:
                if token_cache is None:
                    raise ValueError("token cache is required for a full-length arena probe")
                if token_cache.get("tokens") is None:
                    token_cache["tokens"] = request_json(url + "/tokenize",
                        {"content": article_text, "add_special": False}, timeout=180)["tokens"]
                source_tokens = token_cache["tokens"]
                prompt_count = prompt_tokens_for_context(context, args.decode_tokens, args.mtp_lengths)
                if source_tokens is None or len(source_tokens) < prompt_count:
                    raise RuntimeError(f"prefill text has fewer than {prompt_count} tokens")
                warmups = [(source_tokens[:256], 4)]
            for warmup_prompt, n_predict in warmups:
                result = stream_completion(url + "/completion", {
                    "prompt": warmup_prompt,
                    "n_predict": n_predict,
                    "temperature": 0,
                    "seed": 123,
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "stream": True,
                    "return_tokens": True,
                }, timeout=120)
                if result["timings"].get("predicted_n") != n_predict:
                    raise RuntimeError(f"arena warmup did not complete: {result['timings']}")
            if article_text is not None:
                if full_workload:
                    result = stream_completion(url + "/completion", {
                        "prompt": source_tokens[:prompt_count],
                        "n_predict": args.decode_tokens,
                        "temperature": 0,
                        "seed": 123,
                        "ignore_eos": True,
                        "cache_prompt": False,
                        "stream": True,
                        "return_tokens": True,
                    }, timeout=1800)
                    timing = result["timings"]
                    if timing.get("prompt_n") != prompt_count or timing.get("predicted_n") != args.decode_tokens or len(result["tokens"]) != args.decode_tokens:
                        raise RuntimeError(f"arena full workload did not complete: timings={timing}, token_ids={len(result['tokens'])}")
                    if result_cache is not None:
                        first_text = request_json(url + "/detokenize", {"tokens": result["tokens"][:10]}, timeout=10)["content"]
                        last_text = request_json(url + "/detokenize", {"tokens": result["tokens"][-10:]}, timeout=10)["content"]
                else:
                    probe_prompt_prefix(url, source_tokens[:prompt_count], args.decode_tokens, min(prompt_count, 2048))
        except Exception as error:
            failure = error
        finally:
            stop_process(process)
    label = "full" if full_workload else "fast"
    if is_arena_capacity_failure(log_path.read_text(errors="replace")):
        print(f"arena probe ({label}): context={context} mtp={mtp_length} arena={arena_mib} MiB -> capacity failure", flush=True)
        return False
    if failure is not None:
        raise RuntimeError(f"arena probe failed at context={context}, mtp={mtp_length}, arena={arena_mib} MiB; see {log_path}: {failure}") from failure
    if full_workload and result_cache is not None:
        result_cache[arena_mib] = {
            "result": result,
            "first_text": first_text,
            "last_text": last_text,
            "log_path": log_path,
            "wall_seconds": time.monotonic() - started,
        }
    print(f"arena probe ({label}): context={context} mtp={mtp_length} arena={arena_mib} MiB -> fits", flush=True)
    return True


def arena_for_point(args: argparse.Namespace, context: int, mtp_length: int, logs: Path, url: str,
                    article_text: str | None = None, token_cache: dict[str, list[int] | None] | None = None,
                    result_cache: dict[int, dict] | None = None) -> int:
    if not args.auto_max_arena:
        return args.arena_mib
    fast_max = find_max_arena_mib(args.arena_mib,
        lambda candidate: probe_arena(args, context, mtp_length, candidate, logs, url, article_text, token_cache))
    if article_text is None:
        return fast_max
    checked: dict[int, bool] = {}
    def full_fits(candidate: int) -> bool:
        if candidate > fast_max:
            return False
        if candidate not in checked:
            checked[candidate] = probe_arena(args, context, mtp_length, candidate, logs, url,
                article_text, token_cache, full_workload=True, result_cache=result_cache)
        return checked[candidate]
    if full_fits(fast_max):
        return fast_max
    if fast_max == 1:
        raise RuntimeError("no arena completed the full benchmark workload")
    return find_max_arena_mib(max(1, fast_max - 16), full_fits)


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
    if args.no_kv_stream_rs_rollback and not any(args.mtp_lengths):
        raise SystemExit("--no-kv-stream-rs-rollback requires an MTP length")
    args.model = args.model.resolve()
    args.prefill_text = args.prefill_text.resolve()
    args.server = args.server.resolve()
    args.output = args.output.resolve()
    if args.arena_mib < 1:
        raise SystemExit("--arena-mib must be positive")
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
    token_cache: dict[str, list[int] | None] = {"tokens": None}
    try:
        for context, mtp_length in (
            (context, length)
            for context in range(args.min_context, args.max_context + 1, args.context_step)
            for length in args.mtp_lengths
        ):
            prompt_tokens = prompt_tokens_for_context(context, args.decode_tokens, args.mtp_lengths)
            full_results: dict[int, dict] = {}
            arena_mib = arena_for_point(args, context, mtp_length, logs, url, article_text, token_cache, full_results)
            cached = full_results.get(arena_mib)
            if cached is not None:
                log_path = cached["log_path"]
                result = cached["result"]
                timing = result["timings"]
                decoded_tokens = result["tokens"]
                decoded_first_10_text = cached["first_text"]
                decoded_last_10_text = cached["last_text"]
                wall_seconds = cached["wall_seconds"]
            else:
                log_path = logs / log_name(context, arena_mib, mtp_length)
                env = server_environment(args.uvm)
                command = server_command(args, context, mtp_length, arena_mib)
                started = time.monotonic()
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    try:
                        wait_ready(url, process, log_path)
                        if tokens is None:
                            tokens = token_cache["tokens"]
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
                            raise RuntimeError(f"incomplete benchmark response: timings={timing}, stop_type={result['stop_type']}")
                        decoded_tokens = result["tokens"]
                        if len(decoded_tokens) != args.decode_tokens:
                            raise RuntimeError(f"server returned {len(decoded_tokens)} token IDs for {args.decode_tokens} generated tokens")
                        decoded_first_10_text = request_json(
                            url + "/detokenize", {"tokens": decoded_tokens[:10]}, timeout=10,
                        )["content"]
                        decoded_last_10_text = request_json(
                            url + "/detokenize", {"tokens": decoded_tokens[-10:]}, timeout=10,
                        )["content"]
                    finally:
                        stop_process(process)

                wall_seconds = time.monotonic() - started

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
                "arena_mib": arena_mib,
                "prefill_tps": float(timing["prompt_per_second"]),
                "decode_tps": decode_tps,
                "prompt_ms": float(timing["prompt_ms"]),
                "predicted_ms": float(timing["predicted_ms"]),
                "wall_seconds": wall_seconds,
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
