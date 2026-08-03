#!/usr/bin/env python3
"""Collect restartable DFlash acceptance telemetry through llama-server."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCHEMA = "llama.dflash.telemetry-run.v1"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select_prompts(path: Path, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    eligible = 0
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw in source:
            prompt = " ".join(raw.strip().split())
            if len(prompt) < 64 or len(prompt) > 1200:
                continue
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(prompt)
                continue
            replacement = rng.randrange(eligible)
            if replacement < count:
                reservoir[replacement] = prompt
    if len(reservoir) < count:
        raise RuntimeError(
            f"{path}: only {len(reservoir)} eligible prompts for requested count {count}"
        )
    rng.shuffle(reservoir)
    return reservoir


def request_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(process: subprocess.Popen, base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with code {process.returncode}")
        try:
            health = request_json(f"{base_url}/health", None, 2.0)
            if health.get("status") == "ok":
                return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"llama-server did not become healthy: {last_error}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def count_telemetry(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as source:
        # Accept both the legacy v1 schema (--telemetry-v1-compat) and the
        # default unified v3 schema.
        return sum(
            '"schema":"llama.dflash.acceptance.v1"' in line
            or '"schema":"llama.spec_calib.v3"' in line
            for line in source
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DFlash acceptance telemetry")
    parser.add_argument("--server", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--request-log", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=640)
    parser.add_argument("--port", type=int, default=18099)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    server = Path(args.server)
    target_model = Path(args.target_model)
    draft_model = Path(args.draft_model)
    calibration_data = Path(args.calibration_data)
    telemetry = Path(args.telemetry)
    state_path = Path(args.state)
    server_log = Path(args.server_log)
    for required in (server, target_model, draft_model, calibration_data):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.requests <= 0 or args.max_tokens <= 0:
        raise ValueError("--requests and --max-tokens must be positive")

    prompts = select_prompts(calibration_data, args.requests, args.seed)
    if args.validate_only:
        print(f"DFlash telemetry preflight OK: prompts={len(prompts)}")
        return

    telemetry.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    server_log.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA:
            raise ValueError(f"{state_path}: unsupported state schema")
        state.pop("prompt_digest", None)
        next_request = int(state.get("next_request", 0))
    else:
        next_request = 0
        state = {
            "schema": SCHEMA,
            "selection": {
                "seed": args.seed,
                "source_bytes": calibration_data.stat().st_size,
                "requests": args.requests,
            },
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "next_request": 0,
            "completed": False,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "elapsed_seconds_total": 0.0,
        }
        atomic_json(state_path, state)

    if next_request >= args.requests:
        print(
            f"Using completed DFlash telemetry: {telemetry} "
            f"events={count_telemetry(telemetry)}"
        )
        return

    environment = os.environ.copy()
    environment["LLAMA_DFLASH_TELEMETRY"] = str(telemetry)
    command = [
        str(server),
        "-m", str(target_model),
        "-md", str(draft_model),
        "--spec-type", "draft-dflash",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "-c", str(args.context_size),
        "-b", "128",
        "-ub", "32",
        "-ngl", "all",
        "-ngld", "all",
        "--spec-draft-n-max", "8",
        "--spec-draft-n-min", "1",
        "--spec-draft-p-min", "0.05",
        "--parallel", "1",
        "--no-webui",
    ]
    print("+", " ".join(command), file=sys.stderr, flush=True)
    with server_log.open("a", encoding="utf-8") as server_output:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=server_output,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}"
            wait_for_server(process, base_url, 180.0)
            for index in range(next_request, args.requests):
                payload = {
                    "messages": [{"role": "user", "content": prompts[index]}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "stream": False,
                }
                started = time.monotonic()
                response = None
                error = None
                for attempt in range(3):
                    try:
                        response = request_json(
                            f"{base_url}/v1/chat/completions", payload, 600.0
                        )
                        error = None
                        break
                    except (OSError, ValueError, urllib.error.URLError) as exc:
                        error = str(exc)
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"llama-server exited with code {process.returncode}"
                            ) from exc
                        time.sleep(2.0 * (attempt + 1))
                if response is None:
                    raise RuntimeError(f"request {index} failed: {error}")

                usage = response.get("usage", {})
                state["prompt_tokens_total"] = int(state.get("prompt_tokens_total", 0)) + int(
                    usage.get("prompt_tokens") or 0
                )
                state["completion_tokens_total"] = int(state.get("completion_tokens_total", 0)) + int(
                    usage.get("completion_tokens") or 0
                )
                state["elapsed_seconds_total"] = float(state.get("elapsed_seconds_total", 0.0)) + (
                    time.monotonic() - started
                )
                state["next_request"] = index + 1
                state["telemetry_events"] = count_telemetry(telemetry)
                state["completed"] = index + 1 >= args.requests
                atomic_json(state_path, state)
                print(
                    f"DFlash telemetry request {index + 1}/{args.requests}: "
                    f"events={state['telemetry_events']}",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            stop_server(process)

    events = count_telemetry(telemetry)
    if events < args.requests:
        raise RuntimeError(
            f"DFlash emitted only {events} acceptance events for {args.requests} requests"
        )
    print(f"Collected {events} DFlash acceptance events in {telemetry}")


if __name__ == "__main__":
    main()
