#!/usr/bin/env python3
"""Run the canonical P5 llama-server slot save/restore protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


RequestJson = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]
FRANCE_PROMPT = "What is the capital of France?"
GERMANY_PROMPT = "What is the capital of Germany?"


def completion_payload(prompt: str, id_slot: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0,
        "seed": 123,
        "cache_prompt": True,
        "id_slot": id_slot,
    }


def exercise_state_roundtrip(
    request_json: RequestJson,
    *,
    filename: str,
    expected_tokens: int,
    expected_bytes: int,
) -> dict[str, Any]:
    first = request_json("POST", "/completion", completion_payload(FRANCE_PROMPT, 1))
    save = request_json("POST", "/slots/1?action=save", {"filename": filename})
    changed = request_json("POST", "/completion", completion_payload(GERMANY_PROMPT, 1))
    restore = request_json("POST", "/slots/0?action=restore", {"filename": filename})
    restored = request_json("POST", "/completion", completion_payload(GERMANY_PROMPT, 0))
    original = request_json("POST", "/completion", completion_payload(GERMANY_PROMPT, 1))

    checks = {
        "saved_token_count": save.get("n_saved") == expected_tokens,
        "saved_byte_count": save.get("n_written") == expected_bytes,
        "restored_token_count": restore.get("n_restored") == expected_tokens,
        "restored_byte_count": restore.get("n_read") == expected_bytes,
        "restored_content_matches_original": restored.get("content") == original.get("content"),
        "source_slot_reuse_content_matches": changed.get("content") == original.get("content"),
        "restored_prompt_n": (restored.get("timings") or {}).get("prompt_n") == 2,
        "changed_prompt_n": (changed.get("timings") or {}).get("prompt_n") == 2,
    }
    failure_names = {
        "saved_token_count": "save token count",
        "saved_byte_count": "save byte count",
        "restored_token_count": "restore token count",
        "restored_byte_count": "restore byte count",
        "restored_content_matches_original": "cross-slot deterministic continuation",
        "source_slot_reuse_content_matches": "source-slot deterministic continuation",
        "restored_prompt_n": "restored prompt reuse",
        "changed_prompt_n": "changed prompt reuse",
    }
    failures = [failure_names[name] for name, passed in checks.items() if not passed]
    return {
        "first": first,
        "save": save,
        "changed": changed,
        "restore": restore,
        "restored": restored,
        "original": original,
        "checks": checks,
        "expected": {"tokens": expected_tokens, "bytes": expected_bytes},
        "failures": failures,
        "pass": not failures,
    }


def make_requester(base_url: str, timeout_s: float) -> RequestJson:
    def request_json(method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"expected object response from {path}")
        return parsed

    return request_json


def wait_for_health(base_url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    return False


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--health-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--expected-tokens", type=int, default=12)
    parser.add_argument("--expected-bytes", type=int, default=836576)
    parser.add_argument("--filename", default="p5-state-slot1.bin")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server_bin = args.server_bin.resolve()
    model = args.model.resolve()
    out_dir = args.out_dir.resolve()
    if not server_bin.is_file() or not os.access(server_bin, os.X_OK):
        print(f"invalid server binary: {server_bin}", file=sys.stderr)
        return 2
    if not model.is_file() or model.stat().st_size <= 0:
        print(f"invalid model file: {model}", file=sys.stderr)
        return 2
    if args.expected_tokens <= 0 or args.expected_bytes <= 0:
        print("expected token and byte counts must be positive", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "server-state-roundtrip.json"
    log_path = out_dir / "server.log"
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(server_bin),
        "--model",
        str(model),
        "--n-gpu-layers",
        "99",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--ctx-size",
        "1024",
        "--parallel",
        "2",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--slot-save-path",
        str(out_dir),
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            text=True,
        )
        try:
            if not wait_for_health(base_url, args.health_timeout):
                result = {
                    "pass": False,
                    "failures": ["server health timeout"],
                    "command": command,
                    "server_returncode": process.poll(),
                }
            else:
                result = exercise_state_roundtrip(
                    make_requester(base_url, args.request_timeout),
                    filename=args.filename,
                    expected_tokens=args.expected_tokens,
                    expected_bytes=args.expected_bytes,
                )
                result["command"] = command
        except (OSError, ValueError, urllib.error.URLError) as error:
            result = {
                "pass": False,
                "failures": [f"protocol error: {error}"],
                "command": command,
            }
        finally:
            stop_server(process)

    result["server_returncode"] = process.returncode
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": result["pass"], "failures": result["failures"], "result": str(result_path)}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
