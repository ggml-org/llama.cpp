#!/usr/bin/env python3
"""Capture raw benchmark stdout and classify llama-bench JSONL records."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time
from typing import Any


def atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_complete_line(output, line: bytes) -> None:
    output.write(line)
    if not line.endswith(b"\n"):
        output.write(b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=pathlib.Path, required=True)
    parser.add_argument("--completed-at", type=pathlib.Path, required=True)
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    parser.add_argument("--non-json", type=pathlib.Path, required=True)
    parser.add_argument("--classification", type=pathlib.Path, required=True)
    parser.add_argument("--max-non-json-lines", type=int, required=True)
    args = parser.parse_args()
    if args.max_non_json_lines < 0:
        raise ValueError("--max-non-json-lines must be non-negative")

    total_lines = 0
    total_bytes = 0
    json_lines = 0
    non_json_lines = 0
    blank_lines = 0
    malformed_json_like_lines = 0
    unterminated_final_data = False

    with (
        args.result.open("ab") as result,
        args.completed_at.open("a", encoding="ascii") as completed,
        args.raw.open("ab") as raw,
        args.non_json.open("ab") as non_json,
    ):
        for line in sys.stdin.buffer:
            total_lines += 1
            total_bytes += len(line)
            raw.write(line)
            stripped = line.strip()
            if not line.endswith(b"\n"):
                unterminated_final_data = True
            if not stripped:
                blank_lines += 1
                non_json.write(line)
                continue
            try:
                value = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                non_json_lines += 1
                non_json.write(line)
                if stripped.startswith(b"{"):
                    # llama-bench JSONL records are objects. RCCL warnings may
                    # legitimately begin with a bracketed timestamp.
                    malformed_json_like_lines += 1
                continue
            # Preserve every valid JSON value. summarize-tg.py remains the
            # fail-closed authority for object shape and benchmark contract.
            del value
            json_lines += 1
            append_complete_line(result, line)
            completed.write(f"{time.time_ns()}\n")

    success = (
        malformed_json_like_lines == 0
        and not unterminated_final_data
        and non_json_lines <= args.max_non_json_lines
    )
    classification = {
        "schema_version": 1,
        "consumer_success": success,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "json_lines": json_lines,
        "non_json_lines": non_json_lines,
        "blank_lines": blank_lines,
        "malformed_json_like_lines": malformed_json_like_lines,
        "unterminated_final_data": unterminated_final_data,
        "max_non_json_lines": args.max_non_json_lines,
        "excessive_non_json_output": non_json_lines > args.max_non_json_lines,
        "raw_stream_preserved": True,
        "json_completion_timestamps": json_lines,
    }
    atomic_json(args.classification, classification)
    if not success:
        print(
            "stdout classification failed: "
            f"non_json={non_json_lines}/{args.max_non_json_lines} "
            f"malformed_json_like={malformed_json_like_lines} "
            f"unterminated={int(unterminated_final_data)}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"stdout capture failed: {exc}", file=sys.stderr)
        raise SystemExit(2)