#!/usr/bin/env python3
"""Measure process-start-to-first-row latency for SYCL llama-bench builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


class ColdJitError(RuntimeError):
    """A cold-JIT campaign could not produce admissible evidence."""


def parse_assignments(values: list[str], *, separator: str = "=") -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, found, item = value.partition(separator)
        if not found or not key or not item:
            raise ColdJitError(f"expected NAME{separator}VALUE, got {value!r}")
        parsed[key] = item
    return parsed


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ColdJitError("cannot summarize an empty sample")
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    dof = len(values) - 1
    if dof <= 0:
        critical = 0.0
    elif dof <= 30:
        critical = T95.get(dof, 1.96)
    else:
        # Asymptotic expansion of the two-sided 95% Student-t quantile.
        z = 1.959963984540054
        z2 = z * z
        z3 = z2 * z
        z5 = z3 * z2
        z7 = z5 * z2
        term1 = (z3 + z) / (4.0 * dof)
        term2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * dof**2)
        term3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * dof**3)
        critical = z + term1 + term2 + term3
    ci95 = critical * stdev / len(values) ** 0.5
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": stdev,
        "ci95": ci95,
        "lower95": mean - ci95,
        "upper95": mean + ci95,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holder_snapshot(render_node: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["fuser", "-v", render_node],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode == 1 and not combined.strip():
        return {
            "returncode": proc.returncode,
            "pids": [],
            "output": combined,
        }
    if proc.returncode != 0:
        detail = combined.strip() or f"fuser exited {proc.returncode} without details"
        raise ColdJitError(f"fuser failed for {render_node}: {detail}")
    return {
        "returncode": proc.returncode,
        "pids": sorted({int(value) for value in re.findall(r"\b\d+\b", combined)}),
        "output": combined,
    }


def kill_and_reap(proc: subprocess.Popen[str]) -> None:
    proc.kill()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _stdout_reader(
    stream: Any, messages: queue.Queue[tuple[float, str] | None]
) -> None:
    for line in stream:
        messages.put((time.monotonic(), line))
    messages.put(None)


def effective_library_path(bin_dir: Path, env_extra: dict[str, str]) -> str:
    inherited = env_extra.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    return str(bin_dir) + (f":{inherited}" if inherited else "")


def run_sample(
    *,
    bench: Path,
    bin_dir: Path,
    model: Path,
    timeout_s: int,
    env_extra: dict[str, str],
    stderr_path: Path,
) -> dict[str, Any]:
    argv = [
        str(bench),
        "-m",
        str(model),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-p",
        "512",
        "-n",
        "128",
        "-b",
        "512",
        "-ub",
        "512",
        "--no-warmup",
        "-r",
        "1",
        "-o",
        "jsonl",
    ]
    env = os.environ.copy()
    env.update(env_extra)
    env["SYCL_CACHE_PERSISTENT"] = "0"
    env["LD_LIBRARY_PATH"] = effective_library_path(bin_dir, env_extra)

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    stdout_lines: list[str] = []
    first_row_s: float | None = None
    with stderr_path.open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        messages: queue.Queue[tuple[float, str] | None] = queue.Queue()
        reader = threading.Thread(
            target=_stdout_reader, args=(proc.stdout, messages), daemon=True
        )
        reader.start()
        deadline = started + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                kill_and_reap(proc)
                raise ColdJitError(f"llama-bench timed out after {timeout_s}s")
            try:
                message = messages.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if proc.poll() is not None and not reader.is_alive():
                    break
                continue
            if message is None:
                break
            observed, line = message
            stdout_lines.append(line)
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                kill_and_reap(proc)
                raise ColdJitError("llama-bench emitted a non-object JSON row")
            avg_ts = row.get("avg_ts")
            if (
                isinstance(avg_ts, bool)
                or not isinstance(avg_ts, (int, float))
                or not math.isfinite(avg_ts)
            ):
                kill_and_reap(proc)
                raise ColdJitError(f"llama-bench emitted invalid avg_ts: {avg_ts!r}")
            rows.append(row)
            if first_row_s is None:
                first_row_s = observed - started
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                returncode = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                kill_and_reap(proc)
                raise ColdJitError(f"llama-bench timed out after {timeout_s}s")
        else:
            kill_and_reap(proc)
            raise ColdJitError(f"llama-bench timed out after {timeout_s}s")
        reader.join(timeout=1)

    wall_s = time.monotonic() - started
    pp_rows = [
        row for row in rows if row.get("n_prompt") == 512 and row.get("n_gen") == 0
    ]
    tg_rows = [
        row for row in rows if row.get("n_prompt") == 0 and row.get("n_gen") == 128
    ]
    return {
        "argv": argv,
        "returncode": returncode,
        "first_valid_row_s": first_row_s,
        "process_wall_s": wall_s,
        "rows": rows,
        "pp512": pp_rows[-1] if pp_rows else None,
        "tg128": tg_rows[-1] if tg_rows else None,
        "stdout": "".join(stdout_lines),
        "valid": returncode == 0
        and isinstance(first_row_s, (int, float))
        and math.isfinite(first_row_s)
        and bool(pp_rows)
        and bool(tg_rows),
        "effective_ld_library_path": env["LD_LIBRARY_PATH"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", default=[], help="NAME=GGUF (repeatable)"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--repetitions", type=int, default=6, help="sample 0 is discarded"
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    args = parser.parse_args()

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        product_path = args.out_dir / "product.json"
        product_path.unlink(missing_ok=True)
        if args.repetitions < 3:
            raise ColdJitError(
                "--repetitions must be >= 3 because sample 0 is discarded"
            )
        models = {
            name: Path(path).resolve()
            for name, path in parse_assignments(args.model).items()
        }
        if not models:
            raise ColdJitError("at least one --model NAME=GGUF is required")
        missing_models = [str(path) for path in models.values() if not path.is_file()]
        if missing_models:
            raise ColdJitError(f"model files do not exist: {', '.join(missing_models)}")
        bin_dir = args.bin_dir.resolve()
        bench = bin_dir / "llama-bench"
        if not bench.is_file() or not os.access(bench, os.X_OK):
            raise ColdJitError(f"missing executable {bench}")
        env_extra = parse_assignments(args.env)

        before = holder_snapshot(args.render_node)
        if before["pids"]:
            raise ColdJitError(f"render node is not idle: {before['output'].strip()}")

        model_results: dict[str, Any] = {}
        for model_name, model in models.items():
            samples: list[dict[str, Any]] = []
            for repetition in range(args.repetitions):
                stderr_path = args.out_dir / f"{model_name}-rep{repetition}.stderr.log"
                sample = run_sample(
                    bench=bench,
                    bin_dir=bin_dir,
                    model=model,
                    timeout_s=args.timeout,
                    env_extra=env_extra,
                    stderr_path=stderr_path,
                )
                sample_path = args.out_dir / f"{model_name}-rep{repetition}.json"
                sample_path.write_text(
                    json.dumps(sample, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"{model_name} rep={repetition} valid={sample['valid']} first_row={sample['first_valid_row_s']}",
                    flush=True,
                )
                samples.append(sample)
            retained = samples[1:]
            if not all(sample["valid"] for sample in samples):
                raise ColdJitError(f"{model_name} has an invalid sample")
            model_results[model_name] = {
                "model": str(model),
                "samples": samples,
                "retained_repetitions": list(range(1, args.repetitions)),
                "first_valid_row_s": summarize(
                    [sample["first_valid_row_s"] for sample in retained]
                ),
                "process_wall_s": summarize(
                    [sample["process_wall_s"] for sample in retained]
                ),
                "pp512_ts": summarize(
                    [sample["pp512"]["avg_ts"] for sample in retained]
                ),
                "tg128_ts": summarize(
                    [sample["tg128"]["avg_ts"] for sample in retained]
                ),
            }

        after = holder_snapshot(args.render_node)
        if after["pids"] != before["pids"]:
            raise ColdJitError(
                f"render-node holders changed: before={before['pids']} after={after['pids']}"
            )
        product = {
            "bin_dir": str(bin_dir),
            "bench_sha256": sha256_file(bench),
            "environment": {
                **env_extra,
                "SYCL_CACHE_PERSISTENT": "0",
                "LD_LIBRARY_PATH": effective_library_path(bin_dir, env_extra),
            },
            "holder_before": before,
            "holder_after": after,
            "models": model_results,
        }
        product_path = args.out_dir / "product.json"
        product_path.write_text(
            json.dumps(product, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {product_path}")
        return 0
    except (ColdJitError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
