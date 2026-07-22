#!/usr/bin/env python3
"""Benchmark fork-only Arc A770 surfaces against upstream controls.

This runner is intentionally explicit about environment-controlled modes:
- default: lets TURBO_LAYER_ADAPTIVE / TURBO_AUTO_ASYMMETRIC apply.
- pure: disables both auto policies so requested turbo K/V is what runs.
- xmx: enables GGML_SYCL_FA_XMX to route eligible FA cases to the XMX path.

It writes JSONL records for each subprocess plus a compact Markdown summary.

`--campaign product` adds a sole-tenancy product/depth harness used for the
SYCL performance plan. One `llama-bench -r 1` invocation per sample with
`-m MODEL -ngl 99 -fa on -ctk KV -ctv KV -n 128 -b 512 -ub 512 --no-warmup
-o json` (plus `-d DEPTH` only when DEPTH > 0). Six samples per cell,
paired percent samples (candidate/baseline - 1)*100 with 95% t-interval.
The runner probes `fuser /dev/dri/renderD128` immediately before each
leg; if any holder is reported, the leg is aborted with exit 70 and the
holder text is printed to stderr. The runner does NOT kill any
Arc-using process; the authorized orchestrator command (run by the
human) is:
    sudo systemctl stop llama-sycl.cpp.service
    sudo fuser -k /dev/dri/renderD128
repeated until the fuser exit signals no holder.

By default both arms use `--bin-dir`. Supplying `--candidate-bin-dir`
routes the candidate arm through an independently built `llama-bench`,
which is required for compile-time/code-generation comparisons.
The canonical baseline is the empty env and runs alone (six launches per
cell) when no `--env` is supplied. `--env NAME=VALUE` (repeatable) enables
the candidate arm; both arms run the same number of launches under
alternating pair order.

Pairing rule: paired percent samples align by sample index (rep 1 of
baseline is paired with rep 1 of candidate, etc.). A cell with any
missing or failed retained sample is recorded as invalid and emits no
candidate/paired stats; the failure list names the dropped reps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/mnt/mrgr")

# Model paths are relative to MODELS_ROOT (default /mnt/mrgr); override the env var to relocate.
DEFAULT_MODELS = [
    ("llama31-8b-heretic", "models/llama31-8b-heretic/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf"),
    ("qwen3-coder-30b-a3b", "gguf/Qwen3-Coder-30B-A3B-UD-Q3_K_XL/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf"),
]

FA_ROUTE_RE = re.compile(
    r"GGML_SYCL_FA_ROUTE: route=(?P<route>VEC|TILE|XMX|Q8_GQA) "
    r"phase=(?P<phase>decode|prefill) q_tokens=(?P<q_tokens>\d+) "
    r"head_dim=(?P<head_dim>\d+) gqa=(?P<gqa>\d+) "
    r"type_k=(?P<type_k>\S+) type_v=(?P<type_v>\S+) "
    r"k_quants_first=(?P<k_quants_first>[01]) "
    r"v_quants_first=(?P<v_quants_first>[01])"
)
FA_PROFILE_RE = re.compile(
    r"GGML_SYCL_FA_PROFILE: route=(?P<route>VEC|TILE) "
    r"layout=(?P<layout>canonical|quants-first) "
    r"launches=(?P<launches>\d+) conversion_us=(?P<conversion_us>\d+) "
    r"conversion_bytes=(?P<conversion_bytes>\d+) stage1_us=(?P<stage1_us>\d+) "
    r"combine_us=(?P<combine_us>\d+) gqa=(?P<gqa>\d+) "
    r"repeated_packed_kv_bytes=(?P<repeated_packed_kv_bytes>\d+)"
)
GRAPH_PROFILE_RE = re.compile(
    r"GGML_SYCL_GRAPH_PROFILE: graph_calls=(?P<graph_calls>\d+) "
    r"direct_calls=(?P<direct_calls>\d+) nodes=(?P<nodes>\d+) "
    r"prepare_us=(?P<prepare_us>\d+) record_us=(?P<record_us>\d+) "
    r"finalize_calls=(?P<finalize_calls>\d+) finalize_us=(?P<finalize_us>\d+) "
    r"update_calls=(?P<update_calls>\d+) update_fallbacks=(?P<update_fallbacks>\d+) "
    r"update_us=(?P<update_us>\d+) submit_us=(?P<submit_us>\d+) "
    r"wait_us=(?P<wait_us>\d+) direct_enqueue_us=(?P<direct_enqueue_us>\d+)"
)




def _effective_env(env_extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
    for knob in (
        "GGML_SYCL_FA_XMX",
        "GGML_SYCL_FA_FORCE_VEC_STANDARD",
        "GGML_SYCL_FA_Q8_GQA_TILE",
        "GGML_SYCL_FA_Q8_GQA_DIRECT",
        "GGML_SYCL_FA_PROFILE",
        "GGML_SYCL_Q8_KV_QUANTS_FIRST",
        "LLAMA_ENABLE_INNERQ",
        "TURBO_LAYER_ADAPTIVE",
        "TURBO_AUTO_ASYMMETRIC",
    ):
        env.pop(knob, None)
    env.update(env_extra)
    return env


def _redacted_env(env: dict[str, str]) -> dict[str, str]:
    secret_markers = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "AUTH")
    return {
        key: "<redacted>" if any(marker in key.upper() for marker in secret_markers) else value
        for key, value in sorted(env.items())
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_command(argv: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _cmake_cache_values(cache_path: Path) -> dict[str, str]:
    prefixes = ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "GGML_SYCL", "LLAMA_")
    values: dict[str, str] = {}
    if not cache_path.is_file():
        return values
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("//", "#")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key_with_type, value = line.split("=", 1)
        key = key_with_type.split(":", 1)[0]
        if key.startswith(prefixes):
            values[key] = value
    return values


def collect_product_provenance(
    bin_dir: Path,
    candidate_bin_dir: Path,
    baseline_env: dict[str, str],
    candidate_env: dict[str, str] | None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    bench_path = bin_dir / "llama-bench"
    sycl_lib = bin_dir / "libggml-sycl.so"
    candidate_bench_path = candidate_bin_dir / "llama-bench"
    candidate_sycl_lib = candidate_bin_dir / "libggml-sycl.so"
    baseline_effective = _effective_env(baseline_env)
    candidate_effective = _effective_env(candidate_env or {})
    return {
        "repository_commit": _capture_command(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
        ),
        "llama_cli_version": _capture_command(
            [str(bin_dir / "llama-cli"), "--version"], baseline_effective
        ),
        "sha256": {
            str(bench_path): _sha256_file(bench_path),
            str(sycl_lib): _sha256_file(sycl_lib) if sycl_lib.is_file() else None,
            str(candidate_bench_path): _sha256_file(candidate_bench_path),
            str(candidate_sycl_lib): (
                _sha256_file(candidate_sycl_lib)
                if candidate_sycl_lib.is_file()
                else None
            ),
        },
        "cmake_cache": _cmake_cache_values(
            bin_dir.parent / "CMakeCache.txt"
        ),
        "candidate_cmake_cache": _cmake_cache_values(
            candidate_bin_dir.parent / "CMakeCache.txt"
        ),
        "sycl_ls": _capture_command(["sycl-ls"], baseline_effective),
        "kernel": _capture_command(["uname", "-a"]),
        "compute_runtime": _capture_command(
            [
                "pacman",
                "-Q",
                "intel-compute-runtime",
                "intel-graphics-compiler",
                "level-zero-loader",
            ]
        ),
        "baseline_effective_env": _redacted_env(baseline_effective),
        "candidate_effective_env": _redacted_env(candidate_effective),
    }


def run(argv: list[str], env_extra: dict[str, str], timeout_s: int, cwd: Path | None = None) -> dict[str, Any]:
    env = _effective_env(env_extra)
    t0 = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_s": round(time.time() - t0, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": 124,
            "elapsed_s": round(time.time() - t0, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout_s": timeout_s,
        }
    except OSError as exc:
        return {
            "ok": False,
            "returncode": -1,
            "elapsed_s": round(time.time() - t0, 3),
            "stdout": "",
            "stderr": str(exc),
        }


def parse_bench(stdout: str) -> list[dict[str, Any]]:
    try:
        start = stdout.find("[")
        end = stdout.rfind("]")
        if start != -1 and end != -1:
            data = json.loads(stdout[start:end + 1])
        else:
            data = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        rows.append({
            "build_commit": row.get("build_commit"),
            "model_type": row.get("model_type"),
            "type_k": row.get("type_k"),
            "type_v": row.get("type_v"),
            "flash_attn": row.get("flash_attn"),
            "n_prompt": row.get("n_prompt"),
            "n_gen": row.get("n_gen"),
            "avg_ts": row.get("avg_ts"),
            "stddev_ts": row.get("stddev_ts"),
            "samples_ts": row.get("samples_ts"),
        })
    return rows


def bench_case(bin_dir: Path, model: str, kv: tuple[str, str], fa: str, p: int, n: int, reps: int) -> list[str]:
    return [
        str(bin_dir / "llama-bench"),
        "-m", model,
        "-ngl", "99",
        "-fa", fa,
        "-ctk", kv[0],
        "-ctv", kv[1],
        "-p", str(p),
        "-n", str(n),
        "-r", str(reps),
        "-o", "json",
    ]


# ---------------------------------------------------------------------------
# --campaign product: sole-tenancy product/depth harness
# ---------------------------------------------------------------------------

DEFAULT_PRODUCT_DEPTHS: tuple[int, ...] = (0, 2048, 4096, 8192, 16384)
DEFAULT_PRODUCT_KV_TYPES: tuple[tuple[str, str], ...] = (("f16", "f16"), ("q8_0", "q8_0"))
DEFAULT_PRODUCT_REPETITIONS: int = 6

# Treat both the current i915 driver and the xe driver as Arc fault sources.
DMESG_RE = re.compile(
    r"(i915|xe).*(?:reset|hang|timeout|GPU HANG|device.?lost)", re.IGNORECASE
)

SOLE_TENANCY_EXIT = 70
QK8_0 = 32
Q8_0_BLOCK_BYTES = 34


class SoleTenancyViolation(Exception):
    """Raised when /dev/dri/renderD128 has any holder.

    The runner is a non-mutating probe: it does NOT kill the holder. The
    authorized external orchestrator command (run by the human) is the
    only authority for killing Arc-using processes. The campaign catches
    this exception, prints the holder text to stderr, and exits 70.
    """

    def __init__(self, holder_lines: list[str], probe_error: str | None = None):
        self.holder_lines = holder_lines
        self.probe_error = probe_error
        head = "Sole tenancy violated: /dev/dri/renderD128 has holders."
        if probe_error:
            head += f" Probe error: {probe_error}"
        head += (
            " Aborting leg. Run the authorized orchestrator command: "
            "sudo systemctl stop llama-sycl.cpp.service && "
            "sudo fuser -k /dev/dri/renderD128, then repeat the non-mutating "
            "probe until fuser exits 1 with no output."
        )
        super().__init__(head + "\n" + "\n".join(holder_lines))


def check_sole_tenancy(
    fuser_path: str = "/dev/dri/renderD128",
    runner=subprocess.run,
    fuser_timeout: int = 30,
) -> None:
    """Non-mutating probe: raise SoleTenancyViolation if any holder exists."""
    try:
        proc = runner(
            ["fuser", fuser_path],
            check=False, capture_output=True, text=True, timeout=fuser_timeout,
        )
    except Exception as exc:  # pragma: no cover
        raise SoleTenancyViolation([], probe_error=str(exc)) from exc
    output = "\n".join(
        part.strip() for part in (proc.stdout, proc.stderr) if part and part.strip()
    )
    if proc.returncode == 1 and not output:
        return
    holders = output.splitlines() or [
        f"<fuser exited {proc.returncode} without holder or error details>"
    ]
    raise SoleTenancyViolation(holders)


def capture_dmesg(
    out_path: Path,
    since: str = "-1h",
    runner=subprocess.run,
) -> int:
    """Capture i915/xe reset, hang, timeout, or device-loss lines."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = runner(
            ["dmesg", "-T", "--since", since],
            check=False, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            proc = runner(
                ["sudo", "-n", "dmesg", "-T", "--since", since],
                check=False, capture_output=True, text=True, timeout=30,
            )
    except Exception as exc:  # pragma: no cover
        out_path.write_text(f"dmesg failed: {exc}\n", encoding="utf-8")
        return -1
    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout or "unknown dmesg error").strip()
        out_path.write_text(f"dmesg failed: {error}\n", encoding="utf-8")
        return -1
    matches = [
        line for line in (proc.stdout or "").splitlines() if DMESG_RE.search(line)
    ]
    out_path.write_text(
        "\n".join(matches) + ("\n" if matches else ""), encoding="utf-8"
    )
    return len(matches)


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("inf")
    if df in table:
        return table[df]
    return 1.960


def _sample_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "mean": 0.0, "stddev": 0.0, "ci95": 0.0, "n": 0}
    n = len(values)
    median = statistics.median(values)
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if n > 1 else 0.0
    ci = (stddev / math.sqrt(n)) * _t_critical_95(n - 1) if n > 1 else 0.0
    return {"median": median, "mean": mean, "stddev": stddev, "ci95": ci, "n": n}


def _paired_percent(candidate: list[float], baseline: list[float]) -> dict[str, float]:
    if not candidate or not baseline or len(candidate) != len(baseline):
        return {"pct_median": 0.0, "pct_mean": 0.0, "pct_stddev": 0.0, "pct_ci95": 0.0, "n": 0}
    pairs = [(c / b - 1.0) * 100.0 for c, b in zip(candidate, baseline) if b > 0]
    if not pairs:
        return {"pct_median": 0.0, "pct_mean": 0.0, "pct_stddev": 0.0, "pct_ci95": 0.0, "n": 0}
    base = _sample_stats(pairs)
    return {
        "pct_median": base["median"],
        "pct_mean": base["mean"],
        "pct_stddev": base["stddev"],
        "pct_ci95": base["ci95"],
        "n": base["n"],
    }


def _parse_depths(arg: str | None) -> tuple[int, ...]:
    if arg is None:
        return DEFAULT_PRODUCT_DEPTHS
    if not arg.strip():
        raise ValueError("depth list cannot be empty")
    out: list[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError("depth list contains an empty entry")
        depth = int(tok)
        if depth < 0:
            raise ValueError(f"depth cannot be negative: {depth}")
        out.append(depth)
    return tuple(out)


def _parse_kv_types(arg: str | None) -> tuple[tuple[str, str], ...]:
    if arg is None:
        return DEFAULT_PRODUCT_KV_TYPES
    if not arg.strip():
        raise ValueError("KV type list cannot be empty")
    out: list[tuple[str, str]] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError("KV type list contains an empty entry")
        parts = tok.split("/")
        if len(parts) != 2:
            raise ValueError(f"KV type must be K/V, got {tok!r}")
        k, v = (part.strip() for part in parts)
        if not k or not v:
            raise ValueError(f"KV type names cannot be empty, got {tok!r}")
        out.append((k, v))
    return tuple(out)


def _parse_env_list(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--env expects NAME=VALUE, got {raw!r}")
        k, v = raw.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"--env has empty name in {raw!r}")
        if k in out:
            raise ValueError(f"--env has duplicate name {k!r}")
        out[k] = v
    return out


def _product_bench_argv(bin_dir: Path, model: str, kv: tuple[str, str], depth: int) -> list[str]:
    """Canonical per-sample llama-bench command for the product campaign."""
    argv = [
        str(bin_dir / "llama-bench"),
        "-m", model,
        "-ngl", "99",
        "-fa", "on",
        "-ctk", kv[0],
        "-ctv", kv[1],
        "-p", "512",
        "-n", "128",
        "-b", "512",
        "-ub", "512",
        "--no-warmup",
        "-r", "1",
        "-o", "json",
    ]
    if depth > 0:
        argv += ["-d", str(depth)]
    return argv


def _select_prompt_row(
    bench: list[dict[str, Any]], n_prompt: int, n_gen: int
) -> dict[str, Any] | None:
    """Select one exact llama-bench row and reject non-positive throughput."""
    for row in bench:
        try:
            row_prompt = int(row.get("n_prompt", 0) or 0)
            row_gen = int(row.get("n_gen", 0) or 0)
            avg_ts = float(row.get("avg_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if row_prompt == n_prompt and row_gen == n_gen and avg_ts > 0:
            return row
    return None


def _select_product_rows(
    bench: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Select the exact pp512 and tg128 rows from llama-bench JSON."""
    return {
        "pp512": _select_prompt_row(bench, n_prompt=512, n_gen=0),
        "tg128": _select_prompt_row(bench, n_prompt=0, n_gen=128),
    }


def _parse_fa_route_records(logs: str) -> list[dict[str, Any]]:
    """Parse structured one-time SYCL FlashAttention route records."""
    records: list[dict[str, Any]] = []
    for match in FA_ROUTE_RE.finditer(logs):
        groups = match.groupdict()
        records.append(
            {
                "route": groups["route"],
                "phase": groups["phase"],
                "q_tokens": int(groups["q_tokens"]),
                "head_dim": int(groups["head_dim"]),
                "gqa": int(groups["gqa"]),
                "type_k": groups["type_k"],
                "type_v": groups["type_v"],
                "k_quants_first": groups["k_quants_first"] == "1",
                "v_quants_first": groups["v_quants_first"] == "1",
            }
        )
    return records


def _parse_fa_profile_records(logs: str) -> list[dict[str, Any]]:
    """Parse aggregate SYCL q8 FlashAttention profile records."""
    records: list[dict[str, Any]] = []
    for match in FA_PROFILE_RE.finditer(logs):
        groups = match.groupdict()
        records.append(
            {
                "route": groups["route"],
                "layout": groups["layout"],
                "launches": int(groups["launches"]),
                "conversion_us": int(groups["conversion_us"]),
                "conversion_bytes": int(groups["conversion_bytes"]),
                "stage1_us": int(groups["stage1_us"]),
                "combine_us": int(groups["combine_us"]),
                "gqa": int(groups["gqa"]),
                "repeated_packed_kv_bytes": int(groups["repeated_packed_kv_bytes"]),
            }
        )
    return records


def _parse_graph_profile_records(logs: str) -> list[dict[str, int]]:
    """Parse aggregate SYCL graph lifecycle profile records."""
    return [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in GRAPH_PROFILE_RE.finditer(logs)
    ]


def _kv_bytes_per_element(kv_type: str) -> float:
    if kv_type == "f16":
        return 2.0
    if kv_type == "q8_0":
        return Q8_0_BLOCK_BYTES / QK8_0
    raise ValueError(f"effective KV bandwidth does not support {kv_type!r}")


def _effective_kv_read_bytes_per_step(
    kv: tuple[str, str], depth: int, layers: int, query_heads: int, head_dim: int
) -> float:
    """Requested K+V bytes read per generated token, without cache effects."""
    return (
        layers
        * query_heads
        * depth
        * head_dim
        * (_kv_bytes_per_element(kv[0]) + _kv_bytes_per_element(kv[1]))
    )


def _annotate_effective_kv_bandwidth(
    cell: dict[str, Any], model_shape: tuple[int, int, int] | None
) -> None:
    cell["effective_kv_read_bytes_per_step"] = None
    cell["effective_kv_gbps"] = {"baseline": None, "candidate": None}
    if model_shape is None:
        return
    layers, query_heads, head_dim = model_shape
    read_bytes = _effective_kv_read_bytes_per_step(
        tuple(cell["kv"]), cell["depth"], layers, query_heads, head_dim
    )
    tg128 = cell["metrics"]["tg128"]
    cell["effective_kv_read_bytes_per_step"] = read_bytes
    cell["effective_kv_gbps"] = {
        "baseline": read_bytes * tg128["baseline_stats"]["median"] / 1e9,
        "candidate": (
            read_bytes * tg128["candidate_stats"]["median"] / 1e9
            if tg128["candidate_stats"]["n"] > 0
            else None
        ),
    }


def _candidate_env_log_assertions(
    cells: list[dict[str, Any]], candidate_env: dict[str, str] | None
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Require requested candidate values when the backend logs that key."""
    if candidate_env is None:
        return {}, True
    all_samples: list[dict[str, Any]] = []
    candidate_samples: list[dict[str, Any]] = []
    for cell in cells:
        for arm_name, samples in cell["samples"].items():
            all_samples.extend(samples)
            if arm_name == "candidate":
                candidate_samples.extend(samples)
    assertions: dict[str, dict[str, Any]] = {}
    all_valid = True
    for key, value in candidate_env.items():
        key_re = re.compile(rf"(?m)^\s*{re.escape(key)}\s*:")
        value_re = re.compile(
            rf"(?m)^\s*{re.escape(key)}\s*:\s*{re.escape(value)}(?:\s|$)"
        )
        backend_logs_key = any(
            key_re.search(
                sample.get("stdout", "") + "\n" + sample.get("stderr", "")
            )
            for sample in all_samples
        )
        matched_samples = sum(
            bool(
                value_re.search(
                    sample.get("stdout", "") + "\n" + sample.get("stderr", "")
                )
            )
            for sample in candidate_samples
        )
        valid = not backend_logs_key or matched_samples == len(candidate_samples)
        assertions[key] = {
            "requested_value": value,
            "backend_logs_key": backend_logs_key,
            "candidate_samples": len(candidate_samples),
            "candidate_samples_with_requested_value": matched_samples,
            "valid": valid,
        }
        all_valid = all_valid and valid
    return assertions, all_valid


def _write_product_summary_md(summary: dict[str, Any], md_path: Path) -> None:
    lines = [
        f"# Product campaign: {summary['model_name']}",
        "",
        f"- bin-dir: {summary['bin_dir']}",
        f"- baseline label: {summary['baseline_label']}",
        f"- candidate label: {summary['candidate_label']}",
        f"- baseline env: {summary['baseline_env']}",
        f"- candidate env: {summary['candidate_env']}",
        f"- candidate_enabled: {summary['candidate_enabled']}",
        f"- model shape: {summary['model_shape']}",
        f"- candidate env log assertions: {summary['candidate_env_log_assertions']}",
        f"- dmesg fault hits before={summary['dmesg_before_hits']} after={summary['dmesg_after_hits']} new={len(summary['dmesg_new_matches'])}",
        "",
        "| depth | kv | metric | valid | baseline median tok/s | baseline mean | baseline stddev | baseline 95% CI | candidate median tok/s | candidate mean | candidate stddev | candidate 95% CI | paired median % | paired mean % | paired stddev | paired 95% CI | effective KV B/step | baseline effective GB/s | candidate effective GB/s | n |",
        "|---:|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        read_bytes = cell["effective_kv_read_bytes_per_step"]
        gbps = cell["effective_kv_gbps"]
        read_bytes_text = f"{read_bytes:.0f}" if read_bytes is not None else "n/a"
        baseline_gbps_text = (
            f"{gbps['baseline']:.3f}" if gbps["baseline"] is not None else "n/a"
        )
        candidate_gbps_text = (
            f"{gbps['candidate']:.3f}" if gbps["candidate"] is not None else "n/a"
        )
        for metric_name in ("pp512", "tg128"):
            metric = cell["metrics"][metric_name]
            b = metric["baseline_stats"]
            c = metric["candidate_stats"]
            p = metric["paired"]
            valid = "Y" if cell["valid"] else "N"
            lines.append(
                "| {d} | {kt}/{vt} | {metric} | {v} | {bm:.2f} | {bmean:.2f} | {bsd:.2f} | +/- {bc:.2f} | {cm:.2f} | {cmean:.2f} | {csd:.2f} | +/- {cc:.2f} | {pm:+.2f} | {pmean:+.2f} | {psd:.2f} | +/- {pc:.2f} | {read_bytes} | {bgbps} | {cgbps} | {n} |".format(
                    d=cell["depth"], kt=cell["kv"][0], vt=cell["kv"][1],
                    metric=metric_name, v=valid, bm=b["median"], bmean=b["mean"],
                    bsd=b["stddev"], bc=b["ci95"], cm=c["median"],
                    cmean=c["mean"], csd=c["stddev"], cc=c["ci95"],
                    pm=p["pct_median"], pmean=p["pct_mean"], psd=p["pct_stddev"],
                    pc=p["pct_ci95"], read_bytes=read_bytes_text,
                    bgbps=baseline_gbps_text, cgbps=candidate_gbps_text, n=p["n"],
                )
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _align_cell_samples_by_rep(
    cell_samples: dict[str, list[dict[str, Any]]],
    retained_reps: list[int],
    metric_name: str,
) -> tuple[dict[str, list[float]], list[str]]:
    """Pair one metric across arms by repetition index."""
    value_key = f"{metric_name}_ts"
    by_arm_rep: dict[str, dict[int, dict[str, Any]]] = {}
    for arm_name, samples in cell_samples.items():
        by_arm_rep[arm_name] = {entry["rep"]: entry for entry in samples if "rep" in entry}
    failures: list[str] = []
    aligned: dict[str, list[float]] = {arm: [] for arm in cell_samples}
    for rep in retained_reps:
        for arm_name in cell_samples:
            entry = by_arm_rep[arm_name].get(rep)
            if entry is None:
                failures.append(f"{metric_name} {arm_name} rep={rep} (no record)")
                continue
            ts = float(entry.get(value_key, 0.0))
            if not math.isfinite(ts) or ts <= 0:
                failures.append(f"{metric_name} {arm_name} rep={rep} (missing, non-finite, or non-positive avg_ts)")
            aligned[arm_name].append(ts)
    return aligned, failures


def run_product_cell(
    bin_dir: Path,
    model_path: str,
    kv: tuple[str, str],
    depth: int,
    baseline_env: dict[str, str],
    candidate_env: dict[str, str] | None,
    repetitions: int,
    timeout_s: int,
    samples_dir: Path,
    cell_idx: int,
    candidate_bin_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one paired product cell and retain both pp512 and tg128."""
    if repetitions < 3:
        raise ValueError(
            "repetitions must be >= 3 (sample 0 is always discarded, need at least 2 for CI)"
        )
    samples_dir.mkdir(parents=True, exist_ok=True)
    candidate_bin_dir = candidate_bin_dir or bin_dir
    has_candidate = candidate_env is not None
    arms: list[tuple[str, dict[str, str], Path]] = [
        ("baseline", baseline_env, bin_dir)
    ]
    if has_candidate:
        arms.append(("candidate", candidate_env, candidate_bin_dir))
    cell_samples: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _, _ in arms
    }
    for rep in range(repetitions):
        order = list(arms) if rep % 2 == 0 else list(reversed(arms))
        for arm_name, arm_env, arm_bin_dir in order:
            check_sole_tenancy()
            argv = _product_bench_argv(arm_bin_dir, model_path, kv, depth)
            label = (
                f"[cell {cell_idx} d={depth} kv={kv[0]}/{kv[1]} rep={rep}] "
                f"arm={arm_name}"
            )
            print(label, flush=True)
            result = run(argv, arm_env, timeout_s)
            rows = _select_product_rows(parse_bench(result.get("stdout", "")))
            route_records = _parse_fa_route_records(
                result.get("stdout", "") + "\n" + result.get("stderr", "")
            )
            profile_records = _parse_fa_profile_records(
                result.get("stdout", "") + "\n" + result.get("stderr", "")
            )
            graph_profile_records = _parse_graph_profile_records(
                result.get("stdout", "") + "\n" + result.get("stderr", "")
            )
            sample_values: dict[str, float] = {}
            for metric_name, row in rows.items():
                sample_values[f"{metric_name}_ts"] = (
                    float(row["avg_ts"])
                    if result["ok"] and row is not None and row.get("avg_ts")
                    else 0.0
                )
            sample_record = {
                "rep": rep,
                "ok": result["ok"],
                "returncode": result["returncode"],
                "pp512_ts": sample_values["pp512_ts"],
                "tg128_ts": sample_values["tg128_ts"],
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "fa_route_records": route_records,
                "fa_profile_records": profile_records,
                "graph_profile_records": graph_profile_records,
            }
            sample_path = samples_dir / (
                f"cell{cell_idx:02d}_{kv[0]}_{kv[1]}_d{depth}"
                f"_{arm_name}_rep{rep}.json"
            )
            sample_path.write_text(
                json.dumps(
                    {
                        "argv": argv,
                        "baseline_env": baseline_env,
                        "candidate_env": candidate_env,
                        "active_arm": arm_name,
                        "active_env": arm_env,
                        "active_bin_dir": str(arm_bin_dir),
                        "result_ok": result["ok"],
                        "returncode": result["returncode"],
                        "elapsed_s": result["elapsed_s"],
                        "selected_rows": rows,
                        "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "fa_route_records": route_records,
                        "fa_profile_records": profile_records,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            cell_samples[arm_name].append(sample_record)

    retained_reps = list(range(1, repetitions))
    expected = len(retained_reps)
    failures: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}
    cell_valid = True
    zero_stats = {"median": 0.0, "mean": 0.0, "stddev": 0.0, "ci95": 0.0, "n": 0}
    zero_paired = {
        "pct_median": 0.0,
        "pct_mean": 0.0,
        "pct_stddev": 0.0,
        "pct_ci95": 0.0,
        "n": 0,
    }
    for metric_name in ("pp512", "tg128"):
        aligned, metric_failures = _align_cell_samples_by_rep(
            cell_samples, retained_reps, metric_name
        )
        failures.extend(metric_failures)
        baseline = aligned["baseline"]
        candidate = aligned.get("candidate", [])
        metric_valid = len(baseline) == expected and all(ts > 0 for ts in baseline)
        if has_candidate:
            metric_valid = (
                metric_valid
                and len(candidate) == expected
                and all(ts > 0 for ts in candidate)
            )
        cell_valid = cell_valid and metric_valid
        metrics[metric_name] = {
            "retained_baseline_ts": baseline,
            "retained_candidate_ts": candidate if has_candidate else [],
            "baseline_stats": _sample_stats(baseline),
            "candidate_stats": _sample_stats(candidate) if has_candidate else dict(zero_stats),
            "paired": (
                _paired_percent(candidate, baseline)
                if has_candidate and metric_valid
                else dict(zero_paired)
            ),
        }

    if not cell_valid:
        for metric in metrics.values():
            metric["retained_baseline_ts"] = []
            metric["retained_candidate_ts"] = []
            metric["baseline_stats"] = dict(zero_stats)
            metric["candidate_stats"] = dict(zero_stats)
            metric["paired"] = dict(zero_paired)
    return {
        "cell": cell_idx,
        "depth": depth,
        "kv": list(kv),
        "samples": cell_samples,
        "metrics": metrics,
        "failures": failures,
        "valid": cell_valid,
    }


def run_product_campaign_main(ns: argparse.Namespace) -> int:
    bin_dir = Path(ns.bin_dir).resolve()
    candidate_bin_dir = Path(
        getattr(ns, "candidate_bin_dir", None) or ns.bin_dir
    ).resolve()
    bench_path = bin_dir / "llama-bench"
    candidate_bench_path = candidate_bin_dir / "llama-bench"
    if not bench_path.is_file() or not os.access(bench_path, os.X_OK):
        print(f"--bin-dir must contain an executable regular llama-bench: {bench_path}", file=sys.stderr, flush=True)
        return 2
    if not candidate_bench_path.is_file() or not os.access(candidate_bench_path, os.X_OK):
        print(
            "--candidate-bin-dir must contain an executable regular "
            f"llama-bench: {candidate_bench_path}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    model = Path(ns.model).resolve()
    if not model.is_file() or model.stat().st_size <= 0:
        print(f"--model must be a non-empty regular file: {model}", file=sys.stderr, flush=True)
        return 2

    try:
        baseline_env = _parse_env_list(getattr(ns, "baseline_env", []) or [])
        candidate_env = _parse_env_list(getattr(ns, "env", []) or []) or None
        depths = _parse_depths(ns.depths)
        kv_types = _parse_kv_types(ns.kv_types)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2

    shape_values = (getattr(ns, "model_layers", None), getattr(ns, "query_heads", None), getattr(ns, "head_dim", None))
    if any(value is not None for value in shape_values) and not all(value is not None for value in shape_values):
        print("--model-layers, --query-heads, and --head-dim must be all present or all absent", file=sys.stderr, flush=True)
        return 2
    model_shape: tuple[int, int, int] | None = None
    if all(value is not None for value in shape_values):
        model_shape = tuple(int(value) for value in shape_values)
        if any(value <= 0 for value in model_shape):
            print("model shape values must be positive", file=sys.stderr, flush=True)
            return 2
        if any("q8_0" in kv for kv in kv_types) and model_shape[2] % QK8_0 != 0:
            print(f"--head-dim must be divisible by QK8_0={QK8_0} for q8_0", file=sys.stderr, flush=True)
            return 2
    timeout_s = int(ns.timeout)
    repetitions = int(getattr(ns, "repetitions", DEFAULT_PRODUCT_REPETITIONS) or DEFAULT_PRODUCT_REPETITIONS)
    if repetitions < 3:
        print(
            "--repetitions must be >= 3 (sample 0 is always discarded, need at least 2 for CI)",
            file=sys.stderr,
            flush=True,
        )
        return 2


    out_dir = Path(ns.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    dmesg_before_path = out_dir / "dmesg.before.txt"
    dmesg_after_path = out_dir / "dmesg.after.txt"
    try:
        check_sole_tenancy()
    except SoleTenancyViolation as exc:
        sys.stderr.write(str(exc) + "\n")
        (out_dir / "sole-tenancy-violation.txt").write_text("\n".join(exc.holder_lines) + "\n", encoding="utf-8")
        return SOLE_TENANCY_EXIT

    dmesg_before_n = capture_dmesg(dmesg_before_path)
    if dmesg_before_n < 0:
        print("unable to capture pre-run dmesg; campaign is not gateable", file=sys.stderr)
        return 1
    provenance = collect_product_provenance(
        bin_dir, candidate_bin_dir, baseline_env, candidate_env
    )
    (out_dir / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    cells: list[dict[str, Any]] = []
    cell_idx = 0
    try:
        for depth in depths:
            for kv in kv_types:
                cell_idx += 1
                cell = run_product_cell(
                    bin_dir=bin_dir,
                    candidate_bin_dir=candidate_bin_dir,
                    model_path=str(model),
                    kv=kv,
                    depth=depth,
                    baseline_env=baseline_env,
                    candidate_env=candidate_env,
                    repetitions=repetitions,
                    timeout_s=timeout_s,
                    samples_dir=samples_dir,
                    cell_idx=cell_idx,
                )
                _annotate_effective_kv_bandwidth(cell, model_shape)
                cells.append(cell)
    except SoleTenancyViolation as exc:
        sys.stderr.write(str(exc) + "\n")
        (out_dir / "sole-tenancy-violation.txt").write_text("\n".join(exc.holder_lines) + "\n", encoding="utf-8")
        return SOLE_TENANCY_EXIT

    dmesg_after_n = capture_dmesg(dmesg_after_path)
    expected_pairs = repetitions - 1
    invalid_cell_ids: list[int] = []
    invalid_diagnostics: list[str] = []
    for cell in cells:
        cell_diag = list(cell["failures"])
        for metric_name, metric in cell["metrics"].items():
            n_base = len(metric["retained_baseline_ts"])
            if n_base != expected_pairs:
                cell_diag.append(f"{metric_name} baseline has {n_base} retained samples (expected {expected_pairs})")
            if candidate_env is not None:
                n_candidate = len(metric["retained_candidate_ts"])
                if n_candidate != expected_pairs:
                    cell_diag.append(f"{metric_name} candidate has {n_candidate} retained samples (expected {expected_pairs})")
                n_pairs = metric["paired"]["n"]
                if n_pairs != expected_pairs:
                    cell_diag.append(f"{metric_name} paired has {n_pairs} pairs (expected {expected_pairs})")
        if cell_diag or not cell["valid"]:
            cell["valid"] = False
            invalid_cell_ids.append(cell["cell"])
            tag = f"cell {cell['cell']} d={cell['depth']} kv={cell['kv']}"
            for diagnostic in cell_diag or ["marked invalid by per-cell gate"]:
                invalid_diagnostics.append(f"{tag}: {diagnostic}")

    env_log_assertions, env_logs_valid = _candidate_env_log_assertions(cells, candidate_env)
    if not env_logs_valid:
        for key, assertion in env_log_assertions.items():
            if not assertion["valid"]:
                invalid_diagnostics.append(f"candidate env log mismatch for {key}: {assertion}")

    dmesg_new_matches: list[str] = []
    if dmesg_after_n < 0:
        invalid_diagnostics.append("unable to capture post-run dmesg")
    else:
        unmatched_before = dmesg_before_path.read_text(encoding="utf-8").splitlines()
        for line in dmesg_after_path.read_text(encoding="utf-8").splitlines():
            if line in unmatched_before:
                unmatched_before.remove(line)
            else:
                dmesg_new_matches.append(line)
        if dmesg_new_matches:
            invalid_diagnostics.append(f"{len(dmesg_new_matches)} new i915/xe fault line(s) after campaign")

    all_valid = not invalid_cell_ids and env_logs_valid and not dmesg_new_matches and dmesg_after_n >= 0
    summary = {
        "model_name": model.name, "model_path": str(model), "bin_dir": str(bin_dir),
        "candidate_bin_dir": str(candidate_bin_dir),
        "baseline_label": ns.baseline_label, "candidate_label": ns.candidate_label,
        "baseline_env": baseline_env, "candidate_env": candidate_env or {},
        "candidate_enabled": candidate_env is not None,
        "candidate_env_log_assertions": env_log_assertions,
        "model_shape": ({"model_layers": model_shape[0], "query_heads": model_shape[1], "head_dim": model_shape[2]} if model_shape is not None else None),
        "provenance": provenance,
        "dmesg_before_hits": dmesg_before_n, "dmesg_after_hits": dmesg_after_n,
        "dmesg_before_path": str(dmesg_before_path), "dmesg_after_path": str(dmesg_after_path),
        "dmesg_new_matches": dmesg_new_matches,
        "depths": list(depths), "kv_types": [list(kv) for kv in kv_types],
        "repetitions": repetitions, "expected_retained_per_arm": expected_pairs,
        "all_cells_valid": all_valid, "invalid_cell_ids": invalid_cell_ids,
        "invalid_cell_count": len(invalid_cell_ids), "invalid_diagnostics": invalid_diagnostics,
        "cells": cells,
    }
    json_path = out_dir / "product.json"
    json_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_product_summary_md(summary, out_dir / "product.md")
    print(f"wrote {json_path}")
    print(f"wrote {out_dir / 'product.md'}")
    if not all_valid:
        sys.stderr.write(f"product campaign is not gateable; {len(invalid_diagnostics)} diagnostic(s), invalid cells={invalid_cell_ids}\n")
        for line in invalid_diagnostics:
            sys.stderr.write(f"  {line}\n")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork-bin", default="build-port/bin")
    ap.add_argument("--upstream-bin", default=os.environ.get("UPSTREAM_BIN", "/mnt/mrgr/llama-cpp-sycl-turbo/compare/llama.cpp/build-sycl-a770/bin"))
    ap.add_argument("--out-dir", default="bench-a770-fork-unique")
    ap.add_argument("--quick", action="store_true", help="Use p64/n16/r1 for all models")
    ap.add_argument("--models", nargs="*", choices=[m[0] for m in DEFAULT_MODELS], help="Subset of models")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--campaign", choices=["product"], default=None,
        help="Use a non-default harness (currently: product = sole-tenancy product/depth sweep).",
    )
    ap.add_argument(
        "--candidate-bin-dir",
        help="[product] directory containing the candidate llama-bench; "
        "defaults to --bin-dir for environment-only comparisons",
    )
    # Product-campaign subcommand flags. Ignored by the legacy matrix path.
    ap.add_argument("--bin-dir", help="[product] directory containing llama-bench (and llama-server for spec-decode legs).")
    ap.add_argument("--model", help="[product] path to a GGUF model.")
    ap.add_argument("--depths", help="[product] comma-separated depths (default: 0,2048,4096,8192,16384).")
    ap.add_argument("--kv-types", help="[product] comma-separated K/V pairs (default: f16/f16,q8_0/q8_0).")
    ap.add_argument("--model-layers", type=int,
                    help="[product] model layer count for requested-KV bandwidth.")
    ap.add_argument("--query-heads", type=int,
                    help="[product] query head count for requested-KV bandwidth.")
    ap.add_argument("--head-dim", type=int,
                    help="[product] attention head dimension for requested-KV bandwidth.")
    ap.add_argument(
        "--baseline-env", action="append", default=[],
        help="[product] repeatable NAME=VALUE env applied only to the baseline arm.",
    )
    ap.add_argument(
        "--env", action="append", default=[],
        help="[product] repeatable NAME=VALUE env applied to the candidate arm. Omit to run the canonical baseline alone (six launches per cell).",
    )
    ap.add_argument("--baseline-label", default="stock",
                    help="[product] literal label for the baseline arm.")
    ap.add_argument("--candidate-label", default="candidate",
                    help="[product] literal label for the candidate arm.")
    ap.add_argument("--repetitions", type=int, default=DEFAULT_PRODUCT_REPETITIONS,
                    help="[product] samples per arm per cell (default 6; sample 0 is discarded).")
    ns = ap.parse_args()

    if ns.campaign == "product":
        if not ns.bin_dir or not ns.model or not ns.out_dir:
            ap.error("--campaign product requires --bin-dir, --model, --out-dir")
        if ns.candidate_bin_dir and not ns.env:
            ap.error("--candidate-bin-dir requires --env to enable the candidate arm")
        return run_product_campaign_main(ns)

    # Legacy matrix path: unchanged from the original runner.
    fork_bin = Path(ns.fork_bin).resolve()
    upstream_bin = Path(ns.upstream_bin).resolve()
    out_dir = Path(ns.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    md_path = out_dir / "summary.md"

    selected = set(ns.models or [m[0] for m in DEFAULT_MODELS])
    models = [(name, str((Path(MODELS_ROOT) / rel).resolve())) for name, rel in DEFAULT_MODELS if name in selected]

    cases: list[dict[str, Any]] = []
    for name, model in models:
        p, n, reps = (64, 16, 1) if ns.quick else (512, 64, 2)
        base = {"model_name": name, "model_path": model, "p": p, "n": n, "reps": reps}
        for kv in [("f16", "f16"), ("q8_0", "q8_0")]:
            cases.append({**base, "repo": "upstream", "label": f"upstream-{kv[0]}-{kv[1]}", "bin": upstream_bin, "kv": kv, "fa": "on", "env": {}})
            cases.append({**base, "repo": "fork", "label": f"fork-{kv[0]}-{kv[1]}", "bin": fork_bin, "kv": kv, "fa": "on", "env": {}})
        for kv in [("f16", "f16"), ("q8_0", "q8_0"), ("turbo3", "turbo3")]:
            cases.append({**base, "repo": "fork", "label": f"fork-xmx-default-{kv[0]}-{kv[1]}", "bin": fork_bin, "kv": kv, "fa": "on", "env": {"GGML_SYCL_FA_XMX": "1"}})
        for kv in [("turbo2", "turbo2"), ("turbo3", "turbo3"), ("turbo4", "turbo4"), ("q8_0", "turbo3")]:
            pure_env = {"TURBO_LAYER_ADAPTIVE": "0", "TURBO_AUTO_ASYMMETRIC": "0"}
            cases.append({**base, "repo": "fork", "label": f"fork-default-{kv[0]}-{kv[1]}", "bin": fork_bin, "kv": kv, "fa": "on", "env": {}})
            cases.append({**base, "repo": "fork", "label": f"fork-pure-{kv[0]}-{kv[1]}", "bin": fork_bin, "kv": kv, "fa": "on", "env": pure_env})
            cases.append({**base, "repo": "fork", "label": f"fork-xmx-pure-{kv[0]}-{kv[1]}", "bin": fork_bin, "kv": kv, "fa": "on", "env": {"GGML_SYCL_FA_XMX": "1", **pure_env}})
        cases.append({**base, "repo": "fork", "label": "fork-nonfa-turbo3-turbo3", "bin": fork_bin, "kv": ("turbo3", "turbo3"), "fa": "off", "env": {"TURBO_LAYER_ADAPTIVE": "0", "TURBO_AUTO_ASYMMETRIC": "0"}})

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, case in enumerate(cases, 1):
            argv = bench_case(case["bin"], case["model_path"], case["kv"], case["fa"], case["p"], case["n"], case["reps"])
            print(f"[{i}/{len(cases)}] {case['model_name']} {case['label']}", flush=True)
            result = run(argv, case["env"], ns.timeout)
            record = {k: v for k, v in case.items() if k != "bin"}
            record["bin"] = str(case["bin"])
            record["argv"] = argv
            record["result"] = {k: v for k, v in result.items() if k not in {"stdout", "stderr"}}
            record["bench"] = parse_bench(result.get("stdout", ""))
            record["stderr_tail"] = result.get("stderr", "")[-4000:]
            record["stdout_tail"] = result.get("stdout", "")[-4000:] if not record["bench"] else ""
            jf.write(json.dumps(record, sort_keys=True) + "\n")
            jf.flush()

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = ["# Arc A770 fork-unique benchmark summary", "", f"JSONL: `{os.path.relpath(jsonl_path)}`", "", "| model | case | status | pp tok/s | tg tok/s |", "|---|---|---:|---:|---:|"]
    for rec in records:
        pp = tg = ""
        for row in rec.get("bench", []):
            if row.get("n_prompt", 0):
                pp = f"{row.get('avg_ts', 0):.2f}"
            if row.get("n_gen", 0):
                tg = f"{row.get('avg_ts', 0):.2f}"
        status = "ok" if rec["result"].get("ok") else f"fail({rec['result'].get('returncode')})"
        lines.append(f"| {rec['model_name']} | {rec['label']} | {status} | {pp} | {tg} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {jsonl_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
