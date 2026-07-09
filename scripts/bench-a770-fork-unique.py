#!/usr/bin/env python3
"""Benchmark fork-only Arc A770 surfaces against upstream controls.

This runner is intentionally explicit about environment-controlled modes:
- default: lets TURBO_LAYER_ADAPTIVE / TURBO_AUTO_ASYMMETRIC apply.
- pure: disables both auto policies so requested turbo K/V is what runs.
- xmx: enables GGML_SYCL_FA_XMX to route eligible FA cases to the XMX path.

It writes JSONL records for each subprocess plus a compact Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/mnt/mrgr")

# Model paths are relative to MODELS_ROOT (default /mnt/mrgr); override the env var to relocate.
DEFAULT_MODELS = [
    ("llama31-8b-heretic", "models/llama31-8b-heretic/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf"),
    ("mistral-7b", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
    ("qwen3-coder-30b-a3b", "gguf/Qwen3-Coder-30B-A3B-UD-Q3_K_XL/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf"),
]


def run(argv: list[str], env_extra: dict[str, str], timeout_s: int, cwd: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
    # Clear fork-controlled knobs so an ambient shell export can't leak into a
    # case that declares defaults; env_extra is the only per-case source.
    for _knob in ("GGML_SYCL_FA_XMX", "TURBO_LAYER_ADAPTIVE", "TURBO_AUTO_ASYMMETRIC"):
        env.pop(_knob, None)
    env.update(env_extra)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork-bin", default="build-port/bin")
    ap.add_argument("--upstream-bin", default=os.environ.get("UPSTREAM_BIN", "/mnt/mrgr/llama-cpp-sycl-turbo/compare/llama.cpp/build-sycl-a770/bin"))
    ap.add_argument("--out-dir", default="bench-a770-fork-unique")
    ap.add_argument("--quick", action="store_true", help="Use p64/n16/r1 for all models")
    ap.add_argument("--models", nargs="*", choices=[m[0] for m in DEFAULT_MODELS], help="Subset of models")
    ap.add_argument("--timeout", type=int, default=900)
    ns = ap.parse_args()

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
