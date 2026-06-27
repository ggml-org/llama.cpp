#!/usr/bin/env python3
"""A770 SYCL spec-decode + KV-type benchmark harness for llama-cpp-turboquant.

Launches ``llama-server`` once per config arm, runs a fixed prompt suite
(``prompts.jsonl``) through the OpenAI-compatible ``/v1/chat/completions``
endpoint, and records server-reported throughput and draft acceptance.

Why a server + HTTP harness and not ``llama-bench``: ``llama-bench``'s
``test_gen``/``test_prompt`` call ``llama_decode()`` with no speculative
context, so it cannot measure speculative decoding at all. Acceptance and tg
are read straight from the server response ``timings`` object
(``predicted_per_second``, ``prompt_per_second``, ``draft_n``,
``draft_n_accepted``) -- authoritative, no log scraping required.

stdlib only (no third-party deps). Mirrors (does not import) the HTTP/JSONL
pattern of ``Luce-Org-lucebox-hub/harness/benchmarks/generation_benchmark.py``
and the spec-timings handling of ``tools/server/bench/speed-bench/speed_bench.py``.

Config via environment:
  MODEL          gguf path (default: on-disk Llama-3.1-8B-heretic Q4_K_M)
  SERVER_BIN     llama-server binary (default: main-checkout build)
  PORT           server port (default 8771; prod is 8767)
  CTX            context size (default 16384)
  THREADS        CPU threads (default 12)
  REPEATS        measured runs per prompt, median reported (default 2)
  MODE           'baseline' (6-arm sweep) | 'deadoff' (R2 A/B) (default baseline)
  KV             KV cache type for MODE=deadoff (default q8_0)
  SETVARS        oneAPI setvars.sh (default /opt/intel/oneapi/setvars.sh)
  HEALTH_TIMEOUT seconds to wait for /health (default 180)
  REQ_TIMEOUT    per-request HTTP timeout seconds (default 300)
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
PROMPTS = Path(os.environ.get("PROMPTS", str(HERE / "prompts.jsonl")))

MODEL = os.environ.get(
    "MODEL",
    "/home/svnbjrn/models/llama31-8b-heretic/Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf",
)
SERVER_BIN = os.environ.get(
    "SERVER_BIN",
    "/home/svnbjrn/projects/trb/llama-cpp-turboquant/build/bin/llama-server",
)
PORT = int(os.environ.get("PORT", "8771"))
CTX = int(os.environ.get("CTX", "16384"))
THREADS = int(os.environ.get("THREADS", "12"))
REPEATS = int(os.environ.get("REPEATS", "2"))
MODE = os.environ.get("MODE", "baseline")
KV = os.environ.get("KV", "q8_0")
SETVARS = os.environ.get("SETVARS", "/opt/intel/oneapi/setvars.sh")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "180"))
REQ_TIMEOUT = float(os.environ.get("REQ_TIMEOUT", "300"))

BASE = f"http://127.0.0.1:{PORT}"
NGRAM_MOD_PARAMS = [
    "--spec-ngram-mod-n-match", os.environ.get("NMATCH", "24"),
    "--spec-ngram-mod-n-min", os.environ.get("NMIN", "48"),
    "--spec-ngram-mod-n-max", os.environ.get("NMAX", "64"),
]


def build_arms() -> list[dict[str, Any]]:
    """Return the config matrix for the active MODE.

    Each arm: {name, kv, spec_label, extra:[server flags]}.
    """
    if MODE == "deadoff":
        common = ["--spec-type", "ngram-mod", *NGRAM_MOD_PARAMS]
        return [
            {"name": f"deadoff0-{KV}", "kv": KV, "spec_label": "ngram-mod (dead-off 0)",
             "extra": [*common, "--spec-ngram-mod-dead-off", "0"]},
            {"name": f"deadoff3-{KV}", "kv": KV, "spec_label": "ngram-mod (dead-off 3)",
             "extra": [*common, "--spec-ngram-mod-dead-off", "3"]},
        ]
    # baseline: {none, ngram-mod, ngram-mod+ngram-map-k4v} x {q8_0, f16}
    spec_variants = [
        ("none", "none", ["--spec-type", "none"]),
        ("ngrammod", "ngram-mod", ["--spec-type", "ngram-mod", *NGRAM_MOD_PARAMS]),
        ("ngrammod+mapk4v", "ngram-mod,ngram-map-k4v",
         ["--spec-type", "ngram-mod,ngram-map-k4v", *NGRAM_MOD_PARAMS]),
    ]
    arms: list[dict[str, Any]] = []
    for kv in ("q8_0", "f16"):
        for short, label, flags in spec_variants:
            arms.append({"name": f"{short}-{kv}", "kv": kv, "spec_label": label, "extra": flags})
    return arms


def load_prompts() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with PROMPTS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if not prompts:
        raise SystemExit(f"no prompts in {PROMPTS}")
    return prompts


def server_command(arm: dict[str, Any]) -> str:
    args = [
        SERVER_BIN,
        "-m", MODEL,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--ctx-size", str(CTX),
        "--device", "SYCL0",
        "--n-gpu-layers", "999",
        "--no-mmap",
        "--flash-attn", "on",
        "--parallel", "1",
        "--threads", str(THREADS),
        "--cache-type-k", arm["kv"],
        "--cache-type-v", arm["kv"],
        *arm["extra"],
    ]
    quoted = " ".join(shlex.quote(a) for a in args)
    return f"source {shlex.quote(SETVARS)} >/dev/null 2>&1 && exec {quoted}"


def wait_health(timeout: float) -> bool:
    deadline = time.time() + timeout
    url = f"{BASE}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    return False


def post_chat(messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "cache_prompt": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    return content if isinstance(content, str) else ""


def run_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    n_predict = int(prompt.get("n_predict", 256))
    messages = prompt.get("messages") or [{"role": "user", "content": prompt.get("prompt", "")}]
    runs: list[dict[str, Any]] = []
    last_text = ""
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        resp = post_chat(messages, n_predict)
        elapsed = time.perf_counter() - t0
        timings = resp.get("timings") or {}
        usage = resp.get("usage") or {}
        last_text = extract_text(resp)
        completion_tokens = usage.get("completion_tokens")
        tg = timings.get("predicted_per_second")
        if tg is None and isinstance(completion_tokens, int) and elapsed > 0:
            tg = completion_tokens / elapsed  # wall-clock fallback
        draft_n = timings.get("draft_n")
        draft_acc = timings.get("draft_n_accepted")
        accept_rate = (draft_acc / draft_n) if (draft_n and draft_acc is not None) else None
        runs.append({
            "tg": tg,
            "pp": timings.get("prompt_per_second"),
            "elapsed_s": elapsed,
            "completion_tokens": completion_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "draft_n": draft_n,
            "draft_n_accepted": draft_acc,
            "accept_rate": accept_rate,
        })
    tgs = [r["tg"] for r in runs if isinstance(r["tg"], (int, float))]
    pps = [r["pp"] for r in runs if isinstance(r["pp"], (int, float))]
    accs = [r["accept_rate"] for r in runs if isinstance(r["accept_rate"], (int, float))]
    return {
        "id": prompt["id"],
        "n_predict": n_predict,
        "runs": runs,
        "tg_median": statistics.median(tgs) if tgs else None,
        "pp_median": statistics.median(pps) if pps else None,
        "accept_rate_median": statistics.median(accs) if accs else None,
        "draft_reported": any(r["draft_n"] is not None for r in runs),
        "completion_tokens": runs[-1]["completion_tokens"],
        "text_preview": last_text[:160].replace("\n", " "),
    }


def scan_log(logpath: Path) -> dict[str, Any]:
    """Best-effort: pull the flash-attn init line and any draft-acceptance lines."""
    fa_lines: list[str] = []
    acc_lines: list[str] = []
    try:
        text = logpath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"fa_lines": [], "acceptance_lines": []}
    for line in text.splitlines():
        low = line.lower()
        if ("flash" in low or "fattn" in low or "flash_attn" in low) and "warn" not in low:
            fa_lines.append(line.strip())
        if "draft acceptance" in low or "statistics" in low:
            acc_lines.append(line.strip())
    # de-dup, cap
    fa_lines = list(dict.fromkeys(fa_lines))[:8]
    acc_lines = acc_lines[-12:]
    return {"fa_lines": fa_lines, "acceptance_lines": acc_lines}


def start_server(arm: dict[str, Any], logpath: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["ZES_ENABLE_SYSMAN"] = "1"
    env["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
    with logpath.open("w", encoding="utf-8") as logf:
        return subprocess.Popen(
            ["bash", "-c", server_command(arm)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        # SIGINT triggers clean shutdown + spec stats; fall back to TERM/KILL.
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def run_arm(arm: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    logpath = RESULTS / f"{arm['name']}.log"
    print(f"\n=== arm {arm['name']}  (kv={arm['kv']}, spec={arm['spec_label']}) ===", flush=True)
    proc = start_server(arm, logpath)
    try:
        if not wait_health(HEALTH_TIMEOUT):
            print(f"  !! server failed health within {HEALTH_TIMEOUT}s (see {logpath})", flush=True)
            return {"arm": arm, "error": "health_timeout", "prompts": []}
        # SYCL JIT warmup: run the first prompt once (discarded) so kernel
        # compilation (incl. batched-verify when spec fires) is not charged to
        # the first measured generation.
        print("  warmup...", flush=True)
        try:
            wp = prompts[0]
            post_chat(wp.get("messages") or [{"role": "user", "content": wp.get("prompt", "")}],
                      int(wp.get("n_predict", 256)))
        except Exception as e:  # noqa: BLE001 - warmup failures are non-fatal
            print(f"  warmup error (continuing): {e}", flush=True)
        results = []
        for p in prompts:
            r = run_prompt(p)
            tg = r["tg_median"]
            acc = r["accept_rate_median"]
            print(f"  {r['id']:<12} tg={tg:.2f} t/s" if tg is not None else f"  {r['id']:<12} tg=n/a",
                  (f"  accept={acc:.3f}" if acc is not None else "  accept=n/a"),
                  f"  ctok={r['completion_tokens']}", flush=True)
            results.append(r)
        log_scan = scan_log(logpath)
        spec_missing = arm["spec_label"] != "none" and not any(r.get("draft_reported") for r in results)
        if spec_missing:
            print(f"  !! WARNING: arm '{arm['name']}' expects spec ({arm['spec_label']}) "
                  f"but server reported no draft stats - spec may be disabled", flush=True)
        return {"arm": arm, "error": None, "prompts": results, "log_scan": log_scan,
                "spec_stats_missing": spec_missing}
    finally:
        stop_server(proc)
        time.sleep(2.0)  # let the GPU/Level-Zero context fully release before next arm


def md_table(summary: list[dict[str, Any]], prompt_ids: list[str], field: str, fmt: str) -> str:
    arm_names = [a["arm"]["name"] for a in summary]
    header = "| prompt | " + " | ".join(arm_names) + " |"
    sep = "|" + "---|" * (len(arm_names) + 1)
    lines = [header, sep]
    for pid in prompt_ids:
        cells = []
        for a in summary:
            val = None
            if not a.get("error"):
                for pr in a["prompts"]:
                    if pr["id"] == pid:
                        val = pr.get(field)
                        break
            cells.append(format(val, fmt) if isinstance(val, (int, float)) else "n/a")
        lines.append(f"| {pid} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    prompts = load_prompts()
    prompt_ids = [p["id"] for p in prompts]
    arms = build_arms()
    only = os.environ.get("ONLY", "").strip()
    if only:
        arms = [a for a in arms if only in a["name"]]
    print(f"MODE={MODE}  MODEL={MODEL}")
    print(f"SERVER_BIN={SERVER_BIN}")
    print(f"PORT={PORT} CTX={CTX} THREADS={THREADS} REPEATS={REPEATS}")
    print(f"arms: {[a['name'] for a in arms]}")

    out_path = RESULTS / ("summary.json" if MODE == "baseline" else f"summary_{MODE}.json")
    summary: list[dict[str, Any]] = []
    for arm in arms:
        summary.append(run_arm(arm, prompts))
        # write incrementally so a mid-sweep failure does not lose completed arms
        out = {
            "mode": MODE,
            "model": MODEL,
            "server_bin": SERVER_BIN,
            "ctx": CTX,
            "threads": THREADS,
            "repeats": REPEATS,
            "prompt_ids": prompt_ids,
            "arms": summary,
        }
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n\n## Throughput (tg, tokens/s, median of repeats)\n")
    print(md_table(summary, prompt_ids, "tg_median", ".2f"))
    print("\n## Draft acceptance rate (median)\n")
    print(md_table(summary, prompt_ids, "accept_rate_median", ".3f"))
    print("\n## Prompt throughput (pp, tokens/s, median)\n")
    print(md_table(summary, prompt_ids, "pp_median", ".1f"))
    print(f"\nsummary -> {out_path}")
    for a in summary:
        if a.get("error"):
            print(f"  ARM ERROR: {a['arm']['name']}: {a['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
