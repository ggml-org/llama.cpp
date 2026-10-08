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
  MODE           'baseline' (6-arm sweep) | 'deadoff' | 'stress' | 'ab' | 'acceptance-curve' (default baseline)
  KV             KV cache type for MODE=deadoff/stress (default q8_0)
  SETVARS        oneAPI setvars.sh (default /opt/intel/oneapi/setvars.sh);
                 empty = inherit the caller's oneAPI environment
  HEALTH_TIMEOUT seconds to wait for /health (default 180)
  REQ_TIMEOUT    per-request HTTP timeout seconds (default 300)
  PLACEMENT      'full' (--device SYCL0 -ngl 999 --no-mmap, default) | 'fit'
                 (--fit on --fit-target FIT_TARGET, for models larger than VRAM)
  DRAFT_MODEL    optional --spec-draft-model gguf
  SERVER_EXTRA   extra server flags appended to every arm (shell-split)

MODE=ab is a paired A/B of two server builds under one speculative config:
  SERVER_BIN_A / SERVER_BIN_B   the two llama-server binaries; each arm loads
                                the shared libraries next to its own binary
  NAME_A / NAME_B               arm labels (default a / b)
  SPEC_ARGS      speculative flags for both arms (default: ngram-mod)
  LAUNCHES       server launches per arm (default 4), run in ABBA order so
                 drift cancels; it must be even and positive. The first request
                 of every launch is discarded
  OUT_TAG        suffix for the summary and log file names
It refuses to start (exit 70) while another process holds the render node, and
(exit 2) unless xe or i915 serves the render node. It
reports new i915/xe fault lines from dmesg, and prints paired 95% CIs per prompt.
It exits non-zero when a launch failed (a failed or unverified warmup fails it
too), a response failed the target-argmax verifier, a speculative arm reported
no draft statistics, a fault line appeared, or the kernel log could not be
compared (unreadable, wrapped or cleared during the run). The verifier needs
the target's argmax on every response row, so it needs a server that returns
the probabilities of accepted draft tokens (the port of upstream llama.cpp PR
27196); against an older build every speculative launch fails verification.
Both arms run with LLAMA_TRACE=1 so the log shows how many draft rounds were
verified and how many restored a speculative checkpoint.
Worked run and how to read the output:
docs/research/speculative/sycl-a770-spec-checkpoint-on-device-ab-2026-10-04.md
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
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

PLACEMENT = os.environ.get("PLACEMENT", "full")
FIT_TARGET = os.environ.get("FIT_TARGET", "1024")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "")
SPEC_ARGS = shlex.split(os.environ.get("SPEC_ARGS", ""))
SERVER_EXTRA = shlex.split(os.environ.get("SERVER_EXTRA", ""))
LAUNCHES = int(os.environ.get("LAUNCHES", "4"))
OUT_TAG = os.environ.get("OUT_TAG", "")
RENDER_NODE = os.environ.get("RENDER_NODE", "/dev/dri/renderD128")
EXIT_GPU_BUSY = 70
EXIT_USAGE = 2
# A kernel log line is an Arc fault when it names one of the two drivers and a failure term, in either
# order: i915 prints its fence timeout as "Fence expiration time out i915-<bdf>:...". "hang" also
# covers i915's "GPU HANG"; "timed?[ _-]?out" covers timeout, xe's "Timedout job", "timed out",
# "timed-out" and "time out"; "fault" counts only as a word of its own or after "page" (xe's "Fault
# response: Unsuccessful", "PageFault", "GTT fault"), so "default" does not match; "banned" is i915's
# verdict on a guilty context.
GPU_DRIVER_RE = re.compile(r"\b(?:i915|xe)\b", re.IGNORECASE)
GPU_FAULT_TERM_RE = re.compile(r"reset|hang|hung|timed?[ _-]?out|GuC|wedged|banned|CAT error|"
                               r"\b(?:page.?)?fault|device.?lost", re.IGNORECASE)
# two-sided 95% Student t quantiles, index = degrees of freedom (capped at 15)
T95 = [float("nan"), 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
       2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131]

BASE = f"http://127.0.0.1:{PORT}"
NGRAM_MOD_PARAMS = [
    "--spec-ngram-mod-n-match", os.environ.get("NMATCH", "24"),
    "--spec-ngram-mod-n-min", os.environ.get("NMIN", "48"),
    "--spec-ngram-mod-n-max", os.environ.get("NMAX", "64"),
]


def build_arms() -> list[dict[str, Any]]:
    """Return the config matrix for the active MODE.

    Each arm: {name, kv, spec_label, extra:[server flags]}; MODE=ab arms also
    carry their own server_bin.
    """
    if MODE == "ab":
        spec = SPEC_ARGS or ["--spec-type", "ngram-mod", *NGRAM_MOD_PARAMS]
        label = spec[spec.index("--spec-type") + 1] if "--spec-type" in spec[:-1] else "custom"
        return [
            {"name": os.environ.get(f"NAME_{key}", key.lower()), "kv": KV, "spec_label": label,
             "extra": spec, "server_bin": os.environ[f"SERVER_BIN_{key}"]}
            for key in ("A", "B")
        ]
    if MODE in {"deadoff", "stress"}:
        common = ["--spec-type", "ngram-mod", *NGRAM_MOD_PARAMS]
        arms = [
            {"name": f"deadoff0-{KV}", "kv": KV, "spec_label": "ngram-mod (dead-off 0)",
             "extra": [*common, "--spec-ngram-mod-dead-off", "0"]},
            {"name": f"deadoff3-{KV}", "kv": KV, "spec_label": "ngram-mod (dead-off 3)",
             "extra": [*common, "--spec-ngram-mod-dead-off", "3"]},
        ]
        if MODE == "stress":
            arms.insert(0, {
                "name": f"none-{KV}",
                "kv": KV,
                "spec_label": "none",
                "extra": ["--spec-type", "none"],
            })
        return arms
    if MODE == "acceptance-curve":
        # Request-total traces, not individual controller verification rounds.
        mtp_base = ["--spec-type", "draft-mtp"]
        return [
            {"name": "adaptive-3-7", "kv": KV, "spec_label": "draft-mtp adaptive 3-7",
             "extra": ["--spec-type", "draft-mtp-adaptive",
                       "--spec-draft-n-min-adaptive", "3",
                       "--spec-draft-n-max", "7"]},
            {"name": "fixed-3", "kv": KV, "spec_label": "draft-mtp fixed 3",
             "extra": [*mtp_base,
                       "--spec-draft-n-min", "3",
                       "--spec-draft-n-max", "3"]},
            {"name": "fixed-7", "kv": KV, "spec_label": "draft-mtp fixed 7",
             "extra": [*mtp_base,
                       "--spec-draft-n-min", "7",
                       "--spec-draft-n-max", "7"]},
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
    if PLACEMENT == "fit":
        placement = ["--fit", "on", "--fit-target", FIT_TARGET]
    else:
        placement = ["--device", "SYCL0", "--n-gpu-layers", "999", "--no-mmap"]
    args = [
        arm.get("server_bin", SERVER_BIN),
        "-m", MODEL,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--ctx-size", str(CTX),
        *placement,
        "--flash-attn", "on",
        "--parallel", "1",
        "--threads", str(THREADS),
        "--cache-type-k", arm["kv"],
        "--cache-type-v", arm["kv"],
        *arm["extra"],
        *(["--spec-draft-model", DRAFT_MODEL] if DRAFT_MODEL else []),
        *SERVER_EXTRA,
    ]
    quoted = " ".join(shlex.quote(a) for a in args)
    prefix = f"source {shlex.quote(SETVARS)} >/dev/null && " if SETVARS else ""
    return f"{prefix}exec {quoted}"


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


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def apply_chat_template(messages: list[dict[str, str]]) -> str:
    response = post_json("/apply-template", {"messages": messages})
    prompt = response.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("/apply-template did not return a prompt string")
    return prompt


def post_completion(prompt: str, n_predict: int) -> dict[str, Any]:
    return post_json("/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 123,
        "stream": False,
        "cache_prompt": False,
        "return_tokens": True,
        "n_probs": 1,
        "post_sampling_probs": False,
    })


def analyze_native_response(resp: dict[str, Any]) -> dict[str, Any]:
    tokens = resp.get("tokens")
    probs = resp.get("completion_probabilities")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ValueError("/completion did not return integer token IDs")
    if not isinstance(probs, list) or len(probs) != len(tokens):
        raise ValueError("/completion probability rows do not match returned tokens")

    failures: list[dict[str, Any]] = []
    rows_with_argmax = 0
    for index, (token, row) in enumerate(zip(tokens, probs)):
        logprob = row.get("logprob") if isinstance(row, dict) else None
        top = row.get("top_logprobs") if isinstance(row, dict) else None
        top_id = top[0].get("id") if isinstance(top, list) and top and isinstance(top[0], dict) else None
        if not isinstance(top_id, int) or isinstance(top_id, bool):
            # without the target's argmax the row proves nothing: a server that leaves the top list of
            # accepted draft tokens empty fails here
            top_id = None
        rows_with_argmax += top_id is not None
        if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
            failures.append({"index": index, "token": token, "reason": "nonfinite_target_logprob"})
        elif top_id is None:
            failures.append({"index": index, "token": token, "reason": "missing_target_argmax"})
        elif row.get("id") != token or top_id != token:
            failures.append({
                "index": index,
                "token": token,
                "target_argmax": top_id,
                "reason": "generated_token_not_target_argmax",
            })

    token_bytes = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return {
        "token_ids": tokens,
        "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "content_sha256": hashlib.sha256(str(resp.get("content", "")).encode("utf-8")).hexdigest(),
        "verifier_rows": len(probs),
        "verifier_rows_with_argmax": rows_with_argmax,
        "verifier_invariant_ok": not failures,
        "verifier_failures": failures,
    }


def run_prompt(prompt: dict[str, Any], repeats: int | None = None) -> dict[str, Any]:
    n_predict = int(prompt.get("n_predict", 256))
    messages = prompt.get("messages") or [{"role": "user", "content": prompt.get("prompt", "")}]
    formatted_prompt = apply_chat_template(messages)
    runs: list[dict[str, Any]] = []
    last_text = ""
    for _ in range(REPEATS if repeats is None else repeats):
        t0 = time.perf_counter()
        resp = post_completion(formatted_prompt, n_predict)
        elapsed = time.perf_counter() - t0
        timings = resp.get("timings") or {}
        evidence = analyze_native_response(resp)
        last_text = resp.get("content") if isinstance(resp.get("content"), str) else ""
        completion_tokens = timings.get("predicted_n")
        if not isinstance(completion_tokens, int):
            completion_tokens = len(evidence["token_ids"])
        tg = timings.get("predicted_per_second")
        if tg is None and elapsed > 0:
            tg = completion_tokens / elapsed
        draft_n = timings.get("draft_n")
        draft_acc = timings.get("draft_n_accepted")
        accept_rate = (draft_acc / draft_n) if (draft_n and draft_acc is not None) else None
        runs.append({
            "tg": tg,
            "pp": timings.get("prompt_per_second"),
            "elapsed_s": elapsed,
            "completion_tokens": completion_tokens,
            "prompt_tokens": timings.get("prompt_n"),
            "draft_n": draft_n,
            "draft_n_accepted": draft_acc,
            "target_evaluations": completion_tokens + (draft_n - draft_acc if draft_n and draft_acc is not None else 0),
            "accept_rate": accept_rate,
            **evidence,
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
        "all_verifier_invariants_ok": all(r["verifier_invariant_ok"] for r in runs),
        "completion_tokens": runs[-1]["completion_tokens"],
        "text_preview": last_text[:160].replace("\n", " "),
    }


def parse_rejection_records(text: str) -> list[dict[str, int]]:
    pattern = re.compile(r"task\s+(\d+)\s+\|\s+accepted\s+(\d+)/\s*(\d+)\s+draft tokens")
    generated_by_task: dict[int, int] = {}
    records: list[dict[str, int]] = []
    for match in pattern.finditer(text):
        task, accepted, drafted = (int(value) for value in match.groups())
        generated = generated_by_task.get(task, 0)
        if accepted < drafted:
            records.append({
                "task": task,
                "accepted": accepted,
                "drafted": drafted,
                "rejection_position": generated + accepted,
            })
        generated_by_task[task] = generated + accepted + 1
    return records


def scan_log(logpath: Path) -> dict[str, Any]:
    """Best-effort: pull FA, draft-acceptance, and hard-off trace lines."""
    fa_lines: list[str] = []
    acc_lines: list[str] = []
    hard_off_lines: list[str] = []
    try:
        text = logpath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"fa_lines": [], "acceptance_lines": [], "hard_off_lines": [], "rejection_records": [],
                "draft_rounds": 0, "checkpoint_restores": 0, "checkpoint_creates_dbg": 0}
    draft_rounds = 0
    checkpoint_restores = 0
    for line in text.splitlines():
        low = line.lower()
        # LLAMA_TRACE lines, one per verified draft; the debug-level twin ends in ", new n_tokens"
        if "accepted" in low and "draft tokens" in low and "new n_tokens" not in low:
            draft_rounds += 1
            if "(restore checkpoint)" in low:
                checkpoint_restores += 1
        # runtime diagnostics only: "flash" alone also matches model paths such as Qwen3.8-Flash-Next
        if ("flash_attn" in low or "fattn" in low or "flash attention" in low) and "warn" not in low:
            fa_lines.append(line.strip())
        if "draft acceptance" in low or "statistics" in low:
            acc_lines.append(line.strip())
        if "dead ngram-mod fires" in low and "disabling for seq" in low:
            hard_off_lines.append(line.strip())
    return {
        "fa_lines": list(dict.fromkeys(fa_lines))[:8],
        "acceptance_lines": acc_lines[-12:],
        "hard_off_lines": hard_off_lines,
        "rejection_records": parse_rejection_records(text),
        "draft_rounds": draft_rounds,
        "checkpoint_restores": checkpoint_restores,
        # only printed at debug verbosity (SERVER_EXTRA="-lv 5")
        "checkpoint_creates_dbg": text.count("created speculative checkpoint"),
    }


def library_path(bin_dir: str, inherited: str) -> str:
    """LD_LIBRARY_PATH with bin_dir first. No empty entry: the loader reads one as the current directory."""
    return os.pathsep.join(part for part in [bin_dir, *inherited.split(os.pathsep)] if part)


def arm_library_path(exe: Path) -> str:
    """The LD_LIBRARY_PATH an arm's server runs with: its own build directory first."""
    return library_path(str(exe.parent), os.environ.get("LD_LIBRARY_PATH", ""))


def start_server(arm: dict[str, Any], logpath: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["ZES_ENABLE_SYSMAN"] = "1"
    env["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
    if "server_bin" in arm:
        # the build-tree RUNPATH points at one build dir; make each arm load its own libraries
        env["LD_LIBRARY_PATH"] = arm_library_path(Path(arm["server_bin"]).resolve())
        env.setdefault("LLAMA_TRACE", "1")
    with logpath.open("w", encoding="utf-8") as logf:
        return subprocess.Popen(
            ["bash", "-c", server_command(arm)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            # Keep curve servers inside the wrapper timeout's process group so its
            # final SIGKILL reaches the server even if Python cannot finish cleanup.
            start_new_session=MODE != "acceptance-curve",
        )


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    for sig, grace in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            group = os.getpgid(proc.pid)
            if group == os.getpgrp():
                proc.send_signal(sig)  # shared timeout group: do not signal ourselves
            else:
                os.killpg(group, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            if sig == signal.SIGKILL:
                raise  # do not report successful cleanup while the child is alive


def curve_path(arm_name: str) -> Path:
    """Per-request JSONL totals consumed by test-spec-adaptive-curve."""
    return RESULTS / f"acceptance_curve_{Path(MODEL).stem}_{arm_name}.jsonl"


def run_arm(arm: dict[str, Any], prompts: list[dict[str, Any]], tag: str = "") -> dict[str, Any]:
    logpath = RESULTS / f"{arm['name']}{tag}.log"
    print(f"\n=== arm {arm['name']}  (kv={arm['kv']}, spec={arm['spec_label']}) ===", flush=True)
    proc = start_server(arm, logpath)
    try:
        if not wait_health(HEALTH_TIMEOUT):
            print(f"  !! server failed health within {HEALTH_TIMEOUT}s (see {logpath})", flush=True)
            return {"arm": arm, "error": "health_timeout", "prompts": []}
        # SYCL JIT warmup: run the first prompt once (discarded) so kernel
        # compilation (incl. batched-verify when spec fires) is not charged to
        # the first measured generation. A failed warmup fails the launch: the
        # first measured request would absorb that work, and a warmup response
        # that fails the verifier must not slip past the launch gate.
        print("  warmup...", flush=True)
        try:
            wp = prompts[0]
            messages = wp.get("messages") or [{"role": "user", "content": wp.get("prompt", "")}]
            evidence = analyze_native_response(
                post_completion(apply_chat_template(messages), int(wp.get("n_predict", 256))))
        except Exception as e:  # noqa: BLE001 - any failure, request or response, fails the launch
            print(f"  !! warmup failed: {e}", flush=True)
            return {"arm": arm, "error": f"warmup failed: {e}", "prompts": []}
        if not evidence["verifier_invariant_ok"]:
            print(f"  !! warmup response failed the target-argmax verifier: {evidence['verifier_failures'][:3]}",
                  flush=True)
            return {"arm": arm, "error": "warmup response failed the target-argmax verifier", "prompts": []}
        results = []
        interleave = MODE == "acceptance-curve"
        requests = (p for _ in range(REPEATS if interleave else 1) for p in prompts)
        for p in requests:
            r = run_prompt(p, repeats=1) if interleave else run_prompt(p)
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


def run_arm_curve(arm: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    """Publish a trace only after the launch and GPU fault gates pass."""
    path = curve_path(arm["name"])
    path.unlink(missing_ok=True)
    before = dmesg_lines()
    if not before:
        return {"arm": arm, "error": "kernel log unreadable or empty", "rows": 0}
    launch = run_arm(arm, prompts, tag="-curve")
    problems = launch_problems(launch)
    faults = new_gpu_faults(before, dmesg_lines())
    if faults is None or faults:
        problems.append(f"GPU fault gate failed: {faults}")
    rows = []
    for prompt in launch["prompts"]:
        for run in prompt["runs"]:
            drafted, accepted = run.get("draft_n"), run.get("draft_n_accepted")
            if (not isinstance(drafted, int) or isinstance(drafted, bool) or
                    not isinstance(accepted, int) or isinstance(accepted, bool) or
                    drafted <= 0 or not 0 <= accepted <= drafted):
                problems.append(f"{prompt['id']}: invalid or missing draft statistics")
                continue
            rows.append({"n_draft": drafted, "n_accepted": accepted, "prompt_id": prompt["id"]})
    if not rows:
        problems.append("no measured requests")
    if problems:
        return {"arm": arm, "error": "; ".join(problems), "rows": 0}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"arm": arm, "error": None, "curve_path": str(path), "rows": len(rows)}


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


def gpu_holders() -> list[str]:
    """What holds the render node; empty only when fuser reports an idle node.

    fuser exits 1 without any output for an idle node. It also exits 1 for a node that does not
    exist, with the reason on stderr. Every outcome but the first counts as held, as
    check_sole_tenancy in scripts/bench-a770-fork-unique.py does.
    """
    try:
        proc = subprocess.run(["fuser", RENDER_NODE], check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [f"<fuser failed: {e}>"]
    output = [line.strip() for part in (proc.stdout, proc.stderr) for line in part.splitlines() if line.strip()]
    if proc.returncode == 1 and not output:
        return []
    return output or [f"<fuser exited {proc.returncode} without holder or error details>"]


def dmesg_lines() -> list[str] | None:
    """The kernel log, or None when dmesg is not readable."""
    for cmd in (["dmesg"], ["sudo", "-n", "dmesg"]):
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            return proc.stdout.splitlines()
    return None


def is_gpu_fault(line: str) -> bool:
    return bool(GPU_DRIVER_RE.search(line) and GPU_FAULT_TERM_RE.search(line))


def kmsg_lines_since(before: list[str], after: list[str]) -> list[str] | None:
    """Lines the kernel log gained between two reads.

    The ring buffer drops its oldest lines first, so the second read starts with a suffix of the
    first for as long as any line of the first survives. None means no line survived (the buffer
    wrapped past the first read or was cleared), so lines logged in between may be gone.
    """
    for dropped in range(len(before)):
        kept = len(before) - dropped
        if kept <= len(after) and after[:kept] == before[dropped:]:
            return after[kept:]
    return None


def new_gpu_faults(before: list[str] | None, after: list[str] | None) -> list[str] | None:
    """i915/xe fault lines logged between two kernel log reads, or None when that cannot be told."""
    if before is None or after is None:
        return None
    gained = kmsg_lines_since(before, after)
    if gained is None:
        return None
    return [line for line in gained if is_gpu_fault(line)]


def workload_mismatches(launches_a: list[dict[str, Any]], launches_b: list[dict[str, Any]],
                        prompt_ids: list[str]) -> list[dict[str, Any]]:
    """Launch pairs and prompts whose two arms did not generate the same ordered token streams.

    The paired delta compares launch i of one arm with launch i of the other. It measures speed only
    where both generated the same tokens; elsewhere it also carries the workload difference (other
    lengths, other acceptance). Not a gate: temperature-0 output is not reproducible on SYCL.
    """
    def streams(launch: dict[str, Any], pid: str) -> list[str] | None:
        for prompt in launch.get("prompts", []):
            if prompt.get("id") == pid:
                return [run.get("token_sha256") for run in prompt.get("runs", [])]
        return None

    mismatches = []
    for i, (la, lb) in enumerate(zip(launches_a, launches_b)):
        for pid in prompt_ids:
            sa, sb = streams(la, pid), streams(lb, pid)
            if sa is None or sb is None or sa != sb:
                mismatches.append({"launch": i, "prompt": pid})
    return mismatches


def argmax_evidence(launches: list[dict[str, Any]]) -> dict[str, int]:
    """Response rows over all launches, and how many carried a top list the verifier could check."""
    runs = [run for launch in launches for prompt in launch.get("prompts", []) for run in prompt.get("runs", [])]
    counted = all("verifier_rows_with_argmax" in run for run in runs)
    # summaries written before the harness counted the rows carry no count: unknown, not zero
    return {"rows": sum(run.get("verifier_rows", 0) for run in runs),
            "rows_with_argmax": sum(run["verifier_rows_with_argmax"] for run in runs) if counted else None}


def launch_problems(launch: dict[str, Any]) -> list[str]:
    """Why a launch cannot enter the paired statistics; empty for a usable one."""
    if launch.get("error"):
        return [str(launch["error"])]
    problems = ["no draft statistics"] if launch.get("spec_stats_missing") else []
    for prompt in launch["prompts"]:
        if not isinstance(prompt.get("tg_median"), (int, float)):
            problems.append(f"{prompt['id']}: no timings")
        if not prompt.get("all_verifier_invariants_ok", False):
            problems.append(f"{prompt['id']}: response failed the target-argmax verifier")
    return problems


def ab_exit_code(ok: bool, new_faults: list[str] | None) -> int:
    """0 only for complete launches and an evaluated, empty fault gate."""
    return 0 if ok and new_faults == [] else 1


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


CORE_LIBRARY_PREFIXES = ("libllama", "libggml", "libmtmd")


def build_hashes(exe: Path, resolved: dict[str, str] | None = None) -> dict[str, str | None]:
    """sha256 of the server binary and of every shared library in its directory, keyed by file name.

    A shared build keeps core code (libllama, the ggml backends) in libraries the server loads from
    its own directory (library_path puts it first), so the binary alone can be byte-identical across
    builds that differ. Symlinks are skipped: they name a library that is hashed under its own name.
    Backends loaded with dlopen sit in that directory too. A llama or ggml library the loader resolves
    elsewhere (resolved, from resolved_libraries) is hashed under its path; runtimes such as oneAPI
    and the system libraries are not hashed.
    """
    hashes = {exe.name: file_sha256(exe)}
    for lib in sorted(exe.parent.glob("lib*.so*")):
        if lib.is_file() and not lib.is_symlink():
            hashes[lib.name] = file_sha256(lib)
    for soname, path in sorted((resolved or {}).items()):
        lib = Path(path)
        if soname.startswith(CORE_LIBRARY_PREFIXES) and lib.is_absolute() and lib.resolve().parent != exe.parent.resolve():
            hashes[str(lib)] = file_sha256(lib)
    return hashes


LDD_LINE_RE = re.compile(r"^\s*(\S+) => (.+?)(?: \(0x[0-9a-f]+\))?$")


def resolved_libraries(exe: Path, ld_library_path: str) -> dict[str, str] | None:
    """Shared library -> path as the dynamic loader resolves them for exe (ldd), None when ldd fails."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ld_library_path
    try:
        proc = subprocess.run(["ldd", str(exe)], check=False, capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    libraries = {}
    for line in proc.stdout.splitlines():
        match = LDD_LINE_RE.match(line)
        if match:
            libraries[match.group(1)] = match.group(2).strip()
    return libraries


def render_driver(render_node: str, drm_sysfs_root: str = "/sys/class/drm") -> str | None:
    """Kernel driver bound to the render node's device (xe or i915 on an Arc), None when unknown."""
    link = Path(drm_sysfs_root) / Path(render_node).name / "device" / "driver"
    try:
        return Path(os.readlink(link)).name
    except OSError:
        return None


def launch_order(launches: int) -> list[tuple[int, int]]:
    """Arm indices of each launch pair: AB, BA, AB, ... Only an even count balances the positions."""
    return [(0, 1) if i % 2 == 0 else (1, 0) for i in range(launches)]


def paired_stats(a: list[float], b: list[float]) -> dict[str, Any]:
    """Paired B-A difference with a 95% t interval; pair i is the i-th launch of each arm."""
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0}
    a, b = a[:n], b[:n]
    diffs = [y - x for x, y in zip(a, b)]
    a_mean = statistics.fmean(a)
    delta = statistics.fmean(diffs)
    half = T95[min(n - 1, 15)] * statistics.stdev(diffs) / math.sqrt(n) if n >= 2 else None
    return {
        "n": n,
        "a_mean": a_mean,
        "b_mean": statistics.fmean(b),
        "delta": delta,
        "ci95_half": half,
        "delta_pct": 100.0 * delta / a_mean,
        "ci95_half_pct": 100.0 * half / a_mean if half is not None else None,
    }


def run_ab(arms: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> int:
    if LAUNCHES < 2 or LAUNCHES % 2:
        # with an odd number of AB pairs no order gives both arms the same mean position
        print(f"!! LAUNCHES={LAUNCHES}: MODE=ab needs an even, positive launch count per arm, "
              "so that the ABBA order balances drift", flush=True)
        return EXIT_USAGE
    prompt_ids = [p["id"] for p in prompts]
    suffix = f"-{OUT_TAG}" if OUT_TAG else ""
    out_path = RESULTS / f"summary_ab{suffix}.json"
    a, b = arms
    if a["name"] == b["name"]:
        # launches are kept per name: one arm's launches would stand in for the other's
        print(f"!! both arms are named {a['name']!r}; set NAME_A and NAME_B to different labels", flush=True)
        return EXIT_USAGE
    for arm in arms:
        exe = Path(arm["server_bin"]).resolve()
        arm["libraries"] = resolved_libraries(exe, arm_library_path(exe))
        arm["sha256"] = build_hashes(exe, arm["libraries"])
        print(f"arm {arm['name']}: {arm['server_bin']} ({len(arm['sha256'])} files hashed)")
        if arm["libraries"] is None:
            print(f"!! ldd failed for {exe}: the libraries this arm loads are not recorded", flush=True)
        else:
            for soname, path in sorted(arm["libraries"].items()):
                if path == "not found":
                    print(f"!! arm {arm['name']}: the loader finds no {soname}", flush=True)
                elif soname.startswith(CORE_LIBRARY_PREFIXES) and not path.startswith(str(exe.parent) + os.sep):
                    print(f"!! arm {arm['name']} loads {soname} from {path}, outside its build directory", flush=True)
    # numbers from one kernel driver are no baseline for the other (AGENTS.md, "Kernel Driver")
    kernel_driver = render_driver(RENDER_NODE)
    if kernel_driver not in ("xe", "i915"):
        # the fault gate knows only these two, so another GPU's faults would pass it unseen
        print(f"!! {RENDER_NODE} is served by {kernel_driver or 'an unknown driver'}, not xe or i915; "
              "set RENDER_NODE to the Arc's render node", flush=True)
        return EXIT_USAGE
    print(f"kernel driver of {RENDER_NODE}: {kernel_driver}")

    kmsg_before = dmesg_lines()
    launches: dict[str, list[dict[str, Any]]] = {a["name"]: [], b["name"]: []}
    out: dict[str, Any] = {
        "mode": MODE, "model": MODEL, "draft_model": DRAFT_MODEL, "placement": PLACEMENT, "ctx": CTX,
        "threads": THREADS, "repeats": REPEATS, "launches_per_arm": LAUNCHES, "prompt_ids": prompt_ids,
        "kernel_driver": kernel_driver, "arms": arms, "server_extra": SERVER_EXTRA, "launches": launches,
    }
    for i, pair in enumerate(launch_order(LAUNCHES)):
        for arm in (arms[pair[0]], arms[pair[1]]):
            holders = gpu_holders()
            if holders:
                print(f"!! {RENDER_NODE} is held by {' '.join(holders)}; refusing to time a shared GPU", flush=True)
                return EXIT_GPU_BUSY
            launches[arm["name"]].append(run_arm(arm, prompts, tag=f"{suffix}-L{i}"))
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def tg_of(launch: dict[str, Any], pid: str) -> float | None:
        for pr in launch["prompts"]:
            if pr["id"] == pid and isinstance(pr["tg_median"], (int, float)):
                return float(pr["tg_median"])
        return None

    def series(name: str, pid: str | None) -> list[float]:
        vals = []
        for launch in launches[name]:
            per_prompt = [tg_of(launch, p) for p in ([pid] if pid else prompt_ids)]
            if launch_problems(launch) or any(v is None for v in per_prompt):
                continue
            vals.append(statistics.fmean(per_prompt))
        return vals

    ok = all(len(series(arm["name"], None)) == LAUNCHES for arm in arms)
    stats = {pid or "all": paired_stats(series(a["name"], pid), series(b["name"], pid))
             for pid in [*prompt_ids, None]} if ok else {}

    def hashes(name: str, pid: str) -> set[str]:
        return {run["token_sha256"] for launch in launches[name] for pr in launch["prompts"]
                if pr["id"] == pid for run in pr["runs"]}

    tokens = {pid: {"a_distinct": len(hashes(a["name"], pid)), "b_distinct": len(hashes(b["name"], pid)),
                    "identical_across_arms": hashes(a["name"], pid) == hashes(b["name"], pid)}
              for pid in prompt_ids}
    events = {arm["name"]: {key: sum(launch.get("log_scan", {}).get(key, 0) for launch in launches[arm["name"]])
                            for key in ("draft_rounds", "checkpoint_restores", "checkpoint_creates_dbg")}
              for arm in arms}
    mismatched = workload_mismatches(launches[a["name"]], launches[b["name"]], prompt_ids)
    evidence = {arm["name"]: argmax_evidence(launches[arm["name"]]) for arm in arms}
    new_faults = new_gpu_faults(kmsg_before, dmesg_lines())
    out.update({"paired_tg": stats, "token_identity": tokens, "checkpoint_events": events,
                "paired_workload_mismatches": mismatched, "paired_tg_isolates_arms": ok and not mismatched,
                "argmax_evidence": evidence,
                "dmesg_new_gpu_faults": new_faults})
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # a delta over pairs that generated different tokens also measures the workload difference
    kind = "exploratory: arms generated different tokens" if mismatched else "same token streams in every pair"
    print(f"\n\n## Paired tg, {b['name']} minus {a['name']} ({LAUNCHES} launches per arm, ABBA order; {kind})\n")
    print(f"| prompt | {a['name']} t/s | {b['name']} t/s | delta t/s | delta % | 95% CI half-width % |")
    print("|---|---|---|---|---|---|")
    for pid, st in stats.items():
        half = f"{st['ci95_half_pct']:.2f}" if st.get("ci95_half_pct") is not None else "n/a"
        print(f"| {pid} | {st['a_mean']:.2f} | {st['b_mean']:.2f} | {st['delta']:+.2f} | {st['delta_pct']:+.2f} | {half} |")
    if mismatched:
        print(f"!! {len(mismatched)} of {len(prompt_ids) * LAUNCHES} paired (launch, prompt) cells generated different "
              "token streams in the two arms: their delta mixes speed with a different workload")
    if not ok:
        print("!! at least one launch is unusable; no paired statistics")
        for arm in arms:
            for i, launch in enumerate(launches[arm["name"]]):
                for problem in launch_problems(launch):
                    print(f"   {arm['name']} launch {i}: {problem}")
    print("\n## Checkpoint activity (summed over launches)\n")
    for name, ev in events.items():
        print(f"  {name}: draft rounds {ev['draft_rounds']}, checkpoint restores {ev['checkpoint_restores']}, "
              f"checkpoint creates (debug log only) {ev['checkpoint_creates_dbg']}")
    print("\n## Token identity (temperature 0)\n")
    for pid, t in tokens.items():
        print(f"  {pid}: distinct streams {a['name']}={t['a_distinct']} {b['name']}={t['b_distinct']}, "
              f"identical across arms: {t['identical_across_arms']}")
    print("\n## Verifier evidence\n")
    for name, ev in evidence.items():
        print(f"  {name}: {ev['rows_with_argmax']} of {ev['rows']} response rows carried the target argmax; "
              "a row without it fails the verifier")
    if new_faults is None:
        print("\n!! dmesg unreadable, wrapped or cleared during the run: GPU fault gate NOT evaluated")
    else:
        print(f"\nnew i915/xe fault lines in dmesg: {len(new_faults)}")
        for line in new_faults[:5]:
            print(f"  {line}")
    print(f"kernel driver: {kernel_driver or 'unknown'}")
    print(f"\nsummary -> {out_path}")
    return ab_exit_code(ok, new_faults)


def main() -> int:
    prompts = load_prompts()
    prompt_ids = [p["id"] for p in prompts]
    arms = build_arms()
    if MODE == "ab":
        print(f"MODE=ab  MODEL={MODEL}  DRAFT_MODEL={DRAFT_MODEL or '-'}")
        print(f"PORT={PORT} CTX={CTX} THREADS={THREADS} REPEATS={REPEATS} LAUNCHES={LAUNCHES} PLACEMENT={PLACEMENT}")
        return run_ab(arms, prompts)
    if MODE == "acceptance-curve":
        print(f"MODE=acceptance-curve  MODEL={MODEL}")
        print(f"SERVER_BIN={SERVER_BIN}")
        print(f"PORT={PORT} CTX={CTX} THREADS={THREADS} REPEATS={REPEATS}")
        print(f"arms: {[a['name'] for a in arms]}")
        # Invalidate the complete old campaign before any arm can publish new data.
        for arm in arms:
            curve_path(arm["name"]).unlink(missing_ok=True)
        driver = render_driver(RENDER_NODE)
        if driver not in {"xe", "i915"} or not prompts or REPEATS < 1:
            print("!! requires xe/i915, a nonempty prompt set and positive REPEATS")
            return EXIT_USAGE
        print(f"kernel driver: {driver}")
        print("Request totals only; sequential arms do not establish a throughput improvement.")
        results = []
        for arm in arms:
            holders = gpu_holders()
            if holders:
                print(f"!! {RENDER_NODE} is held by {' '.join(holders)}; "
                      "refusing to time a shared GPU", flush=True)
                return EXIT_GPU_BUSY
            results.append(run_arm_curve(arm, prompts))
        print("\n## Acceptance curve summary\n")
        for r in results:
            name = r["arm"]["name"]
            err = r.get("error")
            path = r.get("curve_path", curve_path(name))
            rows = r.get("rows", 0)
            if err:
                print(f"  {name}: ERROR ({err}); no trace published")
            else:
                print(f"  {name}: {rows} rows -> {path}")
        return 1 if any(r.get("error") for r in results) else 0
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
