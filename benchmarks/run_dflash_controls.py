#!/usr/bin/env python3
"""Run pinned ordinary DFlash2 controls. No streaming, UVM, fit retries or implicit offload."""

import argparse
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/server-ab"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from server_ab import request_json, stream_completion, wait_for_health
from inspect_dflash_artifacts import inspect_artifacts, read_json, verify_file

SPEC = importlib.util.spec_from_file_location("dflash_sweep", ROOT / "benchmarks/run-fixed-span-sweep.py")
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


def server_command(args, context, depth):
    if depth not in (1, 3, 7) or not 512 <= context <= 262144:
        raise ValueError("ordinary control requires D=1,3,7 and a valid context")
    return [str(args.server), "--model", str(args.target), "--model-draft", str(args.draft),
            "--spec-type", "draft-dflash", "--spec-draft-n-max", str(depth),
            "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
            "--ctx-size", str(context), "--batch-size", "256", "--ubatch-size", "256",
            "--parallel", "1", "--n-gpu-layers", "999", "--spec-draft-ngl", "999",
            "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
            "--spec-draft-type-k", args.draft_k, "--spec-draft-type-v", args.draft_v,
            "--fit", "off", "--no-mmproj", "--host", "127.0.0.1", "--port", str(args.port),
            "--threads", "8", "--threads-batch", "8", "-lv", "3"]


def validate_result(result, prompt_tokens, decode_tokens):
    timing = result.get("timings", {})
    if (timing.get("prompt_n") != prompt_tokens or timing.get("predicted_n") != decode_tokens
            or len(result.get("tokens", [])) != decode_tokens):
        raise ValueError("incomplete ordinary control response")
    for name in ("prompt_per_second", "predicted_per_second"):
        value = timing.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid ordinary control timing: {name}")


def failure_kind(text):
    return "capacity_failure" if "out of memory" in text.lower() else "failed"


def summarize_trace(rows, depth):
    rows = [r for r in rows if not r.get("warmup", False)]
    processes = [r for r in rows if r.get("event") == "process"]
    if any(r.get("ret") != 0 for r in processes):
        raise ValueError("failed traced process")
    injection = [r for r in processes if r.get("role") == "draft" and r.get("kind") == "features"]
    draft = [r for r in processes if r.get("role") == "draft" and r.get("kind") == "tokens"]
    target = [r for r in processes if r.get("role") == "target" and r.get("kind") == "tokens"]
    lattice = [r for r in rows if r.get("event") == "tensor" and r.get("name") == "lattice" and r.get("file")]
    attention = [r for r in rows if r.get("event") == "attention" and r.get("role") == "draft"]
    if not injection or not draft or not target or not lattice or not attention:
        raise ValueError("trace does not cover injection, prediction, verification, lattice and SWA masks")
    if any(r.get("feature_width") != 25600 or not 1 <= r.get("rows", 0) <= 256 for r in injection):
        raise ValueError("invalid feature injection geometry")
    if any(r.get("rows") != depth+1 for r in draft):
        raise ValueError("draft block width differs from configured control")
    if depth+1 not in {r.get("rows") for r in target}:
        raise ValueError("target verification block was not observed")
    if any(r.get("query_rows") != depth+1 or not r.get("visible_rows") or not r.get("mask_file") for r in attention):
        raise ValueError("incomplete draft mask evidence")
    return {"injection_rows": sorted({r["rows"] for r in injection}),
            "draft_widths": sorted({r["rows"] for r in draft}),
            "target_widths": sorted({r["rows"] for r in target}),
            "visible_swa_rows": sorted({v for r in attention for v in r["visible_rows"]}),
            "process_count": len(processes), "attention_count": len(attention)}


def validate_payloads(directory, rows):
    files = {}
    def visit(value):
        if isinstance(value, dict):
            if "file" in value:
                name = value["file"]
                if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                    raise ValueError("invalid trace payload filename")
                path = directory / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("missing or redirected trace payload")
                if not 0 < value.get("bytes", 0) <= 64*1024*1024:
                    raise ValueError("invalid trace payload size")
                pin = {"size": value["bytes"], "sha256": value["sha256"]}
                if name in files and files[name] != pin:
                    raise ValueError("conflicting trace payload identity")
                files[name] = pin
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(rows)
    if sum(pin["size"] for pin in files.values()) > 512*1024*1024:
        raise ValueError("trace exceeds payload budget")
    for name, pin in files.items():
        verify_file(directory / name, pin)
    return len(files)


def stop_process(process):
    # Nsight owns a child server; stop the entire session and let the profiler flush.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_point(args, context, depth, token_cache, mode):
    directory = args.output / f"context-{context}-d-{depth}-{mode}"
    directory.mkdir()
    log_path = directory / "server.log"
    command = server_command(args, context, depth)
    env = SWEEP.server_environment(False)
    # An inherited profiler would invalidate the uninstrumented control.
    env.pop("LD_PRELOAD", None)
    env.pop("DFLASH_CONTROL_TRACE", None)
    env.pop("DFLASH_CONTROL_CAPTURE", None)
    env.pop("LLAMA_RS_HOST_SPILL", None)
    if mode != "timing":
        env["LD_PRELOAD"] = str(args.trace_library)
        env["DFLASH_CONTROL_TRACE"] = str(directory)
        env["DFLASH_CONTROL_CAPTURE"] = "1" if mode == "capture" else "0"
    if mode == "nsight":
        command = [str(args.nsys), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--cuda-graph-trace=node",
                   "--cuda-memory-usage=true", "--output", str(directory / "native"), *command]
    row = {"status": "failed", "context": context, "depth": depth, "mode": mode,
           "command": command, "uvm": False, "log": str(log_path), "directory": str(directory)}
    process = None
    url = f"http://127.0.0.1:{args.port}"
    with log_path.open("x") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=True)
            wait_for_health(url, 180, process)
            if "tokens" not in token_cache:
                token_cache["tokens"] = request_json(url+"/tokenize", {
                    "content": args.prefill_text.read_text(), "add_special": False}, timeout=180)["tokens"]
            count = min(args.prompt_tokens, context-args.decode_tokens-depth-8)
            if count < 256 or len(token_cache["tokens"]) < count:
                raise ValueError("insufficient context/corpus for ordinary control")
            result = stream_completion(url+"/completion", {"prompt": token_cache["tokens"][:count],
                "n_predict": args.decode_tokens, "temperature": 0, "seed": 123, "ignore_eos": True,
                "cache_prompt": False, "stream": True, "return_tokens": True}, timeout=600)
            validate_result(result, count, args.decode_tokens)
            row.update(status="ok", prompt_tokens=count, decode_tokens=args.decode_tokens, result=result)
        except Exception as error:
            row["error"] = str(error)
        finally:
            if process is not None:
                stop_process(process)
    if row["status"] != "ok":
        row["status"] = failure_kind(log_path.read_text(errors="replace"))
    elif mode == "capture":
        try:
            trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
            row["trace"] = summarize_trace(trace, depth)
            row["trace"]["verified_payloads"] = validate_payloads(directory, trace)
        except Exception as error:
            row.update(status="failed", error=str(error))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "draft", "draft-config", "server", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--contexts", default="4096")
    parser.add_argument("--depths", default="1,3,7")
    parser.add_argument("--prompt-tokens", type=int, default=2304)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--draft-k", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--draft-v", choices=("f16", "q4_0"), default="f16")
    parser.add_argument("--prefill-text", type=Path, default=ROOT/"benchmarks/data/online-articles-262144-words.txt")
    parser.add_argument("--trace-library", type=Path)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--port", type=int, default=12351)
    args = parser.parse_args()
    contexts = [int(value) for value in args.contexts.split(",")]
    depths = [int(value) for value in args.depths.split(",")]
    if args.decode_tokens < 8 or args.prompt_tokens < 256 or not 1 <= args.port <= 65535:
        parser.error("invalid token or port bounds")
    for context in contexts:
        for depth in depths:
            server_command(args, context, depth)
    for name in ("target", "draft", "draft_config", "server", "prefill_text", "trace_library", "nsys"):
        value = getattr(args, name)
        if value is not None:
            if not value.is_file():
                parser.error(f"missing input: {value}")
            setattr(args, name, value.resolve())
    if args.nsys and not args.trace_library:
        parser.error("Nsight phase attribution requires --trace-library")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.jsonl"
    if path.exists():
        parser.error("results already exist; choose another output directory")
    manifest = read_json(ROOT/"benchmarks/data/dflash2-qwen27b-artifacts.json")
    report = inspect_artifacts({"target": args.target, "draft": args.draft, "draft_config": args.draft_config}, manifest)
    (args.output / "artifacts.json").write_text(json.dumps(report, indent=2)+"\n")
    modes = ["timing"] + (["capture"] if args.trace_library else []) + (["nsight"] if args.nsys else [])
    token_cache = {}
    with path.open("x") as output:
        for context in contexts:
            for depth in depths:
                for mode in modes:
                    started = time.monotonic()
                    row = run_point(args, context, depth, token_cache, mode)
                    row["wall_seconds"] = time.monotonic()-started
                    output.write(json.dumps(row)+"\n")
                    output.flush()
                    print(json.dumps({key: value for key, value in row.items() if key != "result"}), flush=True)
                    if row["status"] == "capacity_failure":
                        break  # Do not repeat a known physical-fit failure under instrumentation.
                    if row["status"] != "ok":
                        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
