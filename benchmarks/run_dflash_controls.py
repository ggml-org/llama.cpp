#!/usr/bin/env python3
"""Run matched native/streamed DFlash2 controls with explicit fitting and profiling options."""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import signal
import struct
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
KV_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")


def effective_microbatch(args, context):
    batch=getattr(args,"batch_size",256)
    ubatch=getattr(args,"ubatch_size",256)
    if type(batch) is not int or not 1 <= batch <= 2**32-1 or type(ubatch) is not int or not 0 <= ubatch <= 2**32-1:
        raise ValueError("batch must be positive uint32; microbatch may be zero to inherit")
    return min(batch,context,ubatch or batch)


def prompt_count(args, context, depth):
    maximum=getattr(args,"max_depth",depth)
    headroom=getattr(args,"prompt_headroom",None)
    headroom=maximum+8 if headroom is None else headroom
    if type(headroom) is not int or headroom < maximum+1:
        raise ValueError("prompt headroom must cover the configured verification width")
    count=min(args.prompt_tokens,context-args.decode_tokens-headroom)
    if count < 256:
        raise ValueError("context does not leave room for the prompt")
    return count


def auto_arena_for_point(args, context, depth, token_cache):
    if not getattr(args,"auto_max_arena",False):
        return args.arena_mib,None
    cache={}
    probe_root=args.output/"fit-probes"
    probe_root.mkdir(parents=True,exist_ok=True)
    def run(candidate,full):
        config=copy.copy(args)
        config.arena_mib=candidate
        config.output=probe_root/f"context-{context}-d-{depth}-arena-{candidate}-{'full' if full else 'fast'}"
        config.output.mkdir()
        if not full:
            config.prompt_tokens=min(args.prompt_tokens,2560)
            config.decode_tokens=min(args.decode_tokens,32)
        started=time.monotonic()
        row=run_point(config,context,depth,token_cache,"timing")
        row["wall_seconds"]=time.monotonic()-started
        row["arena_mib"]=candidate
        row["full_workload"]=full
        with (probe_root/"results.jsonl").open("a") as output:
            output.write(json.dumps(row)+"\n")
        print(json.dumps({"arena_probe":candidate,"context":context,"depth":depth,"full":full,"status":row["status"]}),flush=True)
        if row["status"] == "quota_failure":
            minimum=SWEEP.arena_minimum_mib(Path(row["log"]).read_text(errors="replace"))
            raise SWEEP.ArenaTooSmallError(minimum or 0)
        if row["status"] not in ("ok","capacity_failure"):
            raise RuntimeError(f"arena probe failed: {row.get('error',row)}")
        return row
    def probe(candidate):
        if candidate not in cache:
            cache[candidate]=run(candidate,False)
        return cache[candidate]["status"] == "ok"
    upper=1048576
    start=args.arena_mib
    while upper >= 1:
        candidate=SWEEP.find_max_arena_mib(start,probe,upper_limit=upper)
        row=run(candidate,True)
        if row["status"] == "ok":
            return candidate,row
        upper=candidate-1
        start=min(start,upper)
    raise RuntimeError("no arena fits the full workload")


def server_command(args, context, depth):
    if depth not in (1, 3, 7) or not 512 <= context <= 262144:
        raise ValueError("ordinary control requires D=1,3,7 and a valid context")
    batch = getattr(args,"batch_size",256)
    ubatch = getattr(args,"ubatch_size",256)
    key = getattr(args,"target_k","q8_0")
    value = getattr(args,"target_v","q4_0")
    effective_microbatch(args,context)
    if any(t not in KV_TYPES for t in (key,value,args.draft_k,args.draft_v)):
        raise ValueError("unknown KV type")
    command = [str(args.server), "--model", str(args.target), "--model-draft", str(args.draft),
            "--spec-type", "draft-dflash", "--spec-draft-n-max", str(depth),
            "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
            "--ctx-size", str(context), "--batch-size", str(batch), "--ubatch-size", str(ubatch),
            "--parallel", "1", "--n-gpu-layers", "999", "--spec-draft-ngl", "999",
            "--flash-attn", "on", "--cache-type-k", key, "--cache-type-v", value,
            "--spec-draft-type-k", args.draft_k, "--spec-draft-type-v", args.draft_v,
            "--fit", "off", "--no-mmproj", "--host", "127.0.0.1", "--port", str(args.port),
            "--threads", "8", "--threads-batch", "8", "-lv", str(getattr(args,"verbosity",3))]
    arena = getattr(args, "arena_mib", 0)
    if arena < 0:
        raise ValueError("negative arena size")
    if arena:
        command += ["--kv-stream-arena-mib", str(arena)]
    return command


def validate_result(result, prompt_tokens, decode_tokens):
    timing = result.get("timings", {})
    if (timing.get("prompt_n") != prompt_tokens or timing.get("predicted_n") != decode_tokens
            or len(result.get("tokens", [])) != decode_tokens):
        raise ValueError("incomplete ordinary control response")
    if type(timing.get("draft_n")) is not int or timing["draft_n"] <= 0:
        raise ValueError("DFlash control completed without generating drafts")
    for name in ("prompt_per_second", "predicted_per_second"):
        value = timing.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid ordinary control timing: {name}")


def failure_kind(text):
    if "shared arena quota insufficient" in text.lower():
        return "quota_failure"
    return "capacity_failure" if "out of memory" in text.lower() else "failed"


def summarize_trace(rows, depth, max_injection_rows=256):
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
    if any(r.get("feature_width") != 25600 or not 1 <= r.get("rows", 0) <= max_injection_rows for r in injection):
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
                if not 0 < value.get("bytes", 0) <= 128*1024*1024:
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
    if sum(pin["size"] for pin in files.values()) > 1024*1024*1024:
        raise ValueError("trace exceeds payload budget")
    for name, pin in files.items():
        verify_file(directory / name, pin)
    return len(files)


def compare_control_rows(actual, control, artifacts_actual, artifacts_control, trace_actual, trace_control, *, allow_uvm_difference=False):
    for row in (actual, control):
        if row.get("status") != "ok" or row.get("mode") != "capture":
            raise ValueError("matching control requires successful captured runs")
    for field in ("context", "depth", "prompt_tokens", "decode_tokens"):
        if field not in actual or actual[field] != control.get(field):
            raise ValueError(f"nonmatching control profile: {field}")
    if any(type(row.get("uvm")) is not bool for row in (actual,control)) or (
            actual["uvm"] != control["uvm"] and not allow_uvm_difference):
        raise ValueError("nonmatching control profile: uvm")
    for option in ("--batch-size", "--ubatch-size", "--cache-type-k", "--cache-type-v",
                   "--spec-draft-type-k", "--spec-draft-type-v", "--spec-draft-n-max",
                   "--spec-draft-n-min", "--spec-draft-p-min", "--n-gpu-layers", "--spec-draft-ngl"):
        values = []
        for row in (actual, control):
            command = row["command"]
            if command.count(option) != 1 or command.index(option)+1 == len(command):
                raise ValueError(f"missing/ambiguous control option: {option}")
            values.append(command[command.index(option)+1])
        if values[0] != values[1]:
            raise ValueError(f"nonmatching control option: {option}")
    if not artifacts_actual.get("sha256") or artifacts_actual["sha256"] != artifacts_control.get("sha256"):
        raise ValueError("nonmatching control artifacts")
    if actual["result"]["tokens"] != control["result"]["tokens"]:
        raise ValueError("matching control generated different token IDs")
    def signature(value):
        if isinstance(value, dict):
            excluded = {"file", "mask_file"}
            if value.get("name") == "lattice" and value.get("canonical_sha256"):
                excluded.add("sha256")
            return {key: signature(child) for key, child in value.items() if key not in excluded}
        if isinstance(value, list):
            return [signature(child) for child in value]
        return value
    # Device free-memory readings are observations, not arithmetic outputs.
    expected = [signature(row) for row in trace_control if row.get("event") != "memory"]
    measured = [signature(row) for row in trace_actual if row.get("event") != "memory"]
    if not expected or len(expected) != len(measured):
        raise ValueError("matching control trace record count differs")
    for index, (left, right) in enumerate(zip(expected, measured)):
        if left != right:
            raise ValueError(f"matching control first differs at trace record {index}: {left.get('event')}")
    return {"token_ids_equal": True, "trace_records_equal": len(expected)}


def canonical_lattice(data, rows, width, top_k=16, vocab=248320):
    if (not 2 <= rows <= 8 or not 1 <= top_k <= 16 or not top_k+top_k*top_k <= width <= 5120
            or len(data) != rows*width*4):
        raise ValueError("invalid selector lattice geometry")
    if any(data[:width*4]):
        raise ValueError("selector anchor row is not zero")
    result = []
    previous = []
    for index in range(1, rows):
        raw = data[index*width*4:(index+1)*width*4]
        if any(raw[(top_k+top_k*top_k)*4:]):
            raise ValueError("selector padding is not zero")
        values = struct.unpack("<"+"f"*width, raw)
        ids = list(values[:top_k])
        if any(not math.isfinite(v) or int(v) != v or not 0 <= v < vocab for v in ids) or len(set(ids)) != top_k:
            raise ValueError("invalid or duplicate selector candidate ID")
        scores = values[top_k:top_k+top_k*top_k]
        if any(not math.isfinite(value) for value in scores):
            raise ValueError("invalid selector score")
        if index == 1:
            if any(raw[top_k*4:(top_k+top_k)*4] != raw[(top_k+k*top_k)*4:(top_k+(k+1)*top_k)*4]
                   for k in range(1, top_k)):
                raise ValueError("anchor scores are not repeated")
            entries = [(int(ids[j]), struct.pack("<f", scores[j]).hex()) for j in range(top_k)]
        else:
            entries = [(int(previous[k]), int(ids[j]), struct.pack("<f", scores[k*top_k+j]).hex())
                       for k in range(top_k) for j in range(top_k)]
        result.append(sorted(entries))
        previous = ids
    return hashlib.sha256(json.dumps([rows, width, top_k, result], separators=(",", ":")).encode()).hexdigest()


def load_control_trace(directory):
    rows = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    validate_payloads(directory, rows)
    for row in rows:
        if row.get("event") == "tensor" and row.get("name") == "lattice":
            data = (directory / row["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("lattice payload changed after verification")
            row["canonical_sha256"] = canonical_lattice(data, row["rows"], row["width"])
    return rows


def fit_ledger(row, trace, native_context=262144):
    context, depth = row["context"], row["depth"]
    if context < 2304 or context % 256 or native_context < context or native_context % 256:
        raise ValueError("fit ledger requires padded context beyond the bounded draft window")
    snapshots = {}
    for event in trace:
        if event.get("event") == "memory":
            snapshots[event["role"]] = event
    target, draft = snapshots["target"], snapshots["draft"]
    if len(target["devices"]) != 1:
        raise ValueError("fit ledger requires the single-device control profile")
    device = target["devices"][0]
    name, total = device["name"], device["total"]
    def buffer_bytes(snapshot, field):
        return sum(entry[field] for entry in snapshot["buffers"] if entry["buffer"] == name)
    def component_bytes(component):
        return sum(entry["bytes"] for entry in target["components"]
                   if entry["buffer"] == name and entry["name"] == component)
    weights = buffer_bytes(target, "model")+buffer_bytes(draft, "model")
    kv = component_bytes("target_kv")
    recurrent = component_bytes("target_recurrent")
    draft_kv = buffer_bytes(draft, "context")
    if (not kv or kv % context or not recurrent or recurrent % (depth+1)
            or not draft_kv or not weights or not total):
        raise ValueError("invalid measured cache/weight ledger")
    full_kv = (kv//context)*native_context
    lower = weights+full_kv+recurrent+draft_kv
    current_recurrent = recurrent//(depth+1)
    return {"native_context": native_context, "device_total_bytes": total,
            "ordinary_lower_bound_bytes": lower, "target_full_history_kv_bytes": full_kv,
            "ordinary_full_context_fits": False if lower > total else None,
            "streamed_parent_upper_bound_before_scratch_bytes": max(0, total-weights-current_recurrent),
            "target_elastic_upper_bound_before_scratch_bytes": max(0, total-weights-current_recurrent-draft_kv),
            "streamed_fit_qualified": False,
            "excluded": ["compute", "output/features", "rollback staging", "native/library resources"]}


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
    env = SWEEP.server_environment(getattr(args,"uvm",False))
    # An inherited profiler would invalidate the uninstrumented control.
    env.pop("LD_PRELOAD", None)
    env.pop("DFLASH_CONTROL_TRACE", None)
    env.pop("DFLASH_CONTROL_CAPTURE", None)
    env.pop("LLAMA_RS_HOST_SPILL", None)
    env.pop("LLAMA_RS_STAGED_PUBLICATION", None)
    if getattr(args, "rs_host_spill", False):
        env["LLAMA_RS_HOST_SPILL"] = "1"
        env["LLAMA_RS_STAGED_PUBLICATION"] = "1"
    if mode != "timing":
        env["LD_PRELOAD"] = str(args.trace_library)
        env["DFLASH_CONTROL_TRACE"] = str(directory)
        env["DFLASH_CONTROL_CAPTURE"] = "1" if mode == "capture" else "0"
    if mode == "nsight":
        command = [str(args.nsys), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--cuda-graph-trace=node",
                   "--cuda-memory-usage=true", "--output", str(directory / "native"), *command]
    row = {"status": "failed", "context": context, "depth": depth, "mode": mode,"arena_mib":args.arena_mib,
           "command": command, "uvm": getattr(args,"uvm",False),
           "numerical_control_only":getattr(args,"uvm",False),"log": str(log_path), "directory": str(directory)}
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
            count = prompt_count(args,context,depth)
            if count < 256 or len(token_cache["tokens"]) < count:
                raise ValueError("insufficient context/corpus for ordinary control")
            result = stream_completion(url+"/completion", {"prompt": token_cache["tokens"][:count],
                "n_predict": args.decode_tokens, "temperature": 0, "seed": 123, "ignore_eos": True,
                "cache_prompt": False, "stream": True, "return_tokens": True}, timeout=args.request_timeout)
            validate_result(result, count, args.decode_tokens)
            row.update(status="ok", prompt_tokens=count, decode_tokens=args.decode_tokens, result=result)
        except Exception as error:
            row["error"] = str(error)
        finally:
            if process is not None:
                stop_process(process)
    log_text=log_path.read_text(errors="replace")
    if row["status"] == "ok":
        try:
            SWEEP.validate_execution_log(log_text)
        except RuntimeError as error:
            row.update(status="failed",error=str(error))
    if row["status"] != "ok":
        row["status"] = failure_kind(log_text)
    elif mode == "capture":
        try:
            trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
            row["trace"] = summarize_trace(trace, depth,max_injection_rows=effective_microbatch(args,context))
            row["trace"]["verified_payloads"] = validate_payloads(directory, trace)
            if context >= 2304 and not args.arena_mib and not args.rs_host_spill:
                row["fit_ledger"] = fit_ledger(row, trace)
            if args.arena_mib:
                row["declared_arena_bytes"] = args.arena_mib*1024*1024
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
    parser.add_argument("--prompt-headroom",type=int,
                        help="positions beyond decode tokens; use 15 for matched DFlash D=3/7 and target/MTP comparisons")
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--ubatch-size",type=int,default=256)
    parser.add_argument("--verbosity",type=int,default=3,
                        help="use 5 for allocator diagnostics in a GGML_ALLOCATOR_DEBUG build; not for timing")
    parser.add_argument("--target-k",choices=KV_TYPES,default="q8_0")
    parser.add_argument("--target-v",choices=KV_TYPES,default="q4_0")
    parser.add_argument("--draft-k", choices=KV_TYPES, default="f16")
    parser.add_argument("--draft-v", choices=KV_TYPES, default="f16")
    parser.add_argument("--prefill-text", type=Path, default=ROOT/"benchmarks/data/online-articles-262144-words.txt")
    parser.add_argument("--trace-library", type=Path)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--compare-to", type=Path,
                        help="require byte-exact captured data against matching independent native controls")
    parser.add_argument("--uvm",action="store_true",help="native numerical control only; not a physical-fit or speed control")
    parser.add_argument("--compare-numerical-only",action="store_true",
                        help="allow UVM to differ in a matching numerical comparison; keep all other profile checks")
    parser.add_argument("--port", type=int, default=12351)
    parser.add_argument("--arena-mib", type=int, default=0,
                        help="qualify streamed execution in this declared arena; zero uses native storage")
    parser.add_argument("--auto-max-arena",action="store_true",
                        help="find each context/depth's largest physically fitting arena, then qualify the full workload")
    parser.add_argument("--rs-host-spill", action="store_true",
                        help="use the same two-slot staged rollback in the native control to keep it physically fitting")
    parser.add_argument("--request-timeout", type=float, default=1800,
                        help="completion timeout in seconds, including long-context prefill")
    args = parser.parse_args()
    contexts = [int(value) for value in args.contexts.split(",")]
    depths = [int(value) for value in args.depths.split(",")]
    args.max_depth=max(depths)
    if args.auto_max_arena and args.arena_mib < 1:
        parser.error("--auto-max-arena needs a positive --arena-mib starting point")
    if args.uvm and (args.arena_mib or args.auto_max_arena):
        parser.error("UVM is only allowed for independent native numerical controls")
    if args.compare_numerical_only and not args.compare_to:
        parser.error("--compare-numerical-only needs --compare-to")
    if (args.arena_mib < 0 or args.decode_tokens < 8 or args.prompt_tokens < 256 or not 1 <= args.port <= 65535
            or not math.isfinite(args.request_timeout) or not 1 <= args.request_timeout <= 43200):
        parser.error("invalid token or port bounds")
    for context in contexts:
        for depth in depths:
            server_command(args, context, depth)
            prompt_count(args,context,depth)
    for name in ("target", "draft", "draft_config", "server", "prefill_text", "trace_library", "nsys"):
        value = getattr(args, name)
        if value is not None:
            if not value.is_file():
                parser.error(f"missing input: {value}")
            setattr(args, name, value.resolve())
    if args.nsys and not args.trace_library:
        parser.error("Nsight phase attribution requires --trace-library")
    if args.compare_to and not args.trace_library:
        parser.error("matching controls require --trace-library")
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
    controls = []
    control_artifacts = {}
    if args.compare_to:
        controls = [json.loads(line) for line in (args.compare_to/"results.jsonl").read_text().splitlines()]
        control_artifacts = read_json(args.compare_to/"artifacts.json")
    with path.open("x") as output:
        for context in contexts:
            for depth in depths:
                arena,qualified=auto_arena_for_point(args,context,depth,token_cache)
                args.arena_mib=arena
                for mode in modes:
                    started = time.monotonic()
                    row = qualified if qualified is not None and mode == "timing" else run_point(args, context, depth, token_cache, mode)
                    if args.compare_to and row["status"] == "ok" and mode == "capture":
                        try:
                            matches = [r for r in controls if r.get("context") == context and r.get("depth") == depth
                                       and r.get("mode") == "capture"]
                            if len(matches) != 1:
                                raise ValueError("missing or ambiguous independent control")
                            control = matches[0]
                            row["comparison"] = compare_control_rows(row, control, report, control_artifacts,
                                load_control_trace(Path(row["directory"])), load_control_trace(Path(control["directory"])),
                                allow_uvm_difference=args.compare_numerical_only)
                        except Exception as error:
                            row.update(status="failed", error=str(error))
                    if qualified is None or mode != "timing":
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
