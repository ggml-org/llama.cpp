#!/usr/bin/env python3
"""Find a physical arena limit with runtime probes of a text or image workload.

Does not stop containers, download models, enable UVM, or reserve arbitrary VRAM.
The result is workload-specific, not a guarantee for other image/prompt shapes.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/server-ab"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from server_ab import request_json
from run_dflash_controls import ngram_command, ngram_settings, speculative_source_stats, validate_mixed_sources

SPEC = importlib.util.spec_from_file_location("arena_sweep", ROOT / "benchmarks/run-fixed-span-sweep.py")
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)
MARKER = "<__media__>"
REMOVED_ENV = ("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_CUDA_PREFER_MODEL_WEIGHTS", "GGML_CUDA_PREFER_KV_HOST",
    "GGML_CUDA_KV_ACCESSED_BY_GPU", "GGML_CUDA_DISABLE_GRAPHS", "LLAMA_RS_HOST_SPILL", "LLAMA_RS_STAGED_PUBLICATION",
    "LD_PRELOAD", "DFLASH_CONTROL_TRACE", "DFLASH_CONTROL_CAPTURE")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT/"build-v2/bin/llama-server")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context-size", type=int, default=262144)
    parser.add_argument("--cache-type-k", choices=SWEEP.KV_TYPES, default="q8_0")
    parser.add_argument("--cache-type-v", choices=SWEEP.KV_TYPES, default="q4_0")
    parser.add_argument("--spec-mode", choices=("none", "mtp", "dflash", "dflash-k4v"), default="none")
    parser.add_argument("--ngram-n", type=int, help="mixed K4V lookup length (default: 12)")
    parser.add_argument("--ngram-m", type=int, help="mixed K4V maximum proposals (default: 7; supported: 1-7)")
    parser.add_argument("--ngram-min-hits", type=int, help="mixed K4V minimum hits (default: 2)")
    parser.add_argument("--draft-length", type=int)
    parser.add_argument("--model-draft", type=Path)
    parser.add_argument("--mmproj", type=Path, help="configure the deferred projector; images are optional")
    parser.add_argument("--image", type=Path, action="append", default=[], help="add image-workload validation; repeat for a batch")
    parser.add_argument("--prompt-file", type=Path, help="literal image-request prompt; one <__media__> per image")
    parser.add_argument("--prefill-text", type=Path, default=ROOT/"benchmarks/data/online-articles-262144-words.txt")
    parser.add_argument("--prompt-tokens", type=int, help="text-only prefill length; default fills context minus outputs and headroom")
    parser.add_argument("--probe-mode", choices=("full", "preflight"), default="full",
                        help="full: test the requested workload for every candidate; preflight: use bounded text history plus real decoding")
    parser.add_argument("--validate-selected", action="store_true",
                        help="after a preflight search, run the full requested workload once at the selected arena; fail if it does not fit")
    parser.add_argument("--copy-probe", action="store_true",
                        help="also validate a separate 32-output copying path; always enabled for mixed K4V, optional for matched non-mixed sizing")
    parser.add_argument("--no-webui", action="store_true", help="disable the test server's web UI")
    parser.add_argument("--verbosity", type=int, default=3, help="server log threshold 0-5; route proofs require at least 4")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--image-min-tokens", type=int, default=64)
    parser.add_argument("--image-max-tokens", type=int, default=4096)
    parser.add_argument("--initial-arena-mib", type=int, default=2240, help="optional initial search guess, not the final arena size or a safety reserve")
    parser.add_argument("--port", type=int, default=1249)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, default=ROOT/"benchmarks/results/arena-fit")
    args = parser.parse_args(argv)
    if args.validate_selected and args.probe_mode != "preflight":
        parser.error("--validate-selected requires --probe-mode preflight")
    if args.draft_length is None:
        args.draft_length = {"none":0, "mtp":3, "dflash":7, "dflash-k4v":7}[args.spec_mode]
    if not (args.spec_mode == "none" and args.draft_length == 0 or
            args.spec_mode == "mtp" and 1 <= args.draft_length <= 3 or
            args.spec_mode in ("dflash", "dflash-k4v") and 1 <= args.draft_length <= 7):
        parser.error("target-only uses length 0; MTP supports 1-3; DFlash supports 1-7")
    try:
        ngram_settings(args)
    except ValueError as error:
        parser.error(str(error))
    if args.spec_mode in ("dflash", "dflash-k4v") and not args.model_draft:
        parser.error("--model-draft is required for DFlash")
    if args.spec_mode == "none" and args.model_draft:
        parser.error("--model-draft requires MTP or DFlash")
    if args.image and not args.mmproj or args.mmproj and args.spec_mode in ("dflash", "dflash-k4v"):
        parser.error("--image requires --mmproj; vision+DFlash is unsupported")
    if args.prompt_file and not args.image or args.prompt_tokens is not None and args.image:
        parser.error("--prompt-file requires image validation; --prompt-tokens is for text probes")
    if args.copy_probe and args.image:
        parser.error("--copy-probe is a text-only resource check; omit it for image validation")
    if (not 512 <= args.context_size <= 262144 or not 8 <= args.decode_tokens < args.context_size-15 or
            not 1 <= args.batch_size <= 2**31-1 or not 0 <= args.ubatch_size <= 2**31-1 or
            not 1 <= args.initial_arena_mib <= 1048576 or not 1 <= args.port <= 65535 or not 0 <= args.verbosity <= 5 or
            not math.isfinite(args.request_timeout) or not 1 <= args.request_timeout <= 43200 or
            not 1 <= args.image_min_tokens <= args.image_max_tokens or
            (args.image and args.image_max_tokens > args.context_size) or len(args.image) > 32):
        parser.error("invalid context, output, batch, image, arena, timeout or port bounds")
    if args.prompt_tokens is not None and not 1 <= args.prompt_tokens <= args.context_size-args.decode_tokens-15:
        parser.error("prefill must leave room for outputs and 15 speculative headroom positions")
    return args


def server_command(args, arena_mib):
    command = [str(args.server), "--model", str(args.model), "--ctx-size", str(args.context_size),
        "--batch-size", str(args.batch_size), "--ubatch-size", str(args.ubatch_size), "--parallel", "1",
        "--n-gpu-layers", "999", "--flash-attn", "on", "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v, "--shared-device-memory-mib", str(arena_mib), "--fit", "off",
        "--spec-type", {"none":"none", "mtp":"draft-mtp", "dflash":"draft-dflash", "dflash-k4v":"ngram-map-k4v,draft-dflash"}[args.spec_mode],
        "--host", "127.0.0.1", "--port", str(args.port), "--offline", "--no-context-shift", "--cache-ram", "0",
        "-lv", str(max(args.verbosity, 4 if args.spec_mode == "dflash-k4v" or args.copy_probe else 0))]
    if args.no_webui:
        command.append("--no-webui")
    if args.draft_length:
        command += ["--spec-draft-n-max", str(args.draft_length), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"]
    if args.model_draft:
        command += ["--model-draft", str(args.model_draft), "--spec-draft-ngl", "999"]
    command += ngram_command(args)
    if args.mmproj:
        command += ["--mmproj", str(args.mmproj), "--image-min-tokens", str(args.image_min_tokens),
            "--image-max-tokens", str(args.image_max_tokens)]
    else:
        command.append("--no-mmproj")
    return command


def environment():
    env = SWEEP.server_environment(False)
    for name in list(env):
        if name.startswith("LLAMA_ARG_") or name in REMOVED_ENV:
            env.pop(name)
    env["LLAMA_MEDIA_MARKER"] = MARKER
    return env


def bounded_bytes(path, maximum=64*1048576):
    with path.open("rb") as source:
        data = source.read(maximum+1)
    if not data or len(data) > maximum:
        raise ValueError(f"input must contain 1 byte to {maximum} bytes: {path}")
    return data


def requested_text_tokens(args):
    return args.prompt_tokens or args.context_size-args.decode_tokens-15


def probe_text_tokens(args):
    requested = requested_text_tokens(args)
    if args.probe_mode == "preflight":
        # Exercise wide prefill and decode transitions without filling the whole history.
        return min(requested, max(2048, 2*max(args.batch_size, args.ubatch_size)))
    return requested


def payload(args, prompt, images):
    if images:
        if not isinstance(prompt, str) or prompt.count(MARKER) != len(images):
            raise ValueError("image prompt needs exactly one media marker per image")
        prompt = {"prompt_string":prompt, "multimodal_data":images}
    return {"prompt":prompt, "n_predict":args.decode_tokens, "temperature":0, "seed":123,
        "ignore_eos":True, "cache_prompt":False, "return_tokens":True, "stream":False}


def validate_result(args, result, expected_prompt):
    timing = result.get("timings", {})
    if len(result.get("tokens", [])) != args.decode_tokens or timing.get("predicted_n") != args.decode_tokens:
        raise RuntimeError("incomplete decode response")
    count = timing.get("prompt_n")
    if type(count) is not int or count <= 0 or count+args.decode_tokens+15 > args.context_size:
        raise RuntimeError("request exceeded the configured context/headroom or omitted prompt timing")
    if expected_prompt is not None and count != expected_prompt:
        raise RuntimeError("incomplete prefill response")
    if args.spec_mode != "none" and (type(timing.get("draft_n")) is not int or timing["draft_n"] <= 0):
        raise RuntimeError("speculative workload completed without generating drafts")


def mixed_route_probe(args, url, prompt, candidate):
    settings = ngram_settings(args)
    if settings is None and not args.copy_probe:
        return None
    count = 32
    length = min(len(prompt), args.context_size-count-15)
    tail = 2*settings["ngram_n"]+settings["ngram_m"]+2 if settings else 33
    if length < tail:
        raise ValueError("mixed route proof needs at least 2*N+M+2 prompt tokens with decode headroom")
    ids = request_json(url+"/tokenize", {"content":"B", "add_special":False}, timeout=args.request_timeout)["tokens"]
    if len(ids) != 1 or type(ids[0]) is not int or not 0 <= ids[0] <= 2**31-1:
        raise ValueError("mixed route proof requires a single ordinary token for B")
    proof_prompt = list(prompt[:length])
    proof_prompt[-tail:] = ids*tail
    config = argparse.Namespace(**vars(args))
    config.decode_tokens = count
    request = payload(config, proof_prompt, [])
    request["logit_bias"] = [[ids[0], 1000]]
    result = request_json(url+"/completion", request, timeout=args.request_timeout)
    path = args.output/f"arena-{candidate}-mixed-proof.json"
    path.write_text(json.dumps({"payload":request, "result":result}, indent=2)+"\n")
    validate_result(config, result, length)
    if result["tokens"] != ids*count:
        raise RuntimeError("mixed route proof violated the target sampling constraint")
    return {"response":str(path), "response_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompt_tokens":length, "decode_tokens":count, "ngram":settings, "timings":result["timings"],
            "scope":"Separate target-bias-constrained text path/resource check; mixed mode additionally proves both sources. Not performance or model quality evidence"}


def probe(args, candidate):
    try:
        with socket.socket() as available:
            if os.name == "posix":
                available.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            available.bind(("127.0.0.1", args.port))
    except OSError as error:
        raise RuntimeError(f"port {args.port} is unavailable; choose --port for this finder") from error
    log_path = args.output/f"arena-{candidate}.log"
    command = server_command(args, candidate)
    failure = None
    result = {}
    mixed_proof = None
    response_path = None
    prompt_sha256 = None
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment())
        try:
            url = f"http://127.0.0.1:{args.port}"
            SWEEP.wait_ready(url, process, log_path)
            expected = None
            if args.image:
                images = [base64.b64encode(bounded_bytes(path)).decode("ascii") for path in args.image]
                prompt = bounded_bytes(args.prompt_file).decode("utf-8") if args.prompt_file else \
                    "Describe these images: " + " ".join([MARKER]*len(images))
            else:
                expected = probe_text_tokens(args)
                corpus = bounded_bytes(args.prefill_text).decode("utf-8")
                tokens = request_json(url+"/tokenize", {"content":corpus, "add_special":True}, timeout=args.request_timeout)["tokens"]
                if len(tokens) < expected:
                    raise ValueError("corpus is too short; provide more text or set --prompt-tokens")
                prompt, images = tokens[:expected], []
            result = request_json(url+"/completion", payload(args,prompt,images), timeout=args.request_timeout)
            response_path = args.output/f"arena-{candidate}-response.json"
            response_path.write_text(json.dumps(result, indent=2)+"\n")
            if not args.image:
                prompt_sha256 = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            validate_result(args, result, expected)
            mixed_proof = mixed_route_probe(args, url, prompt, candidate)
        except Exception as error:
            failure = error
        finally:
            SWEEP.stop_process(process)
    text = log_path.read_text(errors="replace")
    if re.search(r"illegal memory access|misaligned address|device-side assert|unspecified launch failure", text, re.IGNORECASE):
        raise RuntimeError(f"probe reported a device fault, not an arena capacity boundary; see {log_path}")
    if failure is None:
        try:
            SWEEP.validate_execution_log(text)
            if args.spec_mode == "dflash-k4v":
                if mixed_proof is None:
                    raise RuntimeError("missing completed mixed route probe")
                mixed_proof["source_stats"] = validate_mixed_sources(text,
                    generated_tokens=result["timings"]["draft_n"]+mixed_proof["timings"]["draft_n"],
                    accepted_tokens=result["timings"]["draft_n_accepted"]+mixed_proof["timings"]["draft_n_accepted"])
            elif mixed_proof is not None:
                mixed_proof["source_stats"] = speculative_source_stats(text)
                for source_key, timing_key in (("generated_tokens", "draft_n"), ("accepted_tokens", "draft_n_accepted")):
                    expected = result["timings"].get(timing_key, 0)+mixed_proof["timings"].get(timing_key, 0)
                    if sum(row[source_key] for row in mixed_proof["source_stats"].values()) != expected:
                        raise ValueError("copy-path source totals differ from completed request counters")
        except (RuntimeError, ValueError) as error:
            failure = error
    minimum = SWEEP.arena_minimum_mib(text)
    if minimum is not None:
        raise SWEEP.ArenaTooSmallError(minimum)
    if failure and not SWEEP.is_arena_capacity_failure(text):
        raise RuntimeError(f"probe failed; not a physical OOM: {failure}; see {log_path}") from failure
    return {"arena_mib":candidate, "fits":failure is None, "log":str(log_path), "command":command,
        "log_sha256":hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "response":str(response_path) if response_path else None,
        "response_sha256":hashlib.sha256(response_path.read_bytes()).hexdigest() if response_path else None,
        "prompt_ids_sha256":prompt_sha256, "mixed_proof":mixed_proof,
        "probe_mode":args.probe_mode, "timings":result.get("timings"), "error":str(failure) if failure else None}


def fit(args):
    records = []
    def checked(candidate):
        try:
            record = probe(args,candidate)
        except SWEEP.ArenaTooSmallError as error:
            record = {"arena_mib":candidate,"fits":False,"minimum_mib":error.minimum_mib}
            with (args.output/"probes.jsonl").open("a") as output:
                output.write(json.dumps(record)+"\n")
            raise
        records.append(record)
        with (args.output/"probes.jsonl").open("a") as output:
            output.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)
        return record["fits"]
    maximum = SWEEP.find_max_arena_mib(args.initial_arena_mib, checked)
    next_rejected = any(row["arena_mib"] == maximum+1 and not row["fits"] for row in records)
    selected = next((row for row in records if row["arena_mib"] == maximum and row.get("fits")), None)
    runtime_validated = selected is not None and selected.get("timings") is not None
    selected_validation = None
    if args.validate_selected:
        full = argparse.Namespace(**vars(args))
        full.probe_mode, full.validate_selected = "full", False
        full.output = args.output/"selected-validation"
        full.output.mkdir(exist_ok=False)
        try:
            selected_validation = probe(full, maximum)
        except Exception as error:
            (full.output/"record.json").write_text(json.dumps({"arena_mib":maximum,"fits":False,"error":str(error)},indent=2)+"\n")
            raise RuntimeError(f"selected arena full workload validation failed: {error}; see {full.output}") from error
        (full.output/"record.json").write_text(json.dumps(selected_validation,indent=2)+"\n")
        if not selected_validation["fits"]:
            raise RuntimeError(f"selected arena failed full workload validation; no fit published; see {full.output}")
    full_validated = bool(selected_validation and selected_validation["fits"]) or runtime_validated and (args.probe_mode == "full" or bool(args.image))
    report = {"arena_mib":maximum, "initial_arena_mib":args.initial_arena_mib,
        "next_mib_rejected":maximum+1 if next_rejected else None,
        "command":server_command(args,maximum), "uvm":False, "reserve_mib":0,
        "context_size":args.context_size, "cache_type_k":args.cache_type_k, "cache_type_v":args.cache_type_v,
        "spec_mode":args.spec_mode, "draft_length":args.draft_length, "images":[str(path) for path in args.image],
        "ngram":ngram_settings(args), "mixed_routes_validated":args.spec_mode == "dflash-k4v" and bool(selected and selected.get("mixed_proof")),
        "copy_path_validated":bool(selected and selected.get("mixed_proof")), "headless":args.no_webui,
        "model_draft":str(args.model_draft) if args.model_draft else None,
        "prompt_file":str(args.prompt_file) if args.prompt_file else None,
        "projector_configured":bool(args.mmproj), "vision_execution_validated":bool(args.image) and runtime_validated,
        "probe_mode":args.probe_mode, "runtime_probe_validated":runtime_validated,
        "full_workload_validated":full_validated, "selected_validation":selected_validation,
        "prompt_tokens":None if args.image else probe_text_tokens(args),
        "requested_prompt_tokens":None if args.image else requested_text_tokens(args),
        "decode_tokens":args.decode_tokens,
        "scope":"Preflight executes bounded text history plus the requested decode, or the complete supplied image request. Full text history is validated only in full mode or by --validate-selected. Different image shapes, prompts, other processes or driver allocations can change fit."}
    removed = sorted(set(REMOVED_ENV) | {name for name in os.environ if name.startswith("LLAMA_ARG_")})
    report["environment"] = {name:None for name in removed}
    report["environment"]["LLAMA_MEDIA_MARKER"] = MARKER
    launch = ["env"] + [item for name in removed for item in ("-u",name)] + ["LLAMA_MEDIA_MARKER="+MARKER] + report["command"]
    report["shell_command"] = shlex.join(launch)
    (args.output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output/"command.txt").write_text(report["shell_command"]+"\n")
    return report


def main(argv=None):
    args = arguments(argv)
    for name in ("server", "model", "model_draft", "mmproj", "prompt_file", "prefill_text"):
        path = getattr(args,name)
        if path is not None:
            path = path.resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"not a file: {path}")
            setattr(args,name,path)
    args.image = [path.resolve(strict=True) for path in args.image]
    if any(not path.is_file() for path in args.image):
        raise ValueError("image paths must be files")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    print("Free the GPU before fitting. This tool does not stop containers. UVM is disabled for every probe.", flush=True)
    report = fit(args)
    print("\nSelected command (POSIX shell):\n"+report["shell_command"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
