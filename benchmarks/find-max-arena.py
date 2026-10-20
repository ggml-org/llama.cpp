#!/usr/bin/env python3
"""Find a physical arena limit by executing a configured text or image workload.

Does not stop containers, download models, enable UVM, or reserve arbitrary VRAM.
The result is workload-specific, not a guarantee for other image/prompt shapes.
"""

import argparse
import base64
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/server-ab"))
from server_ab import request_json

SPEC = importlib.util.spec_from_file_location("arena_sweep", ROOT / "benchmarks/run-fixed-span-sweep.py")
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)
MARKER = "<__media__>"
REMOVED_ENV = ("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_CUDA_PREFER_MODEL_WEIGHTS", "GGML_CUDA_PREFER_KV_HOST",
    "GGML_CUDA_KV_ACCESSED_BY_GPU", "GGML_CUDA_DISABLE_GRAPHS", "LLAMA_RS_HOST_SPILL", "LLAMA_RS_STAGED_PUBLICATION")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT/"build-v2/bin/llama-server")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context-size", type=int, default=262144)
    parser.add_argument("--cache-type-k", choices=SWEEP.KV_TYPES, default="q8_0")
    parser.add_argument("--cache-type-v", choices=SWEEP.KV_TYPES, default="q4_0")
    parser.add_argument("--spec-mode", choices=("none", "mtp", "dflash"), default="none")
    parser.add_argument("--draft-length", type=int)
    parser.add_argument("--model-draft", type=Path)
    parser.add_argument("--mmproj", type=Path, help="configure the deferred projector; images are optional")
    parser.add_argument("--image", type=Path, action="append", default=[], help="add image-workload validation; repeat for a batch")
    parser.add_argument("--prompt-file", type=Path, help="literal image-request prompt; one <__media__> per image")
    parser.add_argument("--prefill-text", type=Path, default=ROOT/"benchmarks/data/online-articles-262144-words.txt")
    parser.add_argument("--prompt-tokens", type=int, help="text-only prefill length; default fills context minus outputs and headroom")
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
    if args.draft_length is None:
        args.draft_length = {"none":0, "mtp":3, "dflash":7}[args.spec_mode]
    if not (args.spec_mode == "none" and args.draft_length == 0 or
            args.spec_mode == "mtp" and 1 <= args.draft_length <= 3 or
            args.spec_mode == "dflash" and 1 <= args.draft_length <= 7):
        parser.error("target-only uses length 0; MTP supports 1-3; DFlash supports 1-7")
    if args.spec_mode == "dflash" and not args.model_draft:
        parser.error("--model-draft is required for DFlash")
    if args.spec_mode == "none" and args.model_draft:
        parser.error("--model-draft requires MTP or DFlash")
    if args.image and not args.mmproj or args.mmproj and args.spec_mode == "dflash":
        parser.error("--image requires --mmproj; vision+DFlash is unsupported")
    if args.prompt_file and not args.image or args.prompt_tokens is not None and args.image:
        parser.error("--prompt-file requires image validation; --prompt-tokens is for text probes")
    if (not 512 <= args.context_size <= 262144 or not 8 <= args.decode_tokens < args.context_size-15 or
            not 1 <= args.batch_size <= 2**31-1 or not 0 <= args.ubatch_size <= 2**31-1 or
            not 1 <= args.initial_arena_mib <= 1048576 or not 1 <= args.port <= 65535 or
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
        "--spec-type", {"none":"none", "mtp":"draft-mtp", "dflash":"draft-dflash"}[args.spec_mode],
        "--host", "127.0.0.1", "--port", str(args.port), "--offline", "--no-context-shift", "--cache-ram", "0", "-lv", "3"]
    if args.draft_length:
        command += ["--spec-draft-n-max", str(args.draft_length), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"]
    if args.model_draft:
        command += ["--model-draft", str(args.model_draft), "--spec-draft-ngl", "999"]
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
    if args.spec_mode != "none" and not timing.get("draft_n", 0) > 0:
        raise RuntimeError("speculative workload completed without generating drafts")


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
                expected = args.prompt_tokens or args.context_size-args.decode_tokens-15
                corpus = bounded_bytes(args.prefill_text).decode("utf-8")
                tokens = request_json(url+"/tokenize", {"content":corpus, "add_special":True}, timeout=args.request_timeout)["tokens"]
                if len(tokens) < expected:
                    raise ValueError("corpus is too short; provide more text or set --prompt-tokens")
                prompt, images = tokens[:expected], []
            result = request_json(url+"/completion", payload(args,prompt,images), timeout=args.request_timeout)
            validate_result(args, result, expected)
        except Exception as error:
            failure = error
        finally:
            SWEEP.stop_process(process)
    text = log_path.read_text(errors="replace")
    if failure is None:
        try:
            SWEEP.validate_execution_log(text)
        except RuntimeError as error:
            failure = error
    minimum = SWEEP.arena_minimum_mib(text)
    if minimum is not None:
        raise SWEEP.ArenaTooSmallError(minimum)
    if failure and not SWEEP.is_arena_capacity_failure(text):
        raise RuntimeError(f"probe failed; not a physical OOM: {failure}; see {log_path}") from failure
    return {"arena_mib":candidate, "fits":failure is None, "log":str(log_path), "command":command,
        "timings":result.get("timings"), "error":str(failure) if failure else None}


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
    report = {"arena_mib":maximum, "initial_arena_mib":args.initial_arena_mib,
        "next_mib_rejected":maximum+1 if next_rejected else None,
        "command":server_command(args,maximum), "uvm":False, "reserve_mib":0,
        "context_size":args.context_size, "cache_type_k":args.cache_type_k, "cache_type_v":args.cache_type_v,
        "spec_mode":args.spec_mode, "draft_length":args.draft_length, "images":[str(path) for path in args.image],
        "model_draft":str(args.model_draft) if args.model_draft else None,
        "prompt_file":str(args.prompt_file) if args.prompt_file else None,
        "projector_configured":bool(args.mmproj), "vision_execution_validated":bool(args.image),
        "prompt_tokens":None if args.image else args.prompt_tokens or args.context_size-args.decode_tokens-15,
        "decode_tokens":args.decode_tokens,
        "scope":"Tested workload only; different image dimensions/counts, prompt shapes, other processes or driver allocations can change fit."}
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
