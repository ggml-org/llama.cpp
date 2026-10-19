#!/usr/bin/env python3
"""Offline native-versus-streamed Gemma target and assistant server qualification."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/server-ab"))
from server_ab import request_json, stream_completion, wait_for_health

KV_TYPES=("f32","f16","bf16","q8_0","q5_0","q5_1","q4_0","q4_1")


def server_command(args, streamed, depth, port):
    key=getattr(args,"cache_type_k","q8_0")
    value=getattr(args,"cache_type_v","q4_0")
    command = [str(args.server.resolve()), "--model", str(args.model.resolve()),
               "--host", "127.0.0.1", "--port", str(port),
               "--offline", "--fit", "off", "--parallel", "1", "--ctx-size", str(args.context),
               "--batch-size", str(args.batch_size), "--ubatch-size", str(args.ubatch_size),
               "--n-gpu-layers", "999", "--flash-attn", "on", "--cache-type-k", key, "--cache-type-v", value,
               "--no-mmproj", "--no-context-shift",
               "--cache-ram", "2048", "--ctx-checkpoints", "4", "--no-webui", "--log-verbosity", "3"]
    if depth:
        command += ["--model-draft", str(args.assistant.resolve()), "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", str(depth), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0", "--spec-draft-ngl", "999"]
    if streamed:
        command += ["--kv-stream-arena-mib", str(args.arena_mib)]
    elif depth:
        # Native startup uses independent draft defaults; match the target for this shared-KV control.
        command += ["--spec-draft-type-k", key, "--spec-draft-type-v", value]
    return command


def draft_counts(result,depth,label):
    timing=result.get("timings",{})
    generated,accepted=timing.get("draft_n",0),timing.get("draft_n_accepted",0)
    if type(generated) is not int or type(accepted) is not int or generated<0 or accepted<0 or accepted>generated:
        raise RuntimeError("invalid Gemma draft/acceptance counts: "+label)
    if depth and not generated:
        raise RuntimeError("Gemma assistant did not draft: "+label)
    if not depth and (generated or accepted):
        raise RuntimeError("target-only Gemma unexpectedly drafted: "+label)
    return generated,accepted


@contextlib.contextmanager
def server(args, streamed, depth):
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    label = f"{'streamed' if streamed else 'native'}-mtp{depth}"
    log_path = args.output / (label + ".log")
    command = server_command(args, streamed, depth, port)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GGML_", "LLAMA_ARG_"))}
    (args.output / (label + "-command.json")).write_text(json.dumps({"command": command, "uvm": False}, indent=2))
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=os.name == "posix")
        try:
            base = f"http://127.0.0.1:{port}"
            wait_for_health(base, timeout=180, process=process)
            yield base
        finally:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=20)


def cancel_after_first_output(base, payload):
    body = json.dumps(dict(payload, stream=True, n_predict=256)).encode()
    request = urllib.request.Request(base + "/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError("cancelled generation failed: " + str(event))
            if event.get("tokens") or event.get("content"):
                return
    raise RuntimeError("cancellation did not observe any generated output")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with args.corpus.open(encoding="utf-8") as corpus:
        text = corpus.read(262144)
    results = []
    for depth in args.depths:
        controls = {}
        draft_controls = {}
        for streamed in (False, True):
            with server(args, streamed, depth) as base:
                tokens = request_json(base + "/tokenize", {"content": text, "add_special": True})["tokens"]
                if len(tokens) < args.prefill + 64:
                    raise RuntimeError("qualification corpus is too short")
                prompt = tokens[:args.prefill]
                payload = {"prompt": prompt, "n_predict": args.predict, "temperature": 0, "seed": 123,
                           "ignore_eos": True, "cache_prompt": False, "return_tokens": True, "stream": True}
                cases = [("fresh", payload), ("cached", dict(payload, cache_prompt=True)),
                         ("changed", dict(payload, prompt=tokens[args.prefill:args.prefill+64]))]
                for label, body in cases:
                    result = stream_completion(base + "/completion", body, timeout=600)
                    if result.get("timings", {}).get("predicted_n") != args.predict or len(result.get("tokens", [])) != args.predict:
                        raise RuntimeError("incomplete Gemma completion: " + label)
                    counts=draft_counts(result,depth,label)
                    if streamed:
                        if controls[label] != result["tokens"]:
                            raise RuntimeError(f"native/streamed token mismatch: depth={depth} case={label}")
                        if draft_controls[label]!=counts:
                            raise RuntimeError(f"native/streamed draft/acceptance mismatch: depth={depth} case={label}")
                    else:
                        controls[label] = result["tokens"]
                        draft_controls[label]=counts
                    results.append({"depth": depth, "streamed": streamed, "case": label, "result": result})
                    (args.output / "results.json").write_text(json.dumps(results, indent=2))
                cancel_after_first_output(base, payload)
                result = stream_completion(base + "/completion", dict(payload, prompt=tokens[args.prefill:args.prefill+64]), timeout=600)
                if len(result.get("tokens", [])) != args.predict or result.get("timings", {}).get("predicted_n") != args.predict:
                    raise RuntimeError("incomplete post-cancellation request")
                if result["tokens"] != controls["changed"]:
                    raise RuntimeError("cancellation changed a fresh request")
                counts=draft_counts(result,depth,"cancel-recovery")
                if streamed and counts!=draft_controls["cancel-recovery"]:
                    raise RuntimeError(f"native/streamed draft/acceptance mismatch: depth={depth} case=cancel-recovery")
                if not streamed:
                    draft_controls["cancel-recovery"]=counts
                results.append({"depth": depth, "streamed": streamed, "case": "cancel-recovery", "result": result})
                (args.output / "results.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/data/online-articles-262144-words.txt")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--prefill", type=int, default=1333)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--arena-mib", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--cache-type-k",choices=KV_TYPES,default="q8_0")
    parser.add_argument("--cache-type-v",choices=KV_TYPES,default="q4_0")
    parser.add_argument("--depths", default="1,2,3")
    args = parser.parse_args()
    args.depths = [int(value) for value in args.depths.split(",")]
    if not args.depths or any(depth not in (0, 1, 2, 3) for depth in args.depths) or len(set(args.depths)) != len(args.depths):
        parser.error("depths must be distinct values from 0,1,2,3; 0 disables the assistant")
    if min(args.context, args.prefill, args.predict, args.arena_mib, args.batch_size, args.ubatch_size) <= 0:
        parser.error("sizes must be positive")
    if args.prefill + args.predict + 4 >= args.context or min(args.batch_size, args.ubatch_size) < max(args.depths) + 1:
        parser.error("context and microbatch must cover the prompt, outputs and verification")
    run(args)


if __name__ == "__main__":
    main()
