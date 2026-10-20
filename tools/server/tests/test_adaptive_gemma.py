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


def slot_directory(args,streamed,depth):
    return (args.output/f"{'streamed' if streamed else 'native'}-mtp{depth}-slots").resolve()


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
    if getattr(args,"lifecycle",False):
        command += ["--slot-save-path",str(slot_directory(args,streamed,depth))]
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


def prompt_counts(result,length,cached,label):
    timing=result.get("timings",{})
    reused,processed=timing.get("cache_n"),timing.get("prompt_n")
    if type(reused) is not int or type(processed) is not int or reused<0 or processed<=0 or reused+processed!=length or (not cached and reused):
        raise RuntimeError("invalid Gemma prompt accounting: "+label)
    if cached and not reused:
        raise RuntimeError("Gemma repeated prompt did not produce a cache hit: "+label)
    return reused,processed


@contextlib.contextmanager
def server(args, streamed, depth, phase=None):
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    label = f"{'streamed' if streamed else 'native'}-mtp{depth}"
    if phase:
        label += "-"+phase
    if getattr(args,"lifecycle",False):
        slot_directory(args,streamed,depth).mkdir(parents=True,exist_ok=True)
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
        prompt_controls = {}
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
                    if getattr(args,"require_cache_hit",False):
                        accounting=prompt_counts(result,len(body["prompt"]),body["cache_prompt"],label)
                        if streamed and accounting!=prompt_controls[label]:
                            raise RuntimeError(f"native/streamed prompt accounting mismatch: depth={depth} case={label}")
                        prompt_controls[label]=accounting
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
                if getattr(args,"require_cache_hit",False):
                    accounting=prompt_counts(result,64,False,"cancel-recovery")
                    if streamed and accounting!=prompt_controls["cancel-recovery"]:
                        raise RuntimeError(f"native/streamed prompt accounting mismatch: depth={depth} case=cancel-recovery")
                    prompt_controls["cancel-recovery"]=accounting
                if streamed and counts!=draft_controls["cancel-recovery"]:
                    raise RuntimeError(f"native/streamed draft/acceptance mismatch: depth={depth} case=cancel-recovery")
                if not streamed:
                    draft_controls["cancel-recovery"]=counts
                results.append({"depth": depth, "streamed": streamed, "case": "cancel-recovery", "result": result})
                (args.output / "results.json").write_text(json.dumps(results, indent=2))
    return results


def run_lifecycle(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with args.corpus.open(encoding="utf-8") as corpus:
        text=corpus.read(16*1024*1024)
    results=[]
    actions=[]
    streamed_only=getattr(args,"streamed_only",False)
    for depth in args.depths:
        native={}
        for streamed in ((True,) if streamed_only else (False,True)):
            references={}
            saved_tokens=saved_bytes=0
            path=slot_directory(args,streamed,depth)/"lifecycle.bin"

            def completion(base,label,body,reference=None):
                result=stream_completion(base+"/completion",body,timeout=600)
                (args.output/"last-response.json").write_text(json.dumps({"depth":depth,"streamed":streamed,
                    "case":label,"prompt_tokens":len(body["prompt"]),"result":result},indent=2))
                if result.get("timings",{}).get("predicted_n")!=body["n_predict"] or len(result.get("tokens",[]))!=body["n_predict"]:
                    raise RuntimeError("incomplete Gemma lifecycle completion: "+label)
                counts=draft_counts(result,depth,label)
                accounting=prompt_counts(result,len(body["prompt"]),body["cache_prompt"],label)
                if reference and result["tokens"]!=references[reference]:
                    raise RuntimeError("Gemma lifecycle continuation token mismatch: "+label)
                if streamed and not streamed_only:
                    tokens,expected_counts,expected_prompt=native[label]
                    if result["tokens"]!=tokens:
                        raise RuntimeError("native/streamed lifecycle token mismatch: "+label)
                    if counts!=expected_counts or accounting!=expected_prompt:
                        raise RuntimeError("native/streamed lifecycle accounting mismatch: "+label)
                elif not streamed:
                    native[label]=(result["tokens"],counts,accounting)
                references[label]=result["tokens"]
                results.append({"depth":depth,"streamed":streamed,"case":label,"result":result,
                    "comparison":"streamed-self-reference" if streamed_only else "native-matched"})
                (args.output/"results.json").write_text(json.dumps(results,indent=2))
                print(json.dumps({"depth":depth,"streamed":streamed,"case":label,"cache_n":accounting[0],
                    "prompt_n":accounting[1],"outputs":len(result["tokens"])}),flush=True)

            def state_action(base,action,phase):
                nonlocal saved_tokens,saved_bytes
                response=request_json(base+"/slots/0?action="+action,
                    None if action=="erase" else {"filename":path.name},timeout=600,method="POST")
                if action=="save":
                    saved_tokens=response.get("n_saved")
                    saved_bytes=response.get("n_written")
                    if type(saved_tokens) is not int or not 0<saved_tokens<=args.context or type(saved_bytes) is not int or saved_bytes<=0 or not path.is_file() or path.stat().st_size!=saved_bytes:
                        raise RuntimeError("incomplete Gemma state save")
                elif action=="restore":
                    if response.get("n_restored")!=saved_tokens or response.get("n_read")!=saved_bytes or path.stat().st_size!=saved_bytes:
                        raise RuntimeError("incomplete Gemma state restore")
                elif type(response.get("n_erased")) is not int or response["n_erased"]<0:
                    raise RuntimeError("invalid Gemma state erase")
                actions.append({"depth":depth,"streamed":streamed,"phase":phase,"action":action,
                    "path":str(path),"response":response})
                (args.output/"state-actions.json").write_text(json.dumps(actions,indent=2))

            with server(args,streamed,depth) as base:
                tokens=request_json(base+"/tokenize",{"content":text,"add_special":True})["tokens"]
                if len(tokens)<args.prefill+64:
                    raise RuntimeError("qualification corpus is too short")
                high={"prompt":tokens[:args.prefill],"n_predict":args.predict,"temperature":0,"seed":123,
                    "ignore_eos":True,"cache_prompt":False,"return_tokens":True,"stream":True}
                low=dict(high,prompt=tokens[args.prefill:args.prefill+64],n_predict=min(64,args.predict))
                completion(base,"low-fresh",low)
                completion(base,"low-cached",dict(low,cache_prompt=True))
                completion(base,"high-fresh",high)
                completion(base,"high-cached",dict(high,cache_prompt=True))
                state_action(base,"save","initial")
                completion(base,"low-after-high",low,"low-fresh")
                state_action(base,"erase","initial")
                state_action(base,"restore","initial")
                # Replayed native states can select different batch/row layouts; use their matching native case as the oracle.
                completion(base,"restored-high",dict(high,cache_prompt=True))
                cancel_after_first_output(base,high)
                completion(base,"cancel-recovery",low,"low-fresh")
            with server(args,streamed,depth,"restart") as base:
                state_action(base,"restore","restart")
                completion(base,"restarted-restored",dict(high,cache_prompt=True))
                completion(base,"restarted-fresh",high,"high-fresh")
                completion(base,"restarted-low",low,"low-fresh")
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
    parser.add_argument("--require-cache-hit",action="store_true",help="require real repeated-prompt reuse and equal native/streamed prompt accounting")
    parser.add_argument("--lifecycle",action="store_true",help="qualify low/high/low, slot save/restore and process restart")
    parser.add_argument("--streamed-only",action="store_true",help="explicit lifecycle self-consistency mode where native allocations do not fit")
    args = parser.parse_args()
    args.depths = [int(value) for value in args.depths.split(",")]
    if not args.depths or any(depth not in (0, 1, 2, 3) for depth in args.depths) or len(set(args.depths)) != len(args.depths):
        parser.error("depths must be distinct values from 0,1,2,3; 0 disables the assistant")
    if min(args.context, args.prefill, args.predict, args.arena_mib, args.batch_size, args.ubatch_size) <= 0:
        parser.error("sizes must be positive")
    if args.prefill + args.predict + 4 >= args.context or min(args.batch_size, args.ubatch_size) < max(args.depths) + 1:
        parser.error("context and microbatch must cover the prompt, outputs and verification")
    if args.streamed_only and not args.lifecycle:
        parser.error("--streamed-only requires --lifecycle and is not a native-equivalence test")
    if args.lifecycle and 64+min(64,args.predict)+4>=args.context:
        parser.error("lifecycle context must also fit the 64-token low request")
    if args.lifecycle:
        run_lifecycle(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
