#!/usr/bin/env python3
"""Offline native-versus-streamed Gemma target and assistant server qualification."""

import argparse
import base64
import copy
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import struct
import sys
import urllib.request
import zlib

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
               "--no-context-shift",
               "--cache-ram", "2048", "--ctx-checkpoints", "4", "--no-webui", "--log-verbosity", "3"]
    if getattr(args,"mmproj",None):
        command += ["--mmproj",str(args.mmproj.resolve()),"--image-min-tokens",str(args.image_min_tokens),
                    "--image-max-tokens",str(args.image_max_tokens)]
    else:
        command += ["--no-mmproj"]
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


def vision_prompt_counts(result,cached,label):
    # Image placeholders are not decoder positions. Validate the actual expanded extent.
    total=result.get("tokens_evaluated")
    if type(total) is not int or total<=0:
        raise RuntimeError("invalid Gemma vision evaluated extent: "+label)
    return (*prompt_counts(result,total,cached,label),total)


def vision_cases(args,prefix,images):
    first,changed,large=images
    marker="<__media__>"
    def body(prompt,media=(),cached=False):
        return {"prompt":{"prompt_string":prompt,"multimodal_data":list(media)} if media else prompt,
                "n_predict":args.predict,"temperature":0,"top_k":1,"seed":123,"ignore_eos":True,
                "return_tokens":True,"cache_prompt":cached,"stream":True,"id_slot":0}
    prompt=prefix+f" Describe this pattern: {marker} One sentence: "
    cases=[("text",body("The capital of France is")),
            ("image_fresh",body(prompt,[first])),
            ("same_image_cached",body(prompt,[first],True)),
            ("followup_cached",body(prompt+" Now focus on its colors: ",[first],True)),
            ("changed_image",body(prompt,[changed],True)),
            ("multiple_images",body(f"First: {marker} Then: {marker} Compare: ",[first,large])),
            ("adjacent_images",body(f"Compare: {marker}{marker} One sentence: ",[first,changed])),
            ("image_last",body("Describe: "+marker,[large]))]
    if getattr(args,"vision_boundaries",False):
        cases.extend([("image_only",body(marker,[first])),("image_first",body(marker+" Describe this pattern: ",[large]))])
    if getattr(args,"vision_aspects",False):
        from test_adaptive_vision import png
        cases.extend([("portrait_image",body(prompt,[png(384,512,3)])),("narrow_image",body(prompt,[png(32,1024,4)]))])
    return cases


def vision_context_overflow(args,valid):
    body=copy.deepcopy(valid)
    body["prompt"]["prompt_string"]=" green"*(2*args.context)+"<__media__>"
    body["cache_prompt"]=False;body["stream"]=False
    return body


def vision_after_decode(initial,result,image):
    content=result.get("content")
    if not isinstance(content,str) or not content:raise RuntimeError("missing generated text before later image")
    body=copy.deepcopy(initial)
    body["prompt"]["prompt_string"]+=content+" Compare this new image: <__media__> One sentence: "
    body["prompt"]["multimodal_data"].append(image)
    body["cache_prompt"]=True
    return body


def vision_rejection_cases(valid):
    from test_adaptive_vision import png
    image=base64.b64decode(png(256,256))
    large=bytearray(image)
    large[16:24]=struct.pack(">II",(1<<24)+1,1)
    large[29:33]=struct.pack(">I",zlib.crc32(large[12:29]))
    overflow=bytearray(image)
    overflow[16:24]=struct.pack(">II",65536,65536)
    overflow[29:33]=struct.pack(">I",zlib.crc32(overflow[12:29]))
    audio=struct.pack("<4sI4s4sIHHIIHH4sI",b"RIFF",356,b"WAVE",b"fmt ",16,1,1,16000,32000,2,16,b"data",320)+bytes(320)
    records=[]
    for name,data in (("empty-image",b""),("truncated-png",image[:24]),("garbage-image",b"not an image"),
            ("oversized-png-header",large),("overflow-pixel-product",overflow),("unsupported-audio",audio)):
        body=copy.deepcopy(valid);body.update(stream=False,n_predict=8)
        body["prompt"]["multimodal_data"]=[base64.b64encode(data).decode()]
        records.append((name,"/completion",body,"Failed to load image or audio file"))
    records.append(("file-url","/v1/chat/completions",{"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"file:///not-a-real-vision-fixture.png"}}]}],"max_tokens":8},
        "file:// URLs are not allowed"))
    return records


def check_vision_rejection(status,result,expected,label):
    error=result.get("error",{})
    message=error.get("message") if isinstance(error,dict) else error
    if (not 400<=status<600 or not isinstance(message,str) or expected.lower() not in message.lower() or
            re.search(r"out of memory|CUDA error|illegal memory|GGML_ASSERT",message,re.I)):
        raise RuntimeError("unexpected Gemma vision rejection: "+label+" "+str((status,result)))


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
    if getattr(args,"mmproj",None):
        env["LLAMA_MEDIA_MARKER"]="<__media__>"
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


def run_vision(args):
    """Compare each image scenario with its own native replay/layout, never another case."""
    from test_adaptive_vision import png,request

    args.output.mkdir(parents=True,exist_ok=True)
    text=args.corpus.read_text(encoding="utf-8")
    images=(png(256,256),png(256,256,1),png(512,384,2))
    results=[]
    actions=[]
    rejections=[]
    streamed_only=getattr(args,"streamed_only",False)
    for depth in args.depths:
        native={}
        for streamed in ((True,) if streamed_only else (False,True)):
            saved_tokens=saved_bytes=0
            path=slot_directory(args,streamed,depth)/"vision-lifecycle.bin"
            references={}

            def completion(base,label,body,cached=False):
                result=stream_completion(base+"/completion",body,timeout=600)
                request_sha=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
                row={"depth":depth,"streamed":streamed,"case":label,"result":result,
                     "request_sha256":request_sha,"comparison":"streamed-self-reference" if streamed_only else "native-matched"}
                (args.output/"last-response.json").write_text(json.dumps(row,indent=2))
                tokens=result.get("tokens",[])
                if (result.get("timings",{}).get("predicted_n")!=body["n_predict"] or len(tokens)!=body["n_predict"] or
                        any(type(token) is not int or token<0 for token in tokens)):
                    raise RuntimeError("incomplete Gemma vision completion: "+label)
                counts=draft_counts(result,depth,label)
                accounting=vision_prompt_counts(result,cached,label)
                reference_case="image_fresh" if label in ("save-image","restarted-fresh-image","context-limit-recovery","invalid-state-recovery") else \
                    "text" if label in ("low-after-image","cancel-recovery","restarted-text") else None
                if reference_case and tokens!=references[reference_case]:
                    raise RuntimeError("Gemma vision uncached self-reference mismatch: "+label)
                if streamed and not streamed_only:
                    reference=native[label]
                    if tokens!=reference[0]:raise RuntimeError("native/streamed vision token mismatch: "+label)
                    if (counts,accounting,request_sha)!=reference[1:]:
                        raise RuntimeError("native/streamed vision accounting/request mismatch: "+label)
                elif not streamed:
                    native[label]=(tokens,counts,accounting,request_sha)
                references[label]=tokens
                results.append(row)
                (args.output/"results.json").write_text(json.dumps(results,indent=2))
                print(json.dumps({"depth":depth,"streamed":streamed,"case":label,"cache_n":accounting[0],
                    "prompt_n":accounting[1],"evaluated":accounting[2],"outputs":len(tokens)}),flush=True)
                return result

            def rejection(base,label,endpoint,body,expected,error_type=None):
                status,result=request(int(base.rsplit(":",1)[1]),endpoint,body)
                rejections.append({"depth":depth,"streamed":streamed,"case":label,"status":status,"result":result,
                    "endpoint":endpoint,"request_sha256":hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()})
                (args.output/"rejections.json").write_text(json.dumps(rejections,indent=2))
                check_vision_rejection(status,result,expected,label)
                if error_type and result.get("error",{}).get("type")!=error_type:
                    raise RuntimeError("wrong rejection type: "+label)

            def state_action(base,action,phase):
                nonlocal saved_tokens,saved_bytes
                response=request_json(base+"/slots/0?action="+action,
                    None if action=="erase" else {"filename":path.name},timeout=600,method="POST")
                if action=="save":
                    saved_tokens=response.get("n_saved");saved_bytes=response.get("n_written")
                    if (type(saved_tokens) is not int or not 0<saved_tokens<=args.context or
                            type(saved_bytes) is not int or saved_bytes<=0 or not path.is_file() or path.stat().st_size!=saved_bytes):
                        raise RuntimeError("incomplete Gemma vision state save")
                elif action=="restore":
                    if response.get("n_restored")!=saved_tokens or response.get("n_read")!=saved_bytes or path.stat().st_size!=saved_bytes:
                        raise RuntimeError("incomplete Gemma vision state restore")
                elif type(response.get("n_erased")) is not int or response["n_erased"]<0:
                    raise RuntimeError("invalid Gemma vision state erase")
                with path.open("rb") as state:
                    digest=hashlib.file_digest(state,"sha256").hexdigest()
                actions.append({"depth":depth,"streamed":streamed,"phase":phase,"action":action,"path":str(path),
                    "response":response,"file_sha256":digest})
                (args.output/"state-actions.json").write_text(json.dumps(actions,indent=2))

            with server(args,streamed,depth) as base:
                tokens=request_json(base+"/tokenize",{"content":text,"add_special":True})["tokens"]
                if len(tokens)<args.prefill:raise RuntimeError("qualification corpus is too short")
                prefix=request_json(base+"/detokenize",{"tokens":tokens[:args.prefill]})["content"]
                cases=vision_cases(args,prefix,images)
                for label,body in cases:
                    result=completion(base,label,body,cached=body["cache_prompt"])
                    if label=="image_fresh" and getattr(args,"vision_followup_after_decode",False):
                        completion(base,"image-after-decode",vision_after_decode(body,result,images[1]),True)
                high=dict(cases[1][1],cache_prompt=True)
                low=cases[0][1]
                if getattr(args,"vision_boundaries",False):
                    rejection(base,"context-limit","/completion",vision_context_overflow(args,high),"context size","exceed_context_size_error")
                    completion(base,"context-limit-recovery",dict(high,cache_prompt=False))
                if getattr(args,"vision_rejections",False):
                    completion(base,"before-rejections",dict(high,cache_prompt=False))
                    port=int(base.rsplit(":",1)[1])
                    for label,path_suffix,body,expected in vision_rejection_cases(high):
                        rejection(base,label,path_suffix,body,expected)
                        if request(port,"/health")!=(200,{"status":"ok"}):raise RuntimeError("server unhealthy after rejection")
                        completion(base,"after-rejection-"+label,high,True)
                if args.lifecycle:
                    completion(base,"save-image",dict(high,cache_prompt=False))
                    completion(base,"save-image-cached",high,True)
                    state_action(base,"save","initial")
                    completion(base,"low-after-image",low)
                    state_action(base,"erase","initial")
                    state_action(base,"restore","initial")
                    completion(base,"restored-image",high,True)
                    if getattr(args,"vision_boundaries",False):
                        with path.open("rb") as complete:
                            original=hashlib.file_digest(complete,"sha256").hexdigest()
                            complete.seek(0);truncated_bytes=complete.read(min(4096,max(1,saved_bytes//2)))
                        truncated=path.with_name("vision-truncated.bin")
                        with truncated.open("xb") as output:output.write(truncated_bytes)
                        rejection(base,"truncated-state","/slots/0?action=restore",{"filename":truncated.name},"Unable to restore slot")
                        with path.open("rb") as complete:
                            if path.stat().st_size!=saved_bytes or hashlib.file_digest(complete,"sha256").hexdigest()!=original:
                                raise RuntimeError("valid state file changed during truncated restore")
                        completion(base,"invalid-state-recovery",dict(high,cache_prompt=False))
                cancel_after_first_output(base,dict(high,cache_prompt=False))
                completion(base,"cancel-recovery",low)
            if args.lifecycle:
                with server(args,streamed,depth,"restart") as base:
                    state_action(base,"restore","restart")
                    completion(base,"restarted-restored-image",high,True)
                    completion(base,"restarted-fresh-image",dict(high,cache_prompt=False))
                    completion(base,"restarted-text",low)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--mmproj",type=Path,help="enable paired Gemma image/projector qualification")
    parser.add_argument("--image-min-tokens",type=int,default=70)
    parser.add_argument("--image-max-tokens",type=int,default=70)
    parser.add_argument("--vision-rejections",action="store_true",help="check malformed/oversized/unsupported media and cached recovery")
    parser.add_argument("--vision-followup-after-decode",action="store_true",help="append another image after retaining the generated continuation")
    parser.add_argument("--vision-boundaries",action="store_true",help="qualify image-only/first prompts, context rejection and optional truncated-state recovery")
    parser.add_argument("--vision-aspects",action="store_true",help="include portrait and narrow/tall image requests")
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
    if args.mmproj and (args.image_min_tokens not in (70,140,280,560,1120) or
            args.image_max_tokens not in (70,140,280,560,1120) or args.image_min_tokens>args.image_max_tokens or
            args.ubatch_size<args.image_max_tokens or args.batch_size<args.ubatch_size or
            args.prefill+2*args.image_max_tokens+args.predict+128>=args.context):
        parser.error("vision requires supported 70/140/280/560/1120 budgets, a whole image microbatch and sufficient context")
    if args.vision_rejections and not args.mmproj:
        parser.error("--vision-rejections requires --mmproj")
    if args.vision_boundaries and not args.mmproj:parser.error("--vision-boundaries requires --mmproj")
    if args.vision_aspects and not args.mmproj:parser.error("--vision-aspects requires --mmproj")
    if args.vision_followup_after_decode and (not args.mmproj or args.prefill+2*args.image_max_tokens+2*args.predict+128>=args.context):
        parser.error("later image requires --mmproj and context for both generated continuations")
    if args.mmproj:
        run_vision(args)
    elif args.lifecycle:
        run_lifecycle(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
