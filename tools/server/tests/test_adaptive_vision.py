#!/usr/bin/env python3
"""Offline serial vision qualification, including bounded embedded MTP."""

import argparse
import base64
import contextlib
import http.client
import json
import os
import signal
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib


def png(width, height, variant=0):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(((x * 7 + y + variant * 31) % 256,
                         (y * 13 + variant * 47) % 256, (x + y * 3) % 256))
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")).decode()


def request(port, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


@contextlib.contextmanager
def server(args, mode, extra=(), reject=False, env_extra=None):
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    log_path = args.output / f"{mode}.log"
    command = [str(args.server.resolve()), "--model", str(args.model.resolve()), "--mmproj", str(args.mmproj.resolve()),
               "--host", "127.0.0.1", "--port", str(port), "--offline", "--fit", "off", "--parallel", "1",
               "--ctx-size", str(args.context), "-b", str(args.batch_size), "-ub", str(args.ubatch_size), "-ngl", "999", "-fa", "on",
               "-ctk", "q8_0", "-ctv", "q4_0", "--spec-type", "none", "--no-context-shift", "--no-warmup",
               "--cache-ram", str(args.cache_ram_mib), "--ctx-checkpoints", "8", "--image-min-tokens", "64",
               "--image-max-tokens", "4096", "--slots", "--no-webui", "--log-verbosity", str(getattr(args, "log_verbosity", 4))]
    if mode.startswith("arena"):
        command += ["--shared-device-memory-mib", str(args.arena_mib)]
    if getattr(args,"mtp_length",0):
        command += ["--spec-type","draft-mtp","--spec-draft-n-max",str(args.mtp_length),
                    "--spec-draft-type-k","q8_0","--spec-draft-type-v","q4_0"]
    command += list(extra)
    env = dict(os.environ, LLAMA_MEDIA_MARKER="<__media__>")
    env.update(env_extra or {})
    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    if mode == "stock" and getattr(args, "stock_uvm", False):
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=os.name == "posix")
        try:
            if reject:
                assert process.wait(timeout=30) != 0, f"unsupported startup admitted: {extra}"
                diagnostic = log_path.read_text().lower()
                assert "vision arena" in diagnostic or "attached mtp kv streaming supports at most 3 draft tokens" in diagnostic, log_path
                yield None
                return
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{mode} startup failed: {log_path}\n{log_path.read_text()[-4000:]}")
                try:
                    if request(port, "/health")[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError(f"{mode} startup timed out: {log_path}")
            yield port
        finally:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)


def completion(prompt, images=(), cache=False, decode=16):
    return {"prompt": {"prompt_string": prompt, "multimodal_data": list(images)} if images else prompt,
            "n_predict": decode, "temperature": 0, "top_k": 1, "seed": 42,
            "ignore_eos": True, "return_tokens": True, "cache_prompt": cache, "id_slot": 0}


def run(args, mode):
    image, changed, large = png(256, 256), png(256, 256, 1), png(512, 384, 2)
    marker = "<__media__>"
    background = " green" * args.background_tokens
    initial = background + f" Describe this pattern: {marker} One sentence: "
    cases = [
        ("text", completion("The capital of France is", decode=args.decode)),
        ("image", completion(initial, [image], decode=args.decode)),
        ("image_uncached_control", completion(initial, [image], decode=args.decode)),
        ("same_image_cached", completion(initial, [image], True, args.decode)),
        ("followup_cached", completion(initial + " Now focus on its colors: ", [image], True, args.decode)),
        ("changed_image", completion(initial, [changed], True, args.decode)),
        ("multiple_images", completion(f"First: {marker} Then: {marker} Compare in one sentence: ", [image, large], decode=args.decode)),
        ("adjacent_images", completion(f"Compare: {marker}{marker} One sentence: ", [image, changed], decode=args.decode)),
        ("after_other_prompt", completion(initial, [image], True, args.decode)),
    ]
    results = {}
    with server(args, mode) as port:
        for name, body in cases:
            status, result = request(port, "/completion", body)
            assert status == 200, (mode, name, status, result)
            assert len(result.get("tokens", [])) == args.decode, (name, result)
            results[name] = {"tokens": result["tokens"], "content": result["content"],
                             "cached": result.get("tokens_cached"), "timings": result.get("timings")}
            (args.output / f"{mode}-partial.json").write_text(json.dumps(results,indent=2))
            print(mode, name, results[name]["cached"], flush=True)
        assert results["image"]["tokens"] == results["image_uncached_control"]["tokens"]
        assert (results["changed_image"]["timings"] or {})["prompt_n"] > (results["same_image_cached"]["timings"] or {})["prompt_n"]
        assert any((results[name]["timings"] or {}).get("prompt_n", args.context) < args.background_tokens
                   for name in ("same_image_cached", "followup_cached")) or args.background_tokens == 0, results

        chat = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this pattern briefly."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}]}],
            "temperature": 0, "max_tokens": 8}
        status, result = request(port, "/v1/chat/completions", chat)
        assert status == 200 and result.get("choices"), (status, result)

        # Stop an in-flight request through socket disconnect, then require a clean next request.
        stream = completion(initial, [large], decode=512)
        stream.update(stream=True, return_progress=True)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        connection.request("POST", "/completion", json.dumps(stream), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        response.readline()
        response.close()
        connection.close()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status, slots = request(port, "/slots")
            if status == 200 and all(not slot["is_processing"] for slot in slots):
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("cancelled slot did not return to idle")
        status, result = request(port, "/completion", completion("The capital of France is", decode=args.decode))
        assert status == 200 and result["tokens"] == results["text"]["tokens"], (status, result)
        if mode == "arena" and not args.skip_budget_rejection:
            status, result = request(port, "/completion", completion(f"Describe: {marker} Answer: ", [png(1536, 1536)]))
            assert status >= 400, ("oversized vision workspace was not rejected", result)
            status, result = request(port, "/completion", completion("The capital of France is", decode=args.decode))
            assert status == 200 and result["tokens"] == results["text"]["tokens"], result
        assert request(port, "/health")[0] == 200
    assert "found better prompt" in (args.output / f"{mode}.log").read_text(), "RAM prompt-cache restoration was not exercised"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--arena-mib", type=int, default=1024)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ubatch-size", type=int, default=64)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--mtp-length", type=int, choices=(0,1,2,3), default=0)
    parser.add_argument("--cache-ram-mib", type=int, default=2048)
    parser.add_argument("--background-tokens", type=int, default=128)
    parser.add_argument("--mode", choices=("arena", "stock", "compare"), default="compare")
    parser.add_argument("--stock-uvm", action="store_true",
                        help="allow only the eager correctness control to oversubscribe VRAM; the arena run keeps UVM disabled")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-rejections", action="store_true")
    parser.add_argument("--skip-budget-rejection", action="store_true",
                        help="skip the oversized-image rejection assertion when testing a larger arena")
    args = parser.parse_args()
    if min(args.arena_mib, args.context, args.decode, args.batch_size, args.ubatch_size) <= 0 or args.background_tokens < 0:
        parser.error("memory/context/batch/decode values must be positive, background must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in (("stock", "arena") if args.mode == "compare" else (args.mode,)):
        results[mode] = run(args, mode)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
    if args.mode == "compare":
        for name in results["stock"]:
            assert results["stock"][name]["tokens"] == results["arena"][name]["tokens"], name
        print("Stock and arena token IDs match for every request.")
    if args.check_rejections:
        cases = [("parallel", ("--parallel", "2")), ("mtp", ("--spec-type", "draft-mtp")),
                 ("cpu-projector", ("--no-mmproj-offload",)), ("kv-quant", ("-ctk", "f16")),
                 ("fit", ("--fit", "on")), ("embedding", ("--embedding",))]
        cases[1] = ("mtp-length",("--spec-type","draft-mtp","--spec-draft-n-max","4"))
        for name, extra in cases:
            with server(args, f"arena-reject-{name}", extra, reject=True):
                print("startup rejected:", name)


if __name__ == "__main__":
    main()
