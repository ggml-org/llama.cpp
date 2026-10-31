#!/usr/bin/env python3
"""Measure prompt reuse in llama-server for chats that contain media (images or audio).

Each scenario sends two chat requests to the same slot. The second request shares a prefix with
the first one, so the server should process again only the part of the prompt that changed.
For each request the script reports the prompt size, the tokens reused from the cache and the
tokens processed again.

Start the server first, for example:
    llama-server -m model.gguf --mmproj mmproj.gguf
and run:
    python scripts/server-test-mtmd-cache.py
or let the script start the server, the arguments after "--" are passed to it:
    python scripts/server-test-mtmd-cache.py --server build/bin/llama-server -- -m model.gguf --mmproj mmproj.gguf
With --parallel N each scenario runs on N slots at the same time (the server needs -np N or more).
"""

import argparse
import atexit
import base64
import concurrent.futures
import io
import json
import logging
import math
import struct
import subprocess
import sys
import time
import wave
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("server-test-mtmd-cache")

Message = dict[str, Any]

Q_FIRST = "Describe it in one short sentence."
Q_SECOND = "Answer with a single word: what is it?"
FILLER = " ".join(
    f"Note {i}: the archive keeps printed newspapers from many decades, sorted by date and by city."
    for i in range(80)
)


@dataclass
class Scenario:
    desc: str
    first: Callable[[], list[Message]]
    second: Callable[[], list[Message]]


@dataclass
class Result:
    scenario: str
    request: str
    slot: int
    prompt: int
    reused: int
    processed: int
    prompt_ms: float
    wall_ms: float
    output: str


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def user(*parts: dict) -> Message:
    return {"role": "user", "content": list(parts)}


def build_scenarios(m0: dict, m1: dict) -> dict[str, Scenario]:
    return {
        "edit": Scenario(
            "change the text after the media",
            lambda: [user(m0, text(Q_FIRST))],
            lambda: [user(m0, text(Q_SECOND))],
        ),
        "two": Scenario(
            "change the text after the second of two media",
            lambda: [user(m0, m1, text(Q_FIRST))],
            lambda: [user(m0, m1, text(Q_SECOND))],
        ),
        "long": Scenario(
            "change the start of a long text after the media",
            lambda: [user(m0, text("Context A. " + FILLER + "\n" + Q_FIRST))],
            lambda: [user(m0, text("Context B. " + FILLER + "\n" + Q_FIRST))],
        ),
    }


def make_png(width: int, height: int, seed: int) -> bytes:
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: none
        for x in range(width):
            raw += bytes(
                (
                    (x * (seed + 3)) % 256,
                    (y * (seed + 5)) % 256,
                    ((x + y) * 7 + seed * 50) % 256,
                )
            )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def make_wav(seconds: float, freq: float, rate: int = 16000) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(int(seconds * rate))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def image_part(data: bytes, fmt: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{fmt};base64,{base64.b64encode(data).decode()}"
        },
    }


def audio_part(data: bytes, fmt: str) -> dict:
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt},
    }


def file_part(path: Path) -> dict:
    ext = path.suffix.lower().lstrip(".")
    if ext in ("wav", "mp3", "flac"):
        return audio_part(path.read_bytes(), ext)
    return image_part(path.read_bytes(), "jpeg" if ext == "jpg" else ext)


def media_parts(args: argparse.Namespace, seed: int) -> tuple[dict, dict]:
    if args.file:
        parts = [file_part(Path(f)) for f in args.file]
        return parts[0], parts[-1]
    if args.media == "audio":
        f0 = 440.0 + 50.0 * seed
        return audio_part(make_wav(args.audio_seconds, f0), "wav"), audio_part(
            make_wav(args.audio_seconds, f0 * 1.5), "wav"
        )
    w, h = (int(v) for v in args.image_size.lower().split("x"))
    return image_part(make_png(w, h, 2 * seed), "png"), image_part(
        make_png(w, h, 2 * seed + 1), "png"
    )


def send(
    args: argparse.Namespace, messages: list[Message], cache_prompt: bool, slot: int
) -> Optional[dict]:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": args.n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": slot,
        "cache_prompt": cache_prompt,
    }
    kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}
    if args.thinking:
        kwargs["enable_thinking"] = args.thinking == "on"
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    t0 = time.time()
    try:
        res = requests.post(
            f"{args.base_url}/v1/chat/completions", json=payload, timeout=args.timeout
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"request failed: {e}")
        return None
    if res.status_code != 200:
        logger.error(f"request failed: HTTP {res.status_code}: {res.text[:500]}")
        return None
    data = res.json()
    data["wall_ms"] = (time.time() - t0) * 1000
    return data


def to_result(scenario: str, request: str, slot: int, data: dict) -> Result:
    timings = data.get("timings", {})
    message = data["choices"][0]["message"]
    reused = int(timings.get("cache_n", 0))
    processed = int(timings.get("prompt_n", 0))
    return Result(
        scenario=scenario,
        request=request,
        slot=slot,
        prompt=int(data.get("usage", {}).get("prompt_tokens", reused + processed)),
        reused=reused,
        processed=processed,
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
        wall_ms=float(data["wall_ms"]),
        output=(message.get("reasoning_content") or "")
        + (message.get("content") or ""),
    )


def log_result(r: Result, show_slot: bool) -> None:
    slot = f" slot {r.slot}" if show_slot else ""
    logger.info(
        f"  {r.request:<8}{slot} prompt {r.prompt:6d}  reused {r.reused:6d}  processed {r.processed:6d}  ({r.prompt_ms:8.1f} ms)"
    )


def check_server(args: argparse.Namespace) -> bool:
    try:
        props = requests.get(f"{args.base_url}/props", timeout=30).json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"cannot reach the server at {args.base_url}: {e}")
        return False
    modalities = props.get("modalities")
    needed = "audio" if args.media == "audio" else "vision"
    if not args.file and modalities is not None and not modalities.get(needed, False):
        logger.error(f"the loaded model does not support {needed} input")
        return False
    n_slots = int(props.get("total_slots", 1))
    if args.id_slot + args.parallel > n_slots:
        logger.error(
            f"slots {args.id_slot} to {args.id_slot + args.parallel - 1} are needed but the server has {n_slots}, start it with -np {args.id_slot + args.parallel} or more"
        )
        return False
    return True


def run_phase(
    args: argparse.Namespace,
    scenario: str,
    request: str,
    messages: list[list[Message]],
    cache_prompt: bool,
) -> list[Optional[Result]]:
    """send one request per slot at the same time, index i goes to slot id_slot + i"""

    def one(i: int) -> Optional[Result]:
        slot = args.id_slot + i
        data = send(args, messages[i], cache_prompt, slot)
        return to_result(scenario, request, slot, data) if data else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(messages)) as pool:
        results = list(pool.map(one, range(len(messages))))
    for r in results:
        if r is not None:
            log_result(r, args.parallel > 1)
    return results


def start_server(
    args: argparse.Namespace, server_args: list[str]
) -> Optional[subprocess.Popen]:
    cmd = [args.server, "--host", args.host, "--port", str(args.port), *server_args]
    logger.info(f"starting: {' '.join(cmd)}")
    log = open(args.server_log, "w") if args.server_log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    hint = (
        f"see {args.server_log}" if args.server_log else "use --server-log to see why"
    )
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        if proc.poll() is not None:
            logger.error(f"the server exited with code {proc.returncode}, {hint}")
            return None
        try:
            if requests.get(f"{args.base_url}/health", timeout=5).status_code == 200:
                return proc
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    logger.error(f"the server was not ready after {args.timeout} seconds, {hint}")
    stop_server(proc)
    return None


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure prompt reuse in llama-server for chats with media."
    )
    parser.add_argument("--host", default="localhost", help="server host")
    parser.add_argument("--port", type=int, default=9931, help="server port")
    parser.add_argument(
        "--server",
        default=None,
        help='llama-server binary to start, the arguments after "--" are passed to it (default: use a running server)',
    )
    parser.add_argument(
        "--server-log", default=None, help="file for the output of the started server"
    )
    parser.add_argument(
        "--media",
        choices=["image", "audio"],
        default="image",
        help="type of the generated media",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="media file to use instead of the generated media, give it twice to use two files in the 'two' scenario",
    )
    parser.add_argument(
        "--image-size", default="768x768", help="size of the generated images, WxH"
    )
    parser.add_argument(
        "--audio-seconds", type=float, default=3.0, help="length of the generated audio"
    )
    parser.add_argument(
        "--scenarios",
        default="edit,two,long",
        help="comma-separated list of scenarios to run",
    )
    parser.add_argument(
        "--n-predict", type=int, default=8, help="tokens to generate per request"
    )
    parser.add_argument(
        "--id-slot", type=int, default=0, help="first slot used for the requests"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="run each scenario on this many slots at the same time, the server needs at least that many slots (-np)",
    )
    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default=None,
        help="turn thinking on or off for templates that support it (default: server default)",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default=None,
        help="JSON object, for example '{\"enable_thinking\": false}'",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="send the second request again without the cache and compare the outputs",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="timeout in seconds for each request and for the server to start",
    )
    parser.add_argument("--json", default=None, help="write the results to this file")

    argv = sys.argv[1:]
    server_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, server_args = argv[:i], argv[i + 1 :]
    args = parser.parse_args(argv)
    args.base_url = f"http://{args.host}:{args.port}"
    if server_args and not args.server:
        parser.error('the arguments after "--" need --server')

    if args.server:
        proc = start_server(args, server_args)
        if proc is None:
            return 1
        atexit.register(stop_server, proc)

    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if not check_server(args):
        return 1

    # each slot gets its own media and session, so nothing is shared between slots
    per_slot = [
        build_scenarios(*media_parts(args, seed)) for seed in range(args.parallel)
    ]
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [n for n in names if n not in per_slot[0]]
    if unknown:
        logger.error(
            f"unknown scenarios: {', '.join(unknown)} (available: {', '.join(per_slot[0])})"
        )
        return 1

    run_id = int(time.time() * 1000) % 100000000
    results: list[Result] = []
    failed = False

    for name in names:
        systems = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Session {run_id}-{name}"
                + (f"-{i}." if args.parallel > 1 else "."),
            }
            for i in range(args.parallel)
        ]
        logger.info(f"{name}: {per_slot[0][name].desc}")

        firsts = run_phase(
            args,
            name,
            "first",
            [[systems[i]] + per_slot[i][name].first() for i in range(args.parallel)],
            cache_prompt=True,
        )
        messages = [
            [systems[i]] + per_slot[i][name].second() for i in range(args.parallel)
        ]
        seconds = run_phase(args, name, "second", messages, cache_prompt=True)
        colds = (
            run_phase(args, name, "uncached", messages, cache_prompt=False)
            if args.check
            else [None] * args.parallel
        )
        for i in range(args.parallel):
            first, second, cold = firsts[i], seconds[i], colds[i]
            if first is None or second is None or (args.check and cold is None):
                failed = True
                continue
            results += [first, second]
            if cold is not None:
                results.append(cold)
                same = "same as" if cold.output == second.output else "DIFFERENT from"
                slot = f" on slot {args.id_slot + i}" if args.parallel > 1 else ""
                logger.info(
                    f"  the output of the second request{slot} is {same} the uncached run"
                )

    logger.info("")
    logger.info("prompt tokens processed by the second request:")
    for r in results:
        if r.request == "second":
            slot = f" slot {r.slot}" if args.parallel > 1 else ""
            logger.info(f"  {r.scenario:<10}{slot} {r.processed:6d} of {r.prompt:6d}")

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
