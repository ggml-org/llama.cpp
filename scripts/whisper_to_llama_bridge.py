#!/usr/bin/env python3
"""
Run `whisper-cli` on an audio file and POST the transcript to a running llama-server.

Usage (example):
  python scripts/whisper_to_llama_bridge.py --audio my.wav \
    --whisper-bin "build\\bin\\Release\\whisper-cli.exe" \
    --llama-url http://127.0.0.1:8080 --llama-endpoint /v1/completions \
    --llama-model my-model

The script calls `whisper-cli` (built from whisper.cpp) and posts the resulting
transcript as the `prompt` to the llama-server completions endpoint. No Python
dependencies required (uses stdlib only).
"""
import argparse
import json
import subprocess
import sys
from urllib import request, error


def run_whisper(whisper_bin: str, audio: str, model: str | None) -> str:
    cmd = [whisper_bin, "-f", audio]
    if model:
        cmd += ["-m", model]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    # prefer stdout, fallback to stderr if empty
    out = proc.stdout.strip() if proc.stdout.strip() else proc.stderr.strip()
    # remove common deprecation/warning lines
    lines = [l for l in out.splitlines() if not l.startswith("WARNING:")]
    # join and return
    return "\n".join(lines).strip()


def post_to_llama(url: str, endpoint: str, prompt: str, model: str | None, api_key: str | None, timeout: int = 30):
    payload = {"prompt": prompt}
    if model:
        payload["model"] = model

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url.rstrip("/") + endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), body
    except error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except Exception as e:
        return None, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Path to audio file (wav preferred)")
    p.add_argument("--whisper-bin", default=r"build\\bin\\Release\\whisper-cli.exe", help="Path to whisper-cli binary")
    p.add_argument("--whisper-model", default=None, help="Optional whisper model file for whisper-cli (-m)")
    p.add_argument("--llama-url", default="http://127.0.0.1:8080", help="Base URL of llama-server")
    p.add_argument("--llama-endpoint", default="/v1/completions", help="API endpoint to POST transcript to")
    p.add_argument("--llama-model", default=None, help="Model name to include in the JSON body")
    p.add_argument("--api-key", default=None, help="API key for llama-server (optional)")
    args = p.parse_args()

    transcript = run_whisper(args.whisper_bin, args.audio, args.whisper_model)
    if not transcript:
        print("No transcript produced by whisper-cli", file=sys.stderr)
        sys.exit(2)

    code, body = post_to_llama(args.llama_url, args.llama_endpoint, transcript, args.llama_model, args.api_key)
    if code is None:
        print("Error posting to llama-server:", body, file=sys.stderr)
        sys.exit(3)

    print(f"llama-server response code: {code}")
    print(body)


if __name__ == "__main__":
    main()
