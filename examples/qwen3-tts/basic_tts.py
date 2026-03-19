#!/usr/bin/env python3
"""
Basic TTS demo — synthesize text to speech with Qwen3-TTS.

This is the simplest usage: just provide text and get audio.

Usage:
    python basic_tts.py --text "Hello, this is a test of Qwen3 text to speech."
    python basic_tts.py --text "The quick brown fox jumps over the lazy dog." --output fox.wav
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def find_binary():
    """Locate the llama-qwen3tts binary."""
    candidates = [
        os.path.join(REPO_ROOT, "build", "bin", "Release", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "Release", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "bin", "Debug", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "Debug", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "bin", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "llama-qwen3tts"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_models():
    """Locate GGUF model files. Prefers f16 vocoder over bf16 (ggml conv1d requires f16)."""
    models_dir = os.path.join(REPO_ROOT, "models")
    talker = None
    cp = None
    vocoder = None
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            fl = f.lower()
            if "qwen3tts" not in fl or not fl.endswith(".gguf"):
                continue
            if "talker" in fl:
                if talker is None or "bf16" in fl:
                    talker = os.path.join(models_dir, f)
            elif "cp" in fl:
                if cp is None or "bf16" in fl:
                    cp = os.path.join(models_dir, f)
            elif "tokenizer" in fl:
                if vocoder is None or "f16" in fl:
                    vocoder = os.path.join(models_dir, f)
    return talker, cp, vocoder


def main():
    parser = argparse.ArgumentParser(description="Basic Qwen3-TTS synthesis")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--language", type=str, default="english",
                        help="Language (english, chinese, french, etc.)")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Max frames to generate (12 frames ≈ 1 second)")
    parser.add_argument("--temp", type=float, default=0.9, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p / nucleus sampling")
    parser.add_argument("--rep-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--greedy", action="store_true", help="Force greedy decoding")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--streaming-text", action="store_true", help="Enable streaming text mode")
    parser.add_argument("--binary", type=str, default=None, help="Path to llama-qwen3tts binary")
    parser.add_argument("--model-talker", type=str, default=None)
    parser.add_argument("--model-cp", type=str, default=None)
    parser.add_argument("--model-vocoder", type=str, default=None)
    args = parser.parse_args()

    binary = args.binary or find_binary()
    if not binary:
        print("ERROR: Cannot find llama-qwen3tts binary.")
        print("  Build it first: cmake --build build --target llama-qwen3tts")
        sys.exit(1)

    talker, cp, vocoder = find_models()
    talker = args.model_talker or talker
    cp = args.model_cp or cp
    vocoder = args.model_vocoder or vocoder

    if not talker or not cp:
        print("ERROR: Cannot find model files in models/ directory.")
        print("  Expected: *talker*.gguf, *cp*.gguf, *tokenizer*.gguf")
        sys.exit(1)

    cmd = [
        binary,
        "--model-talker", talker,
        "--model-cp", cp,
        "--text", args.text,
        "--output", args.output,
        "--language", args.language,
        "--max-tokens", str(args.max_tokens),
        "--temp", str(args.temp),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--rep-penalty", str(args.rep_penalty),
    ]
    if vocoder:
        cmd += ["--model-vocoder", vocoder]
    if args.greedy:
        cmd += ["--greedy"]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.streaming_text:
        cmd += ["--streaming-text"]

    print(f"Synthesizing: \"{args.text}\"")
    print(f"Language: {args.language}")
    print(f"Output: {args.output}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = result.stdout + result.stderr

    if os.path.isfile(args.output) and os.path.getsize(args.output) > 44:
        size = os.path.getsize(args.output)
        # Show performance stats if available
        for line in combined.splitlines():
            if any(k in line for k in ["Prefill:", "Decode:", "Real-time", "Wrote"]):
                print(line.strip())
        print(f"\nDone. Output: {args.output} ({size:,} bytes)")
    else:
        print(f"\nERROR: synthesis failed")
        print(combined)
        sys.exit(1)


if __name__ == "__main__":
    main()
