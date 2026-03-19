#!/usr/bin/env python3
"""
Voice cloning demo — clone a speaker's voice from reference audio.

Supports two modes:
  1. X-vector mode (simple): only needs reference audio
  2. ICL mode (higher quality): needs reference audio + transcript + codec codes

Usage:
    # X-vector mode (simpler, uses speaker embedding only)
    python voice_cloning.py \
        --ref-audio speaker.wav \
        --text "This sentence will be spoken in the cloned voice."

    # ICL mode (higher quality, also uses reference text and codec tokens)
    python voice_cloning.py \
        --ref-audio speaker.wav \
        --ref-text "The transcript of the reference audio" \
        --ref-codes ref_codes.txt \
        --text "This sentence will be spoken in the cloned voice."

    # Extract ref_codes first (requires HuggingFace model):
    python ../../tools/tts/extract_ref_codes.py \
        --ref-audio speaker.wav \
        --output ref_codes.txt
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def find_binary():
    candidates = [
        os.path.join(REPO_ROOT, "build", "bin", "Release", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "Release", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "bin", "Debug", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "Debug", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "bin", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "llama-qwen3tts"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_models():
    """Locate GGUF model files. Prefers f16 vocoder over bf16 (ggml conv1d requires f16)."""
    models_dir = os.path.join(REPO_ROOT, "models")
    talker = cp = vocoder = None
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
    parser = argparse.ArgumentParser(description="Qwen3-TTS voice cloning")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference audio WAV")
    parser.add_argument("--ref-text", type=str, default=None,
                        help="Reference transcript (enables ICL mode when combined with --ref-codes)")
    parser.add_argument("--ref-codes", type=str, default=None,
                        help="Precomputed codec codes file (use extract_ref_codes.py)")
    parser.add_argument("--output", type=str, default="cloned_output.wav")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    if not os.path.isfile(args.ref_audio):
        print(f"ERROR: reference audio not found: {args.ref_audio}")
        sys.exit(1)

    binary = find_binary()
    if not binary:
        print("ERROR: Cannot find llama-qwen3tts binary.")
        sys.exit(1)

    talker, cp, vocoder = find_models()
    if not talker or not cp:
        print("ERROR: Cannot find model files.")
        sys.exit(1)

    icl_mode = args.ref_text and args.ref_codes
    if args.ref_text and not args.ref_codes:
        print("NOTE: --ref-text without --ref-codes uses x-vector mode.")
        print("  For ICL mode, extract codes first:")
        print(f"    python ../../tools/tts/extract_ref_codes.py --ref-audio {args.ref_audio} --output ref_codes.txt")
        print()

    mode_name = "ICL (in-context learning)" if icl_mode else "x-vector"
    print(f"Voice cloning mode: {mode_name}")
    print(f"Reference audio: {args.ref_audio}")
    if icl_mode:
        print(f"Reference text: \"{args.ref_text}\"")
        print(f"Reference codes: {args.ref_codes}")
    print(f"Target text: \"{args.text}\"")
    print(f"Output: {args.output}")
    print()

    cmd = [
        binary,
        "--model-talker", talker,
        "--model-cp", cp,
        "--ref-audio", args.ref_audio,
        "--text", args.text,
        "--output", args.output,
        "--language", args.language,
        "--max-tokens", str(args.max_tokens),
    ]
    if vocoder:
        cmd += ["--model-vocoder", vocoder]
    if args.ref_text:
        cmd += ["--ref-text", args.ref_text]
    if args.ref_codes:
        cmd += ["--ref-codes", args.ref_codes]

    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = result.stdout + result.stderr

    if os.path.isfile(args.output) and os.path.getsize(args.output) > 44:
        size = os.path.getsize(args.output)
        for line in combined.splitlines():
            if any(k in line for k in ["Prefill:", "Decode:", "Real-time", "Wrote", "Speaker embed", "ICL"]):
                print(line.strip())
        print(f"\nDone. Output: {args.output} ({size:,} bytes)")
    else:
        print(f"\nERROR: synthesis failed")
        print(combined)
        sys.exit(1)


if __name__ == "__main__":
    main()
