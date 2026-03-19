#!/usr/bin/env python3
"""
Performance benchmark — measure token/s, prefill rate, and real-time factor.

Runs the model with various text lengths and reports throughput statistics.

Usage:
    python benchmark.py
    python benchmark.py --runs 3 --max-tokens 100
"""

import argparse
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

BENCHMARK_TEXTS = [
    ("short", "Hello world."),
    ("medium", "The quick brown fox jumps over the lazy dog. This is a standard test sentence for speech synthesis quality evaluation."),
    ("long", "In the beginning, there was silence. Then came the first whisper of sound, a gentle breeze that carried the promise of something extraordinary. As the world awakened from its slumber, birds began to sing their morning melodies, each note a testament to the beauty of nature. The sun rose slowly over the horizon, painting the sky in brilliant shades of orange and pink."),
]


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


def parse_performance(output: str) -> dict:
    """Extract performance metrics from llama-qwen3tts output."""
    metrics = {}

    m = re.search(r"Prefill:\s+(\d+) tokens in ([\d.]+) ms\s+\(([\d.]+) tok/s\)", output)
    if m:
        metrics["prefill_tokens"] = int(m.group(1))
        metrics["prefill_ms"] = float(m.group(2))
        metrics["prefill_tok_s"] = float(m.group(3))

    m = re.search(r"Decode:\s+(\d+) frames in ([\d.]+) ms\s+\(([\d.]+) frames/s\)", output)
    if m:
        metrics["decode_frames"] = int(m.group(1))
        metrics["decode_ms"] = float(m.group(2))
        metrics["decode_fps"] = float(m.group(3))

    m = re.search(r"Talker:\s+([\d.]+) ms total\s+\(([\d.]+) ms/frame\)", output)
    if m:
        metrics["talker_ms"] = float(m.group(1))
        metrics["talker_ms_per_frame"] = float(m.group(2))

    m = re.search(r"CP:\s+([\d.]+) ms total\s+\(([\d.]+) ms/frame\)", output)
    if m:
        metrics["cp_ms"] = float(m.group(1))
        metrics["cp_ms_per_frame"] = float(m.group(2))

    m = re.search(r"Head:\s+([\d.]+) ms total\s+\(([\d.]+) ms/frame\)", output)
    if m:
        metrics["head_ms"] = float(m.group(1))
        metrics["head_ms_per_frame"] = float(m.group(2))

    m = re.search(r"Real-time factor:\s+([\d.]+)x\s+\(([\d.]+)s audio in ([\d.]+)s\)", output)
    if m:
        metrics["rtf"] = float(m.group(1))
        metrics["audio_s"] = float(m.group(2))
        metrics["wall_s"] = float(m.group(3))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS performance benchmark")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per test")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max frames to generate per test")
    parser.add_argument("--with-vocoder", action="store_true",
                        help="Include vocoder in benchmark (adds overhead)")
    args = parser.parse_args()

    binary = find_binary()
    if not binary:
        print("ERROR: Cannot find llama-qwen3tts binary.")
        sys.exit(1)

    talker, cp, vocoder = find_models()
    if not talker or not cp:
        print("ERROR: Cannot find model files.")
        sys.exit(1)

    print("=" * 80)
    print("Qwen3-TTS Performance Benchmark")
    print("=" * 80)
    print(f"  Binary:    {binary}")
    print(f"  Talker:    {os.path.basename(talker)}")
    print(f"  CP:        {os.path.basename(cp)}")
    print(f"  Max frames: {args.max_tokens}")
    print(f"  Runs per test: {args.runs}")
    print()

    results = []
    for label, text in BENCHMARK_TEXTS:
        run_metrics = []
        for run in range(args.runs):
            output_file = f"_bench_{label}.wav"
            cmd = [
                binary,
                "--model-talker", talker,
                "--model-cp", cp,
                "--text", text,
                "--output", output_file,
                "--language", "english",
                "--max-tokens", str(args.max_tokens),
            ]
            if args.with_vocoder and vocoder:
                cmd += ["--model-vocoder", vocoder]

            result = subprocess.run(cmd, capture_output=True, text=True)
            combined = result.stdout + result.stderr
            metrics = parse_performance(combined)

            if metrics:
                run_metrics.append(metrics)

            if os.path.isfile(output_file):
                os.remove(output_file)

        if not run_metrics:
            print(f"  {label}: FAILED")
            continue

        avg = {}
        for key in run_metrics[0]:
            vals = [m[key] for m in run_metrics if key in m]
            avg[key] = sum(vals) / len(vals)

        results.append((label, text, avg))

    print(f"\n{'='*80}")
    print("Results:")
    print(f"{'='*80}")
    print(f"{'Test':<10} {'Prefill':>10} {'Prefill':>10} {'Decode':>10} {'Talker':>10} {'CP':>10} {'RTF':>8}")
    print(f"{'':.<10} {'(tok/s)':>10} {'(ms)':>10} {'(fr/s)':>10} {'(ms/fr)':>10} {'(ms/fr)':>10} {'':>8}")
    print("-" * 80)

    for label, text, avg in results:
        print(f"{label:<10} "
              f"{avg.get('prefill_tok_s', 0):>10.1f} "
              f"{avg.get('prefill_ms', 0):>10.1f} "
              f"{avg.get('decode_fps', 0):>10.2f} "
              f"{avg.get('talker_ms_per_frame', 0):>10.1f} "
              f"{avg.get('cp_ms_per_frame', 0):>10.1f} "
              f"{avg.get('rtf', 0):>8.2f}x")

    print()
    print("Key metrics:")
    print("  Prefill tok/s  — how fast the model processes the input text")
    print("  Decode fr/s    — audio frames generated per second (12 = real-time)")
    print("  Talker ms/fr   — time per frame for the 28-layer Talker")
    print("  CP ms/fr       — time per frame for the 5-layer Code Predictor (15 steps)")
    print("  RTF            — real-time factor (>1.0 = faster than real-time)")


if __name__ == "__main__":
    main()
