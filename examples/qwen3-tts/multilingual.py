#!/usr/bin/env python3
"""
Multilingual TTS demo — synthesize speech in 10 different languages.

Generates one WAV file per language, demonstrating the same sentence
translated into each supported language.

Usage:
    python multilingual.py
    python multilingual.py --output-dir multilingual_output/
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SAMPLES = {
    "english":    "Hello, this is a demonstration of Qwen3 text to speech synthesis.",
    "chinese":    "你好，这是千问三号文本转语音合成的演示。",
    "german":     "Hallo, dies ist eine Demonstration der Qwen3 Sprachsynthese.",
    "spanish":    "Hola, esta es una demostración de la síntesis de voz Qwen3.",
    "french":     "Bonjour, ceci est une démonstration de la synthèse vocale Qwen3.",
    "italian":    "Ciao, questa è una dimostrazione della sintesi vocale Qwen3.",
    "japanese":   "こんにちは、Qwen3のテキスト読み上げ合成のデモンストレーションです。",
    "korean":     "안녕하세요, Qwen3 텍스트 음성 합성 시연입니다.",
    "portuguese": "Olá, esta é uma demonstração da síntese de voz Qwen3.",
    "russian":    "Здравствуйте, это демонстрация синтеза речи Qwen3.",
}


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
    parser = argparse.ArgumentParser(description="Qwen3-TTS multilingual demo")
    parser.add_argument("--output-dir", type=str, default="multilingual_output",
                        help="Output directory for WAV files")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Max frames per utterance")
    parser.add_argument("--languages", type=str, nargs="+", default=None,
                        help="Subset of languages to generate (default: all)")
    args = parser.parse_args()

    binary = find_binary()
    if not binary:
        print("ERROR: Cannot find llama-qwen3tts binary.")
        sys.exit(1)

    talker, cp, vocoder = find_models()
    if not talker or not cp:
        print("ERROR: Cannot find model files.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    langs = args.languages or list(SAMPLES.keys())
    for lang in langs:
        if lang not in SAMPLES:
            print(f"WARNING: no sample text for '{lang}', skipping")
            continue

        text = SAMPLES[lang]
        output = os.path.join(args.output_dir, f"{lang}.wav")
        print(f"\n{'='*60}")
        print(f"Language: {lang}")
        print(f"Text: {text}")
        print(f"Output: {output}")
        print(f"{'='*60}")

        cmd = [
            binary,
            "--model-talker", talker,
            "--model-cp", cp,
            "--text", text,
            "--output", output,
            "--language", lang,
            "--max-tokens", str(args.max_tokens),
        ]
        if vocoder:
            cmd += ["--model-vocoder", vocoder]

        result = subprocess.run(cmd, capture_output=True, text=True)
        combined = result.stdout + result.stderr
        if os.path.isfile(output) and os.path.getsize(output) > 44:
            for line in combined.splitlines():
                if "Real-time" in line or "Wrote" in line:
                    print(f"  {line.strip()}")
        else:
            print(f"  FAILED")
            for line in combined.splitlines()[-5:]:
                print(f"    {line}")

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
