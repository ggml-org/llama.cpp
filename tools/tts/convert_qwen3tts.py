#!/usr/bin/env python3
"""
Convert Qwen3-TTS model to GGUF format.

Produces three GGUF files:
  1. Talker (28-layer Qwen2 + speaker encoder)
  2. Code Predictor (5-layer)
  3. Tokenizer/Vocoder (separate script)

Usage:
    python tools/tts/convert_qwen3tts.py \
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --outdir models/ \
        --outtype f16
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from convert_hf_to_gguf import Qwen3TTSTalkerModel, Qwen3TTSCodePredictorModel, ModelBase
import gguf


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS to GGUF (talker + code predictor)")
    parser.add_argument("--model", "-m", type=Path, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--outdir", "-d", type=Path, default=Path("."),
                        help="Output directory for GGUF files")
    parser.add_argument("--outtype", "-t", choices=["f16", "f32", "bf16", "q8_0"], default="f16",
                        help="Output data type")
    parser.add_argument("--only", choices=["talker", "cp", "both"], default="both",
                        help="Which model to convert")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    ftype_map = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }
    output_type = ftype_map[args.outtype]

    import torch
    with torch.inference_mode():
        if args.only in ("talker", "both"):
            print("=" * 60)
            print("  Converting Talker model")
            print("=" * 60)
            fname_talker = args.outdir / f"qwen3tts-talker-{args.outtype}.gguf"
            talker = Qwen3TTSTalkerModel(args.model, output_type, fname_talker)
            talker.write()
            print(f"Talker GGUF written to: {fname_talker}")
            print()

        if args.only in ("cp", "both"):
            print("=" * 60)
            print("  Converting Code Predictor model")
            print("=" * 60)
            fname_cp = args.outdir / f"qwen3tts-cp-{args.outtype}.gguf"
            cp = Qwen3TTSCodePredictorModel(args.model, output_type, fname_cp)
            cp.write()
            print(f"Code Predictor GGUF written to: {fname_cp}")
            print()

    print("Done! Use convert_qwen3tts_tokenizer.py for the vocoder GGUF.")


if __name__ == "__main__":
    main()
