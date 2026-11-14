#!/usr/bin/env python3
"""Generate a CustomVoice PCM WAV and its transcript for clone_voice.py."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output WAV path")
    parser.add_argument("--speaker", default="Serena")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--text", default="下过雨的早晨，空气特别清新。我们慢慢走，不用着急。")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--instruct", help="speaking style for 1.7B CustomVoice")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not args.text.strip() or args.output.suffix.lower() != ".wav":
        parser.error("reference text and a .wav output path are required")

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    torch.manual_seed(args.seed)
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=args.device, dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    )
    if args.instruct and model.model.tts_model_size == "0b6":
        parser.error("--instruct requires the 1.7B CustomVoice checkpoint")
    with torch.inference_mode():
        waves, rate = model.generate_custom_voice(
            text=args.text, language=args.language, speaker=args.speaker, instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
        )
    wave = np.asarray(waves[0])
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise RuntimeError("model did not return valid mono audio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wave, rate, subtype="PCM_16")
    args.output.with_suffix(".txt").write_text(args.text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
