#!/usr/bin/env python3
"""Offline voice cloning for the M5-TTS android engine.

Clones a speaker once from reference audio + its transcript and caches the
result in voices.json (codec codes, 16 books per frame). Every later synthesis
run consumes the cache directly - no encoder, no reference audio on device.

    # clone (or replace) a voice
    ./clone_voice.py ref.wav "参考音频的逐字转写" --name alice \
        --model /tmp/m5-official/onnx --voices model/voices.json

    # inspect what is cached
    ./clone_voice.py --list --voices model/voices.json

The reference audio is preprocessed exactly like M5's infer_onnx.read_reference
(mono mixdown, resample to 24 kHz, peak normalize, silence trim with 0.15 s
gaps cut and 0.05 s padding) so cloned voices match the reference pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path

import wave

import numpy as np
import onnxruntime as ort


def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Linear-interpolation resample; only used for references not at 24 kHz.
    (M5's infer_onnx uses scipy resample_poly - import it when available.)"""
    try:
        from math import gcd
        from scipy.signal import resample_poly
        divisor = gcd(sr, target)
        return resample_poly(audio, target // divisor, sr // divisor).astype(np.float32)
    except Exception:
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=int(len(audio) * target / sr), endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def read_reference(path: Path, sample_rate: int = 24000) -> np.ndarray:
    """Mirrors M5's infer_onnx.read_reference: mono mixdown, resample, peak
    normalize, silence trim (0.15 s gaps cut, 0.05 s padding kept)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width} in {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"empty or invalid reference audio: {path}")
    if sr != sample_rate:
        audio = _resample(audio, sr, sample_rate)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    threshold = 0.02 if len(audio) > 10 * sample_rate else 0.015
    speech = np.flatnonzero(np.abs(audio) > threshold)
    if not len(speech):
        raise ValueError("reference audio contains no speech above the silence threshold")
    cuts = np.flatnonzero(np.diff(speech) > int(0.15 * sample_rate))
    starts = np.r_[speech[0], speech[cuts + 1]]
    final_end = speech[-1] + 1 if len(audio) - speech[-1] - 1 >= int(0.15 * sample_rate) else len(audio)
    ends = np.r_[speech[cuts] + 1, final_end]
    padding = int(0.05 * sample_rate)
    return np.concatenate([audio[max(0, s - padding):min(len(audio), e + padding)]
                           for s, e in zip(starts, ends)]).astype(np.float32)


def encode_reference(encoder_path: Path, identity_path: Path, audio: np.ndarray) -> np.ndarray:
    """Runs codec_encoder.onnx; returns codes as [T, 16] int64."""
    identity = json.loads(identity_path.read_text())["encoder_sha256"]
    hasher = hashlib.sha256()
    with encoder_path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest != identity:
        raise ValueError(f"codec encoder SHA256 mismatch: {digest[:16]} != {identity[:16]}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    codes = session.run(["codes"], {"audio": audio[None]})[0]
    codes = codes[0, :, : (len(audio) + 1919) // 1920].T
    if codes.shape[1:] != (16,) or not 1 <= len(codes) <= 1000:
        raise ValueError(f"invalid reference codes shape {codes.shape}")
    if np.any((codes < 0) | (codes > 2047)):
        raise ValueError("reference codes out of range")
    return codes.astype(np.int64)


def merge_voice(voices_path: Path, name: str, reference_text: str, codes: np.ndarray,
                encoder_sha: str) -> str:
    """Adds or replaces one voice in voices.json. Returns 'added' or 'replaced'."""
    if voices_path.exists():
        data = json.loads(voices_path.read_text(encoding="utf-8"))
    else:
        data = {}
    if "encoder_sha256" in data and data["encoder_sha256"] != encoder_sha:
        raise ValueError("voices.json was built with a different codec encoder; rebuild all voices")
    data["encoder_sha256"] = encoder_sha
    data.setdefault("voices", [])
    entry = {"name": name, "reference_text": reference_text, "codes": codes.tolist()}
    for i, voice in enumerate(data["voices"]):
        if voice.get("name") == name:
            data["voices"][i] = entry
            action = "replaced"
            break
    else:
        data["voices"].append(entry)
        action = "added"
    voices_path.parent.mkdir(parents=True, exist_ok=True)
    voices_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", nargs="?", type=Path,
                        help="reference wav (speaker to clone)")
    parser.add_argument("transcript", nargs="?",
                        help="verbatim transcript of the reference audio")
    parser.add_argument("--name", help="voice name used later by --voice NAME")
    parser.add_argument("--model", type=Path, default=Path("/tmp/m5-official/onnx"),
                        help="dir with codec_encoder.onnx / codec_identity.json")
    parser.add_argument("--voices", type=Path, default=Path("voices.json"),
                        help="voices.json to update (created if missing)")
    parser.add_argument("--list", action="store_true", help="list cached voices and exit")
    args = parser.parse_args()

    if args.list:
        if not args.voices.exists():
            print(f"{args.voices}: not found")
            return 1
        data = json.loads(args.voices.read_text(encoding="utf-8"))
        for i, voice in enumerate(data.get("voices", [])):
            frames = len(voice.get("codes", []))
            print(f"[{i}] {voice.get('name')!r}  frames={frames}  "
                  f"ref={voice.get('reference_text', '')[:32]!r}")
        return 0

    if not args.reference or not args.transcript:
        parser.error("reference wav and transcript are required (or use --list)")
    if not args.name:
        args.name = args.reference.stem
    transcript = args.transcript.strip()
    if not transcript:
        parser.error("transcript is empty")

    audio = read_reference(args.reference, 24000)
    codes = encode_reference(args.model / "codec_encoder.onnx",
                             args.model / "codec_identity.json", audio)
    identity = json.loads((args.model / "codec_identity.json").read_text())["encoder_sha256"]
    action = merge_voice(args.voices, args.name, transcript, codes, identity)
    seconds = len(audio) / 24000
    print(f"{action} voice {args.name!r}: {seconds:.2f}s reference -> "
          f"{len(codes)} frames of codes, cached in {args.voices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
