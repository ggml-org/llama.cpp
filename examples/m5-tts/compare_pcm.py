#!/usr/bin/env python3
"""Compare aligned Android benchmark WAVs with identical sample counts."""

import argparse
import json
import wave

import numpy as np


def read_pcm(path):
    with wave.open(path, "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 24000):
            raise ValueError(f"expected mono 24 kHz PCM16: {path}")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference, candidate = map(read_pcm, (args.reference, args.candidate))
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError(f"sample count mismatch: {reference.size}, {candidate.size}")
    power = np.dot(reference, reference)
    if not power:
        raise ValueError("silent reference")
    error = candidate - reference
    noise = np.dot(error, error)
    print(json.dumps({
        "samples": int(reference.size),
        "sample_exact": bool(not noise),
        "max_abs_pcm16": int(np.abs(error).max()),
        "snr_db": float(10 * np.log10(power / noise)) if noise else None,
    }))


if __name__ == "__main__":
    main()
