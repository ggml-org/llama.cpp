#!/usr/bin/env python3
"""Prepare the official Android tokenizer and a reference voice for app_process."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m5_source", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("transcript")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sys.path[:0] = [str(args.m5_source), str(args.m5_source / "android")]
    from infer_onnx import read_reference
    from package_model import export_tokenizer

    encoder = args.model / "codec_encoder.onnx"
    identity = json.loads((args.model / "codec_identity.json").read_text())["encoder_sha256"]
    with encoder.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != identity:
            raise ValueError("codec encoder SHA256 mismatch")
    transcript = args.transcript.strip()
    if not transcript:
        raise ValueError("reference transcript is empty")
    audio = read_reference(args.reference, 24000)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(encoder), options, providers=["CPUExecutionProvider"])
    codes = session.run(["codes"], {"audio": audio[None]})[0]
    codes = codes[0, :, : (len(audio) + 1919) // 1920].T
    if codes.shape[1:] != (16,) or not 1 <= len(codes) <= 1000 or np.any((codes < 0) | (codes > 2047)):
        raise ValueError("invalid reference audio codes")
    args.output.mkdir(parents=True, exist_ok=True)
    export_tokenizer(args.model / "tokenizer.json", args.output / "tokenizer.bin")
    voice = {"name": args.reference.stem, "reference_text": transcript, "codes": codes.tolist()}
    (args.output / "voices.json").write_text(
        json.dumps({"encoder_sha256": identity, "voices": [voice]}, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
