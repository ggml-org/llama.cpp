#!/usr/bin/env python3
"""Capture an official depth frame and its sixteen sampling decisions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("m5_source", "model", "voices", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--text", default="你好，今天天气很好。")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    sys.path.insert(0, str(args.m5_source.resolve()))
    from infer_onnx import conditioned_text
    from tokenizers import Tokenizer

    root = args.model
    voice = json.loads(args.voices.read_text())["voices"][0]
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    ids = tokenizer.encode(conditioned_text(voice["reference_text"] + args.text, "Chinese")).ids
    text = np.load(root / "text_embeddings.npy", mmap_mode="r")[ids]
    audio = np.load(root / "audio_embeddings.npy", mmap_mode="r")
    codes = np.asarray(voice["codes"])
    prompt = audio[codes + 1 + np.arange(16) * 2051].sum(axis=1)
    prefix = np.concatenate([text, prompt])[None]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    backbone = ort.InferenceSession(str(root / "backbone.onnx"), options, providers=["CPUExecutionProvider"])
    hidden = backbone.run(["last_hidden"], {
        "hidden": prefix,
        "positions": np.arange(prefix.shape[1], dtype=np.int64)[None],
        "past": np.empty((8, 2, 1, 2, 0, 64), dtype=np.float32),
    })[0]

    graph = onnx.load(root / "depth_frame.onnx", load_external_data=True)
    nodes = {node.name: node for node in graph.graph.node}
    outputs = ["codes", "next_hidden"]

    def expose(name, shape):
        graph.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        outputs.append(name)

    for index in range(16):
        suffix = f"_{index}" if index else ""
        expose(nodes[f"/ArgMax{suffix}"].input[0], [1, 2051])
    for index in range(15):
        layer_suffix = f"_{index * 5}" if index else ""
        norm_suffix = f"_{index}" if index else ""
        shape = [1, 1 if index else 2, 512]
        expose(nodes[f"/core/input_layernorm{layer_suffix}/Cast"].input[0], shape)
        expose(nodes[f"/core/norm{norm_suffix}/Mul_1"].output[0], shape)
    session = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
    noise = np.random.default_rng(args.seed).exponential(size=(16, 1, 2051)).astype(np.float32)
    noise = np.maximum(noise, np.finfo(np.float32).tiny)
    feed = {"hidden": hidden, "noise": noise, "allow_eos": np.array([False])}
    codes, next_hidden, *extra = session.run(outputs, feed)
    scores = np.stack(extra[:16])
    np.testing.assert_array_equal(np.argmax(scores, axis=-1).reshape(1, 16), codes)
    np.savez(args.output, **feed, codes=codes, next_hidden=next_hidden, scores=scores,
             first_local=extra[16], first_decoded=extra[17],
             local_sequence=np.concatenate(extra[16::2], axis=1),
             decoded_steps=np.concatenate([step[:, -1:] for step in extra[17::2]], axis=1))
    print(f"prefix={prefix.shape[1]} codes={codes.tolist()}")


if __name__ == "__main__":
    main()
