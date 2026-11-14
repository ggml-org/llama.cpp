#!/usr/bin/env python3
"""Export the M5-TTS llama-30M temporal backbone for embedding input/output."""

import argparse
import json
from pathlib import Path

import numpy as np

from gguf import GGMLQuantizationType, GGUFWriter, quantize


WIDTH = 512
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 64
LAYERS = 8
CONTEXT = 2048
FFN = 2048
ROPE_BASE = 500_000.0
ROPE_SCALE = 32.0

FIELDS = (
    ("input_layernorm", "attn_norm", (WIDTH,), None),
    ("self_attn.q_proj", "attn_q", (WIDTH, WIDTH), HEADS),
    ("self_attn.k_proj", "attn_k", (KV_HEADS * HEAD_DIM, WIDTH), KV_HEADS),
    ("self_attn.v_proj", "attn_v", (KV_HEADS * HEAD_DIM, WIDTH), None),
    ("self_attn.o_proj", "attn_output", (WIDTH, WIDTH), None),
    ("post_attention_layernorm", "ffn_norm", (WIDTH,), None),
    ("mlp.gate_proj", "ffn_gate", (FFN, WIDTH), None),
    ("mlp.up_proj", "ffn_up", (FFN, WIDTH), None),
    ("mlp.down_proj", "ffn_down", (WIDTH, FFN), None),
)

def rope_factors():
    freq = ROPE_BASE ** (-np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
    wavelength = 2 * np.pi / freq
    smooth = np.clip((CONTEXT / wavelength - 1) / 3, 0, 1)
    return (1 / ((1 - smooth) / ROPE_SCALE + smooth)).astype(np.float32)


def interleave_rope(weight, heads):
    return weight.reshape(heads, 2, HEAD_DIM // 2, WIDTH).swapaxes(1, 2).reshape(weight.shape)


def convert(checkpoint, output):
    import torch
    from safetensors import safe_open

    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("backbone_flavor") != "llama-30M":
        raise ValueError("expected the M5-TTS llama-30M temporal backbone")

    tensors = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt") as source:
        def load(name, shape, heads=None):
            tensor = source.get_tensor("backbone." + name)
            if tensor.shape != shape or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f"unexpected shape or dtype for {name}: {tensor.shape}, {tensor.dtype}")
            tensor = tensor.float().numpy()
            return interleave_rope(tensor, heads) if heads else tensor

        tensors["output_norm.weight"] = load("norm.weight", (WIDTH,))
        for layer in range(LAYERS):
            for source_name, target_name, shape, heads in FIELDS:
                tensors[f"blk.{layer}.{target_name}.weight"] = load(
                    f"layers.{layer}.{source_name}.weight", shape, heads
                )

    write_gguf(output, tensors, rope_factors())


def write_gguf(output, tensors, factors, layers=LAYERS, name="M5-TTS temporal backbone", quant=None):
    writer = GGUFWriter(output, "llama")
    writer.add_name(name)
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(WIDTH)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_rope_freq_base(ROPE_BASE)
    writer.add_tokenizer_model("none")
    writer.add_vocab_size(1)
    # llama's loader expects an embedding tensor. Inference supplies hidden vectors directly.
    writer.add_tensor("token_embd.weight", np.zeros((1, WIDTH), dtype=np.float32))
    writer.add_tensor("rope_freqs.weight", factors)
    for name, tensor in tensors.items():
        if quant is not None and tensor.ndim == 2:
            writer.add_tensor(name, quantize(tensor, quant), raw_dtype=quant)
        else:
            writer.add_tensor(name, tensor)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def convert_onnx(model, output, depth=False, quant=None):
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(model).graph
    initializers = {item.name: item for item in graph.initializer}
    nodes = {node.name: node for node in graph.node if node.op_type == "MatMul"}

    aliases = {node.output[0]: node.input[0] for node in graph.node if node.op_type == "Identity"}

    def read(name, shape):
        while name in aliases:
            name = aliases[name]
        array = numpy_helper.to_array(initializers[name])
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError(f"unexpected ONNX initializer {name}: {array.shape}, {array.dtype}")
        return array

    tensors = {"output_norm.weight": read("core.norm.weight", (WIDTH,))}
    layers = 5 if depth else LAYERS
    for layer in range(layers):
        suffix = f"_{layer}" if layer else ""
        for source_name, target_name, shape, heads in FIELDS:
            if len(shape) == 1:
                tensor = read(f"core.layers.{layer}.{source_name}.weight", shape)
            else:
                projection = source_name.rsplit(".", 1)[-1]
                scope = "/core/mlp" if source_name.startswith("mlp.") else "/core"
                node = nodes[f"{scope}/{projection}{suffix}/MatMul"]
                tensor = read(node.input[1], shape[::-1]).T
                if heads:
                    tensor = interleave_rope(tensor, heads)
            tensors[f"blk.{layer}.{target_name}.weight"] = np.ascontiguousarray(tensor)
    if depth:
        write_gguf(output, tensors, rope_factors(), layers, "M5-TTS depth core", quant)
        return
    inv_freq = read("core.inv_freq", (HEAD_DIM // 2,))
    if not np.isfinite(inv_freq).all() or np.any(inv_freq <= 0):
        raise ValueError("invalid ONNX rotary frequencies")
    base_freq = ROPE_BASE ** (-np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
    write_gguf(output, tensors, base_freq / inv_freq, quant=quant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--depth", action="store_true", help="extract the five-layer core from depth_frame.onnx")
    parser.add_argument("--quant", choices=["Q8_0", "Q4_0"], help="quantize ONNX model projection weights")
    args = parser.parse_args()
    if args.quant and args.checkpoint.suffix != ".onnx":
        parser.error("--quant requires an ONNX input")
    if args.depth and args.checkpoint.suffix != ".onnx":
        parser.error("--depth requires depth_frame.onnx")
    if args.checkpoint.suffix == ".onnx":
        convert_onnx(args.checkpoint, args.output, args.depth,
                     GGMLQuantizationType[args.quant] if args.quant else None)
    else:
        convert(args.checkpoint, args.output)
