#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf
from gguf import GGUFValueType


TOKENIZER_KEYS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.chat_template",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.add_bos_token",
)


def field_payload(field: gguf.ReaderField):
    main_type = field.types[0]
    if main_type == GGUFValueType.STRING:
        return field.parts[-1].tobytes()

    if main_type == GGUFValueType.ARRAY:
        sub_type = field.types[-1]
        if sub_type == GGUFValueType.STRING:
            return [field.parts[idx].tobytes() for idx in field.data]
        return field.contents()

    return field.contents()


def copy_field(writer: gguf.GGUFWriter, reader: gguf.GGUFReader, key: str) -> None:
    field = reader.fields.get(key)
    if field is None:
        return

    value_type = field.types[0]
    sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
    writer.add_key_value(key, field_payload(field), value_type, sub_type=sub_type)


def scalar(reader: gguf.GGUFReader, key: str):
    field = reader.fields.get(key)
    if field is None:
        raise KeyError(f"missing GGUF metadata key: {key}")
    return field.contents()


def tensor_map(reader: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    return {tensor.name: tensor for tensor in reader.tensors}


class SafeTensorIndex:
    def __init__(self, directory: Path):
        self._files: dict[str, Path] = {}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    self._files[key] = path

        if not self._files:
            raise FileNotFoundError(f"no .safetensors files found in {directory}")

    def get(self, name: str) -> torch.Tensor:
        path = self._files.get(name)
        if path is None:
            raise KeyError(f"missing assistant tensor: {name}")

        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def tensor_to_numpy(index: SafeTensorIndex, name: str) -> np.ndarray:
    tensor = index.get(name)
    dtype = torch.float32 if tensor.ndim <= 1 else torch.float16
    return tensor.detach().to(dtype=dtype).contiguous().numpy()


def add_assistant_tensor(writer: gguf.GGUFWriter, index: SafeTensorIndex, src: str, dst: str) -> None:
    print(f"add {dst} <- {src}", flush=True)
    writer.add_tensor(dst, tensor_to_numpy(index, src))


def add_metadata(writer: gguf.GGUFWriter, target: gguf.GGUFReader, config: dict) -> None:
    text_config = config["text_config"]
    layer_types = text_config["layer_types"]
    is_swa = [layer_type == "sliding_attention" for layer_type in layer_types]
    n_kv_swa = text_config["num_key_value_heads"]
    n_kv_full = text_config["num_global_key_value_heads"]

    writer.add_name("Gemma 4 31B IT Assistant MTP")
    writer.add_description("Gemma 4 assistant decoder converted for native llama.cpp MTP.")
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16))
    writer.add_quantization_version(int(scalar(target, "general.quantization_version")))

    writer.add_vocab_size(int(text_config["vocab_size"]))
    writer.add_context_length(int(scalar(target, "gemma4.context_length")))
    writer.add_embedding_length(int(text_config["hidden_size"]))
    writer.add_embedding_length_out(int(config["backbone_hidden_size"]))
    writer.add_block_count(int(text_config["num_hidden_layers"]))
    writer.add_feed_forward_length(int(text_config["intermediate_size"]))
    writer.add_head_count(int(text_config["num_attention_heads"]))
    writer.add_head_count_kv([n_kv_swa if layer_is_swa else n_kv_full for layer_is_swa in is_swa])

    writer.add_rope_freq_base(float(scalar(target, "gemma4.rope.freq_base")))
    writer.add_rope_freq_base_swa(float(scalar(target, "gemma4.rope.freq_base_swa")))
    writer.add_key_length(int(text_config["global_head_dim"]))
    writer.add_value_length(int(text_config["global_head_dim"]))
    writer.add_key_length_swa(int(text_config["head_dim"]))
    writer.add_value_length_swa(int(text_config["head_dim"]))
    writer.add_rope_dimension_count(int(scalar(target, "gemma4.rope.dimension_count")))
    writer.add_rope_dimension_count_swa(int(scalar(target, "gemma4.rope.dimension_count_swa")))
    writer.add_sliding_window(int(text_config["sliding_window"]))
    writer.add_sliding_window_pattern(is_swa)
    writer.add_layer_norm_rms_eps(float(text_config["rms_norm_eps"]))

    for key in TOKENIZER_KEYS:
        copy_field(writer, target, key)


def add_tensors(writer: gguf.GGUFWriter, target: gguf.GGUFReader, config: dict, index: SafeTensorIndex) -> None:
    target_tensors = tensor_map(target)

    add_assistant_tensor(writer, index, "model.embed_tokens.weight", "token_embd.weight")
    add_assistant_tensor(writer, index, "model.norm.weight", "output_norm.weight")
    add_assistant_tensor(writer, index, "pre_projection.weight", "mtp_pre_proj.weight")
    add_assistant_tensor(writer, index, "post_projection.weight", "mtp_post_proj.weight")

    token_embd = target_tensors["token_embd.weight"]
    print("add mtp_input_embd.weight <- target token_embd.weight", flush=True)
    writer.add_tensor("mtp_input_embd.weight", token_embd.data, raw_dtype=token_embd.tensor_type)

    rope_freqs = target_tensors["rope_freqs.weight"]
    print("add rope_freqs.weight <- target rope_freqs.weight", flush=True)
    writer.add_tensor("rope_freqs.weight", np.asarray(rope_freqs.data, dtype=np.float32))

    n_layer = int(config["text_config"]["num_hidden_layers"])
    for i in range(n_layer):
        prefix = f"model.layers.{i}"
        add_assistant_tensor(writer, index, f"{prefix}.input_layernorm.weight", f"blk.{i}.attn_norm.weight")
        add_assistant_tensor(writer, index, f"{prefix}.post_attention_layernorm.weight", f"blk.{i}.post_attention_norm.weight")
        add_assistant_tensor(writer, index, f"{prefix}.self_attn.q_norm.weight", f"blk.{i}.attn_q_norm.weight")
        add_assistant_tensor(writer, index, f"{prefix}.self_attn.q_proj.weight", f"blk.{i}.attn_q.weight")
        add_assistant_tensor(writer, index, f"{prefix}.self_attn.o_proj.weight", f"blk.{i}.attn_output.weight")
        add_assistant_tensor(writer, index, f"{prefix}.pre_feedforward_layernorm.weight", f"blk.{i}.ffn_norm.weight")
        add_assistant_tensor(writer, index, f"{prefix}.mlp.gate_proj.weight", f"blk.{i}.ffn_gate.weight")
        add_assistant_tensor(writer, index, f"{prefix}.mlp.up_proj.weight", f"blk.{i}.ffn_up.weight")
        add_assistant_tensor(writer, index, f"{prefix}.mlp.down_proj.weight", f"blk.{i}.ffn_down.weight")
        add_assistant_tensor(writer, index, f"{prefix}.post_feedforward_layernorm.weight", f"blk.{i}.post_ffw_norm.weight")
        add_assistant_tensor(writer, index, f"{prefix}.layer_scalar", f"blk.{i}.layer_output_scale.weight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Gemma 4 assistant safetensors to a llama.cpp Gemma4 MTP GGUF.")
    parser.add_argument("--assistant-dir", required=True, type=Path, help="HF snapshot directory for google/gemma-4-31B-it-assistant")
    parser.add_argument("--target-gguf", required=True, type=Path, help="Target Gemma 4 GGUF to copy tokenizer and backbone embeddings from")
    parser.add_argument("--outfile", required=True, type=Path, help="Output assistant MTP GGUF")
    args = parser.parse_args()

    with (args.assistant_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    target = gguf.GGUFReader(args.target_gguf)
    index = SafeTensorIndex(args.assistant_dir)
    writer = gguf.GGUFWriter(args.outfile, "gemma4_mtp", use_temp_file=True)

    add_metadata(writer, target, config)
    add_tensors(writer, target, config, index)

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


if __name__ == "__main__":
    main()
