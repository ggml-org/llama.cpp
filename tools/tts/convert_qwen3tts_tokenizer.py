#!/usr/bin/env python3
"""
Convert Qwen3-TTS-Tokenizer-12Hz model to GGUF format.

This produces the vocoder GGUF file used by llama-qwen3tts to decode
speech codes into audio waveforms.

Usage:
    python tools/tts/convert_qwen3tts_tokenizer.py \
        --input models/Qwen3-TTS-Tokenizer-12Hz \
        --output models/qwen3-tts-tokenizer-f16.gguf \
        --type f16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DIRECT_TENSOR_MAP = {
    "encoder.downsample.conv.weight": "tok_enc.downsample.weight",
    "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight": "tok_enc.vq_acoustic.input_proj.weight",
    "encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight": "tok_enc.vq_acoustic.output_proj.weight",
    "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight": "tok_enc.vq_semantic.input_proj.weight",
    "encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight": "tok_enc.vq_semantic.output_proj.weight",
    "decoder.pre_conv.conv.bias": "tok_dec.pre_conv.bias",
    "decoder.pre_conv.conv.weight": "tok_dec.pre_conv.weight",
    "decoder.pre_transformer.input_proj.bias": "tok_dec.pre_tfm.input_proj.bias",
    "decoder.pre_transformer.input_proj.weight": "tok_dec.pre_tfm.input_proj.weight",
    "decoder.pre_transformer.output_proj.bias": "tok_dec.pre_tfm.output_proj.bias",
    "decoder.pre_transformer.output_proj.weight": "tok_dec.pre_tfm.output_proj.weight",
    "decoder.pre_transformer.norm.weight": "tok_dec.pre_tfm.norm.weight",
    "decoder.quantizer.rvq_first.input_proj.weight": "tok_dec.vq_first.input_proj.weight",
    "decoder.quantizer.rvq_first.output_proj.weight": "tok_dec.vq_first.output_proj.weight",
    "decoder.quantizer.rvq_rest.input_proj.weight": "tok_dec.vq_rest.input_proj.weight",
    "decoder.quantizer.rvq_rest.output_proj.weight": "tok_dec.vq_rest.output_proj.weight",
    "decoder.decoder.0.conv.weight": "tok_dec.dec.0.conv.weight",
    "decoder.decoder.0.conv.bias": "tok_dec.dec.0.conv.bias",
    "decoder.decoder.5.alpha": "tok_dec.dec.5.snake.alpha",
    "decoder.decoder.5.beta": "tok_dec.dec.5.snake.beta",
    "decoder.decoder.6.conv.weight": "tok_dec.dec.6.conv.weight",
    "decoder.decoder.6.conv.bias": "tok_dec.dec.6.conv.bias",
}

ENCODER_PATTERNS = [
    (r"encoder\.encoder\.layers\.(\d+)\.conv\.weight", "tok_enc.conv.{}.weight"),
    (r"encoder\.encoder\.layers\.(\d+)\.conv\.bias", "tok_enc.conv.{}.bias"),
    (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.weight", "tok_enc.res.{}.blk.{}.weight"),
    (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.bias", "tok_enc.res.{}.blk.{}.bias"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.weight", "tok_enc.blk.{}.attn_norm.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.bias", "tok_enc.blk.{}.attn_norm.bias"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight", "tok_enc.blk.{}.ffn_norm.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.bias", "tok_enc.blk.{}.ffn_norm.bias"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight", "tok_enc.blk.{}.attn_q.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight", "tok_enc.blk.{}.attn_k.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight", "tok_enc.blk.{}.attn_v.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight", "tok_enc.blk.{}.attn_output.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale", "tok_enc.blk.{}.attn_scale"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc1\.weight", "tok_enc.blk.{}.ffn_up.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc2\.weight", "tok_enc.blk.{}.ffn_down.weight"),
    (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale", "tok_enc.blk.{}.ffn_scale"),
    (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum", "tok_enc.vq_acoustic.{}.codebook"),
    (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.cluster_usage", "tok_enc.vq_acoustic.{}.usage"),
    (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.initialized", "tok_enc.vq_acoustic.{}.initialized"),
    (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum", "tok_enc.vq_semantic.{}.codebook"),
    (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.cluster_usage", "tok_enc.vq_semantic.{}.usage"),
    (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.initialized", "tok_enc.vq_semantic.{}.initialized"),
]

DECODER_PATTERNS = [
    (r"decoder\.decoder\.(\d+)\.block\.0\.alpha", "tok_dec.dec.{}.snake.alpha"),
    (r"decoder\.decoder\.(\d+)\.block\.0\.beta", "tok_dec.dec.{}.snake.beta"),
    (r"decoder\.decoder\.(\d+)\.block\.1\.conv\.weight", "tok_dec.dec.{}.conv_t.weight"),
    (r"decoder\.decoder\.(\d+)\.block\.1\.conv\.bias", "tok_dec.dec.{}.conv_t.bias"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act1\.alpha", "tok_dec.dec.{}.res.{}.act1.alpha"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act1\.beta", "tok_dec.dec.{}.res.{}.act1.beta"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act2\.alpha", "tok_dec.dec.{}.res.{}.act2.alpha"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act2\.beta", "tok_dec.dec.{}.res.{}.act2.beta"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv1\.conv\.weight", "tok_dec.dec.{}.res.{}.conv1.weight"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv1\.conv\.bias", "tok_dec.dec.{}.res.{}.conv1.bias"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv2\.conv\.weight", "tok_dec.dec.{}.res.{}.conv2.weight"),
    (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv2\.conv\.bias", "tok_dec.dec.{}.res.{}.conv2.bias"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.input_layernorm\.weight", "tok_dec.pre_tfm.blk.{}.attn_norm.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight", "tok_dec.pre_tfm.blk.{}.ffn_norm.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_q.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_k.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_v.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_output.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale", "tok_dec.pre_tfm.blk.{}.attn_scale"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.gate_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_gate.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.up_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_up.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.down_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_down.weight"),
    (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale", "tok_dec.pre_tfm.blk.{}.ffn_scale"),
    (r"decoder\.quantizer\.rvq_first\.vq\.layers\.(\d+)\._codebook\.embedding_sum", "tok_dec.vq_first.{}.codebook"),
    (r"decoder\.quantizer\.rvq_first\.vq\.layers\.(\d+)\._codebook\.cluster_usage", "tok_dec.vq_first.{}.usage"),
    (r"decoder\.quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.embedding_sum", "tok_dec.vq_rest.{}.codebook"),
    (r"decoder\.quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.cluster_usage", "tok_dec.vq_rest.{}.usage"),
    (r"decoder\.upsample\.(\d+)\.0\.conv\.weight", "tok_dec.upsample.{}.conv.weight"),
    (r"decoder\.upsample\.(\d+)\.0\.conv\.bias", "tok_dec.upsample.{}.conv.bias"),
    (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.weight", "tok_dec.upsample.{}.dwconv.weight"),
    (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.bias", "tok_dec.upsample.{}.dwconv.bias"),
    (r"decoder\.upsample\.(\d+)\.1\.gamma", "tok_dec.upsample.{}.gamma"),
    (r"decoder\.upsample\.(\d+)\.1\.norm\.weight", "tok_dec.upsample.{}.norm.weight"),
    (r"decoder\.upsample\.(\d+)\.1\.norm\.bias", "tok_dec.upsample.{}.norm.bias"),
    (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.weight", "tok_dec.upsample.{}.pwconv1.weight"),
    (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.bias", "tok_dec.upsample.{}.pwconv1.bias"),
    (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.weight", "tok_dec.upsample.{}.pwconv2.weight"),
    (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.bias", "tok_dec.upsample.{}.pwconv2.bias"),
]


def map_tensor_name(hf_name: str) -> str | None:
    if hf_name in DIRECT_TENSOR_MAP:
        return DIRECT_TENSOR_MAP[hf_name]

    all_patterns = ENCODER_PATTERNS + DECODER_PATTERNS
    for pattern, template in all_patterns:
        m = re.match(pattern, hf_name)
        if m:
            return template.format(*m.groups())

    return None


def get_tensors(input_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    tokenizer_dir = input_dir / "speech_tokenizer"
    search_dir = tokenizer_dir if tokenizer_dir.exists() else input_dir
    safetensor_files = sorted(search_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {search_dir}")
    for sf_path in safetensor_files:
        logger.info("Loading tensors from %s", sf_path.name)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def convert_dtype(tensor: torch.Tensor, output_type: str, tensor_name: str = "") -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    data = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
    n_dims = len(data.shape)

    if n_dims <= 1:
        return data.astype(np.float32), gguf.GGMLQuantizationType.F32

    # Encoder tensors benefit from F32 precision to avoid VQ code flips
    if tensor_name.startswith("tok_enc."):
        return data.astype(np.float32), gguf.GGMLQuantizationType.F32

    if output_type == "f32":
        return data.astype(np.float32), gguf.GGMLQuantizationType.F32
    elif output_type == "bf16":
        bf16_type = gguf.GGMLQuantizationType.BF16
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy(), bf16_type
        t_bf16 = tensor.float().to(torch.bfloat16)
        return t_bf16.view(torch.uint16).numpy(), bf16_type
    elif output_type == "f16":
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16
    elif output_type == "q8_0":
        if any(x in tensor_name for x in ["codebook", "_norm", "norm.", "scale", "alpha", "beta"]):
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        data = data.astype(np.float32)
        try:
            quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            return quantized, gguf.GGMLQuantizationType.Q8_0
        except Exception:
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
    else:
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16


def convert(input_dir: Path, output_path: Path, output_type: str = "f16") -> None:
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    encoder_config = config.get("encoder_config", {})
    decoder_config = config.get("decoder_config", {})

    logger.info("Converting Qwen3-TTS-Tokenizer-12Hz to GGUF format")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arch = "qwen3-tts-tokenizer"
    writer = gguf.GGUFWriter(path=None, arch=arch)

    writer.add_name("Qwen3-TTS-Tokenizer-12Hz")
    writer.add_type(gguf.GGUFType.MODEL)
    ftype_map = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
                 "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}
    writer.add_file_type(ftype_map.get(output_type, gguf.LlamaFileType.MOSTLY_F16))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    writer.add_uint32(f"{arch}.num_codebooks", decoder_config.get("num_quantizers", 16))
    writer.add_uint32(f"{arch}.codebook_size", encoder_config.get("codebook_size", 2048))
    writer.add_uint32(f"{arch}.sample_rate", config.get("input_sample_rate", 24000))
    writer.add_float32(f"{arch}.frame_rate", encoder_config.get("_frame_rate", 12.5))
    writer.add_uint32(f"{arch}.encoder.hidden_size", encoder_config.get("hidden_size", 512))
    writer.add_uint32(f"{arch}.encoder.num_layers", encoder_config.get("num_hidden_layers", 8))
    writer.add_uint32(f"{arch}.encoder.num_heads", encoder_config.get("num_attention_heads", 8))
    writer.add_uint32(f"{arch}.encoder.num_quantizers", encoder_config.get("num_quantizers", 32))
    writer.add_uint32(f"{arch}.encoder.valid_quantizers", config.get("encoder_valid_num_quantizers", 16))
    writer.add_uint32(f"{arch}.encoder.codebook_dim", encoder_config.get("codebook_dim", 256))
    writer.add_uint32(f"{arch}.decoder.hidden_size", decoder_config.get("hidden_size", 512))
    writer.add_uint32(f"{arch}.decoder.num_layers", decoder_config.get("num_hidden_layers", 8))
    writer.add_uint32(f"{arch}.decoder.num_heads", decoder_config.get("num_attention_heads", 16))
    writer.add_uint32(f"{arch}.decoder.latent_dim", decoder_config.get("latent_dim", 1024))
    writer.add_uint32(f"{arch}.decoder.codebook_dim", decoder_config.get("codebook_dim", 512))
    writer.add_uint32(f"{arch}.decoder.semantic_codebook_size", decoder_config.get("semantic_codebook_size", 4096))
    writer.add_array(f"{arch}.upsample_rates", decoder_config.get("upsample_rates", [8, 5, 4, 3]))

    all_tensors = list(get_tensors(input_dir))

    codebook_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for hf_name, tensor in all_tensors:
        if "embedding_sum" in hf_name or "embed_sum" in hf_name:
            sum_key = "embedding_sum" if "embedding_sum" in hf_name else "embed_sum"
            base = hf_name.replace(sum_key, "")
            codebook_pairs.setdefault(base, {})["embed_sum"] = tensor
        elif "cluster_usage" in hf_name:
            base = hf_name.replace("cluster_usage", "")
            codebook_pairs.setdefault(base, {})["cluster_usage"] = tensor

    tensor_count = 0
    skipped_count = 0

    for hf_name, tensor in tqdm(all_tensors, desc="Converting"):
        ggml_name = map_tensor_name(hf_name)
        if ggml_name is None:
            skipped_count += 1
            continue

        if "cluster_usage" in hf_name:
            skipped_count += 1
            continue

        if "embedding_sum" in hf_name or "embed_sum" in hf_name:
            sum_key = "embedding_sum" if "embedding_sum" in hf_name else "embed_sum"
            base = hf_name.replace(sum_key, "")
            pair = codebook_pairs.get(base, {})
            if "cluster_usage" in pair:
                tensor = pair["embed_sum"] / pair["cluster_usage"].clamp(min=1e-5).unsqueeze(1)

        data, dtype = convert_dtype(tensor, output_type, ggml_name)
        writer.add_tensor(ggml_name, data, raw_dtype=dtype)
        tensor_count += 1

    logger.info("Converted %d tensors, skipped %d", tensor_count, skipped_count)

    logger.info("Writing GGUF file to %s", output_path)
    writer.write_header_to_file(path=output_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS-Tokenizer-12Hz model to GGUF")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output GGUF file path")
    parser.add_argument("--type", "-t", choices=["f16", "f32", "bf16", "q8_0"], default="f16", help="Output data type")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    convert(args.input, args.output, args.type)


if __name__ == "__main__":
    main()
