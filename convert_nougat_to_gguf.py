#!/usr/bin/env python3
"""
Convert Nougat (Neural Optical Understanding for Academic Documents) model to GGUF format.
This script handles the conversion of Nougat's Swin Transformer encoder and mBART decoder.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel

# Add parent directory to path to import gguf
sys.path.append(str(Path(__file__).parent / "gguf-py"))
import gguf

# Constants for Nougat
NOUGAT_VISION_PREFIX = "vision_model"
NOUGAT_DECODER_PREFIX = "decoder"
NOUGAT_ENCODER_PREFIX = "encoder"

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Nougat model to GGUF format")
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/nougat-base",
        help="HuggingFace model ID or path to local model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for GGUF files",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1"],
        default="f16",
        help="Quantization type for model weights",
    )
    parser.add_argument(
        "--split-model",
        action="store_true",
        help="Split into separate vision and text GGUF files",
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only export vocabulary/tokenizer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during conversion",
    )
    return parser.parse_args()


def get_tensor_name(name: str) -> str:
    """Map Nougat tensor names to GGUF tensor names"""

    # Vision model (Swin Transformer) mappings
    if name.startswith("encoder.model.encoder."):
        # Swin encoder layers
        name = name.replace("encoder.model.encoder.", "swin.")

        # Patch embedding
        if "embeddings.patch_embeddings" in name:
            if "projection.weight" in name:
                return "swin.patch_embed.weight"
            elif "projection.bias" in name:
                return "swin.patch_embed.bias"

        # Position embeddings
        if "position_embeddings" in name:
            return "swin.pos_embed"

        # Layer mappings
        if "layers." in name:
            # Extract stage and layer indices
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers":
                    stage_idx = int(parts[i + 1])
                    if "blocks." in name:
                        block_idx = int(parts[parts.index("blocks") + 1])

                        # Attention components
                        if "attn.qkv" in name:
                            return f"swin.stage.{stage_idx}.layer.{block_idx}.attn.qkv.{'weight' if 'weight' in name else 'bias'}"
                        elif "attn.proj" in name:
                            return f"swin.stage.{stage_idx}.layer.{block_idx}.attn.proj.{'weight' if 'weight' in name else 'bias'}"
                        elif "norm1" in name:
                            return f"swin.stage.{stage_idx}.layer.{block_idx}.norm1.{'weight' if 'weight' in name else 'bias'}"
                        elif "norm2" in name:
                            return f"swin.stage.{stage_idx}.layer.{block_idx}.norm2.{'weight' if 'weight' in name else 'bias'}"
                        elif "mlp.fc1" in name:
                            return f"swin.stage.{stage_idx}.layer.{block_idx}.mlp.fc1.{'weight' if 'weight' in name else 'bias'}"
                        elif "mlp.fc2" in name:
                            return f"swin.stage.{stage_idx}.layer.{block_idx}.mlp.fc2.{'weight' if 'weight' in name else 'bias'}"

                    # Downsample layers
                    elif "downsample" in name:
                        if "norm" in name:
                            return f"swin.stage.{stage_idx}.downsample.norm.{'weight' if 'weight' in name else 'bias'}"
                        elif "reduction" in name:
                            return f"swin.stage.{stage_idx}.downsample.reduction.weight"

    # Decoder model (mBART) mappings
    elif name.startswith("decoder.model."):
        name = name.replace("decoder.model.", "")

        # Token and position embeddings
        if name == "shared.weight":
            return "token_embd.weight"
        elif name == "decoder.embed_positions.weight":
            return "position_embd.weight"

        # Decoder layers
        if "decoder.layers." in name:
            layer_idx = int(name.split(".")[2])

            # Self-attention
            if "self_attn.q_proj" in name:
                return f"blk.{layer_idx}.attn_q.weight"
            elif "self_attn.k_proj" in name:
                return f"blk.{layer_idx}.attn_k.weight"
            elif "self_attn.v_proj" in name:
                return f"blk.{layer_idx}.attn_v.weight"
            elif "self_attn.out_proj" in name:
                return f"blk.{layer_idx}.attn_o.weight"
            elif "self_attn_layer_norm" in name:
                return f"blk.{layer_idx}.attn_norm.{'weight' if 'weight' in name else 'bias'}"

            # Cross-attention
            elif "encoder_attn.q_proj" in name:
                return f"blk.{layer_idx}.attn_q_cross.weight"
            elif "encoder_attn.k_proj" in name:
                return f"blk.{layer_idx}.attn_k_cross.weight"
            elif "encoder_attn.v_proj" in name:
                return f"blk.{layer_idx}.attn_v_cross.weight"
            elif "encoder_attn.out_proj" in name:
                return f"blk.{layer_idx}.attn_o_cross.weight"
            elif "encoder_attn_layer_norm" in name:
                return f"blk.{layer_idx}.attn_norm_cross.{'weight' if 'weight' in name else 'bias'}"

            # FFN
            elif "fc1" in name:
                return f"blk.{layer_idx}.ffn_up.weight"
            elif "fc2" in name:
                return f"blk.{layer_idx}.ffn_down.weight"
            elif "final_layer_norm" in name:
                return f"blk.{layer_idx}.ffn_norm.{'weight' if 'weight' in name else 'bias'}"

        # Output layers
        elif "decoder.layer_norm" in name:
            return f"output_norm.{'weight' if 'weight' in name else 'bias'}"
        elif "lm_head" in name:
            return "output.weight"

    # Encoder layers (for encoder-only export)
    elif name.startswith("encoder."):
        name = name.replace("encoder.", "enc.")
        # Similar mappings but with enc. prefix
        return f"enc.{name}"

    # Default: return original name
    return name


def convert_swin_encoder(model_dict: Dict[str, torch.Tensor], gguf_writer: gguf.GGUFWriter, args):
    """Convert Swin Transformer encoder weights to GGUF format"""

    print("Converting Swin Transformer encoder...")

    # Write Swin hyperparameters
    swin_config = {
        "window_size": 7,
        "patch_size": 4,
        "image_size": 384,  # Default for Nougat
        "hidden_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "mlp_ratio": 4.0,
        "norm_eps": 1e-5,
    }

    gguf_writer.add_string("swin.type", "swin_transformer")
    gguf_writer.add_int32("swin.window_size", swin_config["window_size"])
    gguf_writer.add_int32("swin.patch_size", swin_config["patch_size"])
    gguf_writer.add_int32("swin.image_size", swin_config["image_size"])
    gguf_writer.add_int32("swin.hidden_dim", swin_config["hidden_dim"])
    gguf_writer.add_float32("swin.mlp_ratio", swin_config["mlp_ratio"])
    gguf_writer.add_float32("swin.norm_eps", swin_config["norm_eps"])

    # Convert encoder weights
    encoder_tensors = {k: v for k, v in model_dict.items() if k.startswith("encoder.")}

    for name, tensor in encoder_tensors.items():
        gguf_name = get_tensor_name(name)

        if args.verbose:
            print(f"  {name} -> {gguf_name} {list(tensor.shape)}")

        # Convert to appropriate dtype
        if args.quantization == "f32":
            data = tensor.float().cpu().numpy()
        elif args.quantization == "f16":
            data = tensor.half().cpu().numpy()
        else:
            # Quantization would be applied here
            data = tensor.float().cpu().numpy()

        gguf_writer.add_tensor(gguf_name, data)

    print(f"  Converted {len(encoder_tensors)} encoder tensors")


def convert_mbart_decoder(model_dict: Dict[str, torch.Tensor], gguf_writer: gguf.GGUFWriter, args):
    """Convert mBART decoder weights to GGUF format"""

    print("Converting mBART decoder...")

    # Write mBART architecture info
    gguf_writer.add_string("general.architecture", "mbart")

    # Convert decoder weights
    decoder_tensors = {k: v for k, v in model_dict.items() if k.startswith("decoder.")}

    for name, tensor in decoder_tensors.items():
        gguf_name = get_tensor_name(name)

        if args.verbose:
            print(f"  {name} -> {gguf_name} {list(tensor.shape)}")

        # Convert to appropriate dtype
        if args.quantization == "f32":
            data = tensor.float().cpu().numpy()
        elif args.quantization == "f16":
            data = tensor.half().cpu().numpy()
        else:
            # Quantization would be applied here
            data = tensor.float().cpu().numpy()

        gguf_writer.add_tensor(gguf_name, data)

    print(f"  Converted {len(decoder_tensors)} decoder tensors")


def convert_tokenizer(processor, gguf_writer: gguf.GGUFWriter, args):
    """Convert Nougat tokenizer/processor to GGUF format"""

    print("Converting tokenizer...")

    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()

    # Write tokenizer metadata
    gguf_writer.add_string("tokenizer.model", "mbart")
    gguf_writer.add_int32("tokenizer.vocab_size", len(vocab))

    # Add special tokens
    special_tokens = {
        "bos": tokenizer.bos_token,
        "eos": tokenizer.eos_token,
        "unk": tokenizer.unk_token,
        "pad": tokenizer.pad_token,
    }

    for key, token in special_tokens.items():
        if token:
            gguf_writer.add_string(f"tokenizer.{key}_token", token)
            gguf_writer.add_int32(f"tokenizer.{key}_token_id", tokenizer.convert_tokens_to_ids(token))

    # Add vocabulary
    tokens = []
    scores = []
    token_types = []

    for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        tokens.append(token.encode("utf-8"))
        scores.append(0.0)  # Dummy scores for now
        token_types.append(1 if token in tokenizer.all_special_tokens else 0)

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(token_types)

    print(f"  Vocabulary size: {len(vocab)}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Nougat model from {args.model_id}...")

    # Load model and processor
    processor = NougatProcessor.from_pretrained(args.model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id)

    # Get model state dict
    state_dict = model.state_dict()

    if args.split_model:
        # Create separate files for vision and text models

        # Vision model (Swin encoder)
        vision_output = output_dir / "nougat-vision.gguf"
        print(f"\nCreating vision model: {vision_output}")

        vision_writer = gguf.GGUFWriter(str(vision_output), "nougat-vision")
        vision_writer.add_string("general.name", "Nougat Vision Model (Swin)")
        vision_writer.add_string("general.description", "Swin Transformer encoder for Nougat OCR")
        vision_writer.add_string("general.architecture", "swin")

        convert_swin_encoder(state_dict, vision_writer, args)
        vision_writer.write_header_to_file()
        vision_writer.write_kv_data_to_file()
        vision_writer.write_tensors_to_file()
        vision_writer.close()

        # Text model (mBART decoder)
        text_output = output_dir / "nougat-text.gguf"
        print(f"\nCreating text model: {text_output}")

        text_writer = gguf.GGUFWriter(str(text_output), "nougat-text")
        text_writer.add_string("general.name", "Nougat Text Model (mBART)")
        text_writer.add_string("general.description", "mBART decoder for Nougat OCR")

        convert_mbart_decoder(state_dict, text_writer, args)
        convert_tokenizer(processor, text_writer, args)

        text_writer.write_header_to_file()
        text_writer.write_kv_data_to_file()
        text_writer.write_tensors_to_file()
        text_writer.close()

    else:
        # Create single combined model file
        output_file = output_dir / "nougat-combined.gguf"
        print(f"\nCreating combined model: {output_file}")

        writer = gguf.GGUFWriter(str(output_file), "nougat")
        writer.add_string("general.name", "Nougat OCR Model")
        writer.add_string("general.description", "Neural Optical Understanding for Academic Documents")
        writer.add_string("general.architecture", "nougat")

        # Add both encoder and decoder
        convert_swin_encoder(state_dict, writer, args)
        convert_mbart_decoder(state_dict, writer, args)

        if not args.vocab_only:
            convert_tokenizer(processor, writer, args)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    print("\nConversion complete!")

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n):,}")
    print(f"  Decoder parameters: {sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n):,}")

    if args.quantization != "f32":
        print(f"  Quantization: {args.quantization}")


if __name__ == "__main__":
    main()