#!/usr/bin/env python3
"""
Nougat Model Surgery Script
Splits the Nougat model into separate vision encoder (Swin) and text decoder (mBART) components.
Also creates the multimodal projector that connects them.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, NougatProcessor

# Add parent directory to import gguf
sys.path.append(str(Path(__file__).parent.parent / "gguf-py"))
import gguf


class NougatModelSurgeon:
    """Handles splitting and converting Nougat model components"""

    def __init__(self, model_id: str, output_dir: str, verbose: bool = False):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Load the model
        print(f"Loading Nougat model from {model_id}...")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.processor = NougatProcessor.from_pretrained(model_id)

    def extract_vision_encoder(self) -> Dict[str, torch.Tensor]:
        """Extract Swin Transformer vision encoder weights"""
        print("Extracting vision encoder (Swin Transformer)...")

        vision_dict = {}
        encoder = self.model.encoder

        # Get all encoder parameters
        for name, param in encoder.named_parameters():
            # Map to our Swin naming convention
            mapped_name = self._map_swin_tensor_name(name)
            vision_dict[mapped_name] = param.detach().cpu()

            if self.verbose:
                print(f"  {name} -> {mapped_name} {list(param.shape)}")

        print(f"  Extracted {len(vision_dict)} vision encoder tensors")
        return vision_dict

    def extract_text_decoder(self) -> Dict[str, torch.Tensor]:
        """Extract mBART text decoder weights"""
        print("Extracting text decoder (mBART)...")

        decoder_dict = {}
        decoder = self.model.decoder

        # Get all decoder parameters
        for name, param in decoder.named_parameters():
            # Map to our mBART naming convention
            mapped_name = self._map_mbart_tensor_name(name)
            decoder_dict[mapped_name] = param.detach().cpu()

            if self.verbose:
                print(f"  {name} -> {mapped_name} {list(param.shape)}")

        print(f"  Extracted {len(decoder_dict)} text decoder tensors")
        return decoder_dict

    def extract_projector(self) -> Dict[str, torch.Tensor]:
        """Extract multimodal projector that connects vision and text models"""
        print("Extracting multimodal projector...")

        projector_dict = {}

        # In Nougat, the projection happens through the decoder's cross-attention
        # We need to extract the projection matrices that connect encoder outputs to decoder

        # Look for cross-attention weights in decoder
        for name, param in self.model.decoder.named_parameters():
            if "encoder_attn" in name:
                # These are the cross-attention weights that project from vision to text
                mapped_name = self._map_projector_tensor_name(name)
                projector_dict[mapped_name] = param.detach().cpu()

                if self.verbose:
                    print(f"  {name} -> {mapped_name} {list(param.shape)}")

        # If there's a specific projection layer between encoder and decoder
        if hasattr(self.model, "enc_to_dec_proj"):
            projector_dict["mm.projector.weight"] = self.model.enc_to_dec_proj.weight.detach().cpu()
            if hasattr(self.model.enc_to_dec_proj, "bias"):
                projector_dict["mm.projector.bias"] = self.model.enc_to_dec_proj.bias.detach().cpu()

        print(f"  Extracted {len(projector_dict)} projector tensors")
        return projector_dict

    def _map_swin_tensor_name(self, name: str) -> str:
        """Map HuggingFace Swin tensor names to our convention"""

        # Remove model prefix
        if name.startswith("model.encoder."):
            name = name[len("model.encoder."):]
        elif name.startswith("encoder."):
            name = name[len("encoder."):]

        # Patch embeddings
        if "embeddings.patch_embeddings" in name:
            if "projection.weight" in name:
                return "swin.patch_embed.weight"
            elif "projection.bias" in name:
                return "swin.patch_embed.bias"
            elif "norm" in name:
                return f"swin.patch_embed.norm.{'weight' if 'weight' in name else 'bias'}"

        # Position embeddings
        if "position_embeddings" in name:
            return "swin.pos_embed"

        # Parse layer structure
        if "layers." in name:
            parts = name.split(".")
            stage_idx = None
            layer_idx = None

            # Find stage and layer indices
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    stage_idx = int(parts[i + 1])
                if part == "blocks" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])

            if stage_idx is not None:
                # Layer-specific components
                if layer_idx is not None:
                    # Attention
                    if "attn.qkv" in name:
                        suffix = "weight" if "weight" in name else "bias"
                        return f"swin.stage.{stage_idx}.layer.{layer_idx}.attn.qkv.{suffix}"
                    elif "attn.proj" in name:
                        suffix = "weight" if "weight" in name else "bias"
                        return f"swin.stage.{stage_idx}.layer.{layer_idx}.attn.proj.{suffix}"
                    # Norms
                    elif "norm1" in name:
                        suffix = "weight" if "weight" in name else "bias"
                        return f"swin.stage.{stage_idx}.layer.{layer_idx}.norm1.{suffix}"
                    elif "norm2" in name:
                        suffix = "weight" if "weight" in name else "bias"
                        return f"swin.stage.{stage_idx}.layer.{layer_idx}.norm2.{suffix}"
                    # MLP/FFN
                    elif "mlp.fc1" in name:
                        suffix = "weight" if "weight" in name else "bias"
                        return f"swin.stage.{stage_idx}.layer.{layer_idx}.mlp.fc1.{suffix}"
                    elif "mlp.fc2" in name:
                        suffix = "weight" if "weight" in name else "bias"
                        return f"swin.stage.{stage_idx}.layer.{layer_idx}.mlp.fc2.{suffix}"
                    # Relative position bias
                    elif "relative_position_bias_table" in name:
                        return f"swin.stage.{stage_idx}.layer.{layer_idx}.attn.relative_position_bias_table"

                # Downsample layers between stages
                elif "downsample" in name:
                    if "norm" in name:
                        suffix = "weight" if "weight" in name else "bias"
                        return f"swin.stage.{stage_idx}.downsample.norm.{suffix}"
                    elif "reduction" in name:
                        return f"swin.stage.{stage_idx}.downsample.reduction.weight"

        # Output normalization
        if "layernorm" in name or "layer_norm" in name:
            if "final" in name or "output" in name:
                suffix = "weight" if "weight" in name else "bias"
                return f"swin.norm.{suffix}"

        # Default mapping
        return f"swin.{name}"

    def _map_mbart_tensor_name(self, name: str) -> str:
        """Map HuggingFace mBART tensor names to our convention"""

        # Remove model prefix
        if name.startswith("model.decoder."):
            name = name[len("model.decoder."):]
        elif name.startswith("decoder."):
            name = name[len("decoder."):]

        # Embeddings
        if name == "embed_tokens.weight" or name == "shared.weight":
            return "token_embd.weight"
        elif "embed_positions" in name:
            return "position_embd.weight"

        # Parse decoder layers
        if "layers." in name:
            parts = name.split(".")
            layer_idx = int(parts[1])

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
                suffix = "weight" if "weight" in name else "bias"
                return f"blk.{layer_idx}.attn_norm.{suffix}"

            # Cross-attention (encoder-decoder attention)
            elif "encoder_attn.q_proj" in name:
                return f"blk.{layer_idx}.attn_q_cross.weight"
            elif "encoder_attn.k_proj" in name:
                return f"blk.{layer_idx}.attn_k_cross.weight"
            elif "encoder_attn.v_proj" in name:
                return f"blk.{layer_idx}.attn_v_cross.weight"
            elif "encoder_attn.out_proj" in name:
                return f"blk.{layer_idx}.attn_o_cross.weight"
            elif "encoder_attn_layer_norm" in name:
                suffix = "weight" if "weight" in name else "bias"
                return f"blk.{layer_idx}.attn_norm_cross.{suffix}"

            # FFN
            elif "fc1" in name:
                return f"blk.{layer_idx}.ffn_up.weight"
            elif "fc2" in name:
                return f"blk.{layer_idx}.ffn_down.weight"
            elif "final_layer_norm" in name:
                suffix = "weight" if "weight" in name else "bias"
                return f"blk.{layer_idx}.ffn_norm.{suffix}"

        # Output layers
        elif "layernorm" in name or "layer_norm" in name:
            suffix = "weight" if "weight" in name else "bias"
            return f"output_norm.{suffix}"
        elif "lm_head" in name or "output_projection" in name:
            return "output.weight"

        # Default mapping
        return name

    def _map_projector_tensor_name(self, name: str) -> str:
        """Map cross-attention tensors to projector names"""

        # Extract layer index from name
        if "layers." in name:
            parts = name.split(".")
            layer_idx = int(parts[1])

            if "encoder_attn.q_proj" in name:
                return f"mm.layer.{layer_idx}.q_proj.weight"
            elif "encoder_attn.k_proj" in name:
                return f"mm.layer.{layer_idx}.k_proj.weight"
            elif "encoder_attn.v_proj" in name:
                return f"mm.layer.{layer_idx}.v_proj.weight"
            elif "encoder_attn.out_proj" in name:
                return f"mm.layer.{layer_idx}.out_proj.weight"

        return f"mm.{name}"

    def save_component(self, tensors: Dict[str, torch.Tensor], filename: str, arch_name: str, description: str):
        """Save component tensors to GGUF file"""

        output_path = self.output_dir / filename
        print(f"Saving {arch_name} to {output_path}...")

        writer = gguf.GGUFWriter(str(output_path), arch_name)
        writer.add_string("general.name", arch_name)
        writer.add_string("general.description", description)
        writer.add_string("general.architecture", arch_name.lower())

        # Add tensors
        for name, tensor in tensors.items():
            data = tensor.float().cpu().numpy()
            writer.add_tensor(name, data)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        print(f"  Saved {len(tensors)} tensors")

    def perform_surgery(self):
        """Main surgery operation - split model into components"""

        print("\n" + "=" * 60)
        print("Starting Nougat Model Surgery")
        print("=" * 60)

        # Extract components
        vision_tensors = self.extract_vision_encoder()
        text_tensors = self.extract_text_decoder()
        projector_tensors = self.extract_projector()

        # Save components
        print("\nSaving components...")

        self.save_component(
            vision_tensors,
            "nougat-vision-swin.gguf",
            "Nougat-Vision-Swin",
            "Swin Transformer vision encoder from Nougat OCR model"
        )

        self.save_component(
            text_tensors,
            "nougat-text-mbart.gguf",
            "Nougat-Text-mBART",
            "mBART text decoder from Nougat OCR model"
        )

        if projector_tensors:
            self.save_component(
                projector_tensors,
                "nougat-projector.gguf",
                "Nougat-Projector",
                "Multimodal projector connecting vision and text models"
            )

        # Save configuration
        self.save_config()

        print("\n" + "=" * 60)
        print("Surgery Complete!")
        print(f"Output files saved to: {self.output_dir}")
        print("=" * 60)

    def save_config(self):
        """Save model configuration for reconstruction"""

        config = {
            "model_id": self.model_id,
            "vision_config": {
                "architecture": "swin",
                "image_size": 384,
                "patch_size": 4,
                "window_size": 7,
                "num_channels": 3,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
            },
            "text_config": {
                "architecture": "mbart",
                "vocab_size": self.processor.tokenizer.vocab_size,
                "max_position_embeddings": 1024,
                "hidden_size": self.model.config.decoder.hidden_size,
                "num_layers": self.model.config.decoder.num_hidden_layers,
                "num_attention_heads": self.model.config.decoder.num_attention_heads,
            },
            "components": {
                "vision": "nougat-vision-swin.gguf",
                "text": "nougat-text-mbart.gguf",
                "projector": "nougat-projector.gguf" if self.extract_projector() else None,
            }
        }

        config_path = self.output_dir / "nougat-config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nConfiguration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Nougat Model Surgery - Split model into components")
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/nougat-base",
        help="HuggingFace model ID or path to local model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/nougat-surgery",
        help="Output directory for split components"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output showing tensor mappings"
    )

    args = parser.parse_args()

    surgeon = NougatModelSurgeon(args.model_id, args.output_dir, args.verbose)
    surgeon.perform_surgery()


if __name__ == "__main__":
    main()