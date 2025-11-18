#!/usr/bin/env python3
"""
Model Conversion Helper
=======================

Automated helper script for converting HuggingFace models to GGUF format
and quantizing them to various formats.

Requirements:
    - llama.cpp repository cloned
    - pip install huggingface_hub

Usage:
    python convert_model_helper.py --model meta-llama/Llama-2-7b-hf --output ./models
"""

import argparse
import subprocess
import os
import json
import shutil
from pathlib import Path
from typing import List, Optional


class ModelConverter:
    """Automated model conversion and quantization"""

    def __init__(self, llamacpp_dir: str = "./llama.cpp"):
        """
        Initialize converter

        Args:
            llamacpp_dir: Path to llama.cpp repository
        """
        self.llamacpp_dir = Path(llamacpp_dir)

        # Check if llama.cpp exists
        if not self.llamacpp_dir.exists():
            raise FileNotFoundError(
                f"llama.cpp directory not found: {llamacpp_dir}\n"
                f"Clone it with: git clone https://github.com/ggerganov/llama.cpp"
            )

        self.convert_script = self.llamacpp_dir / "convert_hf_to_gguf.py"
        self.quantize_bin = self.llamacpp_dir / "quantize"

        # Check if conversion script exists
        if not self.convert_script.exists():
            raise FileNotFoundError(f"Conversion script not found: {self.convert_script}")

    def download_model(self, model_id: str, output_dir: str):
        """
        Download model from HuggingFace

        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            output_dir: Directory to save the model

        Returns:
            Path to downloaded model directory
        """
        print(f"\n{'='*80}")
        print(f"DOWNLOADING MODEL: {model_id}")
        print('='*80)

        output_path = Path(output_dir) / model_id.split('/')[-1]
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "huggingface-cli",
            "download",
            model_id,
            "--local-dir",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Downloaded to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"✗ Download failed: {e}")
            raise

    def get_model_info(self, model_dir: Path) -> dict:
        """
        Extract model information from config.json

        Args:
            model_dir: Path to model directory

        Returns:
            Dictionary with model info
        """
        config_path = model_dir / "config.json"

        if not config_path.exists():
            return {"error": "config.json not found"}

        with open(config_path) as f:
            config = json.load(f)

        # Extract key info
        info = {
            "architecture": config.get("architectures", ["Unknown"])[0],
            "hidden_size": config.get("hidden_size", 0),
            "num_layers": config.get("num_hidden_layers", 0),
            "vocab_size": config.get("vocab_size", 0),
        }

        # Estimate parameters (rough)
        if info["hidden_size"] > 0 and info["num_layers"] > 0:
            # Rough estimate: embeddings + layers × (4*h² + 3*h*ffn) + output
            h = info["hidden_size"]
            l = info["num_layers"]
            v = info["vocab_size"]
            ffn = config.get("intermediate_size", h * 4)

            params = v * h + l * (4 * h * h + 3 * h * ffn) + v * h
            info["parameters_approx"] = params
            info["parameters_b"] = f"{params / 1e9:.1f}B"

        return info

    def convert_to_gguf(self, model_dir: Path, output_file: Optional[Path] = None,
                       outtype: str = "f16", vocab_only: bool = False) -> Path:
        """
        Convert HuggingFace model to GGUF

        Args:
            model_dir: Path to HuggingFace model directory
            output_file: Output GGUF file path (default: auto-generated)
            outtype: Output type (f32, f16, q8_0)
            vocab_only: Convert only vocabulary

        Returns:
            Path to output GGUF file
        """
        print(f"\n{'='*80}")
        print(f"CONVERTING TO GGUF ({outtype})")
        print('='*80)

        if output_file is None:
            output_file = model_dir.parent / f"{model_dir.name}-{outtype}.gguf"

        cmd = [
            "python",
            str(self.convert_script),
            str(model_dir),
            "--outfile",
            str(output_file),
            "--outtype",
            outtype
        ]

        if vocab_only:
            cmd.append("--vocab-only")

        try:
            subprocess.run(cmd, check=True, cwd=self.llamacpp_dir)
            print(f"✓ Converted to {output_file}")
            print(f"  Size: {output_file.stat().st_size / 1024**3:.2f} GB")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"✗ Conversion failed: {e}")
            raise

    def quantize_model(self, input_file: Path, quant_type: str) -> Path:
        """
        Quantize GGUF model

        Args:
            input_file: Input F16/F32 GGUF file
            quant_type: Quantization type (q4_k_m, q5_k_m, etc.)

        Returns:
            Path to quantized model
        """
        print(f"\n{'='*80}")
        print(f"QUANTIZING TO {quant_type.upper()}")
        print('='*80)

        # Build quantize tool if not exists
        if not self.quantize_bin.exists():
            print("Building quantize tool...")
            subprocess.run(["make", "quantize"], cwd=self.llamacpp_dir, check=True)

        output_file = input_file.with_name(
            input_file.stem.replace("-f16", f"-{quant_type}") + ".gguf"
        )

        cmd = [
            str(self.quantize_bin),
            str(input_file),
            str(output_file),
            quant_type
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Quantized to {output_file}")
            print(f"  Size: {output_file.stat().st_size / 1024**3:.2f} GB")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"✗ Quantization failed: {e}")
            raise

    def full_conversion_pipeline(self, model_id: str, output_dir: str,
                                 quant_types: Optional[List[str]] = None,
                                 download: bool = True,
                                 keep_f16: bool = True) -> dict:
        """
        Complete conversion pipeline

        Args:
            model_id: HuggingFace model ID or local path
            output_dir: Output directory
            quant_types: List of quantization types (default: q4_k_m, q5_k_m, q8_0)
            download: Whether to download from HuggingFace
            keep_f16: Keep F16 model after quantization

        Returns:
            Dictionary with output file paths
        """
        if quant_types is None:
            quant_types = ["q4_k_m", "q5_k_m", "q8_0"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "model_id": model_id,
            "files": {}
        }

        # Download or use local model
        if download:
            model_dir = self.download_model(model_id, str(output_dir))
        else:
            model_dir = Path(model_id)

        # Get model info
        info = self.get_model_info(model_dir)
        results["info"] = info

        print(f"\n{'='*80}")
        print("MODEL INFORMATION")
        print('='*80)
        print(f"Architecture: {info.get('architecture', 'Unknown')}")
        print(f"Parameters: {info.get('parameters_b', 'Unknown')}")
        print(f"Layers: {info.get('num_layers', 'Unknown')}")
        print(f"Hidden size: {info.get('hidden_size', 'Unknown')}")

        # Convert to F16
        f16_file = self.convert_to_gguf(model_dir, outtype="f16")
        results["files"]["f16"] = str(f16_file)

        # Quantize
        for quant_type in quant_types:
            try:
                quant_file = self.quantize_model(f16_file, quant_type)
                results["files"][quant_type] = str(quant_file)
            except Exception as e:
                print(f"Warning: Failed to quantize to {quant_type}: {e}")

        # Remove F16 if requested
        if not keep_f16 and len(results["files"]) > 1:
            print(f"\nRemoving F16 file: {f16_file}")
            f16_file.unlink()
            del results["files"]["f16"]

        # Save manifest
        manifest_file = output_dir / f"{model_dir.name}_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print("CONVERSION COMPLETE")
        print('='*80)
        print(f"Manifest saved to: {manifest_file}")
        print("\nGenerated files:")
        for name, path in results["files"].items():
            size_gb = Path(path).stat().st_size / 1024**3
            print(f"  {name.upper():10} - {size_gb:7.2f} GB - {path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Automated model conversion helper"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output",
        default="./converted_models",
        help="Output directory"
    )
    parser.add_argument(
        "--llamacpp-dir",
        default="./llama.cpp",
        help="Path to llama.cpp repository"
    )
    parser.add_argument(
        "--quant-types",
        nargs='+',
        default=["q4_k_m", "q5_k_m", "q8_0"],
        help="Quantization types to generate"
    )
    parser.add_argument(
        "--no-download",
        action='store_true',
        help="Don't download, use local model"
    )
    parser.add_argument(
        "--keep-f16",
        action='store_true',
        help="Keep F16 model after quantization"
    )

    args = parser.parse_args()

    try:
        converter = ModelConverter(args.llamacpp_dir)

        converter.full_conversion_pipeline(
            model_id=args.model,
            output_dir=args.output,
            quant_types=args.quant_types,
            download=not args.no_download,
            keep_f16=args.keep_f16
        )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


# Example usage:
"""
# Download and convert Llama-2-7B
python convert_model_helper.py \\
    --model meta-llama/Llama-2-7b-hf \\
    --output ./models \\
    --llamacpp-dir ../llama.cpp

# Convert local model with specific quantizations
python convert_model_helper.py \\
    --model ./my-model \\
    --output ./output \\
    --quant-types q4_k_s q4_k_m q5_k_m q6_k \\
    --no-download \\
    --keep-f16

# Minimal conversion (Q4_K_M only)
python convert_model_helper.py \\
    --model mistralai/Mistral-7B-v0.1 \\
    --quant-types q4_k_m
"""
