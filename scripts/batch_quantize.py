#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch quantize GGUF models in a directory.")
    parser.add_argument("input_dir", type=str, help="Directory containing unquantized .gguf files")
    parser.add_argument("output_dir", type=str, help="Directory to save quantized .gguf files")
    parser.add_argument("--quantize-bin", type=str, default="./llama-quantize", help="Path to the quantize binary")
    parser.add_argument("--method", type=str, default="Q4_K_M", help="Quantization method (e.g., Q4_K_M, Q5_K_M)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        sys.exit(1)

    gguf_files = list(input_path.glob("*.gguf"))
    if not gguf_files:
        print(f"No .gguf files found in {input_path}")
        return

    for gguf_file in gguf_files:
        # Skip if it already looks quantized
        if any(q in gguf_file.name for q in ["_Q", "-Q", "Q4", "Q5", "Q8"]):
            print(f"Skipping {gguf_file.name} (already appears quantized)")
            continue
            
        output_file = output_path / f"{gguf_file.stem}-{args.method}.gguf"
        print(f"\nQuantizing {gguf_file.name} to {args.method}...")
        
        cmd = [args.quantize_bin, str(gguf_file), str(output_file), args.method]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully quantized: {output_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to quantize {gguf_file.name}. Error: {e}")

if __name__ == "__main__":
    main()
