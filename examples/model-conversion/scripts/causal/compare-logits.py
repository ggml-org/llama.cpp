#!/usr/bin/env python3

import numpy as np
import sys
import os
from pathlib import Path

def quick_logits_check(pytorch_file, llamacpp_file):
    """Lightweight sanity check before NMSE"""

    try:
        pytorch_logits = np.fromfile(pytorch_file, dtype=np.float32)
        llamacpp_logits = np.fromfile(llamacpp_file, dtype=np.float32)
    except Exception as e:
        print(f"❌ NOK: Failed to load files - {e}")
        return False

    # Check shapes match
    if pytorch_logits.shape != llamacpp_logits.shape:
        print(f"❌ NOK: Shape mismatch - PyTorch: {pytorch_logits.shape}, llama.cpp: {llamacpp_logits.shape}")
        return False

    # Calculate key metrics
    diff = pytorch_logits - llamacpp_logits
    abs_diff = np.abs(diff)
    max_diff = np.max(abs_diff)

    # Get top 10 predictions from both models
    pytorch_top10 = np.argsort(pytorch_logits)[-10:][::-1]
    llamacpp_top10 = np.argsort(llamacpp_logits)[-10:][::-1]
    print(f"Top 10 PyTorch logits: {pytorch_logits[pytorch_top10]}")
    print(f"Top 10 llama.cpp logits: {llamacpp_logits[llamacpp_top10]}")
    print(f"Max absolute difference: {max_diff:.4f}")

    if max_diff > 1.0:
        print(f"❌ NOK: Large differences detected - max diff: {max_diff:.4f}")
        return False

    return True

def main():
    model_path = os.getenv('MODEL_PATH')
    if not model_path:
        print("Error: MODEL_PATH environment variable not set")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    data_dir = Path("data")

    pytorch_file = data_dir / f"pytorch-{model_name}.bin"
    llamacpp_file = data_dir / f"llamacpp-{model_name}.bin"

    if not pytorch_file.exists():
        print(f"Error: PyTorch logits file not found: {pytorch_file}")
        print("Please run scripts/run-org-model.sh first to generate this file.")
        sys.exit(1)

    if not llamacpp_file.exists():
        print(f"Error: llama.cpp logits file not found: {llamacpp_file}")
        print("Please run scripts/run-converted-model.sh first to generate this file.")
        sys.exit(1)

    print("Checked all required files were found. Proceeding...\n")


    print("🔍 GGML Model Validation for model ", model_name)
    print("=" * 40)
    print(f"PyTorch logits  : {pytorch_file}")
    print(f"llama.cpp logits: {llamacpp_file}")
    print()

    success = quick_logits_check(pytorch_file, llamacpp_file)

    # Exit with appropriate code
    if success:
        print("✅ OK: Lightweight model check successful!")
        print("       Ok to proceed with NMSE check...")
        sys.exit(0)
    else:
        print(f"❌ NOK: Top 10 predictions don't match - generation will differ")
        sys.exit(1)

if __name__ == "__main__":
    main()
