#!/usr/bin/env python3

import argparse
import os
import json
from safetensors import safe_open
from collections import defaultdict

parser = argparse.ArgumentParser(description='Process model with specified path')
parser.add_argument('--model-path', '-m', help='Path to the model')
args = parser.parse_args()

model_path = os.environ.get('MODEL_PATH', args.model_path)
if model_path is None:
    parser.error("Model path must be specified either via --model-path argument or MODEL_PATH environment variable")

# Check if there's an index file (multi-file model)
index_path = os.path.join(model_path, "model.safetensors.index.json")
single_file_path = os.path.join(model_path, "model.safetensors")

if os.path.exists(index_path):
    # Multi-file model
    print("Multi-file model detected")

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    # Get the weight map (tensor_name -> file_name)
    weight_map = index_data.get("weight_map", {})

    # Group tensors by file for efficient processing
    file_tensors = defaultdict(list)
    for tensor_name, file_name in weight_map.items():
        file_tensors[file_name].append(tensor_name)

    print("Tensors in model:")

    # Process each shard file
    for file_name, tensor_names in file_tensors.items():
        file_path = os.path.join(model_path, file_name)
        print(f"\n--- From {file_name} ---")

        with safe_open(file_path, framework="pt") as f:  # type: ignore
            for tensor_name in sorted(tensor_names):
                tensor = f.get_tensor(tensor_name)
                print(f"- {tensor_name} : shape = {tensor.shape}, dtype = {tensor.dtype}")

elif os.path.exists(single_file_path):
    # Single file model (original behavior)
    print("Single-file model detected")

    with safe_open(single_file_path, framework="pt") as f:  # type: ignore
        keys = f.keys()
        print("Tensors in model:")
        for key in sorted(keys):
            tensor = f.get_tensor(key)
            print(f"- {key} : shape = {tensor.shape}, dtype = {tensor.dtype}")

else:
    print(f"Error: Neither 'model.safetensors.index.json' nor 'model.safetensors' found in {model_path}")
    print("Available files:")
    if os.path.exists(model_path):
        for item in sorted(os.listdir(model_path)):
            print(f"  {item}")
    else:
        print(f"  Directory {model_path} does not exist")
    exit(1)
