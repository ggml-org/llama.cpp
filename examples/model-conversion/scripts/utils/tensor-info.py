#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from safetensors import safe_open


MODEL_SAFETENSORS_FILE = "model.safetensors"
MODEL_SAFETENSORS_INDEX = "model.safetensors.index.json"


def get_weight_map(model_path: Path) -> Optional[dict[str, str]]:
    index_file = model_path / MODEL_SAFETENSORS_INDEX

    if index_file.exists():
        with open(index_file, 'r') as f:
            index = json.load(f)
            return index.get("weight_map", {})

    return None


def get_all_tensor_names(model_path: Path) -> list[str]:
    weight_map = get_weight_map(model_path)

    if weight_map is not None:
        return list(weight_map.keys())

    single_file = model_path / MODEL_SAFETENSORS_FILE
    if single_file.exists():
        try:
            with safe_open(single_file, framework="pt", device="cpu") as f:
                return list(f.keys())
        except Exception as e:
            print(f"Error reading {single_file}: {e}")
            sys.exit(1)

    print(f"Error: No safetensors files found in {model_path}")
    sys.exit(1)


def find_tensor_file(model_path: Path, tensor_name: str) -> Optional[str]:
    weight_map = get_weight_map(model_path)

    if weight_map is not None:
        return weight_map.get(tensor_name)

    single_file = model_path / MODEL_SAFETENSORS_FILE
    if single_file.exists():
        return single_file.name

    return None


def normalize_tensor_name(tensor_name: str) -> str:
    normalized = re.sub(r'\.\d+\.', '.#.', tensor_name)
    normalized = re.sub(r'\.\d+$', '.#', normalized)
    return normalized


def list_all_tensors(model_path: Path, unique: bool = False):
    tensor_names = get_all_tensor_names(model_path)

    if unique:
        seen = set()
        for tensor_name in sorted(tensor_names):
            normalized = normalize_tensor_name(tensor_name)
            if normalized not in seen:
                seen.add(normalized)
                print(normalized)
    else:
        for tensor_name in sorted(tensor_names):
            print(tensor_name)


def print_tensor_info(model_path: Path, tensor_name: str):
    tensor_file = find_tensor_file(model_path, tensor_name)

    if tensor_file is None:
        print(f"Error: Could not find tensor '{tensor_name}' in model index")
        print(f"Model path: {model_path}")
        sys.exit(1)

    file_path = model_path / tensor_file

    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                tensor_slice = f.get_slice(tensor_name)
                shape = tensor_slice.get_shape()
                print(f"Tensor: {tensor_name}")
                print(f"File:   {tensor_file}")
                print(f"Shape:  {shape}")
            else:
                print(f"Error: Tensor '{tensor_name}' not found in {tensor_file}")
                sys.exit(1)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Print tensor information from a safetensors model"
    )
    parser.add_argument(
        "tensor_name",
        nargs="?",  # optional (if --list is used for example)
        help="Name of the tensor to inspect"
    )
    parser.add_argument(
        "-m", "--model-path",
        type=Path,
        help="Path to the model directory (default: MODEL_PATH environment variable)"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List unique tensor patterns in the model (layer numbers replaced with #)"
    )

    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_path_str = os.environ.get("MODEL_PATH")
        if model_path_str is None:
            print("Error: --model-path not provided and MODEL_PATH environment variable not set")
            sys.exit(1)
        model_path = Path(model_path_str)

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    if not model_path.is_dir():
        print(f"Error: Model path is not a directory: {model_path}")
        sys.exit(1)

    if args.list:
        list_all_tensors(model_path, unique=True)
    else:
        if args.tensor_name is None:
            print("Error: tensor_name is required when not using --list")
            sys.exit(1)
        print_tensor_info(model_path, args.tensor_name)


if __name__ == "__main__":
    main()
