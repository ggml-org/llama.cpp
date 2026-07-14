#!/usr/bin/env python3

import os
import sys
import torch
import transformers
import json
import textwrap
import numpy as np
from pathlib import Path


def get_model_name_from_env_path(env_path_name):
    model_path = os.getenv(env_path_name)
    if not model_path:
        print(f"Error: {env_path_name} environment variable not set")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    name = os.path.basename(os.path.normpath(model_path))
    if name.endswith(".gguf"):
        name = name[:-5]

    return name


def summarize(tensor: torch.Tensor, name: str, max_seq: int = 3, max_vals: int = 3):
    """
    Print a tensor in llama.cpp debug style.

    Supports:
    - 2D tensors (seq, hidden)
    - 3D tensors (batch, seq, hidden)
    - 4D tensors (batch, seq, heads, dim_per_head) via flattening heads × dim_per_head

    Shows first and last max_vals of each vector per sequence position.
    """
    t = tensor.detach().to(torch.float32).cpu()

    # Determine dimensions
    if t.ndim == 3:
        _, s, _ = t.shape
    elif t.ndim == 2:
        _, s = 1, t.shape[0]
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        _, s, _, _ = t.shape
    else:
        print(f"Skipping tensor due to unsupported dimensions: {t.ndim}")
        return

    ten_shape = t.shape

    print(f"ggml_debug: {name} = (f32)  ... = {{{ten_shape}}}")
    print("                                     [")
    print("                                      [")

    # Determine indices for first and last sequences
    first_indices = list(range(min(s, max_seq)))
    last_indices = list(range(max(0, s - max_seq), s))

    # Check if there's an overlap between first and last indices or if we're at the edge case of s = 2 * max_seq
    has_overlap = bool(set(first_indices) & set(last_indices)) or (max_seq * 2 == s)

    # Combine indices
    if has_overlap:
        # If there's overlap, just use the combined unique indices
        indices = sorted(list(set(first_indices + last_indices)))
        separator_index = None
    else:
        # If no overlap, we'll add a separator between first and last sequences
        indices = first_indices + last_indices
        separator_index = len(first_indices)

    for i, si in enumerate(indices):
        # Add separator if needed
        if separator_index is not None and i == separator_index:
            print("                                       ...")

        # Extract appropriate slice
        vec = t[0, si]
        if vec.ndim == 2:  # 4D case: flatten heads × dim_per_head
            flat = vec.flatten().tolist()
        else:  # 2D or 3D case
            flat = vec.tolist()

        # First and last slices
        first = flat[:max_vals]
        last = flat[-max_vals:] if len(flat) >= max_vals else flat
        first_str = ", ".join(f"{v:12.4f}" for v in first)
        last_str = ", ".join(f"{v:12.4f}" for v in last)

        print(f"                                       [{first_str}, ..., {last_str}]")

    print("                                      ],")
    print("                                     ]")
    print(f"                                     sum = {t.sum().item():.6f}\n")


def debug_hook(name):
    def fn(_m, input, output):
        if isinstance(input, torch.Tensor):
            summarize(input, name + "_in")
        elif isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            summarize(input[0], name + "_in")
        if isinstance(output, torch.Tensor):
            summarize(output, name + "_out")
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            summarize(output[0], name + "_out")

    return fn


def setup_rope_debug(model_module_path: str, function_name: str = "apply_rotary_pos_emb"):
    """
    Apply monkey patch to dump RoPE activations for debugging.

    Args:
        model_module_path: Path to the model module (e.g., "transformers.models.apertus.modeling_apertus")
        function_name: Name of the RoPE function to patch (default: "apply_rotary_pos_emb")

    Example:
        from utils.common import setup_rope_debug
        setup_rope_debug("transformers.models.apertus.modeling_apertus")
    """
    import importlib

    # Import the module and get the original function
    module = importlib.import_module(model_module_path)
    orig_rope = getattr(module, function_name)

    # Set torch print options for better debugging
    torch.set_printoptions(threshold=float('inf'))
    torch.set_printoptions(precision=6, sci_mode=False)

    def debug_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # log inputs
        summarize(q, "RoPE.q_in")
        summarize(k, "RoPE.k_in")

        # call original
        q_out, k_out = orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)

        # log outputs
        summarize(q_out, "RoPE.q_out")
        summarize(k_out, "RoPE.k_out")

        return q_out, k_out

    # Patch it
    setattr(module, function_name, debug_rope)
    print(f"RoPE debug patching applied to {model_module_path}.{function_name}")


def save_output_data(data, tokens, prompt, model_name, type_suffix="", output_dir="data"):
    """
    Save output data (logits/embeddings), tokens, and prompt to files.

    Args:
        data:        numpy array of floats (logits or embeddings)
        tokens:      list or array of token IDs
        prompt:      string containing the input prompt
        model_name:  name of the model
        type_suffix: optional suffix like "-embeddings" (default: "")
        output_dir:  directory to save files (default: "data")

    Creates the following files in output_dir:
        - pytorch-{model_name}{type_suffix}.bin
        - pytorch-{model_name}{type_suffix}.txt
        - pytorch-{model_name}{type_suffix}-prompt.txt
        - pytorch-{model_name}{type_suffix}-tokens.bin
    """
    data_dir = Path(output_dir)
    data_dir.mkdir(exist_ok=True)
    base_path = data_dir / f"pytorch-{model_name}{type_suffix}"

    # Convert and flatten logits/embeddings
    data = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    data = data.flatten() if data.ndim > 1 else data

    # Save logits/embedding files
    data.astype(np.float32).tofile(f"{base_path}.bin")
    print(f"Data saved to {base_path}.bin")

    with open(f"{base_path}.txt", "w") as f:
        f.writelines(f"{i}: {value:.6f}\n" for i, value in enumerate(data))
    print(f"Data saved to {base_path}.txt")

    # Convert and flatten tokens
    tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
    tokens = tokens.flatten() if tokens.ndim > 1 else tokens

    # Save token binary file
    tokens.astype(np.int32).tofile(f"{base_path}-tokens.bin")
    print(f"Tokens saved to {base_path}-tokens.bin")

    # Save prompt file
    with open(f"{base_path}-prompt.txt", "w") as f:
        f.write(f"prompt: {prompt}\n")
        f.write(f"n_tokens: {len(tokens)}\n")
        f.write(f"token ids: {', '.join(str(int(tid)) for tid in tokens)}\n")
    print(f"Prompt saved to {base_path}-prompt.txt")


def compare_tokens(original, converted, type_suffix="", output_dir="data"):
    data_dir = Path(output_dir)

    # Read tokens from both models
    tokens1_file = data_dir / f"{original}{type_suffix}-tokens.bin"
    tokens2_file = data_dir / f"{converted}{type_suffix}-tokens.bin"

    if not tokens1_file.exists():
        print(f"Error: Token file not found: {tokens1_file}")
        return False

    if not tokens2_file.exists():
        print(f"Error: Token file not found: {tokens2_file}")
        return False

    tokens1 = np.fromfile(tokens1_file, dtype=np.int32)
    tokens2 = np.fromfile(tokens2_file, dtype=np.int32)

    print(f"\nComparing tokens between:")
    print(f"  Original : {original} ({len(tokens1)} tokens)")
    print(f"  Converted: {converted} ({len(tokens2)} tokens)")

    if len(tokens1) != len(tokens2):
        print(f"\n❌ Token count mismatch: {len(tokens1)} vs {len(tokens2)}")
        return False

    if np.array_equal(tokens1, tokens2):
        print(f"\n✅ All {len(tokens1)} tokens match!")
        return True

    mismatches = np.where(tokens1 != tokens2)[0]
    print(f"\n❌ Found {len(mismatches)} mismatched tokens:")

    num_to_show = min(len(mismatches), 10)
    for idx in mismatches[:num_to_show]:
        print(f"  Position {idx}: {tokens1[idx]} vs {tokens2[idx]}")

    if len(mismatches) > num_to_show:
        print(f"  ... and {len(mismatches) - num_to_show} more mismatches")

    return False


def show_version_warning(current_version, model_version):
    if not model_version:
        return False

    try:
        from packaging.version import parse, InvalidVersion
        try:
            return parse(current_version) < parse(model_version)
        except InvalidVersion:
            return current_version != model_version
    except ImportError:
        return current_version != model_version

def get_model_transformers_version(model_path):
    if not model_path:
        return None

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("transformers_version")
    except (IOError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read or parse {config_path}: {e}", file=sys.stderr)
        return None

def exit_with_warning(message, model_path):
    print(message)

    if model_path and transformers is not None:
        model_transformers_version = get_model_transformers_version(model_path)
        transformers_version       = transformers.__version__
        if show_version_warning(transformers_version, model_transformers_version):
            warning_message = f"""
                =====================================================================
                Verification failure might be due to a transformers version mismatch:

                Current transformers version: {transformers_version}
                Model's required version    : {model_transformers_version}

                Consider installing the version specified by the model's config:
                pip install transformers=={model_transformers_version}
                =====================================================================
            """
            print(textwrap.dedent(warning_message))
    sys.exit(1)
