#!/usr/bin/env python3

import os
import sys
import torch


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
