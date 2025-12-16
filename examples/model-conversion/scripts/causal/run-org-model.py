#!/usr/bin/env python3

import argparse
import os
import importlib
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import numpy as np

### If you want to dump RoPE activations, apply this monkey patch to the model
### class from Transformers that you are running (replace apertus.modeling_apertus
### with the proper package and class for your model
### === START ROPE DEBUG ===
# from transformers.models.apertus.modeling_apertus import apply_rotary_pos_emb

# orig_rope = apply_rotary_pos_emb
# torch.set_printoptions(threshold=float('inf'))
# torch.set_printoptions(precision=6, sci_mode=False)

# def debug_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     # log inputs
#     summarize(q, "RoPE.q_in")
#     summarize(k, "RoPE.k_in")

#     # call original
#     q_out, k_out = orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)

#     # log outputs
#     summarize(q_out, "RoPE.q_out")
#     summarize(k_out, "RoPE.k_out")

#     return q_out, k_out

# # Patch it
# import transformers.models.apertus.modeling_apertus as apertus_mod  # noqa: E402
# apertus_mod.apply_rotary_pos_emb = debug_rope
### == END ROPE DEBUG ===


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
        elif isinstance(input, (tuple, list)) and isinstance(input[0], torch.Tensor):
            summarize(input[0], name + "_in")
        if isinstance(output, torch.Tensor):
            summarize(output, name + "_out")
        elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            summarize(output[0], name + "_out")

    return fn


unreleased_model_name = os.getenv("UNRELEASED_MODEL_NAME")

parser = argparse.ArgumentParser(description="Process model with specified path")
parser.add_argument("--model-path", "-m", help="Path to the model")
args = parser.parse_args()

model_path = os.environ.get("MODEL_PATH", args.model_path)
if model_path is None:
    parser.error(
        "Model path must be specified either via --model-path argument or MODEL_PATH environment variable"
    )


print("Loading model and tokenizer using AutoTokenizer:", model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print("Model type:       ", config.model_type)
print("Vocab size:       ", config.vocab_size)
print("Hidden size:      ", config.hidden_size)
print("Number of layers: ", config.num_hidden_layers)
print("BOS token id:     ", config.bos_token_id)
print("EOS token id:     ", config.eos_token_id)

if unreleased_model_name:
    model_name_lower = unreleased_model_name.lower()
    unreleased_module_path = (
        f"transformers.models.{model_name_lower}.modular_{model_name_lower}"
    )
    class_name = f"{unreleased_model_name}ForCausalLM"
    print(f"Importing unreleased model module: {unreleased_module_path}")

    try:
        model_class = getattr(
            importlib.import_module(unreleased_module_path), class_name
        )
        model = model_class.from_pretrained(
            model_path
        )  # Note: from_pretrained, not fromPretrained
    except (ImportError, AttributeError) as e:
        print(f"Failed to import or load model: {e}")
        exit(1)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", offload_folder="offload", trust_remote_code=True, config=config
    )

for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # only leaf modules
        module.register_forward_hook(debug_hook(name))

model_name = os.path.basename(model_path)
# Printing the Model class to allow for easier debugging. This can be useful
# when working with models that have not been publicly released yet and this
# migth require that the concrete class is imported and used directly instead
# of using AutoModelForCausalLM.
print(f"Model class: {model.__class__.__name__}")

device = next(model.parameters()).device
if os.getenv("MODEL_TESTING_PROMPT"):
    prompt = os.getenv("MODEL_TESTING_PROMPT")
else:
    prompt = "Hello, my name is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print(f"Input tokens: {input_ids}")
print(f"Input text: {repr(prompt)}")
print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

with torch.no_grad():
    outputs = model(input_ids.to(model.device))
    logits = outputs.logits

    # Extract logits for the last token (next token prediction)
    last_logits = logits[0, -1, :].float().cpu().numpy()

    print(f"Logits shape: {logits.shape}")
    print(f"Last token logits shape: {last_logits.shape}")
    print(f"Vocab size: {len(last_logits)}")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    bin_filename = data_dir / f"pytorch-{model_name}.bin"
    txt_filename = data_dir / f"pytorch-{model_name}.txt"

    # Save to file for comparison
    last_logits.astype(np.float32).tofile(bin_filename)

    # Also save as text file for easy inspection
    with open(txt_filename, "w") as f:
        for i, logit in enumerate(last_logits):
            f.write(f"{i}: {logit:.6f}\n")

    # Print some sample logits for quick verification
    print(f"First 10 logits: {last_logits[:10]}")
    print(f"Last 10 logits: {last_logits[-10:]}")

    # Show top 5 predicted tokens
    top_indices = np.argsort(last_logits)[-5:][::-1]
    print("Top 5 predictions:")
    for idx in top_indices:
        token = tokenizer.decode([idx])
        print(f"  Token {idx} ({repr(token)}): {last_logits[idx]:.6f}")

    print(f"Saved bin logits to: {bin_filename}")
    print(f"Saved txt logist to: {txt_filename}")
