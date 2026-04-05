#!/usr/bin/env python3

import argparse
import os
import sys
import importlib
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.common import debug_hook, save_output_data

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process model with specified path")
    parser.add_argument("--model-path", "-m", help="Path to the model")
    parser.add_argument("--prompt-file", "-f", help="Optional prompt file", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--device", "-d", help="Device to use (cpu, cuda, mps, auto)", default="auto")
    parser.add_argument("--dump-tensors", help="Directory to save intermediate tensors as float32 .bin files for comparison with llama.cpp debug output", metavar="DIR")
    parser.add_argument("--dump-layer", type=int, default=0, help="Which layer to dump tensors from (default: 0)")
    return parser.parse_args()

def load_model_and_tokenizer(model_path, device="auto"):
    print("Loading model and tokenizer using AutoTokenizer:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    multimodal = False
    full_config = config

    # Determine device_map based on device argument
    if device == "cpu":
        device_map = {"": "cpu"}
        print("Forcing CPU usage")
    elif device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    print("Model type:       ", config.model_type)
    if "vocab_size" not in config and "text_config" in config:
        config = config.text_config
        multimodal = True

    def print_if_exists(label, obj, attr, default="N/A"):
        val = getattr(obj, attr) if hasattr(obj, attr) else default
        print(f"{label}", val)

    print_if_exists("Vocab size:       ", config, "vocab_size")
    print_if_exists("Hidden size:      ", config, "hidden_size")
    print_if_exists("Number of layers: ", config, "num_hidden_layers")
    print_if_exists("BOS token id:     ", config, "bos_token_id")
    print_if_exists("EOS token id:     ", config, "eos_token_id")

    unreleased_model_name = os.getenv("UNRELEASED_MODEL_NAME")
    if unreleased_model_name:
        model_name_lower = unreleased_model_name.lower()
        unreleased_module_path = (
            f"transformers.models.{model_name_lower}.modular_{model_name_lower}"
        )
        class_name = f"{unreleased_model_name}ForCausalLM"
        print(f"Importing unreleased model module: {unreleased_module_path}")

        try:
            model_class = getattr(importlib.import_module(unreleased_module_path), class_name)
            model = model_class.from_pretrained(
                    model_path,
                    device_map=device_map,
                    offload_folder="offload",
                    trust_remote_code=True,
                    config=config
            )
        except (ImportError, AttributeError) as e:
            print(f"Failed to import or load model: {e}")
            exit(1)
    else:
        if multimodal:
            model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    device_map=device_map,
                    offload_folder="offload",
                    trust_remote_code=True,
                    config=full_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    offload_folder="offload",
                    trust_remote_code=True,
                    config=config
            )

    print(f"Model class: {model.__class__.__name__}")

    return model, tokenizer, config

def enable_torch_debugging(model):
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # only leaf modules
                module.register_forward_hook(debug_hook(name))


def save_tensor(t, name, dump_dir):
    """Save a tensor as flat float32 binary + shape sidecar, matching llama.cpp layout.

    llama.cpp stores tensors with ne[0] (innermost) = last PyTorch dim,
    so for a [batch, seq, hidden] tensor the flat layout is identical to
    PyTorch's contiguous layout when batch=1: seq rows of hidden floats.
    """
    import pathlib
    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)

    arr = t.detach().float().cpu()
    # Drop batch dim if present (batch=1 always for inference)
    if arr.ndim == 4:
        arr = arr[0]           # [seq, heads, dim] or similar
    elif arr.ndim == 3:
        arr = arr[0]           # [seq, hidden]

    flat = arr.contiguous().numpy().flatten().astype("float32")
    shape = list(arr.shape)

    flat.tofile(f"{dump_dir}/{name}.bin")
    with open(f"{dump_dir}/{name}.shape", "w") as f:
        f.write(" ".join(str(d) for d in shape) + "\n")
    print(f"  Saved tensor '{name}' shape={shape} ({flat.size} elements)")


def save_tensor_attn_qk(t, name, dump_dir):
    """Save Q or K tensor from PyTorch [batch, seq, n_heads, head_dim]
    matching GGML layout ne=[head_dim, n_heads, seq] (both give same flat order)."""
    import pathlib
    pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)

    arr = t.detach().float().cpu()
    if arr.ndim == 4:
        arr = arr[0]  # drop batch → [seq, n_heads, head_dim]
    # ndim==3: already [seq, n_heads, head_dim]

    flat = arr.contiguous().numpy().flatten().astype("float32")
    shape = list(arr.shape)

    flat.tofile(f"{dump_dir}/{name}.bin")
    with open(f"{dump_dir}/{name}.shape", "w") as f:
        f.write(" ".join(str(d) for d in shape) + "\n")
    print(f"  Saved tensor '{name}' shape={shape} ({flat.size} elements)")


def register_dump_hooks(model, config, dump_dir, target_layer=0):
    """Register forward hooks that save key intermediate tensors for a given layer.

    Tensor names mirror the llama.cpp cb() names with the layer suffix -N.
    """
    il = target_layer
    hooks = []

    # Find the text model regardless of multimodal wrapper.
    text_model = getattr(model, "model", model)
    if not hasattr(text_model, "embed_tokens"):
        text_model = getattr(text_model, "language_model", text_model)
        text_model = getattr(text_model, "model", text_model)

    layer0  = text_model.layers[0]
    layerN  = text_model.layers[il]

    # ---- inp_scaled: input to layer0.input_layernorm = inpL in llama.cpp ----
    def inp_scaled_hook(_m, inp):
        t = inp[0] if isinstance(inp, tuple) else inp
        save_tensor(t, "inp_scaled", dump_dir)
    hooks.append(layer0.input_layernorm.register_forward_pre_hook(inp_scaled_hook))

    def make_save_hook(tensor_name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            save_tensor(t, tensor_name, dump_dir)
        return hook

    # ---- target layer tensors ----
    hooks.append(layerN.input_layernorm.register_forward_hook(
        make_save_hook(f"attn_norm-{il}")))

    if hasattr(layerN, "self_attn"):
        if hasattr(layerN.self_attn, "q_norm"):
            def q_norm_hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                save_tensor_attn_qk(t, f"Qcur_normed-{il}", dump_dir)
            hooks.append(layerN.self_attn.q_norm.register_forward_hook(q_norm_hook))

        if hasattr(layerN.self_attn, "k_norm"):
            def k_norm_hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                save_tensor_attn_qk(t, f"Kcur_normed-{il}", dump_dir)
            hooks.append(layerN.self_attn.k_norm.register_forward_hook(k_norm_hook))

        def attn_raw_out_hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            save_tensor(t, f"attn_raw_out-{il}", dump_dir)
        hooks.append(layerN.self_attn.register_forward_hook(attn_raw_out_hook))

    if hasattr(layerN, "post_attention_layernorm"):
        hooks.append(layerN.post_attention_layernorm.register_forward_hook(
            make_save_hook(f"attn_post_norm-{il}")))

    if hasattr(layerN, "pre_feedforward_layernorm"):
        hooks.append(layerN.pre_feedforward_layernorm.register_forward_hook(
            make_save_hook(f"ffn_norm-{il}")))
        def attn_out_hook(_m, inp):
            t = inp[0] if isinstance(inp, tuple) else inp
            save_tensor(t, f"attn_out-{il}", dump_dir)
        hooks.append(layerN.pre_feedforward_layernorm.register_forward_pre_hook(attn_out_hook))
    elif hasattr(layerN, "mlp"):
        def attn_out_hook_dense(_m, inp):
            t = inp[0] if isinstance(inp, tuple) else inp
            save_tensor(t, f"attn_out-{il}", dump_dir)
        hooks.append(layerN.mlp.register_forward_pre_hook(attn_out_hook_dense))

    if hasattr(layerN, "post_feedforward_layernorm"):
        hooks.append(layerN.post_feedforward_layernorm.register_forward_hook(
            make_save_hook(f"ffn_post_norm-{il}")))

    def layerN_post_hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        save_tensor(t, f"l_out-{il}", dump_dir)
    hooks.append(layerN.register_forward_hook(layerN_post_hook))

    # ---- final norm (last token only) ----
    if hasattr(text_model, "norm"):
        def result_norm_hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.ndim == 3:
                t = t[:, -1:, :]
            save_tensor(t, "result_norm", dump_dir)
        hooks.append(text_model.norm.register_forward_hook(result_norm_hook))

    print(f"Registered {len(hooks)} tensor-dump hooks (layer {il}) → {dump_dir}/")
    return hooks

def get_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, encoding='utf-8') as f:
            return f.read()
    elif os.getenv("MODEL_TESTING_PROMPT"):
        return os.getenv("MODEL_TESTING_PROMPT")
    else:
        return "Hello, my name is"

def main():
    args = parse_arguments()
    model_path = os.environ.get("MODEL_PATH", args.model_path)
    if model_path is None:
        print("Error: Model path must be specified either via --model-path argument or MODEL_PATH environment variable")
        sys.exit(1)


    model, tokenizer, config = load_model_and_tokenizer(model_path, args.device)

    if args.verbose:
        enable_torch_debugging(model)

    dump_hooks = []
    if args.dump_tensors:
        dump_hooks = register_dump_hooks(model, config, args.dump_tensors, args.dump_layer)

    model_name = os.path.basename(model_path)

    # Iterate over the model parameters (the tensors) and get the first one
    # and use it to get the device the model is on.
    device = next(model.parameters()).device
    prompt = get_prompt(args)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    token_ids = input_ids[0].cpu().tolist()

    print(f"Input tokens: {input_ids}")
    print(f"Input text: {repr(prompt)}")
    print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    batch_size = 512

    with torch.no_grad():
        past = None
        outputs = None
        for i in range(0, input_ids.size(1), batch_size):
            print(f"Processing chunk with tokens {i} to {i + batch_size}")
            chunk = input_ids[:, i:i + batch_size]
            outputs = model(chunk.to(model.device), past_key_values=past, use_cache=True)
            past = outputs.past_key_values

        logits = outputs.logits # type: ignore

        # Extract logits for the last token (next token prediction)
        last_logits = logits[0, -1, :].float().cpu().numpy()

        print(f"Logits shape: {logits.shape}")
        print(f"Last token logits shape: {last_logits.shape}")
        print(f"Vocab size: {len(last_logits)}")

        # Print some sample logits for quick verification
        print(f"First 10 logits: {last_logits[:10]}")
        print(f"Last 10 logits: {last_logits[-10:]}")

        # Show top 5 predicted tokens
        top_indices = np.argsort(last_logits)[-5:][::-1]
        print("Top 5 predictions:")
        for idx in top_indices:
            token = tokenizer.decode([idx])
            print(f"  Token {idx} ({repr(token)}): {last_logits[idx]:.6f}")

        save_output_data(last_logits, token_ids, prompt, model_name)

if __name__ == "__main__":
    main()
