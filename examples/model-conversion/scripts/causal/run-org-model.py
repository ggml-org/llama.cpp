#!/usr/bin/env python3

import argparse
import os
import importlib
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import numpy as np

unreleased_model_name = os.getenv('UNRELEASED_MODEL_NAME')

parser = argparse.ArgumentParser(description='Process model with specified path')
parser.add_argument('--model-path', '-m', help='Path to the model')
args = parser.parse_args()

model_path = os.environ.get('MODEL_PATH', args.model_path)
if model_path is None:
    parser.error("Model path must be specified either via --model-path argument or MODEL_PATH environment variable")

config = AutoConfig.from_pretrained(model_path)

print("Model type:       ", config.model_type)
print("Vocab size:       ", config.vocab_size)
print("Hidden size:      ", config.hidden_size)
print("Number of layers: ", config.num_hidden_layers)
print("BOS token id:     ", config.bos_token_id)
print("EOS token id:     ", config.eos_token_id)

print("Loading model and tokenizer using AutoTokenizer:", model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

if unreleased_model_name:
    model_name_lower = unreleased_model_name.lower()
    unreleased_module_path = f"transformers.models.{model_name_lower}.modular_{model_name_lower}"
    class_name = f"{unreleased_model_name}ForCausalLM"
    print(f"Importing unreleased model module: {unreleased_module_path}")

    try:
        model_class = getattr(importlib.import_module(unreleased_module_path), class_name)
        model = model_class.from_pretrained(model_path)  # Note: from_pretrained, not fromPretrained
    except (ImportError, AttributeError) as e:
        print(f"Failed to import or load model: {e}")
        exit(1)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)

model_name = os.path.basename(model_path)
# Printing the Model class to allow for easier debugging. This can be useful
# when working with models that have not been publicly released yet and this
# migth require that the concrete class is imported and used directly instead
# of using AutoModelForCausalLM.
print(f"Model class: {model.__class__.__name__}")

prompt = "Hello, my name is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

print(f"Input tokens: {input_ids}")
print(f"Input text: {repr(prompt)}")
print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

    # Extract logits for the last token (next token prediction)
    last_logits = logits[0, -1, :].cpu().numpy()

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
