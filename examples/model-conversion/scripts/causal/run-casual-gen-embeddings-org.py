#!/usr/bin/env python3

import argparse
import os
import sys
import importlib
import torch
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.common import save_output_data

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

if unreleased_model_name:
    model_name_lower = unreleased_model_name.lower()
    unreleased_module_path = f"transformers.models.{model_name_lower}.modular_{model_name_lower}"
    class_name = f"{unreleased_model_name}ForCausalLM"
    print(f"Importing unreleased model module: {unreleased_module_path}")

    try:
        model_class = getattr(importlib.import_module(unreleased_module_path), class_name)
        model = model_class.from_pretrained(model_path)
    except (ImportError, AttributeError) as e:
        print(f"Failed to import or load model: {e}")
        print("Falling back to AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)
print(f"Model class: {type(model)}")
#print(f"Model file: {type(model).__module__}")

model_name = os.path.basename(model_path)
print(f"Model name: {model_name}")

prompt = "Hello world today"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # ty: ignore[call-non-callable]
token_ids = input_ids[0].cpu().tolist()
print(f"Input tokens: {input_ids}")
print(f"Input text: {repr(prompt)}")
print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")  # ty: ignore[unresolved-attribute]

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

    # Extract hidden states from the last layer
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    # Index -1 gets the last layer, shape: [batch_size, seq_len, hidden_size]
    last_hidden_states = outputs.hidden_states[-1]

    # Get embeddings for all tokens
    token_embeddings = last_hidden_states[0].float().cpu().numpy()  # Remove batch dimension

    print(f"Hidden states shape: {last_hidden_states.shape}")
    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"Hidden dimension: {token_embeddings.shape[-1]}")
    print(f"Number of tokens: {token_embeddings.shape[0]}")

    print(token_embeddings)
    save_output_data(token_embeddings, token_ids, prompt, model_name, type_suffix="-embeddings")

    # Print embeddings per token in the requested format
    print("\nToken embeddings:")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # ty: ignore[unresolved-attribute]
    for i, embedding in enumerate(token_embeddings):
        # Format: show first few values, ..., then last few values
        if len(embedding) > 10:
            # Show first 3 and last 3 values with ... in between
            first_vals = " ".join(f"{val:8.6f}" for val in embedding[:3])
            last_vals = " ".join(f"{val:8.6f}" for val in embedding[-3:])
            print(f"embedding {i}: {first_vals}  ... {last_vals}")
        else:
            # If embedding is short, show all values
            vals = " ".join(f"{val:8.6f}" for val in embedding)
            print(f"embedding {i}: {vals}")

    # Also show token info for reference
    print(f"\nToken reference:")
    for i, token in enumerate(tokens):
        print(f"  Token {i}: {repr(token)}")

