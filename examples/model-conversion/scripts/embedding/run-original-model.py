#!/usr/bin/env python3

import argparse
import os
import numpy as np
import importlib
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch

unreleased_model_name = os.getenv('UNRELEASED_MODEL_NAME')

parser = argparse.ArgumentParser(description='Process model with specified path')
parser.add_argument('--model-path', '-m', help='Path to the model')
parser.add_argument('--prompts-file', '-p', help='Path to file containing prompts (one per line)')
parser.add_argument('--use-sentence-transformers', action='store_true',
                    help='Use SentenceTransformer to apply all numbered layers (01_Pooling, 02_Dense, 03_Dense, 04_Normalize)')
args = parser.parse_args()

def read_prompt_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompts file '{file_path}' not found")
        exit(1)
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        exit(1)

model_path = os.environ.get('EMBEDDING_MODEL_PATH', args.model_path)
if model_path is None:
    parser.error("Model path must be specified either via --model-path argument or EMBEDDING_MODEL_PATH environment variable")

# Determine if we should use SentenceTransformer
use_sentence_transformers = args.use_sentence_transformers or os.environ.get('USE_SENTENCE_TRANSFORMERS', '').lower() in ('1', 'true', 'yes')

if use_sentence_transformers:
    from sentence_transformers import SentenceTransformer
    print("Using SentenceTransformer to apply all numbered layers")
    model = SentenceTransformer(model_path)
    tokenizer = model.tokenizer
    config = model[0].auto_model.config  # type: ignore
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)

    # This can be used to override the sliding window size for manual testing. This
    # can be useful to verify the sliding window attention mask in the original model
    # and compare it with the converted .gguf model.
    if hasattr(config, 'sliding_window'):
        original_sliding_window = config.sliding_window
        #original_sliding_window = 6
        print(f"Modified sliding window: {original_sliding_window} -> {config.sliding_window}")

    print(f"Using unreleased model: {unreleased_model_name}")
    if unreleased_model_name:
        model_name_lower = unreleased_model_name.lower()
        unreleased_module_path = f"transformers.models.{model_name_lower}.modular_{model_name_lower}"
        class_name = f"{unreleased_model_name}Model"
        print(f"Importing unreleased model module: {unreleased_module_path}")

        try:
            model_class = getattr(importlib.import_module(unreleased_module_path), class_name)
            model = model_class.from_pretrained(model_path, config=config)
        except (ImportError, AttributeError) as e:
            print(f"Failed to import or load model: {e}")
            exit(1)
    else:
        model = AutoModel.from_pretrained(model_path, config=config)
    print(f"Model class: {type(model)}")
    print(f"Model file: {type(model).__module__}")

# Verify the model is using the correct sliding window
if not use_sentence_transformers:
    if hasattr(model.config, 'sliding_window'):  # type: ignore
        print(f"Model's sliding_window: {model.config.sliding_window}")  # type: ignore
    else:
        print("Model config does not have sliding_window attribute")

model_name = os.path.basename(model_path)

if args.prompts_file:
    prompt_text = read_prompt_from_file(args.prompts_file)
    texts = [prompt_text]
else:
    texts = ["Hello world today"]

with torch.no_grad():
    if use_sentence_transformers:
        embeddings = model.encode(texts, convert_to_numpy=True)
        all_embeddings = embeddings  # Shape: [batch_size, hidden_size]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        tokens = encoded['input_ids'][0]
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        for i, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
            print(f"{token_id:6d} -> '{token_str}'")

        print(f"Embeddings shape (after all SentenceTransformer layers): {all_embeddings.shape}")
        print(f"Embedding dimension: {all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else all_embeddings.shape[0]}")  # type: ignore
    else:
        # Standard approach: use base model output only
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokens = encoded['input_ids'][0]
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        for i, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
            print(f"{token_id:6d} -> '{token_str}'")

        outputs = model(**encoded)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        all_embeddings = hidden_states[0].cpu().numpy()  # Shape: [seq_len, hidden_size]

        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"All embeddings shape: {all_embeddings.shape}")
        print(f"Embedding dimension: {all_embeddings.shape[1]}")

    if len(all_embeddings.shape) == 1:
        n_embd = all_embeddings.shape[0]  # type: ignore
        n_embd_count = 1
        all_embeddings = all_embeddings.reshape(1, -1)
    else:
        n_embd = all_embeddings.shape[1]  # type: ignore
        n_embd_count = all_embeddings.shape[0]  # type: ignore

    print()

    for j in range(n_embd_count):
        embedding = all_embeddings[j]
        print(f"embedding {j}: ", end="")

        # Print first 3 values
        for i in range(min(3, n_embd)):
            print(f"{embedding[i]:9.6f} ", end="")

        print(" ... ", end="")

        # Print last 3 values
        for i in range(n_embd - 3, n_embd):
            print(f"{embedding[i]:9.6f} ", end="")

        print()  # New line

    print()

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    bin_filename = data_dir / f"pytorch-{model_name}-embeddings.bin"
    txt_filename = data_dir / f"pytorch-{model_name}-embeddings.txt"

    flattened_embeddings = all_embeddings.flatten()
    flattened_embeddings.astype(np.float32).tofile(bin_filename)

    with open(txt_filename, "w") as f:
        idx = 0
        for j in range(n_embd_count):
            for value in all_embeddings[j]:
                f.write(f"{idx}: {value:.6f}\n")
                idx += 1
    print(f"Total values: {len(flattened_embeddings)} ({n_embd_count} embeddings × {n_embd} dimensions)")
    print("")
    print(f"Saved bin embeddings to: {bin_filename}")
    print(f"Saved txt embeddings to: {txt_filename}")
