#!/usr/bin/env python3

import numpy as np
import argparse
import os
import importlib

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel

unreleased_model_name = os.getenv('UNRELEASED_MODEL_NAME')

def cosine_similarity(a, b=None):
    a = np.asarray(a)
    if b is None:
        b = a
    else:
        b = np.asarray(b)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)

    a_norms = np.where(a_norms == 0, 1e-8, a_norms)
    b_norms = np.where(b_norms == 0, 1e-8, b_norms)

    a_normalized = a / a_norms
    b_normalized = b / b_norms

    # Compute cosine similarity
    return np.dot(a_normalized, b_normalized.T)

def load_embeddings_from_file(filename, n_tokens, n_embd):
    embeddings = np.fromfile(filename, dtype=np.float32)
    return embeddings.reshape(n_tokens, n_embd)

def test_single_prompt_similarity(python_emb, cpp_emb, tokens, prompt):
    np.set_printoptions(suppress=True, precision=6)
    print("pytorch embeddings:");
    print(python_emb)
    print("llama.cpp embeddings:");
    print(cpp_emb)
    print(f"\n=== Prompt: '{prompt}' ===")
    print(f"Tokens: {tokens}")
    print(f"Embeddings shape: Python {python_emb.shape}, llama.cpp {cpp_emb.shape}")

    n_tokens = len(tokens)

    # 1. Direct embedding comparison
    print(f"\n1. Raw Embedding Magnitude Comparison:")
    # Check if the distance of each token embedding from the origin and compare
    # if the vectors are on the same "sphere". This does not tell us about
    # direction (meaning of the token embedding), just magnitude.
    for i in range(n_tokens):
        py_mag = np.linalg.norm(python_emb[i]) # calculate standard euclidean norm for Python embeddings
        cpp_mag = np.linalg.norm(cpp_emb[i])   # calculate standard euclidean norm for llama.cpp embeddings
        ratio = py_mag / cpp_mag if cpp_mag > 0 else float('inf')
        print(f"   Token {i} ({tokens[i]}): Python={py_mag:.3f}, llama.cpp={cpp_mag:.3f}, ratio={ratio:.3f}")

    # 2. Cosine similarity between tokens within each model
    # Here we check the direction of token embeddings to see if the have the
    # same meaning (similarity). This is done by calculating cosine similarity
    # of a pair of token embeddings within each model.
    print(f"\n2. Within-Model Token Similarities:")
    print("   Python model:")
    for i in range(n_tokens):
        for j in range(i+1, n_tokens):
            sim = cosine_similarity([python_emb[i]], [python_emb[j]])[0][0]
            print(f"     {tokens[i]} ↔ {tokens[j]}: {sim:.4f}")

    print("   llama.cpp model:")
    for i in range(n_tokens):
        for j in range(i+1, n_tokens):
            sim = cosine_similarity([cpp_emb[i]], [cpp_emb[j]])[0][0]
            print(f"     {tokens[i]} ↔ {tokens[j]}: {sim:.4f}")

    # 3. Cross-model similarity (same token position)
    print(f"\n3. Cross-Model Same-Token Similarities:")
    for i in range(n_tokens):
        sim = cosine_similarity([python_emb[i]], [cpp_emb[i]])[0][0]
        print(f"   Token {i} ({tokens[i]}): {sim:.4f}")

    # 4. Similarity matrix comparison
    print(f"\n4. Similarity Matrix Differences:")
    py_sim_matrix = cosine_similarity(python_emb)
    cpp_sim_matrix = cosine_similarity(cpp_emb)
    diff_matrix = np.abs(py_sim_matrix - cpp_sim_matrix)

    print(f"   Max difference: {np.max(diff_matrix):.4f}")
    print(f"   Mean difference: {np.mean(diff_matrix):.4f}")
    print(f"   RMS difference: {np.sqrt(np.mean(diff_matrix**2)):.4f}")

    return {
        'cross_model_similarities': [cosine_similarity([python_emb[i]], [cpp_emb[i]])[0][0] for i in range(n_tokens)],
        'similarity_matrix_diff': diff_matrix,
        'max_diff': np.max(diff_matrix),
        'mean_diff': np.mean(diff_matrix),
        'rms_diff': np.sqrt(np.mean(diff_matrix**2))
    }

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

def main():
    parser = argparse.ArgumentParser(description='Test semantic similarity between Python and llama.cpp embeddings')
    parser.add_argument('--model-path', '-m', required=True, help='Path to the original Python model')
    parser.add_argument('--python-embeddings', '-pe', help='Path to pytorch embeddings "logits" binary file')
    parser.add_argument('--cpp-embeddings', '-ce', help='Path to llama.cpp embeddings "logits" binary file')
    parser.add_argument('--causal', '-c', default=False, help='if the model is causal (default: false)', action='store_true')
    parser.add_argument('--prompt', '-p', default='Hello world today', help='Test prompt')
    parser.add_argument('--prompts-file', '-pf', help='Path to file containing prompts')

    args = parser.parse_args()

    if args.prompts_file:
        prompt = read_prompt_from_file(args.prompts_file)
    else:
        prompt = args.prompt

    print("Semantic Similarity Test Between Python and llama.cpp Embedding Models")
    print("=" * 70)

    # Single prompt detailed comparison
    print(f"\nTesting with prompt: '{prompt}'")

    # Load the python model to get configuration information and also to load the tokenizer.
    print("Loading model and tokenizer using AutoTokenizer:", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)

    if unreleased_model_name:
        model_name_lower = unreleased_model_name.lower()
        unreleased_module_path = f"transformers.models.{model_name_lower}.modular_{model_name_lower}"
        if args.causal:
            class_name = f"{unreleased_model_name}ForCausalLM"
        else:
            class_name = f"{unreleased_model_name}Model"
        print(f"Model class: {class_name}")
        print(f"Importing unreleased model module: {unreleased_module_path}")

        try:
            model_class = getattr(importlib.import_module(unreleased_module_path), class_name)
            model = model_class.from_pretrained(args.model_path)
        except (ImportError, AttributeError) as e:
            print(f"Failed to import or load model: {e}")
            exit(1)
    else:
        if args.causal:
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
        else:
            model = AutoModel.from_pretrained(args.model_path)

    encoded = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    n_tokens = len(tokens)
    print(f"n_tokens: {n_tokens}");
    print(f"hidden_size: {model.config.hidden_size}")

    # Load binary embeddings from data directory.
    llamacpp_embeddings = load_embeddings_from_file(args.cpp_embeddings, n_tokens, model.config.hidden_size)
    python_embeddings = load_embeddings_from_file(args.python_embeddings, n_tokens, model.config.hidden_size)

    # Run comparison
    results = test_single_prompt_similarity(python_embeddings, llamacpp_embeddings, tokens, prompt)

    # Summary
    print(f"\n=== SUMMARY ===")
    avg_cross_sim = np.mean(results['cross_model_similarities'])
    print(f"Average cross-model similarity: {avg_cross_sim:.4f}")
    print(f"Similarity matrix RMS difference: {results['rms_diff']:.4f}")

    # Quality assessment
    if avg_cross_sim > 0.95:
        print("✅ EXCELLENT: Models are highly similar")
    elif avg_cross_sim > 0.90:
        print("✅ VERY GOOD: Models are very similar")
    elif avg_cross_sim > 0.80:
        print("⚠️  GOOD: Models are reasonably similar")
    elif avg_cross_sim > 0.70:
        print("⚠️  FAIR: Models have some differences")
    else:
        print("❌ POOR: Models are significantly different")

if __name__ == "__main__":
    main()
