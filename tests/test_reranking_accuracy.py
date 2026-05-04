#!/usr/bin/env python3
"""End-to-end reranking test for jina-reranker-v3.

Uses llama-embedding --pooling last to get 512-dim projected embeddings,
then computes cosine similarity for relevance scoring.
"""
import subprocess
import numpy as np
import sys

MODEL = "/home/elias/projects/jina-test/jina-reranker-v3-v2.gguf"
LLAMA = "./build/bin/llama-embedding"

def get_embedding(text):
    result = subprocess.run(
        [LLAMA, "-m", MODEL, "--pooling", "last", "-n", "1", "-p", text,
         "--embd-output-format", "raw"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()[-200:]}", file=sys.stderr)
        return None
    embed_strs = result.stdout.strip().split()
    return np.array([float(x) for x in embed_strs], dtype=np.float32)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("=" * 60)
print("jina-reranker-v3 Reranking Accuracy Test")
print("=" * 60)

# Test 1: Unit vectors
print("\n--- Test 1: Unit vectors ---")
e = get_embedding("hello world")
assert e is not None, "Failed to get embedding"
assert e.shape == (512,), f"Wrong shape: {e.shape}"
norm = np.linalg.norm(e)
print(f"  Shape: {e.shape}, Norm: {norm:.6f}")
assert abs(norm - 1.0) < 0.01, f"Embeddings not normalized: {norm}"
print("  PASS")

# Test 2: Deterministic
print("\n--- Test 2: Deterministic ---")
e2 = get_embedding("hello world")
sim = cosine_sim(e, e2)
print(f"  Self-similarity: {sim:.6f}")
assert abs(sim - 1.0) < 0.001, f"Not deterministic: {sim}"
print("  PASS")

# Test 3: Different texts produce different embeddings
print("\n--- Test 3: Different texts -> different embeddings ---")
e_a = get_embedding("Python programming")
e_b = get_embedding("Cooking Italian pasta")
sim = cosine_sim(e_a, e_b)
print(f"  Similarity: {sim:.4f}")
assert sim < 0.99, f"Too similar for different topics: {sim}"
print("  PASS")

# Test 4: Relevant docs cluster together
print("\n--- Test 4: Relevant docs cluster together ---")
docs = {
    "python1": "Python is a high-level programming language",
    "python2": "Programming in Python uses indentation",
    "cooking": "Cooking pasta requires boiling water",
}
embeddings = {}
for name, doc in docs.items():
    embeddings[name] = get_embedding(doc)

rel_rel = cosine_sim(embeddings["python1"], embeddings["python2"])
cross = cosine_sim(embeddings["python1"], embeddings["cooking"])
print(f"  Relevant-relevant: {rel_rel:.4f}")
print(f"  Cross-topic:       {cross:.4f}")
assert rel_rel > cross, f"Relevant docs should be closer: {rel_rel} vs {cross}"
print("  PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
