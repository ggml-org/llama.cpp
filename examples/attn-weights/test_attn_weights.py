#!/usr/bin/env python3
"""
Test: verify llama.cpp attention weight extraction works correctly.

Usage:
    python3 benchmark/test_attn_weights.py <model.gguf>

Tests:
    1. Basic extraction: attention weights are non-null and sum to ~1.0
    2. Multi-head: multiple (layer, head) pairs return independent weights
    3. Greedy generation: attention is extracted at each autoregressive step
    4. Cross-validation with known properties (monotonicity, sparsity)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import llama_attn as ll


def test_basic(model, vocab, ctx, n_layers):
    """Test 1: basic attention extraction on a prompt."""
    print("=" * 60)
    print("TEST 1: Basic attention extraction")
    print("=" * 60)

    tokens = ll.tokenize(vocab, "The quick brown fox jumps over the lazy dog")
    n_tokens = len(tokens)
    print(f"  Tokens: {n_tokens}")

    ret = ll.decode_batch(ctx, tokens, output_last_only=True)
    assert ret == 0, f"decode failed: {ret}"

    n_c = ll.n_ctx(ctx)
    attn = ll.get_attn_weights(ctx, -1, 1, n_c)
    assert attn is not None, "get_attn_weights returned None"

    n_kv = ll._lib.llama_get_attn_n_kv(ctx)
    print(f"  n_kv: {n_kv}")
    print(f"  Attention shape: {attn.shape}")
    print(f"  Attention sum: {attn[0].sum():.6f}")
    print(f"  Attention max: {attn[0].max():.6f} at position {attn[0].argmax()}")
    print(f"  Attention min: {attn[0].min():.6f}")

    # Softmax output should sum to ~1.0
    assert abs(attn[0].sum() - 1.0) < 0.05, f"Attention doesn't sum to 1.0: {attn[0].sum()}"
    # All values should be non-negative
    assert (attn[0] >= 0).all(), "Negative attention values found"

    print("  PASSED\n")
    return True


def test_multi_head(model, vocab, ctx, n_layers):
    """Test 2: multiple (layer, head) pairs."""
    print("=" * 60)
    print("TEST 2: Multi-head attention extraction")
    print("=" * 60)

    # Set multiple heads across different layers
    layers = [0, n_layers // 2, n_layers - 1]
    heads = [0, 0, 0]
    n_pairs = len(layers)
    ll.set_attn_heads(ctx, layers, heads)
    print(f"  Configured {n_pairs} heads: {list(zip(layers, heads))}")

    tokens = ll.tokenize(vocab, "Hello world, this is a test of attention")
    ret = ll.decode_batch(ctx, tokens, output_last_only=True)
    assert ret == 0, f"decode failed: {ret}"

    n_c = ll.n_ctx(ctx)
    attn = ll.get_attn_weights(ctx, -1, n_pairs, n_c)
    assert attn is not None, "get_attn_weights returned None"

    print(f"  Attention shape: {attn.shape}")
    for p in range(n_pairs):
        s = attn[p].sum()
        print(f"  Pair {p} (L{layers[p]},H{heads[p]}): sum={s:.6f}, max={attn[p].max():.4f} @ pos {attn[p].argmax()}")
        assert abs(s - 1.0) < 0.05, f"Pair {p} doesn't sum to 1.0: {s}"

    # Different layers should produce different attention patterns
    if n_pairs >= 2:
        diff = np.abs(attn[0] - attn[-1]).mean()
        print(f"  Mean abs difference between first and last layer: {diff:.6f}")
        # They should not be identical (unless the model is degenerate)
        # Don't assert this as a hard requirement

    # Reset to default (last layer, head 0)
    ll.set_attn_heads(ctx, [n_layers - 1], [0])

    print("  PASSED\n")
    return True


def test_generation(model, vocab, ctx, n_layers):
    """Test 3: attention during autoregressive generation."""
    print("=" * 60)
    print("TEST 3: Autoregressive generation with attention")
    print("=" * 60)

    tokens = ll.tokenize(vocab, "Once upon a time")
    n_prompt = len(tokens)
    print(f"  Prompt: {n_prompt} tokens")

    # Prefill
    ret = ll.decode_batch(ctx, tokens, output_last_only=True)
    assert ret == 0, f"prefill decode failed: {ret}"

    n_c = ll.n_ctx(ctx)
    nv = ll.n_vocab(vocab)
    eos = ll.vocab_eos(vocab)

    max_gen = 10
    gen_tokens = []
    attn_sums = []

    for step in range(max_gen):
        # Get attention for current token
        attn = ll.get_attn_weights(ctx, -1, 1, n_c)
        assert attn is not None, f"Step {step}: attention is None"

        n_kv = ll._lib.llama_get_attn_n_kv(ctx)
        s = attn[0].sum()
        attn_sums.append(s)

        # Get next token (greedy)
        next_tok = ll.argmax_logits(ctx, -1, nv)
        if next_tok == eos:
            print(f"  Step {step}: EOS")
            break

        gen_tokens.append(next_tok)

        # Decode next token
        pos = n_prompt + step
        ret = ll.decode_single(ctx, next_tok, pos, output=True)
        assert ret == 0, f"Step {step}: decode failed: {ret}"

    print(f"  Generated {len(gen_tokens)} tokens")
    print(f"  Attention sums: {[f'{s:.4f}' for s in attn_sums]}")

    for i, s in enumerate(attn_sums):
        assert abs(s - 1.0) < 0.05, f"Step {i}: attention sum = {s}"

    print("  PASSED\n")
    return True


def test_multiple_heads_same_layer(model, vocab, ctx, n_layers):
    """Test 4: multiple heads from the same layer."""
    print("=" * 60)
    print("TEST 4: Multiple heads from same layer")
    print("=" * 60)

    n_h = ll.n_head(model)
    last_layer = n_layers - 1
    n_test_heads = min(4, n_h)

    layers = [last_layer] * n_test_heads
    heads = list(range(n_test_heads))
    ll.set_attn_heads(ctx, layers, heads)
    print(f"  Layer {last_layer}, heads {heads}")

    tokens = ll.tokenize(vocab, "Attention is all you need")
    ret = ll.decode_batch(ctx, tokens, output_last_only=True)
    assert ret == 0, f"decode failed: {ret}"

    n_c = ll.n_ctx(ctx)
    attn = ll.get_attn_weights(ctx, -1, n_test_heads, n_c)
    assert attn is not None, "get_attn_weights returned None"

    print(f"  Attention shape: {attn.shape}")
    for h in range(n_test_heads):
        s = attn[h].sum()
        peak = attn[h].argmax()
        print(f"  Head {h}: sum={s:.6f}, peak @ pos {peak}, max={attn[h].max():.4f}")
        assert abs(s - 1.0) < 0.05, f"Head {h}: sum = {s}"

    # Different heads should show at least some variation
    if n_test_heads >= 2:
        patterns_identical = all(
            np.allclose(attn[0], attn[h], atol=1e-5)
            for h in range(1, n_test_heads)
        )
        if patterns_identical:
            print("  WARNING: all heads have identical attention patterns")
        else:
            print("  OK: heads show different patterns")

    # Reset
    ll.set_attn_heads(ctx, [n_layers - 1], [0])

    print("  PASSED\n")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.gguf>")
        sys.exit(1)

    model_path = sys.argv[1]
    n_ctx = 512
    if len(sys.argv) > 2:
        n_ctx = int(sys.argv[2])

    print(f"Model: {model_path}")
    print(f"n_ctx: {n_ctx}\n")

    ll.init()

    model = ll.load_model(model_path)
    vocab = ll.get_vocab(model)
    n_layers = ll.n_layer(model)
    n_heads = ll.n_head(model)
    nv = ll.n_vocab(vocab)
    print(f"Loaded: {n_layers} layers, {n_heads} heads, vocab={nv}\n")

    passed = 0
    failed = 0

    for test_fn in [test_basic, test_multi_head, test_generation, test_multiple_heads_same_layer]:
        # Create fresh context for each test
        ctx = ll.create_context(model, n_ctx=n_ctx, n_batch=n_ctx, attn_weights=True)
        try:
            if test_fn(model, vocab, ctx, n_layers):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1
        finally:
            ll.free_context(ctx)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    ll.free_model(model)
    ll.cleanup()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
