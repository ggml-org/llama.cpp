#!/usr/bin/env python3
"""
spec_harness.py — Reusable speculative decoding validation tool.

Reads binary captures from llama-spec-harness and validates the EAGLE3
decoder against ground-truth next tokens. Determines whether a draft model
is undertrained vs whether the C++ pipeline is broken.

Usage:
  python scripts/spec_harness.py validate \
    --capture /tmp/spec_harness/capture_france.bin \
    --eagle3-model /path/to/Bonsai-4B-EAGLE3/ \
    [--with-kv-history]  # test with accumulated KV (Option B behavior)

  python scripts/spec_harness.py feature-stats \
    --capture /tmp/spec_harness/capture_france.bin
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ── Binary capture format ──────────────────────────────────────────────
# Header: magic[4]='SPEC', version(u32), n_embd(u32), n_layers(u32),
#         layer_ids[3](i32), n_records(u32)
# Per-record: token_id(i32), next_token_id(i32), features(f32 × n_layers × n_embd)

HEADER_FMT = '<4s I I I 3i I'  # magic, version, n_embd, n_layers, 3×layer_id, n_records
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def read_capture(path):
    """Read a binary capture file. Returns header dict and list of records."""
    with open(path, 'rb') as f:
        raw = f.read(HEADER_SIZE)
        (magic, version, n_embd, n_layers,
         l0, l1, l2, n_records) = struct.unpack(HEADER_FMT, raw)

        assert magic == b'SPEC', f"Bad magic: {magic}"
        assert version == 1, f"Unsupported version: {version}"

        header = {
            'n_embd': n_embd,
            'n_layers': n_layers,
            'layer_ids': [l0, l1, l2],
            'n_records': n_records,
        }

        feat_size = n_layers * n_embd
        record_fmt = f'<ii{feat_size}f'
        record_size = struct.calcsize(record_fmt)

        records = []
        for _ in range(n_records):
            raw = f.read(record_size)
            if len(raw) < record_size:
                break
            vals = struct.unpack(record_fmt, raw)
            token_id = vals[0]
            next_token_id = vals[1]
            features = np.array(vals[2:], dtype=np.float32)
            records.append({
                'token_id': token_id,
                'next_token_id': next_token_id,
                'features': features,
            })

    return header, records


def feature_stats(header, records):
    """Print feature statistics from a capture file."""
    n_embd = header['n_embd']
    n_layers = header['n_layers']
    layer_ids = header['layer_ids']

    import logging; logging.warning(f"=== Feature Statistics ({len(records)} records) ===")
    import logging; logging.warning(f"n_embd={n_embd}, n_layers={n_layers}, layers={layer_ids[:n_layers]}")
    import logging; logging.warning()

    all_feats = np.stack([r['features'] for r in records])  # [N, n_layers*n_embd]

    for i in range(n_layers):
        layer_feats = all_feats[:, i*n_embd:(i+1)*n_embd]
        import logging; logging.warning(f"Layer {layer_ids[i]}:")
        import logging; logging.warning(f"  mean={layer_feats.mean():.4f}  std={layer_feats.std():.4f}")
        import logging; logging.warning(f"  min={layer_feats.min():.4f}   max={layer_feats.max():.4f}")
        import logging; logging.warning(f"  per-token norms: mean={np.linalg.norm(layer_feats, axis=1).mean():.2f}")
        import logging; logging.warning()

    # Combined features
    import logging; logging.warning(f"Combined ({n_layers}×{n_embd} = {n_layers*n_embd}):")
    import logging; logging.warning(f"  mean={all_feats.mean():.4f}  std={all_feats.std():.4f}")
    import logging; logging.warning(f"  min={all_feats.min():.4f}   max={all_feats.max():.4f}")


def rms_norm(x, w, eps=1e-6):
    """RMS normalization."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * w


def eagle3_decoder_forward(tensors, token_id, g_embd, kv_cache=None, position=0):
    """
    Full EAGLE3 decoder forward pass (1-layer transformer).

    Args:
        tensors: dict of safetensors weights
        token_id: current token ID (int)
        g_embd: g_embeddings tensor [n_embd]
        kv_cache: optional dict with 'K' and 'V' tensors [n_kv_heads, seq_len, head_dim]
        position: absolute position for RoPE

    Returns:
        logits: [vocab_size] tensor
        new_kv_cache: updated KV cache
        prenorm: pre-norm hidden state for recurrence [n_embd]
    """
    n_embd = 2560
    n_heads = 32
    n_kv_heads = 8
    head_dim = 80
    n_rep = n_heads // n_kv_heads  # 4
    rope_theta = 5000000.0

    # Token embedding
    tok_vec = tensors['embed_tokens.weight'][token_id].float()

    # Norms
    tok_normed = rms_norm(tok_vec, tensors['midlayer.input_layernorm.weight'].float())
    g_normed = rms_norm(g_embd, tensors['midlayer.hidden_norm.weight'].float())

    # Concatenate: [tok_normed; g_normed] → [5120]
    concat_input = torch.cat([tok_normed, g_normed])

    # QKV projections
    Q = tensors['midlayer.self_attn.q_proj.weight'].float() @ concat_input  # [2560]
    K = tensors['midlayer.self_attn.k_proj.weight'].float() @ concat_input  # [640]
    V = tensors['midlayer.self_attn.v_proj.weight'].float() @ concat_input  # [640]

    # Reshape for heads
    Q = Q.view(n_heads, head_dim)      # [32, 80]
    K = K.view(n_kv_heads, head_dim)   # [8, 80]
    V = V.view(n_kv_heads, head_dim)   # [8, 80]

    # RoPE on Q and K
    def apply_rope(x, pos, theta=rope_theta):
        """Apply rotary position embeddings."""
        d = x.shape[-1]
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        angles = pos * freqs
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        out_even = x_even * cos_a - x_odd * sin_a
        out_odd  = x_even * sin_a + x_odd * cos_a
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

    Q = apply_rope(Q, position)
    K = apply_rope(K, position)

    # KV cache management
    if kv_cache is not None:
        K_cache = torch.cat([kv_cache['K'], K.unsqueeze(1)], dim=1)  # [8, seq+1, 80]
        V_cache = torch.cat([kv_cache['V'], V.unsqueeze(1)], dim=1)  # [8, seq+1, 80]
    else:
        K_cache = K.unsqueeze(1)  # [8, 1, 80]
        V_cache = V.unsqueeze(1)  # [8, 1, 80]

    new_kv_cache = {'K': K_cache, 'V': V_cache}

    # GQA: repeat K/V for each head group
    K_expanded = K_cache.repeat_interleave(n_rep, dim=0)  # [32, seq, 80]
    V_expanded = V_cache.repeat_interleave(n_rep, dim=0)  # [32, seq, 80]

    # Attention: Q [32, 80] @ K^T [32, 80, seq] → [32, seq]
    Q_3d = Q.unsqueeze(1)  # [32, 1, 80]
    scores = torch.bmm(Q_3d, K_expanded.transpose(1, 2)) / (head_dim ** 0.5)  # [32, 1, seq]
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.bmm(attn_weights, V_expanded)  # [32, 1, 80]
    attn_output = attn_output.squeeze(1).reshape(-1)    # [2560]

    # O projection + residual on g_embd
    attn_output = tensors['midlayer.self_attn.o_proj.weight'].float() @ attn_output
    ffn_inp = attn_output + g_embd

    # FFN
    ffn_normed = rms_norm(ffn_inp, tensors['midlayer.post_attention_layernorm.weight'].float())
    gate = F.silu(tensors['midlayer.mlp.gate_proj.weight'].float() @ ffn_normed)
    up = tensors['midlayer.mlp.up_proj.weight'].float() @ ffn_normed
    ffn_out = tensors['midlayer.mlp.down_proj.weight'].float() @ (gate * up)
    decoder_out = ffn_out + ffn_inp

    # Prenorm output (for recurrence — becomes next g_embd)
    prenorm = decoder_out.clone()

    # Output norm → logits (using tok_embd as lm_head, since lm_head is untrained)
    out_normed = rms_norm(decoder_out, tensors['norm.weight'].float())
    logits = tensors['embed_tokens.weight'].float() @ out_normed  # [vocab]

    return logits, new_kv_cache, prenorm


def validate(header, records, eagle3_path, with_kv_history=False):
    """
    Validate EAGLE3 decoder against ground-truth next tokens.

    Tests the full pipeline: features → FC → decoder → prediction.
    """
    import safetensors.torch as st

    import logging; logging.warning(f"Loading EAGLE3 model from {eagle3_path}...")
    tensors = st.load_file(str(eagle3_path / 'model.safetensors'))

    n_embd = header['n_embd']
    fc_weight = tensors['fc.weight'].float()  # [2560, 7680]

    import logging; logging.warning(f"FC weight shape: {fc_weight.shape}")
    import logging; logging.warning(f"Using embed_tokens as lm_head (lm_head is untrained)")
    import logging; logging.warning(f"KV history: {'ENABLED' if with_kv_history else 'DISABLED (single-token, matches C++)'}")
    import logging; logging.warning()

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = len(records)

    kv_cache = None
    spreads = []
    confidences = []

    for idx, rec in enumerate(records):
        token_id = rec['token_id']
        next_token_id = rec['next_token_id']
        features = torch.from_numpy(rec['features'])

        # FC projection: features → g_embd
        g_embd = fc_weight @ features  # [2560]

        # Decoder forward
        position = idx  # approximate absolute position
        if not with_kv_history:
            kv_cache = None  # reset each step (matches C++ Option A)

        logits, kv_cache, prenorm = eagle3_decoder_forward(
            tensors, token_id, g_embd, kv_cache, position
        )

        # Check accuracy
        spread = (logits.max() - logits.min()).item()
        spreads.append(spread)

        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max().item()
        confidences.append(top_prob)

        top_k = logits.topk(10)
        predicted = top_k.indices[0].item()
        top5_ids = set(top_k.indices[:5].tolist())
        top10_ids = set(top_k.indices[:10].tolist())

        if predicted == next_token_id:
            top1_correct += 1
        if next_token_id in top5_ids:
            top5_correct += 1
        if next_token_id in top10_ids:
            top10_correct += 1

        # Per-step output
        status = "✓" if predicted == next_token_id else "✗"
        in_top5 = "T5" if next_token_id in top5_ids else "  "
        import logging; logging.warning(f"  [{idx:3d}] {status} {in_top5} | tok={token_id:6d} → "
              f"pred={predicted:6d} truth={next_token_id:6d} | "
              f"spread={spread:7.1f} conf={top_prob:.3f}")

    import logging; logging.warning()
    import logging; logging.warning("=" * 70)
    import logging; logging.warning(f"=== Speculative Harness Report ===")
    import logging; logging.warning(f"=" * 70)
    import logging; logging.warning(f"EAGLE3 model:  {eagle3_path}")
    import logging; logging.warning(f"Records:       {total}")
    import logging; logging.warning(f"KV history:    {'Yes' if with_kv_history else 'No (single-token)'}")
    import logging; logging.warning()
    import logging; logging.warning(f"Draft Accuracy:")
    import logging; logging.warning(f"  Top-1:  {top1_correct/total:6.1%} ({top1_correct}/{total})")
    import logging; logging.warning(f"  Top-5:  {top5_correct/total:6.1%} ({top5_correct}/{total})")
    import logging; logging.warning(f"  Top-10: {top10_correct/total:6.1%} ({top10_correct}/{total})")
    import logging; logging.warning()
    import logging; logging.warning(f"Logit Diagnostics:")
    import logging; logging.warning(f"  Mean spread:     {np.mean(spreads):8.1f}")
    import logging; logging.warning(f"  Min spread:      {np.min(spreads):8.1f}")
    import logging; logging.warning(f"  Max spread:      {np.max(spreads):8.1f}")
    import logging; logging.warning(f"  Mean confidence: {np.mean(confidences):8.4f}")
    import logging; logging.warning(f"  Max confidence:  {np.max(confidences):8.4f}")
    import logging; logging.warning()

    # Verdict
    top5_pct = top5_correct / total
    mean_spread = np.mean(spreads)

    if top5_pct < 0.05:
        verdict = "MODEL_UNDERTRAINED"
        explanation = ("The EAGLE3 model cannot predict next tokens even with perfect features. "
                      "This is NOT a code bug — the model needs more training.")
    elif top5_pct < 0.15:
        verdict = "MODEL_WEAK"
        explanation = ("The EAGLE3 model has some predictive ability but is too weak for "
                      "useful speculative decoding. Consider retraining or finding better weights.")
    elif mean_spread < 10:
        verdict = "PIPELINE_BROKEN"
        explanation = ("Features or decoder weights are corrupted — logit spread is too low. "
                      "Debug the C++ extraction/decoder pipeline.")
    else:
        verdict = "MODEL_OK"
        explanation = ("The EAGLE3 model has meaningful predictive ability. "
                      "If C++ speculative decoding still fails, the issue is in the C++ pipeline.")

    import logging; logging.warning(f"Verdict: {verdict}")
    import logging; logging.warning(f"  {explanation}")
    import logging; logging.warning()

    return verdict


def main():
    parser = argparse.ArgumentParser(description='Speculative Decoding Validation Harness')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # feature-stats
    p_stats = subparsers.add_parser('feature-stats', help='Print feature statistics')
    p_stats.add_argument('--capture', required=True, help='Path to capture .bin file')

    # validate
    p_val = subparsers.add_parser('validate', help='Validate EAGLE3 decoder accuracy')
    p_val.add_argument('--capture', required=True, help='Path to capture .bin file')
    p_val.add_argument('--eagle3-model', required=True, help='Path to EAGLE3 HF model dir')
    p_val.add_argument('--with-kv-history', action='store_true',
                       help='Accumulate KV cache across tokens (tests Option B behavior)')

    args = parser.parse_args()

    if args.command == 'feature-stats':
        header, records = read_capture(args.capture)
        feature_stats(header, records)

    elif args.command == 'validate':
        header, records = read_capture(args.capture)
        eagle3_path = Path(args.eagle3_model)
        validate(header, records, eagle3_path, args.with_kv_history)


if __name__ == '__main__':
    main()
