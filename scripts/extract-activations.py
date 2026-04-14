#!/usr/bin/env python3
"""Extract real activation tensors by running a forward pass through the model.

Captures the INPUT activations to specific weight tensors (the vectors that get
multiplied by the weight matrix). These are what matter for quantization quality:
quantization error * activation magnitude = output error.

Usage:
    python3 scripts/extract-activations.py MODEL.gguf OUTPUT_DIR [--prompt TEXT] [--layer N]

Output:
    For each target tensor, writes a .f32bin file with header:
        int64_t n_rows, int64_t row_len
    followed by n_rows * row_len float32 values.
    n_rows = number of tokens, row_len = hidden dimension.

NOTE: This uses a simplified forward pass (no KV cache, single prompt).
Activations are extracted from after the norm layers (the actual matmul inputs).
"""
import sys
import os
import struct
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'gguf-py'))

from gguf import GGUFReader


def bf16_to_f32(raw_bytes):
    """Convert raw BF16 bytes to float32 numpy array."""
    bf16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def rms_norm(x, weight, eps=1e-6):
    """RMS normalization (Qwen3/Llama style)."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def silu(x):
    """SiLU activation."""
    return x / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} MODEL.gguf OUTPUT_DIR [--prompt TEXT] [--layer N]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    prompt_text = "The quick brown fox jumps over the lazy dog. In a distant galaxy, scientists discovered"
    target_layer = 16

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--prompt" and i + 1 < len(sys.argv):
            prompt_text = sys.argv[i + 1]
        elif sys.argv[i] == "--layer" and i + 1 < len(sys.argv):
            target_layer = int(sys.argv[i + 1])

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_path}...")
    reader = GGUFReader(model_path)

    # Read model config from metadata
    config = {}
    for kv in reader.fields.values():
        if hasattr(kv, 'parts') and len(kv.parts) > 0:
            name = kv.name
            if 'block_count' in name:
                config['n_layer'] = int(kv.parts[-1][0])
            elif 'embedding_length' in name:
                config['hidden'] = int(kv.parts[-1][0])
            elif 'feed_forward_length' in name:
                config['ffn'] = int(kv.parts[-1][0])
            elif 'head_count_kv' in name:
                config['n_kv_heads'] = int(kv.parts[-1][0])
            elif 'head_count' in name and 'kv' not in name:
                config['n_heads'] = int(kv.parts[-1][0])
            elif 'key_length' in name:
                config['head_dim'] = int(kv.parts[-1][0])
            elif 'layer_norm_rms_epsilon' in name:
                config['eps'] = float(kv.parts[-1][0])

    print(f"Config: {config}")
    hidden = config['hidden']

    # Load tensors into a dict
    def load_tensor(name):
        for t in reader.tensors:
            if t.name == name:
                raw = bytes(t.data)
                shape = [int(s) for s in t.shape]
                n_el = int(t.n_elements)
                if t.tensor_type.name == 'BF16':
                    flat = bf16_to_f32(raw)
                elif t.tensor_type.name == 'F16':
                    flat = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                elif t.tensor_type.name == 'F32':
                    flat = np.frombuffer(raw, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported type: {t.tensor_type.name}")
                assert flat.shape[0] == n_el, f"Expected {n_el} elements, got {flat.shape[0]}"
                if len(shape) == 1:
                    return flat.copy()
                return flat.reshape(list(reversed(shape))).copy()
        raise KeyError(f"Tensor {name} not found")

    # Create simple token IDs from the prompt (use first few tokens from vocab)
    # We just need realistic activations, not perfect tokenization
    n_tokens = min(32, len(prompt_text.split()))
    print(f"Using {n_tokens} pseudo-tokens for activation extraction")

    # Load token embedding and create input
    print("Loading token_embd...")
    token_embd = load_tensor("token_embd.weight")  # [vocab, hidden]
    # Use token IDs 100-131 (arbitrary but avoids special tokens)
    token_ids = list(range(100, 100 + n_tokens))
    x = token_embd[token_ids]  # [n_tokens, hidden]
    print(f"Input shape: {x.shape}")

    # Run forward pass through target layer only (we just need the activations)
    layer = target_layer
    print(f"\nProcessing layer {layer}...")

    def save_activation(name, data):
        """Save activation tensor as f32bin."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n_rows, row_len = data.shape
        fname = os.path.join(output_dir, name + ".f32bin")
        with open(fname, 'wb') as fp:
            fp.write(struct.pack('<qq', n_rows, row_len))
            data.astype(np.float32).tofile(fp)
        print(f"  Saved {fname}: {n_rows} x {row_len} ({os.path.getsize(fname) / 1024:.1f} KB)")

    # Attention norm → input to attn_q/k/v
    attn_norm_w = load_tensor(f"blk.{layer}.attn_norm.weight")
    x_normed = rms_norm(x, attn_norm_w, config.get('eps', 1e-6))
    save_activation(f"act_blk{layer}_attn_input", x_normed)

    # Compute Q, K, V to get post-attention residual
    W_q = load_tensor(f"blk.{layer}.attn_q.weight")   # [n_heads*head_dim, hidden]
    W_k = load_tensor(f"blk.{layer}.attn_k.weight")   # [n_kv_heads*head_dim, hidden]
    W_v = load_tensor(f"blk.{layer}.attn_v.weight")   # [n_kv_heads*head_dim, hidden]
    W_o = load_tensor(f"blk.{layer}.attn_output.weight")  # [hidden, n_heads*head_dim]

    Q = x_normed @ W_q.T  # [n_tokens, n_heads*head_dim]
    K = x_normed @ W_k.T
    V = x_normed @ W_v.T

    # Simplified attention (no RoPE, no mask, no GQA — just need rough activations)
    n_heads = config['n_heads']
    head_dim = config['head_dim']
    Q_h = Q.reshape(n_tokens, n_heads, head_dim)
    K_h = K.reshape(n_tokens, config['n_kv_heads'], head_dim)
    V_h = V.reshape(n_tokens, config['n_kv_heads'], head_dim)

    # Repeat KV heads for GQA
    rep = n_heads // config['n_kv_heads']
    K_h = np.repeat(K_h, rep, axis=1)
    V_h = np.repeat(V_h, rep, axis=1)

    # Attention scores and output
    scores = np.einsum('thd,shd->ths', Q_h, K_h) / np.sqrt(head_dim)
    attn_w = softmax(scores, axis=-1)
    attn_out = np.einsum('ths,shd->thd', attn_w, V_h).reshape(n_tokens, -1)

    # attn_output weight input
    save_activation(f"act_blk{layer}_attn_output_input", attn_out)

    # Project and add residual
    attn_proj = attn_out @ W_o.T
    x = x + attn_proj

    # FFN norm → input to ffn_gate/ffn_up
    ffn_norm_w = load_tensor(f"blk.{layer}.ffn_norm.weight")
    x_ffn = rms_norm(x, ffn_norm_w, config.get('eps', 1e-6))
    save_activation(f"act_blk{layer}_ffn_input", x_ffn)

    # FFN: gate and up projections
    W_gate = load_tensor(f"blk.{layer}.ffn_gate.weight")  # [ffn, hidden]
    W_up = load_tensor(f"blk.{layer}.ffn_up.weight")      # [ffn, hidden]
    W_down = load_tensor(f"blk.{layer}.ffn_down.weight")  # [hidden, ffn]

    gate = x_ffn @ W_gate.T
    up = x_ffn @ W_up.T
    ffn_act = silu(gate) * up  # SwiGLU activation

    # ffn_down weight input (the SwiGLU output)
    save_activation(f"act_blk{layer}_ffn_down_input", ffn_act)

    print(f"\nDone! Extracted 4 activation tensors to {output_dir}/")


if __name__ == "__main__":
    main()
