"""Test fp32 K projection to reduce fp16 drift."""
import torch
import torch.nn.functional as F
import math
import pathlib
import importlib.util

# Load the existing module
spec = importlib.util.spec_from_file_location(
    "export_gemma4_prefill",
    "/Users/user/Developer/GitHub/llama.cpp/tools/ane-mtp/export-gemma4-prefill.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Gemma4InitialSlab = mod.Gemma4InitialSlab
rms_norm = mod.rms_norm
rope = mod.rope


class Gemma4FP32K(Gemma4InitialSlab):
    """Same as base, but K projection runs in fp32 and stays fp32 through rope."""
    def forward(self, token_ids, positions):
        batch, sequence = token_ids.shape
        current = F.embedding(token_ids.to(torch.int64), self.embedding) * math.sqrt(self.hidden)
        normed = rms_norm(current, self.attn_norm)
        # K projection in fp32 to avoid K matmul fp16 drift
        keys = F.linear(normed.float(), self.k.float()).reshape(batch, sequence, self.kv_heads, self.head_dim)
        # Keep Q and V in fp16 for the rest of the model
        query = F.linear(normed, self.q).reshape(batch, sequence, self.heads, self.head_dim)
        values = F.linear(normed, self.v).reshape(batch, sequence, self.kv_heads, self.head_dim)
        query = rope(rms_norm(query, self.q_norm), positions, 10000.0).float()
        keys = rope(rms_norm(keys, self.k_norm), positions, 10000.0)
        values = rms_norm(values, self.k_norm) * self.v_norm_inverse
        expanded_k = keys.repeat_interleave(self.heads // self.kv_heads, dim=2)
        expanded_v = values.repeat_interleave(self.heads // self.kv_heads, dim=2)
        query_heads = query.permute(0, 2, 1, 3)
        key_heads = expanded_k.permute(0, 2, 3, 1)
        scores = torch.matmul(query_heads, key_heads)
        causal = positions[:, None, :] <= positions[:, :, None]
        scores = torch.where(causal[:, None], scores, torch.full_like(scores, -1.0e4))
        probs = torch.softmax(scores.float(), dim=-1).to(current.dtype)
        attended = torch.matmul(probs, expanded_v.permute(0, 2, 1, 3))
        attended = attended.permute(0, 2, 1, 3).reshape(batch, sequence, -1)
        attended = rms_norm(F.linear(attended, self.o), self.post_attn) + current
        ffn_input = rms_norm(attended, self.ffn_norm)
        ffn = F.gelu(F.linear(ffn_input, self.gate), approximate='tanh') * F.linear(ffn_input, self.up)
        output = (rms_norm(F.linear(ffn, self.down), self.post_ffn) + attended) * self.scale
        return output, keys.reshape(batch, sequence, -1), values.reshape(batch, sequence, -1)


# Compute the fp32-K reference outputs to compare against fp16
import numpy as np
source = pathlib.Path('/Volumes/Julian T7/models/gemma-4-12B-it-qat-q4_0-unquantized')
m_fp32k = Gemma4FP32K(source, 3840, 16, 8, 256).eval()
for bucket in (128, 256, 512):
    tokens = torch.zeros((1, bucket), dtype=torch.int32)
    pos = torch.arange(bucket, dtype=torch.int32).reshape(1, -1)
    with torch.no_grad():
        h, k, v = m_fp32k(tokens, pos)
    h_np = h.float().numpy()
    k_np = k.float().numpy()
    v_np = v.float().numpy()
    np.save(f'/tmp/fp32k_ref_s{bucket}_hidden.npy', h_np)
    np.save(f'/tmp/fp32k_ref_s{bucket}_key.npy', k_np)
    np.save(f'/tmp/fp32k_ref_s{bucket}_value.npy', v_np)
    print(f'saved fp32k reference for s{bucket}: h range [{h_np.min():.4f}, {h_np.max():.4f}], k range [{k_np.min():.4f}, {k_np.max():.4f}]')
