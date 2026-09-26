"""Verify the converted F16 GGUF tensors match the PyTorch reference checkpoint.

Per-tensor name/shape/dtype comparison plus a max-abs-value sanity check on a
few representative tensors (token_embd, blk.0.attn_qkv, head.0.attn_qkv,
scorer.1, type_emb).
"""
import os
import sys

import numpy as np

from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-py"))
import gguf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "laya-f16.gguf")
MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", os.path.join(ROOT, "multilingual"))
ST_PATH = os.environ.get("LAYA_ST_PATH", os.path.join(MODEL_DIR, "model.safetensors"))


def main():
    st = load_file(ST_PATH)
    rd = gguf.GGUFReader(GGUF_PATH)

    gguf_tensors = {}
    for t in rd.tensors:
        shape = tuple(int(d) for d in t.shape)
        gguf_tensors[t.name] = shape

    tm = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LAYA, 22)

    mismatches = []
    missing = []
    for src, t in st.items():
        if src == "temperature":
            continue
        # mirror conversion/laya.py filter_tensors (encoder. prefix strip +
        # nn.MultiheadAttention in_proj rename) then map via TensorNameMap
        name = src
        if name.startswith("encoder."):
            name = name[len("encoder."):]
        if name.endswith("self_attn.in_proj_weight"):
            name = name[:-len("in_proj_weight")] + "in_proj.weight"
        elif name.endswith("self_attn.in_proj_bias"):
            name = name[:-len("in_proj_bias")] + "in_proj.bias"
        mapped = tm.get_name(name, (".weight", ".bias"))
        if mapped is None:
            missing.append((src, name, "no mapping"))
            continue
        if mapped not in gguf_tensors:
            missing.append((src, name, mapped))
            continue
        gshape = gguf_tensors[mapped]
        sshape = tuple(t.shape)
        if tuple(reversed(sshape)) != gshape:
            mismatches.append((src, mapped, sshape, gshape))

    print(f"source tensors (excl temperature): {len(st) - 1}")
    print(f"gguf tensors: {len(gguf_tensors)}")
    print(f"missing: {len(missing)}")
    for src, name, mapped in missing:
        print("  MISSING", src, "->", name, "=>", mapped)
    print(f"shape mismatches: {len(mismatches)}")
    for src, name, ss, gs in mismatches:
        print("  SHAPE", src, "->", name, "src", ss, "gguf", gs)

    # value sanity: find the gguf tensor by name and compare max abs
    probe = {
        "token_embd.weight": "encoder.embeddings.tok_embeddings.weight",
        "blk.0.attn_qkv.weight": "encoder.layers.0.attn.Wqkv.weight",
        "blk.0.ffn_up.weight": "encoder.layers.0.mlp.Wi.weight",
        "head.0.attn_qkv.weight": "head.layers.0.self_attn.in_proj_weight",
        "head.0.ffn_up.weight": "head.layers.0.linear1.weight",
        "scorer.1.weight": "scorer.1.weight",
        "scorer.3.weight": "scorer.3.weight",
        "act_head.0.weight": "act_head.0.weight",
        "act_head.2.weight": "act_head.2.weight",
        "type_emb.weight": "type_emb.weight",
        "token_embd_norm.weight": "encoder.embeddings.norm.weight",
        "output_norm.weight": "encoder.final_norm.weight",
    }
    for gguf_name, src_name in probe.items():
        if gguf_name not in gguf_tensors:
            print("  PROBE MISSING", gguf_name)
            continue
        t = next(x for x in rd.tensors if x.name == gguf_name)
        data = t.data.reshape(t.shape[::-1])
        ref = st[src_name].numpy()
        print(f"{gguf_name:28s} max|ref|={np.abs(ref).max():.4f} "
              f"max|gguf|={np.abs(data).max():.4f} "
              f"corr={np.corrcoef(ref.ravel(), data.ravel())[0,1]:.6f}")


if __name__ == "__main__":
    main()
