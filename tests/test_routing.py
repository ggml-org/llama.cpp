import subprocess, sys

FFN_NAMES = [
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
    "blk.15.ffn_gate.weight",
    "blk.39.ffn_down.weight",
]
ATTN_NAMES = [
    "token_embd.weight",
    "output_norm.weight",
    "output.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
]

FFN_PATTERNS = ["ffn_norm","ffn_gate","ffn_up","ffn_down",
                "ffn_gate_exps","ffn_up_exps","ffn_down_exps"]

def is_ffn_tensor(name):
    return any(p in name for p in FFN_PATTERNS)

fail = 0
for n in FFN_NAMES:
    if not is_ffn_tensor(n):
        print(f"FAIL: expected FFN but got ATTN for '{n}'"); fail += 1
for n in ATTN_NAMES:
    if is_ffn_tensor(n):
        print(f"FAIL: expected ATTN but got FFN for '{n}'"); fail += 1

print(f"Routing test: {len(FFN_NAMES)+len(ATTN_NAMES)} tensors checked, {fail} failures")
sys.exit(fail)
