#!/usr/bin/env python3
"""
Compare expected tensor names from C++ with actual tensor names in GGUF file
"""
import gguf

print("=" * 80)
print("NLLB Tensor Name Verification")
print("=" * 80)

# Read GGUF file
reader = gguf.GGUFReader('nllb-600m.gguf')
actual_tensors = set(t.name for t in reader.tensors)

print(f"\nTotal tensors in GGUF: {len(actual_tensors)}")

# Expected tensor names from C++ code
expected_base = [
    "token_embd.weight",
    "position_embd.weight",
    "output.weight",
    "enc.output_norm.weight",
    "enc.output_norm.bias",
    "dec.output_norm.weight",
    "dec.output_norm.bias",
]

# Encoder layers (12 layers)
for i in range(12):
    expected_base.extend([
        f"enc.blk.{i}.attn_norm.weight",
        f"enc.blk.{i}.attn_norm.bias",
        f"enc.blk.{i}.attn_q.weight",
        f"enc.blk.{i}.attn_q.bias",
        f"enc.blk.{i}.attn_k.weight",
        f"enc.blk.{i}.attn_k.bias",
        f"enc.blk.{i}.attn_v.weight",
        f"enc.blk.{i}.attn_v.bias",
        f"enc.blk.{i}.attn_o.weight",
        f"enc.blk.{i}.attn_o.bias",
        f"enc.blk.{i}.ffn_norm.weight",
        f"enc.blk.{i}.ffn_norm.bias",
        f"enc.blk.{i}.ffn_up.weight",
        f"enc.blk.{i}.ffn_up.bias",
        f"enc.blk.{i}.ffn_down.weight",
        f"enc.blk.{i}.ffn_down.bias",
    ])

# Decoder layers (12 layers)
for i in range(12):
    expected_base.extend([
        f"dec.blk.{i}.attn_norm.weight",
        f"dec.blk.{i}.attn_norm.bias",
        f"dec.blk.{i}.attn_q.weight",
        f"dec.blk.{i}.attn_q.bias",
        f"dec.blk.{i}.attn_k.weight",
        f"dec.blk.{i}.attn_k.bias",
        f"dec.blk.{i}.attn_v.weight",
        f"dec.blk.{i}.attn_v.bias",
        f"dec.blk.{i}.attn_o.weight",
        f"dec.blk.{i}.attn_o.bias",
        f"dec.blk.{i}.cross_attn_norm.weight",
        f"dec.blk.{i}.cross_attn_norm.bias",
        f"dec.blk.{i}.cross_attn_q.weight",
        f"dec.blk.{i}.cross_attn_q.bias",
        f"dec.blk.{i}.cross_attn_k.weight",
        f"dec.blk.{i}.cross_attn_k.bias",
        f"dec.blk.{i}.cross_attn_v.weight",
        f"dec.blk.{i}.cross_attn_v.bias",
        f"dec.blk.{i}.cross_attn_o.weight",
        f"dec.blk.{i}.cross_attn_o.bias",
        f"dec.blk.{i}.ffn_norm.weight",
        f"dec.blk.{i}.ffn_norm.bias",
        f"dec.blk.{i}.ffn_up.weight",
        f"dec.blk.{i}.ffn_up.bias",
        f"dec.blk.{i}.ffn_down.weight",
        f"dec.blk.{i}.ffn_down.bias",
    ])

expected_tensors = set(expected_base)

print(f"Expected tensors from C++: {len(expected_tensors)}")

# Find missing and extra tensors
missing = expected_tensors - actual_tensors
extra = actual_tensors - expected_tensors

if missing:
    print(f"\n❌ MISSING TENSORS IN GGUF ({len(missing)}):")
    for name in sorted(missing)[:20]:  # Show first 20
        print(f"  - {name}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")

if extra:
    print(f"\n❓ EXTRA TENSORS IN GGUF ({len(extra)}):")
    for name in sorted(extra)[:20]:  # Show first 20
        print(f"  + {name}")
    if len(extra) > 20:
        print(f"  ... and {len(extra) - 20} more")

if not missing and not extra:
    print("\n✅ ALL TENSORS MATCH PERFECTLY!")
else:
    print(f"\n⚠️  Mismatch detected!")
    print(f"   Expected: {len(expected_tensors)}")
    print(f"   Actual: {len(actual_tensors)}")
    print(f"   Missing: {len(missing)}")
    print(f"   Extra: {len(extra)}")

print("\n" + "=" * 80)

