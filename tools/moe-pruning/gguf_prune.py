"""
gguf-prune: REAP-based expert pruning directly on a GGUF file.

Slices the expert dimension of the four stacked MoE weight tensors per layer:
    blk.{il}.ffn_up_exps      [n_embd, intermediate, n_experts]
    blk.{il}.ffn_down_exps    [intermediate, n_embd, n_experts]
    blk.{il}.ffn_gate_inp     [n_embd, n_experts]
    blk.{il}.ffn_exp_probs_b  [n_experts]  (score-correction bias, if present)

Quantized blocks (Q4_K, Q6_K, …) are preserved as raw bytes — slicing the
expert axis (last dim) is safe because each expert is independently quantised
in ggml, so dropping experts = dropping whole quantisation blocks.

Metadata patched:
    {arch}.expert_count  → keep_n
    (expert_used_count = top-k routing k, NOT touched)

Usage:
    # keep top 20% of experts (26/128) per MoE layer
    python gguf_prune.py \\
        --input  nemotron.gguf \\
        --stats  expert_stats.json \\
        --output nemotron-pruned.gguf \\
        --keep_ratio 0.20

    # or keep an absolute number
    python gguf_prune.py \\
        --input  nemotron.gguf \\
        --stats  expert_stats.json \\
        --output nemotron-pruned.gguf \\
        --keep_n 32
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


# ── Constants ─────────────────────────────────────────────────────────────────

# Base tensor names that carry the expert dimension (last axis in ggml layout).
# Some GGUFs append parameter tails like ".weight" / ".bias".
EXPERT_BASE_SUFFIXES = {
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_inp",
}


def is_expert_suffix(suffix: str) -> bool:
    """Return True if a tensor suffix is one of the MoE expert tensors to prune."""
    if suffix in ("ffn_exp_probs_b", "exp_probs_b", "exp_probs_b.bias"):
        return True
    return any(suffix == base or suffix.startswith(base + ".") for base in EXPERT_BASE_SUFFIXES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def layer_and_suffix(name: str) -> tuple[int, str] | tuple[None, None]:
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def pick_experts(layer_stats: dict, keep_n: int) -> list[int]:
    """
    Return sorted indices of the top `keep_n` experts by REAP score.
    Falls back to 'importance_score' (weighted frequency) if 'reap' absent.
    """
    if "reap" in layer_stats:
        scores = np.array(layer_stats["reap"], dtype=np.float64)
    elif "importance_score" in layer_stats:
        scores = np.array(layer_stats["importance_score"], dtype=np.float64)
    else:
        raise KeyError(
            "Layer stats has neither 'reap' nor 'importance_score'. "
            "Run expert-profile / nemotron_reap.py profile first."
        )
    return sorted(np.argsort(scores)[-keep_n:].tolist())


def slice_expert_axis(data: np.ndarray, keep: list[int]) -> np.ndarray:
    """
    Slice the expert axis of reader tensor data keeping only `keep` indices.

    GGUFReader reshapes tensors to NumPy with reversed ggml dims, so for MoE
    tensors where experts are the last ggml dim, expert is axis 0 in `data`.
    This also preserves quantized row-byte alignment (axis -1 is byte-packed
    rows for quantized tensors and must not be sliced for expert pruning).
    """
    return np.take(data, keep, axis=0)


def copy_field(writer: GGUFWriter, field, reader: GGUFReader) -> bool:
    """Copy a single metadata field to writer. Returns False if skipped."""
    key = field.name
    val_type = field.types[0]
    part = field.parts[-1]

    if val_type == GGUFValueType.STRING:
        # Preserve raw bytes: GGUF metadata can contain non-UTF8 strings.
        writer.add_key_value(key, bytes(part), GGUFValueType.STRING)
    elif val_type == GGUFValueType.UINT8:
        writer.add_uint8(key, int(part[0]))
    elif val_type == GGUFValueType.INT8:
        writer.add_int8(key, int(part[0]))
    elif val_type == GGUFValueType.UINT16:
        writer.add_uint16(key, int(part[0]))
    elif val_type == GGUFValueType.INT16:
        writer.add_int16(key, int(part[0]))
    elif val_type == GGUFValueType.UINT32:
        writer.add_uint32(key, int(part[0]))
    elif val_type == GGUFValueType.INT32:
        writer.add_int32(key, int(part[0]))
    elif val_type == GGUFValueType.FLOAT32:
        writer.add_float32(key, float(part[0]))
    elif val_type == GGUFValueType.UINT64:
        writer.add_uint64(key, int(part[0]))
    elif val_type == GGUFValueType.INT64:
        writer.add_int64(key, int(part[0]))
    elif val_type == GGUFValueType.FLOAT64:
        writer.add_float64(key, float(part[0]))
    elif val_type == GGUFValueType.BOOL:
        writer.add_bool(key, bool(part[0]))
    elif val_type == GGUFValueType.ARRAY:
        elem_type = field.types[1]
        if elem_type == GGUFValueType.STRING:
            # ReaderField.data stores indices of ARRAY payload items; for
            # STRING arrays this points at each string byte payload.
            vals = [bytes(field.parts[idx]) for idx in field.data]
            writer.add_key_value(key, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)
        else:
            # ReaderField.data stores part-indices, not payload values.
            vals = field.contents()
            if not isinstance(vals, list):
                print(f"  WARNING: skipping array field {key!r} (unexpected non-list contents)")
                return False
            writer.add_array(key, vals)
    else:
        print(f"  WARNING: skipping field {key!r} (unsupported type {val_type})")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="REAP expert pruning on a GGUF file")
    ap.add_argument("--input",      required=True,              help="Input .gguf path")
    ap.add_argument("--stats",      required=True,              help="expert_stats.json from expert-profile")
    ap.add_argument("--output",     required=True,              help="Output .gguf path")
    ap.add_argument("--keep_ratio", type=float, default=None,   help="Fraction to keep, e.g. 0.20")
    ap.add_argument("--keep_n",     type=int,   default=None,   help="Absolute count to keep, e.g. 32")
    ap.add_argument("--n_experts",  type=int,   default=128,    help="Experts per MoE layer in source model")
    args = ap.parse_args()

    if args.keep_ratio is None and args.keep_n is None:
        ap.error("Provide --keep_ratio or --keep_n")
    if args.keep_ratio is not None and args.keep_n is not None:
        ap.error("Provide --keep_ratio OR --keep_n, not both")

    keep_n = args.keep_n if args.keep_n is not None else max(1, int(args.n_experts * args.keep_ratio))
    print(f"[gguf-prune] keeping {keep_n}/{args.n_experts} experts per MoE layer")

    # ── Load stats ─────────────────────────────────────────────────────────────
    with open(args.stats) as f:
        stats = {int(k): v for k, v in json.load(f).items()}
    print(f"[gguf-prune] stats loaded for {len(stats)} MoE layers")

    # ── Open source GGUF ───────────────────────────────────────────────────────
    print(f"[gguf-prune] reading  {args.input}")
    reader = GGUFReader(args.input, mode="r")

    arch_field = reader.get_field("general.architecture")
    arch = str(bytes(arch_field.parts[-1]), "utf-8") if arch_field else "nemotron_h_moe"
    print(f"[gguf-prune] arch     {arch}")

    expert_count_key = f"{arch}.expert_count"

    # ── Compute kept indices per layer ─────────────────────────────────────────
    kept: dict[int, list[int]] = {}
    for tensor in reader.tensors:
        il, suffix = layer_and_suffix(tensor.name)
        if il is None or suffix is None or not is_expert_suffix(suffix):
            continue
        if il in kept:
            continue  # already computed for this layer
        if il not in stats:
            print(f"  Layer {il:3d}: no stats — keeping ALL {args.n_experts} experts")
            kept[il] = list(range(args.n_experts))
        else:
            kept[il] = pick_experts(stats[il], keep_n)
            never = stats[il].get("never_activated", "?")
            crit  = "reap" if "reap" in stats[il] else "importance_score"
            print(f"  Layer {il:3d}: keep {kept[il][:4]}…  never_activated={never}  criterion={crit}")

    # ── Build output GGUF ──────────────────────────────────────────────────────
    print(f"\n[gguf-prune] writing  {args.output}")
    writer = GGUFWriter(args.output, arch=arch)

    # --- metadata: copy all fields, replace expert_count ---
    for field in reader.fields.values():
        # Reader exposes synthetic header fields (GGUF.*) that are not KV
        # metadata and must not be copied back as normal keys.
        if field.name.startswith("GGUF."):
            continue
        # Writer already sets general.architecture from ctor; avoid duplicate warning.
        if field.name in (expert_count_key, "general.architecture"):
            continue  # replaced below
        copy_field(writer, field, reader)

    writer.add_expert_count(keep_n)
    print(f"[gguf-prune] patched  {expert_count_key} → {keep_n}")

    # --- tensors ---
    n_pruned = 0
    for tensor in reader.tensors:
        il, suffix = layer_and_suffix(tensor.name)
        is_expert = il is not None and suffix is not None and is_expert_suffix(suffix)

        if is_expert:
            assert il is not None
            k = kept[il]
            data = slice_expert_axis(tensor.data, k)
            writer.add_tensor(
                tensor.name,
                data,
                raw_dtype=tensor.tensor_type,
            )
            n_pruned += 1
        else:
            writer.add_tensor(
                tensor.name,
                tensor.data,
                raw_dtype=tensor.tensor_type,
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    out = Path(args.output)
    size_gb = out.stat().st_size / 1024**3
    print(f"\n[gguf-prune] done")
    print(f"  Expert tensors sliced : {n_pruned}")
    print(f"  MoE layers pruned     : {len(kept)}")
    print(f"  Experts per layer     : {keep_n}/{args.n_experts}")
    print(f"  Output size           : {size_gb:.2f} GB  →  {out}")


if __name__ == "__main__":
    main()
