#!/usr/bin/env python3
"""
tile640_quantize_v3.py — Tessera quantization using the T640 physical layout.

Reads BF16 weights from a HuggingFace safetensors model and writes a Tessera GGUF.

Changes from v2:
  - Adds `--imatrix <path>` for activation-aware outlier ranking. The imatrix
    is a numpy .npz file (or llama-imatrix .gguf/.dat file) that contains
    per-position mean(|x|) statistics from a calibration forward pass.
  - Outlier selection score changes from `|w|` to `|w × x̂|` when an imatrix
    is provided. This is the AWQ insight: positions with large activation
    magnitude matter more for the matmul output, even if their weight is small.
  - Fixes the 3D-expert `.weight.weight` naming bug: 3D expert names like
    `blk.0.ffn_gate_up_exps.weight` are already the weight name (no extra
    `.weight` suffix appended).
  - Fixes the pass-through F16 name mapping: all F16 pass-through tensors
    get the HF->GGUF name mapping applied, not just the quantized ones.
  - All tensor names are kept ≤64 chars to satisfy GGUF format constraint.

The imatrix file format is a numpy .npz with two arrays per weight:
  - `<key>.in_sum2`: float32[shape[0]] — sum of x² per input position
  - `<key>.counts`:  int64[1]          — number of samples seen
The key is the weight's name in the calibration model (HF-style for the
prism calibrator, GGUF-style for llama-imatrix). We provide a name mapper.

For pruning outliers the imatrix isn't strictly needed — pure `|w|` is the
fallback. Imatrix is a quality refinement, not a correctness requirement.
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# CHAMP-Q channel permutation. Imported lazily in main() so the helper
# module path can be customised via TESSERA_TOOLS / sys.path; the
# functions themselves have no side effects on import.
try:
    from tools.tessera.champq_permute import (  # type: ignore
        CHAMPQPolicy,
        apply_champq_permutation,
        compute_champq_permutation,
        decode_q_to_weight,
        invert_champq_permutation,
    )
    _CHAMPQ_AVAILABLE = True
except ImportError:
    try:
        # tools/ is on sys.path when quantize_v3 is run as a script.
        from tessera.champq_permute import (  # type: ignore
            CHAMPQPolicy,
            apply_champq_permutation,
            compute_champq_permutation,
            decode_q_to_weight,
            invert_champq_permutation,
        )
        _CHAMPQ_AVAILABLE = True
    except ImportError:
        _CHAMPQ_AVAILABLE = False

# PE-QAT: parameter-efficient quantization-aware training policy. The
# trainer lives in tools.tessera.pe_qat and is imported as a normal
# module import (it has no platform-specific deps). The policy is
# consumed in the main quantize loop via apply_pe_qat_to_weight().
try:
    from tools.tessera.pe_qat import (  # type: ignore
        PE_QAT_POLICY_SCHEMA,
        _pe_qat_policy_for,
        apply_pe_qat_to_weight,
    )
    _PE_QAT_AVAILABLE = True
except ImportError:
    try:
        # tools/ is on sys.path when quantize_v3 is run as a script.
        from tessera.pe_qat import (  # type: ignore
            PE_QAT_POLICY_SCHEMA,
            _pe_qat_policy_for,
            apply_pe_qat_to_weight,
        )
        _PE_QAT_AVAILABLE = True
    except ImportError:
        _PE_QAT_AVAILABLE = False

ACCELERATE_BACKEND = None
ANE_BACKEND = None
if sys.platform == "darwin" and os.environ.get("TESSERA_ACCELERATE", "1") != "0":
    try:
        tessera_tools = Path(
            os.environ.get(
                "TESSERA_TOOLS",
                "/Users/user/Developer/GitHub/llama.cpp/tools/tessera",
            )
        )
        if str(tessera_tools) not in sys.path:
            sys.path.insert(0, str(tessera_tools))
        from apple_accelerate import AccelerateBackend

        ACCELERATE_BACKEND = AccelerateBackend()
    except Exception as exc:
        print(
            f"WARN: Apple Accelerate backend unavailable: {exc}",
            file=sys.stderr,
        )

ane_model = os.environ.get("TESSERA_ANE_MODEL")
if sys.platform == "darwin" and ane_model:
    try:
        tessera_tools = Path(
            os.environ.get(
                "TESSERA_TOOLS",
                "/Users/user/Developer/GitHub/llama.cpp/tools/tessera",
            )
        )
        if str(tessera_tools) not in sys.path:
            sys.path.insert(0, str(tessera_tools))
        from apple_ane_quantizer import ANEQuantizerBackend

        ANE_BACKEND = ANEQuantizerBackend(Path(ane_model))
        print(
            "Using Tessera Core ML quantizer asset "
            f"(exact canonical path, ANE proposals available): {ane_model}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"WARN: ANE quantizer backend unavailable: {exc}", file=sys.stderr)

# ── Optional MLX backend (much faster for matmul ops) ────────────────────
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("WARN: mlx.core not available, falling back to numpy (slow)", file=sys.stderr)

# ── Tile640 format constants (must match ggml C++ + bonsai_ternary.rs) ──
TILE640_PAGE_SIZE = 640
TILE640_LANE_SIZE = 20
TILE640_LANES_PER_PAGE = 32
TILE640_WORDS_PER_PAGE = 32
LARGE_MATRIX_ROW_BALANCED_THRESHOLD = int(
    os.environ.get("TESSERA_ROW_BALANCED_THRESHOLD", "250000000")
)


def import_gguf(gguf_py: Optional[str]) -> Tuple[Any, Any, Any, Any]:
    """Import llama.cpp's GGUF API, optionally from an explicit gguf-py tree."""
    if gguf_py:
        gguf_path = str(Path(gguf_py).expanduser().resolve())
        if gguf_path not in sys.path:
            sys.path.insert(0, gguf_path)
    try:
        import gguf
        from gguf import GGUFReader, GGUFValueType, GGUFWriter
    except ImportError as exc:
        location = f" at {gguf_py}" if gguf_py else ""
        raise RuntimeError(
            f"gguf Python library not available{location}; pass --gguf-py "
            "pointing to llama.cpp/gguf-py"
        ) from exc
    return gguf, GGUFReader, GGUFValueType, GGUFWriter


def normalize_hf_tensor_name(name: str, base_block_count: int) -> Optional[str]:
    """Normalize Qwen3.6 checkpoint names for llama.cpp's official tensor map."""
    if name.startswith((
        "model.visual.",
        "visual.",
        "model.embed_vision.",
        "model.embed_audio.",
        "model.vision_embedder.",
        "model.audio_embedder.",
        "vision_tower.",
        "model.vision_tower.",
        "audio_tower.",
        "model.audio_tower.",
        "mm_projector.",
        "model.mm_projector.",
    )):
        return None
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model."):]
    elif name.startswith("language_model.model."):
        name = "model." + name[len("language_model.model."):]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model."):]
    router_scale = re.match(r"model\.layers\.(\d+)\.router\.scale$", name)
    if router_scale:
        return f"model.layers.{router_scale.group(1)}.router.proj.scale"
    expert_scale = re.match(
        r"model\.layers\.(\d+)\.router\.per_expert_scale$", name
    )
    if expert_scale:
        return f"blk.{expert_scale.group(1)}.ffn_down_exps.scale"

    mtp_name = name
    if mtp_name.startswith("model.mtp."):
        mtp_name = mtp_name[len("model."):]
    mtp_remapper = {
        "fc": "eh_proj",
        "pre_fc_norm_embedding": "enorm",
        "pre_fc_norm_hidden": "hnorm",
        "norm": "shared_head.norm",
    }
    top_level_mtp = re.match(r"mtp\.([^.]+)\.(weight|bias)$", mtp_name)
    if top_level_mtp and top_level_mtp.group(1) in mtp_remapper:
        return (
            f"model.layers.{base_block_count}."
            f"{mtp_remapper[top_level_mtp.group(1)]}."
            f"{top_level_mtp.group(2)}"
        )
    match = re.match(r"mtp\.layers\.(\d+)\.(.+)$", mtp_name)
    if match:
        return f"model.layers.{base_block_count + int(match.group(1))}.{match.group(2)}"
    if mtp_name.endswith(".dt_bias"):
        return mtp_name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
    return name


def mapped_tensor_parts(mapped_name: str) -> Tuple[int, Optional[str]]:
    """Return (block index, canonical tensor stem) for an official GGUF name."""
    match = re.match(r"blk\.(\d+)\.(.+?)(?:\.(?:weight|bias))?$", mapped_name)
    if not match:
        return -1, None
    return int(match.group(1)), match.group(2)


def component_name(mapped_weight_name: str, suffix: str) -> str:
    if not mapped_weight_name.endswith(".weight"):
        raise ValueError(f"Tile640 tensor is not a weight: {mapped_weight_name}")
    return mapped_weight_name[:-len("weight")] + suffix


def evolved_component_name(name: str, component: str) -> str:
    """Name a generic Tile640 component using llama.cpp tensor suffix rules."""
    tail = name.rsplit(".", 1)[-1]
    if tail in {"weight", "bias"}:
        return f"{name}_{component}"
    return f"{name}.weight_{component}"


def evolved_matrix_view(arr: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
    """Return the 2-D quantization view and its GGML-order matrix shape.

    GGUFReader exposes tensor data in NumPy order (the reverse of GGML ne).
    For convolution kernels, keep the output-channel axis as rows and flatten
    every input/kernel axis into K. The logical GGUF shape remains separate so
    the loader can reshape the decoded matrix for convolution.
    """
    if arr.ndim <= 2:
        row_width = arr.shape[-1]
        rows = arr.reshape(-1, row_width)
    else:
        rows = arr.reshape(arr.shape[0], -1)
        row_width = rows.shape[1]
    return rows, row_width, [int(row_width), int(rows.shape[0])]


def copy_gguf_metadata(
    writer: Any,
    metadata_path: str,
    GGUFReader: Any,
    GGUFValueType: Any,
    expected_architecture: str,
) -> int:
    """Copy model/tokenizer metadata from a loadable GGUF into the Tessera GGUF."""
    reader = GGUFReader(metadata_path, "r")
    architecture = reader.get_field("general.architecture")
    if architecture is None:
        raise ValueError(f"{metadata_path}: missing general.architecture")
    architecture_name = architecture.contents()
    if architecture_name != expected_architecture:
        raise ValueError(
            f"{metadata_path}: expected {expected_architecture} metadata, found {architecture_name!r}"
        )

    excluded = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
        "general.file_type",
        "split.no",
        "split.count",
        "split.tensors.count",
    }
    copied = 0
    for key, field in reader.fields.items():
        if key in excluded or key.startswith("tessera.") or key.startswith("tile640."):
            continue
        value_type = field.types[0]
        value = field.contents()
        if value_type == GGUFValueType.ARRAY:
            if len(field.types) < 2:
                raise ValueError(f"{metadata_path}: array metadata {key!r} has no subtype")
            writer.add_key_value(key, value, value_type, field.types[-1])
        else:
            writer.add_key_value(key, value, value_type)
        copied += 1

    required = {
        "general.type",
        f"{expected_architecture}.block_count",
        f"{expected_architecture}.embedding_length",
        "tokenizer.ggml.model",
        "tokenizer.ggml.tokens",
    }
    missing = sorted(key for key in required if key not in reader.fields)
    if missing:
        raise ValueError(
            f"{metadata_path}: incomplete model metadata; missing {', '.join(missing)}"
        )
    return copied


def apply_gemma4_metadata_overrides(
    writer,
    metadata_reader,
    architecture_name: str,
    sliding_window: int,
    GGUFValueType,
) -> int:
    """Override gemma4 sliding_window metadata if it disagrees with the
    canonical config. Google shipped a gemma 3 QAT gguf with
    `sliding_window=1024` when the config says 512; the same defect was
    reproduced in the gemma 4 12B QAT gguf. Forcing the correct value at
    export time means downstream llama.cpp can load the model without a
    manual `--override-kv` argument.

    Returns the number of overrides applied (0 if architecture is not
    gemma4-family or the source already matches).
    """
    if architecture_name not in ("gemma4", "gemma4-assistant"):
        return 0
    overrides = 0
    swa_keys = [
        "gemma4.attention.sliding_window",
    ]
    # MTP blocks for gemma 4 12B share the same gemma4 attention metadata
    # layout but use a separate key prefix.
    for swa_key in swa_keys:
        field = metadata_reader.get_field(swa_key)
        if field is None:
            continue
        try:
            current = int(field.contents())
        except Exception:
            current = None
        if current is None or current == sliding_window:
            continue
        writer.add_uint32(swa_key, sliding_window)
        overrides += 1
        print(
            f"  gemma4 metadata override: {swa_key} {current} -> {sliding_window}",
            file=sys.stderr,
        )
    # The MTP prefix uses the gemma4-assistant arch_name; replicate the same
    # override there if a field exists.
    mtp_swa_key = "mtp.gemma4.attention.sliding_window"
    mtp_field = metadata_reader.get_field(mtp_swa_key)
    if mtp_field is not None:
        try:
            mtp_current = int(mtp_field.contents())
        except Exception:
            mtp_current = None
        if mtp_current is not None and mtp_current != sliding_window:
            writer.add_uint32(mtp_swa_key, sliding_window)
            overrides += 1
            print(
                f"  gemma4 metadata override: {mtp_swa_key} "
                f"{mtp_current} -> {sliding_window}",
                file=sys.stderr,
            )
    return overrides


def add_tessera_metadata(
    writer: Any,
    calibrated: bool,
    unified: bool,
    unsloth_prior: bool = False,
    global_residual_budget: bool = False,
    epoch_receipt: Optional[Dict[str, Any]] = None,
    source_receipt: Optional[Dict[str, Any]] = None,
    imatrix_paths: Optional[List[str]] = None,
    imatrix_merge_policy: Optional[str] = None,
) -> None:
    profile = "TSQ-T640-AWQ-SR" if calibrated else "TSQ-T640-SR"
    if unified:
        profile += "-U"
    features = [
        "ternary-core",
        "interleaved-fused-dequant",
        "per-page-bf16-scale",
        "per-lane-int8-scale",
        "sparse-f16-residual",
        "reconstruction-error-residual-repair",
        "exact-sensitive-tensors",
        "complete-learned-tensor-coverage",
    ]
    if calibrated:
        features.extend(["imatrix", "awq", "activation-weighted-residual-repair"])
    if global_residual_budget:
        features.append("global-residual-budget")
    if unsloth_prior:
        features.append("unsloth-dynamic-quant-prior")
    if unified:
        features.append("unified-multimodal")

    writer.add_string("tessera.name", "Tessera Quantization")
    writer.add_uint32("tessera.version", 1)
    writer.add_string("tessera.profile", profile)
    writer.add_array("tessera.features", features)
    writer.add_string("tessera.core.type", "balanced-ternary")
    writer.add_uint32("tessera.core.levels", 3)
    writer.add_string("tessera.layout", "T640")
    writer.add_uint32("tessera.layout.version", 1)
    writer.add_uint32("tessera.layout.page_size", TILE640_PAGE_SIZE)
    writer.add_uint32("tessera.layout.lane_size", TILE640_LANE_SIZE)
    writer.add_uint32("tessera.layout.lanes_per_page", TILE640_LANES_PER_PAGE)
    writer.add_uint32("tessera.layout.words_per_page", TILE640_WORDS_PER_PAGE)
    writer.add_string("tessera.scale.page_type", "bf16")
    writer.add_string("tessera.scale.lane_type", "int8")
    writer.add_string("tessera.residual.type", "row-sparse")
    writer.add_string("tessera.residual.value_type", "f16")
    writer.add_bool("tessera.sensitive.exact", True)
    writer.add_bool("tessera.calibration.imatrix", calibrated)
    writer.add_bool("tessera.calibration.awq", calibrated)
    writer.add_bool("tessera.calibration.unsloth_prior", unsloth_prior)
    if imatrix_paths:
        writer.add_array("tessera.calibration.imatrix_paths", imatrix_paths)
        writer.add_string(
            "tessera.calibration.imatrix_merge_policy",
            imatrix_merge_policy or "single",
        )
        writer.add_uint32(
            "tessera.calibration.imatrix_source_count",
            len(imatrix_paths),
        )
    writer.add_string("tessera.coverage", "all-learned-tensors")
    writer.add_bool("tessera.passthrough", False)
    writer.add_bool("tessera.unified", unified)
    if epoch_receipt is not None:
        if epoch_receipt.get("schema") != "llama.tessera.epoch.v1":
            raise ValueError("Tessera epoch receipt has an unsupported schema")
        writer.add_uint32("tessera.dataset.epoch", int(epoch_receipt["epoch"]))
        writer.add_string(
            "tessera.dataset.model_fingerprint",
            str(epoch_receipt["model_fingerprint"]),
        )
        writer.add_string(
            "tessera.dataset.evidence_digest",
            str(epoch_receipt["evidence_digest"]),
        )
        writer.add_uint64(
            "tessera.dataset.observer_calibration_tokens",
            int(epoch_receipt.get("observer_calibration_tokens", 0)),
        )
        writer.add_uint64(
            "tessera.dataset.acceptance_observations",
            int(epoch_receipt.get("acceptance_observations", 0)),
        )
    if source_receipt is not None:
        if source_receipt.get("schema") != "llama.tessera.source-epoch.v1":
            raise ValueError("Tessera source receipt has an unsupported schema")
        writer.add_uint32("tessera.source.epoch", int(source_receipt["epoch"]))
        writer.add_string("tessera.source.digest", str(source_receipt["source_digest"]))
        writer.add_string("tessera.source.artifact_digest", str(source_receipt["artifact_digest"]))
        writer.add_uint64("tessera.source.logical_bytes", int(source_receipt["logical_bytes"]))
        writer.add_uint64("tessera.source.tensor_count", int(source_receipt["tensor_count"]))
        lineage = source_receipt.get("lineage") or {}
        if lineage.get("parent_source_digest"):
            writer.add_string(
                "tessera.source.parent_digest",
                str(lineage["parent_source_digest"]),
            )
        if lineage.get("training_corpus_epoch") is not None:
            writer.add_uint32(
                "tessera.source.training_corpus_epoch",
                int(lineage["training_corpus_epoch"]),
            )
        if lineage.get("training_corpus_digest"):
            writer.add_string(
                "tessera.source.training_corpus_digest",
                str(lineage["training_corpus_digest"]),
            )
        if lineage.get("telemetry_epoch") is not None:
            writer.add_uint32(
                "tessera.source.telemetry_epoch",
                int(lineage["telemetry_epoch"]),
            )


def copy_embedded_mmproj_metadata(
    writer: Any,
    mmproj_path: str,
    GGUFReader: Any,
    GGUFValueType: Any,
) -> Tuple[Any, int]:
    """Merge projector metadata into a text-model GGUF without replacing its identity."""
    reader = GGUFReader(mmproj_path, "r")
    excluded = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
        "general.type",
        "general.name",
        "general.basename",
        "general.size_label",
        "split.no",
        "split.count",
        "split.tensors.count",
    }
    copied = 0
    for key, field in reader.fields.items():
        if key in excluded or key.startswith("general."):
            continue
        value_type = field.types[0]
        value = field.contents()
        if value_type == GGUFValueType.ARRAY:
            writer.add_key_value(key, value, value_type, field.types[-1])
        else:
            writer.add_key_value(key, value, value_type)
        copied += 1
    return reader, copied


# ═══════════════════════════════════════════════════════════════════════
# Name mapping: HuggingFace -> GGUF (matches llama.cpp's HF->GGUF logic)
# ═══════════════════════════════════════════════════════════════════════

def hf_to_gguf_layer(wname: str) -> Tuple[int, Optional[str]]:
    """
    Map an HF safetensors tensor name to (layer_idx, gguf_short_name).
    Returns (-1, None) for non-layer tensors.
    """
    # Strip "model.language_model." or "model." prefix
    s = wname
    for prefix in ["model.language_model.", "language_model.model.", "model.", "language_model."]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    # Find layer number
    m = re.match(r"layers\.(\d+)\.(.*)$", s)
    if not m:
        # Could be a top-level tensor (tok_embd, output, norm, etc.)
        return (-1, None)
    layer_idx = int(m.group(1))
    rest = m.group(2)
    return (layer_idx, _hf_tail_to_gguf(rest))


def _hf_tail_to_gguf(tail: str) -> Optional[str]:
    """
    Convert the part of an HF name after "layers.N." to a GGUF short name.
    Examples:
      self_attn.q_proj.weight        -> attn_q
      self_attn.q_norm.weight        -> attn_q_norm
      linear_attn.in_proj_qkv.weight -> wqkv
      linear_attn.in_proj_z.weight   -> wqkv_gate (for Qwen3.5/3.6)
      linear_attn.out_proj.weight    -> ssm_out
      mlp.gate_proj.weight           -> ffn_gate_exps  (or shared if shared_expert)
      mlp.up_proj.weight             -> ffn_up_exps
      mlp.down_proj.weight           -> ffn_down_exps
      mlp.shared_expert.gate_proj    -> ffn_gate_shexp
      mlp.shared_expert.up_proj      -> ffn_up_shexp
      mlp.shared_expert.down_proj    -> ffn_down_shexp
      mlp.gate.weight                -> ffn_gate_inp  (MoE router)
      mlp.shared_expert.gate         -> ffn_gate_inp_shexp (shared expert gate)
    """
    parts = tail.split(".")
    if len(parts) == 2 and parts[1] == "weight":
        # 2D weight, e.g. self_attn.q_proj.weight, mlp.gate_proj.weight
        kind, proj = parts
    elif len(parts) == 2:
        # 2D weight without ".weight" suffix, e.g. shared_expert_gate
        kind, proj = parts
    elif len(parts) == 3 and parts[0] == "mlp" and parts[1] == "experts":
        # 3D MoE expert, e.g. mlp.experts.gate_up_proj
        kind, _, proj = parts
    elif len(parts) == 3 and parts[0] == "mlp" and parts[1] == "shared_expert":
        # Shared expert, e.g. mlp.shared_expert.gate_proj
        kind, _, proj = parts
    elif len(parts) == 3 and parts[0] == "mlp" and parts[2] == "weight":
        # e.g. mlp.shared_expert_gate.weight (Qwen3.5 shared expert gate)
        kind, proj, _ = parts
    elif len(parts) == 3 and parts[1] == "weight":
        # e.g. linear_attn.in_proj_qkv.weight
        kind, _, proj = parts
    else:
        return None

    # self_attn.*
    if kind == "self_attn":
        return {
            "q_proj": "attn_q",
            "k_proj": "attn_k",
            "v_proj": "attn_v",
            "o_proj": "attn_out",
            "q_norm": "attn_q_norm",
            "k_norm": "attn_k_norm",
        }.get(proj, None)

    # linear_attn.*  (gated delta net in Qwen3.5/3.6)
    if kind == "linear_attn":
        return {
            "in_proj_qkv": "wqkv",
            "in_proj_z":   "wqkv_gate",
            "in_proj_a":   "ssm_a",
            "in_proj_b":   "ssm_b",
            "in_proj_q":   None,
            "in_proj_k":   None,
            "in_proj_v":   None,
            "out_proj":    "ssm_out",
            "conv1d":      "ssm_conv1d",
            "dt_bias":     "ssm_dt",
            "A_log":       "ssm_a",  # in some HF versions
            "norm":        "ssm_norm",
            "beta":        "ssm_beta",
            "alpha":       "ssm_alpha",
        }.get(proj, None)

    # mlp.* — expert and shared expert
    if kind == "mlp":
        if proj == "gate" and len(parts) == 2:
            return "ffn_gate_inp"  # MoE router
        if proj == "shared_expert_gate" and len(parts) == 2:
            return "ffn_gate_inp_shexp"
        if proj in ("gate_proj", "up_proj", "down_proj") and len(parts) == 2:
            # Top-level mlp.gate_proj (Qwen3.5 dense FFN, no MoE)
            return f"ffn_{proj[:4] if proj != 'down_proj' else 'down'}"
        if len(parts) == 3 and parts[1] == "experts":
            return {
                "gate_proj":   "ffn_gate_exps",
                "up_proj":     "ffn_up_exps",
                "down_proj":   "ffn_down_exps",
                "gate_up_proj": "ffn_gate_up_exps",  # merged
            }.get(proj, None)
        if len(parts) == 3 and parts[1] == "shared_expert":
            return {
                "gate_proj": "ffn_gate_shexp",
                "up_proj":   "ffn_up_shexp",
                "down_proj": "ffn_down_shexp",
            }.get(proj, None)
        if proj == "shared_expert_gate":
            return "ffn_gate_inp_shexp"

    return None


# Tensor classes that should be quantized to Tile640
QUANT_2D_SHORT = {
    "attn_q", "attn_k", "attn_v", "attn_out",
    "attn_qkv", "attn_gate", "ssm_out",
    "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_inp",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "token_embd",
    "per_layer_token_embd", "per_layer_model_proj",
    "inp_gate", "proj",
}

# 3D MoE experts (gate, up, down are quantized; gate_up_proj is the merged form)
QUANT_3D_SHORT = {
    "ffn_down_exps",
    "ffn_gate_up_exps",  # merged gate + up
    "ffn_gate_exps",     # unmerged, 3D
    "ffn_up_exps",       # unmerged, 3D
}

# Tensor classes that stay F16 (sensitive or tiny)
F16_KEEP_SHORT = {
    "ffn_gate_inp",        # MoE router
    "ffn_gate_inp_shexp",  # shared expert gate
    "attn_q_norm", "attn_k_norm", "ssm_norm",  # norms
    "ssm_a", "ssm_dt",     # SSM state
}

# Top-level (non-layer) tensors that need special handling
TOP_LEVEL_F16 = {
    "model.language_model.embed_tokens.weight": "token_embd",
    "model.language_model.norm.weight":         "output_norm",
    "lm_head.weight":                            "output",
    "output.weight":                             "output",
    "model.norm.weight":                         "output_norm",
}


# ═══════════════════════════════════════════════════════════════════════
# Imatrix loading
# ═══════════════════════════════════════════════════════════════════════

def load_imatrix(path: str) -> Dict[str, np.ndarray]:
    """
    Load an imatrix file. Supports:
      - .npz (prism oqe format): keys like `<name>.in_sum2` and `<name>.counts`
      - .gguf (llama-imatrix format with --output-format gguf)
      - .dat  (llama-imatrix legacy format)
    Returns: dict mapping weight name (as in the imatrix) to float32[in_dim] RMS.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"imatrix not found: {path}")
    if path.endswith(".npz"):
        return _load_imatrix_npz(path)
    elif path.endswith(".gguf"):
        return _load_imatrix_gguf(path)
    else:
        raise ValueError(f"unsupported imatrix format: {path}")


def _load_imatrix_npz(path: str) -> Dict[str, np.ndarray]:
    """Load prism-style oqe imatrix (.npz)."""
    raw = np.load(path)
    # Group by base name (strip .in_sum2 / .counts suffix)
    by_weight: Dict[str, Dict[str, np.ndarray]] = {}
    for key in raw.files:
        if key.endswith(".in_sum2"):
            base = key[:-len(".in_sum2")]
            by_weight.setdefault(base, {})["sum2"] = raw[key]
        elif key.endswith(".counts"):
            base = key[:-len(".counts")]
            by_weight.setdefault(base, {})["counts"] = raw[key]

    out: Dict[str, np.ndarray] = {}
    for base, d in by_weight.items():
        if "sum2" not in d or "counts" not in d:
            continue
        s2 = d["sum2"].astype(np.float32)
        cn = d["counts"].astype(np.float32)
        if cn[0] <= 0:
            continue
        # RMS: sqrt(mean(x²)) — proxy for mean(|x|) for rank-ordering
        rms = np.sqrt(s2 / cn[0])
        out[base] = rms
    print(f"  imatrix: {len(out)} weight entries loaded from {path}", file=sys.stderr)
    return out


def _load_imatrix_gguf(path: str) -> Dict[str, np.ndarray]:
    """Load llama-imatrix output (GGUF format)."""
    try:
        from gguf import GGUFReader
    except ImportError:
        raise RuntimeError("gguf python lib not available; install via llama.cpp/gguf-py")
    r = GGUFReader(path)
    tensors: Dict[str, np.ndarray] = {}
    for t in r.tensors:
        if t.tensor_type.name == "F32":
            tensors[t.name] = np.array(t.data, dtype=np.float32)
    out: Dict[str, np.ndarray] = {}
    for name, sum2 in tensors.items():
        if not name.endswith(".in_sum2"):
            continue
        base = name[:-len(".in_sum2")]
        counts = tensors.get(f"{base}.counts")
        if counts is None:
            continue
        sum2 = sum2.reshape(-1, sum2.shape[-1] if sum2.ndim > 1 else sum2.size)
        counts = counts.reshape(-1)
        valid = counts > 0
        if not np.any(valid):
            continue
        weighted_sum = sum2[valid].sum(axis=0)
        total_count = counts[valid].sum()
        pooled = np.sqrt(weighted_sum / total_count).astype(np.float32)
        if counts.size == 1:
            out[base] = pooled
            continue

        # llama-imatrix records one activation observer per expert for
        # ggml_mul_mat_id. Preserve that conditional information instead of
        # collapsing all routed experts to one vector. Experts not selected by
        # the calibration corpus receive the layer-pooled observer, matching
        # the intent of Unsloth/llmcompressor's calibrate-all-experts behavior
        # without fabricating zero-importance channels.
        expert_rms = np.empty_like(sum2, dtype=np.float32)
        expert_rms[valid] = np.sqrt(
            sum2[valid] / counts[valid, np.newaxis]
        ).astype(np.float32)
        expert_rms[~valid] = pooled
        out[base] = expert_rms
    print(f"  imatrix (gguf): {len(out)} weight entries", file=sys.stderr)
    return out


def merge_imatrix_geomean(
    primary: Dict[str, np.ndarray],
    *others: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Geometric-mean merge of multiple imatrices.

    For each tensor key present in `primary`, the merged importance is
    ``(primary * other_1 * other_2 * ...)^(1/N)`` where N is the number
    of inputs that actually measured the key. Sources that don't have a
    given key are *skipped* (they didn't measure it, so they have no
    opinion), which means a missing source does not artificially deflate
    the merged value.

    Geometric mean is the standard "balanced" merge for activation-aware
    quantization: it preserves the relative ordering of channels, doesn't
    over-emphasise the most pessimistic calibration, and stays numerically
    stable (no overflow) since importance is positive.

    Tensors that exist in `primary` but not in any merge source are kept
    as-is (geometric mean of one value = that value). Tensors that exist
    only in merge sources are NOT introduced into the result (we can't
    synthesize importance we don't have).

    Args:
        primary:  the main imatrix dict (tessera will use this as-is for
            keys not present in any merge source).
        *others:  one or more additional imatrix dicts to combine.

    Returns:
        New dict with the same keys as `primary`, with per-tensor
        geometric-mean importance values.
    """
    if not others:
        return primary
    merged: Dict[str, np.ndarray] = {}
    for key, base_val in primary.items():
        # Geometric mean: product^(1/N) over available sources.
        # In log-space: mean of log-importance, then exp.
        # We use the multiplicative form because all sources store
        # sqrt(E[x^2]) which is strictly non-negative.
        prod = np.asarray(base_val, dtype=np.float64)
        n_sources = 1
        for other in others:
            v = other.get(key)
            if v is None:
                continue
            arr = np.asarray(v, dtype=np.float64)
            if arr.shape != base_val.shape:
                # Shape mismatch (e.g. MoE expert fan-in differs) — skip this
                # source for this key rather than broadcasting garbage.
                print(
                    f"  imatrix-merge: skipping {key} from a merge source "
                    f"(shape {arr.shape} != primary {base_val.shape})",
                    file=sys.stderr,
                )
                continue
            prod = prod * arr
            n_sources += 1
        # (a*b*...)^(1/N) — use exp(mean(log)) for numerical stability
        log_prod = np.log(np.maximum(prod, 1e-30))
        merged[key] = np.exp(log_prod / n_sources).astype(np.float32)
    return merged


def lookup_acts(wname_hf: str, imatrix: Dict[str, np.ndarray],
                gguf_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Look up the imatrix entry for a given weight name. Tries HF and GGUF variants.

    Args:
        wname_hf:    HF-style name from the safetensors (e.g.
                     `model.language_model.layers.0.self_attn.q_proj.weight`)
        imatrix:     dict of {key: act_scales} from the imatrix file
        gguf_name:   optional GGUF-style name (e.g. `blk.0.attn_q.weight` or
                     `blk.0.attn_q`). If provided, tried as a fallback.

    Returns the per-position activation magnitude (RMS), or None if no match.
    """
    candidates = [wname_hf]
    # Strip "model." prefixes one at a time
    for prefix in ["model.language_model.", "language_model.model.",
                   "model.", "language_model."]:
        if wname_hf.startswith(prefix):
            candidates.append(wname_hf[len(prefix):])

    # Try stripping ".weight" suffix
    if wname_hf.endswith(".weight"):
        for c in list(candidates):
            candidates.append(c[:-len(".weight")])

    if gguf_name is not None:
        candidates.append(gguf_name)
        if gguf_name.endswith(".weight"):
            candidates.append(gguf_name[:-len(".weight")])

    for c in candidates:
        if c in imatrix:
            return imatrix[c]

    # Qwen3.6 stores expert gate+up weights fused in the HF checkpoint while
    # the canonical calibration GGUF executes and observes the two projections
    # separately. They consume the same routed input, so combine the two
    # observer arrays (normally they are bit-identical).
    fused_candidates: List[np.ndarray] = []
    for c in candidates:
        if "ffn_gate_up_exps" not in c:
            continue
        for split_name in (
            c.replace("ffn_gate_up_exps", "ffn_gate_exps"),
            c.replace("ffn_gate_up_exps", "ffn_up_exps"),
        ):
            value = imatrix.get(split_name)
            if value is not None:
                fused_candidates.append(value)
    if fused_candidates:
        reference_shape = fused_candidates[0].shape
        compatible = [value for value in fused_candidates if value.shape == reference_shape]
        return np.mean(np.stack(compatible), axis=0, dtype=np.float32)

    # The normal text context does not execute the MTP block. When its mapped
    # GGUF layer has no direct observer, use the mean observer for the same
    # projection across calibrated base layers. This is preferable to
    # magnitude-only selection and is shape checked before use.
    if gguf_name is not None:
        match = re.match(r"blk\.\d+\.(.+)$", gguf_name)
        if match:
            suffix = match.group(1)
            suffixes = [suffix]
            if "ffn_gate_up_exps" in suffix:
                suffixes = [
                    suffix.replace("ffn_gate_up_exps", "ffn_gate_exps"),
                    suffix.replace("ffn_gate_up_exps", "ffn_up_exps"),
                ]
            layer_values = [
                value
                for key, value in imatrix.items()
                if any(re.fullmatch(rf"blk\.\d+\.{re.escape(item)}", key)
                       for item in suffixes)
            ]
            if layer_values:
                reference_shape = layer_values[0].shape
                compatible = [value for value in layer_values if value.shape == reference_shape]
                if compatible:
                    return np.mean(np.stack(compatible), axis=0, dtype=np.float32)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Tile640 quantization (MLX or numpy)
# ═══════════════════════════════════════════════════════════════════════

def ternarize_with_acts(weights: np.ndarray, act_scales: Optional[np.ndarray],
                         outlier_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize a 2D weight matrix to ternary {-1, 0, +1} + outliers.
    Outlier selection criterion:
      - With act_scales:   |w[r,c] × x̂[c]|   (AWQ-style, calibrated)
      - Without act_scales: |w[r,c]|          (magnitude-only)
    Returns: (ternary, outlier_indices_flat, outlier_values)
    """
    out_dim, in_dim = weights.shape
    n = out_dim * in_dim
    abs_flat = np.abs(weights).astype(np.float32).flatten()

    if act_scales is not None:
        assert act_scales.shape == (in_dim,), f"act_scales shape mismatch: {act_scales.shape} vs {(in_dim,)}"
        # Calibrated score: |w| * x̂, tiled across rows
        score = np.abs(weights.astype(np.float32)) * act_scales.astype(np.float32).reshape(1, in_dim)
        score_flat = score.flatten()
    else:
        score_flat = abs_flat

    outlier_count = max(0, int(np.ceil(n * outlier_frac)))
    outlier_count = min(outlier_count, n)

    if outlier_count:
        # Stable argsort (descending) — must match Rust's sort_by for determinism.
        sorted_idx = np.argsort(-score_flat, kind='stable')
        outlier_idx = sorted_idx[:outlier_count].astype(np.int64)
    else:
        outlier_idx = np.zeros(0, dtype=np.int64)
    outlier_vals = weights.flatten()[outlier_idx].astype(np.float32)

    # Compute ternary values
    abs_sum = abs_flat.sum()
    threshold = abs_sum / n
    is_outlier = np.zeros(n, dtype=bool)
    is_outlier[outlier_idx] = True
    is_nonzero = (abs_flat >= threshold) & (~is_outlier)
    is_pos = (weights.flatten() > 0) & is_nonzero

    ternary = np.zeros(n, dtype=np.int8)
    ternary[is_pos] = 1
    ternary[is_nonzero & ~is_pos] = -1

    return ternary, outlier_idx, outlier_vals


def ternarize_with_acts_mlx(weights: np.ndarray, act_scales: Optional[np.ndarray],
                             outlier_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as ternarize_with_acts but using MLX for ~30-40x speedup on Apple Silicon."""
    if not HAS_MLX:
        return ternarize_with_acts(weights, act_scales, outlier_frac)

    out_dim, in_dim = weights.shape
    n = out_dim * in_dim
    weights_mx = mx.array(weights.astype(np.float32))
    abs_mx = mx.abs(weights_mx)
    abs_flat = np.array(abs_mx.flatten())

    if act_scales is not None:
        act_mx = mx.array(act_scales.astype(np.float32).reshape(1, in_dim))
        score_mx = abs_mx * act_mx
        score_flat = np.array(mx.flatten(score_mx))
    else:
        score_flat = abs_flat

    outlier_count = max(0, int(np.ceil(n * outlier_frac)))
    outlier_count = min(outlier_count, n)

    if outlier_count:
        sorted_idx_mx = mx.argsort(-mx.array(score_flat))
        mx.eval(sorted_idx_mx)
        sorted_idx = np.array(sorted_idx_mx, dtype=np.int64)
        outlier_idx = sorted_idx[:outlier_count]
        outlier_vals = np.array(
            weights_mx.flatten()[sorted_idx_mx][:outlier_count],
            dtype=np.float32,
        )
    else:
        outlier_idx = np.zeros(0, dtype=np.int64)
        outlier_vals = np.zeros(0, dtype=np.float32)

    # Build ternary
    abs_sum = float(mx.sum(abs_mx).item())
    threshold = abs_sum / n
    is_outlier = np.zeros(n, dtype=bool)
    is_outlier[outlier_idx] = True
    is_nonzero = (abs_flat >= threshold) & (~is_outlier)
    is_pos = (weights.flatten() > 0) & is_nonzero
    ternary = np.zeros(n, dtype=np.int8)
    ternary[is_pos] = 1
    ternary[is_nonzero & ~is_pos] = -1

    return ternary, outlier_idx, outlier_vals


def select_repair_residuals(
    weights: np.ndarray,
    core_weights: np.ndarray,
    ternary: np.ndarray,
    act_scales: Optional[np.ndarray],
    outlier_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out_dim, in_dim = core_weights.shape
    count = min(
        core_weights.size,
        max(0, int(np.ceil(core_weights.size * outlier_frac))),
    )
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    page_scales, lane_scales = compute_scales(
        core_weights, ternary, out_dim, in_dim
    )
    pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
    padded_in_dim = pages_per_row * TILE640_PAGE_SIZE
    scale_per_lane = (
        page_scales.astype(np.float32).reshape(out_dim, pages_per_row, 1)
        * lane_scales.astype(np.float32).reshape(
            out_dim, pages_per_row, TILE640_LANES_PER_PAGE
        )
        / 127.0
    )
    importance = None
    if act_scales is not None:
        if act_scales.shape != (in_dim,):
            raise ValueError(
                f"act_scales shape mismatch: {act_scales.shape} vs {(in_dim,)}"
            )
        importance = np.maximum(act_scales.astype(np.float32), 1e-8)

    # A single global MLX argpartition over the billion-element tied token
    # embedding returned duplicated zero indices on Apple GPU.  Besides being
    # invalid CSR, global selection can starve most lookup rows.  Large
    # get-rows matrices are therefore selected independently per row in
    # bounded chunks.  This keeps every row within its physical width and
    # makes the residual budget useful for arbitrary vocabulary tokens.
    if core_weights.size >= LARGE_MATRIX_ROW_BALANCED_THRESHOLD:
        per_row = min(in_dim, max(1, int(np.ceil(in_dim * outlier_frac))))
        index_parts = []
        value_parts = []
        row_chunk = 256
        ternary_rows = ternary.reshape(out_dim, in_dim)
        for row_begin in range(0, out_dim, row_chunk):
            row_end = min(out_dim, row_begin + row_chunk)
            chunk_scales = np.repeat(
                scale_per_lane[row_begin:row_end],
                TILE640_LANE_SIZE,
                axis=-1,
            ).reshape(row_end - row_begin, padded_in_dim)[:, :in_dim]
            chunk_reconstructed = (
                ternary_rows[row_begin:row_end].astype(np.float32)
                * chunk_scales
            )
            chunk_weights = weights[row_begin:row_end].astype(
                np.float32, copy=False
            )
            if ACCELERATE_BACKEND is not None:
                chunk_error = ACCELERATE_BACKEND.weighted_square_error(
                    chunk_weights, chunk_reconstructed, importance
                )
            else:
                chunk_error = np.square(chunk_weights - chunk_reconstructed)
                if importance is not None:
                    chunk_error *= np.square(importance).reshape(1, in_dim)
            selected = np.argpartition(
                -chunk_error, per_row - 1, axis=1
            )[:, :per_row]
            rows = np.arange(row_begin, row_end, dtype=np.int64)[:, None]
            global_indices = rows * in_dim + selected.astype(np.int64)
            index_parts.append(global_indices.reshape(-1))
            value_parts.append(
                np.take_along_axis(chunk_weights, selected, axis=1).reshape(-1)
            )
        indices = np.concatenate(index_parts)
        values = np.concatenate(value_parts).astype(np.float32, copy=False)
        expected = out_dim * per_row
        if indices.size != expected or np.unique(indices).size != expected:
            raise ValueError("large-matrix residual selection produced duplicate indices")
        return indices, values

    expanded = np.repeat(
        scale_per_lane, TILE640_LANE_SIZE, axis=-1
    ).reshape(out_dim, padded_in_dim)[:, :in_dim]
    reconstructed = ternary.reshape(out_dim, in_dim).astype(np.float32) * expanded
    if ACCELERATE_BACKEND is not None:
        error = ACCELERATE_BACKEND.weighted_square_error(
            weights.astype(np.float32, copy=False),
            reconstructed,
            importance,
        )
    else:
        error = np.square(weights.astype(np.float32) - reconstructed)
        if importance is not None:
            error *= np.square(importance).reshape(1, in_dim)
    flat_error = error.reshape(-1)
    if count == flat_error.size:
        indices = np.arange(count, dtype=np.int64)
    elif HAS_MLX:
        # Partition on the Apple GPU instead of fully sorting every weight.
        # Sort only the selected tail to keep deterministic descending order.
        error_mx = mx.array(flat_error)
        selected_mx = mx.argpartition(-error_mx, count - 1)[:count]
        mx.eval(selected_mx)
        selected = np.asarray(selected_mx, dtype=np.int64)
        order = np.lexsort((selected, -flat_error[selected]))
        indices = selected[order]
    else:
        selected = np.argpartition(-flat_error, count - 1)[:count]
        order = np.lexsort((selected, -flat_error[selected]))
        indices = selected[order].astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= core_weights.size):
        raise ValueError("residual selection produced an out-of-range index")
    if indices.size != core_weights.size and np.unique(indices).size != indices.size:
        raise ValueError("residual selection produced duplicate indices")
    values = weights.reshape(-1)[indices].astype(np.float32)
    return indices, values


def pack_tile640(ternary_flat: np.ndarray, out_dim: int, in_dim: int) -> Tuple[np.ndarray, int]:
    """Pack flat {-1,0,+1} array into base-3 u32 words, 20 trits per u32, LSB-first.
    Returns (packed_u32, pages_per_row).
    """
    pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
    padded_in_dim = pages_per_row * TILE640_PAGE_SIZE
    rows = ternary_flat.reshape(out_dim, in_dim)
    if padded_in_dim != in_dim:
        rows = np.pad(rows, ((0, 0), (0, padded_in_dim - in_dim)))
    t = rows.reshape(out_dim, pages_per_row, TILE640_LANES_PER_PAGE, TILE640_LANE_SIZE)
    # Convert {-1, 0, +1} -> {2, 0, 1} (base-3 digits, LSB-first)
    trits = np.where(t > 0, 1, np.where(t < 0, 2, 0)).astype(np.uint32)
    pow3 = np.array([3 ** i for i in range(TILE640_LANE_SIZE)], dtype=np.uint32)
    words = (trits * pow3).sum(axis=-1, dtype=np.uint32)
    return words.flatten(), pages_per_row


def compute_scales(weights: np.ndarray, ternary_flat: np.ndarray,
                   out_dim: int, in_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-lane scales and encode them relative to an FP16 page scale.

    For a fixed ternary sign pattern, mean(abs(weight)) over retained nonzero
    values is the least-squares optimal scalar for each lane.

    Returns (page_scales_f16[out, pages], lane_scales_i8[out, pages, 32])."""
    pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
    padded_in_dim = pages_per_row * TILE640_PAGE_SIZE
    w = weights.reshape(out_dim, in_dim).astype(np.float32)
    t = ternary_flat.reshape(out_dim, in_dim)
    if padded_in_dim != in_dim:
        pad = padded_in_dim - in_dim
        w = np.pad(w, ((0, 0), (0, pad)))
        t = np.pad(t, ((0, 0), (0, pad)))
    page_weights = w.reshape(out_dim * pages_per_row, TILE640_PAGE_SIZE)
    page_ternary = t.reshape(out_dim * pages_per_row, TILE640_PAGE_SIZE)
    if ANE_BACKEND is not None and page_weights.shape[0] >= 64:
        # Core ML functions have static row dimensions.  Fill the largest
        # buckets first and leave a short tail to Accelerate/NumPy.  Keeping
        # the program resident amortizes Core ML dispatch over whole pages.
        lane_target_flat = np.empty(
            (page_weights.shape[0], TILE640_LANES_PER_PAGE), dtype=np.float32
        )
        cursor = 0
        remaining = page_weights.shape[0]
        for bucket in (1024, 256, 64):
            while remaining >= bucket:
                end = cursor + bucket
                lane_target_flat[cursor:end] = ANE_BACKEND.lane_targets(
                    page_weights[cursor:end], page_ternary[cursor:end]
                )
                cursor = end
                remaining -= bucket
        if remaining:
            if ACCELERATE_BACKEND is not None:
                lane_target_flat[cursor:] = ACCELERATE_BACKEND.lane_targets(
                    page_weights[cursor:],
                    page_ternary[cursor:],
                    TILE640_LANE_SIZE,
                )
            else:
                tail_weights = page_weights[cursor:].reshape(
                    remaining, TILE640_LANES_PER_PAGE, TILE640_LANE_SIZE
                )
                tail_retained = page_ternary[cursor:].reshape(
                    remaining, TILE640_LANES_PER_PAGE, TILE640_LANE_SIZE
                ) != 0
                tail_count = tail_retained.sum(axis=-1)
                lane_target_flat[cursor:] = np.divide(
                    (np.abs(tail_weights) * tail_retained).sum(axis=-1),
                    tail_count,
                    out=np.zeros_like(tail_count, dtype=np.float32),
                    where=tail_count != 0,
                )
        lane_target = lane_target_flat.reshape(
            out_dim, pages_per_row, TILE640_LANES_PER_PAGE
        )
    elif ACCELERATE_BACKEND is not None:
        lane_target = ACCELERATE_BACKEND.lane_targets(
            page_weights,
            page_ternary,
            TILE640_LANE_SIZE,
        ).reshape(out_dim, pages_per_row, TILE640_LANES_PER_PAGE)
    else:
        w_lanes = w.reshape(
            out_dim,
            pages_per_row,
            TILE640_LANES_PER_PAGE,
            TILE640_LANE_SIZE,
        )
        t_lanes = t.reshape(
            out_dim,
            pages_per_row,
            TILE640_LANES_PER_PAGE,
            TILE640_LANE_SIZE,
        )
        retained = t_lanes != 0
        retained_count = retained.sum(axis=-1)
        lane_target = np.divide(
            (np.abs(w_lanes) * retained).sum(axis=-1),
            retained_count,
            out=np.zeros_like(retained_count, dtype=np.float32),
            where=retained_count != 0,
        )
    page_max = lane_target.max(axis=-1)
    page_max = np.where(page_max < 1e-30, 1.0, page_max)
    raw = (lane_target / page_max[:, :, None]) * 127.0
    lane_scales = np.clip(np.round(raw), 1, 127).astype(np.int8)
    page_scales_f16 = page_max.astype(np.float16)
    if not np.all(np.isfinite(page_scales_f16)):
        raise ValueError("Tile640 page scale is not representable as finite F16")
    return page_scales_f16, lane_scales.reshape(out_dim * pages_per_row, TILE640_LANES_PER_PAGE).flatten()


def normalized_awq_scale(act_scales: np.ndarray, alpha: float) -> np.ndarray:
    """Build a numerically bounded AWQ weight scale.

    Importance matrices can contain zero or near-zero channels. Inverting a
    raw ``max(x, 1e-8) ** alpha`` scale can overflow F16 and poison inference.
    The absolute normalization is immaterial because the reciprocal is applied
    to the activation, so normalize first and bound only the unsupported tail.
    """
    values = np.asarray(act_scales, dtype=np.float32)
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size == 0:
        return np.ones(values.shape, dtype=np.float32)

    reference = float(np.median(finite_positive))
    relative = np.nan_to_num(
        values / max(reference, 1e-8),
        nan=1.0,
        posinf=256.0,
        neginf=1.0 / 256.0,
    )
    relative = np.clip(relative, 1.0 / 256.0, 256.0)
    scale = np.power(relative, alpha, dtype=np.float32)
    if not np.all(np.isfinite(scale)):
        raise ValueError("AWQ produced a non-finite weight scale")
    return scale


def awq_scale_search(weights: np.ndarray, act_scales: np.ndarray,
                     outlier_frac: float,
                     alpha_grid: List[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                     tensor_name: str = "",
                     ) -> Tuple[float, np.ndarray]:
    """AWQ scale search: find per-channel scale s^c that minimizes ternary-quant MSE.

    The optimization target is selected by the module-level `AWQ_SEARCH_TARGET`:
      * "per-row" (default) minimizes importance-weighted per-row reconstruction
        error. Mathematically equivalent to layer-output error when the
        calibration input is diagonal-covariance (synthetic with
        correlation=0), so for a diagonal-only signal this target and
        "layer-output" produce the same alpha.
      * "layer-output" minimizes ||(W_q - W) · X||² with X drawn from a
        per-tensor cache (real calibration activations) or reconstructed
        from the imatrix with banded cross-channel correlation (synthetic
        fallback). For real per-layer X this recovers the original AWQ
        behavior.

    For each candidate alpha, the per-channel scale is s[c]^alpha where s is
    the per-input-channel activation magnitude (RMS). The scaled weights
    W'[r,c] = W[r,c] * s[c]^alpha are ternarized. We pick the alpha that
    minimizes the chosen error.

    Returns (best_alpha, per_channel_scale[in_dim]).
    """
    out_dim, in_dim = weights.shape
    if out_dim > 1024:
        row_ids = np.linspace(0, out_dim - 1, 1024, dtype=np.int64)
        W = weights[row_ids].astype(np.float32)
        out_dim = W.shape[0]
    else:
        W = weights.astype(np.float32)
    s = act_scales.astype(np.float32).reshape(1, in_dim)
    # Avoid division-by-zero at the matmul: clamp s to a small floor
    s_safe = np.maximum(s, 1e-8)

    # Look up real X for the layer-output target. Fall back to synthetic
    # if the tensor is not in the cache.
    real_X = None
    if AWQ_SEARCH_TARGET == "layer-output" and tensor_name:
        lookup = tensor_name
        if lookup.endswith(".weight"):
            lookup = lookup[:-len(".weight")]
        if lookup in CALIBRATION_ACTIVATIONS:
            real_X = CALIBRATION_ACTIVATIONS[lookup]
            if real_X.shape[1] != in_dim:
                print(
                    f"WARN: real_X for {lookup} has in_dim {real_X.shape[1]}, "
                    f"expected {in_dim}; using synthetic",
                    file=sys.stderr,
                )
                real_X = None

    best_alpha = 0.0
    best_err = float("inf")
    best_scale = np.ones(in_dim, dtype=np.float32)

    for alpha in alpha_grid:
        if alpha == 0.0:
            scale = np.ones(in_dim, dtype=np.float32)
        else:
            scale = normalized_awq_scale(s_safe.flatten(), alpha)
        W_scaled = W * scale.reshape(1, in_dim)
        ternary, outlier_idx, outlier_vals = ternarize_with_acts_mlx(W_scaled, act_scales, outlier_frac)
        page_scales, lane_scales = compute_scales(W_scaled, ternary, out_dim, in_dim)
        pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
        padded_in_dim = pages_per_row * TILE640_PAGE_SIZE
        scale_per_lane = (
            page_scales.astype(np.float32).reshape(out_dim, pages_per_row, 1)
            * lane_scales.astype(np.float32).reshape(out_dim, pages_per_row, TILE640_LANES_PER_PAGE)
            / 127.0
        )
        expanded = np.repeat(scale_per_lane, TILE640_LANE_SIZE, axis=-1).reshape(out_dim, padded_in_dim)[:, :in_dim]
        dequant = ternary.reshape(out_dim, in_dim).astype(np.float32) * expanded
        if outlier_idx.size:
            dequant.flat[outlier_idx] = outlier_vals
        effective = dequant / scale.reshape(1, in_dim)
        if AWQ_SEARCH_TARGET == "layer-output":
            if real_X is not None:
                X = real_X.astype(np.float32, copy=False)
            else:
                X = _synthetic_calibration_input(
                    act_scales,
                    batch=AWQ_SYNTHETIC_BATCH,
                    correlation=AWQ_SYNTHETIC_CORRELATION,
                    seed=AWQ_SYNTHETIC_SEED,
                )
            WX_ref = W @ X.T
            WX_q = effective @ X.T
            err = float(np.mean((WX_q - WX_ref) ** 2))
        else:
            err = float(np.mean((effective - W) ** 2 * np.square(act_scales.reshape(1, in_dim))))
        if err < best_err:
            best_err = err
            best_alpha = alpha
            best_scale = scale
    return best_alpha, best_scale


CALIBRATION_ACTIVATIONS: Dict[str, np.ndarray] = {}

# Module-level knobs for the AWQ layer-output search. These are set from the
# CLI in main() and read by `awq_scale_search` when target == "layer-output".
# Defaults preserve the legacy per-row search when --awq-search-target is not
# explicitly set.
AWQ_SEARCH_TARGET: str = "per-row"  # "per-row" | "layer-output"
AWQ_SYNTHETIC_BATCH: int = 32
AWQ_SYNTHETIC_CORRELATION: float = 0.25
AWQ_SYNTHETIC_SEED: int = 0


def load_calibration_activations(path: str) -> Dict[str, np.ndarray]:
    """Load per-layer calibration activations from a .npz file.

    The expected layout is one array per layer/tensor, keyed by the canonical
    GGUF tensor name (e.g. `blk.5.attn_q`). Each array has shape
    `(batch, in_dim)` and dtype float32. Tensors that do not appear in the
    archive fall back to the synthetic approximation in the AWQ search.

    The archive is loaded once and cached at module level so subsequent
    `awq_scale_search` calls within the same quantization run don't reopen
    the file.
    """
    if not CALIBRATION_ACTIVATIONS and os.path.exists(path):
        archive = np.load(path)
        for key in archive.files:
            CALIBRATION_ACTIVATIONS[key] = np.asarray(archive[key], dtype=np.float32)
        print(
            f"  loaded {len(CALIBRATION_ACTIVATIONS)} calibration activation "
            f"arrays from {path}",
            file=sys.stderr,
        )
    return CALIBRATION_ACTIVATIONS


def _synthetic_calibration_input(
    act_scales: np.ndarray,
    batch: int = 32,
    correlation: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic calibration input X (shape: batch x in_dim) whose
    per-channel variance matches the imatrix.

    With `correlation=0.0`, X_c is i.i.d. normal scaled by sqrt(E[x_c²]).
    With `correlation>0`, a Toeplitz-style banded correlation is added
    (so columns within `1/correlation` of each other have a non-zero
    covariance). This breaks the algebraic equivalence with the per-row
    reconstruction error and produces a meaningfully different layer-output
    error signal. correlation in [0, 1); 0.0 is the diagonal-only case.
    """
    in_dim = act_scales.shape[0]
    importance = np.maximum(act_scales.astype(np.float32), 1e-8)
    # Per-channel variance is importance^2 (imatrix records sqrt(E[x^2]))
    variance = importance * importance
    rng = np.random.default_rng(seed)
    # Start with i.i.d. standard normal
    Z = rng.standard_normal((batch, in_dim)).astype(np.float32)
    if correlation > 0.0:
        # Toeplitz band: columns within `band` of each other share a
        # correlation `correlation`. Implemented as a sparse mask multiply
        # (no Cholesky needed for small batch).
        band = max(1, int(round(1.0 / max(correlation, 1e-3))))
        Z_corr = np.zeros_like(Z)
        for offset in range(-band, band + 1):
            weight = correlation ** abs(offset)
            if offset == 0:
                Z_corr += weight * Z
            elif offset > 0 and offset < in_dim:
                Z_corr[:, offset:] += weight * Z[:, :-offset]
                Z_corr[:, :-offset] += weight * Z[:, offset:]
        # Re-normalize so per-column variance is still ~1
        col_var = Z_corr.var(axis=0, keepdims=True) + 1e-8
        Z_corr = Z_corr / np.sqrt(col_var)
        Z = Z_corr
    return Z * np.sqrt(variance).reshape(1, in_dim)


def awq_scale_search_layer_output(
    weights: np.ndarray,
    act_scales: np.ndarray,
    outlier_frac: float,
    alpha_grid: List[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    synthetic_batch: int = 32,
    correlation: float = 0.25,
    seed: int = 0,
    real_X: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """AWQ scale search with layer-output MSE as the optimization target.

    The original AWQ paper picks alpha by minimizing
        L(s) = || Q(W·s^α) · (1/s^α · X)  -  W·X ||²
    on real per-layer activation snapshots X. When those snapshots are
    unavailable, we approximate with synthetic X reconstructed from the
    imatrix: `X[c] = sqrt(E[x_c²]) · z_c` with optional banded correlation
    to break the algebraic equivalence with the per-row error.

    With `real_X` supplied (shape: batch x in_dim, from a calibration forward
    pass), the synthetic approximation is replaced by the real data and
    the search recovers the original AWQ behavior.

    Returns (best_alpha, per_channel_scale[in_dim]).
    """
    out_dim, in_dim = weights.shape
    if out_dim > 1024:
        row_ids = np.linspace(0, out_dim - 1, 1024, dtype=np.int64)
        W = weights[row_ids].astype(np.float32)
        out_dim_eff = W.shape[0]
    else:
        W = weights.astype(np.float32)
        out_dim_eff = out_dim

    if real_X is not None:
        if real_X.ndim != 2 or real_X.shape[1] != in_dim:
            raise ValueError(
                f"real_X must have shape (batch, {in_dim}), got {real_X.shape}"
            )
        X = real_X.astype(np.float32, copy=False)
    else:
        X = _synthetic_calibration_input(
            act_scales, batch=synthetic_batch, correlation=correlation, seed=seed
        )

    # Reference output (W · X) computed once. shape: (out_dim_eff, batch)
    WX_ref = W @ X.T

    s = act_scales.astype(np.float32).reshape(1, in_dim)
    s_safe = np.maximum(s, 1e-8)

    best_alpha = 0.0
    best_err = float("inf")
    best_scale = np.ones(in_dim, dtype=np.float32)

    for alpha in alpha_grid:
        if alpha == 0.0:
            scale = np.ones(in_dim, dtype=np.float32)
        else:
            scale = normalized_awq_scale(s_safe.flatten(), alpha)
        # Scale weights, ternarize, then dequantize back in the original scale.
        # The "inverse" 1/s^α is what the runtime applies to the input, so
        # the dequantized weight (W_q) needs to be divided by s^α to compute
        # the *effective* weight that gets multiplied by the *original* X.
        W_scaled = W * scale.reshape(1, in_dim)
        q_int = np.clip(np.round(W_scaled), -1, 1).astype(np.float32)
        W_eff = q_int / scale.reshape(1, in_dim)
        # Outliers are stored separately and contribute to the layer output
        # independently; for the AWQ alpha search we ignore them because
        # they are by construction high-magnitude and stored in F16 (no
        # quantization error of their own).
        WX_q = W_eff @ X.T
        diff = WX_q - WX_ref
        # Per-row L2, then mean over rows. Using the Frobenius norm divided
        # by (out_dim * batch) gives a comparable scale across layers.
        err = float(np.mean(diff * diff))
        if err < best_err:
            best_err = err
            best_alpha = alpha
            best_scale = scale
    return best_alpha, best_scale


def quantize_2d(weights: np.ndarray, out_dim: int, in_dim: int,
                outlier_frac: float, act_scales: Optional[np.ndarray] = None,
                awq_alpha: Optional[float] = 0.0,
                awq_clip: float = 1.0,
                tensor_name: str = "",
                ternary_threshold: float = 1.0,
                lrq_u: Optional[np.ndarray] = None,
                lrq_v: Optional[np.ndarray] = None,
                lrq_agg: str = "mean",
                ) -> Dict[str, np.ndarray]:
    """Full 2D weight quantization: ternary + pack + scales + outliers.

    AWQ step (when act_scales and awq_alpha > 0):
        W' = W * s[c]^awq_alpha   (per-input-channel scaling)
        ternarize and scale W'
        stored_input_scale[c] = 1.0 / s[c]^awq_alpha   (applied to input at runtime)

    LRQ step (when lrq_u and lrq_v are provided):
        Reconstruct S = U @ V (shape (out_dim, in_dim)). Aggregate across
        the output dimension to a per-input-channel scale
        ``s_agg[c] = mean_r(S[r, c])`` (or RMS for ``lrq_agg="rms"``).
        Apply the per-input-channel scale to the weight, ternarize, and
        store ``1.0 / s_agg`` as the input scale. The full U and V are not
        applied at runtime (we do not touch the runtime); they are kept in
        the policy as audit metadata so a future runtime extension can
        apply the low-rank correction additively.

    `ternary_threshold` is a multiplier on the per-row mean(|W|) used as the
    {-1, 0, +1} cutoff. Default 1.0 = legacy tessera behaviour. A value >1
    produces a sparser quantization (more zeros), <1 denser. Per-tensor
    calibrated via tools/tessera/per_tensor_calibrate.py.
    """
    if lrq_u is not None and lrq_v is not None:
        # Reconstruct the rank-r scale and aggregate it to per-input-channel.
        # The aggregation preserves energy ("rms") or the linear mean
        # ("mean"); both reduce to the AWQ convention at rank 1.
        s = np.asarray(lrq_u, dtype=np.float32) @ np.asarray(lrq_v, dtype=np.float32)
        if lrq_agg == "rms":
            s_agg = np.sqrt(np.mean(s * s, axis=0) + 1e-12).astype(np.float32)
        else:
            s_agg = np.mean(s, axis=0).astype(np.float32)
        # Clamp before inversion so the F16 reciprocal is always finite.
        s_agg = np.maximum(s_agg, np.float32(1e-6))
        w_scale = s_agg.reshape(1, in_dim)
        weights_scaled = (weights.astype(np.float32) * w_scale)
        input_scale = (1.0 / s_agg).astype(np.float32)
        resolved_alpha = 0.0  # AWQ path is bypassed when LRQ is active
    elif act_scales is not None and awq_alpha is None:
        awq_alpha, _ = awq_scale_search(
            weights, act_scales, outlier_frac, tensor_name=tensor_name
        )
        resolved_alpha = 0.0 if awq_alpha is None else awq_alpha
        input_scale = np.ones(in_dim, dtype=np.float32)
        if act_scales is not None and resolved_alpha > 0.0:
            # Per-channel scale on weights. Normalize and bound the telemetry
            # before inversion so the serialized F16 reciprocal is always finite.
            w_scale = normalized_awq_scale(act_scales, resolved_alpha)
            weights_scaled = (weights.astype(np.float32) * w_scale.reshape(1, in_dim))
            # inverse goes to the input
            input_scale = 1.0 / w_scale
        else:
            weights_scaled = weights.astype(np.float32)
    else:
        resolved_alpha = 0.0 if awq_alpha is None else awq_alpha
        input_scale = np.ones(in_dim, dtype=np.float32)
        if act_scales is not None and resolved_alpha > 0.0:
            # Per-channel scale on weights. Normalize and bound the telemetry
            # before inversion so the serialized F16 reciprocal is always finite.
            w_scale = normalized_awq_scale(act_scales, resolved_alpha)
            weights_scaled = (weights.astype(np.float32) * w_scale.reshape(1, in_dim))
            # inverse goes to the input
            input_scale = 1.0 / w_scale
        else:
            weights_scaled = weights.astype(np.float32)

    if not 0.7 <= awq_clip <= 1.0:
        raise ValueError(f"AWQ clip must be in [0.7, 1.0], got {awq_clip}")
    if not 0.3 <= ternary_threshold <= 3.0:
        raise ValueError(
            f"ternary_threshold must be in [0.3, 3.0], got {ternary_threshold}"
        )
    if awq_clip < 1.0:
        row_limit = (
            np.max(np.abs(weights_scaled), axis=1, keepdims=True)
            * np.float32(awq_clip)
        )
        core_weights = np.clip(weights_scaled, -row_limit, row_limit)
    else:
        core_weights = weights_scaled
    ternary, _, _ = ternarize_with_acts_mlx(
        core_weights, None, 0.0
    )
    # Apply per-tensor calibrated threshold as a multiplier on the per-row
    # mean(|W|) cutoff. The default 1.0 keeps legacy behaviour. Per-tensor
    # values from the calibration pass go through here.
    if ternary_threshold != 1.0:
        per_row_threshold = (
            np.mean(np.abs(core_weights), axis=1, keepdims=True)
            * np.float32(ternary_threshold)
        )
        # Re-ternarize: bump positives above the new threshold up to +1,
        # bump positives below the new threshold (but non-zero) down to 0.
        # We rebuild from scratch because the existing ternarize helper
        # hard-codes the threshold to mean(|W|).
        flat = core_weights.flatten()
        flat_ternary = np.zeros(flat.size, dtype=np.int8)
        # The legacy path kept positions where |w| >= mean(|W|); scale that
        # by the calibrated multiplier. Broadcast (out_dim,) over the
        # flattened (out_dim * in_dim,) weight via row-major repeat:
        # `per_row_threshold` is per-row, so the same value applies to
        # every element of that row in the row-major flat layout. The v1
        # comparison used `per_row_threshold.reshape(-1)` (shape out_dim)
        # against an (out_dim * in_dim,) array, which broadcasts the
        # threshold vector against the flat array as if it were a
        # column-major layout — silently shifting the threshold for
        # every row past row 0 and corrupting the ternarized output.
        threshold_flat = np.repeat(per_row_threshold.reshape(-1), in_dim)
        abs_flat = np.abs(flat)
        keep = abs_flat >= threshold_flat
        # We also need to know sign for the kept positions.
        flat_ternary[keep & (flat > 0)] = 1
        flat_ternary[keep & (flat < 0)] = -1
        ternary = flat_ternary
    outlier_idx, outlier_vals = select_repair_residuals(
        weights_scaled,
        core_weights,
        ternary,
        act_scales,
        outlier_frac,
    )
    if outlier_idx.size:
        ternary[outlier_idx] = 0
    packed, pages_per_row = pack_tile640(ternary, out_dim, in_dim)
    page_scales_f16, lane_scales = compute_scales(core_weights, ternary, out_dim, in_dim)

    # Build outlier (row, col) from flat indices. Residual = original weight
    # at the outlier position (ternary is 0 there by construction).
    if outlier_idx.size:
        outlier_rows = (outlier_idx // in_dim).astype(np.int64)
        order = np.argsort(outlier_rows, kind="stable")
        outlier_rows = outlier_rows[order]
        outlier_cols = (outlier_idx[order] % in_dim).astype(np.int32)
        outlier_resid = outlier_vals[order].astype(np.float16)
        row_counts = np.bincount(outlier_rows, minlength=out_dim)
        outlier_row_offsets = np.empty(out_dim + 1, dtype=np.int32)
        outlier_row_offsets[0] = 0
        np.cumsum(row_counts, out=outlier_row_offsets[1:])
    else:
        outlier_row_offsets = np.zeros(out_dim + 1, dtype=np.int32)
        outlier_cols = np.zeros(0, dtype=np.int32)
        outlier_resid = np.zeros(0, dtype=np.float16)

    input_scale_f16 = input_scale.astype(np.float16)
    if not np.all(np.isfinite(input_scale_f16)):
        raise ValueError("AWQ input scale is not representable as finite F16")

    return {
        # GGUF has no U32 tensor type. Tile640 consumes the words as raw
        # 32-bit bit patterns, so store the identical bytes as signed I32.
        "packed": packed.astype(np.uint32).view(np.int32),
        # Component tensors are intentionally flat. The runtime reconstructs
        # rows/pages from the logical shape and requires ne[0] to be the total
        # page-scale count rather than the pages-per-row axis.
        "page_scales": page_scales_f16.reshape(-1),
        "lane_scales": lane_scales,
        "outlier_row_offsets": outlier_row_offsets,
        "outlier_cols": outlier_cols,
        "outlier_vals": outlier_resid,
        "input_scale": input_scale_f16,  # applied to input at runtime
        "awq_alpha": resolved_alpha,
        "awq_clip": awq_clip,
    }


DEFAULT_GEMMA4_SENSITIVE_PATTERNS: Tuple[str, ...] = (
    # QK-norm scales. Pre-normalization of Q and K in gemma 4 removes the
    # natural 1/sqrt(d_k) dampening, so any precision loss in these scales
    # propagates unattenuated through attention. Force exact (outlier_frac=1).
    "attn_q_norm",
    "attn_k_norm",
    # Post-norm scales. Gemma 4 has both pre-norm and post-norm RMSNorm; the
    # post-norm sits between the attention/FFN sublayer output and the next
    # residual stream, where it interacts directly with the precision-sensitive
    # parts of the network. Gemma 4 uses the suffix `_norm` (not the
    # `_layernorm` suffix seen in earlier architectures).
    "post_attention_norm",
    "post_feedforward_norm",
    "pre_feedforward_norm",
    # Attention output projection. Reads from the value projection whose
    # weights are multiplied by softmax(QK^T) — anything that perturbs the
    # attention output gets amplified through the residual stream.
    ".attn_output.",
    # FFN down projection. Output of the FFN block, feeds into post-norm.
    ".ffn_down.",
    # Embedding and output (lm_head) tensors. The output projection is
    # particularly sensitive because it produces the final logits.
    "token_embd",
    "output.",
    "output_norm",
)


def is_gemma4_sensitive_tensor(
    tensor_name: str, extra_patterns: Optional[List[str]] = None
) -> bool:
    """True iff `tensor_name` matches any of the gemma 4 sensitive patterns
    (default or user-supplied). The quantizer forces these to exact
    (outlier_frac=1.0) Tile640 encoding, which is functionally F16 storage
    with Tile640 wrapping — preserves the full precision of the tensor
    while keeping the runtime layout uniform.
    """
    patterns = DEFAULT_GEMMA4_SENSITIVE_PATTERNS
    if extra_patterns:
        patterns = patterns + tuple(extra_patterns)
    name = tensor_name.lower()
    return any(p.lower() in name for p in patterns)


def _imatrix_mse_row_scale(
    W_row: np.ndarray,
    importance: np.ndarray,
    grid: int = 20,
    maxshrink: float = 0.95,
    norm: float = 3.0,
) -> float:
    """Find the per-row scale that minimizes importance-weighted reconstruction
    error under 8-bit symmetric quantization. This is the vllm `imatrix_mse`
    observer (RFC #2456) applied at row granularity instead of group.

    The error is `err = sum_c(importance_c * |W_c - s * round(W_c/s)|^p)` for
    candidate scales `s` in `[maxshrink * max(|W|), max(|W|)]`. We pick the
    scale that minimizes the error; ties broken toward the larger scale
    (preserves the natural dynamic range of the row).
    """
    max_abs = float(np.max(np.abs(W_row)))
    if max_abs <= 1e-12:
        return 1.0
    lo = maxshrink * max_abs
    hi = max_abs
    if hi - lo < 1e-9:
        return hi
    candidates = np.linspace(lo, hi, grid + 1, dtype=np.float32)
    best_err = float("inf")
    best_scale = float(hi)
    for s in candidates:
        if s <= 1e-12:
            continue
        q = np.clip(np.round(W_row / s), -127, 127)
        recon = q.astype(np.float32) * s
        diff = np.abs(W_row - recon).astype(np.float32)
        if norm == 2.0:
            err = float(np.sum(importance * diff * diff))
        elif norm == 1.0:
            err = float(np.sum(importance * diff))
        else:
            err = float(np.sum(importance * np.power(diff, norm)))
        if err < best_err:
            best_err = err
            best_scale = float(s)
    return best_scale


def quantize_2d_imatrix_mse(
    weights: np.ndarray,
    out_dim: int,
    in_dim: int,
    outlier_frac: float,
    act_scales: np.ndarray,
    mse_norm: float = 3.0,
    mse_grid: int = 20,
    mse_maxshrink: float = 0.95,
    awq_clip: float = 1.0,
) -> Dict[str, np.ndarray]:
    """2D weight quantization with imatrix-weighted MSE range selection.

    This is the vllm `imatrix_mse` observer analogue, applied at row
    granularity. For each row we do an MSE grid search for the scale that
    minimizes importance-weighted reconstruction error under 8-bit
    symmetric quantization, then ternarize, treat outliers, and pack into
    Tile640.

    AWQ per-channel pre-scaling is **disabled** in this path: the importance
    signal is already consumed by the range selection, so the per-channel
    scale would double-count it.
    """
    if act_scales is None:
        raise ValueError("imatrix_mse range selection requires act_scales")
    W = weights.astype(np.float32, copy=False)
    importance = np.power(
        np.maximum(act_scales.astype(np.float32), 1e-8), mse_norm
    )

    pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
    padded_in_dim = pages_per_row * TILE640_PAGE_SIZE

    # Per-row MSE-optimal scale. We sample down for very large matrices to
    # keep the grid search cost bounded; the resulting scale is then broadcast
    # to all rows via the page structure.
    if out_dim > 4096:
        sample_rows = np.linspace(0, out_dim - 1, 4096, dtype=np.int64)
        sample_scales = np.array(
            [
                _imatrix_mse_row_scale(
                    W[r], importance, mse_grid, mse_maxshrink, mse_norm
                )
                for r in sample_rows
            ],
            dtype=np.float32,
        )
        # Per-row scale from a nearest-neighbor projection so the per-row
        # signal is preserved (different rows genuinely have different
        # dynamic ranges in MoE / wide FFN layers).
        scales_per_row = np.interp(
            np.arange(out_dim, dtype=np.float32),
            sample_rows.astype(np.float32),
            sample_scales,
        ).astype(np.float32)
    else:
        scales_per_row = np.array(
            [
                _imatrix_mse_row_scale(
                    W[r], importance, mse_grid, mse_maxshrink, mse_norm
                )
                for r in range(out_dim)
            ],
            dtype=np.float32,
        )

    # Apply AWQ-style clipping on the original weights. This is a hard cap on
    # the magnitude that participates in ternarization, controlled by the
    # calibration policy or a CLI default.
    if not 0.7 <= awq_clip <= 1.0:
        raise ValueError(f"AWQ clip must be in [0.7, 1.0], got {awq_clip}")
    if awq_clip < 1.0:
        row_limit = (
            np.max(np.abs(W), axis=1, keepdims=True) * np.float32(awq_clip)
        )
        core_weights = np.clip(W, -row_limit, row_limit)
    else:
        core_weights = W

    # Rescale each row so the max(|W|/scale) is ~127/127, then ternarize.
    # The MSE scale becomes the page scale; the ternarized values become the
    # 2-bit pattern.
    inv_scales = 1.0 / np.maximum(scales_per_row, 1e-8)
    weights_normalized = core_weights * inv_scales.reshape(out_dim, 1)
    if padded_in_dim > in_dim:
        weights_normalized = np.pad(
            weights_normalized, ((0, 0), (0, padded_in_dim - in_dim))
        )
    ternary, _, _ = ternarize_with_acts_mlx(weights_normalized, None, 0.0)
    # ternarize_with_acts_mlx returns a flat 1D array; reshape to 2D for
    # the rest of the pipeline (select_repair_residuals, pack_tile640,
    # compute_scales all expect 2D shapes).
    ternary_2d = ternary.reshape(out_dim, padded_in_dim)[:, :in_dim]

    # Outliers are selected against the original (un-normalized) weights, with
    # the importance signal still applied so high-importance positions
    # preferentially become residuals.
    outlier_idx, outlier_vals = select_repair_residuals(
        W,
        weights_normalized[:, :in_dim],
        ternary_2d,
        act_scales,
        outlier_frac,
    )
    if outlier_idx.size:
        ternary_2d.flat[outlier_idx] = 0
    packed, _ = pack_tile640(ternary_2d, out_dim, in_dim)
    _, lane_scales = compute_scales(
        weights_normalized, ternary_2d, out_dim, in_dim
    )

    # Page scales = MSE-optimal scale per row, in FP16, replicated across pages
    # in the row. Replicating rather than averaging keeps the dynamic range
    # information intact; pages are independent quantization blocks.
    page_scales_f16 = np.repeat(
        scales_per_row.astype(np.float16).reshape(out_dim, 1),
        pages_per_row,
        axis=1,
    )

    if outlier_idx.size:
        outlier_rows = (outlier_idx // in_dim).astype(np.int64)
        order = np.argsort(outlier_rows, kind="stable")
        outlier_rows = outlier_rows[order]
        outlier_cols = (outlier_idx[order] % in_dim).astype(np.int32)
        outlier_resid = outlier_vals[order].astype(np.float16)
        row_counts = np.bincount(outlier_rows, minlength=out_dim)
        outlier_row_offsets = np.empty(out_dim + 1, dtype=np.int32)
        outlier_row_offsets[0] = 0
        np.cumsum(row_counts, out=outlier_row_offsets[1:])
    else:
        outlier_row_offsets = np.zeros(out_dim + 1, dtype=np.int32)
        outlier_cols = np.zeros(0, dtype=np.int32)
        outlier_resid = np.zeros(0, dtype=np.float16)

    return {
        "packed": packed.astype(np.uint32).view(np.int32),
        "page_scales": page_scales_f16.reshape(-1),
        "lane_scales": lane_scales,
        "outlier_row_offsets": outlier_row_offsets,
        "outlier_cols": outlier_cols,
        "outlier_vals": outlier_resid,
        "input_scale": np.ones(in_dim, dtype=np.float16),  # AWQ disabled
        "awq_alpha": 0.0,
        "awq_clip": awq_clip,
    }


def _septq_banded_cholesky(H: np.ndarray, bandwidth: int) -> np.ndarray:
    """Compute a lower-triangular Cholesky factor L with H = L @ L.T.

    Two strategies are used depending on banded-Cholesky feasibility:

    1. **Banded Cholesky** (cost O(n * bandwidth^2)): the standard
       outer-product Cholesky restricted to the band. Works when the
       off-band energy of H is small relative to the diagonal (i.e., H is
       "approximately" banded with the requested bandwidth).

    2. **Full Cholesky + banded read-out** (fallback, cost O(n^3) via
       BLAS): when the banded Cholesky fails because the off-band energy
       of H is non-negligible (the common case for a rank-deficient H
       from a small calibration set), compute the full Cholesky factor
       and return it. The GPTQ-M helper does a banded forward-substitution
       on the returned L which gives the exact banded portion of L^{-1}
       regardless of whether L is full or banded.

    The fallback fires only when the banded Cholesky would produce a
    non-positive-definite s = H[j,j] - sum_{k=k_min}^{j-1} L[j,k]^2.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got {H.shape}")
    n = H.shape[0]
    if bandwidth < 0:
        raise ValueError(f"bandwidth must be >= 0, got {bandwidth}")
    L = np.zeros((n, n), dtype=H.dtype)
    fallback = False
    for j in range(n):
        k_min = max(0, j - bandwidth)
        if k_min < j:
            s = H[j, j] - L[j, k_min:j] @ L[j, k_min:j]
        else:
            s = H[j, j]
        if s <= 0.0:
            fallback = True
            break
        L[j, j] = np.sqrt(s)
        i_max = min(n, j + bandwidth + 1)
        if i_max > j + 1 and k_min < j:
            L[j + 1:i_max, j] = (
                H[j + 1:i_max, j] - L[j + 1:i_max, k_min:j] @ L[j, k_min:j]
            ) / L[j, j]
        elif i_max > j + 1:
            L[j + 1:i_max, j] = H[j + 1:i_max, j] / L[j, j]
    if not fallback:
        return L
    # Fallback: full Cholesky via BLAS. The banded forward-sub in
    # _septq_gptq_M produces the banded portion of L^{-1} regardless
    # of whether L is full or banded, so the rest of the GPTQ-M path
    # is unchanged.
    L_full = np.linalg.cholesky(H)
    return L_full


def _septq_gptq_M(L: np.ndarray, bandwidth: int) -> np.ndarray:
    """Build the strictly upper-triangular GPTQ update matrix from L = chol(H).

    With L lower triangular, (L^{-1})_{jj} = 1 / L[j, j], so the closed-form
    per-column update is ``M[j, k] = (L^{-1})_{k, j} * L[j, j]`` for k > j.
    M is upper triangular with bandwidth ``bandwidth`` (M[j, k] = 0 for
    k - j > bandwidth). The banded portion of L^{-1} comes from a banded
    forward-substitution: for each j, solve L x = e_j keeping only x[k] for
    k in (j, j + bandwidth + 1). Cost is O(n * bandwidth^2).
    """
    n = L.shape[0]
    if bandwidth < 0:
        raise ValueError(f"bandwidth must be >= 0, got {bandwidth}")
    M = np.zeros((n, n), dtype=L.dtype)
    for j in range(n):
        x = np.zeros(n, dtype=L.dtype)
        x[j] = 1.0 / L[j, j]
        k_max = min(n, j + bandwidth + 1)
        for k in range(j + 1, k_max):
            row_min = max(0, k - bandwidth)
            s = -np.dot(L[k, row_min:k], x[row_min:k])
            x[k] = s / L[k, k]
        if k_max > j + 1:
            M[j, j + 1:k_max] = x[j + 1:k_max] * L[j, j]
    return M


def _septq_build_hessian(
    in_dim: int,
    act_scales: Optional[np.ndarray],
    calibration_activations: Optional[np.ndarray],
    ridge_fraction: float = 1e-4,
) -> np.ndarray:
    """Build the symmetric positive-definite Hessian H for the SEPTQ update.

    With calibration activations X of shape (n_samples, in_dim), H is the
    standard second-moment matrix H = X^T X / n_samples, optionally ridged
    for numerical stability. Without activations we fall back to a diagonal
    Hessian with H[j, j] = act_scales[j]^2 (the imatrix RMS proxy).
    """
    if calibration_activations is not None:
        X = np.asarray(calibration_activations, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != in_dim:
            raise ValueError(
                f"calibration_activations must be (n_samples, {in_dim}); got {X.shape}"
            )
        n = X.shape[0]
        H = (X.T @ X) / np.float32(max(n, 1))
        diag_mean = float(np.mean(np.diag(H)))
        ridge = max(np.float32(ridge_fraction) * np.float32(diag_mean),
                    np.float32(1e-2) * np.float32(diag_mean))
        if ridge > 0:
            H = H + np.eye(in_dim, dtype=np.float32) * ridge
    elif act_scales is not None:
        if act_scales.shape != (in_dim,):
            raise ValueError(
                f"act_scales shape mismatch: {act_scales.shape} vs {(in_dim,)}"
            )
        diag = np.maximum(act_scales.astype(np.float32), 1e-8) ** 2
        H = np.diag(diag).astype(np.float32)
    else:
        H = np.eye(in_dim, dtype=np.float32)
    return H


def quantize_2d_septq(weights: np.ndarray, out_dim: int, in_dim: int,
                      septq_ratio: float,
                      act_scales: Optional[np.ndarray] = None,
                      septq_iterations: int = 1,
                      ternary_threshold: float = 1.0,
                      tensor_name: str = "",
                      calibration_activations: Optional[np.ndarray] = None,
                      septq_hessian_mode: str = "banded",
                      septq_hessian_bandwidth: int = 32,
                      septq_importance_weight: str = "quant_error_h",
                      septq_importance_lambda: float = 0.0,
                      ) -> Dict[str, np.ndarray]:
    """2D weight quantization using the SEPTQ recipe (KDD 2025).

    SEPTQ is a two-step PTQ method:

    1. **Static global importance.** Compute a per-element importance score
       ``s[i,j] = (W[i,j] - Q(W[i,j]))^2 * H[j,j]`` where H is the Hessian of
       the calibration activations and Q is a baseline quantizer. The top-k%
       elements by importance form a static global mask M.
    2. **Column-wise quantization with error compensation.** Build the
       strictly upper-triangular GPTQ update matrix M = (H^{-1} / diag(H))
       restricted to a band of width ``septq_hessian_bandwidth``. Quantize
       the masked elements in one vectorized pass to {-1, 0, +1}, then
       apply the cross-column update ``W[:, k] -= sum_j e_j * M[j, k]`` as a
       single (out_dim, in_dim) @ (in_dim, in_dim) matmul.
       {-1, 0, +1} and apply the cross-column update. The diagonal-H
       approximation (this commit) makes the cross-column update a
       no-op; the banded GPTQ-M path is added in a follow-up commit.

    The end result is a mixed-precision representation: the "important"
    elements are quantized to ternary {-1, 0, +1} and the "unimportant"
    elements are kept at full precision. This is the opposite of the
    standard Tessera flow where outliers are the "important" elements.

    The diagonal-only approximation is used for the Hessian inverse: we use
    H[j,j] (the column-wise importance from the imatrix) for the importance
    scoring, and skip the cross-column error compensation. With a strictly
    diagonal H_inv the cross-column update is identically zero, so this is
    a tractable simplification of the full SEPTQ algorithm. The mask
    selection uses argpartition (O(n)) instead of a full sort (O(n log n))
    to make the importance ranking cheaper.

    Two Hessian modes are supported:

    * ``"banded"`` (default when ``calibration_activations`` is provided):
      compute H = X^T X / n from the calibration activations, take the
      banded Cholesky factor with bandwidth ``septq_hessian_bandwidth``,
      and use the GPTQ update. This is the full SEPTQ algorithm modulo
      the band truncation.
    * ``"diagonal"``: H is a diagonal matrix with H[j, j] = act_scales[j]^2
      (the imatrix RMS proxy). M is identically zero, so the cross-column
      update is a no-op; only the static mask is active. This is the
      v1 SEPTQ behaviour and is kept as a fast ablation baseline.

    When ``calibration_activations`` is None and ``septq_hessian_mode`` is
    ``"banded"``, the function silently falls back to the diagonal proxy.
    The main script has no activations, so the banded mode is currently
    only exercised by the A/B harness.

    The output dict has the same shape as ``quantize_2d``. The
    ``outlier_*`` entries store the UNIMPORTANT elements (full precision);
    the ternary stores the IMPORTANT elements. The ``outlier_frac``
    analogue is ``1 - septq_ratio``.

    Args:
        weights: 2D weight matrix [out_dim, in_dim], float32.
        out_dim: output dimension.
        in_dim: input dimension.
        septq_ratio: fraction of elements to quantize (0, 1]. 1.0 = all
            quantized (equivalent to RTN); 0.5 = half quantized, half
            kept full precision.
        act_scales: per-input-channel activation magnitude (RMS), used as
            a proxy for the Hessian diagonal H[j,j]. If None, uniform
            column importance is assumed.
        septq_iterations: number of column-wise passes. The mask is fixed
            at the first iteration (static global mask per the paper);
            subsequent passes re-ternarize with the same mask. The default
            of 1 is the canonical SEPTQ setting.
        ternary_threshold: multiplier on per-row mean(|W|) for the {-1, 0,
            +1} cutoff. Default 1.0 = legacy tessera behaviour.
        tensor_name: for diagnostics only.
        calibration_activations: optional (n_samples, in_dim) calibration
            activations used to build the full Hessian in banded mode.
        septq_hessian_mode: ``"banded"`` (default) or ``"diagonal"``.
        septq_hessian_bandwidth: band radius for banded Cholesky; only
            entries with |i - j| <= bandwidth participate in H^{-1}. The
            default of 32 is a conservative trade-off: small enough to be
            cheap (O(n * b^2) factorization) and large enough to capture
            most of the cross-column benefit on typical layer activations.
        septq_importance_weight: importance score mode. ``"quant_error_h"``
            (default) is the original ``(W - Q(W))^2 * h_diag``. The
            other modes are designed for heavy-tailed weights where the
            original score lets outliers dominate the mask and forces
            them into {-1, 0, +1}, which destroys their full-precision
            values. See the per-mode descriptions in the source.
        septq_importance_lambda: weight on the ``1/(|W| + eps)`` term
            in the ``hybrid`` importance mode. Ignored for other modes.
    """
    if not 0.0 < septq_ratio <= 1.0:
        raise ValueError(f"septq_ratio must be in (0, 1], got {septq_ratio}")
    if septq_iterations < 1:
        raise ValueError(f"septq_iterations must be >= 1, got {septq_iterations}")
    if not 0.3 <= ternary_threshold <= 3.0:
        raise ValueError(
            f"ternary_threshold must be in [0.3, 3.0], got {ternary_threshold}"
        )
    if septq_hessian_mode not in ("diagonal", "banded"):
        raise ValueError(
            f"septq_hessian_mode must be 'diagonal' or 'banded', got {septq_hessian_mode}"
        )
    if septq_hessian_bandwidth < 0:
        raise ValueError(
            f"septq_hessian_bandwidth must be >= 0, got {septq_hessian_bandwidth}"
        )
    if septq_importance_weight not in (
        "quant_error_h", "inv_abs_w", "inv_cdf", "hybrid",
    ):
        raise ValueError(
            f"septq_importance_weight must be one of "
            f"'quant_error_h', 'inv_abs_w', 'inv_cdf', 'hybrid'; "
            f"got {septq_importance_weight!r}"
        )
    if septq_importance_lambda < 0:
        raise ValueError(
            f"septq_importance_lambda must be >= 0, got {septq_importance_lambda}"
        )
    # Bandwidth > in_dim - 1 is the same as full Cholesky. Clamp here so
    # the inner helpers can assume 0 <= bandwidth < in_dim.
    effective_bandwidth = min(septq_hessian_bandwidth, in_dim - 1)

    W = weights.astype(np.float32, copy=False)
    n = out_dim * in_dim
    abs_W = np.abs(W)

    # Resolve Hessian mode: banded requires activations; fall back to diagonal
    # when activations are not available so the main-script path is unchanged.
    effective_mode = septq_hessian_mode
    if effective_mode == "banded" and calibration_activations is None:
        effective_mode = "diagonal"

    # Hessian diagonal proxy. The imatrix's RMS per channel is a cheap
    # surrogate for H[j,j] = E[x_j^2] over the calibration distribution.
    if act_scales is not None:
        if act_scales.shape != (in_dim,):
            raise ValueError(
                f"act_scales shape mismatch: {act_scales.shape} vs {(in_dim,)}"
            )
        h_diag = np.maximum(act_scales.astype(np.float32), 1e-8)
    else:
        h_diag = np.ones(in_dim, dtype=np.float32)

    # Per-row mean(|W|) threshold for the baseline ternarizer. Matches the
    # standard Tessera behaviour at ternary_threshold=1.0.
    row_mean_abs = abs_W.mean(axis=1, keepdims=True)
    threshold_1d = (row_mean_abs * np.float32(ternary_threshold)).reshape(-1)
    keep_2d = abs_W >= threshold_1d.reshape(out_dim, 1)

    # Step 1: baseline ternarization to compute the quantization error used
    # in the importance score. We ternarize using the same mean(|W|) rule
    # as the standard Tessera flow so the SEPTQ importance ranking is
    # directly comparable.
    sign_W = np.sign(W).astype(np.int8)
    ternary_init = np.where(keep_2d, sign_W, np.int8(0))
    quant_error_init = (W - ternary_init.astype(np.float32)) ** 2

    # Step 2: per-element importance. The original score is
    # ``(W - Q(W))^2 * h_diag`` which puts the largest values on the
    # heavy-tail elements (large |W| -> large ternarization error
    # because Q(W) is sign(W) for |W| > row_mean). On heavy-tailed
    # weights this lets outliers dominate the mask and forces them
    # into {-1, 0, +1}, which destroys their full-precision values.
    # The weighted modes downweight outliers so the mask focuses on
    # the bulk where ternarization actually helps the MSE.
    base_importance_2d = quant_error_init * h_diag.reshape(1, in_dim)
    if septq_importance_weight == "quant_error_h":
        importance_2d = base_importance_2d
    elif septq_importance_weight == "inv_abs_w":
        # Divide by |W| (with a small eps for stability) to downweight
        # large-|W| elements. The mask then focuses on the bulk where
        # ternarization is most useful.
        weight_2d = np.float32(1.0) / (abs_W + np.float32(1e-8))
        importance_2d = base_importance_2d * weight_2d
    elif septq_importance_weight == "inv_cdf":
        # Use the empirical 1 - CDF(|W|) as the weight. The CDF is
        # computed per row (matching the per-row ternarization rule).
        # This is the most aggressive downweighting: elements in the
        # top of the row's magnitude distribution get weight ~0, so
        # the mask never picks them.
        abs_W_2d = abs_W
        # Per-row ranks via argsort; convert to a [0, 1] CDF value.
        row_ranks = np.empty_like(abs_W_2d, dtype=np.float32)
        for r in range(out_dim):
            order = np.argsort(abs_W_2d[r], kind="stable")
            row_ranks[r, order] = np.arange(in_dim, dtype=np.float32) / max(in_dim - 1, 1)
        cdf_weight_2d = np.float32(1.0) - row_ranks
        importance_2d = base_importance_2d * cdf_weight_2d
    elif septq_importance_weight == "hybrid":
        # Additive hybrid: original score plus a lambda-weighted
        # 1/(|W| + eps) term. With lambda = 0 this is the original;
        # larger lambda progressively downweights outliers. The
        # 1/(|W| + eps) term is scaled by h_diag so both terms share
        # the same activation weighting and the units are comparable.
        inv_abs_2d = np.float32(1.0) / (abs_W + np.float32(1e-8))
        # Scale lambda to the bulk of the base importance so the
        # lambda = 1 setting produces a comparable contribution from
        # both terms. The base importance at a typical element is
        # roughly E[(W - Q(W))^2] * E[H[j,j]]; we approximate the
        # scale with the median of the base importance, which is
        # robust to outliers.
        base_scale = float(np.median(base_importance_2d))
        inv_scale = float(np.median(inv_abs_2d * h_diag.reshape(1, in_dim)))
        if inv_scale > 0 and base_scale > 0:
            normalized_lambda = (
                np.float32(septq_importance_lambda) *
                np.float32(base_scale / inv_scale)
            )
        else:
            normalized_lambda = np.float32(0.0)
        importance_2d = base_importance_2d + normalized_lambda * inv_abs_2d * h_diag.reshape(1, in_dim)
    else:
        # Unreachable; the constructor validates the choice.
        raise AssertionError(f"unreachable importance mode {septq_importance_weight!r}")
    importance_flat = importance_2d.reshape(-1)

    # Step 3: static global mask. We only need the top-k elements by
    # importance, so we find the k-th largest value with argpartition
    # (O(n)) and use it as a threshold. Cost is O(n) instead of O(n log n)
    # for a full sort. With float32 importance values exact ties at the
    # threshold are vanishingly rare, so the mask has exactly k True
    # entries in practice. If there are ties the mask may have a few
    # more than k entries; that is fine for SEPTQ which only uses the
    # ratio approximately. Determinism is preserved: argpartition is
    # quickselect and the threshold comparison is exact.
    k = max(1, min(n, int(np.ceil(n * septq_ratio))))
    if k >= n:
        mask_2d = np.ones((out_dim, in_dim), dtype=bool)
    else:
        kth_idx = np.argpartition(-importance_flat, k - 1)[k - 1]
        threshold = importance_flat[kth_idx]
        mask_flat = importance_flat >= threshold
        # If the threshold selection overshot (rare, only when there are
        # exact float32 ties at the boundary), cap to exactly k True
        # entries by clearing the lowest-importance over-shoots. Cost is
        # O(n) once more; in practice this branch almost never fires
        # for float32 importance values.
        excess = int(mask_flat.sum()) - k
        if excess > 0:
            true_idx = np.flatnonzero(mask_flat)
            true_importance = importance_flat[true_idx]
            tail = np.argpartition(true_importance, excess)[:excess]
            mask_flat[true_idx[tail]] = False
        mask_2d = mask_flat.reshape(out_dim, in_dim)

    # Step 4: vectorized column-wise quantization with error compensation.
    # The v1 per-column Python loop was the dominant cost. It is replaced
    # by a few vectorized ops: build the quantized mask, the ternary, and
    # the per-position error matrix in a single pass each. The diagonal-H
    # approximation is preserved as the "diagonal" mode (no cross-column
    # update). The "banded" mode applies the GPTQ-M update via
    # W_compensated = W - error_2d @ M with M derived from the banded
    # Cholesky of H = X^T X / n.
    quantized_mask_2d = mask_2d & keep_2d
    ternary_2d = np.where(quantized_mask_2d, sign_W, np.int8(0))
    # E is the per-position error at quantized positions, 0 elsewhere.
    # Casting ternary_2d to float32 once and broadcasting is faster than
    # the per-column `astype` of the original loop.
    error_2d = (ternary_2d.astype(np.float32) - W) * quantized_mask_2d.astype(np.float32)

    if effective_mode == "banded":
        H = _septq_build_hessian(
            in_dim, act_scales, calibration_activations
        )
        L = _septq_banded_cholesky(H, effective_bandwidth)
        M = _septq_gptq_M(L, effective_bandwidth)
        # W_compensated = W - E @ M. With banded M (b << in_dim) this is
        # a dense matmul that BLAS handles in well under a second on
        # 4096^2 inputs. E @ M is O(out_dim * in_dim^2) for the dense
        # matmul even when M is banded (BLAS doesn't know about the zeros
        # outside the band); a custom banded matmul would be O(out_dim *
        # in_dim * bandwidth) but would require a separate code path.
        W_compensated = W - error_2d @ M
    else:
        # Diagonal-only approximation: H_inv is diagonal, so M is
        # identically zero. W_compensated is just W (the per-column
        # local absorption in the v1 loop was a no-op because error_2d
        # is zero at non-quantized positions; it is dropped here for
        # clarity). Avoid the matmul entirely; BLAS would still take
        # ~100ms even on a zero matrix.
        W_compensated = W

    ternary_final = ternary_2d.reshape(-1)

    # Step 5: pack into the Tessera format. The "outliers" are the
    # UNIMPORTANT elements (stored as full precision after the
    # cross-column update has been applied). The ternary stores the
    # IMPORTANT elements. The page/lane scales are computed against the
    # original weights at the ternary positions only (the pack reflects
    # the quantization; the outlier storage reflects the compensated
    # W so that reconstruction at inference matches the optimized W).
    unimportant_2d = np.where(~mask_2d, W_compensated, np.float32(0.0))
    outlier_idx = np.where((~mask_2d).reshape(-1))[0].astype(np.int64)
    outlier_vals = unimportant_2d.reshape(-1)[outlier_idx].astype(np.float32)

    packed, pages_per_row = pack_tile640(ternary_final, out_dim, in_dim)
    page_scales_f16, lane_scales = compute_scales(W, ternary_final, out_dim, in_dim)

    if outlier_idx.size:
        outlier_rows = (outlier_idx // in_dim).astype(np.int64)
        order = np.argsort(outlier_rows, kind="stable")
        outlier_rows = outlier_rows[order]
        outlier_cols = (outlier_idx[order] % in_dim).astype(np.int32)
        outlier_resid = outlier_vals[order].astype(np.float16)
        row_counts = np.bincount(outlier_rows, minlength=out_dim)
        outlier_row_offsets = np.empty(out_dim + 1, dtype=np.int32)
        outlier_row_offsets[0] = 0
        np.cumsum(row_counts, out=outlier_row_offsets[1:])
    else:
        outlier_row_offsets = np.zeros(out_dim + 1, dtype=np.int32)
        outlier_cols = np.zeros(0, dtype=np.int32)
        outlier_resid = np.zeros(0, dtype=np.float16)

    return {
        "packed": packed.astype(np.uint32).view(np.int32),
        "page_scales": page_scales_f16.reshape(-1),
        "lane_scales": lane_scales,
        "outlier_row_offsets": outlier_row_offsets,
        "outlier_cols": outlier_cols,
        "outlier_vals": outlier_resid,
        "input_scale": np.ones(in_dim, dtype=np.float16),  # AWQ disabled
        "awq_alpha": 0.0,
        "awq_clip": 1.0,
        # Extra keys (not consumed by the GGUF writer; used by the A/B harness
        # to reconstruct the quantized weight without unpacking Tile640).
        "_ternary": ternary_final,
        "_mask_2d": mask_2d,
    }


def quantize_3d(weights: np.ndarray, n_experts: int, out_dim: int, in_dim: int,
                outlier_frac, imatrix: Optional[Dict[str, np.ndarray]] = None,
                wname_hf: str = "", gguf_name: str = "",
                awq_alpha: Optional[float] = 0.0,
                awq_clip: float = 1.0,
                ) -> List[Dict[str, np.ndarray]]:
    """Quantize a 3D expert weight (n_experts, out, in) by quantizing each expert
    independently. Returns a list of n_experts per-expert quant dicts."""
    results = []
    act_scales = None
    if imatrix is not None and (wname_hf or gguf_name):
        act_scales = lookup_acts(wname_hf, imatrix, gguf_name)
    pooled_act_scales = (
        np.mean(act_scales, axis=0, dtype=np.float32)
        if act_scales is not None and act_scales.ndim == 2
        else act_scales
    )
    if pooled_act_scales is not None and awq_alpha is None:
        expert_ids = np.linspace(0, n_experts - 1, min(n_experts, 4), dtype=np.int64)
        row_ids = np.linspace(0, out_dim - 1, min(out_dim, 256), dtype=np.int64)
        merged = weights[expert_ids][:, row_ids, :].reshape(-1, in_dim)
        search_fraction = (
            float(np.median(np.asarray(outlier_frac, dtype=np.float64)))
            if not np.isscalar(outlier_frac) else float(outlier_frac)
        )
        awq_alpha, _ = awq_scale_search(
            merged, pooled_act_scales, search_fraction, tensor_name=out_name
        )
    for ex in range(n_experts):
        w_ex = weights[ex]  # (out_dim, in_dim)
        expert_act_scales = (
            act_scales[ex]
            if act_scales is not None and act_scales.ndim == 2
            else act_scales
        )
        results.append(
            quantize_2d(
                w_ex,
                out_dim,
                in_dim,
                (
                    float(outlier_frac[ex])
                    if not np.isscalar(outlier_frac)
                    else float(outlier_frac)
                ),
                expert_act_scales,
                (
                    float(awq_alpha[ex])
                    if isinstance(awq_alpha, (list, tuple, np.ndarray))
                    else awq_alpha
                ),
                (
                    float(awq_clip[ex])
                    if isinstance(awq_clip, (list, tuple, np.ndarray))
                    else awq_clip
                ),
            )
        )
        if (ex + 1) % 16 == 0:
            sys.stderr.write(f"    expert {ex+1}/{n_experts}\r")
            sys.stderr.flush()
    sys.stderr.write("\n")
    return results


# ═══════════════════════════════════════════════════════════════════════
# GGUF output naming
# ═══════════════════════════════════════════════════════════════════════

def gguf_name_for(short: str, il: int, suffix: str = "weight") -> str:
    """Build a GGUF tensor name from a short name + layer index.

    The C++ loader's `tn(LLM_TENSOR_X, "weight", il)` produces
    `blk.<il>.<X>.weight` for both 2D and 3D tensors, so we always append
    the suffix. e.g. for `ffn_gate_up_exps` + suffix `weight` →
    `blk.0.ffn_gate_up_exps.weight`, which the loader expects.
    """
    if short in TOP_LEVEL_F16.values():
        return short  # top-level, no layer prefix
    return f"blk.{il}.{short}.{suffix}"


# ═══════════════════════════════════════════════════════════════════════
# Main quantization loop
# ═══════════════════════════════════════════════════════════════════════

def load_calibration_policy(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as source:
        policy = json.load(source)
    if policy.get("schema") not in {
        "llama.dflash.calibration-policy.v1",
        "llama.speculative.calibration-policy.v1",
    }:
        raise ValueError(f"{path}: unsupported calibration policy schema")
    return policy


HESSIAN_TRACE_SCHEMA = "llama.tessera.hessian-trace-policy.v1"


def load_hessian_trace_policy(path: Optional[str]) -> Optional[dict]:
    """Load an L3 E5 hessian-trace policy produced by l3_hessian_trace.py.

    The policy inherits the speculative calibration-policy parent schema
    so the same root-schema check as ``load_calibration_policy`` applies.
    Returns None when ``path`` is empty.
    """
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as source:
        policy = json.load(source)
    if policy.get("schema") not in {
        "llama.dflash.calibration-policy.v1",
        "llama.speculative.calibration-policy.v1",
    }:
        raise ValueError(f"{path}: unsupported calibration policy schema")
    hessian = policy.get("hessian_trace")
    if not isinstance(hessian, dict):
        raise ValueError(
            f"{path}: missing hessian_trace sub-policy (expected schema {HESSIAN_TRACE_SCHEMA!r})"
        )
    if hessian.get("schema") != HESSIAN_TRACE_SCHEMA:
        raise ValueError(
            f"{path}: hessian_trace.schema must be {HESSIAN_TRACE_SCHEMA!r}, "
            f"got {hessian.get('schema')!r}"
        )
    return policy


def merge_hessian_trace_into_policy(
    calibration_policy: Optional[dict], trace_policy: Optional[dict]
) -> Optional[dict]:
    """Merge the per-tensor trace values into the in-memory calibration policy.

    The merge is additive: each tensor entry under
    ``trace_policy['hessian_trace']['tensors']`` adds a
    ``hessian_trace`` / ``hessian_trace_avg`` /
    ``hessian_trace_per_tile`` triplet to the corresponding tensor_family
    in the calibration policy. New tensor families are created when no
    existing entry matches the tensor name (substring match by default,
    exact match when ``exact`` is set). The merge keeps the parent
    speculative-calibration schema intact; downstream consumers
    (the L5 orchestrator, the quantizer's sensitivity scorer) can then
    read the trace values as a first-class field on each tensor.

    Returns the calibration policy (mutated in place) or None when no
    calibration policy is supplied. The trace-only path (no calibration
    policy) writes the trace policy back as-is so the standalone tool
    can drive the consumer side too.
    """
    if trace_policy is None:
        return calibration_policy
    hessian = trace_policy.get("hessian_trace", {})
    records = hessian.get("tensors", [])
    if calibration_policy is None:
        # No parent calibration policy: emit the trace policy verbatim
        # so the caller still gets a consumable document.
        return trace_policy
    families = calibration_policy.get("tensor_families", {})
    # Pre-compute the matching tensor name once per family for speed.
    matched_keys: set[str] = set()
    for record in records:
        name = record.get("name")
        if not name:
            continue
        per_tile = record.get("hessian_trace_per_tile", [])
        for key, entry in families.items():
            matches = entry.get("match", [])
            if not matches:
                continue
            exact = bool(entry.get("exact", False))
            ok = (
                name in matches
                if exact
                else any(fragment in name for fragment in matches)
            )
            if not ok:
                continue
            entry["hessian_trace"] = record.get("hessian_trace")
            entry["hessian_trace_avg"] = record.get("hessian_trace_avg")
            entry["hessian_trace_per_tile"] = list(per_tile)
            entry["hessian_trace_n_tiles"] = record.get("n_tiles")
            entry["hessian_trace_method"] = record.get("method", hessian.get("method"))
            matched_keys.add(key)
    # Tensors that did not match an existing family get their own entry
    # so the L5 orchestrator can rank them even when no AWQ / LRQ prior
    # exists for the same name.
    for record in records:
        name = record.get("name")
        if not name:
            continue
        entry_key = f"hessian:{name}"
        if entry_key in families:
            continue
        families[entry_key] = {
            "match": [name],
            "exact": True,
            "hessian_trace": record.get("hessian_trace"),
            "hessian_trace_avg": record.get("hessian_trace_avg"),
            "hessian_trace_per_tile": list(record.get("hessian_trace_per_tile", [])),
            "hessian_trace_n_tiles": record.get("n_tiles"),
            "hessian_trace_method": record.get("method", hessian.get("method")),
        }
        matched_keys.add(entry_key)
    calibration_policy["tensor_families"] = families
    calibration_policy["hessian_trace"] = hessian
    return calibration_policy


def lrq_policy_for(
    policy: Optional[dict], tensor_name: str
) -> Optional[Tuple[int, np.ndarray, np.ndarray, str]]:
    """Return (rank, U, V, aggregation) for `tensor_name` if the policy has
    an LRQ entry that matches it, or None when no LRQ data applies.

    The matching rule mirrors ``tensor_policy``: the entry's ``match`` list
    is checked as a substring unless ``exact`` is set. The caller should
    prefer this over AWQ when it returns non-None.

    When the policy carries per-entry ``model_role`` tags (the Phase
    16 unified policy shape), entries are filtered by the tensor's
    inferred role: a ``trunk`` tensor prefers ``model_role=trunk``
    entries and falls back to ``shared_embd`` entries. The legacy
    single-model path (no ``model_role`` metadata) is preserved
    exactly: the first matching entry wins.
    """
    if policy is None:
        return None
    families = policy.get("tensor_families", {})
    has_role_metadata = any(
        isinstance(entry, dict) and "model_role" in entry
        for entry in families.values()
    )
    tensor_role = _infer_tensor_role(tensor_name)
    selected = None
    selected_rank = (-1, -1, -1)
    for family in families.values():
        if "lrq_u" not in family or "lrq_v" not in family:
            continue
        matches = family.get("match", [])
        if not matches:
            continue
        exact = bool(family.get("exact", False))
        matched = (
            tensor_name in matches
            if exact
            else any(fragment in tensor_name for fragment in matches)
        )
        if not matched:
            continue
        rank = int(family.get("lrq_rank", 0))
        try:
            u = np.asarray(family["lrq_u"], dtype=np.float32)
            v = np.asarray(family["lrq_v"], dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if u.ndim != 2 or v.ndim != 2:
            continue
        if rank <= 0:
            rank = min(u.shape[0], v.shape[1])
        if u.shape[1] != rank or v.shape[0] != rank:
            # Malformed entry; skip rather than fail the whole quantize call.
            continue
        agg = str(family.get("lrq_input_scale_agg", "mean"))
        if agg not in ("mean", "rms"):
            agg = "mean"
        if not has_role_metadata:
            # Legacy path: return the first valid match.
            return rank, u, v, agg
        entry_role = family.get("model_role")
        if tensor_role is not None and entry_role == tensor_role:
            role_score = 2
        elif entry_role == UNIFIED_SHARED_EMBD_ROLE:
            role_score = 1
        else:
            role_score = 0
        rank_tuple = (
            role_score,
            int(exact),
            max(len(fragment) for fragment in matches),
        )
        if rank_tuple > selected_rank:
            selected = (rank, u, v, agg)
            selected_rank = rank_tuple
    return selected


def load_pe_qat_policy(path: Optional[str]) -> Optional[dict]:
    """Load a ``llama.tessera.pe-qat-policy.v1`` JSON from disk.

    Returns None if `path` is None/empty.  Raises if the path is set but
    the file is unreadable, the JSON is malformed, or the schema is not
    the PE-QAT schema.  Unlike the calibration policy (which has a
    family/instance split), the PE-QAT policy is consumed wholesale by
    ``pe_qat_policy_for`` per tensor.
    """
    if not path:
        return None
    if not _PE_QAT_AVAILABLE:
        raise RuntimeError(
            "--pe-qat-policy requires tools.tessera.pe_qat; the module "
            "failed to import on this checkout"
        )
    with open(path, "r", encoding="utf-8") as source:
        policy = json.load(source)
    if policy.get("schema") != PE_QAT_POLICY_SCHEMA:
        raise ValueError(
            f"{path}: expected schema {PE_QAT_POLICY_SCHEMA!r}, "
            f"got {policy.get('schema')!r}"
        )
    return policy


def pe_qat_policy_for(
    policy: Optional[dict], tensor_name: str
) -> Optional[dict]:
    """Return the PE-QAT entry for `tensor_name`, or None.

    Delegates to ``tools.tessera.pe_qat._pe_qat_policy_for`` so the
    multi-tensor / single-tensor layout is consistent between the
    trainer, the demo, and the quantizer.  Kept as a thin wrapper so
    the rest of quantize_v3.py does not need to know about the
    internal helper.
    """
    if policy is None:
        return None
    return _pe_qat_policy_for(policy, tensor_name)


# Role names that the unified calibration driver stamps on every
# per-tensor entry. The Python consumer (this file) uses them to
# route per-tensor qtype to the right lane when a unified policy
# carries entries for trunk + dflash + dspark + mtp_nextn +
# shared_embd. The constants are mirrored from
# tools/tessera/per_tensor_calibrate.py::MODEL_ROLES so this module
# has no runtime import dependency.
UNIFIED_MODEL_ROLES = ("trunk", "dflash", "dspark", "mtp_nextn", "shared_embd")
UNIFIED_SHARED_EMBD_ROLE = "shared_embd"


def _infer_tensor_role(tensor_name: str) -> Optional[str]:
    """Best-effort role inference for a tensor name.

    The unified arch mixes tensors from multiple components in a
    single safetensors file. The role of a tensor is determined by
    the naming convention the loader uses, not by the model arch
    string. The patterns are conservative: anything that does not
    match returns ``None``, which signals "no role hint" and the
    consumer falls back to the legacy single-arch behaviour.

    The patterns are ordered most-specific first so the inference
    does not need to enumerate every MTP / DSparc / dflash variant
    explicitly. New patterns can be added here when the loader
    grows a new component.
    """
    if not tensor_name:
        return None
    # Embedding / output are always shared between trunk and
    # dflash (the drafter's tokenizer + output head are aliased to
    # the trunk's). The shared entry is the worst-of calibration
    # baked at calibration time; the GGUF writer still picks the
    # worst-of-the-two per the Phase 16 spec.
    if tensor_name == "token_embd.weight" or tensor_name.startswith("token_embd."):
        return UNIFIED_SHARED_EMBD_ROLE
    if tensor_name == "output.weight" or tensor_name.startswith("output."):
        return UNIFIED_SHARED_EMBD_ROLE
    if tensor_name.startswith("dflash.") or tensor_name.startswith("dflash_"):
        return "dflash"
    # DSpark heads (DeepSeek-style Markov / sparse heads). The
    # naming convention is `markov_*` and `head_*` (the latter is
    # the routed-expert head); both share the dspark calibration
    # lane.
    if tensor_name.startswith("markov_") or tensor_name.startswith("head_"):
        return "dspark"
    # MTP (Multi-Token Prediction) nextn blocks. The nextn tensors
    # live under `blk.<N>.nextn.*` and the MTP-specific shared
    # tensors live under `nextn.*`.
    if ".nextn." in tensor_name or tensor_name.startswith("nextn."):
        return "mtp_nextn"
    # Everything else (blk.<N>.* without the nextn branch, and any
    # tensor not matching a special pattern) is the trunk.
    if tensor_name.startswith("blk."):
        return "trunk"
    return None


def _select_family_for_role(
    families: dict, tensor_name: str, tensor_role: Optional[str]
) -> Optional[dict]:
    """Pick the best per-tensor entry, honouring ``model_role`` when present.

    Selection rules:

    1. **No role metadata**: any entry whose ``match`` field hits
       is a candidate (legacy single-model behaviour). The
       highest-ranked match wins (exact > substring, longer
       substring > shorter).
    2. **Role metadata present**: prefer an entry whose
       ``model_role`` matches the tensor's inferred role. If no
       role-specific entry matches, fall back to ``shared_embd``
       entries (the worst-of-trunk+dflash shared lane). If neither
       matches, no entry is returned and the caller falls back to
       the global defaults.

    The function preserves the existing match-ranking logic so
    pre-Phase-16 policies (no ``model_role`` field) keep behaving
    exactly as they did before.
    """
    has_role_metadata = any(
        isinstance(entry, dict) and "model_role" in entry
        for entry in families.values()
    )
    if not has_role_metadata:
        # Legacy path: any match is a candidate.
        selected = None
        selected_rank = (-1, -1)
        for family in families.values():
            matches = family.get("match", [])
            if not matches:
                continue
            exact = bool(family.get("exact", False))
            matched = (
                tensor_name in matches
                if exact
                else any(fragment in tensor_name for fragment in matches)
            )
            if not matched:
                continue
            rank = (int(exact), max(len(fragment) for fragment in matches))
            if rank > selected_rank:
                selected, selected_rank = family, rank
        return selected

    # Role-aware path: collect every matching entry, score by
    # (role_match, exact, longest_match_fragment). shared_embd is
    # the cross-arch fallback when no role-specific entry exists.
    selected = None
    selected_rank = (-1, -1, -1)
    for family in families.values():
        matches = family.get("match", [])
        if not matches:
            continue
        exact = bool(family.get("exact", False))
        matched = (
            tensor_name in matches
            if exact
            else any(fragment in tensor_name for fragment in matches)
        )
        if not matched:
            continue
        entry_role = family.get("model_role")
        # role_score: 2 = exact role match, 1 = shared_embd
        # fallback, 0 = other (still considered, but ranked
        # below). The "other" rank keeps the function total so
        # pre-Phase-16 entries without model_role still get
        # considered as a last resort.
        if tensor_role is not None and entry_role == tensor_role:
            role_score = 2
        elif entry_role == UNIFIED_SHARED_EMBD_ROLE:
            role_score = 1
        else:
            role_score = 0
        rank = (
            role_score,
            int(exact),
            max(len(fragment) for fragment in matches),
        )
        if rank > selected_rank:
            selected, selected_rank = family, rank
    return selected


def tensor_policy(policy: Optional[dict], tensor_name: str,
                  default_fraction: float, default_alpha: Optional[float]) -> Tuple[float, Optional[float], float, bool, float]:
    """Return (outlier_fraction, awq_alpha, awq_clip, exact, ternary_threshold)
    for `tensor_name` given the per-tensor calibration `policy`.

    `ternary_threshold` is a multiplier on the per-row mean(|W|) used as the
    {-1, 0, +1} cutoff. Default 1.0 = legacy tessera behaviour. Set to a
    different value (typically in [0.5, 2.0]) by per-tensor calibration
    produced via tools/tessera/per_tensor_calibrate.py.

    When the policy contains per-entry ``model_role`` tags (the
    Phase 16 unified policy shape), this function filters entries
    by the tensor's inferred role: prefer the role-specific entry
    (e.g. ``trunk`` for ``blk.0.attn_q.weight``) and fall back to
    ``shared_embd`` for cross-arch tensors (``token_embd``,
    ``output``). The legacy single-model path (no ``model_role``
    metadata) is preserved exactly: the highest-ranked match wins.
    """
    if policy is None:
        return default_fraction, default_alpha, 1.0, False, 1.0
    tensor_role = _infer_tensor_role(tensor_name)
    selected = _select_family_for_role(
        policy.get("tensor_families", {}), tensor_name, tensor_role
    )
    if selected is not None:
        exact = bool(selected.get("exact", False))
        fraction = 1.0 if exact else float(selected.get("outlier_fraction", default_fraction))
        policy_alpha = selected.get("awq_alpha", default_alpha)
        alpha = None if policy_alpha == "auto" else float(policy_alpha)
        clip = float(selected.get("awq_clip", 1.0))
        threshold = float(selected.get("ternary_threshold", 1.0))
        return fraction, alpha, clip, exact, threshold
    return default_fraction, default_alpha, 1.0, False, 1.0


def expert_policy_values(
    policy: Optional[dict],
    tensor_name: str,
    n_experts: int,
    default_fraction: float,
    default_alpha: Optional[float],
    default_clip: float,
) -> Tuple[np.ndarray, List[Optional[float]], np.ndarray]:
    fractions = np.full(n_experts, default_fraction, dtype=np.float64)
    alphas = [default_alpha] * n_experts
    clips = np.full(n_experts, default_clip, dtype=np.float64)
    if policy is None:
        return fractions, alphas, clips
    match = re.match(r"blk\.(\d+)\.", tensor_name)
    if match is None:
        return fractions, alphas, clips
    layer = (
        policy.get("moe_residual_allocation", {})
        .get("layers", {})
        .get(match.group(1), {})
    )
    tensor_key = tensor_name.removeprefix(f"blk.{match.group(1)}.")
    entries = (
        layer.get("tensors", {}).get(tensor_key, {}).get("experts")
        or layer.get("experts", {})
    )
    for expert, entry in entries.items():
        index = int(expert)
        if 0 <= index < n_experts:
            fractions[index] = float(entry["outlier_fraction"])
            if "awq_alpha" in entry:
                alphas[index] = float(entry["awq_alpha"])
            if "awq_clip" in entry:
                clips[index] = float(entry["awq_clip"])
    if np.any((fractions < 0.0001) | (fractions > 0.05)):
        raise ValueError(f"{tensor_name}: MoE residual fraction outside Tessera range")
    return fractions, alphas, clips


def main():
    def parse_awq_alpha(value: str) -> Optional[float]:
        if value.lower() == "auto":
            return None
        alpha = float(value)
        if not 0.0 <= alpha <= 1.0:
            raise argparse.ArgumentTypeError("AWQ alpha must be 'auto' or a number in [0, 1]")
        return alpha

    ap = argparse.ArgumentParser(description="Tessera TSQ-T640 quantization with optional AWQ calibration")
    ap.add_argument("--model-dir", required=True, help="HuggingFace model dir with safetensors")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument(
        "--metadata-from",
        required=True,
        help="Loadable canonical GGUF whose model and tokenizer metadata will be copied",
    )
    ap.add_argument(
        "--architecture",
        choices=(
            "auto",
            "qwen35",
            "qwen35moe",
            "gemma4",
            "gemma4-assistant",
            "dflash",
            "deepseek4",
            "glm4moe",
            "glm-dsa",
            "kimi-linear",
        ),
        default="auto",
        help="Model architecture; default reads general.architecture from --metadata-from",
    )
    ap.add_argument(
        "--vision-from",
        default=None,
        help="Canonical mmproj GGUF to embed in the same output file",
    )
    ap.add_argument(
        "--gguf-py",
        default="/Users/user/Developer/GitHub/llama.cpp/gguf-py",
        help="Path to llama.cpp/gguf-py",
    )
    ap.add_argument("--imatrix", default=None, help="Path to imatrix .npz or .gguf file")
    ap.add_argument(
        "--imatrix-merge",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Additional imatrix to merge with --imatrix via geometric mean. "
            "Repeat for multiple. Currently only the geometric-mean policy "
            "is implemented; the merged importance is "
            "(a * b * c * ...)^(1/N) per tensor, where N counts only the "
            "sources that actually measured that tensor (missing keys are "
            "skipped, not treated as 1.0)."
        ),
    )
    ap.add_argument(
        "--imatrix-merge-policy",
        default="geometric-mean",
        choices=("geometric-mean",),
        help="Merge policy when --imatrix-merge is supplied.",
    )
    ap.add_argument(
        "--adaptive-moe-result",
        default=None,
        help=(
            "Result JSON from tools/tessera/moe-calibrate.py; resolves and "
            "validates the final cumulative routed-expert imatrix"
        ),
    )
    ap.add_argument("--calibration-policy", default=None,
                    help="Acceptance-aware DFlash calibration policy JSON")
    ap.add_argument(
        "--hessian-trace-policy",
        default=None,
        help=(
            "Hessian-trace calibration policy (llama.tessera.hessian-trace-policy.v1) "
            "produced by tools/tessera/l3_hessian_trace.py. The per-tensor "
            "trace and per-tile trace values are merged into the in-memory "
            "calibration_policy so downstream consumers (the L5 orchestrator, "
            "the quantizer's sensitivity scorer) can read them. The parent "
            "schema is llama.speculative.calibration-policy.v1 so this flag is "
            "a no-op consumer-side schema-wise; it is the L3 E5 unlock for "
            "first-class Hessian-trace sensitivity."
        ),
    )
    ap.add_argument(
        "--tessera-epoch-receipt",
        default=None,
        help="Epoch JSON frozen before calibration and embedded in the output GGUF",
    )
    ap.add_argument(
        "--tessera-source-receipt",
        default=None,
        help="Combined BF16 source-epoch receipt embedded in the output GGUF",
    )
    ap.add_argument("--awq-alpha", type=parse_awq_alpha, default=None,
                    help="AWQ scaling strength in [0, 1], or 'auto' to search per matrix. Default: auto.")
    ap.add_argument("--outlier-frac", type=float, default=0.005, help="Fraction of weights as outliers (default 0.005 = 0.5%%)")
    ap.add_argument("--default-outlier-frac", type=float, default=0.001, help="Outlier fraction for pass-through / sensitive weights (default 0.001)")
    ap.add_argument(
        "--force-gemma4-n-swa",
        type=int,
        default=512,
        help=(
            "Override gemma4.attention.sliding_window (and the MTP variant) "
            "to this value at export time. Set to 0 to disable the override. "
            "Default 512 matches the official gemma 4 12B config and repairs "
            "the 1024-instead-of-512 bug Google shipped in the QAT gguf."
        ),
    )
    ap.add_argument(
        "--range-selection",
        choices=("legacy", "imatrix-mse"),
        default="legacy",
        help=(
            "How to choose the ternary code's per-row range. 'legacy' uses the "
            "current max(|W|)/127 scheme. 'imatrix-mse' (vllm imatrix_mse "
            "analogue) does an MSE grid search per row, weighting candidate "
            "ranges by the per-channel imatrix importance, then takes the lane "
            "scale as a percentile of the per-position scale. Implies --awq-"
            "alpha=0 because importance is now baked into the range, not the "
            "pre-quantization scaling."
        ),
    )
    ap.add_argument(
        "--imatrix-mse-norm",
        type=float,
        default=3.0,
        help="p exponent for imatrix_mse importance-weighted error: err = sum(importance * |Q(w)-w|^p)",
    )
    ap.add_argument(
        "--imatrix-mse-grid",
        type=int,
        default=20,
        help="Number of grid steps for imatrix_mse shrink search per row.",
    )
    ap.add_argument(
        "--imatrix-mse-maxshrink",
        type=float,
        default=0.95,
        help="Maximum shrink factor (as fraction of max(|W|)) considered by the imatrix_mse grid search.",
    )
    ap.add_argument(
        "--gemma4-sensitive-patterns",
        nargs="*",
        default=None,
        help=(
            "Substring patterns of canonical tensor names that must be forced "
            "to exact (outlier_frac=1.0) Tile640 encoding. Default for "
            "gemma4/gemma4-assistant covers QK-norm, post-norm, attention "
            "output projection, and FFN down projection."
        ),
    )
    ap.add_argument(
        "--gemma4-kld-threshold",
        type=float,
        default=0.0,
        help=(
            "If > 0, validate the quantized model by computing the per-block "
            "Kullback-Leibler divergence between the FP16 reference block and "
            "the quantized block on a calibration set, and refuse to write the "
            "GGUF if any block exceeds this threshold. The shadow calibration "
            "already does provisional reconstruction; this adds a downstream "
            "accept/reject gate. Disabled by default."
        ),
    )
    ap.add_argument(
        "--awq-search-target",
        choices=("per-row", "layer-output"),
        default="per-row",
        help=(
            "Optimization target for the AWQ alpha search. 'per-row' "
            "(default) minimizes importance-weighted per-row reconstruction "
            "error — the original tessera behavior. 'layer-output' minimizes "
            "||(W_q - W) · X||² on per-tensor X. When no calibration activations "
            "are loaded, the layer-output target uses a synthetic X with banded "
            "cross-channel correlation; for a diagonal-only X this is "
            "mathematically equivalent to per-row, so the gain shows up only "
            "with real per-layer snapshots. See --calibration-activations."
        ),
    )
    ap.add_argument(
        "--calibration-activations",
        default=None,
        help=(
            "Path to a .npz with one float32 (batch, in_dim) array per "
            "canonical tensor name. When set and --awq-search-target="
            "layer-output, these are used as the per-layer X in the AWQ "
            "search. Tensors without an entry fall back to the synthetic "
            "approximation. Generated by a separate calibration forward pass; "
            "not produced by the orchestrator today."
        ),
    )
    ap.add_argument(
        "--awq-synthetic-batch",
        type=int,
        default=32,
        help="Batch size for the synthetic X used by the layer-output AWQ search.",
    )
    ap.add_argument(
        "--awq-synthetic-correlation",
        type=float,
        default=0.25,
        help=(
            "Banded cross-channel correlation coefficient (in [0, 1)) for the "
            "synthetic X. 0.0 is the diagonal-only case (mathematically "
            "equivalent to per-row); 0.25 is the default and gives a "
            "mildly-correlated synthetic input that breaks the equivalence."
        ),
    )
    ap.add_argument("--spool-dir", default="/Volumes/Julian T7/tmp", help="Spool dir for SpooledTemporaryFile (default: /Volumes/Julian T7/tmp)")
    ap.add_argument("--keep-types", nargs="*", default=None, help="Force KEEP_F16 for these tensor names (substring match)")
    ap.add_argument(
        "--inventory-only",
        action="store_true",
        help="Validate metadata and tensor mapping without reading or quantizing weights",
    )
    ap.add_argument(
        "--permute-channels",
        "--champq",
        action="store_true",
        dest="permute_channels",
        help=(
            "Enable CHAMP-Q: permute the input channels of every Tile640 "
            "weight by L2 activation magnitude (or per-row weight L2 if no "
            "imatrix is loaded) before quantizing, then fold the inverse "
            "permutation into the output. The output GGUF is in original "
            "channel order and is bit-compatible with the non-CHAMP-Q path. "
            "Calibration-time only; no runtime cost. See "
            "tools/tessera/champq_permute.py for the algorithm."
        ),
    )
    ap.add_argument(
        "--champq-policy-out",
        default=None,
        help=(
            "Path to write the per-tensor CHAMP-Q permutation policy JSON. "
            "Default: derived from --output (champq-policy.json next to the "
            "GGUF). Set to an empty string to disable the policy file. The "
            "policy is for debugging / A-B comparison; the GGUF itself does "
            "not need it because the output is already in original order."
        ),
    )
    ap.add_argument(
        "--septq",
        action="store_true",
        help=(
            "Use the SEPTQ (KDD 2025) two-step PTQ method instead of the "
            "standard tessera flow. SEPTQ computes a static global importance "
            "mask from a Hessian-based criterion, then quantizes only the "
            "top-k percent elements; the rest are kept at full precision. "
            "See --septq-hessian-mode for the diagonal / banded choice and "
            "--septq-ratio for the quantize fraction. Requires --imatrix for "
            "the Hessian-diagonal importance score; --septq-hessian-mode "
            "banded additionally requires full calibration activations "
            "(currently only the synthetic / A/B harness path supplies them)."
        ),
    )
    ap.add_argument(
        "--septq-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of elements to quantize under SEPTQ. 1.0 = all elements "
            "(equivalent to RTN); 0.5 = half quantized, half kept full precision. "
            "Default 0.5. Only used when --septq is set."
        ),
    )
    ap.add_argument(
        "--septq-iterations",
        type=int,
        default=1,
        help=(
            "Number of column-by-column passes for SEPTQ. The mask is fixed "
            "at the first iteration (static global mask per the paper); "
            "subsequent passes re-ternarize with the same mask. Default 1. "
            "Only used when --septq is set."
        ),
    )
    ap.add_argument(
        "--septq-hessian-mode",
        choices=("diagonal", "banded"),
        default="banded",
        help=(
            "SEPTQ Hessian inverse mode. 'diagonal' (original behaviour) uses "
            "act_scales[j]^2 as the diagonal of H and skips the cross-column "
            "update. 'banded' (default) uses the full H = X^T X / n from the "
            "calibration activations and applies a banded GPTQ-style update "
            "with bandwidth --septq-hessian-bandwidth. The main quantize "
            "script does not have the raw calibration activations and "
            "silently falls back to 'diagonal'; the A/B harness can use the "
            "banded mode when synthetic activations are available."
        ),
    )
    ap.add_argument(
        "--septq-hessian-bandwidth",
        type=int,
        default=32,
        help=(
            "Bandwidth of the banded Cholesky used by the SEPTQ cross-column "
            "update. Default 32. Only used when --septq and "
            "--septq-hessian-mode banded."
        ),
    )
    ap.add_argument(
        "--septq-importance-weight",
        choices=("quant_error_h", "inv_abs_w", "inv_cdf", "hybrid"),
        default="quant_error_h",
        help=(
            "SEPTQ importance score. 'quant_error_h' (default, v1 behaviour) "
            "uses (W - Q(W))^2 * h_diag. 'inv_abs_w' divides by (|W| + eps) "
            "to downweight heavy-tail outliers so the mask focuses on the "
            "bulk. 'inv_cdf' uses 1 - CDF_per_row(|W|) as the weight "
            "(most aggressive). 'hybrid' adds lambda * h_diag / (|W| + eps) "
            "to the original score; --septq-importance-lambda sets lambda. "
            "Only used when --septq is set."
        ),
    )
    ap.add_argument(
        "--septq-importance-lambda",
        type=float,
        default=0.0,
        help=(
            "Lambda for the 'hybrid' importance mode (default 0 = original). "
            "Only used when --septq and --septq-importance-weight hybrid."
        ),
    )
    ap.add_argument(
        "--pe-qat-policy",
        default=None,
        help=(
            "Path to a llama.tessera.pe-qat-policy.v1 JSON produced by "
            "tools/tessera/pe_qat_demo.py (or a production orchestrator). "
            "When set, the trained LoRA delta is merged into each dense "
            "weight and the per-input-channel SmoothQuant factors are "
            "applied before quantization. PE-QAT is checked first in the "
            "policy precedence order (before LRQ / SEPTQ / imatrix-mse) "
            "because it carries the trained LoRA. CHAMP-Q permutation is "
            "skipped on PE-QAT-adjusted weights -- the per-channel s was "
            "trained against the original channel order. Only the 2D "
            "weight path is currently wired; the 3D expert path falls "
            "through to the default quantizer."
        ),
    )
    args = ap.parse_args()
    calibration_policy = load_calibration_policy(args.calibration_policy)
    if args.hessian_trace_policy:
        trace_policy = load_hessian_trace_policy(args.hessian_trace_policy)
        if trace_policy is not None:
            n_tensors = len(trace_policy.get("hessian_trace", {}).get("tensors", []))
            method = trace_policy.get("hessian_trace", {}).get("method", "unknown")
            print(
                f"  hessian-trace: merging {n_tensors} tensor records "
                f"(method={method}) from {args.hessian_trace_policy}",
                file=sys.stderr,
            )
            calibration_policy = merge_hessian_trace_into_policy(
                calibration_policy, trace_policy
            )
    pe_qat_policy = load_pe_qat_policy(args.pe_qat_policy)
    epoch_receipt = (
        json.loads(Path(args.tessera_epoch_receipt).read_text(encoding="utf-8"))
        if args.tessera_epoch_receipt
        else None
    )
    source_receipt = (
        json.loads(Path(args.tessera_source_receipt).read_text(encoding="utf-8"))
        if args.tessera_source_receipt
        else None
    )

    # Wire the AWQ layer-output search knobs. These are read by
    # awq_scale_search() and apply to every tensor the search visits.
    global AWQ_SEARCH_TARGET, AWQ_SYNTHETIC_BATCH
    global AWQ_SYNTHETIC_CORRELATION, AWQ_SYNTHETIC_SEED
    AWQ_SEARCH_TARGET = args.awq_search_target
    AWQ_SYNTHETIC_BATCH = args.awq_synthetic_batch
    AWQ_SYNTHETIC_CORRELATION = args.awq_synthetic_correlation
    if args.calibration_activations:
        load_calibration_activations(args.calibration_activations)
        if args.awq_search_target != "layer-output":
            print(
                "WARN: --calibration-activations provided but --awq-search-target "
                f"is '{args.awq_search_target}'; the activations will be loaded "
                "but unused. Set --awq-search-target=layer-output to consume them.",
                file=sys.stderr,
            )

    # Make the caller-selected gguf-py available before loading a GGUF
    # importance matrix. Previously this import happened only after imatrix
    # loading, so a valid --gguf-py still failed with ModuleNotFoundError.
    gguf, GGUFReader, GGUFValueType, GGUFWriter = import_gguf(args.gguf_py)

    if args.imatrix and args.adaptive_moe_result:
        raise ValueError("--imatrix and --adaptive-moe-result are mutually exclusive")
    if args.adaptive_moe_result:
        adaptive_result = json.loads(
            Path(args.adaptive_moe_result).read_text(encoding="utf-8")
        )
        if (
            adaptive_result.get("schema")
            != "llama.tessera.moe-calibration-result.v1"
            or adaptive_result.get("complete") is not True
        ):
            raise ValueError("adaptive MoE calibration result is incomplete or unsupported")
        args.imatrix = adaptive_result.get("imatrix")
        if not args.imatrix or not Path(args.imatrix).is_file():
            raise ValueError("adaptive MoE calibration result does not resolve to an imatrix")
        print(
            "  adaptive MoE calibration: "
            f"{adaptive_result.get('samples')} samples, "
            f"{adaptive_result.get('rounds')} rounds, "
            f"stop={adaptive_result.get('stop_reason')}",
            file=sys.stderr,
        )

    # Load imatrix if provided
    imatrix = None
    if args.imatrix:
        imatrix = load_imatrix(args.imatrix)
        # Show a few example entries
        sample_keys = list(imatrix.keys())[:5]
        print(f"  imatrix sample keys: {sample_keys}", file=sys.stderr)
        for k in sample_keys[:3]:
            v = imatrix[k]
            print(f"    {k}: shape={v.shape}, mean={v.mean():.4f}, max={v.max():.4f}", file=sys.stderr)
        # Optional: merge with additional imatrices via geometric mean.
        if args.imatrix_merge:
            if args.imatrix_merge_policy != "geometric-mean":
                raise ValueError(
                    f"unsupported imatrix-merge-policy: {args.imatrix_merge_policy}"
                )
            others = [load_imatrix(p) for p in args.imatrix_merge]
            n_primary = len(imatrix)
            n_others = [len(o) for o in others]
            imatrix = merge_imatrix_geomean(imatrix, *others)
            print(
                f"  imatrix-merge: primary={n_primary} entries; "
                f"merge sources={n_others}; "
                f"policy=geometric-mean; "
                f"result={len(imatrix)} entries (key set from primary)",
                file=sys.stderr,
            )
            # Sanity check: log a few key shapes / magnitudes so the user can
            # tell the merge actually took effect (vs silently no-op).
            for k in list(imatrix.keys())[:3]:
                v = imatrix[k]
                print(
                    f"    merged {k}: shape={v.shape}, mean={v.mean():.4f}, "
                    f"max={v.max():.4f}",
                    file=sys.stderr,
                )

    # Setup spool dir
    os.makedirs(args.spool_dir, exist_ok=True)
    os.environ["TMPDIR"] = args.spool_dir

    # Load the authoritative GGUF name map and metadata before inspecting the
    # checkpoint. The HF directory also contains a vision tower; the text GGUF
    # tensor inventory tells us exactly which tensors belong in this model.
    metadata_reader = GGUFReader(args.metadata_from, "r")
    architecture_field = metadata_reader.get_field("general.architecture")
    if architecture_field is None:
        raise ValueError(f"{args.metadata_from}: missing general.architecture")
    architecture_name = architecture_field.contents()
    if args.architecture != "auto" and architecture_name != args.architecture:
        raise ValueError(
            f"{args.metadata_from}: expected {args.architecture} metadata, found {architecture_name!r}"
        )
    arch_enums = {
        "qwen35": gguf.MODEL_ARCH.QWEN35,
        "qwen35moe": gguf.MODEL_ARCH.QWEN35MOE,
        "gemma4": gguf.MODEL_ARCH.GEMMA4,
        "gemma4-assistant": gguf.MODEL_ARCH.GEMMA4_ASSISTANT,
        "dflash": gguf.MODEL_ARCH.DFLASH,
        "deepseek4": gguf.MODEL_ARCH.DEEPSEEK4,
        "glm4moe": gguf.MODEL_ARCH.GLM4_MOE,
        "glm-dsa": gguf.MODEL_ARCH.GLM_DSA,
        "kimi-linear": gguf.MODEL_ARCH.KIMI_LINEAR,
    }
    if architecture_name not in arch_enums:
        raise ValueError(f"unsupported Tessera architecture {architecture_name!r}")
    block_field = metadata_reader.get_field(f"{architecture_name}.block_count")
    if block_field is None:
        raise ValueError(f"{args.metadata_from}: missing {architecture_name}.block_count")
    total_block_count = int(block_field.contents())
    nextn_field = metadata_reader.get_field(f"{architecture_name}.nextn_predict_layers")
    nextn_count = int(nextn_field.contents()) if nextn_field is not None else 0
    base_block_count = total_block_count - nextn_count
    tensor_map = gguf.get_tensor_name_map(arch_enums[architecture_name], total_block_count)
    source_tensor_names = {tensor.name for tensor in metadata_reader.tensors}
    has_embedded_components = (
        metadata_reader.get_field("mtp.component.present") is not None
        or any(name.startswith(("mtp.", "v.", "mm.")) for name in source_tensor_names)
    )

    # Find safetensors shards
    import safetensors
    shard_files = sorted(Path(args.model_dir).glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no .safetensors files in {args.model_dir}")
    print(f"Found {len(shard_files)} safetensors shards", file=sys.stderr)

    # Build inventory: walk all shards, collect (hf_name, shape, dtype) per tensor
    inventory: List[Tuple[str, Tuple[int, ...], str, str, int, Optional[str]]] = []
    replaced_source_tensor_names: set[str] = set()
    skipped_non_text = 0
    has_multimodal_source = False
    source_classifications: Dict[str, int] = {}
    unclassified_source_names: List[str] = []
    def classify_source(kind: str) -> None:
        source_classifications[kind] = source_classifications.get(kind, 0) + 1

    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                if key.startswith((
                    "model.visual.",
                    "visual.",
                    "vision_tower.",
                    "model.vision_tower.",
                    "audio_tower.",
                    "model.audio_tower.",
                    "mm_projector.",
                    "model.mm_projector.",
                    "model.embed_vision.",
                    "model.embed_audio.",
                    "model.vision_embedder.",
                    "model.audio_embedder.",
                )):
                    has_multimodal_source = True
                normalized = normalize_hf_tensor_name(key, base_block_count)
                if normalized is None:
                    classify_source("external-multimodal")
                    skipped_non_text += 1
                    continue
                mapped = tensor_map.get_name(
                    normalized, try_suffixes=(".weight", ".bias", ".scale")
                )
                # Some fused expert arrays are stored in HF without a trailing
                # ".weight", while their canonical GGUF names include it.
                if mapped is not None and mapped not in source_tensor_names:
                    if f"{mapped}.weight" in source_tensor_names:
                        mapped = f"{mapped}.weight"
                    elif f"{mapped}.bias" in source_tensor_names:
                        mapped = f"{mapped}.bias"
                    elif re.fullmatch(r"blk\.\d+\.ffn_gate_up_exps", mapped):
                        # Qwen3.6 checkpoints store gate+up fused. The stock
                        # converter splits this into two GGUF tensors, while
                        # the Tile640 loader intentionally consumes the fused
                        # expert tensor directly.
                        mapped = f"{mapped}.weight"
                is_fused_gate_up = (
                    mapped is not None and
                    re.fullmatch(r"blk\.\d+\.ffn_gate_up_exps\.weight", mapped) is not None
                )
                if mapped is None or (mapped not in source_tensor_names and not is_fused_gate_up):
                    if mapped is None:
                        classify_source("unclassified")
                        unclassified_source_names.append(key)
                    else:
                        classify_source("canonical-converter-transform")
                    skipped_non_text += 1
                    continue
                classify_source("direct-canonical")
                layer_idx, short = mapped_tensor_parts(mapped)
                top_level_direct = {
                    "token_embd.weight": "token_embd",
                    "per_layer_token_embd.weight": "per_layer_token_embd",
                    "per_layer_model_proj.weight": "per_layer_model_proj",
                }
                if mapped in top_level_direct:
                    layer_idx, short = -2, top_level_direct[mapped]
                if layer_idx == -1 or short not in QUANT_2D_SHORT | QUANT_3D_SHORT:
                    # Untouched tensors are copied from the known-good GGUF
                    # below. This preserves converter-specific transforms.
                    continue
                meta = f.get_slice(key).get_shape()
                dt = str(f.get_slice(key).get_dtype())
                inventory.append((key, tuple(meta), dt, mapped, layer_idx, short))
                if is_fused_gate_up:
                    bid = mapped.split(".")[1]
                    replaced_source_tensor_names.update({
                        f"blk.{bid}.ffn_gate_exps.weight",
                        f"blk.{bid}.ffn_up_exps.weight",
                    })
                else:
                    replaced_source_tensor_names.add(mapped)
    print(
        f"Tessera T640 tensors in inventory: {len(inventory)} "
        f"(deferred {skipped_non_text} external/converter-managed tensors; "
        f"replacing {len(replaced_source_tensor_names)} source tensors)",
        file=sys.stderr,
    )
    if unclassified_source_names:
        examples = unclassified_source_names[:12]
        raise ValueError(
            f"{architecture_name}: {len(unclassified_source_names)} source tensors "
            f"are not classified by the family manifest; examples: {examples}"
        )

    tessera_tools = (
        Path(args.gguf_py).expanduser().resolve().parent
        / "tools" / "tessera"
    )
    if str(tessera_tools) not in sys.path:
        sys.path.insert(0, str(tessera_tools))
    from family_coverage import build_coverage_receipt, compact_receipt_json

    canonical_tensors = [
        (tensor.name, tuple(int(value) for value in tensor.shape))
        for tensor in metadata_reader.tensors
    ]
    family_coverage_receipt = build_coverage_receipt(
        architecture_name,
        canonical_tensors,
        base_block_count,
        total_block_count,
        has_multimodal_source=has_multimodal_source,
        has_multimodal_payload=bool(args.vision_from),
        source_classifications=source_classifications,
    )
    print(
        "Tessera family coverage: "
        f"family={family_coverage_receipt['family']} "
        f"features={family_coverage_receipt['features']} "
        f"components={family_coverage_receipt['components']} "
        f"digest={family_coverage_receipt['tensor_manifest_sha256'][:16]}",
        file=sys.stderr,
    )
    if args.inventory_only:
        print("Inventory validation OK", file=sys.stderr)
        return

    # Write GGUF. A complete model/tokenizer header is required by llama.cpp;
    # general.architecture on its own is not a loadable model description.
    writer = GGUFWriter(args.output, arch=architecture_name, use_temp_file=True)
    copied_metadata = copy_gguf_metadata(
        writer, args.metadata_from, GGUFReader, GGUFValueType, architecture_name
    )
    print(
        f"Copied {copied_metadata} metadata entries from {args.metadata_from}",
        file=sys.stderr,
    )
    vision_reader = None
    if args.vision_from:
        vision_reader, vision_metadata = copy_embedded_mmproj_metadata(
            writer, args.vision_from, GGUFReader, GGUFValueType
        )
        print(
            f"Merged {vision_metadata} projector metadata entries from {args.vision_from}",
            file=sys.stderr,
        )
    add_tessera_metadata(
        writer,
        calibrated=imatrix is not None or calibration_policy is not None,
        unified=vision_reader is not None or nextn_count > 0 or has_embedded_components,
        unsloth_prior=bool(calibration_policy and calibration_policy.get("unsloth_bridge")),
        global_residual_budget=bool(
            calibration_policy
            and any(
                family.get("residual_allocation")
                for family in calibration_policy.get("evolution", {}).get("families", {}).values()
            )
        ),
        epoch_receipt=epoch_receipt,
        source_receipt=source_receipt,
        imatrix_paths=([args.imatrix] if args.imatrix else []) + list(args.imatrix_merge or []),
        imatrix_merge_policy=(args.imatrix_merge_policy if args.imatrix_merge else None),
    )
    # Record the new §5.1/§5.2/§5.4 knobs in the GGUF metadata so the output
    # is self-describing — a future audit can read this back without having
    # to infer the production flags from the absence of metadata.
    if architecture_name in ("gemma4", "gemma4-assistant"):
        writer.add_uint32(
            "tessera.gemma4.sliding_window_override",
            int(args.force_gemma4_n_swa),
        )
    if args.range_selection == "imatrix-mse":
        writer.add_string("tessera.range_selection", "imatrix-mse")
        writer.add_float32("tessera.imatrix_mse.norm", float(args.imatrix_mse_norm))
        writer.add_uint32("tessera.imatrix_mse.grid", int(args.imatrix_mse_grid))
        writer.add_float32("tessera.imatrix_mse.maxshrink", float(args.imatrix_mse_maxshrink))
    if args.septq:
        # Record the SEPTQ mode so a future audit can read the quantization
        # policy back from the GGUF without re-deriving it from the absence
        # of the standard tessera metadata. The ratio is the fraction of
        # elements quantized; (1 - ratio) is the residual fraction.
        writer.add_string("tessera.range_selection", "septq")
        writer.add_float32("tessera.septq.ratio", float(args.septq_ratio))
        writer.add_uint32("tessera.septq.iterations", int(args.septq_iterations))
        writer.add_string("tessera.septq.hessian_mode", str(args.septq_hessian_mode))
        writer.add_uint32("tessera.septq.hessian_bandwidth", int(args.septq_hessian_bandwidth))
        writer.add_string("tessera.septq.importance_weight", str(args.septq_importance_weight))
        writer.add_float32("tessera.septq.importance_lambda", float(args.septq_importance_lambda))
    writer.add_string("tessera.awq_search_target", args.awq_search_target)
    if args.calibration_activations:
        writer.add_string(
            "tessera.calibration_activations_source", args.calibration_activations
        )
    if args.awq_search_target == "layer-output":
        writer.add_uint32(
            "tessera.awq_synthetic_batch", int(args.awq_synthetic_batch)
        )
        writer.add_float32(
            "tessera.awq_synthetic_correlation", float(args.awq_synthetic_correlation)
        )
    if args.force_gemma4_n_swa > 0:
        apply_gemma4_metadata_overrides(
            writer,
            metadata_reader,
            architecture_name,
            args.force_gemma4_n_swa,
            GGUFValueType,
        )
    writer.add_string(
        "tessera.coverage.family",
        family_coverage_receipt["family"],
    )
    writer.add_string(
        "tessera.coverage.architecture",
        family_coverage_receipt["architecture"],
    )
    writer.add_string(
        "tessera.coverage.manifest_sha256",
        family_coverage_receipt["tensor_manifest_sha256"],
    )
    writer.add_string(
        "tessera.coverage.receipt",
        compact_receipt_json(family_coverage_receipt),
    )

    t0 = time.time()
    quant_2d = quant_3d = pass_count = 0
    imatrix_hits = imatrix_misses = 0
    alpha_counts: Dict[float, int] = {}

    # CHAMP-Q policy recording. Populated only when --permute-channels is
    # set and the helper module is importable. The default output path
    # sits next to the GGUF; an empty --champq-policy-out disables the
    # file. The policy is for debugging / A-B comparison; the GGUF does
    # not need it because the output is already in original channel
    # order.
    champq_policy: Optional["CHAMPQPolicy"] = None
    if args.permute_channels:
        if not _CHAMPQ_AVAILABLE:
            raise RuntimeError(
                "--permute-channels requires tools.tessera.champq_permute; "
                "the module failed to import on this checkout"
            )
        if args.champq_policy_out is None:
            output_dir = Path(args.output).expanduser().resolve().parent
            champq_policy_path = output_dir / "champq-policy.json"
        elif args.champq_policy_out == "":
            champq_policy_path = None
        else:
            champq_policy_path = Path(args.champq_policy_out).expanduser()
        if champq_policy_path is not None:
            champq_policy = CHAMPQPolicy()
            print(
                f"  CHAMP-Q: permutation policy will be written to "
                f"{champq_policy_path}",
                file=sys.stderr,
            )
        else:
            print(
                "  CHAMP-Q: enabled, permutation policy file disabled via "
                "--champq-policy-out ''",
                file=sys.stderr,
            )

    # PE-QAT policy summary. Loaded eagerly above; per-tensor lookups happen
    # inside the main loop where each tensor name is known. The schema and
    # the number of entries (multi-tensor: dict size; single-tensor: 1) are
    # printed here so a misconfigured policy is visible before any
    # quantization work.
    if pe_qat_policy is not None:
        n_entries = 1
        if isinstance(pe_qat_policy.get("tensors"), dict):
            n_entries = len(pe_qat_policy["tensors"])
        rank = pe_qat_policy.get("rank", "?")
        alpha = pe_qat_policy.get("alpha", "?")
        print(
            f"  PE-QAT: policy loaded, rank={rank} alpha={alpha} "
            f"entries={n_entries}; will merge LoRA + smooth before quantize",
            file=sys.stderr,
        )

    for done, (wname, shape, dtype, out_name, layer_idx, short) in enumerate(inventory):
        # Find which shard holds this tensor
        for shard in shard_files:
            with safetensors.safe_open(str(shard), framework="pt", device="cpu") as f:
                if wname in f.keys():
                    raw = f.get_tensor(wname)
                    break
        else:
            print(f"WARN: tensor {wname} not found in any shard, skipping", file=sys.stderr)
            continue

        arr = raw.float().numpy()

        # Every tensor admitted to the direct inventory must be a Tile640
        # matrix. All remaining canonical tensors are converted to evolved
        # components below; plain source-tensor passthrough is forbidden.
        is_quant_2d = layer_idx != -1 and short in QUANT_2D_SHORT
        is_quant_3d = layer_idx >= 0 and short in QUANT_3D_SHORT

        if is_quant_2d:
            assert arr.ndim == 2, f"expected 2D for {wname}, got {arr.shape}"
            out_dim, in_dim = arr.shape
            gguf_n = out_name
            act_scales = lookup_acts(wname, imatrix, gguf_n) if imatrix else None
            if imatrix is not None:
                if act_scales is None:
                    imatrix_misses += 1
                else:
                    imatrix_hits += 1
            # Preserve MoE routing order exactly: even small router error can
            # choose a different expert and amplify downstream error. This is
            # still an evolved Tile640 tensor, represented entirely by sparse
            # residual components rather than a conventional passthrough.
            direct_exact = short == "ffn_gate_inp"
            # gemma 4 sensitive tensors (QK-norm, post-norm, attention output,
            # FFN down) are forced to exact encoding because QK-norm removes
            # the 1/sqrt(d_k) dampening and amplifies precision error. The
            # user can override the pattern list via --gemma4-sensitive-patterns.
            gemma4_sensitive = (
                architecture_name in ("gemma4", "gemma4-assistant")
                and is_gemma4_sensitive_tensor(
                    out_name, args.gemma4_sensitive_patterns
                )
            )
            if gemma4_sensitive:
                direct_exact = True
            policy_frac, policy_alpha, policy_clip, policy_exact, policy_threshold = tensor_policy(
                calibration_policy, out_name, args.outlier_frac, args.awq_alpha
            )
            direct_exact = direct_exact or policy_exact
            lrq = lrq_policy_for(calibration_policy, out_name) if not direct_exact else None
            use_lrq = lrq is not None
            # PE-QAT is checked before LRQ: a trained LoRA is the most
            # authoritative source for this tensor. direct_exact tensors
            # (e.g. MoE gates) skip PE-QAT -- the weight is already kept
            # at full precision downstream so a merge would be wasted work.
            pe_qat_entry = (
                pe_qat_policy_for(pe_qat_policy, out_name) if not direct_exact else None
            )
            use_pe_qat = pe_qat_entry is not None
            use_imatrix_mse = (
                not use_lrq
                and not use_pe_qat
                and args.range_selection == "imatrix-mse"
                and not direct_exact
                and act_scales is not None
            )
            # CHAMP-Q permute setup. Hoisted before the if/elif so the same
            # permuted (arr, act_scales) feed all paths. LRQ and PE-QAT are
            # both authoritative and skip the permute: their policies already
            # encode the scaling for this tensor, and PE-QAT's per-channel s
            # was trained against the original channel order. direct_exact
            # tensors (all-zero ternary) also skip: channel ordering is
            # irrelevant for them.
            champq_perm: Optional[np.ndarray] = None
            if (
                not use_lrq
                and not use_pe_qat
                and args.permute_channels
                and not direct_exact
                and _CHAMPQ_AVAILABLE
                and in_dim > 1
            ):
                champq_perm = compute_champq_permutation(arr, act_scales)
                champq_inverse = invert_champq_permutation(champq_perm)
                arr = apply_champq_permutation(arr, champq_perm)
                act_scales = (
                    act_scales[champq_perm] if act_scales is not None else None
                )

            # SEPTQ is mutually exclusive with imatrix-mse (both consume the
            # imatrix differently) and is skipped for exact tensors. SEPTQ
            # benefits from the imatrix as the Hessian-diagonal proxy; without
            # one it falls back to uniform column importance (still a valid
            # mixed-precision scheme but loses the calibration signal).
            use_septq = (
                args.septq
                and not direct_exact
                and not use_imatrix_mse
            )

            if use_pe_qat:
                # PE-QAT-mode: the policy carries a trained LoRA + per-channel
                # SmoothQuant factors.  Both are merged into the weight before
                # quantization.  PE-QAT is checked first because the LoRA is
                # the most authoritative per-tensor adjustment available; LRQ
                # is a coarser rank-r scale, AWQ is a heuristic, and SEPTQ
                # / imatrix_mse only refine the range selection.  The clip
                # threshold c is not applied here -- it is consumed at
                # quantization time by the per-output-channel quantizer.
                arr = apply_pe_qat_to_weight(arr, pe_qat_policy, out_name)
                q = quantize_2d(
                    arr,
                    out_dim,
                    in_dim,
                    1.0 if direct_exact else policy_frac,
                    None if direct_exact else act_scales,
                    0.0 if direct_exact else policy_alpha,
                    1.0 if direct_exact else policy_clip,
                    tensor_name=out_name,
                    ternary_threshold=(1.0 if direct_exact else policy_threshold),
                )
                # SmoothQuant split: apply_pe_qat_to_weight multiplied the
                # weight by s, so the runtime must apply 1/s to the input
                # to preserve the matmul equivalence (W*s) @ (x/s) = W @ x.
                # Override the input_scale that quantize_2d set to ones.
                # direct_exact tensors carry no policy entry (and so no s),
                # so this is a no-op for them.
                pe_qat_s = pe_qat_entry.get("per_channel_smooth_s")
                if pe_qat_s is not None and not direct_exact:
                    s_arr = np.asarray(pe_qat_s, dtype=np.float32)
                    if s_arr.ndim == 1 and s_arr.shape[0] == in_dim:
                        q["input_scale"] = (
                            1.0 / np.maximum(s_arr, np.float32(1e-6))
                        ).astype(np.float32)
            elif use_lrq:
                # LRQ-mode: the policy carries a rank-r S = U @ V scale. The
                # AWQ and imatrix_mse paths are bypassed because the policy
                # is the authoritative source for this tensor.
                _lrq_rank, _lrq_u, _lrq_v, _lrq_agg = lrq
                q = quantize_2d(
                    arr,
                    out_dim,
                    in_dim,
                    1.0 if direct_exact else policy_frac,
                    None if direct_exact else act_scales,
                    0.0 if direct_exact else policy_alpha,
                    1.0 if direct_exact else policy_clip,
                    tensor_name=out_name,
                    ternary_threshold=(1.0 if direct_exact else policy_threshold),
                    lrq_u=_lrq_u,
                    lrq_v=_lrq_v,
                    lrq_agg=_lrq_agg,
                )
            elif use_septq:
                q = quantize_2d_septq(
                    arr,
                    out_dim,
                    in_dim,
                    args.septq_ratio,
                    act_scales=act_scales,
                    septq_iterations=args.septq_iterations,
                    ternary_threshold=policy_threshold,
                    tensor_name=out_name,
                    septq_hessian_mode=args.septq_hessian_mode,
                    septq_hessian_bandwidth=args.septq_hessian_bandwidth,
                    septq_importance_weight=args.septq_importance_weight,
                    septq_importance_lambda=args.septq_importance_lambda,
                )
            elif use_imatrix_mse:
                # imatrix_mse range selection: per-row MSE grid search
                # weighted by per-channel importance. AWQ per-channel scaling
                # is bypassed because importance is already consumed by the
                # range selection.
                q = quantize_2d_imatrix_mse(
                    arr,
                    out_dim,
                    in_dim,
                    policy_frac,
                    act_scales,
                    mse_norm=args.imatrix_mse_norm,
                    mse_grid=args.imatrix_mse_grid,
                    mse_maxshrink=args.imatrix_mse_maxshrink,
                    awq_clip=policy_clip,
                )
            else:
                q = quantize_2d(
                    arr,
                    out_dim,
                    in_dim,
                    1.0 if direct_exact else policy_frac,
                    None if direct_exact else act_scales,
                    0.0 if direct_exact else policy_alpha,
                    1.0 if direct_exact else policy_clip,
                    tensor_name=out_name,
                    ternary_threshold=(1.0 if direct_exact else policy_threshold),
                )
            if champq_perm is not None:
                # CHAMP-Q Option A: decode the permuted quantization to F32,
                # undo the input-dim permutation, and re-quantize in the
                # original order. The output Tile640 components are then in
                # the same channel order as the source weight, so the GGUF
                # is interchangeable with the non-CHAMP-Q output.
                w_unpermuted = decode_q_to_weight(q, out_dim, in_dim)
                w_unpermuted = apply_champq_permutation(
                    w_unpermuted, champq_inverse
                )
                if use_imatrix_mse:
                    q = quantize_2d_imatrix_mse(
                        w_unpermuted,
                        out_dim,
                        in_dim,
                        policy_frac,
                        lookup_acts(wname, imatrix, gguf_n) if imatrix else None,
                        mse_norm=args.imatrix_mse_norm,
                        mse_grid=args.imatrix_mse_grid,
                        mse_maxshrink=args.imatrix_mse_maxshrink,
                        awq_clip=policy_clip,
                    )
                else:
                    q = quantize_2d(
                        w_unpermuted,
                        out_dim,
                        in_dim,
                        policy_frac,
                        lookup_acts(wname, imatrix, gguf_n) if imatrix else None,
                        policy_alpha,
                        policy_clip,
                        tensor_name=out_name,
                        ternary_threshold=policy_threshold,
                    )
                if champq_policy is not None:
                    champq_policy.add(out_name, champq_perm)
            alpha = float(q["awq_alpha"])
            alpha_counts[alpha] = alpha_counts.get(alpha, 0) + 1
            writer.add_tensor(component_name(out_name, "weight_packed"),       q["packed"])
            writer.add_tensor(component_name(out_name, "weight_page_scales"),  q["page_scales"])
            writer.add_tensor(component_name(out_name, "weight_lane_scales"),  q["lane_scales"])
            writer.add_tensor(component_name(out_name, "weight_outlier_row_offsets"), q["outlier_row_offsets"])
            writer.add_tensor(component_name(out_name, "weight_outlier_cols"), q["outlier_cols"])
            writer.add_tensor(component_name(out_name, "weight_outlier_vals"), q["outlier_vals"])
            # Per-channel input scale (AWQ). When awq_alpha == 0 this is all 1.0;
            # the C++ side checks for it but treats it as no-op.
            if not np.allclose(q["input_scale"], 1.0, atol=1e-6):
                writer.add_tensor(component_name(out_name, "weight_act_scale"), q["input_scale"])
            quant_2d += 1
            del q
        elif is_quant_3d:
            assert arr.ndim == 3, f"expected 3D for {wname}, got {arr.shape}"
            n_experts, out_dim, in_dim = arr.shape
            gguf_n = out_name
            act_scales = lookup_acts(wname, imatrix, gguf_n) if imatrix else None
            if imatrix is not None:
                if act_scales is None:
                    imatrix_misses += 1
                else:
                    imatrix_hits += 1
            policy_frac, policy_alpha, policy_clip, policy_exact, _policy_threshold = tensor_policy(
                calibration_policy, out_name, args.outlier_frac, args.awq_alpha
            )
            expert_fractions, expert_alphas, expert_clips = expert_policy_values(
                calibration_policy,
                out_name,
                n_experts,
                policy_frac,
                policy_alpha,
                policy_clip,
            )
            # CHAMP-Q for 3D: permute the input channel axis of the expert
            # bank before quantizing, then fold the inverse permutation
            # into the output. The permutation is shared across all
            # experts (the in_dim axis is the same for every expert). See
            # the 2D path above for the rationale. policy_exact forces
            # every expert to exact encoding, where the input-channel
            # ordering is irrelevant, so CHAMP-Q is skipped. PE-QAT, if
            # present for this tensor, also skips CHAMP-Q for the same
            # reason as the 2D path -- the per-channel s was trained
            # against the original channel order. The 3D path itself does
            # not currently apply the PE-QAT merge (the demo only trains
            # 2D layers); the gate is here for symmetry / future 3D
            # support.
            pe_qat_entry_3d = (
                pe_qat_policy_for(pe_qat_policy, out_name)
                if not policy_exact
                else None
            )
            use_pe_qat_3d = pe_qat_entry_3d is not None
            champq_perm_3d: Optional[np.ndarray] = None
            if (
                args.permute_channels
                and not policy_exact
                and not use_pe_qat_3d
                and _CHAMPQ_AVAILABLE
                and in_dim > 1
            ):
                pooled_act: Optional[np.ndarray] = None
                if act_scales is not None:
                    if act_scales.ndim == 2:
                        pooled_act = np.mean(
                            act_scales, axis=0, dtype=np.float32
                        )
                    elif act_scales.shape == (in_dim,):
                        pooled_act = act_scales
                champq_perm_3d = compute_champq_permutation(arr, pooled_act)
                champq_inverse_3d = invert_champq_permutation(champq_perm_3d)
                arr = apply_champq_permutation(arr, champq_perm_3d)
            # Note: 3D path currently does not route through
            # quantize_2d_imatrix_mse because the expert loop is structured
            # differently (per-expert 2D quantize under quantize_3d). The
            # gemma 4 12B model is dense (no MoE), so this is a low-priority
            # extension; the legacy 3D path remains correct.
            qs = quantize_3d(
                arr, n_experts, out_dim, in_dim,
                1.0 if policy_exact else expert_fractions,
                None if policy_exact else imatrix, wname, gguf_n,
                0.0 if policy_exact else expert_alphas,
                1.0 if policy_exact else expert_clips,
            )
            if champq_perm_3d is not None:
                # CHAMP-Q Option A: decode each expert's permuted
                # quantization, undo the input-dim permutation, and
                # re-quantize the un-permuted expert bank with the
                # original imatrix. The result is a Tile640 expert bank
                # in the original channel order, interchangeable with
                # the non-CHAMP-Q output.
                w_orig_3d = np.empty_like(arr)
                for ex in range(n_experts):
                    w_perm = decode_q_to_weight(qs[ex], out_dim, in_dim)
                    w_orig_3d[ex] = apply_champq_permutation(
                        w_perm, champq_inverse_3d
                    )
                qs = quantize_3d(
                    w_orig_3d, n_experts, out_dim, in_dim,
                    expert_fractions,
                    imatrix, wname, gguf_n,
                    expert_alphas,
                    expert_clips,
                )
                if champq_policy is not None:
                    champq_policy.add(out_name, champq_perm_3d)
            if qs:
                alpha = float(qs[0]["awq_alpha"])
                alpha_counts[alpha] = alpha_counts.get(alpha, 0) + 1
            # Concatenate per-expert results into flat arrays
            packed = np.concatenate([q["packed"] for q in qs])
            page_scales = np.concatenate([q["page_scales"] for q in qs])
            lane_scales = np.concatenate([q["lane_scales"] for q in qs])
            outlier_cols = np.concatenate([q["outlier_cols"] for q in qs])
            outlier_vals = np.concatenate([q["outlier_vals"] for q in qs])
            outlier_row_offsets_parts = []
            outlier_base = 0
            for q in qs:
                outlier_row_offsets_parts.append(q["outlier_row_offsets"] + outlier_base)
                outlier_base += q["outlier_cols"].size
            outlier_row_offsets = np.concatenate(outlier_row_offsets_parts)
            # Each expert has its own observer row and therefore its own AWQ
            # transform. GGUF writes [expert, channel] as GGML [channel, expert].
            input_scale = (
                np.stack([q["input_scale"] for q in qs], axis=0)
                if qs else np.ones((n_experts, in_dim), dtype=np.float16)
            )
            writer.add_tensor(component_name(out_name, "weight_packed"),       packed)
            writer.add_tensor(component_name(out_name, "weight_page_scales"),  page_scales)
            writer.add_tensor(component_name(out_name, "weight_lane_scales"),  lane_scales)
            writer.add_tensor(component_name(out_name, "weight_outlier_row_offsets"), outlier_row_offsets)
            writer.add_tensor(component_name(out_name, "weight_outlier_cols"), outlier_cols)
            writer.add_tensor(component_name(out_name, "weight_outlier_vals"), outlier_vals)
            # Per-expert, per-channel input scale selected by matmul_id.
            if not np.allclose(input_scale, 1.0, atol=1e-6):
                writer.add_tensor(component_name(out_name, "weight_act_scale"), input_scale)
            quant_3d += 1
            del qs, packed, page_scales, lane_scales, outlier_row_offsets, outlier_cols, outlier_vals, input_scale
        else:
            raise AssertionError(
                f"direct inventory admitted non-Tile640 tensor {wname} "
                f"(mapped={out_name}, short={short})"
            )

        del arr, raw

        # Progress
        if (done + 1) % 25 == 0 or done + 1 == len(inventory):
            elapsed = time.time() - t0
            rate = (done + 1) / elapsed if elapsed > 0 else 0
            eta = (len(inventory) - done - 1) / rate if rate > 0 else 0
            print(
                f"  [{done+1}/{len(inventory)}] {wname[:80]:80s} "
                f"({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s) "
                f"[quant_2d={quant_2d}, quant_3d={quant_3d}, pass={pass_count}]",
                file=sys.stderr, flush=True,
            )

        # Free memory
        gc.collect()
        try:
            mx.clear_cache() if HAS_MLX else None
        except Exception:
            try:
                mx.metal.clear_cache() if HAS_MLX else None
            except Exception:
                pass

    # Evolved Tile640 stores every remaining canonical tensor in component
    # form. Converter-specific transforms are preserved by quantizing the
    # canonical GGUF arrays rather than trying to reproduce those transforms.
    canonical_quantized = 0
    canonical_total = sum(
        source_tensor.name not in replaced_source_tensor_names
        for source_tensor in metadata_reader.tensors
    )
    canonical_t0 = time.time()
    for source_tensor in metadata_reader.tensors:
        if source_tensor.name in replaced_source_tensor_names:
            continue
        # RoPE factors are generated graph constants, not learned checkpoint
        # parameters. ggml_rope requires F32 during buffer probing and values
        # such as 1e30 are intentionally outside F16, so preserve this one
        # non-weight tensor verbatim.
        if source_tensor.name.endswith("rope_freqs.weight"):
            writer.add_tensor(source_tensor.name, np.asarray(source_tensor.data, dtype=np.float32))
            canonical_quantized += 1
            continue
        source_data = np.asarray(source_tensor.data)
        source_type = source_tensor.tensor_type
        if source_type in {
            gguf.GGMLQuantizationType.MXFP4,
            gguf.GGMLQuantizationType.NVFP4,
        }:
            # DeepSeek V4 Flash/Pro ship routed experts as microscaled FP4.
            # GGUFReader exposes their physical 17-byte blocks; treating those
            # bytes as scalar weights corrupts every expert. Decode to the
            # logical float matrix before applying AWQ and Tessera packing.
            arr = gguf.dequantize(source_data, source_type).astype(
                np.float32, copy=False
            )
        else:
            arr = np.asarray(source_data, dtype=np.float32)
        rows, row_width, matrix_shape = evolved_matrix_view(arr)
        # Vectors, biases, norms, and narrow convolution axes are stored as
        # exact sparse residual rows. They still use Tile640 components and
        # therefore require no source F16/F32 tensor at runtime.
        # Routing logits are disproportionately sensitive: changing their
        # ordering changes which experts execute, amplifying even a small
        # weight error. Unsloth's public MoE calibration path explicitly keeps
        # router and shared-expert gates unquantized. Preserve that behavior
        # without passthrough tensors by storing them as exact Tile640 sparse
        # residual rows.
        is_router = (
            ".ffn_gate_inp.weight" in source_tensor.name
            or ".ffn_gate_inp_shexp.weight" in source_tensor.name
        )
        # gemma 4 sensitive tensors (QK-norm, post-norm, attention output,
        # FFN down) — see is_gemma4_sensitive_tensor for the rationale.
        is_gemma4_sensitive_evo = (
            architecture_name in ("gemma4", "gemma4-assistant")
            and is_gemma4_sensitive_tensor(
                source_tensor.name, args.gemma4_sensitive_patterns
            )
        )
        exact = (
            arr.ndim == 1
            or source_tensor.name.endswith(".bias")
            or row_width < 128
            or is_router
            or is_gemma4_sensitive_evo
        )
        policy_frac, policy_alpha, policy_clip, policy_exact, _policy_threshold = tensor_policy(
            calibration_policy, source_tensor.name, args.default_outlier_frac, 0.0
        )
        exact = exact or policy_exact
        frac = 1.0 if exact else policy_frac
        q = quantize_2d(
            rows, rows.shape[0], row_width, frac, None,
            0.0 if exact else policy_alpha,
            1.0 if exact else policy_clip,
            tensor_name=source_tensor.name,
        )
        writer.add_array(f"tessera.shape.{source_tensor.name}", [int(v) for v in source_tensor.shape])
        writer.add_array(f"tessera.matrix_shape.{source_tensor.name}", matrix_shape)
        writer.add_tensor(evolved_component_name(source_tensor.name, "packed"), q["packed"])
        writer.add_tensor(evolved_component_name(source_tensor.name, "page_scales"), q["page_scales"])
        writer.add_tensor(evolved_component_name(source_tensor.name, "lane_scales"), q["lane_scales"])
        writer.add_tensor(evolved_component_name(source_tensor.name, "outlier_row_offsets"), q["outlier_row_offsets"])
        writer.add_tensor(evolved_component_name(source_tensor.name, "outlier_cols"), q["outlier_cols"])
        writer.add_tensor(evolved_component_name(source_tensor.name, "outlier_vals"), q["outlier_vals"])
        canonical_quantized += 1
        del arr, rows, q
        if canonical_quantized % 25 == 0 or canonical_quantized == canonical_total:
            elapsed = time.time() - canonical_t0
            rate = canonical_quantized / elapsed if elapsed > 0 else 0
            eta = (canonical_total - canonical_quantized) / rate if rate > 0 else 0
            print(
                f"  [canonical {canonical_quantized}/{canonical_total}] "
                f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
    print(
        f"Converted {canonical_quantized} canonical text tensors to Tessera T640",
        file=sys.stderr,
    )
    embedded_vision_tensors = 0
    if vision_reader is not None:
        vision_total = len(vision_reader.tensors)
        vision_t0 = time.time()
        for source_tensor in vision_reader.tensors:
            arr = np.asarray(source_tensor.data, dtype=np.float32)
            rows, row_width, matrix_shape = evolved_matrix_view(arr)
            exact = arr.ndim == 1 or source_tensor.name.endswith(".bias") or row_width < 128
            frac = 1.0 if exact else args.default_outlier_frac
            q = quantize_2d(rows, rows.shape[0], row_width, frac, None, 0.0)
            writer.add_array(f"tessera.shape.{source_tensor.name}", [int(v) for v in source_tensor.shape])
            writer.add_array(f"tessera.matrix_shape.{source_tensor.name}", matrix_shape)
            writer.add_tensor(evolved_component_name(source_tensor.name, "packed"), q["packed"])
            writer.add_tensor(evolved_component_name(source_tensor.name, "page_scales"), q["page_scales"])
            writer.add_tensor(evolved_component_name(source_tensor.name, "lane_scales"), q["lane_scales"])
            writer.add_tensor(evolved_component_name(source_tensor.name, "outlier_row_offsets"), q["outlier_row_offsets"])
            writer.add_tensor(evolved_component_name(source_tensor.name, "outlier_cols"), q["outlier_cols"])
            writer.add_tensor(evolved_component_name(source_tensor.name, "outlier_vals"), q["outlier_vals"])
            embedded_vision_tensors += 1
            del arr, rows, q
            if embedded_vision_tensors % 25 == 0 or embedded_vision_tensors == vision_total:
                elapsed = time.time() - vision_t0
                rate = embedded_vision_tensors / elapsed if elapsed > 0 else 0
                eta = (vision_total - embedded_vision_tensors) / rate if rate > 0 else 0
                print(
                    f"  [vision {embedded_vision_tensors}/{vision_total}] "
                    f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )
                gc.collect()
        print(
            f"Converted and embedded {embedded_vision_tensors} vision tensors as Tessera T640",
            file=sys.stderr,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if imatrix is not None:
        total_lookups = imatrix_hits + imatrix_misses
        coverage = 100.0 * imatrix_hits / total_lookups if total_lookups else 0.0
        print(f"AWQ imatrix coverage: {imatrix_hits}/{total_lookups} tensors ({coverage:.1f}%)", file=sys.stderr)
        print(f"AWQ selected alphas: {dict(sorted(alpha_counts.items()))}", file=sys.stderr)
    if champq_policy is not None and champq_policy.tensors:
        champq_policy.save(str(champq_policy_path))
        print(
            f"  CHAMP-Q: wrote {len(champq_policy.tensors)} permutations "
            f"to {champq_policy_path}",
            file=sys.stderr,
        )
    elif champq_policy is not None:
        print(
            "  CHAMP-Q: enabled but no tensors received a permutation "
            "(everything was direct_exact); policy file not written",
            file=sys.stderr,
        )
    print(f"\nWrote {args.output}", file=sys.stderr)
    print(f"OK: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
