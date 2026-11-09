"""Per-tensor quant type selection for direct safetensors -> quantized GGUF.

Ports the mixed-ftype type table from src/llama-quant.cpp (llama_tensor_get_type_impl,
tensor_get_category, llama_ftype_get_default_type, tensor_type_fallback) so that
convert_hf_to_gguf.py can emit K-quant / I-quant GGUFs in a single pass, with no FP16
intermediate file.

The layer index used by the "use more bits on edge layers" heuristics is the block id
parsed from the tensor name (blk.N.*). This matches the C++ behaviour, where the running
per-category counter is overridden by the parsed block id for MoE models and is identical
to the layer index for dense models.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gguf import GGMLQuantizationType as T
from gguf import LlamaFileType as F


class Cat(Enum):
    OUTPUT = "output"
    TOKEN_EMBD = "token_embd"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_QKV = "attn_qkv"
    ATTN_KV_B = "attn_kv_b"
    ATTN_OUTPUT = "attn_output"
    ATTN_GATE = "attn_gate"
    FFN_UP = "ffn_up"
    FFN_GATE = "ffn_gate"
    FFN_LATENT = "ffn_latent"
    FFN_DOWN = "ffn_down"
    OTHER = "other"


def tensor_category(name: str) -> Cat:
    # substring matching on the gguf tensor name, order mirrors tensor_get_category
    if name == "output.weight":
        return Cat.OUTPUT
    if name in ("token_embd.weight", "per_layer_token_embd.weight"):
        return Cat.TOKEN_EMBD
    if "attn_qkv.weight" in name or "attn_qkv_gate.weight" in name:
        return Cat.ATTN_QKV
    if "attn_kv_b.weight" in name:
        return Cat.ATTN_KV_B
    if "attn_v.weight" in name:
        return Cat.ATTN_V
    if "attn_k.weight" in name:
        return Cat.ATTN_K
    if "attn_q.weight" in name:
        return Cat.ATTN_Q
    if "attn_output.weight" in name:
        return Cat.ATTN_OUTPUT
    if "ffn_up" in name:
        return Cat.FFN_UP
    if "ffn_gate" in name:
        return Cat.FFN_GATE
    if "ffn_latent" in name:
        return Cat.FFN_LATENT
    if "attn_gate.weight" in name:
        return Cat.ATTN_GATE
    if "ffn_down" in name:
        return Cat.FFN_DOWN
    return Cat.OTHER


def category_is_attn_v(cat: Cat) -> bool:
    return cat in (Cat.ATTN_V, Cat.ATTN_QKV, Cat.ATTN_KV_B)


def tensor_is_expert(name: str) -> bool:
    return "_exps." in name or "_chexps." in name


# ftype -> default (base) tensor type, mirrors llama_ftype_get_default_type
FTYPE_DEFAULT: dict[F, T] = {
    F.MOSTLY_Q4_0: T.Q4_0,
    F.MOSTLY_Q4_1: T.Q4_1,
    F.MOSTLY_Q5_0: T.Q5_0,
    F.MOSTLY_Q5_1: T.Q5_1,
    F.MOSTLY_Q8_0: T.Q8_0,
    F.MOSTLY_F16: T.F16,
    F.MOSTLY_BF16: T.BF16,
    F.ALL_F32: T.F32,
    F.MOSTLY_Q1_0: T.Q1_0,
    F.MOSTLY_Q2_0: T.Q2_0,
    F.MOSTLY_MXFP4_MOE: T.MXFP4,
    # K-quants
    F.MOSTLY_Q2_K_S: T.Q2_K,
    F.MOSTLY_Q2_K: T.Q2_K,
    F.MOSTLY_IQ3_XS: T.IQ3_S,
    F.MOSTLY_Q3_K_S: T.Q3_K,
    F.MOSTLY_Q3_K_M: T.Q3_K,
    F.MOSTLY_Q3_K_L: T.Q3_K,
    F.MOSTLY_Q4_K_S: T.Q4_K,
    F.MOSTLY_Q4_K_M: T.Q4_K,
    F.MOSTLY_Q5_K_S: T.Q5_K,
    F.MOSTLY_Q5_K_M: T.Q5_K,
    F.MOSTLY_Q6_K: T.Q6_K,
    F.MOSTLY_TQ1_0: T.TQ1_0,
    F.MOSTLY_TQ2_0: T.TQ2_0,
    F.MOSTLY_IQ2_XXS: T.IQ2_XXS,
    F.MOSTLY_IQ2_XS: T.IQ2_XS,
    F.MOSTLY_IQ2_S: T.IQ2_XS,
    F.MOSTLY_IQ2_M: T.IQ2_S,
    F.MOSTLY_IQ3_XXS: T.IQ3_XXS,
    F.MOSTLY_IQ1_S: T.IQ1_S,
    F.MOSTLY_IQ1_M: T.IQ1_M,
    F.MOSTLY_IQ4_NL: T.IQ4_NL,
    F.MOSTLY_IQ4_XS: T.IQ4_XS,
    F.MOSTLY_IQ3_S: T.IQ3_S,
    F.MOSTLY_IQ3_M: T.IQ3_S,
}


# types gguf-py can quantize in python (produce, not just dequantize); floats need no quantization
PYTHON_QUANTIZABLE = {T.BF16, T.MXFP4, T.Q1_0, T.Q2_0, T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0, T.TQ1_0, T.TQ2_0}
FLOAT_TYPES = {T.F16, T.F32, T.BF16, T.F64}


@dataclass
class QuantCtx:
    n_layer: int
    n_expert: int = 1
    n_gqa: int = 1
    arch: str = ""
    is_70b: bool = False
    has_imatrix: bool = False


def use_more_bits(i_layer: int, n_layers: int) -> bool:
    return i_layer < n_layers // 8 or i_layer >= 7 * n_layers // 8 or (i_layer - n_layers // 8) % 3 == 2


def _bits_per_weight(t: T) -> float:
    from gguf.constants import GGML_QUANT_SIZES
    block_size, type_size = GGML_QUANT_SIZES[t]
    return 8.0 * type_size / block_size


def get_tensor_type(ftype: F, cat: Cat, name: str, ctx: QuantCtx, new_type: T) -> T:
    """Port of llama_tensor_get_type_impl. Returns the target ggml type for a tensor."""
    arch = ctx.arch
    n_expert = ctx.n_expert
    n_gqa = ctx.n_gqa
    n_layer = ctx.n_layer

    # block id (layer index) parsed from the gguf name, e.g. blk.5.ffn_down.weight -> 5
    i_layer = 0
    for part in name.split("."):
        if part.isdecimal():
            i_layer = int(part)
            break

    # by default, for glm5-next, don't let these tensors be quantized below Q8_0
    if arch == "glm5-next" and new_type not in (T.F32, T.BF16, T.F16) and (
        "attn_q_a" in name or "attn_q_b" in name or "nextn.eh_proj" in name
    ):
        return T.Q8_0

    # bumping to q8_0 is preferred since these tensors are not really big
    if arch == "muse-spark" and cat in (Cat.FFN_LATENT, Cat.ATTN_GATE):
        from gguf.constants import GGML_QUANT_SIZES
        if new_type in GGML_QUANT_SIZES and _bits_per_weight(new_type) < _bits_per_weight(T.Q8_0):
            new_type = T.Q8_0

    # for arches that share the same tensor between the token embeddings and the output
    if cat == Cat.OUTPUT:
        if ftype == F.MOSTLY_MXFP4_MOE or arch == "falcon" or _not_divisible(name, new_type):
            new_type = T.Q8_0
        elif ftype in (F.MOSTLY_IQ2_XXS, F.MOSTLY_IQ2_XS, F.MOSTLY_IQ3_XXS,
                       F.MOSTLY_IQ1_S, F.MOSTLY_IQ2_S, F.MOSTLY_IQ2_M, F.MOSTLY_IQ1_M):
            new_type = T.Q5_K
        elif new_type != T.Q8_0:
            new_type = T.Q6_K
    elif ftype == F.MOSTLY_MXFP4_MOE:
        new_type = T.MXFP4 if tensor_is_expert(name) else T.Q8_0
    elif cat == Cat.TOKEN_EMBD:
        if ftype in (F.MOSTLY_IQ2_XXS, F.MOSTLY_IQ2_XS, F.MOSTLY_IQ1_S, F.MOSTLY_IQ1_M):
            new_type = T.Q2_K
        elif ftype in (F.MOSTLY_IQ2_S, F.MOSTLY_IQ2_M, F.MOSTLY_IQ3_XXS):
            new_type = T.IQ3_S
        elif ftype in (F.MOSTLY_TQ1_0, F.MOSTLY_TQ2_0, F.MOSTLY_Q2_0):
            new_type = T.Q4_K
    elif ftype in (F.MOSTLY_IQ2_XXS, F.MOSTLY_IQ2_XS, F.MOSTLY_IQ1_S,
                   F.MOSTLY_IQ2_S, F.MOSTLY_IQ2_M, F.MOSTLY_IQ1_M):
        if category_is_attn_v(cat):
            if n_gqa >= 4 or n_expert >= 4:
                new_type = T.Q4_K
            else:
                new_type = T.IQ3_S if ftype in (F.MOSTLY_IQ2_S, F.MOSTLY_IQ2_M) else T.Q2_K
        elif n_expert == 8 and cat == Cat.ATTN_K:
            new_type = T.Q4_K
        elif cat == Cat.FFN_DOWN:
            if i_layer < n_layer // 8:
                new_type = T.IQ3_S if ftype in (F.MOSTLY_IQ2_S, F.MOSTLY_IQ2_M) else T.Q2_K
        elif cat == Cat.ATTN_OUTPUT:
            if n_expert == 8:
                new_type = T.Q5_K
            else:
                if ftype in (F.MOSTLY_IQ1_S, F.MOSTLY_IQ1_M):
                    new_type = T.IQ2_XXS
                elif ftype in (F.MOSTLY_IQ2_S, F.MOSTLY_IQ2_M):
                    new_type = T.IQ3_S
    elif category_is_attn_v(cat):
        if ftype == F.MOSTLY_Q2_K:
            new_type = T.Q4_K if n_gqa >= 4 else T.Q3_K
        elif ftype == F.MOSTLY_Q2_K_S and n_gqa >= 4:
            new_type = T.Q4_K
        elif ftype == F.MOSTLY_IQ3_XXS:
            new_type = T.Q4_K if n_gqa >= 4 else (T.IQ3_S if not ctx.has_imatrix else T.IQ3_XXS)
        elif (ftype in (F.MOSTLY_IQ3_XS, F.MOSTLY_IQ3_S) and n_gqa >= 4) or ftype == F.MOSTLY_IQ3_M:
            new_type = T.Q4_K
        elif ftype == F.MOSTLY_Q3_K_M:
            new_type = T.Q5_K if i_layer < 2 else T.Q4_K
        elif ftype == F.MOSTLY_Q3_K_L:
            new_type = T.Q5_K
        elif ftype in (F.MOSTLY_IQ4_NL, F.MOSTLY_IQ4_XS) and n_gqa >= 4:
            new_type = T.Q5_K
        elif ftype in (F.MOSTLY_Q4_K_M, F.MOSTLY_Q5_K_M) and use_more_bits(i_layer, n_layer):
            new_type = T.Q6_K
        elif ftype == F.MOSTLY_Q4_K_S and i_layer < 4:
            new_type = T.Q5_K
        if ctx.is_70b and new_type in (T.Q3_K, T.Q4_K):
            new_type = T.Q5_K
        if n_expert == 8:
            new_type = T.Q8_0
    elif cat == Cat.ATTN_K:
        if n_expert == 8:
            new_type = T.Q8_0
        elif ftype == F.MOSTLY_IQ3_XS:
            new_type = T.IQ3_XXS
        elif ftype == F.MOSTLY_IQ3_XXS:
            new_type = T.IQ2_S
    elif cat == Cat.ATTN_Q:
        if ftype == F.MOSTLY_IQ3_XS:
            new_type = T.IQ3_XXS
        elif ftype == F.MOSTLY_IQ3_XXS:
            new_type = T.IQ2_S
    elif cat == Cat.FFN_DOWN:
        if ftype == F.MOSTLY_Q2_K:
            new_type = T.Q3_K
        elif ftype == F.MOSTLY_Q2_K_S:
            if i_layer < n_layer // 8:
                new_type = T.Q4_K
        elif ftype == F.MOSTLY_IQ3_XXS and not ctx.has_imatrix:
            new_type = T.Q4_K if i_layer < n_layer // 8 else T.Q3_K
        elif ftype == F.MOSTLY_Q3_K_M:
            if i_layer < n_layer // 16:
                new_type = T.Q5_K
            elif arch != "falcon" or use_more_bits(i_layer, n_layer):
                new_type = T.Q4_K
            else:
                new_type = T.Q3_K
        elif ftype == F.MOSTLY_IQ3_M and (i_layer < n_layer // 8 or
                                          (n_expert == 8 and use_more_bits(i_layer, n_layer))):
            new_type = T.Q4_K
        elif ftype == F.MOSTLY_Q3_K_L:
            new_type = T.Q4_K if arch == "falcon" else T.Q5_K
        elif ftype == F.MOSTLY_Q4_K_M:
            if arch == "falcon":
                if i_layer < n_layer // 16:
                    new_type = T.Q6_K
                elif use_more_bits(i_layer, n_layer):
                    new_type = T.Q5_K
                else:
                    new_type = T.Q4_K
            elif use_more_bits(i_layer, n_layer):
                new_type = T.Q6_K
        elif i_layer < n_layer // 8 and ftype in (F.MOSTLY_IQ4_NL, F.MOSTLY_IQ4_XS) and not ctx.has_imatrix:
            new_type = T.Q5_K
        elif ftype == F.MOSTLY_Q5_K_M and use_more_bits(i_layer, n_layer):
            new_type = T.Q6_K
        elif ftype == F.MOSTLY_Q4_K_S and arch != "falcon" and i_layer < n_layer // 8:
            new_type = T.Q5_K
    elif cat == Cat.ATTN_OUTPUT:
        if arch != "falcon":
            if n_expert == 8:
                if ftype in (F.MOSTLY_Q2_K, F.MOSTLY_IQ3_XS, F.MOSTLY_IQ3_XXS,
                             F.MOSTLY_Q3_K_S, F.MOSTLY_Q3_K_M, F.MOSTLY_IQ4_NL,
                             F.MOSTLY_Q4_K_S, F.MOSTLY_Q4_K_M, F.MOSTLY_IQ3_S,
                             F.MOSTLY_IQ3_M, F.MOSTLY_IQ4_XS):
                    new_type = T.Q5_K
            else:
                if ftype == F.MOSTLY_Q2_K:
                    new_type = T.Q3_K
                elif ftype == F.MOSTLY_IQ3_XXS:
                    new_type = T.IQ3_S
                elif ftype == F.MOSTLY_Q3_K_M:
                    new_type = T.Q4_K
                elif ftype == F.MOSTLY_Q3_K_L:
                    new_type = T.Q5_K
                elif ftype == F.MOSTLY_IQ3_M:
                    new_type = T.Q4_K
        else:
            if ftype == F.MOSTLY_Q3_K_L:
                new_type = T.Q4_K
    elif cat == Cat.ATTN_QKV:
        if ftype in (F.MOSTLY_Q3_K_M, F.MOSTLY_Q3_K_L, F.MOSTLY_IQ3_M):
            new_type = T.Q4_K
        elif ftype == F.MOSTLY_Q4_K_M:
            new_type = T.Q5_K
        elif ftype == F.MOSTLY_Q5_K_M:
            new_type = T.Q6_K
    elif cat in (Cat.FFN_GATE, Cat.FFN_UP) and ftype == F.MOSTLY_IQ3_XS and (n_layer // 8 <= i_layer < 7 * n_layer // 8):
        new_type = T.IQ3_XXS

    return new_type


def _not_divisible(name: str, t: T) -> bool:
    # placeholder: the C++ checks the output tensor's column count against the block size.
    # the caller supplies ncols via tensor_type_fallback; here we conservatively return
    # False so the output stays at the ftype default (Q8_0 is applied for falcon above).
    return False


def parse_tensor_type_spec(spec: str) -> tuple[str, T]:
    """Parse one 'name=type' spec (name lowercased, type a ggml type name)."""
    if "=" not in spec:
        raise ValueError(f"malformed tensor type '{spec}' (expected name=type)")
    name, _, type_str = spec.partition("=")
    name = name.strip().lower()
    if not name:
        raise ValueError(f"missing tensor name in '{spec}'")
    type_str = type_str.strip()
    if not type_str:
        raise ValueError(f"missing quantization type in '{spec}'")
    try:
        t = T[type_str.upper()]
    except KeyError:
        raise ValueError(f"invalid quantization type '{type_str}' in '{spec}'") from None
    return name, t


def parse_tensor_type_overrides(specs: list[str]) -> list[tuple[str, T]]:
    """Parse a list of 'name=type' specs into (name, type) overrides, first match wins."""
    return [parse_tensor_type_spec(s) for s in specs if s.strip()]


def match_override(overrides: list[tuple[str, T]], name: str) -> T | None:
    """Return the type of the first override whose name is a substring of the (lowercased)
    gguf tensor name, or None. Mirrors llama-quantize --tensor-type matching."""
    name = name.lower()
    for pattern, t in overrides:
        if pattern in name:
            return t
    return None


def tensor_type_fallback(name: str, ncols: int, target: T) -> T:
    """Port of tensor_type_fallback: demote to a compatible type when ncols is not a
    multiple of the target's block size."""
    from gguf.constants import GGML_QUANT_SIZES
    block_size, _ = GGML_QUANT_SIZES[target]
    if ncols % block_size == 0:
        return target

    fallback = {
        T.IQ1_S: T.IQ4_NL, T.IQ1_M: T.IQ4_NL, T.IQ2_XXS: T.IQ4_NL, T.IQ2_XS: T.IQ4_NL,
        T.IQ2_S: T.IQ4_NL, T.IQ3_XXS: T.IQ4_NL, T.IQ3_S: T.IQ4_NL, T.IQ4_XS: T.IQ4_NL,
        T.Q2_0: T.Q4_0, T.Q2_K: T.Q4_0, T.Q3_K: T.Q4_0, T.TQ1_0: T.Q4_0, T.TQ2_0: T.Q4_0,
        T.Q4_K: T.Q5_0, T.Q5_K: T.Q5_1, T.Q6_K: T.Q8_0,
    }.get(target)

    if fallback is None:
        if block_size <= 32:
            return target
        raise ValueError(f"no tensor type fallback is defined for type {target.name}")

    fb_block = GGML_QUANT_SIZES[fallback][0]
    if ncols % fb_block != 0:
        return T.F16
    return fallback
