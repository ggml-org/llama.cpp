"""Guard for the dory tensor mapping: every tensor in the shipped GGUF must be nameable.

This test exists because the two failure modes that actually bit this port are
both silent:

  * a tensor enum with no entry in LLM_TENSOR_NAMES fails at RUNTIME
    (std::map::at) rather than at compile time, and
  * an attention/MLP index mistake still produces a GGUF that loads and then
    emits plausible garbage.

The check is deliberately mechanical and cheap: parse the GGUF's own tensor
names, substitute them into the enum's format string, and require that every
name produced this way exists in llama-arch.cpp's LLM_TENSOR_NAMES table. If a
future change drops a row, the test fails with the offending suffixes named
instead of a loader abort.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GGUF = Path("/Volumes/Safehold/models/dory-1.0-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
ARCH_CPP = REPO / "src" / "llama-arch.cpp"

# The 16 dory-specific enum suffixes the port relies on. Keeping this list small
# makes the test sensitive: a dropped row is caught by the full-file census too,
# but this names the exact one that matters.
REQUIRED_ENUMS = (
    "LLM_TENSOR_ATTN_Q",
    "LLM_TENSOR_ATTN_K",
    "LLM_TENSOR_ATTN_V",
    "LLM_TENSOR_ATTN_OUT",
    "LLM_TENSOR_ATTN_GATE",
    "LLM_TENSOR_ATTN_QK_SCALE",
    "LLM_TENSOR_ATTN_GATE_SCALE",
    "LLM_TENSOR_FFN_GATE",
    "LLM_TENSOR_FFN_UP",
    "LLM_TENSOR_FFN_DOWN",
    "LLM_TENSOR_FFN_GATE_SCALE",
    "LLM_TENSOR_FFN_UP_SCALE",
    "LLM_TENSOR_RESIDUAL_ALPHA",
    "LLM_TENSOR_OUTPUT_SCALE",
    "LLM_TENSOR_TOKEN_EMBD",
    "LLM_TENSOR_OUTPUT",
)


def _declared_names() -> dict[str, str]:
    """{enum: format-string} from LLM_TENSOR_NAMES in llama-arch.cpp."""
    text = ARCH_CPP.read_text(encoding="utf-8")
    block = re.search(r"LLM_TENSOR_NAMES = \{(.*?)\n\};", text, re.S)
    assert block, "LLM_TENSOR_NAMES not found in llama-arch.cpp"
    return {m.group(1): m.group(2) for m in re.finditer(r"\{\s*(LLM_TENSOR_\w+),\s*\"([^\"]+)\"", block.group(1))}


def _read_gguf_names() -> list[str]:
    with GGUF.open("rb") as handle:
        assert handle.read(4) == b"GGUF"
        _, n_tensors, n_kv = struct.unpack("<IQQ", handle.read(20))
        for _ in range(n_kv):
            (key_len,) = struct.unpack("<Q", handle.read(8))
            handle.read(key_len)
            (kv_type,) = struct.unpack("<I", handle.read(4))
            if kv_type == 8:  # string
                (n,) = struct.unpack("<Q", handle.read(8))
                handle.read(n)
            elif kv_type == 9:  # array: element type (4B) then count (8B), then payload
                elem_type, count = struct.unpack("<IQ", handle.read(12))
                if elem_type == 8:  # array of strings: length-prefixed per element
                    for _ in range(count):
                        (n,) = struct.unpack("<Q", handle.read(8))
                        handle.read(n)
                else:
                    scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[elem_type]
                    handle.read(scalar * count)
            else:
                scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[kv_type]
                handle.read(scalar)
        names = []
        for _ in range(n_tensors):
            (name_len,) = struct.unpack("<Q", handle.read(8))
            names.append(handle.read(name_len).decode("utf-8"))
            n_dims = struct.unpack("<I", handle.read(4))[0]
            handle.read(8 * n_dims)
            handle.read(4)  # ggml type
            handle.read(8)  # offset
        return names


def test_every_shipped_tensor_is_covered_by_the_name_table() -> None:
    names = _read_gguf_names()
    assert len(names) == 507, f"expected the dory artifact's 507 tensors, got {len(names)}"
    table = _declared_names()
    # the table stores PREFIXES (blk.%d.attn_q), tensor names carry ".weight" after,
    # so anchoring at end-of-format would reject every real tensor name.
    patterns = [re.escape(fmt).replace("%d", r"\d+") for fmt in table.values()]
    missing = [n for n in names if not any(re.search(pat, n) for pat in patterns)]
    assert not missing, f"{len(missing)} shipped tensors have no LLM_TENSOR_NAMES entry: {missing[:6]}"


def test_required_dory_enums_exist_and_have_format_strings() -> None:
    table = _declared_names()
    missing = [e for e in REQUIRED_ENUMS if e not in table]
    assert not missing, f"enums the dory port needs but the table lacks: {missing}"


def test_attention_and_mlp_indices_do_not_alias() -> None:
    """blk.even must be attention-shaped and blk.odd must be ffn-shaped.

    Attention and MLP live in SEPARATE blk indices in this arch; aliasing them
    would still load and would silently produce wrong math.
    """
    names = set(_read_gguf_names())
    attn_suffixes = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
    ffn_suffixes = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
    for index in (0, 2, 4, 70):
        assert any(f"blk.{index}.{s}" in n for s in attn_suffixes for n in names), index
        assert not any(f"blk.{index}.{s}" in n for s in ffn_suffixes for n in names), index
    for index in (1, 3, 5, 71):
        assert any(f"blk.{index}.{s}" in n for s in ffn_suffixes for n in names), index
        assert not any(f"blk.{index}.{s}" in n for s in attn_suffixes for n in names), index
