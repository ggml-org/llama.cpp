#!/usr/bin/env python3
"""Enforceable Tessera coverage contracts for supported model families."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Sequence


SCHEMA = "llama.tessera.family-coverage.v1"


@dataclass(frozen=True)
class FamilyContract:
    family: str
    architectures: frozenset[str]
    required_text_patterns: tuple[str, ...]
    assistant_architectures: frozenset[str] = frozenset()


CONTRACTS = (
    FamilyContract(
        family="gemma4",
        architectures=frozenset({"gemma4", "gemma4-assistant"}),
        required_text_patterns=(
            r"^token_embd\.weight$",
            r"^output_norm\.weight$",
            r"^blk\.\d+\.attn_norm\.weight$",
            r"^blk\.\d+\.ffn_norm\.weight$",
        ),
        assistant_architectures=frozenset({"gemma4-assistant"}),
    ),
    FamilyContract(
        family="qwen3.6",
        architectures=frozenset({"qwen35", "qwen35moe"}),
        required_text_patterns=(
            r"^token_embd\.weight$",
            r"^output_norm\.weight$",
            r"^blk\.\d+\.attn_norm\.weight$",
            r"^blk\.\d+\.ffn_(?:norm|gate_inp)\.weight$",
        ),
    ),
)


def contract_for_architecture(architecture: str) -> FamilyContract:
    for contract in CONTRACTS:
        if architecture in contract.architectures:
            return contract
    raise ValueError(f"no Tessera family coverage contract for {architecture!r}")


def component_for_tensor(name: str, base_block_count: int) -> str:
    if name.startswith(("v.", "mm.", "vision.", "audio.")):
        return "multimodal"
    if name.startswith("mtp."):
        return "drafter"
    block = re.match(r"^blk\.(\d+)\.", name)
    if block and int(block.group(1)) >= base_block_count:
        return "drafter"
    if any(fragment in name for fragment in (
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_up_exps",
        "ffn_gate_inp",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
        "ffn_latent_",
        "exp_probs_b",
    )):
        return "moe"
    return "text"


def treatment_for_tensor(name: str, shape: Sequence[int]) -> str:
    if name.endswith("rope_freqs.weight"):
        return "generated-exact"
    if name in {
        "token_embd.weight",
        "per_layer_token_embd.weight",
    } or name.endswith(".embed_tokens.weight"):
        return "tile640-row-gather"
    if any(fragment in name for fragment in (
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_up_exps",
    )):
        return "tile640-expert-matmul"
    if (
        len(shape) == 1
        or name.endswith(".bias")
        or (shape and min(shape) < 128)
        or ".ffn_gate_inp" in name
    ):
        return "tile640-exact-residual"
    if len(shape) > 2 or "conv" in name:
        return "tile640-boundary-dequant"
    return "tile640-matmul"


def _require_patterns(names: set[str], patterns: Iterable[str], label: str) -> None:
    missing = [
        pattern for pattern in patterns
        if not any(re.search(pattern, name) for name in names)
    ]
    if missing:
        raise ValueError(f"{label} coverage is missing required tensors: {missing}")


def build_coverage_receipt(
    architecture: str,
    tensors: Sequence[tuple[str, Sequence[int]]],
    base_block_count: int,
    total_block_count: int,
    *,
    has_multimodal_source: bool = False,
    has_multimodal_payload: bool = False,
    source_classifications: dict[str, int] | None = None,
) -> dict:
    contract = contract_for_architecture(architecture)
    if not tensors:
        raise ValueError(f"{architecture}: canonical GGUF has no tensors")

    names = {name for name, _ in tensors}
    _require_patterns(names, contract.required_text_patterns, architecture)

    records = []
    components: Counter[str] = Counter()
    treatments: Counter[str] = Counter()
    for name, shape in sorted(tensors):
        component = component_for_tensor(name, base_block_count)
        treatment = treatment_for_tensor(name, shape)
        records.append({
            "name": name,
            "shape": [int(value) for value in shape],
            "component": component,
            "treatment": treatment,
        })
        components[component] += 1
        treatments[treatment] += 1

    features = {
        "text": components["text"] > 0,
        "moe": components["moe"] > 0,
        "drafter": (
            components["drafter"] > 0
            or architecture in contract.assistant_architectures
            or total_block_count > base_block_count
        ),
        "multimodal": has_multimodal_source or has_multimodal_payload,
    }
    if has_multimodal_source and not has_multimodal_payload:
        raise ValueError(
            f"{architecture}: checkpoint contains multimodal tensors but no "
            "--vision-from payload was supplied"
        )
    if architecture.endswith("moe") and not features["moe"]:
        raise ValueError(f"{architecture}: MoE architecture has no classified MoE tensors")
    if architecture in contract.assistant_architectures and not features["drafter"]:
        raise ValueError(f"{architecture}: assistant architecture has no drafter tensors")

    digest_input = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": SCHEMA,
        "family": contract.family,
        "architecture": architecture,
        "complete": True,
        "base_block_count": int(base_block_count),
        "total_block_count": int(total_block_count),
        "tensor_count": len(records),
        "features": features,
        "components": dict(sorted(components.items())),
        "treatments": dict(sorted(treatments.items())),
        "source_classifications": dict(sorted(
            (source_classifications or {}).items()
        )),
        "tensor_manifest_sha256": hashlib.sha256(digest_input).hexdigest(),
    }


def compact_receipt_json(receipt: dict) -> str:
    if receipt.get("schema") != SCHEMA or receipt.get("complete") is not True:
        raise ValueError("cannot serialize an incomplete Tessera family receipt")
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"))
