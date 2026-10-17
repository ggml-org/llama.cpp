#!/usr/bin/env python3
"""Audit pinned Gemma target/shared-KV assistant metadata without allocating a GPU."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_dflash_artifacts import (
    INTEGER_TYPES, ROOT, check_tensor, integer, metadata, product,
    read_gguf, read_json, scalar, tokenizer, verify_file,
)


def cache_profile(document, architecture):
    if metadata(document, "general.architecture", [["STRING"]]) != architecture:
        raise ValueError(f"expected architecture {architecture}")
    prefix = architecture+"."
    layers = scalar(document, prefix+"block_count", maximum=1024)
    pattern = metadata(document, prefix+"attention.sliding_window_pattern", [["ARRAY", "BOOL"]])
    if not isinstance(pattern, list) or len(pattern) != layers or any(type(local) is not bool for local in pattern):
        raise ValueError("invalid sliding-window pattern")
    head_types = [[kind] for kind in INTEGER_TYPES]+[["ARRAY", kind] for kind in INTEGER_TYPES]
    heads = metadata(document, prefix+"attention.head_count_kv", head_types)
    if type(heads) is int:
        heads = [heads]*layers
    if not isinstance(heads, list) or len(heads) != layers:
        raise ValueError("invalid per-layer KV head count")
    query_heads = scalar(document, prefix+"attention.head_count")
    for count in heads:
        if query_heads % integer(count, "KV heads"):
            raise ValueError("query heads are not divisible by KV heads")
    full = (scalar(document, prefix+"attention.key_length"), scalar(document, prefix+"attention.value_length"))
    local = (scalar(document, prefix+"attention.key_length_swa"), scalar(document, prefix+"attention.value_length_swa"))
    return {
        "layers": layers, "pattern": pattern, "query_heads": query_heads,
        "geometry": [(*(local if windowed else full), heads[i]) for i, windowed in enumerate(pattern)],
        "window": scalar(document, prefix+"attention.sliding_window"),
        "context": scalar(document, prefix+"context_length"),
        "hidden": scalar(document, prefix+"embedding_length"),
        "shared": scalar(document, prefix+"attention.shared_kv_layers", 0, layers),
    }


def validate_pair(target, assistant):
    target_cache = cache_profile(target, "gemma4")
    draft_cache = cache_profile(assistant, "gemma4-assistant")
    if target_cache["shared"]:
        raise ValueError("initial Gemma target profile does not admit inter-layer KV aliases")
    if draft_cache["shared"] != draft_cache["layers"] or scalar(
            assistant, "gemma4-assistant.nextn_predict_layers") != draft_cache["layers"]:
        raise ValueError("assistant must share all its blocks with target sources")
    if draft_cache["context"] < target_cache["context"] or draft_cache["window"] != target_cache["window"]:
        raise ValueError("assistant and target cache coverage differ")
    if scalar(assistant, "gemma4-assistant.embedding_length_out") != target_cache["hidden"]:
        raise ValueError("assistant target hidden width differs")
    target_tokens, draft_tokens = tokenizer(target), tokenizer(assistant)
    for tokens in (target_tokens, draft_tokens):
        if not isinstance(tokens.get("token_text_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", tokens["token_text_sha256"]):
            raise ValueError("missing ordered token text fingerprint")
    # Match the native assistant's token-ID contract; raw token attributes are reported separately.
    if any(target_tokens[name] != draft_tokens[name] for name in
           ("model", "count", "token_text_sha256", "end_token_ids")):
        raise ValueError("ordered tokenizer identities differ")
    vocab = tokenizer(target)["count"]
    for document, hidden in ((target, target_cache["hidden"]), (assistant, draft_cache["hidden"])):
        check_tensor(document, "token_embd.weight", [hidden, vocab])
    for layer, geometry in enumerate(target_cache["geometry"]):
        check_tensor(target, f"blk.{layer}.attn_k.weight", [target_cache["hidden"], product(
            [geometry[0], geometry[2]], "target KV projection")])
    check_tensor(assistant, "nextn.pre_projection.weight", [product([2, target_cache["hidden"]], "input projection"), draft_cache["hidden"]])
    check_tensor(assistant, "nextn.post_projection.weight", [draft_cache["hidden"], target_cache["hidden"]])
    sources = []
    for layer, local in enumerate(draft_cache["pattern"]):
        candidates = [i for i, role in enumerate(target_cache["pattern"]) if role == local]
        if not candidates:
            raise ValueError("target has no source for an assistant cache role")
        source = candidates[-1]
        if draft_cache["geometry"][layer] != target_cache["geometry"][source]:
            raise ValueError("assistant and target source geometry differ")
        sources.append(source)
        width = product([draft_cache["geometry"][layer][0], draft_cache["query_heads"]], "query width")
        check_tensor(assistant, f"blk.{layer}.attn_q.weight", [draft_cache["hidden"], width])
        check_tensor(assistant, f"blk.{layer}.attn_output.weight", [width, draft_cache["hidden"]])
        for name in ("attn_k.weight", "attn_v.weight", "attn_qkv.weight"):
            if f"blk.{layer}.{name}" in assistant["tensors"]:
                raise ValueError("read-only assistant contains a KV producer projection")
    return {
        "metadata_compatible": True, "runtime_qualified": False,
        "target_layers": target_cache["layers"], "target_context": target_cache["context"],
        "full_layer_ids": [i for i, role in enumerate(target_cache["pattern"]) if not role],
        "local_layer_ids": [i for i, role in enumerate(target_cache["pattern"]) if role],
        "target_geometry": [list(shape) for shape in target_cache["geometry"]],
        "sliding_window": target_cache["window"],
        "assistant_blocks": draft_cache["layers"], "assistant_source_layer_ids": sources,
        "new_assistant_kv_histories": 0, "token_ids_sha256": tokenizer(target)["token_ids_sha256"],
        "token_text_sha256": target_tokens["token_text_sha256"],
        "raw_token_attributes_match": target_tokens["token_ids_sha256"] == draft_tokens["token_ids_sha256"],
    }


def inspect_artifacts(target, assistant, manifest):
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported artifact manifest")
    pins = manifest.get("artifacts")
    if not isinstance(pins, dict) or set(pins) != {"target", "assistant"}:
        raise ValueError("manifest requires target and assistant pins")
    paths = {"target": target, "assistant": assistant}
    def signature(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    before = {role: signature(path) for role, path in paths.items()}
    checksums = {role: verify_file(path, pins[role]) for role, path in paths.items()}
    documents = {role: read_gguf(path, (b"<eos>", b"<end_of_turn>")) for role, path in paths.items()}
    report = validate_pair(documents["target"], documents["assistant"])
    for role, path in paths.items():
        if signature(path) != before[role]:
            raise ValueError(f"artifact changed during inspection: {role}")
        report[role+"_tensor_bytes"] = sum(integer(item["bytes"], "tensor bytes", maximum=2**63-1)
            for item in documents[role]["tensors"].values())
    report["sha256"] = checksums
    report["tensor_bytes_are_gpu_allocations"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"benchmarks/data/gemma4-31b-artifacts.json")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_artifacts(args.target, args.assistant, read_json(args.manifest)), indent=2))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"Gemma artifact audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
