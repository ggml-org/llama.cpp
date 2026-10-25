#!/usr/bin/env python3
"""Audit pinned Qwen/DFlash2 metadata without loading model weights or a GPU."""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader

INT_MAX = 2**31-1
SIZE_MAX = 2**63-1
INTEGER_TYPES = {"INT32", "UINT32", "INT64", "UINT64"}


def integer(value, name, minimum=1, maximum=INT_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}: expected integer in [{minimum}, {maximum}]")
    return value


def product(values, name, maximum=SIZE_MAX):
    result = 1
    for value in values:
        value = integer(value, name, maximum=maximum)
        if result > maximum//value:
            raise ValueError(f"{name}: size overflow")
        result *= value
    return result


def metadata(document, name, types):
    fields = document.get("metadata")
    if not isinstance(fields, dict) or name not in fields:
        raise ValueError(f"missing metadata: {name}")
    item = fields[name]
    if not isinstance(item, dict) or item.get("types") not in types or "value" not in item:
        raise ValueError(f"invalid metadata type: {name}")
    return item["value"]


def scalar(document, name, minimum=1, maximum=INT_MAX):
    value = metadata(document, name, [[kind] for kind in INTEGER_TYPES])
    return integer(value, name, minimum, maximum)


def check_tensor(document, name, expected):
    tensors = document.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("invalid tensor inventory")
    item = tensors.get(name)
    if not isinstance(item, dict):
        raise ValueError(f"missing tensor: {name}")
    shape = item.get("shape")
    if not isinstance(shape, list) or not 1 <= len(shape) <= 4:
        raise ValueError(f"invalid tensor rank: {name}")
    for extent in shape:
        integer(extent, name)
    if shape != list(expected):
        raise ValueError(f"tensor shape differs: {name}: {shape} != {list(expected)}")
    try:
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType[item["type"]]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid tensor type: {name}") from exc
    elements = product(shape, name)
    if shape[0] % block:
        raise ValueError(f"invalid quant block alignment: {name}")
    expected_bytes = product([elements//block, size], name)
    if integer(item.get("bytes"), name, maximum=SIZE_MAX) != expected_bytes:
        raise ValueError(f"tensor byte count differs: {name}")


def tokenizer(document):
    item = document.get("tokenizer")
    if not isinstance(item, dict) or not isinstance(item.get("model"), str) or not item["model"]:
        raise ValueError("missing tokenizer identity")
    integer(item.get("count"), "token count", maximum=1000000)
    if not isinstance(item.get("token_ids_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", item["token_ids_sha256"]):
        raise ValueError("invalid ordered-token fingerprint")
    ends = item.get("end_token_ids")
    if not isinstance(ends, list) or not ends or len(ends) > item["count"]:
        raise ValueError("missing end-marker identities")
    for index in ends:
        integer(index, "end-marker ID", 0, item["count"]-1)
    if len(set(ends)) != len(ends):
        raise ValueError("duplicate end-marker ID")
    return item


def validate_pair(target, draft, publisher):
    if not all(isinstance(item, dict) for item in (target, draft, publisher)):
        raise ValueError("profile documents must be objects")
    if metadata(target, "general.architecture", [["STRING"]]) != "qwen35":
        raise ValueError("initial target profile requires dense Qwen35")
    if metadata(draft, "general.architecture", [["STRING"]]) != "dflash":
        raise ValueError("draft architecture is not dflash")
    if publisher.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("publisher configuration is not DFlash2")
    config = publisher.get("dflash_config")
    if not isinstance(config, dict):
        raise ValueError("missing dflash_config")
    hidden = scalar(target, "qwen35.embedding_length")
    total_layers = scalar(target, "qwen35.block_count", maximum=1024)
    nextn = scalar(target, "qwen35.nextn_predict_layers", 0, total_layers-1) if "qwen35.nextn_predict_layers" in target["metadata"] else 0
    trunk = total_layers-nextn
    if integer(publisher.get("num_target_layers"), "num_target_layers", maximum=1024) != trunk:
        raise ValueError("draft was trained for a different target layer count")
    if integer(publisher.get("hidden_size"), "hidden_size") != hidden:
        raise ValueError("target/draft hidden widths differ for this profile")
    source_ids = config.get("target_layer_ids")
    if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 1024:
        raise ValueError("invalid target feature list")
    expected_ids = [integer(index, "target_layer_ids", 0, trunk-2)+1 for index in source_ids]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate feature layers")
    actual_ids = metadata(draft, "dflash.target_layers", [["ARRAY", kind] for kind in INTEGER_TYPES])
    if not isinstance(actual_ids, list):
        raise ValueError("invalid converted feature list")
    for index in actual_ids:
        integer(index, "dflash.target_layers", 1, trunk-1)
    if actual_ids != expected_ids:
        raise ValueError("feature order or output-to-input layer mapping differs")
    encoded_width = product([len(expected_ids), hidden], "encoded input width", INT_MAX)
    if tokenizer(target) != tokenizer(draft):
        raise ValueError("ordered tokenizer identities differ")
    vocab = target["tokenizer"]["count"]
    if integer(publisher.get("vocab_size"), "vocab_size", maximum=1000000) != vocab:
        raise ValueError("publisher vocabulary differs")
    properties = {
        "block_count": ("num_hidden_layers", publisher),
        "embedding_length": ("hidden_size", publisher),
        "feed_forward_length": ("intermediate_size", publisher),
        "context_length": ("max_position_embeddings", publisher),
        "attention.head_count": ("num_attention_heads", publisher),
        "attention.head_count_kv": ("num_key_value_heads", publisher),
        "attention.key_length": ("head_dim", publisher),
        "attention.value_length": ("head_dim", publisher),
        "attention.sliding_window": ("sliding_window", publisher),
        "block_size": ("block_size", config),
        "conv_kernel_size": ("conv_kernel_size", config),
        "conv_group_size": ("conv_group_size", config),
        "selector_rank": ("selector_rank", config),
        "selector_top_k": ("selector_top_k", config),
    }
    checked = {}
    for suffix, (source, container) in properties.items():
        expected = integer(container.get(source), source)
        actual = scalar(draft, "dflash."+suffix)
        if actual != expected:
            raise ValueError(f"publisher/GGUF metadata differs: {suffix}")
        checked[suffix] = actual
    layers = checked["block_count"]
    if layers > 1024 or checked["block_size"] < 2:
        raise ValueError("invalid draft layer or block count")
    heads, kv_heads = checked["attention.head_count"], checked["attention.head_count_kv"]
    if heads % kv_heads or hidden % checked["conv_group_size"]:
        raise ValueError("invalid grouped-head or convolution geometry")
    if checked["attention.sliding_window"] > checked["context_length"] or scalar(target, "qwen35.context_length") > checked["context_length"]:
        raise ValueError("draft context/window cannot cover target native context")
    if publisher.get("is_causal") is not False or publisher.get("use_sliding_window") is not True:
        raise ValueError("publisher must specify non-causal windowed attention")
    causal = metadata(draft, "dflash.attention.causal", [["BOOL"]])
    if causal is not False:
        raise ValueError("draft attention is not non-causal")
    pattern = metadata(draft, "dflash.attention.sliding_window_pattern", [["ARRAY", "BOOL"]])
    if not isinstance(pattern, list) or len(pattern) != layers or any(value is not True for value in pattern):
        raise ValueError("draft cache is not uniformly bounded SWA")
    if publisher.get("layer_types") != ["sliding_attention"]*layers:
        raise ValueError("publisher layer types differ")
    mask_id = integer(config.get("mask_token_id"), "mask_token_id", 0, vocab-1)
    if scalar(draft, "tokenizer.ggml.mask_token_id", 0, vocab-1) != mask_id:
        raise ValueError("special token differs: mask_token_id")
    publisher_eos = integer(publisher.get("eos_token_id"), "eos_token_id", 0, vocab-1)
    gguf_eos = scalar(draft, "tokenizer.ggml.eos_token_id", 0, vocab-1)
    if gguf_eos != scalar(target, "tokenizer.ggml.eos_token_id", 0, vocab-1):
        raise ValueError("target/draft EOS flags differ")
    if publisher_eos not in target["tokenizer"]["end_token_ids"] or gguf_eos not in target["tokenizer"]["end_token_ids"]:
        raise ValueError("EOS alias is not a recognized Qwen end marker")
    rank, top_k = checked["selector_rank"], checked["selector_top_k"]
    lattice = product([top_k, top_k+1], "selector lattice", INT_MAX)
    if lattice > hidden or top_k > vocab:
        raise ValueError("selector lattice does not fit hidden/vocabulary width")
    query_width = product([checked["attention.key_length"], heads], "query projection", INT_MAX)
    kv_width = product([checked["attention.key_length"], kv_heads], "KV projection", INT_MAX)
    projected = product([2, checked["conv_kernel_size"], hidden//checked["conv_group_size"]], "convolution projection", INT_MAX)
    for name in ("token_embd.weight", "output.weight"):
        check_tensor(target, name, [hidden, vocab])
        if name in draft.get("tensors", {}):
            check_tensor(draft, name, [hidden, vocab])
    for name, shape in {
        "fc.weight": [encoded_width, hidden], "enc.output_norm.weight": [hidden],
        "output_norm.weight": [hidden], "selector_hidden.weight": [hidden, rank],
        "selector_predecessor.weight": [rank, vocab], "selector_successor.weight": [rank, vocab],
    }.items():
        check_tensor(draft, name, shape)
    for layer in range(layers):
        for name, shape in {
            "attn_norm.weight": [hidden], "attn_q_norm.weight": [checked["attention.key_length"]],
            "attn_k_norm.weight": [checked["attention.key_length"]],
            "attn_q.weight": [hidden, query_width], "attn_k.weight": [hidden, kv_width],
            "attn_v.weight": [hidden, kv_width], "attn_output.weight": [query_width, hidden],
            "ffn_norm.weight": [hidden], "ffn_gate.weight": [hidden, checked["feed_forward_length"]],
            "ffn_up.weight": [hidden, checked["feed_forward_length"]],
            "ffn_down.weight": [checked["feed_forward_length"], hidden],
            "attn_conv_base": [hidden, checked["conv_kernel_size"], 2],
            "ffn_conv_base": [hidden, checked["conv_kernel_size"], 2],
            "attn_conv_proj.weight": [hidden, projected], "ffn_conv_proj.weight": [hidden, projected],
        }.items():
            check_tensor(draft, f"blk.{layer}.{name}", shape)
    return {
        "metadata_compatible": True, "runtime_qualified": False,
        "vocab_size": vocab, "tokenizer_model": target["tokenizer"]["model"],
        "token_ids_sha256": target["tokenizer"]["token_ids_sha256"],
        "block_size": checked["block_size"], "max_draft_tokens": checked["block_size"]-1,
        "target_input_layers": expected_ids, "encoded_input_width": encoded_width,
        "draft_cache_role": "bounded_resident", "draft_kv_aliases_target": False,
        "draft_layers": layers, "window": checked["attention.sliding_window"],
        "head_dim": checked["attention.key_length"], "kv_heads": kv_heads, "query_heads": heads,
        "target_trunk_layers": trunk, "publisher_eos_token_id": publisher_eos,
        "gguf_eos_token_id": gguf_eos, "mask_token_id": mask_id,
        "warnings": ["Publisher and GGUF select different recognized EOS markers"] if publisher_eos != gguf_eos else [],
        "borrowed_target_weights": [name for name in ("token_embd.weight", "output.weight") if name not in draft["tensors"]],
    }


def read_gguf(path, end_tokens=(b"<|endoftext|>", b"<|im_end|>")):
    reader = GGUFReader(path)
    arch = reader.get_field("general.architecture")
    if arch is None or [kind.name for kind in arch.types] != ["STRING"]:
        raise ValueError("missing GGUF architecture")
    architecture = arch.contents()
    tokens = reader.get_field("tokenizer.ggml.tokens")
    model = reader.get_field("tokenizer.ggml.model")
    types = reader.get_field("tokenizer.ggml.token_type")
    if tokens is None or [kind.name for kind in tokens.types] != ["ARRAY", "STRING"] or not 1 <= len(tokens.data) <= 1000000:
        raise ValueError("invalid ordered vocabulary")
    if model is None or [kind.name for kind in model.types] != ["STRING"]:
        raise ValueError("missing tokenizer model")
    if types is None or [kind.name for kind in types.types] != ["ARRAY", "INT32"] or len(types.data) != len(tokens.data):
        raise ValueError("invalid tokenizer token types")
    digest = hashlib.sha256()
    text_digest = hashlib.sha256()
    end_ids = []
    for index, (token, token_type) in enumerate(zip(tokens.data, types.data)):
        raw = tokens.parts[token].tobytes()
        if raw in end_tokens:
            end_ids.append(index)
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
        digest.update(struct.pack("<i", int(types.parts[token_type][0])))
        text_digest.update(struct.pack("<Q", len(raw)))
        text_digest.update(raw)
    fields = {}
    for name, item in reader.fields.items():
        if name == "general.architecture" or name.startswith(architecture+".") or name in ("tokenizer.ggml.mask_token_id", "tokenizer.ggml.eos_token_id"):
            if len(item.data) > 1024:
                raise ValueError(f"unbounded model metadata: {name}")
            fields[name] = {"types": [kind.name for kind in item.types], "value": item.contents()}
    tensors = {item.name: {"shape": [int(value) for value in item.shape],
                          "type": item.tensor_type.name, "bytes": int(item.n_bytes)}
               for item in reader.tensors}
    return {"metadata": fields, "tensors": tensors,
            "tokenizer": {"model": model.contents(), "count": len(tokens.data),
                          "token_ids_sha256": digest.hexdigest(),
                          "token_text_sha256": text_digest.hexdigest(), "end_token_ids": end_ids}}


def verify_file(path, pin):
    if not isinstance(pin, dict):
        raise ValueError("invalid artifact identity")
    expected_size = integer(pin.get("size"), "artifact size", maximum=SIZE_MAX)
    expected_sha = pin.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha):
        raise ValueError("invalid pinned SHA-256")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != expected_size:
            raise ValueError(f"artifact size differs: {path}")
        digest = hashlib.sha256()
        while block := stream.read(1048576):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"artifact changed during verification: {path}")
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"artifact checksum differs: {path}")
    return expected_sha


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported artifact manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"target", "draft", "draft_config"}:
        raise ValueError("manifest requires target, draft and draft_config")
    for role, pin in artifacts.items():
        if not isinstance(pin, dict):
            raise ValueError(f"invalid artifact identity: {role}")
        for name, pattern in [("revision", "[0-9a-f]{40}"), ("sha256", "[0-9a-f]{64}"),
                              ("repo", "[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"),
                              ("filename", "[A-Za-z0-9][A-Za-z0-9_.-]*")]:
            if not isinstance(pin.get(name), str) or not re.fullmatch(pattern, pin[name]):
                raise ValueError(f"invalid pinned {name}: {role}")
        integer(pin.get("size"), role+" size", maximum=SIZE_MAX)


def read_json(path, maximum=65536):
    if not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"JSON input must be 1 to {maximum} bytes: {path}")
    with path.open("rb") as stream:
        raw = stream.read(maximum+1)
    if len(raw) > maximum:
        raise ValueError("JSON input changed or grew")
    def unique_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key: {name}")
            result[name] = value
        return result
    result = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(result, dict):
        raise ValueError("JSON document must be an object")
    return result


def inspect_artifacts(paths, manifest):
    validate_manifest(manifest)
    if set(paths) != set(manifest["artifacts"]) or not all(isinstance(path, Path) for path in paths.values()):
        raise ValueError("artifact paths must name all three pinned inputs")
    def signature(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    before = {role: signature(path) for role, path in paths.items()}
    checksums = {role: verify_file(path, manifest["artifacts"][role]) for role, path in paths.items()}
    target, draft = read_gguf(paths["target"]), read_gguf(paths["draft"])
    report = validate_pair(target, draft, read_json(paths["draft_config"]))
    for role, path in paths.items():
        if signature(path) != before[role]:
            raise ValueError(f"artifact changed during metadata inspection: {role}")
    report["sha256"] = checksums
    for role, document in (("target", target), ("draft", draft)):
        total = 0
        for item in document["tensors"].values():
            size = integer(item["bytes"], "tensor bytes", maximum=SIZE_MAX)
            if size > SIZE_MAX-total:
                raise ValueError("total tensor bytes overflow")
            total += size
        report[role+"_tensor_bytes"] = total
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"benchmarks/data/dflash2-qwen27b-artifacts.json")
    args = parser.parse_args()
    try:
        manifest = read_json(args.manifest)
        paths = {"target": args.target, "draft": args.draft, "draft_config": args.draft_config}
        report = inspect_artifacts(paths, manifest)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"artifact audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
