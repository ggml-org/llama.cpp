import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "inspect_dflash_artifacts.py"
SPEC = importlib.util.spec_from_file_location("dflash_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def field(value, kind="UINT32"):
    return {"types": [kind], "value": value}


def tensor(*shape):
    return {"shape": list(shape), "type": "F32", "bytes": math.prod(shape)*4}


def profiles():
    tokenizer = {"model": "gpt2", "count": 128, "token_ids_sha256": "a"*64, "end_token_ids": [1, 2]}
    target = {
        "metadata": {"general.architecture": field("qwen35", "STRING"),
                     "qwen35.block_count": field(9), "qwen35.nextn_predict_layers": field(1),
                     "qwen35.embedding_length": field(32), "qwen35.context_length": field(4096),
                     "tokenizer.ggml.eos_token_id": field(1)},
        "tokenizer": copy.deepcopy(tokenizer),
        "tensors": {"token_embd.weight": tensor(32, 128), "output.weight": tensor(32, 128)},
    }
    publisher = {
        "architectures": ["DFlash2DraftModel"], "hidden_size": 32, "head_dim": 8,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 64, "num_target_layers": 8, "max_position_embeddings": 4096,
        "layer_types": ["sliding_attention"]*2, "sliding_window": 128, "use_sliding_window": True,
        "is_causal": False, "eos_token_id": 1, "vocab_size": 128,
        "dflash_config": {"block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
                          "selector_rank": 8, "selector_top_k": 2, "mask_token_id": 127,
                          "target_layer_ids": [1, 5]},
    }
    metadata = {"general.architecture": field("dflash", "STRING")}
    for name, value in {
        "block_count": 2, "embedding_length": 32, "feed_forward_length": 64,
        "context_length": 4096, "attention.head_count": 4, "attention.head_count_kv": 2,
        "attention.key_length": 8, "attention.value_length": 8,
        "attention.sliding_window": 128, "block_size": 8, "conv_kernel_size": 2,
        "conv_group_size": 16, "selector_rank": 8, "selector_top_k": 2,
    }.items():
        metadata["dflash."+name] = field(value)
    metadata["dflash.attention.causal"] = field(False, "BOOL")
    metadata["dflash.attention.sliding_window_pattern"] = {"types": ["ARRAY", "BOOL"], "value": [True, True]}
    metadata["dflash.target_layers"] = {"types": ["ARRAY", "INT32"], "value": [2, 6]}
    metadata["tokenizer.ggml.mask_token_id"] = field(127)
    metadata["tokenizer.ggml.eos_token_id"] = field(1)
    tensors = {"fc.weight": tensor(64, 32), "enc.output_norm.weight": tensor(32),
               "output_norm.weight": tensor(32), "selector_hidden.weight": tensor(32, 8),
               "selector_predecessor.weight": tensor(8, 128),
               "selector_successor.weight": tensor(8, 128)}
    for layer in range(2):
        for name, shape in {
            "attn_norm.weight": (32,), "attn_q_norm.weight": (8,), "attn_k_norm.weight": (8,),
            "attn_q.weight": (32, 32), "attn_k.weight": (32, 16), "attn_v.weight": (32, 16),
            "attn_output.weight": (32, 32), "ffn_norm.weight": (32,),
            "ffn_gate.weight": (32, 64), "ffn_up.weight": (32, 64), "ffn_down.weight": (64, 32),
            "attn_conv_base": (32, 2, 2), "ffn_conv_base": (32, 2, 2),
            "attn_conv_proj.weight": (32, 8), "ffn_conv_proj.weight": (32, 8),
        }.items():
            tensors[f"blk.{layer}.{name}"] = tensor(*shape)
    draft = {"metadata": metadata, "tokenizer": copy.deepcopy(tokenizer), "tensors": tensors}
    return target, draft, publisher


class ProfileTests(unittest.TestCase):
    def test_valid_pair_maps_features_without_mutation_or_kv_aliasing(self):
        documents = profiles()
        saved = copy.deepcopy(documents)
        result = AUDIT.validate_pair(*documents)
        self.assertEqual(result["block_size"], 8)
        self.assertEqual(result["max_draft_tokens"], 7)
        self.assertEqual(result["target_input_layers"], [2, 6])
        self.assertFalse(result["draft_kv_aliases_target"])
        self.assertEqual(result["vocab_size"], 128)
        self.assertEqual(result["token_ids_sha256"], "a"*64)
        self.assertEqual(documents, saved)

    def test_required_fields_cannot_silently_use_loader_defaults(self):
        for key in profiles()[1]["metadata"]:
            with self.subTest(key=key):
                target, draft, config = profiles()
                del draft["metadata"][key]
                with self.assertRaises(ValueError):
                    AUDIT.validate_pair(target, draft, config)

    def test_scalar_type_confusion_and_unbounded_counts_are_rejected(self):
        for value, kind in [(8.0, "FLOAT32"), ("8", "STRING"), (True, "BOOL"),
                            (True, "UINT32"), (0, "UINT32"), (-1, "INT32"), (2**32, "UINT64")]:
            with self.subTest(value=value, kind=kind):
                target, draft, config = profiles()
                draft["metadata"]["dflash.block_size"] = field(value, kind)
                with self.assertRaises(ValueError):
                    AUDIT.validate_pair(target, draft, config)

    def test_wrong_target_or_draft_architecture_is_rejected(self):
        for role, architecture in [(0, "gemma4"), (0, "qwen35moe"), (1, "gemma4-assistant")]:
            target, draft, config = profiles()
            (target, draft)[role]["metadata"]["general.architecture"] = field(architecture, "STRING")
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft, config)
        target, draft, config = profiles()
        config["architectures"] = ["DFlashDraftModel"]
        with self.assertRaises(ValueError):
            AUDIT.validate_pair(target, draft, config)

    def test_publisher_to_gguf_feature_order_must_match(self):
        for ids in [[1, 5], [6, 2], [2, 2], [0, 6], [2, 8], [2.0, 6], [True, 6], [2]]:
            with self.subTest(ids=ids):
                target, draft, config = profiles()
                draft["metadata"]["dflash.target_layers"]["value"] = ids
                with self.assertRaises(ValueError):
                    AUDIT.validate_pair(target, draft, config)

    def test_feature_array_requires_an_integer_element_type(self):
        target, draft, config = profiles()
        draft["metadata"]["dflash.target_layers"]["types"] = ["ARRAY", "FLOAT32"]
        with self.assertRaises(ValueError):
            AUDIT.validate_pair(target, draft, config)

    def test_noncausal_and_all_sliding_attention_are_required(self):
        for mutation in ["causal", "one_full", "wrong_pattern_type", "short_pattern"]:
            with self.subTest(mutation=mutation):
                target, draft, config = profiles()
                if mutation == "causal":
                    draft["metadata"]["dflash.attention.causal"] = field(True, "BOOL")
                elif mutation == "one_full":
                    draft["metadata"]["dflash.attention.sliding_window_pattern"]["value"][0] = False
                elif mutation == "wrong_pattern_type":
                    draft["metadata"]["dflash.attention.sliding_window_pattern"]["types"] = ["ARRAY", "UINT32"]
                else:
                    draft["metadata"]["dflash.attention.sliding_window_pattern"]["value"].pop()
                with self.assertRaises(ValueError):
                    AUDIT.validate_pair(target, draft, config)

    def test_tokenizer_identity_is_not_just_vocabulary_size(self):
        for key, value in [("count", 127), ("model", "llama"), ("token_ids_sha256", "b"*64)]:
            target, draft, config = profiles()
            draft["tokenizer"][key] = value
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft, config)

    def test_required_projection_and_selector_shapes_are_checked(self):
        for name in ["fc.weight", "selector_hidden.weight", "selector_predecessor.weight",
                     "selector_successor.weight", "blk.0.attn_q.weight", "blk.1.attn_k.weight",
                     "blk.0.attn_conv_proj.weight", "blk.1.ffn_conv_base"]:
            with self.subTest(name=name):
                target, draft, config = profiles()
                draft["tensors"][name]["shape"][0] += 1
                with self.assertRaises(ValueError):
                    AUDIT.validate_pair(target, draft, config)

    def test_missing_required_tensors_are_rejected(self):
        for name in profiles()[1]["tensors"]:
            with self.subTest(name=name):
                target, draft, config = profiles()
                del draft["tensors"][name]
                with self.assertRaises(ValueError):
                    AUDIT.validate_pair(target, draft, config)

    def test_geometry_and_encoded_input_overflow_are_rejected(self):
        for key, value in [("hidden_size", 2**31-1), ("num_attention_heads", 2**31),
                           ("selector_top_k", 2**31-1), ("conv_group_size", 3)]:
            target, draft, config = profiles()
            if key in config["dflash_config"]:
                config["dflash_config"][key] = value
            else:
                config[key] = value
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft, config)

    def test_missing_or_malformed_publisher_values_are_rejected(self):
        for key in ["dflash_config", "layer_types", "num_target_layers", "head_dim", "sliding_window"]:
            target, draft, config = profiles()
            del config[key]
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft, config)
        for key, value in [("num_hidden_layers", True), ("vocab_size", "128"), ("is_causal", 0)]:
            target, draft, config = profiles()
            config[key] = value
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft, config)

    def test_shared_target_embedding_and_output_shapes_are_required(self):
        for name in ["token_embd.weight", "output.weight"]:
            target, draft, config = profiles()
            target["tensors"][name]["shape"][0] += 1
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft, config)

    def test_publisher_eos_alias_is_reported_but_only_for_end_markers(self):
        target, draft, config = profiles()
        config["eos_token_id"] = 2
        result = AUDIT.validate_pair(target, draft, config)
        self.assertEqual(result["publisher_eos_token_id"], 2)
        self.assertEqual(result["gguf_eos_token_id"], 1)
        self.assertTrue(result["warnings"])
        config["eos_token_id"] = 3
        with self.assertRaises(ValueError):
            AUDIT.validate_pair(target, draft, config)

    def test_target_and_draft_special_flags_must_still_match(self):
        target, draft, config = profiles()
        draft["metadata"]["tokenizer.ggml.eos_token_id"] = field(2)
        with self.assertRaises(ValueError):
            AUDIT.validate_pair(target, draft, config)

    def test_tensor_types_alignment_and_byte_counts_are_validated(self):
        for key, value in [("type", "INVALID"), ("bytes", True), ("bytes", 1), ("shape", [0, 32]),
                           ("shape", [2**31, 32]), ("shape", [True, 32])]:
            target, draft, config = profiles()
            draft["tensors"]["fc.weight"][key] = value
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft, config)
        target, draft, config = profiles()
        draft["tensors"] = []
        with self.assertRaises(ValueError):
            AUDIT.validate_pair(target, draft, config)


class IntegrityTests(unittest.TestCase):
    def audit_fixture(self, directory):
        documents = profiles()
        paths = {role: Path(directory)/name for role, name in
                 [("target", "target.gguf"), ("draft", "draft.gguf"), ("draft_config", "config.json")]}
        paths["target"].write_bytes(b"target")
        paths["draft"].write_bytes(b"draft")
        paths["draft_config"].write_text(json.dumps(documents[2]))
        artifacts = {}
        for role, path in paths.items():
            raw = path.read_bytes()
            artifacts[role] = {"repo": "fixture/model", "revision": "0"*40, "filename": path.name,
                               "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return paths, {"schema_version": 1, "artifacts": artifacts}, documents

    def test_audit_verifies_all_inputs_before_metadata_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, manifest, documents = self.audit_fixture(directory)
            with patch.object(AUDIT, "read_gguf", side_effect=documents[:2]) as reader:
                report = AUDIT.inspect_artifacts(paths, manifest)
            self.assertEqual(reader.call_count, 2)
            self.assertFalse(report["runtime_qualified"])
            self.assertEqual(set(report["sha256"]), set(paths))
            manifest["artifacts"]["draft"]["sha256"] = "0"*64
            with patch.object(AUDIT, "read_gguf") as reader, self.assertRaises(ValueError):
                AUDIT.inspect_artifacts(paths, manifest)
            reader.assert_not_called()

    def test_replaced_file_after_hash_cannot_publish_a_matching_report(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, manifest, documents = self.audit_fixture(directory)
            def reader(path):
                if path == paths["draft"]:
                    replacement = Path(directory)/"replacement"
                    replacement.write_bytes(b"target")
                    replacement.replace(paths["target"])
                    return documents[1]
                return documents[0]
            with patch.object(AUDIT, "read_gguf", side_effect=reader), self.assertRaises(ValueError):
                AUDIT.inspect_artifacts(paths, manifest)

    def test_checksums_and_sizes_are_verified_not_just_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.gguf"
            path.write_bytes(b"payload")
            pin = {"size": 7, "sha256": hashlib.sha256(b"payload").hexdigest()}
            self.assertEqual(AUDIT.verify_file(path, pin), pin["sha256"])
            for changed in [dict(pin, size=8), dict(pin, sha256="0"*64), dict(pin, size=True),
                            dict(pin, sha256="invalid"), None]:
                with self.subTest(pin=changed), self.assertRaises(ValueError):
                    AUDIT.verify_file(path, changed)

    def test_json_is_bounded_and_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            path.write_text('{"block": 8}')
            self.assertEqual(AUDIT.read_json(path), {"block": 8})
            for data in ["", "[]" , '{"block": 8, "block": 16}', " "*65537]:
                path.write_text(data)
                with self.assertRaises(ValueError):
                    AUDIT.read_json(path)

    def test_manifest_requires_immutable_revisions_and_all_artifacts(self):
        manifest = json.loads((SCRIPT.parent/"data/dflash2-qwen27b-artifacts.json").read_text())
        AUDIT.validate_manifest(manifest)
        for role in manifest["artifacts"]:
            for key, value in [("revision", "main"), ("size", True), ("sha256", "bad"), ("filename", "../model.gguf")]:
                changed = copy.deepcopy(manifest)
                changed["artifacts"][role][key] = value
                with self.subTest(role=role, key=key), self.assertRaises(ValueError):
                    AUDIT.validate_manifest(changed)
        for changed in [dict(manifest, schema_version=True), dict(manifest, artifacts=[]), {}]:
            with self.assertRaises(ValueError):
                AUDIT.validate_manifest(changed)

    def test_real_gguf_reader_preserves_order_and_tensor_metadata(self):
        import numpy as np
        from gguf import GGUFWriter
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for number, words in enumerate([["a", "b", "<|im_end|>"], ["b", "a", "<|im_end|>"]]):
                path = Path(directory)/f"{number}.gguf"
                writer = GGUFWriter(path, "dflash")
                writer.add_tokenizer_model("gpt2")
                writer.add_token_list(words)
                writer.add_token_types([1, 1, 3])
                writer.add_tensor("toy.weight", np.zeros((2, 32), dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                results.append(AUDIT.read_gguf(path))
            self.assertNotEqual(results[0]["tokenizer"]["token_ids_sha256"], results[1]["tokenizer"]["token_ids_sha256"])
            self.assertEqual(results[0]["tokenizer"]["end_token_ids"], [2])
            self.assertEqual(results[0]["tensors"]["toy.weight"], tensor(32, 2))


if __name__ == "__main__":
    unittest.main()
