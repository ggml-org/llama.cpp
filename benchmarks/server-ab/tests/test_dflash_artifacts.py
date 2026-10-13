import copy
import hashlib
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any
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


class OrdinaryControls(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("dflash_controls", SCRIPT.parent / "run_dflash_controls.py")
        assert spec is not None and spec.loader is not None
        self.controls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.controls)

    def test_configurable_batches_and_target_kv_are_not_rewritten(self):
        from types import SimpleNamespace
        args = SimpleNamespace(server=Path("server"),target=Path("target"),draft=Path("draft"),
                               port=12351,draft_k="f16",draft_v="f16",batch_size=512,
                               ubatch_size=128,target_k="q5_1",target_v="q4_1")
        command = self.controls.server_command(args,4096,3)
        for option,value in (("--batch-size","512"),("--ubatch-size","128"),
                             ("--cache-type-k","q5_1"),("--cache-type-v","q4_1")):
            self.assertEqual(command.count(option),1)
            self.assertEqual(command[command.index(option)+1],value)

    def test_invalid_profile_arguments_are_rejected_before_launch(self):
        from types import SimpleNamespace
        base = dict(server=Path("server"),target=Path("target"),draft=Path("draft"),
                    port=12351,draft_k="f16",draft_v="f16")
        for field,value in (("batch_size",0),("ubatch_size",-1),("batch_size",True),
                            ("ubatch_size",2**32),("target_k","not-a-type")):
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                self.controls.server_command(SimpleNamespace(**base,**{field:value}),4096,3)

    def test_native_command_preserves_profile_without_streaming(self):
        from types import SimpleNamespace
        args = SimpleNamespace(server=Path("server"), target=Path("target"), draft=Path("draft"),
                               port=12351, draft_k="f16", draft_v="f16")
        for depth in (1, 3, 7):
            command = self.controls.server_command(args, 4096, depth)
            for option, value in (("--spec-type", "draft-dflash"), ("--spec-draft-n-max", str(depth)),
                                  ("--cache-type-k", "q8_0"), ("--cache-type-v", "q4_0"),
                                  ("--spec-draft-type-k", "f16"), ("--spec-draft-type-v", "f16"),
                                  ("--batch-size", "256"), ("--ubatch-size", "256"), ("--fit", "off"),
                                  ("--n-gpu-layers", "999"), ("--spec-draft-ngl", "999")):
                self.assertEqual(command[command.index(option)+1], value)
            self.assertNotIn("--kv-stream-arena-mib", command)
            self.assertNotIn("--kv-stream-auxiliary-layers", command)
        with self.assertRaises(ValueError):
            self.controls.server_command(args, 4096, 8)

    def test_incomplete_or_nonfinite_results_are_not_successful(self):
        result = {"tokens": [4]*32, "timings": {"prompt_n": 2304, "predicted_n": 32,
                  "prompt_per_second": 500.0, "predicted_per_second": 50.0}}
        self.controls.validate_result(result, 2304, 32)
        for field in ("prompt_n", "predicted_n", "prompt_per_second", "predicted_per_second"):
            broken = {"tokens": result["tokens"], "timings": dict(result["timings"])}
            broken["timings"][field] = float("nan")
            with self.assertRaises(ValueError):
                self.controls.validate_result(broken, 2304, 32)
        with self.assertRaises(ValueError):
            self.controls.validate_result(dict(result, tokens=[4]), 2304, 32)

    def test_streamed_command_declares_one_arena_without_auxiliary_layers(self):
        from types import SimpleNamespace
        args = SimpleNamespace(server=Path("server"),target=Path("target"),draft=Path("draft"),
                               port=12351,draft_k="f16",draft_v="f16",arena_mib=1536)
        command=self.controls.server_command(args,16384,7)
        self.assertEqual(command[command.index("--kv-stream-arena-mib")+1],"1536")
        self.assertNotIn("--kv-stream-pool-mib",command)
        self.assertNotIn("--kv-stream-auxiliary-layers",command)

    def test_capacity_failure_does_not_hide_kernel_or_other_failure(self):
        self.assertEqual(self.controls.failure_kind("shared arena quota insufficient for phase 1 resource 5"),"quota_failure")
        for text in ("CUDA error: out of memory", "cudaMalloc failed: out of memory"):
            self.assertEqual(self.controls.failure_kind(text), "capacity_failure")
        for text in ("missing flash attention kernel", "invalid resource handle", "segmentation fault"):
            self.assertEqual(self.controls.failure_kind(text), "failed")

    def test_trace_requires_all_three_workloads(self):
        rows = [{"event": "process", "role": "target", "kind": "tokens", "rows": 4, "ret": 0},
                {"event": "process", "role": "draft", "kind": "features", "rows": 256, "ret": 0,
                 "feature_width": 25600},
                {"event": "process", "role": "draft", "kind": "tokens", "rows": 4, "ret": 0},
                {"event": "tensor", "role": "draft", "name": "lattice", "file": "lattice.bin"},
                {"event": "attention", "role": "draft", "query_rows": 4,
                 "visible_rows": [2048, 2048, 2048, 2048], "mask_file": "mask.bin"}]
        summary = self.controls.summarize_trace(rows, 3)
        self.assertEqual(summary["injection_rows"], [256])
        self.assertEqual(summary["draft_widths"], [4])
        self.assertEqual(summary["target_widths"], [4])
        for index in range(len(rows)):
            with self.assertRaises(ValueError):
                self.controls.summarize_trace(rows[:index]+rows[index+1:], 3)
        with self.assertRaises(ValueError):
            self.controls.summarize_trace(rows, 7)
        warm = {"event": "process", "role": "draft", "kind": "tokens", "rows": 2,
                "ret": 0, "warmup": True}
        self.assertEqual(self.controls.summarize_trace([warm]+rows, 3)["draft_widths"], [4])
        wider=copy.deepcopy(rows)
        wider[1]["rows"]=512
        self.assertEqual(self.controls.summarize_trace(wider,3,max_injection_rows=512)["injection_rows"],[512])
        with self.assertRaises(ValueError):
            self.controls.summarize_trace(wider,3,max_injection_rows=256)

    def test_trace_payload_checksums_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"a native control payload"
            (root/"payload.bin").write_bytes(content)
            row = {"file": "payload.bin", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            self.assertEqual(self.controls.validate_payloads(root, [row, {"nested": row}]), 1)
            for changed in (dict(row, file="../payload.bin"), dict(row, bytes=0), dict(row, sha256="0"*64)):
                with self.assertRaises(ValueError):
                    self.controls.validate_payloads(root, [changed])
            (root/"payload.bin").write_bytes(b"x"*len(content))
            with self.assertRaises(ValueError):
                self.controls.validate_payloads(root, [row])

    def test_matching_control_rejects_changed_block_and_workload(self):
        from types import SimpleNamespace
        args = SimpleNamespace(server=Path("server"), target=Path("target"), draft=Path("draft"),
                               port=12351, draft_k="f16", draft_v="f16")
        row = {"status": "ok", "mode": "capture", "context": 4096, "depth": 3,
               "prompt_tokens": 2304, "decode_tokens": 64, "uvm": False,
               "command": self.controls.server_command(args, 4096, 3),
               "result": {"tokens": [42]*64}}
        artifacts = {"sha256": {"target": "1"*64, "draft": "2"*64, "draft_config": "3"*64}}
        trace: list[dict[str, Any]] = [{"event": "process", "role": "target", "tokens": [42, 43, 44, 45], "positions": [100, 101, 102, 103]},
                 {"event": "attention", "visible_columns": [[0, 1, 3]], "mask": {"sha256": "a"*64}},
                 {"event": "memory", "devices": [{"free": 100}]}]
        self.controls.compare_control_rows(row, row, artifacts, artifacts, trace, trace)
        changed = copy.deepcopy(trace)
        changed[-1]["devices"][0]["free"] = 99
        self.controls.compare_control_rows(row, row, artifacts, artifacts, trace, changed)
        for index, field, value in ((0, "tokens", [42, 43, 46, 45]),
                                    (0, "positions", [100, 101, 103, 102]),
                                    (1, "visible_columns", [[0, 2, 3]]),
                                    (1, "mask", {"sha256": "b"*64})):
            changed = copy.deepcopy(trace)
            changed[index][field] = value
            with self.assertRaises(ValueError):
                self.controls.compare_control_rows(row, row, artifacts, artifacts, trace, changed)
        for field, value in (("context", 8192), ("depth", 7), ("prompt_tokens", 2303), ("uvm", True)):
            with self.assertRaises(ValueError):
                self.controls.compare_control_rows(row, dict(row, **{field: value}), artifacts, artifacts, trace, trace)
        with self.assertRaises(ValueError):
            self.controls.compare_control_rows(row, row, artifacts, {"sha256": {}}, trace, trace)

    def test_lattice_comparison_preserves_scores_across_unsorted_top_k(self):
        rows: list[list[float]] = [[0]*8, [4, 6, 10, 20, 10, 20, 0, 0], [8, 10, 1, 2, 3, 4, 0, 0]]
        reordered: list[list[float]] = [[0]*8, [6, 4, 20, 10, 20, 10, 0, 0], [10, 8, 4, 3, 2, 1, 0, 0]]
        def encode(value):
            return struct.pack("<24f", *(v for row in value for v in row))
        expected = self.controls.canonical_lattice(encode(rows), 3, 8, 2, 32)
        self.assertEqual(expected, self.controls.canonical_lattice(encode(reordered), 3, 8, 2, 32))
        changed = copy.deepcopy(reordered)
        changed[2][2] += 0.000001
        self.assertNotEqual(expected, self.controls.canonical_lattice(encode(changed), 3, 8, 2, 32))
        for index, value in ((1, 10), (6, 1), (0, float("nan")), (0, 99)):
            changed = copy.deepcopy(rows)
            changed[2][index] = value
            if index == 1:  # Duplicate candidate ID.
                changed[2][0] = 10
            with self.assertRaises(ValueError):
                self.controls.canonical_lattice(encode(changed), 3, 8, 2, 32)

    def test_fit_ledger_is_a_lower_bound_not_an_arena_guarantee(self):
        row = {"context": 4096, "depth": 3}
        trace = [
            {"event": "memory", "role": "target", "buffers": [
                {"buffer": "CUDA0", "model": 1000}], "devices": [{"name": "CUDA0", "total": 10000}],
                "components": [{"name": "target_kv", "buffer": "CUDA0", "bytes": 8192},
                               {"name": "target_recurrent", "buffer": "CUDA0", "bytes": 400}]},
            {"event": "memory", "role": "draft", "buffers": [
                {"buffer": "CUDA0", "model": 100, "context": 20}]}]
        ledger = self.controls.fit_ledger(row, trace, 262144)
        self.assertFalse(ledger["ordinary_full_context_fits"])
        self.assertEqual(ledger["ordinary_lower_bound_bytes"], 1000+100+524288+400+20)
        self.assertEqual(ledger["streamed_parent_upper_bound_before_scratch_bytes"], 10000-1000-100-100)
        self.assertEqual(ledger["target_elastic_upper_bound_before_scratch_bytes"], 10000-1000-100-100-20)
        self.assertFalse(ledger["streamed_fit_qualified"])
        with self.assertRaises(ValueError):
            self.controls.fit_ledger(dict(row, context=1000), trace, 262144)


if __name__ == "__main__":
    unittest.main()
