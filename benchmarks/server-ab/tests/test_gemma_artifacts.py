import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "inspect_gemma_artifacts.py"
SPEC = importlib.util.spec_from_file_location("gemma_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def field(value, kind="UINT32"):
    return {"types": [kind], "value": value}


def tensor(*shape):
    return {"shape": list(shape), "type": "F32", "bytes": math.prod(shape)*4}


def profiles():
    def document(arch, hidden, pattern):
        metadata = {"general.architecture": field(arch, "STRING")}
        for name, value in {
            "block_count": len(pattern), "embedding_length": hidden, "context_length": 4096,
            "attention.head_count": 8, "attention.sliding_window": 1024,
            "attention.key_length": 64, "attention.value_length": 64,
            "attention.key_length_swa": 32, "attention.value_length_swa": 32,
            "attention.shared_kv_layers": 0, "tokenizer.ggml.eos_token_id": 1,
        }.items():
            metadata[arch+"."+name] = field(value)
        metadata[arch+".attention.head_count_kv"] = {
            "types": ["ARRAY", "UINT32"], "value": [4 if local else 2 for local in pattern]}
        metadata[arch+".attention.sliding_window_pattern"] = {
            "types": ["ARRAY", "BOOL"], "value": pattern}
        return {"metadata": metadata, "tensors": {"token_embd.weight": tensor(hidden, 128)},
                "tokenizer": {"model": "llama", "count": 128,
                    "token_ids_sha256": "a"*64, "token_text_sha256": "c"*64, "end_token_ids": [1, 2]}}
    target = document("gemma4", 128, [True]*5+[False]+[True]*5+[False])
    draft = document("gemma4-assistant", 32, [True, True, True, False])
    for name, value in {"embedding_length_out": 128, "nextn_predict_layers": 4,
                        "attention.shared_kv_layers": 4}.items():
        draft["metadata"]["gemma4-assistant."+name] = field(value)
    for index, local in enumerate([True]*5+[False]+[True]*5+[False]):
        target["tensors"][f"blk.{index}.attn_k.weight"] = tensor(128, (32 if local else 64)*(4 if local else 2))
    draft["tensors"]["nextn.pre_projection.weight"] = tensor(256, 32)
    draft["tensors"]["nextn.post_projection.weight"] = tensor(32, 128)
    for index, local in enumerate([True, True, True, False]):
        width = (32 if local else 64)*8
        draft["tensors"][f"blk.{index}.attn_q.weight"] = tensor(32, width)
        draft["tensors"][f"blk.{index}.attn_output.weight"] = tensor(width, 32)
    return target, draft


class GemmaArtifactTests(unittest.TestCase):
    def test_gguf_reader_separates_token_text_order_from_attributes(self):
        from gguf import GGUFWriter
        documents = []
        with tempfile.TemporaryDirectory() as temp:
            for index, (tokens, types) in enumerate((
                    (["<eos>","one","two"], [1,1,1]),
                    (["<eos>","one","two"], [3,1,1]),
                    (["<eos>","two","one"], [3,1,1]))):
                path = Path(temp)/f"tokens-{index}.gguf"
                writer = GGUFWriter(path, "gemma4")
                writer.add_tokenizer_model("gemma4")
                writer.add_token_list(tokens)
                writer.add_token_types(types)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                documents.append(AUDIT.read_gguf(path, (b"<eos>",))["tokenizer"])
        self.assertEqual(documents[0]["end_token_ids"], [0])
        self.assertNotEqual(documents[0]["token_ids_sha256"], documents[1]["token_ids_sha256"])
        self.assertEqual(documents[0]["token_text_sha256"], documents[1]["token_text_sha256"])
        self.assertNotEqual(documents[1]["token_text_sha256"], documents[2]["token_text_sha256"])

    def test_shared_sources_are_roles_not_assistant_layer_count(self):
        pair = profiles()
        before = copy.deepcopy(pair)
        result = AUDIT.validate_pair(*pair)
        self.assertEqual(result["full_layer_ids"], [5, 11])
        self.assertEqual(result["local_layer_ids"], [0,1,2,3,4,6,7,8,9,10])
        self.assertEqual(result["assistant_source_layer_ids"], [10,10,10,11])
        self.assertEqual(result["new_assistant_kv_histories"], 0)
        self.assertFalse(result["runtime_qualified"])
        self.assertEqual(pair, before)

    def test_source_mapping_follows_actual_pattern(self):
        target, draft = profiles()
        target["metadata"]["gemma4.attention.sliding_window_pattern"]["value"][-2:] = [False, True]
        target["metadata"]["gemma4.attention.head_count_kv"]["value"][-2:] = [2, 4]
        self.assertEqual(AUDIT.validate_pair(target, draft)["assistant_source_layer_ids"], [11,11,11,10])

    def test_wrong_architecture_and_missing_metadata_reject(self):
        for role in (0, 1):
            pair = profiles()
            pair[role]["metadata"]["general.architecture"] = field("qwen35", "STRING")
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(*pair)
        for role, prefix in ((0,"gemma4"),(1,"gemma4-assistant")):
            for suffix in ("block_count", "attention.key_length", "attention.head_count_kv",
                           "attention.sliding_window_pattern", "attention.shared_kv_layers"):
                pair = profiles()
                del pair[role]["metadata"][prefix+"."+suffix]
                with self.subTest(role=role, suffix=suffix), self.assertRaises(ValueError):
                    AUDIT.validate_pair(*pair)

    def test_counts_patterns_and_element_types_are_validated(self):
        for value in ([], [True], [True]*13, [True]*11+[1], "local"):
            target, draft = profiles()
            target["metadata"]["gemma4.attention.sliding_window_pattern"]["value"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)
        for value in (True, -1, 0, 2**32, 12.0):
            target, draft = profiles()
            target["metadata"]["gemma4.block_count"]["value"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)

    def test_geometry_and_missing_cache_sources_reject(self):
        for suffix, value in (("attention.key_length",32), ("attention.head_count",7),
                              ("attention.sliding_window",512), ("embedding_length_out",64),
                              ("attention.shared_kv_layers",3), ("nextn_predict_layers",3)):
            target, draft = profiles()
            draft["metadata"]["gemma4-assistant."+suffix] = field(value)
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)
        for local in (True, False):
            target, draft = profiles()
            target["metadata"]["gemma4.attention.sliding_window_pattern"]["value"] = [local]*12
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)

    def test_tokenizer_order_embedding_and_projection_mismatch_reject(self):
        for name, value in (("count",64), ("token_text_sha256","b"*64),
                            ("end_token_ids",[1]), ("model","gpt2")):
            target, draft = profiles()
            draft["tokenizer"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)

    def test_token_attribute_difference_is_reported_not_confused_with_token_reordering(self):
        target, draft = profiles()
        draft["tokenizer"]["token_ids_sha256"] = "b"*64
        result = AUDIT.validate_pair(target, draft)
        self.assertFalse(result["raw_token_attributes_match"])
        del draft["tokenizer"]["token_text_sha256"]
        with self.assertRaises(ValueError):
            AUDIT.validate_pair(target, draft)
        for name in ("token_embd.weight","nextn.pre_projection.weight","nextn.post_projection.weight",
                     "blk.3.attn_q.weight"):
            target, draft = profiles()
            draft["tensors"][name] = tensor(1, 1)
            with self.subTest(name=name), self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)

    def test_draft_kv_projections_and_target_aliasing_are_not_supported(self):
        for suffix in ("attn_k.weight", "attn_v.weight", "attn_qkv.weight"):
            target, draft = profiles()
            draft["tensors"]["blk.0."+suffix] = tensor(32, 32)
            with self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)
        target, draft = profiles()
        target["metadata"]["gemma4.attention.shared_kv_layers"] = field(1)
        with self.assertRaises(ValueError):
            AUDIT.validate_pair(target, draft)

    def test_missing_target_kv_projection_rejects_before_source_binding(self):
        for layer in (0, 5, 10, 11):
            target, draft = profiles()
            del target["tensors"][f"blk.{layer}.attn_k.weight"]
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                AUDIT.validate_pair(target, draft)


if __name__ == "__main__":
    unittest.main()
