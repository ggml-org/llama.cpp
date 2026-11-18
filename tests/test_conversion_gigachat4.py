"""GigaChat4 conversion regressions with small CPU tensors and no model downloads."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from conversion import get_model_class
from conversion.base import ModelBase, gguf
from conversion.deepseek import DeepseekV2Model
from conversion.gigachat4 import GigaChat4Model


class TestGigaChat4Conversion(unittest.TestCase):
    def make_model(self, tensors=None, overrides=None, eager=False, ftype=gguf.LlamaFileType.ALL_F32):
        hparams = {
            "architectures": ["GigaChat4ForCausalLM"],
            "model_type": "gigachat4",
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "hidden_size": 4,
            "intermediate_size": 8,
            "vocab_size": 8,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "kv_lora_rank": 2,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 2,
            "v_head_dim": 2,
            "head_dim": 2,
            "moe_intermediate_size": 4,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_type": "GatedNorm",
            "rms_norm_eps": 1e-6,
            "layernorm_type": "pre_post",
            "layernorm_gating_weight": 2.0,
            "use_mla_scaling_factor": True,
            "gated_attention": True,
            "hidden_act": "silu",
            "full_attention_layers": [1, 3],
            "linear_attention_type": "GigaChat4GatedDeltaNet",
            "linear_gating_type": "gated_rmsnorm_sigmoid",
            "linear_sigmoid_gate_scale": 2.0,
            "linear_attn_o_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "swiglu_limit": 0.0,
            "final_logit_softcapping": None,
            "index_topk": None,
            "rope_interleave": True,
            "rope_theta": 100000.0,
            "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32, "mscale": 1.0, "mscale_all_dim": 1.0},
        }
        hparams.update(overrides or {})
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        (path / "config.json").write_text(json.dumps(hparams))
        save_file(tensors if tensors is not None else {"model.norm.weight": torch.ones(4)}, str(path / "model.safetensors"))
        return GigaChat4Model(path, ftype, path / "out.gguf", hparams=hparams, eager=eager)

    @staticmethod
    def sparse_config():
        return {
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "indexer_norm_type": "LlamaRMSNorm",
        }

    @staticmethod
    def converted(model):
        model.prepare_tensors()
        return {name: gguf.LazyNumpyTensor.to_eager(info.tensor).copy() for name, info in model.gguf_writer.tensors[0].items()}

    def test_norm_weights_and_gate_weights(self):
        names = (
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_self_attn_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.self_attn.norm.weight",
            "model.layers.1.self_attn.q_a_layernorm.weight",
            "model.layers.1.self_attn.kv_a_layernorm.weight",
        )
        weight = torch.tensor([-1.0, 0.0, 0.5, 2.0])
        gate_name = "model.norm.gate_up_projection.weight"
        gate = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        for eager in (False, True):
            with self.subTest(eager=eager):
                model = self.make_model({**{name: weight.clone() for name in names}, gate_name: gate}, eager=eager)
                result = self.converted(model)
                for name in names:
                    np.testing.assert_array_equal(result[model.map_tensor_name(name)], weight.numpy())
                np.testing.assert_array_equal(result["output_norm_gate_up.weight"], gate.numpy())

    def test_norm_gate_rank(self):
        gate = {"model.norm.gate_up_projection.weight": torch.ones((2, 4))}
        norm = {"model.norm.weight": torch.ones(4)}
        # a checkpoint with its gate, an adapter-like input without the output norm, a checkpoint missing the gate
        for tensors, rank in (({**norm, **gate}, 2), ({"model.layers.1.self_attn.q_a_proj.weight": torch.ones((4, 4))}, None), (norm, ValueError)):
            with self.subTest(tensors=sorted(tensors)):
                model = self.make_model(tensors)
                with patch.object(model, "set_vocab"):
                    if rank is ValueError:
                        with self.assertRaises(ValueError):
                            model.prepare_metadata(vocab_only=False)
                        continue
                    model.prepare_metadata(vocab_only=False)
                metadata = model.gguf_writer.kv_data[0]
                if rank is None:
                    self.assertNotIn("gigachat4.norm_gate_rank", metadata)
                else:
                    self.assertEqual(metadata["gigachat4.norm_gate_rank"].value, rank)

    def test_norm_gates_are_not_quantized(self):
        tensors = {
            "model.norm.gate_up_projection.weight": torch.ones(2, 32),
            "model.norm.gate_down_projection.weight": torch.ones(32, 2),
            "model.layers.0.mlp.down_proj.weight": torch.ones(32, 32),
        }
        model = self.make_model(tensors, overrides={"hidden_size": 32, "intermediate_size": 32}, ftype=gguf.LlamaFileType.MOSTLY_Q8_0)
        model.prepare_tensors()
        types = {name: info.dtype for name, info in model.gguf_writer.tensors[0].items()}
        self.assertEqual(types["output_norm_gate_up.weight"], gguf.GGMLQuantizationType.F32)
        self.assertEqual(types["output_norm_gate_down.weight"], gguf.GGMLQuantizationType.F32)
        self.assertEqual(types["blk.0.ffn_down.weight"], gguf.GGMLQuantizationType.Q8_0)

    def test_decoder_cleanup_metadata(self):
        model = self.make_model()
        (model.dir_model / "tokenizer_config.json").write_text(json.dumps({"clean_up_tokenization_spaces": True}))
        with patch.object(model, "get_vocab_base", return_value=(["a"], [gguf.TokenType.NORMAL], "gigachat4")):
            model.set_vocab()
        self.assertFalse(model.gguf_writer.kv_data[0][gguf.Keys.Tokenizer.CLEAN_SPACES].value)

    def test_registration_and_variant_validation(self):
        self.assertIs(ModelBase.from_model_architecture("GigaChat4ForCausalLM"), GigaChat4Model)
        self.assertIs(get_model_class("GigaChat4ForCausalLM"), GigaChat4Model)
        self.assertEqual(self.make_model().gguf_writer.arch, "gigachat4")
        with self.assertRaisesRegex(ValueError, "GigaChat4 requires norm_type"):
            self.make_model(overrides={"norm_type": "ZeroCenteredGatedNorm"}).set_gguf_parameters()

    def test_metadata_and_rope_config_formats(self):
        for sparse in (False, True):
            for new_rope in (False, True):
                with self.subTest(sparse=sparse, rope_parameters=new_rope):
                    overrides = self.sparse_config() if sparse else {}
                    if new_rope:
                        overrides.update({
                            "rope_scaling": None,
                            "rope_parameters": {"rope_type": "yarn", "factor": 32.0, "rope_theta": 100000.0, "original_max_position_embeddings": 32768, "mscale_all_dim": 1.0},
                            "swiglu_limit": 10.0,
                            "final_logit_softcapping": 30.0,
                        })
                    model = self.make_model(overrides=overrides)
                    model.set_gguf_parameters()
                    metadata = {key: entry.value for key, entry in model.gguf_writer.kv_data[0].items()}
                    self.assertEqual(metadata["general.architecture"], "gigachat4")
                    self.assertEqual(metadata["gigachat4.block_count"], 5)
                    self.assertEqual(metadata["gigachat4.nextn_predict_layers"], 1)
                    self.assertEqual(metadata["gigachat4.attention.indexer.head_count"], 64 if sparse else 0)
                    self.assertEqual(metadata["gigachat4.attention.indexer.key_length"], 128 if sparse else 0)
                    self.assertEqual(metadata["gigachat4.attention.indexer.top_k"], 2048 if sparse else 0)
                    self.assertEqual(metadata["gigachat4.attention.indexer.types"], [False, sparse, False, sparse])
                    self.assertEqual(metadata["gigachat4.attention.recurrent_layers"], [True, False, True, False, False])
                    self.assertEqual(metadata["gigachat4.final_logit_softcapping"], 30.0 if new_rope else 0.0)
                    self.assertEqual(metadata["gigachat4.swiglu_clamp_exp"], [10.0 if new_rope else 0.0] * 5)
                    self.assertEqual(metadata["gigachat4.swiglu_clamp_shexp"], metadata["gigachat4.swiglu_clamp_exp"])
                    self.assertEqual(metadata["gigachat4.rope.scaling.factor"], 32.0 if new_rope else 4.0)

    def test_disabled_clamp(self):
        model = self.make_model(overrides={"swiglu_limit": None})
        model.set_gguf_parameters()
        self.assertEqual(model.gguf_writer.kv_data[0]["gigachat4.swiglu_clamp_exp"].value, [0.0] * 5)

    def test_reference_config_defaults(self):
        model = self.make_model(overrides={**self.sparse_config(), "swiglu_limit": 10.0, "final_logit_softcapping": 30.0})
        for key in ("swiglu_limit", "final_logit_softcapping", "index_topk", "linear_attn_o_norm_eps"):
            del model.hparams[key]
        model.set_gguf_parameters()
        metadata = {key: entry.value for key, entry in model.gguf_writer.kv_data[0].items()}
        self.assertEqual(metadata["gigachat4.swiglu_clamp_exp"], [0.0] * 5)
        self.assertEqual(metadata["gigachat4.final_logit_softcapping"], 0.0)
        self.assertEqual(metadata["gigachat4.attention.indexer.top_k"], 0)
        self.assertEqual(metadata["gigachat4.attention.indexer.types"], [False, False, False, False])
        self.assertEqual(metadata["gigachat4.attention.recurrent_layers"], [True, False, True, False, False])
        self.assertEqual(metadata["gigachat4.expert_gating_func"], 2)
        for key in ("scoring_func", "score_function"):
            with self.subTest(key=key):
                model = self.make_model(overrides={key: "softmax"})
                model.set_gguf_parameters()
                self.assertEqual(model.gguf_writer.kv_data[0]["gigachat4.expert_gating_func"].value, 2)
        for key in ("full_attention_layers", "use_mla_scaling_factor", "norm_type", "layernorm_type", "gated_attention",
                    "linear_attention_type", "linear_gating_type", "linear_sigmoid_gate_scale"):
            with self.subTest(key=key):
                model = self.make_model()
                del model.hparams[key]
                with self.assertRaisesRegex(ValueError, key):
                    model.set_gguf_parameters()
        for overrides, error in (({"q_lora_rank": None}, "q_lora_rank"), ({"index_topk": 2048}, "index_n_heads")):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, error):
                    self.make_model(overrides=overrides).set_gguf_parameters()

    def test_mixed_dense_and_sparse_layers(self):
        model = self.make_model(overrides={**self.sparse_config(), "dense_attention_layers": [1]})
        model.set_gguf_parameters()
        self.assertEqual(model.gguf_writer.kv_data[0]["gigachat4.attention.indexer.types"].value, [False, False, False, True])

    def test_zero_top_k_is_dense(self):
        for top_k in (None, 0):
            with self.subTest(top_k=top_k):
                model = self.make_model(overrides={**self.sparse_config(), "index_topk": top_k})
                model.set_gguf_parameters()
                kv = model.gguf_writer.kv_data[0]
                for key in ("top_k", "head_count", "key_length"):
                    self.assertEqual(kv[f"gigachat4.attention.indexer.{key}"].value, 0)
                self.assertEqual(kv["gigachat4.attention.indexer.types"].value, [False, False, False, False])

    def test_norm_epsilon_metadata(self):
        model = self.make_model(overrides={**self.sparse_config(), "rms_norm_eps": 1e-5, "linear_attn_o_norm_eps": 1e-5})
        model.set_gguf_parameters()
        self.assertEqual(model.gguf_writer.kv_data[0]["gigachat4.attention.layer_norm_rms_epsilon"].value, 1e-5)
        with self.assertRaisesRegex(ValueError, "linear_attn_o_norm_eps"):
            self.make_model(overrides={"linear_attn_o_norm_eps": 1e-5}).set_gguf_parameters()
        model = self.make_model()
        del model.hparams["linear_attn_o_norm_eps"]
        model.set_gguf_parameters()
        model = self.make_model()
        del model.hparams["rms_norm_eps"]
        with self.assertRaises(ValueError):
            model.set_gguf_parameters()

    def test_unsupported_variants(self):
        for key, value in (
            ("indexer_norm_type", "LayerNorm"), ("indexer_norm_eps", 1e-5),
            ("indexer_rotate_activation_enabled", True), ("attention_distillation", True),
            ("linear_gating_type", "gated_rmsnorm_sigmoid_zero_centered"),
            ("linear_attention_type", "KimiDeltaAttention"), ("linear_attention_type", "Qwen3NextGatedDeltaNet"),
            ("linear_use_legacy_qkvz_layout", True),
            ("nextn_is_sparse", True), ("nextn_is_linear", True),
            ("branch_multiplier", 0.5), ("linear_attn_output_mult", 2.0),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.make_model(overrides={**self.sparse_config(), key: value}).set_gguf_parameters()

    def test_omitted_parts_are_not_checked(self):
        for key, value in (("indexer_norm_type", "LayerNorm"), ("indexer_rotate_activation_enabled", True)):
            with self.subTest(key=key, value=value):
                self.make_model(overrides={key: value}).set_gguf_parameters()
        with patch.object(GigaChat4Model, "no_mtp", True):
            for key in ("nextn_is_sparse", "nextn_is_linear"):
                with self.subTest(key=key):
                    self.make_model(overrides={key: True}).set_gguf_parameters()
            for layers in ([1, 3], [1, 3, 7], [-1, 1, 3]):
                with self.subTest(full_attention_layers=layers):
                    model = self.make_model(overrides={**self.sparse_config(), "full_attention_layers": layers,
                                                       "dense_attention_layers": [1, 3], "indexer_norm_type": "LayerNorm"})
                    model.set_gguf_parameters()
                    self.assertEqual(model.gguf_writer.kv_data[0]["gigachat4.attention.indexer.top_k"].value, 0)

    def test_indexer_of_the_mtp_block_only(self):
        overrides = {**self.sparse_config(), "dense_attention_layers": [1, 3]}
        for no_mtp, mtp_only, top_k in ((False, False, 2048), (False, True, 2048), (True, False, 0)):
            with self.subTest(no_mtp=no_mtp, mtp_only=mtp_only), patch.object(GigaChat4Model, "no_mtp", no_mtp), \
                    patch.object(GigaChat4Model, "mtp_only", mtp_only):
                model = self.make_model(overrides=overrides)
                model.set_gguf_parameters()
                metadata = model.gguf_writer.kv_data[0]
                self.assertEqual(metadata["gigachat4.attention.indexer.top_k"].value, top_k)
                self.assertEqual(metadata["gigachat4.attention.indexer.types"].value, [False] * 4)

    def test_mtp_export_modes(self):
        tensors = {
            "model.norm.weight": torch.ones(4),
            "model.norm.gate_up_projection.weight": torch.ones((2, 4)),
            "model.norm.gate_down_projection.weight": torch.ones((4, 2)),
            "model.layers.1.self_attn.q_a_proj.weight": torch.ones((4, 4)),
            "model.layers.4.enorm.weight": torch.ones(4),
            "model.layers.4.self_attn.q_a_proj.weight": torch.ones((4, 4)).to(torch.float8_e4m3fn),
            "model.layers.4.self_attn.q_a_proj.weight_scale_inv": torch.ones((2, 2)),
            "model.layers.4.self_attn.indexer.wk.weight": torch.ones((4, 4)),
        }
        mtp = {name for name in tensors if name.startswith("model.layers.4.")}
        model = self.make_model(tensors=tensors)
        self.assertEqual(set(model.model_tensors), set(tensors))
        self.assertEqual(model.block_count, 5)
        model.set_gguf_parameters()
        metadata = model.gguf_writer.kv_data[0]
        self.assertEqual(metadata["gigachat4.nextn_predict_layers"].value, 1)
        self.assertEqual(metadata["gigachat4.attention.recurrent_layers"].value, [True, False, True, False, False])
        with patch.object(GigaChat4Model, "no_mtp", True):
            model = self.make_model(tensors=tensors)
            self.assertEqual(set(model.model_tensors), set(tensors) - mtp)
            self.assertEqual(model.block_count, 4)
            model.set_gguf_parameters()
            self.assertNotIn("gigachat4.nextn_predict_layers", model.gguf_writer.kv_data[0])
        with patch.object(GigaChat4Model, "mtp_only", True):
            model = self.make_model(tensors=tensors)
            shared = {"model.norm.weight", "model.norm.gate_up_projection.weight", "model.norm.gate_down_projection.weight"}
            self.assertEqual(set(model.model_tensors), mtp | shared)
            self.assertEqual(model.block_count, 5)

    def test_indexer_weights_are_unchanged(self):
        weights = {
            "model.layers.1.self_attn.indexer.wq_b.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
            "model.layers.1.self_attn.indexer.wk.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "model.layers.1.self_attn.indexer.k_norm.weight": torch.tensor([0., 0.5, 1., 2.]),
            "model.layers.1.self_attn.indexer.weights_proj.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        }
        model = self.make_model(tensors=weights, overrides={**self.sparse_config(), "index_n_heads": 2, "index_head_dim": 4})
        result = self.converted(model)
        for name, weight in weights.items():
            np.testing.assert_array_equal(result[model.map_tensor_name(name)], weight.numpy())

    def test_block_fp8_dequantizes_before_projection_transforms(self):
        for eager in (False, True):
            with self.subTest(eager=eager):
                names_shapes = {
                    "model.layers.0.self_attn.in_proj_qkvz.weight": (24, 4),
                    "model.layers.0.self_attn.in_proj_ba.weight": (8, 4),
                    "model.layers.1.self_attn.kv_b_proj.weight": (8, 2),
                    "model.layers.1.self_attn.indexer.wq_b.weight": (8, 4),
                    "model.layers.1.self_attn.q_a_proj.weight": (5, 4),
                }
                tensors = {}
                dequantized = {}
                for name, shape in names_shapes.items():
                    weight = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(torch.float8_e4m3fn)
                    scale_shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
                    scales = torch.arange(1, 1 + int(np.prod(scale_shape)), dtype=torch.float32).reshape(scale_shape) / 8
                    tensors[name] = weight
                    tensors[name + "_scale_inv"] = scales
                    dequantized[name] = weight.float() * scales.repeat_interleave(2, 0).repeat_interleave(2, 1)[:shape[0], :shape[1]]
                model = self.make_model(tensors=tensors, eager=eager, overrides={
                    **self.sparse_config(), "index_n_heads": 2, "index_head_dim": 4,
                    "quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 2]},
                })
                model._preserve_fp8 = True
                result = self.converted(model)
                qkvz = dequantized["model.layers.0.self_attn.in_proj_qkvz.weight"].reshape(2, 12, 4)
                np.testing.assert_array_equal(result["blk.0.attn_qkv.weight"], torch.cat([qkvz[:, :2].reshape(4, 4), qkvz[:, 2:4].reshape(4, 4), qkvz[:, 4:8].reshape(8, 4)]).numpy())
                np.testing.assert_array_equal(result["blk.0.attn_gate.weight"], qkvz[:, 8:].reshape(8, 4).numpy())
                ba = dequantized["model.layers.0.self_attn.in_proj_ba.weight"].reshape(2, 4, 4)
                np.testing.assert_array_equal(result["blk.0.ssm_beta.weight"], ba[:, :2].reshape(4, 4).numpy())
                np.testing.assert_array_equal(result["blk.0.ssm_alpha.weight"], ba[:, 2:].reshape(4, 4).numpy())
                kv = dequantized["model.layers.1.self_attn.kv_b_proj.weight"].reshape(2, 4, 2)
                np.testing.assert_array_equal(result["blk.1.attn_k_b.weight"], kv[:, :2].transpose(1, 2).numpy())
                np.testing.assert_array_equal(result["blk.1.attn_v_b.weight"], kv[:, 2:].numpy())
                np.testing.assert_array_equal(result["blk.1.indexer.attn_q_b.weight"], dequantized["model.layers.1.self_attn.indexer.wq_b.weight"].numpy())
                np.testing.assert_array_equal(result["blk.1.attn_q_a.weight"], dequantized["model.layers.1.self_attn.q_a_proj.weight"].numpy())
                self.assertFalse(any("scale" in name for name in result))

    def test_fp8_split_projection_scales(self):
        name = "model.layers.0.self_attn.in_proj_qkvz.weight"
        weight = torch.arange(96, dtype=torch.float32).reshape(24, 4).to(torch.float8_e4m3fn)
        model = self.make_model(tensors={name: weight, name + "_scale_inv": torch.tensor([0.25])},
                                ftype=gguf.LlamaFileType.GUESSED, overrides={"quantization_config": {"quant_method": "fp8"}})
        result = self.converted(model)
        qkvz = weight.float().reshape(2, 12, 4)
        for name, expected in (
            ("blk.0.attn_qkv", torch.cat([qkvz[:, :2].reshape(4, 4), qkvz[:, 2:4].reshape(4, 4), qkvz[:, 4:8].reshape(8, 4)])),
            ("blk.0.attn_gate", qkvz[:, 8:].reshape(8, 4)),
        ):
            self.assertEqual(model.gguf_writer.tensors[0][name + ".weight"].dtype, gguf.GGMLQuantizationType.F8_E4M3)
            np.testing.assert_array_equal(gguf.dequantize(result[name + ".weight"], gguf.GGMLQuantizationType.F8_E4M3), expected.numpy())
            np.testing.assert_array_equal(result[name + ".scale"], [0.25])

    def test_checkpoint_template_is_preserved(self):
        from tokenizers import Tokenizer, models

        model = self.make_model()
        template = "{% for message in messages %}{{ message['content'] }}{{ eos_token }}{% endfor %}"
        (model.dir_model / "chat_template.jinja").write_text(template)
        tokens = ["<unk>", "<s>", "</s>", "a", "b", "ab", "x", "y"]
        tokenizer = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=[("a", "b")], unk_token="<unk>"))
        tokenizer.add_special_tokens(tokens[:3])
        tokenizer.save(str(model.dir_model / "tokenizer.json"))
        for tokenizer_class in ("PreTrainedTokenizerFast", "TokenizersBackend"):
            with self.subTest(tokenizer_class=tokenizer_class):
                (model.dir_model / "tokenizer_config.json").write_text(json.dumps({
                    "tokenizer_class": tokenizer_class, "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
                }))
                with patch.object(DeepseekV2Model, "get_vocab_base_pre", return_value="gigachat"):
                    model.set_vocab()
                metadata = model.gguf_writer.kv_data[0]
                self.assertEqual(metadata["tokenizer.chat_template"].value, template)
                self.assertEqual(metadata["tokenizer.ggml.pre"].value, "gigachat4")
                self.assertEqual(metadata["tokenizer.ggml.tokens"].value, tokens)
                self.assertEqual(metadata["tokenizer.ggml.token_type"].value, [3, 3, 3, 1, 1, 1, 1, 1])
                self.assertEqual(metadata["tokenizer.ggml.bos_token_id"].value, 1)
                self.assertEqual(metadata["tokenizer.ggml.eos_token_id"].value, 2)


if __name__ == "__main__":
    unittest.main()
