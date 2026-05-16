from __future__ import annotations

from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MistralTokenizerType, MistralVocab, _mistral_common_installed, _mistral_import_error_msg, gguf, logger

from .deepseek import DeepseekV2Model
from .llama import LlamaModel

if _mistral_common_installed:
    from mistral_common.tokens.tokenizers.base import TokenizerVersion  # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer  # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer  # type: ignore[import-not-found, ty:unresolved-import]
else:
    TokenizerVersion = None  # type: ignore[assignment]
    Tekkenizer = None  # type: ignore[assignment]
    SentencePieceTokenizer = None  # type: ignore[assignment]


class MistralModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.MISTRAL3
    model_name = "Mistral"
    hf_arch = ""
    is_mistral_format = True
    undo_permute = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for compatibility, we use LLAMA arch for older models
        # TODO: remove this once everyone migrates to newer version of llama.cpp
        if "llama_4_scaling" not in self.hparams:
            self.model_arch = gguf.MODEL_ARCH.LLAMA
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_architecture()
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def dequant_model(self):
        # transform quantization config into HF format
        quant_config = self.hparams.get("quantization")
        if quant_config is not None:
            assert quant_config["qformat_weight"] == "fp8_e4m3"
            self.hparams["quantization_config"] = {
                "activation_scheme": "static",
                "quant_method": "fp8",
                "weight_block_size": None,
            }
        return super().dequant_model()

    @staticmethod
    def get_community_chat_template(vocab: MistralVocab, templates_dir: Path, is_mistral_format: bool):
        assert TokenizerVersion is not None and Tekkenizer is not None and SentencePieceTokenizer is not None, _mistral_import_error_msg
        assert isinstance(vocab.tokenizer, (Tekkenizer, SentencePieceTokenizer)), (
            f"Expected Tekkenizer or SentencePieceTokenizer, got {type(vocab.tokenizer)}"
        )

        if vocab.tokenizer.version == TokenizerVersion.v1:
            return "mistral-v1"
        elif vocab.tokenizer.version == TokenizerVersion.v3 and vocab.tokenizer_type == MistralTokenizerType.spm:
            return "mistral-v3"
        elif vocab.tokenizer.version == TokenizerVersion.v3 and vocab.tokenizer_type == MistralTokenizerType.tekken:
            return "mistral-v3-tekken"
        elif vocab.tokenizer.version == TokenizerVersion.v7 and vocab.tokenizer_type == MistralTokenizerType.spm:
            return "mistral-v7"
        elif vocab.tokenizer.version == TokenizerVersion.v7 and vocab.tokenizer_type == MistralTokenizerType.tekken:
            return "mistral-v7-tekken"
        elif vocab.tokenizer.version == TokenizerVersion.v11:
            template_file = "Mistral-Small-3.2-24B-Instruct-2506.jinja"
        elif vocab.tokenizer.version == TokenizerVersion.v13:
            template_file = "unsloth-mistral-Devstral-Small-2507.jinja"
        else:
            err_message = f"Unknown tokenizer type: {vocab.tokenizer_type} and version {vocab.tokenizer.version}"
            if is_mistral_format:
                err_message += (
                    " . Please pass --disable-mistral-community-chat-template argument to the CLI "
                    "if you want to skip this error and use the Mistral official `mistral-common` pre-processing library."
                )
            raise ValueError(err_message)

        template_path = templates_dir / template_file
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        MistralModel.set_mistral_config(self.gguf_writer, self.hparams)

    @staticmethod
    def set_mistral_config(gguf_writer: gguf.GGUFWriter, hparams: dict):
        if "yarn" in hparams:
            yarn_params = hparams["yarn"]
            mscale_all_dim = 1.0 if not yarn_params["apply_scale"] else 0.0
            gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            gguf_writer.add_rope_scaling_factor(yarn_params["factor"])
            gguf_writer.add_rope_scaling_yarn_beta_fast(yarn_params["beta"])
            gguf_writer.add_rope_scaling_yarn_beta_slow(yarn_params["alpha"])
            gguf_writer.add_rope_scaling_yarn_log_mul(mscale_all_dim)
            gguf_writer.add_rope_scaling_orig_ctx_len(yarn_params["original_max_position_embeddings"])

        if "llama_4_scaling" in hparams:
            gguf_writer.add_attn_temperature_scale(hparams["llama_4_scaling"]["beta"])


class MistralMoeModel(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2
    model_name = "Mistral"
    hf_arch = ""
    is_mistral_format = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Using MistralMoeModel")
        # remap hparams from Mistral MoE format to DeepseekV2 format
        # we do this way to be able to reuse DeepseekV2Model set_gguf_parameters logic
        # ref: https://github.com/vllm-project/vllm/blob/b294e28db2c5dee61bc25157664edcada8b90b31/vllm/transformers_utils/configs/mistral.py
        config = self.hparams
        # Mistral key -> HF key
        config_mapping = {
            "dim": "hidden_size",
            "norm_eps": "rms_norm_eps",
            "n_kv_heads": "num_key_value_heads",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "hidden_dim": "intermediate_size",
        }
        # HF key -> (Mistral key, default value)
        top_level_mapping_with_default = {
            "model_type": ("model_type", "transformer"),
            "hidden_act": ("activation", "silu"),
            "tie_word_embeddings": ("tied_embeddings", False),
            "max_seq_len": ("max_seq_len", config.get("max_position_embeddings", 128_000)),
            "max_position_embeddings": ("max_position_embeddings", 128_000),
        }
        # mapping top-level keys
        for key, new_key in config_mapping.items():
            if key in config:
                config[new_key] = config[key]
        for new_key, (key, default_value) in top_level_mapping_with_default.items():
            config[new_key] = config.get(key, default_value)
        # mapping MoE-specific keys
        moe_config_map = {
            "route_every_n": "moe_layer_freq",
            "first_k_dense_replace": "first_k_dense_replace",
            "num_experts_per_tok": "num_experts_per_tok",
            "num_experts": "n_routed_experts",
            "expert_hidden_dim": "moe_intermediate_size",
            "routed_scale": "routed_scaling_factor",
            "num_shared_experts": "n_shared_experts",
            "num_expert_groups": "n_group",
            "num_expert_groups_per_tok": "topk_group",
        }
        moe = config["moe"]
        for key, new_key in moe_config_map.items():
            if key in moe:
                config[new_key] = moe[key]
        # provide missing values
        config["topk_method"] = None
        config["norm_topk_prob"] = True
        config["scoring_func"] = "softmax"

    def set_vocab(self):
        self._set_vocab_mistral()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        MistralModel.set_mistral_config(self.gguf_writer, self.hparams)
        yarn_params = self.hparams["yarn"]
        self.gguf_writer.add_attn_temperature_length(yarn_params["original_max_position_embeddings"])

        # [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        # note: for legacy reasons, this is not consistent with the other usages of self.gguf_writer.add_rope_scaling_yarn_log_mul
        # ref https://github.com/ggml-org/llama.cpp/pull/17945
        self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1) # mscale_all_dim * 0.1

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # rename certain tensors so that we can reuse DeepseekV2Model modify_tensors logic
        if name.endswith(".qscale_act"):
            name = name.replace(".qscale_act", ".input_scale")
        if name.endswith(".qscale_weight"):
            name = name.replace(".qscale_weight", ".weight_scale")
        if ".wkv_b." in name:
            name = name.replace(".wkv_b.", ".kv_b_proj.")
        if ".experts." in name:
            name = name.replace(".experts.", ".mlp.experts.")
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            name = "model." + name

        return super().filter_tensors((name, gen))
