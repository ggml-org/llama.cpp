from __future__ import annotations

import concurrent.futures
import ctypes
import math
import os
import re

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, MmprojModel, ModelBase, TextModel, gguf, logger

from .qwen import QwenModel

TORCH_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)


@ModelBase.register("DeepseekOCRForCausalLM")
class DeepseekOCRVisionModel(MmprojModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.DEEPSEEKOCR)
        # default values below are taken from HF tranformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_vision_use_gelu(True)
        # calculate proj_scale_factor (used by tinygemma3 test model)
        image_seq_length = self.preprocessor_config.get("image_seq_length", 256)
        n_per_side = int(image_seq_length ** 0.5)
        image_size = self.hparams["image_size"]
        patch_size = self.hparams["patch_size"]
        proj_scale_factor = (image_size // patch_size) // n_per_side
        if proj_scale_factor > 0 and proj_scale_factor != 4:
            # we only need to write this if it's not the default value
            # in this case, we are converting a test model
            self.gguf_writer.add_vision_projector_scale_factor(proj_scale_factor)
        # @bluebread: there's no window_size in config but just add it here anyway
        self.gguf_writer.add_vision_window_size(self.hparams.get("window_size", 14))

        # SAM configuration
        sam_hparams = hparams['sam']
        self.gguf_writer.add_vision_sam_layers_count(sam_hparams['layers'])
        self.gguf_writer.add_vision_sam_embedding_length(sam_hparams['width'])
        self.gguf_writer.add_vision_sam_head_count(sam_hparams['heads'])

    def get_vision_config(self) -> dict[str, Any]:
        vision_config: dict[str, Any] | None = self.global_config.get("vision_config")

        if not vision_config:
            raise ValueError("DeepseekOCR model requires 'vision_config' in the model configuration, but it was not found")

        vision_config['sam'] = vision_config['width']['sam_vit_b']
        vision_config.update(vision_config['width']['clip-l-14-224'])
        vision_config['hidden_size'] = vision_config['width']
        vision_config['num_heads'] = vision_config['heads']
        vision_config['intermediate_size'] = vision_config['heads'] * 4

        return vision_config

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".embeddings." in name or 'pos_embed' in name:
            return gguf.GGMLQuantizationType.F32
        if ".rel_pos_h" in name or '.rel_pos_w' in name:
            return gguf.GGMLQuantizationType.F32
        if ".neck." in name or ".net_" in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Only process vision-related tensors, skip language model tensors
        # Vision components: sam_model, vision_model, projector, image_newline, view_seperator
        # Language model components to skip: lm_head, embed_tokens, layers, norm
        if name.startswith(("lm_head.", "model.embed_tokens.", "model.layers.", "model.norm.")):
            return None

        if name.endswith("pos_embed") or name.endswith("rel_pos_h") or name.endswith("rel_pos_w"):
            name += ".weight"

        return super().filter_tensors((name, gen))


@ModelBase.register("DeepseekForCausalLM")
class DeepseekModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register(
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "KimiVLForConditionalGeneration",
    "KimiK25ForConditionalGeneration",
    "YoutuForCausalLM",
    "YoutuVLForConditionalGeneration",
)
class DeepseekV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    # TODO @ngxson : remove this when we support MTP for deepseek models
    skip_mtp = True

    merge_expert = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams: dict = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

        # special handling for Deepseek OCR
        if self.origin_hf_arch == "DeepseekOCRForCausalLM":
            self.model_arch = gguf.MODEL_ARCH.DEEPSEEK2OCR
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_architecture()
            # default jinja template
            self.gguf_writer.add_chat_template("{% for m in messages %}{{m['content']}}{% endfor %}")

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
            return
        except Exception:
            pass

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokpre = self.get_vocab_base_pre(tokenizer)

        if tokpre == "kimi-k2":
            # Build merges list using the approach similar to HunYuanMoE
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.model._mergeable_ranks  # ty: ignore[unresolved-attribute]
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

            # Build token list
            vocab_size = self.hparams["vocab_size"]
            special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
            reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []

            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
        else:
            raise NotImplementedError(f"Deepseek pre-tokenizer {tokpre!r} is not supported yet!")

    def set_gguf_parameters(self):
        is_ocr = (self.model_arch == gguf.MODEL_ARCH.DEEPSEEK2OCR)

        if is_ocr:
            self.hparams['rope_theta'] = self.hparams.get('rope_theta', 10000.0)
        else:
            # note: deepseek2 using MLA converts into MQA (ie: GQA with 1 group)
            self.hparams["num_key_value_heads"] = 1

        self.hparams['rms_norm_eps'] = self.hparams.get('rms_norm_eps', 1e-6)

        super().set_gguf_parameters()
        hparams = self.hparams

        # first_k_dense_replace: number of leading layers using dense FFN instead of MoE
        # For non-MoE models (like Youtu), set to n_layer to use dense FFN for all layers
        # For MoE models (like DeepSeek-V2), this is the number of leading non-MoE layers
        has_moe = hparams.get("n_routed_experts") is not None
        first_k_dense_replace = hparams.get("first_k_dense_replace")
        if first_k_dense_replace is None:
            # Default: if no MoE, all layers are dense; if MoE, none are dense
            first_k_dense_replace = hparams["num_hidden_layers"] if not has_moe else 0
        self.gguf_writer.add_leading_dense_block_count(first_k_dense_replace)
        kv_lora_rank = hparams.get("kv_lora_rank", 512)
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])

        # note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
        if not is_ocr:
            self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
            self.gguf_writer.add_key_length(kv_lora_rank + hparams["qk_rope_head_dim"])
            self.gguf_writer.add_value_length(kv_lora_rank)
            self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
            self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        # MoE parameters (required by C++ code for DEEPSEEK2 arch)
        # For non-MoE models like Youtu, use intermediate_size as expert_feed_forward_length
        moe_intermediate_size = self.find_hparam(["moe_intermediate_size", "intermediate_size"], optional=False)
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)

        if (n_routed_experts := hparams.get("n_routed_experts")) is not None:
            self.gguf_writer.add_expert_count(n_routed_experts)

        # expert_shared_count is required by C++ code, default to 0 for non-MoE models
        n_shared_experts = hparams.get("n_shared_experts", 0)
        self.gguf_writer.add_expert_shared_count(n_shared_experts)

        # When not set, C++ code will use scale_w = false to skip the no-op scaling
        if (routed_scaling_factor := hparams.get("routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routed_scaling_factor)

        if (norm_topk_prob := hparams.get("norm_topk_prob")) is not None and norm_topk_prob:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if (rope_mscale_all := self.rope_parameters.get("mscale_all_dim")) is not None:
            # [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            # note: for legacy reasons, this is not consistent with the other usages of self.gguf_writer.add_rope_scaling_yarn_log_mul
            # ref https://github.com/ggml-org/llama.cpp/pull/17945
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * rope_mscale_all)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # skip lm_head.weight if tie_word_embeddings is True
        if self.hparams.get("tie_word_embeddings", False):
            if name == "lm_head.weight" or name == "model.lm_head.weight":
                logger.info("Skipping tied output layer 'lm_head.weight' (will use token_embd.weight)")
                return

        # skip Multi-Token Prediction (MTP) layers
        if self.skip_mtp:
            block_count = self.hparams["num_hidden_layers"]
            match = re.match(r"model.layers.(\d+)", name)
            if match and int(match.group(1)) >= block_count:
                return

        # process the experts separately
        if self.merge_expert and name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        # note: MLA with the absorption optimization, needs these two split and k_b_proj transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("DeepseekV4ForCausalLM")
class DeepseekV4Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK4

    # Optional DeepSeek V4 debug / expert-quant knobs. In the pre-#17114
    # monolithic convert_hf_to_gguf.py these were ModelBase.__init__ params
    # wired to --deepseek4-* CLI flags. The refactored conversion/base.py
    # ModelBase.__init__ does not accept them, so they default here; the
    # standard DeepseekV4ForCausalLM conversion path does not require them.
    deepseek4_max_layers: int | None = None
    deepseek4_expert_outtypes: str | None = None
    deepseek4_expert_workers: int = 1

    _experts: list[dict[str, Tensor]] | None = None

    _fp4_table = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.float32)

    _qtype_aliases: dict[str, gguf.GGMLQuantizationType] = {
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q2_k": gguf.GGMLQuantizationType.Q2_K,
        "iq2_xxs": gguf.GGMLQuantizationType.IQ2_XXS,
        "iq2_xs": gguf.GGMLQuantizationType.IQ2_XS,
        "tq1_0": gguf.GGMLQuantizationType.TQ1_0,
        "tq2_0": gguf.GGMLQuantizationType.TQ2_0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._deepseek4_original_block_count = self.block_count
        if self.deepseek4_max_layers is not None:
            if self.deepseek4_max_layers <= 0:
                raise ValueError("--deepseek4-max-layers must be positive")
            if self.deepseek4_max_layers > self.block_count:
                raise ValueError(
                    f"--deepseek4-max-layers={self.deepseek4_max_layers} exceeds model layer count {self.block_count}"
                )
            self.block_count = self.deepseek4_max_layers
            self.hparams["num_hidden_layers"] = self.block_count
            self.hparams["n_layers"] = self.block_count
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
            logger.warning(
                "DeepSeek V4 debug export: writing only the first %d/%d transformer layers",
                self.block_count,
                self._deepseek4_original_block_count,
            )

        self._deepseek4_expert_qtypes = self._parse_expert_outtype_spec(self.deepseek4_expert_outtypes)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        self.hparams["num_key_value_heads"] = self.hparams.get("num_key_value_heads", 1)

        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_attention_output_lora_rank(hparams["o_lora_rank"])
        self.gguf_writer.add_attention_output_group_count(hparams["o_groups"])
        self.gguf_writer.add_attention_compress_ratios(hparams["compress_ratios"])
        self.gguf_writer.add_attention_compress_rope_freq_base(hparams["compress_rope_theta"])

        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams.get("routed_scaling_factor", 1.0))
        self.gguf_writer.add_hash_layer_count(min(hparams["num_hash_layers"], self.block_count))
        if (norm_topk_prob := hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)
        if (swiglu_limit := hparams.get("swiglu_limit")) is not None and float(swiglu_limit) > 0.0:
            self.gguf_writer.add_swiglu_clamp_exp([float(swiglu_limit)] * self.block_count)

        if (sliding_window := hparams.get("sliding_window")) is not None:
            self.gguf_writer.add_sliding_window(sliding_window)

        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])

        if self.deepseek4_max_layers is None and (num_nextn_predict_layers := hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iters(hparams["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_eps(hparams["hc_eps"])

    @staticmethod
    def _strip_model_prefix(name: str) -> str:
        return name.removeprefix("model.")

    def _skip_layer_tensor(self, stripped_name: str) -> bool:
        if self.deepseek4_max_layers is None:
            return False
        match = re.match(r"layers\.(\d+)\.", stripped_name)
        return match is not None and int(match.group(1)) >= self.block_count

    @staticmethod
    def _is_low_bit_ftype(ftype: gguf.LlamaFileType) -> bool:
        return ftype in (
            gguf.LlamaFileType.MOSTLY_TQ1_0,
            gguf.LlamaFileType.MOSTLY_TQ2_0,
            gguf.LlamaFileType.MOSTLY_Q2_K,
            gguf.LlamaFileType.MOSTLY_IQ2_XXS,
            gguf.LlamaFileType.MOSTLY_IQ2_XS,
        )

    @staticmethod
    def _qtype_for_ftype(ftype: gguf.LlamaFileType) -> gguf.GGMLQuantizationType | None:
        return {
            gguf.LlamaFileType.MOSTLY_TQ1_0: gguf.GGMLQuantizationType.TQ1_0,
            gguf.LlamaFileType.MOSTLY_TQ2_0: gguf.GGMLQuantizationType.TQ2_0,
            gguf.LlamaFileType.MOSTLY_Q2_K: gguf.GGMLQuantizationType.Q2_K,
            gguf.LlamaFileType.MOSTLY_IQ2_XXS: gguf.GGMLQuantizationType.IQ2_XXS,
            gguf.LlamaFileType.MOSTLY_IQ2_XS: gguf.GGMLQuantizationType.IQ2_XS,
            gguf.LlamaFileType.MOSTLY_Q8_0: gguf.GGMLQuantizationType.Q8_0,
        }.get(ftype)

    @classmethod
    def _parse_qtype_name(cls, name: str) -> gguf.GGMLQuantizationType:
        qtype = cls._qtype_aliases.get(name.strip().lower())
        if qtype is None:
            allowed = ", ".join(sorted(cls._qtype_aliases))
            raise ValueError(f"unknown DeepSeek V4 expert outtype {name!r}; expected one of: {allowed}")
        return qtype

    @classmethod
    def _parse_expert_outtype_spec(cls, spec: str | None) -> dict[str, gguf.GGMLQuantizationType]:
        if spec is None:
            return {}

        result: dict[str, gguf.GGMLQuantizationType] = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                qtype = cls._parse_qtype_name(item)
                result.update({"w1": qtype, "w2": qtype, "w3": qtype})
                continue
            key, value = (part.strip().lower() for part in item.split("=", 1))
            if key not in ("w1", "w2", "w3", "gate", "down", "up"):
                raise ValueError(f"unknown DeepSeek V4 expert tensor selector {key!r}")
            wid = {"gate": "w1", "down": "w2", "up": "w3"}.get(key, key)
            result[wid] = cls._parse_qtype_name(value)
        return result

    @staticmethod
    def _scale_to_float(scale: Tensor) -> Tensor:
        if TORCH_FLOAT8_E8M0FNU is not None and scale.dtype == TORCH_FLOAT8_E8M0FNU:
            return scale.float()

        if scale.dtype in (torch.uint8, torch.int8):
            e = scale.view(torch.uint8).to(torch.int32)
            bits = torch.where(
                e == 0,
                torch.full_like(e, 0x00400000),
                e << 23,
            )
            return bits.view(torch.float32)

        return scale.float()

    @staticmethod
    def _scale_to_e8m0_bytes(scale: Tensor) -> Tensor:
        if TORCH_FLOAT8_E8M0FNU is not None and scale.dtype == TORCH_FLOAT8_E8M0FNU:
            return scale.view(torch.uint8)
        if scale.dtype in (torch.uint8, torch.int8):
            return scale.view(torch.uint8)

        scale = scale.float()
        e = torch.where(
            scale > 0,
            torch.floor(torch.log2(scale)).to(torch.int32) + 127,
            torch.zeros_like(scale, dtype=torch.int32),
        )
        return torch.clamp(e, 0, 255).to(torch.uint8)

    @classmethod
    def _dequant_fp8_weight(cls, weight: Tensor, scale: Tensor, block_size: Sequence[int]) -> Tensor:
        if len(block_size) != 2:
            raise ValueError(f"DeepSeek V4 expects 2D FP8 block scales, got block size {block_size}")

        block_out, block_in = block_size
        out_dim, in_dim = weight.shape
        if out_dim % block_out != 0 or in_dim % block_in != 0:
            raise ValueError(f"FP8 tensor shape {tuple(weight.shape)} is not divisible by block size {block_size}")

        scale = cls._scale_to_float(scale)
        expected_scale = (out_dim // block_out, in_dim // block_in)
        if tuple(scale.shape) != expected_scale:
            raise ValueError(f"FP8 scale shape {tuple(scale.shape)} does not match expected {expected_scale}")

        weight = weight.reshape(out_dim // block_out, block_out, in_dim // block_in, block_in)
        weight = weight.float() * scale[:, None, :, None]
        return weight.reshape(out_dim, in_dim)

    @classmethod
    def _dequant_fp4_weight(cls, weight: Tensor, scale: Tensor) -> Tensor:
        weight = weight.view(torch.uint8)
        out_dim, packed_in_dim = weight.shape
        in_dim = packed_in_dim * 2
        if in_dim % 32 != 0:
            raise ValueError(f"FP4 packed tensor shape {tuple(weight.shape)} does not contain 32-value blocks")

        n_blocks = in_dim // 32
        scale = cls._scale_to_float(scale)
        if tuple(scale.shape) != (out_dim, n_blocks):
            raise ValueError(f"FP4 scale shape {tuple(scale.shape)} does not match expected {(out_dim, n_blocks)}")

        fp4_table = cls._fp4_table.to(weight.device)
        packed = weight.reshape(out_dim, n_blocks, 16)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        vals = torch.stack((low, high), dim=-1).reshape(out_dim, n_blocks, 32)
        vals = fp4_table[vals.long()] * scale.unsqueeze(-1)
        return vals.reshape(out_dim, in_dim)

    @classmethod
    def _pack_fp4_as_mxfp4(cls, weight: Tensor, scale: Tensor) -> tuple[np.ndarray, list[int]]:
        weight = weight.view(torch.uint8)
        out_dim, packed_in_dim = weight.shape
        in_dim = packed_in_dim * 2
        if in_dim % 32 != 0:
            raise ValueError(f"FP4 packed tensor shape {tuple(weight.shape)} does not contain 32-value blocks")

        n_blocks = in_dim // 32
        scale_e = cls._scale_to_e8m0_bytes(scale)
        if tuple(scale_e.shape) != (out_dim, n_blocks):
            raise ValueError(f"FP4 scale shape {tuple(scale_e.shape)} does not match expected {(out_dim, n_blocks)}")

        packed = weight.reshape(out_dim, n_blocks, 16)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        vals = torch.stack((low, high), dim=-1).reshape(out_dim, n_blocks, 32)
        qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
        raw = torch.cat((scale_e.unsqueeze(-1), qs), dim=-1).reshape(out_dim, n_blocks * 17)
        return raw.numpy(), [out_dim, in_dim]

    _ggml_quant_lib: Any = None

    @classmethod
    def _load_ggml_quant_lib(cls):
        if cls._ggml_quant_lib is not None:
            return cls._ggml_quant_lib

        # This module lives in the conversion/ package; the repo root (where
        # build/bin/libggml.* lands) is its parent's parent. In the pre-#17114
        # monolithic convert_hf_to_gguf.py, __file__ was the repo-root script,
        # so .parent alone was the repo root -- search both so the lookup is
        # correct regardless of package layout.
        repo_root = Path(__file__).resolve().parent.parent
        pkg_root  = Path(__file__).resolve().parent
        candidates = [
            os.environ.get("LLAMA_CPP_LIBGGML"),
            repo_root / "build" / "bin" / "libggml.dylib",
            repo_root / "build" / "bin" / "libggml.so",
            repo_root / "build" / "bin" / "ggml.dll",
            pkg_root  / "build" / "bin" / "libggml.dylib",
            pkg_root  / "build" / "bin" / "libggml.so",
            pkg_root  / "build" / "bin" / "ggml.dll",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if not path.is_file():
                continue
            lib = ctypes.CDLL(str(path))
            lib.ggml_quantize_chunk.restype = ctypes.c_size_t
            lib.ggml_quantize_chunk.argtypes = (
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
            )
            lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
            lib.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)
            cls._ggml_quant_lib = lib
            return lib

        raise RuntimeError(
            "DeepSeek V4 low-bit expert conversion needs llama.cpp's libggml. "
            "Build llama.cpp first or set LLAMA_CPP_LIBGGML to libggml."
        )

    @classmethod
    def _quantize_deepseek4_expert(cls, data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
        c_quantized_types = {
            gguf.GGMLQuantizationType.Q2_K,
            gguf.GGMLQuantizationType.IQ2_XXS,
            gguf.GGMLQuantizationType.IQ2_XS,
        }
        if qtype not in c_quantized_types:
            return gguf.quants.quantize(data, qtype)

        data = np.ascontiguousarray(data, dtype=np.float32)
        out = np.zeros(gguf.quant_shape_to_byte_shape(data.shape, qtype), dtype=np.uint8, order="C")
        lib = cls._load_ggml_quant_lib()
        nrows = math.prod(data.shape[:-1])
        n_per_row = data.shape[-1]
        imatrix = ctypes.cast(0, ctypes.POINTER(ctypes.c_float))
        if lib.ggml_quantize_requires_imatrix(qtype.value):
            qw = np.ascontiguousarray(np.sum(data.reshape(-1, n_per_row) ** 2, axis=0), dtype=np.float32)
            imatrix = qw.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_size = lib.ggml_quantize_chunk(
            qtype.value,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.c_void_p),
            0,
            nrows,
            n_per_row,
            imatrix,
        )
        if result_size != out.size:
            raise RuntimeError(f"ggml_quantize_chunk wrote {result_size} bytes, expected {out.size}")
        return out

    def _write_deepseek4_tid2eid_tensors(self) -> set[str]:
        consumed: set[str] = set()
        for name in list(self.model_tensors.keys()):
            stripped = self._strip_model_prefix(name)
            if self._skip_layer_tensor(stripped):
                consumed.add(name)
                continue
            if re.match(r"layers\.\d+\.ffn\.gate\.tid2eid$", stripped) is None:
                continue

            data = LazyTorchTensor.to_eager(self.model_tensors[name]()).to(torch.int32).numpy()
            new_name = self.map_tensor_name(stripped)
            logger.info(f"{new_name}, int32 --> I32, shape = {{{', '.join(str(n) for n in reversed(data.shape))}}}")
            self.gguf_writer.add_tensor(new_name, data)
            consumed.add(name)
        return consumed

    def _write_deepseek4_expert_tensors(self) -> set[str]:
        default_qtype = self._qtype_for_ftype(self.ftype)
        if default_qtype is None and not self._deepseek4_expert_qtypes:
            if any(re.match(r"(?:model\.)?layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$", name) for name in self.model_tensors):
                raise NotImplementedError(
                    "DeepSeek V4 routed FP4 experts must be converted directly to a compact GGUF type. "
                    "Use --outtype iq2_xxs, iq2_xs, q2_k, tq2_0, tq1_0, or q8_0."
                )
            return set()

        n_experts = self.hparams["n_routed_experts"]
        consumed: set[str] = set()
        groups: dict[tuple[int, str], dict[int, tuple[str, str]]] = {}

        for name in list(self.model_tensors.keys()):
            stripped = self._strip_model_prefix(name)
            if self._skip_layer_tensor(stripped):
                consumed.add(name)
                continue
            match = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$", stripped)
            if match is None:
                continue

            bid = int(match.group(1))
            xid = int(match.group(2))
            wid = match.group(3)
            qtype = self._deepseek4_expert_qtypes.get(wid, default_qtype)
            if qtype is None:
                raise RuntimeError(f"No DeepSeek V4 expert quantization type selected for {wid}")
            scale_name = f"{stripped.removesuffix('.weight')}.scale"
            model_scale_name = scale_name if scale_name in self.model_tensors else f"model.{scale_name}"
            if model_scale_name not in self.model_tensors:
                raise ValueError(f"Missing DeepSeek V4 FP4 scale tensor for {stripped}")

            groups.setdefault((bid, wid), {})[xid] = (name, model_scale_name)
            consumed.update((name, model_scale_name))

        def convert_one(name: str, model_scale_name: str, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
            weight = LazyTorchTensor.to_eager(self.model_tensors[name]())
            scale = LazyTorchTensor.to_eager(self.model_tensors[model_scale_name]())

            if qtype == gguf.GGMLQuantizationType.MXFP4:
                data, _ = self._pack_fp4_as_mxfp4(weight, scale)
                return data

            data = self._dequant_fp4_weight(weight, scale).numpy()
            return self._quantize_deepseek4_expert(data, qtype)

        def add_merged_tensor(bid: int, wid: str, qtype: gguf.GGMLQuantizationType, experts: dict[int, np.ndarray]) -> None:
            missing = sorted(set(range(n_experts)).difference(experts))
            if missing:
                raise ValueError(f"Missing DeepSeek V4 expert tensors for layer {bid} {wid}: {missing[:8]}")

            merged = np.stack([experts[i] for i in range(n_experts)], axis=0)
            merged_name = f"layers.{bid}.ffn.experts.{wid}.weight"
            new_name = self.map_tensor_name(merged_name)
            shape = gguf.quant_shape_from_byte_shape(merged.shape, qtype) if merged.dtype == np.uint8 else merged.shape
            shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"
            logger.info(f"{new_name}, DeepSeek FP4 --> {qtype.name}, shape = {shape_str}")
            self.gguf_writer.add_tensor(new_name, merged, raw_dtype=qtype)

        worker_count = max(1, self.deepseek4_expert_workers)
        for bid, wid in sorted(groups):
            qtype = self._deepseek4_expert_qtypes.get(wid, default_qtype)
            if qtype is None:
                raise RuntimeError(f"No DeepSeek V4 expert quantization type selected for {wid}")
            group = groups[(bid, wid)]
            experts: dict[int, np.ndarray] = {}
            logger.info(
                "DeepSeek V4: quantizing blk.%d %s experts to %s with %d worker%s",
                bid,
                wid,
                qtype.name,
                worker_count,
                "" if worker_count == 1 else "s",
            )

            if worker_count == 1:
                for done, xid in enumerate(sorted(group), start=1):
                    name, model_scale_name = group[xid]
                    experts[xid] = convert_one(name, model_scale_name, qtype)
                    if done % 32 == 0 or done == n_experts:
                        logger.info("DeepSeek V4: blk.%d %s %d/%d experts", bid, wid, done, n_experts)
            else:
                max_pending = worker_count * 2
                pending: dict[concurrent.futures.Future[np.ndarray], int] = {}
                xids = iter(sorted(group))
                done = 0

                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    def submit_next() -> bool:
                        try:
                            xid = next(xids)
                        except StopIteration:
                            return False
                        name, model_scale_name = group[xid]
                        future = executor.submit(convert_one, name, model_scale_name, qtype)
                        pending[future] = xid
                        return True

                    while len(pending) < max_pending and submit_next():
                        pass

                    while pending:
                        finished, _ = concurrent.futures.wait(
                            pending,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in finished:
                            xid = pending.pop(future)
                            experts[xid] = future.result()
                            done += 1
                            if done % 32 == 0 or done == n_experts:
                                logger.info("DeepSeek V4: blk.%d %s %d/%d experts", bid, wid, done, n_experts)
                            submit_next()

            add_merged_tensor(bid, wid, qtype, experts)

        return consumed

    def _prepare_deepseek4_scaled_tensors(self) -> None:
        block_size = (self.hparams.get("quantization_config") or {}).get("weight_block_size", [128, 128])
        consumed: set[str] = set()

        for name in list(self.model_tensors.keys()):
            stripped = self._strip_model_prefix(name)
            if stripped.startswith("mtp.") or self._skip_layer_tensor(stripped):
                consumed.add(name)

        consumed.update(self._write_deepseek4_tid2eid_tensors())
        consumed.update(self._write_deepseek4_expert_tensors())

        for name in list(self.model_tensors.keys()):
            if name in consumed:
                continue
            stripped = self._strip_model_prefix(name)
            if not stripped.endswith(".scale"):
                continue
            if re.match(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.scale$", stripped) is not None:
                continue

            weight_name = f"{stripped.removesuffix('.scale')}.weight"
            model_weight_name = weight_name if weight_name in self.model_tensors else f"model.{weight_name}"
            if model_weight_name not in self.model_tensors:
                raise ValueError(f"Missing DeepSeek V4 FP8 weight tensor for scale {stripped}")

            w = self.model_tensors[model_weight_name]
            s = self.model_tensors[name]
            self.model_tensors[model_weight_name] = (
                lambda w=w, s=s, bs=block_size: self._dequant_fp8_weight(
                    LazyTorchTensor.to_eager(w()),
                    LazyTorchTensor.to_eager(s()),
                    bs,
                )
            )
            consumed.add(name)

        for name in consumed:
            self.model_tensors.pop(name, None)

    def prepare_tensors(self):
        self._prepare_deepseek4_scaled_tensors()

        if any(name.endswith(".scale") for name in self.model_tensors):
            raise NotImplementedError("Unhandled DeepSeek V4 scale tensors remain after conversion preparation")

        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del name
        del new_name
        del bid

        if not self._is_low_bit_ftype(self.ftype) or n_dims <= 1:
            return False

        # DeepSeek V4 routed experts are handled in _write_deepseek4_expert_tensors(),
        # where each expert is converted directly from FP4 to the requested compact
        # GGUF type.  Keep the rest of the model in float form so attention,
        # hyper-connections, indexers, compressors, shared experts and logits do not
        # inherit the global low-bit file type.
        return gguf.GGMLQuantizationType.F16

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        mapped = self._map_tensor_name_deepseek4(name)
        if mapped is not None:
            return mapped
        return super().map_tensor_name(name, try_suffixes)

    def _map_tensor_name_deepseek4(self, name: str) -> str | None:
        if name.startswith("model."):
            name = name.removeprefix("model.")

        top_level: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "embed.weight":    (gguf.MODEL_TENSOR.TOKEN_EMBD, ".weight"),
            "norm.weight":     (gguf.MODEL_TENSOR.OUTPUT_NORM, ".weight"),
            "head.weight":     (gguf.MODEL_TENSOR.OUTPUT, ".weight"),
            "hc_head_base":    (gguf.MODEL_TENSOR.OUTPUT_HC_BASE, ".weight"),
            "hc_head_fn":      (gguf.MODEL_TENSOR.OUTPUT_HC_FN, ".weight"),
            "hc_head_scale":   (gguf.MODEL_TENSOR.OUTPUT_HC_SCALE, ".weight"),
        }
        if name in top_level:
            tensor, suffix = top_level[name]
            return self.format_tensor_name(tensor, suffix=suffix)

        match = re.match(r"layers\.(\d+)\.(.+)", name)
        if match is None:
            return None

        bid = int(match.group(1))
        rest = match.group(2)

        layer_level: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "hc_attn_base":                  (gguf.MODEL_TENSOR.HC_ATTN_BASE, ".weight"),
            "hc_attn_fn":                    (gguf.MODEL_TENSOR.HC_ATTN_FN, ".weight"),
            "hc_attn_scale":                 (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ".weight"),
            "hc_ffn_base":                   (gguf.MODEL_TENSOR.HC_FFN_BASE, ".weight"),
            "hc_ffn_fn":                     (gguf.MODEL_TENSOR.HC_FFN_FN, ".weight"),
            "hc_ffn_scale":                  (gguf.MODEL_TENSOR.HC_FFN_SCALE, ".weight"),
            "attn.attn_sink":                (gguf.MODEL_TENSOR.ATTN_SINKS, ".weight"),
            "attn.wq_a.weight":              (gguf.MODEL_TENSOR.ATTN_Q_A, ".weight"),
            "attn.wq_b.weight":              (gguf.MODEL_TENSOR.ATTN_Q_B, ".weight"),
            "attn.q_norm.weight":            (gguf.MODEL_TENSOR.ATTN_Q_A_NORM, ".weight"),
            "attn.wkv.weight":               (gguf.MODEL_TENSOR.ATTN_KV, ".weight"),
            "attn.kv_norm.weight":           (gguf.MODEL_TENSOR.ATTN_KV_A_NORM, ".weight"),
            "attn.wo_a.weight":              (gguf.MODEL_TENSOR.ATTN_OUT_A, ".weight"),
            "attn.wo_b.weight":              (gguf.MODEL_TENSOR.ATTN_OUT_B, ".weight"),
            "attn.compressor.ape":           (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_APE, ".weight"),
            "attn.compressor.wkv.weight":    (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_KV, ".weight"),
            "attn.compressor.wgate.weight":  (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_GATE, ".weight"),
            "attn.compressor.norm.weight":   (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_NORM, ".weight"),
            "attn.indexer.wq_b.weight":      (gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B, ".weight"),
            "attn.indexer.weights_proj.weight": (gguf.MODEL_TENSOR.INDEXER_PROJ, ".weight"),
            "attn.indexer.compressor.ape":   (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_APE, ".weight"),
            "attn.indexer.compressor.wkv.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_KV, ".weight"),
            "attn.indexer.compressor.wgate.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_GATE, ".weight"),
            "attn.indexer.compressor.norm.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_NORM, ".weight"),
            "attn_norm.weight":              (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
            "ffn_norm.weight":               (gguf.MODEL_TENSOR.FFN_NORM, ".weight"),
            "ffn.shared_experts.w1.weight":  (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "ffn.shared_experts.w3.weight":  (gguf.MODEL_TENSOR.FFN_UP_SHEXP, ".weight"),
            "ffn.shared_experts.w2.weight":  (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
            "ffn.gate.weight":               (gguf.MODEL_TENSOR.FFN_GATE_INP, ".weight"),
            "ffn.gate.bias":                 (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            "ffn.gate.tid2eid":              (gguf.MODEL_TENSOR.FFN_GATE_TID2EID, ".weight"),
            "ffn.experts.w1.weight":         (gguf.MODEL_TENSOR.FFN_GATE_EXP, ".weight"),
            "ffn.experts.w3.weight":         (gguf.MODEL_TENSOR.FFN_UP_EXP, ".weight"),
            "ffn.experts.w2.weight":         (gguf.MODEL_TENSOR.FFN_DOWN_EXP, ".weight"),
        }
        if rest in layer_level:
            tensor, suffix = layer_level[rest]
            return self.format_tensor_name(tensor, bid, suffix=suffix)

        return None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model."):
            name = name.removeprefix("model.")

        # TODO: llama.cpp does not have Multi-Token Prediction for DeepSeek yet.
        if name.startswith("mtp."):
            return

        # process the experts separately
        match = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight", name)
        if match is not None:
            bid = int(match.group(1))
            xid = int(match.group(2))
            wid = match.group(3)
            n_experts = self.hparams["n_routed_experts"]

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["w1", "w3", "w2"]:
                    datas: list[Tensor] = []

                    for expert_id in range(n_experts):
                        ename = f"layers.{bid}.ffn.experts.{expert_id}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"layers.{bid}.ffn.experts.{w_name}.weight"
                    yield self.map_tensor_name(merged_name), data_torch
                return

            del xid, wid
            return

        yield self.map_tensor_name(name), data_torch
