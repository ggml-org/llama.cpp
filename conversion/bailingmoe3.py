from __future__ import annotations

import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, gguf

from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("BailingMoeV3ForCausalLM")
@ModelBase.example("inclusionAI/Ling-3.0-tiny", "inclusionAI/Ling-3.0-flash")
class BailingMoeV3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE3
    supports_mtp_export = True

    _experts: list[dict[str, Tensor]] | None = None
    _main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nextn_layers = self.hparams.get("num_nextn_predict_layers", 0) or 0
        if self.no_mtp:
            nextn_layers = 0
        self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def is_full_attention(self, bid: int) -> bool:
        n_layer = self.hparams["num_hidden_layers"]
        layer_group_size = self.hparams["layer_group_size"]
        return bid >= n_layer or (bid + 1) % layer_group_size == 0 or bid >= n_layer // layer_group_size * layer_group_size

    def set_gguf_parameters(self):
        if not self.hparams.get("no_kda_lora", False):
            raise ValueError("BailingMoeV3 KDA LoRA projections are not supported")
        if not self.hparams.get("kda_safe_gate", False):
            raise ValueError("BailingMoeV3 non-safe KDA gates are not supported")
        if self.hparams.get("gated_attention_proj_granularity_type") != "head_wise":
            raise ValueError("BailingMoeV3 requires head-wise attention gates")

        self.hparams["num_key_value_heads"] = 1
        super().set_gguf_parameters()

        n_head_kv = [1 if self.is_full_attention(il) else 0 for il in range(self.block_count)]
        self.gguf_writer.add_head_count_kv(n_head_kv)

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_ssm_conv_kernel(self.hparams["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(self.hparams["head_dim"])
        self.gguf_writer.add_kda_safe_gate(self.hparams["kda_safe_gate"])
        self.gguf_writer.add_kda_gate_lower_bound(self.hparams["kda_lower_bound"])

        kv_lora_rank = self.hparams["kv_lora_rank"]
        qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
        qk_rope_head_dim = self.hparams["qk_rope_head_dim"]
        if (q_lora_rank := self.hparams.get("q_lora_rank")) is not None:
            self.gguf_writer.add_q_lora_rank(q_lora_rank)
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        self.gguf_writer.add_key_length_mla(qk_nope_head_dim + qk_rope_head_dim)
        self.gguf_writer.add_value_length_mla(self.hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["moe_shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(self.hparams.get("num_shared_experts", 1))
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])

        def clamp_limits(key: str) -> list[float] | None:
            values = self.hparams.get(key)
            if values is None:
                return None
            values = [0.0 if value is None else float(value) for value in values[:self.block_count]]
            return values + [0.0] * (self.block_count - len(values))

        if (values := clamp_limits("expert_swiglu_limit_list")) is not None:
            self.gguf_writer.add_swiglu_clamp_exp(values)
        if (values := clamp_limits("share_expert_swiglu_limit_list")) is not None:
            self.gguf_writer.add_swiglu_clamp_shexp(values)

        if not self.no_mtp and (nextn_layers := self.hparams.get("num_nextn_predict_layers", 0)):
            self.gguf_writer.add_nextn_predict_layers(nextn_layers)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.endswith(".expert_bias"):
            name += ".bias"

        if cls._main_layers is None:
            return super().filter_tensors((name, gen))

        m = re.match(r"model\.layers\.(\d+)\.", name)
        is_mtp = m is not None and int(m.group(1)) >= cls._main_layers

        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and (cls.mtp_shared_embd or name not in (
            "model.word_embeddings.weight", "model.norm.weight", "lm_head.weight",
        )):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")) and data_torch.ndim in (2, 3):
            d_inner = data_torch.shape[0]
            d_conv = data_torch.shape[-1]
            data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        if name.endswith(".A_log"):
            data_torch = torch.exp(data_torch).reshape(-1, 1)

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        if name.endswith(".attention.f_proj.weight"):
            assert bid is not None
            if self.is_full_attention(bid):
                raise ValueError(f"unexpected f_proj on full-attention layer {bid}")
            name = self.format_tensor_name(gguf.MODEL_TENSOR.SSM_F_A, bid)

        if name.endswith(".attention.g_proj.weight"):
            assert bid is not None
            tensor = gguf.MODEL_TENSOR.ATTN_GATE if self.is_full_attention(bid) else gguf.MODEL_TENSOR.SSM_G_A
            name = self.format_tensor_name(tensor, bid)

        if ".mlp.experts." in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for weight_name in ("down_proj", "gate_proj", "up_proj"):
                    tensors = []
                    for expert_id in range(n_experts):
                        expert_name = f"model.layers.{bid}.mlp.experts.{expert_id}.{weight_name}.weight"
                        tensors.append(self._experts[bid].pop(expert_name))
                    merged_name = f"model.layers.{bid}.mlp.experts.{weight_name}.weight"
                    yield from super().modify_tensors(torch.stack(tensors, dim=0), merged_name, bid)
            return

        if name.endswith(".attention.kv_b_proj.weight"):
            assert bid is not None
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            name_k = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid)
            name_v = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid)
            yield from super().modify_tensors(k_b.transpose(1, 2), name_k, bid)
            yield from super().modify_tensors(v_b, name_v, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [name for layer in self._experts for name in layer]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("BailingMoeV3VLForConditionalGeneration")
@ModelBase.example("inclusionAI/Ling-3.0-flash-VL")
class BailingMoeV3VLModel(BailingMoeV3Model):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE3

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # hoist text_config before the shared BailingMoeV3 logic runs:
        # ModelBase.__init__ calls this with the raw VL config, where the text
        # dims still live under text_config
        if "text_config" in self.hparams:
            self.hparams = {**self.hparams, **self.hparams["text_config"]}
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        mrope_section = self.hparams.get("mrope_section")
        if mrope_section is None:
            raise ValueError("BailingMoeV3VL requires mrope_section in the config")
        if sum(mrope_section[:3]) * 2 != self.hparams["qk_rope_head_dim"]:
            raise ValueError(
                f"mrope_section {mrope_section[:3]} counts rope pairs and must sum to"
                f" qk_rope_head_dim / 2 = {self.hparams['qk_rope_head_dim'] // 2}"
            )
        # mrope_section is [t, h, w]; pad to the 4-wide sections array
        self.gguf_writer.add_rope_dimension_sections(list(mrope_section[:3]) + [0])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip projector tensors; the vision tower is skipped by TextModel.filter_tensors
        if name.startswith("linear_proj"):
            return None

        return super().filter_tensors(item)


@ModelBase.register("BailingMoeV3VLForConditionalGeneration")
@ModelBase.example("inclusionAI/Ling-3.0-flash-VL")
class BailingMoeV3VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None

        if self.hparams_vision.get("disable_merger_proj") is not True:
            raise ValueError("BailingMoeV3VL requires disable_merger_proj=true")

        # out_hidden_size is the vision encoder output (post spatial merge, pre linear_proj)
        self.image_emb_dim = self.hparams_vision.get("out_hidden_size")
        if self.image_emb_dim is None:
            raise ValueError("BailingMoeV3VL vision config requires out_hidden_size")

    def set_gguf_parameters(self):
        assert self.hparams_vision is not None
        MmprojModel.set_gguf_parameters(self)  # skip Qwen3VLVisionModel parameters
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LING3VL)
        self.gguf_writer.add_vision_use_gelu(True)

        merge_size = self.hparams_vision.get("spatial_merge_size")
        if merge_size is not None:
            self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

        rms_norm_eps = self.global_config.get("text_config", {}).get("rms_norm_eps", 1e-6)
        self.gguf_writer.add_vision_attention_layernorm_eps(rms_norm_eps)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("lm_head."):
            return None

        if name.startswith("linear_proj"):
            # top-level projector MLP: linear_proj.0 -> mm.0, linear_proj.2 -> mm.2
            parts = name.split(".")
            if len(parts) != 3:
                raise ValueError(f"Unexpected linear_proj tensor: {name}")
            idx, suffix = int(parts[1]), parts[2]
            name = f"mm.{idx}.{suffix}"
            # the qwen3vl filter keeps only visual.*; skip it for the renamed projector tensors
            return MmprojModel.filter_tensors((name, gen))

        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)

        if not name.startswith("visual."):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        assert self.hparams_vision is not None

        if name.startswith("mm.0.") or name.startswith("mm.2."):
            # top-level projector MLP (linear_proj.0 / linear_proj.2, renamed by filter_tensors)
            yield (name, data_torch)
            return

        if name == "visual.merger.norm.weight" or name == "visual.merger.norm.bias":
            # the merger is norm-only for Ling: per-patch LayerNorm before the spatial merge
            new_name = f"mm.input_norm.{name.split('.')[-1]}"
            yield (new_name, data_torch)
            return

        # Ling has no patch bias; the Conv3D split below matches the stock qwen3vl path
        yield from Qwen3VLVisionModel.modify_tensors(self, data_torch, name, bid)
