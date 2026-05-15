from __future__ import annotations

import math
import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, _MISTRAL_COMMON_DATASET_MEAN, _MISTRAL_COMMON_DATASET_STD, gguf

from .qwen import Qwen3Model


@ModelBase.register("StepVLForConditionalGeneration")
class Step3VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None

        if not self.hparams_vision.get("intermediate_size"):
            hidden_size = self.hparams_vision.get("hidden_size") or self.hparams_vision.get("width") or 0
            assert hidden_size > 0
            mlp_ratio = float(self.hparams_vision.get("mlp_ratio", 8960 / 1536))
            self.hparams_vision["intermediate_size"] = int(round(hidden_size * mlp_ratio))

        self.preprocessor_config.setdefault("image_mean", list(_MISTRAL_COMMON_DATASET_MEAN))
        self.preprocessor_config.setdefault("image_std", list(_MISTRAL_COMMON_DATASET_STD))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        projector_stride = int(self.global_config.get("understand_projector_stride", -1))
        hidden_size = int(self.hparams_vision.get("hidden_size", self.hparams_vision.get("width", -1)))
        num_layers = int(self.hparams_vision.get("num_hidden_layers", self.hparams_vision.get("layers", -1)))
        assert (projector_stride, int(self.hparams_vision.get("image_size", -1)), hidden_size, num_layers) == (2, 728, 1536, 47), (
            "current Step3-VL conversion path is only validated for Step3-VL-10B"
        )

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.STEP3VL)
        self.gguf_writer.add_vision_attention_layernorm_eps(float(self.hparams_vision.get("layer_norm_eps", 1e-5)))
        self.gguf_writer.add_vision_projector_scale_factor(projector_stride ** 2)
        # 3024 max resize comes from step3-vl-10b processing_step3.py.
        self.gguf_writer.add_vision_preproc_image_size(3024)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".position_embd." in new_name:
            return gguf.GGMLQuantizationType.F32
        if ("mm.0." in new_name or "mm.1." in new_name) and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(("model.", "lm_head.")):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("vision_model.vit_downsampler"):
            match = re.match(r"vision_model\.vit_downsampler(\d+)\.(weight|bias)", name)
            if match is None:
                raise ValueError(f"Unexpected Step3-VL projector tensor {name!r}")

            proj_id = int(match.group(1)) - 1
            suffix = f".{match.group(2)}"
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, proj_id, suffix=suffix), data_torch)
            return

        if name == "vit_large_projector.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ_FC), data_torch)
            return

        if name.startswith("vision_model."):
            if name == "vision_model.positional_embedding":
                name += ".weight"
            elif name.endswith(".gamma") and ".ls_" in name:
                name = name.removesuffix(".gamma") + ".weight"

            name = name.replace("attn.in_proj_weight", "attn.in_proj.weight")
            name = name.replace("attn.in_proj_bias", "attn.in_proj.bias")

            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("StepVLForConditionalGeneration")
class Step3VLTextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3


@ModelBase.register("Step3p5ForCausalLM")
class Step35Model(TextModel):
    model_arch = gguf.MODEL_ARCH.STEP35

    def set_gguf_parameters(self):
        rope_theta = self.hparams.get("rope_theta")
        if isinstance(rope_theta, list):
            self.hparams["rope_theta"] = float(rope_theta[0])
            self.hparams["local_rope_theta"] = float(rope_theta[1])
            self.rope_parameters["rope_theta"] = self.hparams["rope_theta"]
            self.rope_parameters["sliding_attention"] = {"rope_theta": self.hparams["local_rope_theta"]}

        super().set_gguf_parameters()

        layer_types = self.hparams.get("layer_types") or []
        partial_rotary_factors = self.hparams.get("partial_rotary_factors") or []
        attn_other = self.hparams.get("attention_other_setting") or {}

        n_head_base = self.hparams["num_attention_heads"]
        n_kv_base = self.hparams["num_attention_groups"]

        n_head_swa = attn_other.get("num_attention_heads", n_head_base)
        n_kv_swa = attn_other.get("num_attention_groups", n_kv_base)

        layer_types = layer_types[: self.block_count]
        partial_rotary_factors = partial_rotary_factors[: self.block_count]
        assert [1.0 if lt == "sliding_attention" else 0.5 for lt in layer_types] == partial_rotary_factors
        head_arr = [n_head_swa if lt == "sliding_attention" else n_head_base for lt in layer_types]
        kv_arr = [n_kv_swa if lt == "sliding_attention" else n_kv_base for lt in layer_types]
        swa_pat = [lt == "sliding_attention" for lt in layer_types]

        self.gguf_writer.add_head_count(head_arr)
        self.gguf_writer.add_head_count_kv(kv_arr)

        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(swa_pat)

        self.gguf_writer.add_value_length(self.hparams["head_dim"])

        # MoE params
        self.gguf_writer.add_expert_count(self.hparams["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(self.hparams["moe_top_k"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["share_expert_dim"])

        if (moe_router_scaling_factor := self.hparams.get("moe_router_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(moe_router_scaling_factor)
        if (norm_expert_weight := self.hparams.get("norm_expert_weight")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_expert_weight)

        # leading dense blocks
        leading_dense = 0
        moe_layers_enum = self.hparams.get("moe_layers_enum")
        if isinstance(moe_layers_enum, str) and moe_layers_enum.strip():
            moe_layers = sorted(int(i) for i in moe_layers_enum.strip().split(","))
            if moe_layers:
                leading_dense = max(0, moe_layers[0])
        self.gguf_writer.add_leading_dense_block_count(leading_dense)
        self.gguf_writer.add_moe_every_n_layers(int(self.hparams.get("moe_every_n_layer", 1)))

        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-5))

        # Optional per-layer SwiGLU clamps.
        if (limits := self.hparams.get("swiglu_limits")) is not None:
            limits_f = [0.0 if v is None else float(v) for v in limits[: self.block_count]]
            self.gguf_writer.add_swiglu_clamp_exp(limits_f)
        if (limits_shared := self.hparams.get("swiglu_limits_shared")) is not None:
            limits_shared_f = [0.0 if v is None else float(v) for v in limits_shared[: self.block_count]]
            self.gguf_writer.add_swiglu_clamp_shexp(limits_shared_f)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Map router bias (expert selection bias) to a GGUF bias tensor
        if name.endswith(".moe.router_bias"):
            name += ".bias"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # remove mtp layers
        if (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None:
            il = int(m.group(1))
            n_main = int(self.hparams.get("num_hidden_layers", self.block_count))
            if il >= n_main:
                return
        if name.endswith("norm.weight"):
            data_torch += 1.0

        if name.endswith((".self_attn.g_proj.weight", ".moe.gate.weight", ".moe.up_proj.weight", ".moe.gate_proj.weight", ".moe.down_proj.weight")):
            data_torch = data_torch.squeeze().contiguous()

        yield from super().modify_tensors(data_torch, name, bid)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # Step35 can optionally use Llama-3 style RoPE scaling (HF: rope_scaling.rope_type == "llama3").
        # llama.cpp represents this via a single extra tensor: "rope_freqs.weight" (aka MODEL_TENSOR.ROPE_FREQS).
        rope_params = self.rope_parameters.get("full_attention", self.rope_parameters)
        rope_type = rope_params.get("rope_type") or ""
        if rope_type.lower() != "llama3":
            return

        # Step35 configs can carry per-layer rope_theta as a list; for llama3 rope factors we use the base value.
        rope_theta = self.hparams.get("rope_theta", 10000.0)
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[0]
        base = float(rope_theta)
        if (dim := self.hparams.get("head_dim")) is None:
            dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        dim = int(dim)

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        factor = float(rope_params.get("factor", 8.0))
        low_freq_factor = float(rope_params.get("low_freq_factor", 1.0))
        high_freq_factor = float(rope_params.get("high_freq_factor", 4.0))
        old_context_len = int(rope_params.get("original_max_position_embeddings", self.hparams.get("original_max_position_embeddings", 8192)))

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        rope_factors: list[float] = []
        for freq in freqs:
            wavelen = 2 * math.pi / float(freq)
            if wavelen < high_freq_wavelen:
                rope_factors.append(1.0)
            elif wavelen > low_freq_wavelen:
                rope_factors.append(factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                rope_factors.append(1.0 / ((1.0 - smooth) / factor + smooth))

        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))
