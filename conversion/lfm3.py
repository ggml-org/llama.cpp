from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf


@ModelBase.register("Lfm3ForCausalLM", "Lfm3VlForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM3-1.7B", "LiquidAI/LFM3-1.7B-Base")
class LFM3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.LFM3

    def set_gguf_parameters(self):
        layer_types = self.hparams["layer_types"]
        n_kv_heads = self.hparams["num_key_value_heads"]

        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            n_kv_heads if lt != "conv" else 0
            for lt in layer_types
        ]

        super().set_gguf_parameters()

        self.gguf_writer.add_rope_dimension_count(self.hparams["head_dim"])

        self.gguf_writer.add_sliding_window_pattern([lt == "sliding_attention" for lt in layer_types])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])

        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # Partial RoPE is represented by masking trailing frequency pairs.
        # rope_parameters is keyed by layer type; only full_attention layers use partial RoPE.
        rope_parameters = self.hparams["rope_parameters"]["full_attention"]
        partial = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = self.hparams["head_dim"]
        rope_angles = int(partial * head_dim) // 2
        n_pairs = head_dim // 2
        assert 0 < rope_angles <= n_pairs
        values = [1.0] * rope_angles + [1e30] * (n_pairs - rope_angles)
        rope_freqs = torch.tensor(values, dtype=torch.float32)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), rope_freqs)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # conv op requires 2d tensor
        if "conv.conv" in name:
            data_torch = data_torch.squeeze(1)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Lfm3VlForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM3-VL-2B")
class LFM3VLMmprojModel(MmprojModel):
    has_vision_encoder = True
    has_audio_encoder = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # dynamic resolution, image_size is only set for MmprojModel compatibility
        self.hparams_vision["image_size"] = 512

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        # clip.cpp hardcodes the vision rope base, the RGB patchifier, the 3x3 spatial conv and the GELU projector
        if (self.hparams_vision["rope_parameters"]["rope_theta"] != 1e6 or self.hparams_vision["in_channels"] != 3
                or self.hparams_vision["conv_kernel_size"] != 3):
            raise ValueError("unsupported vision rope_theta / in_channels / conv_kernel_size")
        if (self.global_config["projector_hidden_act"] != "gelu" or not self.global_config["projector_bias"]
                or self.global_config["projector_layer_norm_eps"] != 1e-6):
            raise ValueError("unsupported projector_hidden_act / projector_bias / projector_layer_norm_eps")
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.LFM3VL)
        self.gguf_writer.add_vision_head_count_kv(self.hparams_vision["num_key_value_heads"])
        self.gguf_writer.add_vision_spatial_merge_size(self.global_config["downsample_factor"])
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision["norm_eps"])
        self.gguf_writer.add_vision_use_silu(True)
        size = self.preprocessor_config["size"]
        self.gguf_writer.add_vision_min_pixels(size["shortest_edge"])
        self.gguf_writer.add_vision_max_pixels(size["longest_edge"])
        # PIL resample enum, the torchvision backend has no lanczos
        resample = self.preprocessor_config["resample"]
        resize_algo = {2: "bilinear", 3: "bicubic"}.get(resample)
        if resize_algo is None:
            raise ValueError(f"unsupported resample: {resample}")
        self.gguf_writer.add_vision_image_resize_algo(resize_algo)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if "patch_embd" in new_name:
            return gguf.GGMLQuantizationType.F16
        if "conv2d_spatial" in new_name:
            # CUDA CONV_2D_DW requires an f32 kernel; the tensor is tiny
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if not name.startswith(("model.vision_tower.", "model.multi_modal_projector.")):
            return

        if name == "model.vision_tower.patchifier.weight":
            # qwen2vl-style layout [channel, temporal, ph, pw], one conv kernel per temporal slice
            assert self.hparams_vision is not None
            t = self.hparams_vision["temporal_patch_size"]
            p = self.hparams_vision["patch_size"]
            n_embd = data_torch.shape[0]
            w = data_torch.view(n_embd, 3, t, p, p)
            yield ("v.patch_embd.weight", w[:, :, 0].contiguous())
            if t == 2:
                yield ("v.patch_embd.weight.1", w[:, :, 1].contiguous())
            else:
                assert t == 1, f"unsupported vision_temporal_patch_size {t}"
            return
        if name == "model.vision_tower.embedding_norm.weight":
            yield ("v.embedding_norm.weight", data_torch)
            return
        if "conv.in_proj.weight" in name:
            yield (f"v.blk.{bid}.conv2d_in_proj.weight", data_torch)
            return
        if "conv.spatial_conv.weight" in name:
            yield (f"v.blk.{bid}.conv2d_spatial.weight", data_torch)
            return
        if "conv.out_proj.weight" in name:
            yield (f"v.blk.{bid}.conv2d_out_proj.weight", data_torch)
            return

        # projector: input LayerNorm -> linear_1 -> GELU -> linear_2
        if name.startswith("model.multi_modal_projector."):
            layer, suffix = name.removeprefix("model.multi_modal_projector.").split(".")
            yield ({"layer_norm": "mm.input_norm", "linear_1": "mm.0", "linear_2": "mm.2"}[layer] + "." + suffix, data_torch)
            return

        yield (self.map_tensor_name(name), data_torch)
