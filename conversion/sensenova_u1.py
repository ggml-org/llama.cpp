from __future__ import annotations

from pathlib import Path

import torch

from .base import MmprojModel, ModelBase, gguf
from .qwen import Qwen2Model


@ModelBase.example("sensenova/SenseNova-U1.5-8B-MoT")
@ModelBase.register("NEOChatModel")
class SenseNovaU1Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.SENSENOVA_U1

    @classmethod
    def filter_tensors(cls, item):
        name, _ = item
        if not name.startswith("language_model.") or "_mot_gen" in name:
            return None
        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        if int(self.hparams.get("num_experts") or 0) > 1:
            raise ValueError("SenseNova U1 conversion currently supports the dense understanding branch")
        if self.rope_parameters.get("rope_type", "default") not in ("default", None):
            raise ValueError("SenseNova U1 scaled RoPE is not supported")
        if self.hparams.get("attention_bias", False):
            raise ValueError("SenseNova U1 attention bias is not supported")
        if self.hparams.get("use_sliding_window", False):
            raise ValueError("SenseNova U1 sliding attention is not supported")
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_freq_base_spatial(self.hparams["rope_theta_hw"])

    def get_tensors(self):
        for name, gen in self.model_tensors.items():
            if name.endswith((".q_norm_hw.weight", ".k_norm_hw.weight")):
                continue
            data = gen()
            if name.endswith((".q_norm.weight", ".k_norm.weight")):
                spatial = self.model_tensors[name.replace("_norm.weight", "_norm_hw.weight")]()
                # Keep one weight vector; the graph normalizes its two halves independently.
                data = torch.cat((data, spatial), dim=0)
            yield name, data


@ModelBase.register("NEOVisionModel")
@ModelBase.example("sensenova/SenseNova-U1.5-8B-MoT")
class SenseNovaU1VisionModel(MmprojModel):
    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        else:
            hparams = dict(hparams)

        vision_config = dict(hparams["vision_config"])
        # NEOVisionModel has no transformer blocks; these defaults satisfy
        # the generic mmproj metadata contract without replacing checkpoint values.
        vision_config.setdefault("image_size", 0)
        vision_config.setdefault("intermediate_size", 0)
        vision_config.setdefault("num_hidden_layers", 0)
        vision_config.setdefault("num_attention_heads", 1)
        hparams["vision_config"] = vision_config
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

        self.preprocessor_config.setdefault("image_mean", [0.485, 0.456, 0.406])
        self.preprocessor_config.setdefault("image_std", [0.229, 0.224, 0.225])

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith("vision_model."):
            return None
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type("sensenova_u1")
        vision_config = self.hparams["vision_config"]
        # These defaults match the known U1.5 checkpoint when older configs omit fields.
        hidden_size = int(vision_config.get("hidden_size", 1024))
        min_pixels = int(vision_config.get("min_pixels", 65536))
        max_pixels = int(vision_config.get("max_pixels", 16777216))
        downsample_ratio = float(vision_config.get("downsample_ratio", 0.5))
        if downsample_ratio <= 0:
            raise ValueError(f"SenseNova U1 downsample_ratio must be positive, got {downsample_ratio}")
        merge_size = round(1.0 / downsample_ratio)
        if abs((1.0 / downsample_ratio) - merge_size) > 1e-6:
            raise ValueError(f"SenseNova U1 downsample_ratio must have an integer reciprocal, got {downsample_ratio}")
        norm_eps = float(vision_config.get("layer_norm_eps", vision_config.get("rms_norm_eps", 1e-6)))
        self.gguf_writer.add_vision_head_dim(hidden_size)
        self.gguf_writer.add_vision_attention_layernorm_eps(norm_eps)
        self.gguf_writer.add_vision_min_pixels(min_pixels)
        self.gguf_writer.add_vision_max_pixels(max_pixels)
        self.gguf_writer.add_vision_spatial_merge_size(merge_size)

    def modify_tensors(self, data_torch, name, bid):
        tensor_names = {
            "vision_model.embeddings.patch_embedding.weight": "v.patch_embd.weight",
            "vision_model.embeddings.patch_embedding.bias":   "v.patch_embd.bias",
            "vision_model.embeddings.dense_embedding.weight": self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, 0),
            "vision_model.embeddings.dense_embedding.bias": self.format_tensor_name(
                gguf.MODEL_TENSOR.V_MMPROJ, 0, suffix=".bias"
            ),
        }
        if name not in tensor_names:
            raise ValueError(f"Unexpected SenseNova U1 vision tensor: {name!r}")
        return [(tensor_names[name], data_torch)]

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, 0):
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
