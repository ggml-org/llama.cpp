from __future__ import annotations

import json
from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf


def _unpermute_for_rope(tensor: "Tensor", n_heads: int) -> "Tensor":
    """Invert transformers' `_permute_for_rope`: HF stores Q/K in rotate_half layout,
    llama.cpp consumes the interleaved (NORM) layout."""
    if tensor.ndim == 2:
        dim1, dim2 = tensor.shape
        return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    if tensor.ndim == 1:
        (dim1,) = tensor.shape
        return tensor.view(n_heads, 2, dim1 // n_heads // 2).transpose(1, 2).reshape(dim1)
    raise ValueError(f"_unpermute_for_rope: unexpected shape {tuple(tensor.shape)}")


@ModelBase.register("MuseGlimmerForConditionalGeneration")
@ModelBase.example("meta-models/Muse-Glimmer-30B")
class MuseGlimmerModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MUSE_GLIMMER

    def norm_shift(self, name: str) -> float:
        # All four layer norms use 1, the final norm uses 0.
        return 1.0 if name.endswith("layernorm.weight") else 0.0

    def set_vocab(self):
        self._set_vocab_gpt2()

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.dir_model)
        eot_id = tok.convert_tokens_to_ids("<|eot|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            self.gguf_writer.add_eot_token_id(eot_id)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_final_logit_softcapping(hparams["final_logit_softcapping"])
        self.gguf_writer.add_logit_scale(hparams["output_multiplier"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hparams["layer_types"]])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        shift = self.norm_shift(name)
        if shift != 0.0:
            data_torch = data_torch + shift

        # Invert transformers' `_permute_for_rope` on Q/K, we keep ggml's NORM (interleaved) rope
        if ".self_attn.q_proj." in name:
            data_torch = _unpermute_for_rope(data_torch, int(self.hparams["num_attention_heads"]))
        elif ".self_attn.k_proj." in name:
            data_torch = _unpermute_for_rope(data_torch, int(self.hparams["num_key_value_heads"]))

        # Synthesize QK-norm weights to absorb qk_scale_factor.
        # MuseGlimmer implementation: scaleless RMSNorm followed by qk_scale_factor..
        if bid is not None and name.endswith(f"model.layers.{bid}.self_attn.q_proj.weight"):
            head_dim = self.hparams["head_dim"]
            q_scale = float(self.hparams["qk_scale_factor"])
            yield (
                self.map_tensor_name(f"model.layers.{bid}.self_attn.q_norm.weight"),
                torch.full((head_dim,), q_scale, dtype=torch.float32),
            )
            yield (
                self.map_tensor_name(f"model.layers.{bid}.self_attn.k_norm.weight"),
                torch.ones((head_dim,), dtype=torch.float32),
            )

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MuseGlimmerForConditionalGeneration")
@ModelBase.example("meta-models/Muse-Glimmer-30B")
class MuseGlimmerVisionModel(MmprojModel):
    def get_vision_config(self) -> dict[str, Any] | None:
        c = self.global_config.get("vision_config")
        if not c:
            return None
        # MuseGlimmer actually uses dynamic size, initialize with nominal size
        image_size = c["pos_emb_height"] * c["patch_size"] * c["merge_size"]
        return {**c, "image_size": image_size}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        c = self.hparams_vision  # enriched vision_config from get_vision_config()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MUSE_GLIMMER)
        self.gguf_writer.add_vision_attention_layernorm_eps(float(c["layer_norm_eps"]))
        self.gguf_writer.add_vision_spatial_merge_size(int(c["merge_size"]))

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        keep = ("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.")
        if not any(name.startswith(k) for k in keep):
            return None
        return super().filter_tensors((name, gen))

    # 3-layer projector MLP
    _MM_MLP_MAP = {
        "model.vision_adapter.fc1": (gguf.MODEL_TENSOR.V_MMPROJ, 0),
        "model.vision_adapter.fc2": (gguf.MODEL_TENSOR.V_MMPROJ, 1),
        "model.vision_projection":  (gguf.MODEL_TENSOR.V_MMPROJ, 2),
    }

    def modify_tensors(self, data_torch, name, bid):
        assert self.hparams_vision is not None
        if ".attn.q_proj." in name or ".attn.k_proj." in name:
            n_heads = int(self.hparams_vision["num_attention_heads"])
            data_torch = _unpermute_for_rope(data_torch, n_heads)
        # Lay out the pt=2 temporal slabs of the patch embedding as a conv2d for build_inp()
        if name.endswith("patch_embedder.patch_embedding.weight"):
            n_embd = data_torch.shape[0]
            pt = int(self.hparams_vision["patch_temporal"])
            ps = int(self.hparams_vision["patch_size"])
            data_torch = data_torch.view(n_embd, pt, 3, ps, ps).sum(dim=1)  # (n_embd, 3, ps, ps)
        stem, _, suffix = name.rpartition(".")
        if stem in self._MM_MLP_MAP:
            tensor_key, idx = self._MM_MLP_MAP[stem]
            yield (self.format_tensor_name(tensor_key, bid=idx, suffix="." + suffix), data_torch)
            return
        yield (self.map_tensor_name(name), data_torch)


@ModelBase.register("MuseGlimmerAssistantModel")
@ModelBase.example("meta-models/Muse-Glimmer-30B-assistant")
class MuseGlimmerAssistantModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DFLASH

    def set_vocab(self):
        if self.target_model_dir is None:
            raise ValueError(
                "MuseGlimmerAssistant (DFlash drafter) requires --target-model-dir pointing to the "
                "target MuseGlimmer HF directory"
            )

        original_dir = self.dir_model
        self.dir_model = self.target_model_dir

        from . import get_model_class
        with open(self.target_model_dir / "config.json", "r", encoding="utf-8") as f:
            target_arch = json.load(f)["architectures"][0]
        target_cls = get_model_class(target_arch)
        if target_cls is not type(self):
            target_cls.set_vocab(self)  # ty: ignore[unresolved-attribute]
        else:
            super().set_vocab()

        self.dir_model = original_dir

        mask_token_id = self.hparams.get("mask_token_id")
        if mask_token_id is not None:
            self.gguf_writer.add_mask_token_id(int(mask_token_id))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        h = self.hparams

        self.gguf_writer.add_block_size(int(h["block_size"]))

        # dflash.target_layers[k] refers to the inputs going into the ith layer, which come from the (i-1)th layer's output.
        # The transformers configuration refers to the outputs being recorded.
        self.gguf_writer.add_target_layers([int(x) + 1 for x in h["target_layer_ids"]])

        if h.get("sliding_window") and h.get("layer_types"):
            self.gguf_writer.add_sliding_window(int(h["sliding_window"]))
            self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in h["layer_types"]])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # DFlash defaults to NEOX (rotate_half) rope, matching transformers HF layout for Q/K, QK-norms
        # no permutation needed.
        yield (self.map_tensor_name(name), data_torch)
