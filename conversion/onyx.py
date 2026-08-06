from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf


@ModelBase.register("OnyxForConditionalGeneration")
class OnyxModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ONYX

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
        self.gguf_writer.add_post_norm_rms_eps(hparams["post_norm_eps"])

        # SWA + NoPE: [SW, SW, SW, Full], NoPE used on Full layers. References:
        # https://huggingface.co/someorgtoo-hf/onyx-hf-converted/blob/main/config.json#L19
        # https://huggingface.co/someorgtoo-hf/onyx-hf-converted/blob/main/config.json#L73
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(4)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        shift = self.norm_shift(name)
        if shift != 0.0:
            data_torch = data_torch + shift

        # Synthesize QK-norm weights to absorb qk_scale_factor.
        # Onyx implementation: scaleless RMSNorm followed by qk_scale_factor..
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


@ModelBase.register("OnyxForConditionalGeneration")
class OnyxVisionModel(MmprojModel):
    # fallback for rope_parameters.rope_theta
    ROPE_THETA = 10000.0

    def get_vision_config(self) -> dict[str, Any] | None:
        c = self.global_config.get("vision_config")
        if not c:
            return None
        # Onyx actually uses dynamic size, initialize with nominal size
        image_size = c["pos_emb_height"] * c["patch_size"] * c["merge_size"]
        return {**c, "image_size": image_size}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        c = self.hparams_vision  # enriched vision_config from get_vision_config()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.ONYX)
        self.gguf_writer.add_vision_attention_layernorm_eps(float(c["layer_norm_eps"]))
        self.gguf_writer.add_vision_spatial_merge_size(int(c["merge_size"]))
        self.gguf_writer.add_vision_rope_theta(float(c.get("rope_parameters", {}).get("rope_theta", self.ROPE_THETA)))

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        keep = ("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.")
        if not any(name.startswith(k) for k in keep):
            return None
        return super().filter_tensors((name, gen))

    @staticmethod
    def _unpermute_for_rope(tensor: "Tensor", n_heads: int) -> "Tensor":
        """clip.cpp uses the interleaved convention, so we invert the permutation here."""
        if tensor.ndim == 2:
            dim1, dim2 = tensor.shape
            return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
        if tensor.ndim == 1:
            (dim1,) = tensor.shape
            return tensor.view(n_heads, 2, dim1 // n_heads // 2).transpose(1, 2).reshape(dim1)
        raise ValueError(f"_unpermute_for_rope: unexpected shape {tuple(tensor.shape)}")

    # 3-layer projector MLP
    _MM_MLP_MAP = {
        "model.vision_adapter.fc1": (gguf.MODEL_TENSOR.V_MMPROJ, 0),
        "model.vision_adapter.fc2": (gguf.MODEL_TENSOR.V_MMPROJ, 1),
        "model.vision_projection":  (gguf.MODEL_TENSOR.V_MMPROJ, 2),
    }

    def modify_tensors(self, data_torch, name, bid):
        if ".attn.q_proj." in name or ".attn.k_proj." in name:
            n_heads = int(self.hparams_vision["num_attention_heads"])
            data_torch = self._unpermute_for_rope(data_torch, n_heads)
        stem, _, suffix = name.rpartition(".")
        if stem in self._MM_MLP_MAP:
            tensor_key, idx = self._MM_MLP_MAP[stem]
            yield (self.format_tensor_name(tensor_key, bid=idx, suffix="." + suffix), data_torch)
            return
        yield (self.map_tensor_name(name), data_torch)
