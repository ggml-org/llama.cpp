from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("LocateAnythingForConditionalGeneration")
class LocateAnythingModel(MmprojModel):
    """NVIDIA LocateAnything-3B vision tower + connector.

    The encoder is MoonViT-SO-400M, identical to the one Kimi-K2.5 uses, so we reuse the existing
    MoonViT tensor mapping (rename the ``vision_model.`` prefix to ``vision_tower.``) and the
    interleaved->split Q/K permute. The connector is the "Eagle MLP"
    (LayerNorm(4608) -> Linear -> GELU -> Linear); it maps onto the shared ``mm_projector`` names.

    The text tower is plain Qwen2.5-3B and is converted separately by Qwen2Model, which auto-routes
    via ``text_config.architectures = ["Qwen2ForCausalLM"]`` (no extra code here).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None, "LocateAnything requires vision_config in model config"

        self.patch_size = self.hparams_vision.get("patch_size", 14)
        self.merge_kernel_size = tuple(self.hparams_vision.get("merge_kernel_size", [2, 2]))

        # vision_config has no image_size; derive one from the learned pos-emb grid for the base class.
        pos_emb_h = self.hparams_vision.get("init_pos_emb_height", 64)
        self.hparams_vision["image_size"] = pos_emb_h * self.patch_size

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LOCATEANYTHING)

        # native learned position-embedding grid (interpolated at runtime for other resolutions)
        self.gguf_writer.add_uint32("vision.pos_emb_height", self.hparams_vision.get("init_pos_emb_height", 64))
        self.gguf_writer.add_uint32("vision.pos_emb_width", self.hparams_vision.get("init_pos_emb_width", 64))

        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("layer_norm_eps", 1e-5))
        self.gguf_writer.add_vision_projector_scale_factor(self.merge_kernel_size[0])

        # The HF processor caps images at in_token_limit *patches* (pre-merge) and never upscales.
        # Translate to a pixel budget for the dynamic-size preprocessor; set the minimum to a single
        # merged token so small images are not upscaled.
        in_token_limit = self.preprocessor_config.get("in_token_limit", 25600)
        pixels_per_patch = self.patch_size ** 2
        merge_pixels = (self.patch_size * self.merge_kernel_size[0]) * (self.patch_size * self.merge_kernel_size[1])
        self.gguf_writer.add_vision_min_pixels(merge_pixels)
        self.gguf_writer.add_vision_max_pixels(in_token_limit * pixels_per_patch)

    @staticmethod
    def permute(weights: Tensor, n_head: int) -> Tensor:
        # interleaved -> split RoPE layout, matching Kimi-K2.5 (lets build_rope_2d run in split mode)
        out_dim, in_dim = weights.shape
        head_dim = out_dim // n_head
        w = weights.reshape(n_head, head_dim // 4, 2, 2, in_dim)
        w = w.permute(0, 2, 1, 3, 4)
        return w.reshape(out_dim, in_dim)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not (name.startswith("vision_model") or name.startswith("mlp1")):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        assert self.hparams_vision is not None
        n_head = self.hparams_vision.get("num_attention_heads", 16)

        # Eagle-MLP connector -> shared mm_projector names (mlp1.2 is GELU, no weights):
        #   mlp1.0 (LayerNorm) -> mm_projector.pre_norm
        #   mlp1.1 (Linear)    -> mm_projector.proj.linear_1
        #   mlp1.3 (Linear)    -> mm_projector.proj.linear_2
        if name.startswith("mlp1."):
            name = (name
                    .replace("mlp1.0.", "mm_projector.pre_norm.")
                    .replace("mlp1.1.", "mm_projector.proj.linear_1.")
                    .replace("mlp1.3.", "mm_projector.proj.linear_2."))
            yield from super().modify_tensors(data_torch, name, bid)
            return

        # vision encoder: rename to the shared MoonViT prefix
        if name.startswith("vision_model."):
            name = name.replace("vision_model.", "vision_tower.", 1)

        # keep the fused wqkv (maps to V_ENC_ATTN_QKV), but permute Q/K interleaved->split
        if "wqkv" in name:
            out_dim = data_torch.shape[0]
            qkv_dim = out_dim // 3
            head_dim = qkv_dim // n_head

            if "weight" in name:
                wq, wk, wv = data_torch[:qkv_dim, :], data_torch[qkv_dim:2 * qkv_dim, :], data_torch[2 * qkv_dim:, :]
                wq = self.permute(wq, n_head)
                wk = self.permute(wk, n_head)
                data_torch = torch.cat([wq, wk, wv], dim=0)
            elif "bias" in name:
                bq, bk, bv = data_torch[:qkv_dim], data_torch[qkv_dim:2 * qkv_dim], data_torch[2 * qkv_dim:]
                bq = bq.reshape(n_head, head_dim // 4, 2, 2).permute(0, 2, 1, 3).reshape(-1)
                bk = bk.reshape(n_head, head_dim // 4, 2, 2).permute(0, 2, 1, 3).reshape(-1)
                data_torch = torch.cat([bq, bk, bv], dim=0)

        yield from super().modify_tensors(data_torch, name, bid)
