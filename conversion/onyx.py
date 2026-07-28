from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("OnyxForConditionalGeneration")
class OnyxModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ONYX

    def norm_shift(self, name: str) -> float:
        # All four layer norms use 1, the final norm uses 0.
        return 1.0 if name.endswith("layernorm.weight") else 0.0

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_eot_token_id(200008)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_final_logit_softcapping(hparams["final_logit_softcapping"])
        self.gguf_writer.add_logit_scale(hparams["output_multiplier"])

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
