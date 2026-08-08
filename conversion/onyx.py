from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


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

        # SWA + NoPE: [SW, SW, SW, Full], NoPE used on Full layers. References:
        # https://huggingface.co/someorgtoo-hf/onyx-hf-converted/blob/main/config.json#L19
        # https://huggingface.co/someorgtoo-hf/onyx-hf-converted/blob/main/config.json#L73
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(4)

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
