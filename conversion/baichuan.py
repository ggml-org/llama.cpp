from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("BaichuanForCausalLM", "BaiChuanForCausalLM")
class BaichuanModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BAICHUAN

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        if bid is not None and name == f"model.layers.{bid}.self_attn.W_pack.weight":
            logger.info(f"Unpacking and permuting layer {bid}")
            yield from [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid),
                    self._reverse_hf_permute_part(data_torch, 0, head_count, head_count)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid),
                    self._reverse_hf_permute_part(data_torch, 1, head_count, head_count_kv)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid),
                    self._reverse_hf_part(data_torch, 2)),
            ]
        else:
            yield from self.modify_tensors(data_torch, self.map_tensor_name(name), bid)

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def _reverse_hf_permute_part(
        self, weights: Tensor, n_part: int, n_head: int, n_head_kv: int | None = None,
    ) -> Tensor:
        r = weights.shape[0] // 3
        return self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv)

    def _reverse_hf_part(self, weights: Tensor, n_part: int) -> Tensor:
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]
