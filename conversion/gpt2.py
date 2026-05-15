from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GPT2LMHeadModel")
class GPT2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # we don't need these
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            yield from super().modify_tensors(data_torch, name, bid)
            return

        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")):
            data_torch = data_torch.transpose(1, 0)

        new_name = self.map_tensor_name(name)

        yield from super().modify_tensors(data_torch, new_name, bid)


@ModelBase.register("RuGPT3XLForCausalLM")
class RuGPT3XLModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GPT2

    _qkv_parts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Fuse separate Q, K, V projections into a single QKV tensor
        if ".self_attn.q_proj." in name or ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
            suffix = "weight" if name.endswith(".weight") else "bias"
            part = "q" if ".q_proj." in name else ("k" if ".k_proj." in name else "v")
            key = f"{part}.{suffix}"

            assert bid is not None
            if self._qkv_parts is None:
                self._qkv_parts = [{} for _ in range(self.block_count)]
            self._qkv_parts[bid][key] = data_torch

            q_key, k_key, v_key = f"q.{suffix}", f"k.{suffix}", f"v.{suffix}"
            if all(k in self._qkv_parts[bid] for k in [q_key, k_key, v_key]):
                q = self._qkv_parts[bid].pop(q_key)
                k = self._qkv_parts[bid].pop(k_key)
                v = self._qkv_parts[bid].pop(v_key)
                data_torch = torch.cat([q, k, v], dim=0)
                name = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV, bid, f".{suffix}")
                logger.debug(f"Fused Q/K/V {suffix} for layer {bid} -> {name}")
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._qkv_parts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            parts = [f"({i}){k}" for i, d in enumerate(self._qkv_parts) for k in d.keys()]
            if len(parts) > 0:
                raise ValueError(f"Unprocessed Q/K/V parts: {parts}")
