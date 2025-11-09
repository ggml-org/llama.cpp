from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("MPTForCausalLM")
class MPTModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MPT

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
        except Exception:
            # Fallback for SEA-LION model
            self._set_vocab_sentencepiece()
            self.gguf_writer.add_add_bos_token(False)
            self.gguf_writer.add_pad_token_id(3)
            self.gguf_writer.add_eos_token_id(1)
            self.gguf_writer.add_unk_token_id(0)

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layers"]
        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["d_model"])
        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        if kv_n_heads := self.hparams["attn_config"].get("kv_n_heads"):
            self.gguf_writer.add_head_count_kv(kv_n_heads)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        if self.hparams["attn_config"]["clip_qkv"] is not None:
            self.gguf_writer.add_clamp_kqv(self.hparams["attn_config"]["clip_qkv"])
        if self.hparams["attn_config"]["alibi"]:
            self.gguf_writer.add_max_alibi_bias(self.hparams["attn_config"]["alibi_bias_max"])
        else:
            self.gguf_writer.add_max_alibi_bias(0.0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if "scales" in name:
            new_name = self.map_tensor_name(name, try_suffixes=(".weight", ".bias", ".scales"))
            new_name = new_name.replace("scales", "act.scales")
        else:
            new_name = self.map_tensor_name(name, try_suffixes=(".weight", ".bias"))
        return [(new_name, data_torch)]
