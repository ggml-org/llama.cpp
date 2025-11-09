from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("GPTRefactForCausalLM")
class RefactModel(TextModel):
    model_arch = gguf.MODEL_ARCH.REFACT

    def set_vocab(self):
        super().set_vocab()
        # TODO: how to determine special FIM tokens automatically?
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'eot'])
        special_vocab._set_special_token("prefix", 1)
        special_vocab._set_special_token("suffix", 3)
        special_vocab._set_special_token("middle", 2)
        special_vocab.chat_template = None  # do not add it twice
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        block_count = self.hparams["n_layer"]
        # refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(ff_dim)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_head = self.hparams["n_head"]
        n_head_kv = 1
        head_dim = self.hparams["n_embd"] // n_head
        tensors: list[tuple[str, Tensor]] = []
        if bid is not None:
            if name == f"transformer.h.{bid}.attn.kv.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), data_torch[:n_head_kv * head_dim]))
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), data_torch[n_head_kv * head_dim:]))
            elif name == f"transformer.h.{bid}.attn.q.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), data_torch))
            elif name == f"transformer.h.{bid}.mlp.gate_up_proj.weight":
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), data_torch[:ff_dim]))
                tensors.append((self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), data_torch[ff_dim:]))
        if len(tensors) == 0:
            tensors.append((self.map_tensor_name(name), data_torch))
        return tensors
