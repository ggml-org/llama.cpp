from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


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

        # refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])

        self.gguf_writer.add_feed_forward_length(ff_dim)
        self.gguf_writer.add_block_count(self.block_count)
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

        if bid is not None:
            if name == f"transformer.h.{bid}.attn.kv.weight":
                yield from super().modify_tensors(data_torch[:n_head_kv * head_dim], self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), bid)
                yield from super().modify_tensors(data_torch[n_head_kv * head_dim:], self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), bid)
                return
            if name == f"transformer.h.{bid}.attn.q.weight":
                yield from super().modify_tensors(data_torch, self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), bid)
                return
            if name == f"transformer.h.{bid}.mlp.gate_up_proj.weight":
                yield from super().modify_tensors(data_torch[:ff_dim], self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), bid)
                yield from super().modify_tensors(data_torch[ff_dim:], self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), bid)
                return

        yield from super().modify_tensors(data_torch, name, bid)
