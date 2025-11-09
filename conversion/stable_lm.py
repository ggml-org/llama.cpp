from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("StableLmForCausalLM", "StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM")
class StableLMModel(TextModel):
    model_arch = gguf.MODEL_ARCH.STABLELM

    def set_vocab(self):
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # StableLM 2 1.6B used to have a vocab in a similar format to Qwen's vocab
            self._set_vocab_qwen()

    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        rotary_factor = self.find_hparam(["partial_rotary_factor", "rope_pct"])
        self.gguf_writer.add_rope_dimension_count(int(rotary_factor * (hparams["hidden_size"] // hparams["num_attention_heads"])))
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(["layer_norm_eps", "norm_eps"]))
        self.gguf_writer.add_file_type(self.ftype)
    _q_norms: list[dict[str, Tensor]] | None = None
    _k_norms: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams["num_key_value_heads"]
        if name.find("q_layernorm.norms") != -1:
            assert bid is not None
            if self._q_norms is None:
                self._q_norms = [{} for _ in range(self.block_count)]
            self._q_norms[bid][name] = data_torch
            if len(self._q_norms[bid]) >= n_head:
                return self._stack_qk_norm(bid, n_head, self._q_norms[bid], "q_layernorm")
            else:
                return []
        if name.find("k_layernorm.norms") != -1:
            assert bid is not None
            if self._k_norms is None:
                self._k_norms = [{} for _ in range(self.block_count)]
            self._k_norms[bid][name] = data_torch
            if len(self._k_norms[bid]) >= n_kv_head:
                return self._stack_qk_norm(bid, n_kv_head, self._k_norms[bid], "k_layernorm")
            else:
                return []
        return [(self.map_tensor_name(name), data_torch)]

    def _stack_qk_norm(self, bid: int, n_head: int, norms: dict[str, Tensor], layer_name: str = "q_layernorm"):
        datas: list[Tensor] = []
        # extract the norms in order
        for xid in range(n_head):
            ename = f"model.layers.{bid}.self_attn.{layer_name}.norms.{xid}.weight"
            datas.append(norms[ename])
            del norms[ename]
        data_torch = torch.stack(datas, dim=0)
        merged_name = f"model.layers.{bid}.self_attn.{layer_name}.weight"
        new_name = self.map_tensor_name(merged_name)
        return [(new_name, data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._q_norms is not None or self._k_norms is not None:
            # flatten two `list[dict[str, Tensor]]` into a single `list[str]`
            norms = (
                [k for d in self._q_norms for k in d.keys()] if self._q_norms is not None else []
            ) + (
                [k for d in self._k_norms for k in d.keys()] if self._k_norms is not None else []
            )
            if len(norms) > 0:
                raise ValueError(f"Unprocessed norms: {norms}")
