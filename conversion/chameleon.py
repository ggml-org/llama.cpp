from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf
)
from .llama import LlamaModel
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("ChameleonForConditionalGeneration", "ChameleonForCausalLM")
class ChameleonModel(TextModel):
    model_arch = gguf.MODEL_ARCH.CHAMELEON

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_swin_norm(self.hparams.get("swin_norm", False))

    def set_vocab(self):
        self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ignore image tokenizer for now
        # TODO: remove this once image support is implemented for Chameleon
        if name.startswith("model.vqmodel"):
            return []
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        hidden_dim = self.hparams.get("hidden_size")
        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
        if name.endswith(("q_norm.weight", "q_norm.bias")):
            data_torch = ChameleonModel._reverse_hf_permute(data_torch, n_head, hidden_dim)
        if name.endswith(("k_norm.weight", "k_norm.bias")):
            data_torch = ChameleonModel._reverse_hf_permute(data_torch, n_kv_head, hidden_dim)
        return [(self.map_tensor_name(name), data_torch)]

    # see: https://github.com/huggingface/transformers/blob/72fb02c47dbbe1999ae105319f24631cad6e2e00/src/transformers/models/chameleon/convert_chameleon_weights_to_hf.py#L176-L203
    @staticmethod
    def _reverse_hf_permute(data_torch, n_heads, hidden_dim):
        head_dim = hidden_dim // n_heads
        data_torch = data_torch[0].view(2, head_dim // 2).t().reshape(1, -1)
        data_torch = data_torch.repeat_interleave(n_heads, 0)
        return data_torch
