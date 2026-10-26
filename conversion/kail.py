from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .llama import LlamaModel


@ModelBase.register("KailAForCausalLM")
@ModelBase.example("Kaniskten/Kail_F2")
class KailAModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.KAIL
    undo_permute = True

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_vocab_base_pre(self, tokenizer) -> str:
        # KAIL tokenizer shares the BPE pre-tokenizer fingerprint of its
        # tokenizer lineage; super() validates the fingerprint, we relabel it
        res = super().get_vocab_base_pre(tokenizer)
        if res != "minicpm5":
            raise ValueError(f"unexpected pre-tokenizer fingerprint: {res!r}")
        return "kail"

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # optional metadata for the controller/router inner dimensions;
        # reference implementation: inner_dim = max(hidden_size // reduction, 32)
        for key, default in (("controller_reduction", 4), ("router_reduction", 8)):
            value = self.hparams.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"KAIL: invalid {key}: {value!r}")
            self.gguf_writer.add_uint32(f"kail.{key}", value)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # bare parameter without .weight/.bias suffix; map to blk.N.attn_temp.weight
        if name.endswith("attn_temperature"):
            name += ".weight"
        yield from super().modify_tensors(data_torch, name, bid)
