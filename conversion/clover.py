from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

from .base import ModelBase, gguf
from .qwen import _LinearAttentionVReorderBase

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("CloverMoeModelLM")
@ModelBase.example("Diffuspace/Clover-35B-A3B-Preview")
class CloverMoeModel(_LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.CLOVER_MOE
    supports_mtp_export = False

    def __init__(self, *args, **kwargs):
        self.no_mtp = True
        super().__init__(*args, **kwargs)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_shared_count(self.hparams.get("num_shared_experts", 1))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.endswith((".conv_right_weight", ".bd_softmax_logit_scale", ".bd_softmax_history_bias", ".bd_softmax_history_c")):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".A_log") or (name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight")):
            data_torch = data_torch.float()

        yield from super().modify_tensors(data_torch, name, bid)
