from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("TalkieForCausalLM")
class TalkieModel(TextModel):
    model_arch = gguf.MODEL_ARCH.TALKIE

    _absorbed_gain_tensors = {
        "lm_head": "lm_head_gain.w_g",
        "attn.attn_resid.weight": "attn_gain.a_g",
        "mlp.mlp_resid.weight": "mlp_gain.a_g",
    }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # Talkie used F.rms_norm without an explicit eps
        self.gguf_writer.add_layer_norm_rms_eps(torch.finfo(torch.float32).eps)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        prefix = f"model.blocks.{bid}." if bid is not None else ""
        suffix = name.removeprefix(prefix)

        if suffix in self._absorbed_gain_tensors.values():
            return

        if suffix in self._absorbed_gain_tensors:
            gain = self.model_tensors[prefix + self._absorbed_gain_tensors[suffix]]().float()
            data_torch *= gain
        elif suffix in ("attn.attn_query.weight", "attn.attn_key.weight"):
            # absorb inverse rope
            head_dim = self.hparams["head_dim"]
            by_head = data_torch.view(-1, head_dim, data_torch.shape[1])
            by_head[:, head_dim // 2:] *= -1
        elif suffix == "attn.head_gain.head_g":
            # scalar head gain becomes q norm scale
            head_dim = self.hparams["head_dim"]
            data_torch = data_torch.unsqueeze(-1).expand(-1, head_dim).contiguous()

        if not name.endswith(".weight"):
            name += ".weight"

        yield from super().modify_tensors(data_torch, name, bid)
