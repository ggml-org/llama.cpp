from __future__ import annotations
from .base import (
    ModelBase, gguf
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from .llama import LlamaModel


@ModelBase.register("ApertusForCausalLM")
class ApertusModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.APERTUS
    undo_permute = False
    _alpha_n = {}
    _alpha_p = {}
    _beta = {}
    _eps = {}

    def modify_tensors(self, data_torch, name, bid):
        # Handle xIELU activation parameters
        n_layers = self.hparams["num_hidden_layers"]
        if name.endswith(".act_fn.alpha_n"):
            self._alpha_n[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_n) == n_layers):
                self.gguf_writer.add_xielu_alpha_n([self._alpha_n[k] for k in sorted(self._alpha_n)])
            return []
        if name.endswith(".act_fn.alpha_p"):
            self._alpha_p[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_p) == n_layers):
                self.gguf_writer.add_xielu_alpha_p([self._alpha_p[k] for k in sorted(self._alpha_p)])
            return []
        if name.endswith(".act_fn.beta"):
            self._beta[bid] = data_torch.to("cpu").float().item()
            if (len(self._beta) == n_layers):
                self.gguf_writer.add_xielu_beta([self._beta[k] for k in sorted(self._beta)])
            return []
        if name.endswith(".act_fn.eps"):
            self._eps[bid] = data_torch.to("cpu").float().item()
            if (len(self._eps) == n_layers):
                self.gguf_writer.add_xielu_eps([self._eps[k] for k in sorted(self._eps)])
            return []
        return super().modify_tensors(data_torch, name, bid)
