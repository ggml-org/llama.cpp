from __future__ import annotations

import json

from .base import ModelBase, gguf
from .llama import LlamaModel


@ModelBase.register("ApertusForCausalLM")
@ModelBase.example("swiss-ai/Apertus-8B-Instruct-2509")
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
            return
        if name.endswith(".act_fn.alpha_p"):
            self._alpha_p[bid] = data_torch.to("cpu").float().item()
            if (len(self._alpha_p) == n_layers):
                self.gguf_writer.add_xielu_alpha_p([self._alpha_p[k] for k in sorted(self._alpha_p)])
            return
        if name.endswith(".act_fn.beta"):
            self._beta[bid] = data_torch.to("cpu").float().item()
            if (len(self._beta) == n_layers):
                self.gguf_writer.add_xielu_beta([self._beta[k] for k in sorted(self._beta)])
            return
        if name.endswith(".act_fn.eps"):
            self._eps[bid] = data_torch.to("cpu").float().item()
            if (len(self._eps) == n_layers):
                self.gguf_writer.add_xielu_eps([self._eps[k] for k in sorted(self._eps)])
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register(
    "Apertus1p5ForConditionalGeneration",
    "Apertus1p5TextForCausalLM",
)
@ModelBase.example("swiss-ai/Apertus-v1.5-8B", "swiss-ai/Apertus-v1.5-70B")
class Apertus1p5Model(ApertusModel):
    model_arch = gguf.MODEL_ARCH.APERTUS

    def set_vocab(self):
        super().set_vocab()

        with open(self.dir_model / "tokenizer.json", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        added_tokens = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"]}

        self.gguf_writer.add_eot_token_id(added_tokens["<|assistant_end|>"])
        self.gguf_writer.add_eom_token_id(added_tokens["<|tools_suffix|>"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        n_vocab = self.hparams["vocab_size"]
        n_vocab_out = self.hparams.get("output_vocab_size")
        if n_vocab_out is None:
            n_vocab_out = n_vocab
        if type(n_vocab_out) is not int or not 0 < n_vocab_out <= n_vocab:
            raise ValueError(f"Invalid output vocabulary size {n_vocab_out} for input vocabulary size {n_vocab}")
        self.gguf_writer.add_vocab_size_out(n_vocab_out)
