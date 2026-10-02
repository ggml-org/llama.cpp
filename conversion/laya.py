from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .bert import ModernBertModel


@ModelBase.register_hparams_loader(lambda dir_model: (dir_model / "rl_agent_config.json").is_file())
def _load_laya_hparams(dir_model: Path) -> dict[str, Any]:
    """Laya checkpoints ship no root config.json; the encoder config lives in
    encoder/config.json and the RL-agent hyperparameters in rl_agent_config.json."""
    with open(dir_model / "rl_agent_config.json", encoding="utf-8") as f:
        rl_cfg = json.load(f)

    enc_cfg: dict[str, Any] = {}
    enc_cfg_path = dir_model / "encoder" / "config.json"
    if enc_cfg_path.is_file():
        with open(enc_cfg_path, encoding="utf-8") as f:
            enc_cfg = json.load(f)

    hparams = {
        **enc_cfg,
        # force the laya architecture (root config is absent)
        "architectures": ["LayaModel"],
        "model_type": "laya",
    }
    # RL-agent decision head hyperparameters
    hparams["head_layers"]   = rl_cfg.get("head_layers", 2)
    hparams["max_len"]       = rl_cfg.get("max_len", 1024)
    hparams["head_max_len"]  = rl_cfg.get("head_max_len", 192)
    hparams["n_qtype"]       = 3  # choice / score / noul
    hparams["marker_token_id"] = 4  # <mask>
    hparams["temperature"]   = rl_cfg.get("temperature", [1.0, 1.0, 1.0])
    hparams["act_classes"]   = len(rl_cfg.get("act_costs", {})) + 1
    return hparams


@ModelBase.register("LayaModel")
class LayaModel(ModernBertModel):
    model_arch = gguf.MODEL_ARCH.LAYA

    # laya checkpoints keep the tokenizer files under a tokenizer/ subdirectory
    @property
    def _tokenizer_dir(self) -> Path:
        tok_dir = self.dir_model / "tokenizer"
        return tok_dir if tok_dir.is_dir() else self.dir_model

    def set_vocab(self):
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)
        self.gguf_writer.add_add_sep_token(True)
        self.gguf_writer.add_marker_token_id(self.hparams["marker_token_id"])
        self._set_vocab_gpt2()

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        # tokenizer.json / tokenizer_config.json live in the tokenizer/ subdir
        saved = self.dir_model
        self.dir_model = self._tokenizer_dir
        try:
            return super().get_vocab_base()
        finally:
            self.dir_model = saved

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self._tokenizer_dir, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

        # the laya build_sequence always wraps with [CLS]/[SEP], so the llama.cpp
        # side must add bos/sep the same way; SpecialVocab may have turned sep off
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)
        self.gguf_writer.add_add_sep_token(True)

    def get_vocab_base_pre(self, tokenizer) -> str:
        # The multilingual laya checkpoint reuses the ModernBERT (mmBERT) BPE
        # pre-tokenizer: Metaspace with the U+2581 replacement glyph.
        return "modern-bert"

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # decision head hyperparameters
        self.gguf_writer.add_head_layers(self.hparams["head_layers"])
        self.gguf_writer.add_n_qtype(self.hparams["n_qtype"])
        self.gguf_writer.add_max_len(self.hparams["max_len"])
        self.gguf_writer.add_head_max_len(self.hparams["head_max_len"])
        self.gguf_writer.add_temperature(self.hparams["temperature"])
        self.gguf_writer.add_act_classes(self.hparams["act_classes"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # the temperature buffer is persisted as a KV array, not as a tensor
        if name == "temperature":
            return None

        # encoder tensors reuse the modern-bert naming (strip the encoder. prefix)
        if name.startswith("encoder."):
            name = name[len("encoder."):]

        # PyTorch nn.MultiheadAttention uses in_proj_weight / in_proj_bias; the
        # laya decision head maps them onto the fused qkv gguf tensor.
        if name.endswith("self_attn.in_proj_weight"):
            name = name[:-len("in_proj_weight")] + "in_proj.weight"
        elif name.endswith("self_attn.in_proj_bias"):
            name = name[:-len("in_proj_bias")] + "in_proj.bias"

        return name, gen
