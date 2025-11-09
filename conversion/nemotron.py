from __future__ import annotations
from .base import (
    ModelBase, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .granite import GraniteHybridModel


@ModelBase.register("NemotronForCausalLM")
class NemotronModel(GraniteHybridModel):
    model_arch = gguf.MODEL_ARCH.NEMOTRON

    def set_vocab(self):
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_pad_token_id(0)
        self.gguf_writer.add_unk_token_id(1)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        f_norm_eps = self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon", "norm_eps"])
        self.gguf_writer.add_layer_norm_eps(f_norm_eps)
        # * Partial RoPE
        rot_pct = self.find_hparam(["partial_rotary_factor", "rope_pct", "rope_percent"])
        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_rope_dimension_count(int(rot_pct * n_embd) // n_head)
        # * RopeScaling for Nemotron
        if "rope_scaling" not in self.hparams or self.hparams["rope_scaling"] is None:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        else:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            self.gguf_writer.add_rope_scaling_factor(self.hparams["factor"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # * Adding +1 to LayerNorm's weights here to implement layernorm1p w/o changing anything on the GGML engine side
        #   model.layers.{l}.input_layernorm.weight
        #   model.layers.{l}.post_attention_layernorm.weight
        #   model.norm.weight
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("NemotronHForCausalLM")
class NemotronHModel(GraniteHybridModel):
    """Hybrid mamba2/attention model from NVIDIA"""
    model_arch = gguf.MODEL_ARCH.NEMOTRON_H

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Save the top-level head_dim for later
        self.head_dim = self.hparams.get("head_dim", self.hparams.get("attention_head_dim"))
        assert self.head_dim is not None, "Could not find the attention head dim in config"
        # Don't use expand to calculate d_inner
        self.d_inner = self.find_hparam(["num_heads"]) * self.d_model
        # Update the ssm / attn / mlp layers
        # M: Mamba2, *: Attention, -: MLP
        hybrid_override_pattern = self.hparams["hybrid_override_pattern"]
        self._ssm_layers = [i for i, val in enumerate(hybrid_override_pattern) if val == "M"]
        self._mlp_layers = [i for i, val in enumerate(hybrid_override_pattern) if val == "-"]

    def get_attn_layers(self):
        hybrid_override_pattern = self.hparams["hybrid_override_pattern"]
        assert len(hybrid_override_pattern) == self.block_count, "Mismatch between hybrid override and num_hidden_layers!"
        return [i for i, val in enumerate(hybrid_override_pattern) if val == "*"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_key_length(self.head_dim)
        self.gguf_writer.add_value_length(self.head_dim)
        # Set feed_forward_length
        # NOTE: This will trigger an override warning. This is preferrable to
        #   duplicating all the parent logic
        n_ff = self.find_hparam(["intermediate_size", "n_inner", "hidden_dim"])
        self.gguf_writer.add_feed_forward_length([
            n_ff if i in self._mlp_layers else 0 for i in range(self.block_count)
        ])

    def set_vocab(self):
        super().set_vocab()
        # The tokenizer _does_ add a BOS token (via post_processor type
        # TemplateProcessing) but does not set add_bos_token to true in the
        # config, so we need to explicitly override it here.
        self.gguf_writer.add_add_bos_token(True)
