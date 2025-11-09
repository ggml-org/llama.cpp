from __future__ import annotations
import math
from .base import (
    ModelBase, TextModel, gguf, torch
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("ExaoneForCausalLM")
class ExaoneModel(TextModel):
    model_arch = gguf.MODEL_ARCH.EXAONE

    def set_gguf_parameters(self):
        hparams = self.hparams
        assert (hparams["activation_function"] == "silu")
        max_position_embeddings = hparams["max_position_embeddings"]
        embed_dim = hparams["hidden_size"]
        num_heads = hparams["num_attention_heads"]
        num_kv_heads = hparams.get("num_key_value_heads", num_heads)
        layer_norm_eps = hparams["layer_norm_epsilon"]
        intermediate_size = hparams["intermediate_size"] if "intermediate_size" in hparams else 4 * embed_dim
        num_layers = hparams["num_layers"]
        # ignore for now as EXAONE-3.0-7.8B-Instruct attentino_dropout is 0.0
        # attention_dropout_rate = hparams["attention_dropout"]
        # ignore for now as EXAONE-3.0-7.8B-Instruct embed_dropout is 0.0
        # embed_dropout_rate = hparams["embed_dropout"]
        self.gguf_writer.add_embedding_length(embed_dim)
        self.gguf_writer.add_head_count(num_heads)
        self.gguf_writer.add_head_count_kv(num_kv_heads)
        self.gguf_writer.add_context_length(max_position_embeddings)
        self.gguf_writer.add_layer_norm_rms_eps(layer_norm_eps)
        self.gguf_writer.add_feed_forward_length(intermediate_size)
        self.gguf_writer.add_block_count(num_layers)
        self.gguf_writer.add_file_type(self.ftype)
        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
        rotary_factor = self.find_hparam(["partial_rotary_factor", "rope_pct"], optional=True)
        rotary_factor = rotary_factor if rotary_factor is not None else 1.0
        self.gguf_writer.add_rope_dimension_count(int(rotary_factor * (hparams["hidden_size"] // hparams["num_attention_heads"])))
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "linear" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_scaling := self.find_hparam(["rope_scaling"], optional=True):
            if rope_scaling.get("rope_type", '').lower() == "llama3":
                base = self.hparams.get("rope_theta", 10000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                factor = rope_scaling.get("factor", 8.0)
                low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)
                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                assert low_freq_wavelen != high_freq_wavelen
                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))


@ModelBase.register("Exaone4ForCausalLM")
class Exaone4Model(TextModel):
    model_arch = gguf.MODEL_ARCH.EXAONE4

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if hparams.get("sliding_window") is not None:
            self.gguf_writer.add_sliding_window(hparams["sliding_window"])
            if "layer_types" in hparams:
                self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hparams["layer_types"]])
            elif "sliding_window_pattern" in hparams:
                sliding_window_pattern = []
                if isinstance(hparams["sliding_window_pattern"], str):  # e.g. LLLG
                    for i in range(hparams["num_hidden_layers"]):
                        sliding_window_pattern.append(hparams["sliding_window_pattern"][i % len(hparams["sliding_window_pattern"])] == "L")
                if isinstance(hparams["sliding_window_pattern"], int):  # e.g. 4
                    for i in range(hparams["num_hidden_layers"]):
                        sliding_window_pattern.append((i + 1) % hparams["sliding_window_pattern"] != 0)
                if len(sliding_window_pattern) == hparams["num_hidden_layers"]:
                    self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "linear" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_scaling := self.find_hparam(["rope_scaling"], optional=True):
            if rope_scaling.get("rope_type", '').lower() == "llama3":
                base = self.hparams.get("rope_theta", 10_000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                factor = rope_scaling.get("factor", 16.0)
                low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get("original_max_position_embeddings", 8192)
                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))
