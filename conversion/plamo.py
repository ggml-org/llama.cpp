from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("PlamoForCausalLM")
class PlamoModel(TextModel):
    model_arch = gguf.MODEL_ARCH.PLAMO

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hparams = self.hparams

        self.gguf_writer.add_context_length(4096)  # not in config.json
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(5)  # hparams["num_key_value_heads"]) is wrong
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
        self.gguf_writer.add_file_type(self.ftype)

    def shuffle_attn_q_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(8, 5, 128, 5120)
        data_torch = torch.permute(data_torch, (1, 0, 2, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def shuffle_attn_output_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(5120, 8, 5, 128)
        data_torch = torch.permute(data_torch, (0, 2, 1, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        new_name = self.map_tensor_name(name)

        # shuffle for broadcasting of gqa in ggml_mul_mat
        if new_name.endswith("attn_q.weight"):
            data_torch = self.shuffle_attn_q_weight(data_torch)
        elif new_name.endswith("attn_output.weight"):
            data_torch = self.shuffle_attn_output_weight(data_torch)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Plamo2ForCausalLM", "PLaMo2ForCausalLM")
class Plamo2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.PLAMO2

    def set_vocab(self):
        self._set_vocab_plamo()

    def set_gguf_parameters(self):
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        # Which layers are Mamba layers
        # PLaMo 2 uses mamba_step to indicate the pattern (e.g., 2 means every other layer)
        # This logic matches modeling_plamo.py's is_mamba function
        mamba_step = hparams.get("mamba_step", 2)
        mamba_enabled = hparams.get("mamba_enabled", True)
        num_key_value_heads = []
        num_attention_heads = []

        if mamba_enabled:
            for i in range(self.block_count):
                if self.block_count <= (mamba_step // 2):
                    # use attention in last layer
                    is_mamba = (i != self.block_count - 1)
                else:
                    is_mamba = (i % mamba_step) != (mamba_step // 2)
                if is_mamba:
                    num_key_value_heads.append(0)
                    num_attention_heads.append(0)
                else:
                    num_key_value_heads.append(hparams.get("num_key_value_heads", 4))
                    num_attention_heads.append(hparams.get("num_attention_heads", 32))

        if num_key_value_heads and num_attention_heads:
            self.gguf_writer.add_head_count_kv(num_key_value_heads)
            self.gguf_writer.add_head_count(num_attention_heads)

        self.gguf_writer.add_context_length(hparams.get("max_position_embeddings", 2048))
        self.gguf_writer.add_embedding_length(hparams.get("hidden_size", 4096))
        self.gguf_writer.add_key_length(hparams.get("hidden_size_per_head", 128))
        self.gguf_writer.add_value_length(hparams.get("hidden_size_per_head", 128))
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_layer_norm_rms_eps(hparams.get("rms_norm_eps", 1e-06))
        self.gguf_writer.add_rope_freq_base(self.rope_parameters.get("rope_theta", 10000))

        # Mamba parameters
        self.gguf_writer.add_ssm_state_size(hparams.get("mamba_d_state", 64))
        self.gguf_writer.add_ssm_conv_kernel(hparams.get("mamba_d_conv", 4))
        self.gguf_writer.add_ssm_time_step_rank(hparams.get("mamba_num_heads", 64))
        intermediate_size = hparams.get("mamba_num_heads", 64) * hparams.get("hidden_size_per_head", 128)
        self.gguf_writer.add_ssm_inner_size(intermediate_size)
        self.gguf_writer.add_ssm_group_count(0)

        # MLP feed forward parameters (for attention layers)
        self.gguf_writer.add_feed_forward_length(hparams.get("intermediate_size", 13312))
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif name.endswith(".dt_norm_weight"):
            name = name.rpartition(".dt_norm_weight")[0] + ".dt_norm.weight"
        elif name.endswith(".B_norm_weight"):
            name = name.rpartition(".B_norm_weight")[0] + ".B_norm.weight"
        elif name.endswith(".C_norm_weight"):
            name = name.rpartition(".C_norm_weight")[0] + ".C_norm.weight"
        elif name.endswith(".k_weight"):
            name = name.rpartition(".k_weight")[0] + ".k.weight"
        elif name.endswith(".q_weight"):
            name = name.rpartition(".q_weight")[0] + ".q.weight"
        elif name.endswith(".conv1d.weight"):
            data_torch = torch.squeeze(data_torch)  # remove (, 1, )
            assert data_torch.ndim == 2
        elif name.endswith(".pre_mixer_norm.weight"):
            data_torch += 1.0
        elif name.endswith(".post_mixer_norm.weight"):
            data_torch += 1.0 / 5
        elif name.endswith(".pre_mlp_norm.weight"):
            data_torch += 1.0
        elif name.endswith(".post_mlp_norm.weight"):
            data_torch += 1.0 / (5**1.5)
        elif name.endswith(".norm.weight"):
            data_torch += 1.0

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Plamo3ForCausalLM", "PLaMo3ForCausalLM")
class Plamo3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.PLAMO3

    def set_vocab(self):
        self._set_vocab_plamo()

        tokenizer_config_path = self.dir_model / "tokenizer_config.json"
        tokenizer_config = {}

        if tokenizer_config_path.is_file():
            with open(tokenizer_config_path, encoding="utf-8") as f:
                tokenizer_config = json.load(f)

        chat_template = tokenizer_config.get("chat_template")
        chat_template_jinja = self.dir_model / "chat_template.jinja"

        if chat_template_jinja.is_file():
            with open(chat_template_jinja, encoding="utf-8") as f:
                chat_template = f.read()

        if chat_template:
            self.gguf_writer.add_chat_template(chat_template)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        if (sliding_window := self.find_hparam(["window_size", "sliding_window"], optional=True)) is not None:
            self.gguf_writer.add_sliding_window(sliding_window)
            self.gguf_writer.add_sliding_window_pattern(self.hparams["sliding_window_pattern"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:

        if name.endswith(".pre_mixer_norm.weight"):
            data_torch = data_torch + 1.0
        elif name.endswith(".post_mixer_norm.weight"):
            data_torch = data_torch + 1.0 / 5
        elif name.endswith(".pre_mlp_norm.weight"):
            data_torch = data_torch + 1.0
        elif name.endswith(".post_mlp_norm.weight"):
            data_torch = data_torch + 1.0 / (5**1.5)
        elif name.endswith((".mixer.q_norm.weight", ".mixer.k_norm.weight")):
            data_torch = data_torch + 1.0
        elif name.endswith(".norm.weight"):
            data_torch = data_torch + 1.0

        yield from super().modify_tensors(data_torch, name, bid)
