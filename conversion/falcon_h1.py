from __future__ import annotations
from .base import (
    ModelBase, gguf
)
from .llama import LlamaModel
from .mamba import Mamba2Model
from typing import TYPE_CHECKING, Any, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("FalconH1ForCausalLM")
class FalconH1Model(Mamba2Model):
    model_arch = gguf.MODEL_ARCH.FALCON_H1

    def __init__(self, *args, **kwargs):
        # Set the hparam prefixes for Falcon Mamba2
        self.hparam_prefixes = ["mamba"]
        # Initialize the base Mamba2Model
        super().__init__(*args, **kwargs)
        # Use Llama conversion for attention
        self._transformer_model_class = LlamaModel
        # n_group and d_inner are used during reshape_tensors for mamba2
        self.n_group = self.find_hparam(["n_groups"])
        self.d_inner = self.find_hparam(["mamba_d_ssm"])
        self.d_head = self.find_hparam(["d_head"])
        # Initialize any Falcon Mamba2 specific attributes
        self.has_attention = True  # Falcon Mamba2 has attention components
        # Load Falcon-H1 multipliers from hyperparameters
        self.attention_in_multiplier = self.find_hparam(["attention_in_multiplier"], optional=True)
        self.attention_out_multiplier = self.find_hparam(["attention_out_multiplier"], optional=True)
        self.ssm_in_multiplier = self.find_hparam(["ssm_in_multiplier"], optional=True)
        self.ssm_out_multiplier = self.find_hparam(["ssm_out_multiplier"], optional=True)
        self.mlp_multipliers = self.find_hparam(["mlp_multipliers"], optional=True)
        self.ssm_multipliers = self.find_hparam(["ssm_multipliers"], optional=True)
        self.intermediate_size = self.find_hparam(["intermediate_size"])
        self.key_multiplier = self.find_hparam(["key_multiplier"], optional=True)

    def find_hparam(self, keys: Iterable[str], *args, **kwargs) -> Any:
        prefixed = []
        for pfx in self.hparam_prefixes:
            prefixed.extend(
                "_".join([pfx, k])
                for k in keys
            )
        keys = list(keys) + prefixed
        return super().find_hparam(keys, *args, **kwargs)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        tensors = list(super().modify_tensors(data_torch, name, bid))
        tensor = tensors[0][1]
        if "down_proj" in name:
            tensor = tensor  * self.mlp_multipliers[1]
        elif "gate_proj" in name:
            tensor = tensor * self.mlp_multipliers[0]
        elif "k_proj" in name:
            tensor = tensor * self.key_multiplier * self.attention_in_multiplier
        elif "q_proj" in name:
            tensor = tensor * self.attention_in_multiplier
        elif "v_proj" in name:
            tensor = tensor * self.attention_in_multiplier
        elif "o_proj" in name:
            tensor = tensor * self.attention_out_multiplier
        elif "out_proj" in name:
            tensor = tensor * self.ssm_out_multiplier
        elif "in_proj" in name:
            tensor = tensor * self.ssm_in_multiplier
            zxbcdt_multipliers = self.hparams["ssm_multipliers"]
            intermediate_size = self.hparams["mamba_d_ssm"]
            groups_time_state_size = self.hparams["mamba_n_groups"] * self.hparams["mamba_d_state"]
            tensor[:intermediate_size, :] *= zxbcdt_multipliers[0]
            tensor[intermediate_size:2 * intermediate_size, :] *= zxbcdt_multipliers[1]
            tensor[2 * intermediate_size:2 * intermediate_size + groups_time_state_size, :] *= zxbcdt_multipliers[2]
            tensor[2 * intermediate_size + groups_time_state_size:2 * intermediate_size + 2 * groups_time_state_size, :] *= zxbcdt_multipliers[3]
            tensor[2 * intermediate_size + 2 * groups_time_state_size:, :] *= zxbcdt_multipliers[4]
        elif "lm_head" in name:
            tensor = tensor * self.hparams["lm_head_multiplier"]
        elif "embed_tokens" in name:
            tensor = tensor * self.hparams["embedding_multiplier"]
        elif "mamba.norm" in name:
            tensor = tensor.reshape(self.n_group, self.d_inner // self.n_group)
        tensors = [(tensors[0][0], tensor)]
        return tensors

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        ## General Params ##
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        # Override some Mamba2 defaults
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.hparams.get("max_position_embeddings", 0))
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        ## Attention params ##
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"]) # Override value 0 from Mamba2
        self.gguf_writer.add_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_key_length(self.hparams["head_dim"])
        self.gguf_writer.add_value_length(self.hparams["head_dim"])
        ## Validation ##
        assert self.hparams.get("hidden_act") in [None, "silu"], "Only SILU activation supported"
        assert self.d_inner % self.d_head == 0, f"SSM inner size {self.d_inner} not a multiple of head dim {self.d_head}"
        # Add any other Falcon Mamba2 specific configuration
        self.gguf_writer.add_rope_freq_base(self.find_hparam(["rope_theta"]))
