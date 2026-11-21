from __future__ import annotations

import torch

from .base import ModelBase, TextModel, gguf


@ModelBase.register("Nemotron3_6AcademiaForCausalLM")
class Nemotron3_6AcademiaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.NEMOTRON3_6_ACADEMIA

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        if self.find_hparam(["hidden_act"]) != "relu2":
            raise ValueError("Nemotron3_6Academia requires relu2 activation")
        if self.fuse_qkv:
            raise ValueError("Nemotron3_6Academia requires separate Q/K/V projections")

        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_input_layer_count(self.find_hparam(["n_input_layers"]))
        w.add_recurrent_layer_count(self.find_hparam(["n_recurrent_layers"]))
        w.add_output_layer_count(self.find_hparam(["n_output_layers"]))
        w.add_num_loops(self.find_hparam(["n_recurrent_loops"]))
        patterns = {"linear_attention": "M", "full_attention": "*", "mlp": "-"}
        w.add_layer_pattern("".join(patterns[t] for t in self.find_hparam(["layer_types"])))
        w.add_recurrent_kv_cache_mode(self.hparams.get("recurrent_kv_cache_mode", "per_loop"))
        w.add_loop_conditioning_hidden_size(self.find_hparam(["loop_conditioning_hidden_size"]))
        w.add_loop_conditioning_max_loops(self.find_hparam(["loop_conditioning_max_loops"]))
        w.add_loop_adapter_rank(self.find_hparam(["loop_adapter_rank"]))
        w.add_loop_input_mixing(self.find_hparam(["loop_input_mixing"]))
        w.add_ssm_conv_kernel(self.find_hparam(["linear_conv_kernel_dim"]))
        w.add_ssm_inner_size(self.find_hparam(["linear_num_value_heads"]) * self.find_hparam(["linear_value_head_dim"]))
        w.add_ssm_state_size(self.find_hparam(["linear_key_head_dim"]))
        w.add_ssm_time_step_rank(self.find_hparam(["linear_num_value_heads"]))
        w.add_ssm_group_count(self.find_hparam(["linear_num_key_heads"]))
        w.add_ssm_householder_count(self.find_hparam(["linear_num_householder"]))

    def modify_tensors(self, data_torch, name, bid):
        mapped = self.map_tensor_name(name)
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch.float())
        elif name.endswith(".dt_bias"):
            mapped += ".bias"
        elif not mapped.endswith((".weight", ".bias")):
            mapped += ".weight"
        if name.endswith(".conv1d.weight"):
            data_torch = data_torch.squeeze(1)
        yield mapped, data_torch
