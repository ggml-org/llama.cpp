from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("DbrxForCausalLM")
class DbrxModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DBRX

    def set_gguf_parameters(self):
        ffn_config = self.hparams["ffn_config"]
        attn_config = self.hparams["attn_config"]
        self.gguf_writer.add_block_count(self.hparams["n_layers"])
        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_feed_forward_length(ffn_config["ffn_hidden_size"])
        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        self.gguf_writer.add_head_count_kv(attn_config["kv_n_heads"])
        self.gguf_writer.add_rope_freq_base(attn_config["rope_theta"])
        self.gguf_writer.add_clamp_kqv(attn_config["clip_qkv"])
        self.gguf_writer.add_expert_count(ffn_config["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(ffn_config["moe_top_k"])
        self.gguf_writer.add_layer_norm_eps(1e-5)
        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        n_expert = self.hparams["ffn_config"]["moe_num_experts"]
        n_ff = self.hparams["ffn_config"]["ffn_hidden_size"]
        n_embd = self.hparams["d_model"]
        # Specific behavior for experts tensors: suffix .weight, view as 3D and transpose
        # original implementation expects (n_expert, n_ff, n_embd) for all experts weights
        # But llama.cpp moe graph works differently
        # AND the dimensions in ggml are typically in the reverse order of the pytorch dimensions
        # so (n_expert, n_ff, n_embd) in pytorch is {n_embd, n_ff, n_expert} in ggml_tensor
        exp_tensor_names = {"ffn.experts.mlp.w1": None,       # LLM_TENSOR_FFN_GATE_EXPS ggml_tensor->ne{n_embd, n_ff,   n_expert}
                            "ffn.experts.mlp.w2": (0, 2, 1),  # LLM_TENSOR_FFN_DOWN_EXPS ggml_tensor->ne{n_ff,   n_embd, n_expert}
                            "ffn.experts.mlp.v1": None}       # LLM_TENSOR_FFN_UP_EXPS   ggml_tensor->ne{n_embd, n_ff,   n_expert}
        experts = False
        for exp_tensor_name in exp_tensor_names.keys():
            if name.find(exp_tensor_name) != -1 and name.find(".weight") == -1:
                experts = True
                data_torch = data_torch.view(n_expert, n_ff, n_embd)
                if (permute_tensor := exp_tensor_names[exp_tensor_name]) is not None:
                    data_torch = data_torch.permute(*permute_tensor)
                break
        # map tensor names
        # In MoE models the ffn tensors are typically most of the model weights,
        # and need to be quantizable. Quantize expects tensor names to be suffixed by .weight.
        # Every other model has the weight names ending in .weight,
        # let's assume that is the convention which is not the case for dbrx:
        # https://huggingface.co/databricks/dbrx-instruct/blob/main/model.safetensors.index.json#L15
        new_name = self.map_tensor_name(name if not experts else name + ".weight", try_suffixes=(".weight",))
        return [(new_name, data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid  # unused
        return n_dims > 1
