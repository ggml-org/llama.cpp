from __future__ import annotations

import re

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("HrmTextForCausalLM")
@ModelBase.example("danish-foundation-models/DFM-Mimir")
class HrmTextModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HRM_TEXT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # training-style configs store the per-stack count in num_hidden_layers,
        # transformers-style configs rewrite num_hidden_layers to the expanded
        # cache-slot count and keep the real count in num_layers_per_stack
        self.layers_per_stack = self.hparams.get("num_layers_per_stack") or self.hparams["num_hidden_layers"]
        self.h_cycles = self.hparams["H_cycles"]
        self.l_cycles = self.hparams["L_cycles"]

        # the runtime expands one H and one L stack over h_cycles * (l_cycles + 1)
        # cache slots, so the GGUF block count is the slot count while the file
        # only holds 2 * layers_per_stack physical blocks
        self.block_count = self.layers_per_stack * self.h_cycles * (self.l_cycles + 1)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, 2 * self.layers_per_stack)

        # "default" rope_type carries no scaling; drop it to avoid a misleading
        # unknown-rope-type warning in the generic parameter writer
        if self.rope_parameters.get("rope_type") == "default":
            self.rope_parameters = {k: v for k, v in self.rope_parameters.items() if k != "rope_type"}

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        head_dim = self.hparams.get("head_dim") or self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(head_dim)
        self.gguf_writer.add_embedding_scale(self.hparams["embedding_scale"])
        self.gguf_writer.add_uint32(gguf.Keys.LLM.HRM_LAYERS_PER_STACK.format(arch=self.gguf_writer.arch), self.layers_per_stack)
        self.gguf_writer.add_uint32(gguf.Keys.LLM.HRM_H_CYCLES.format(arch=self.gguf_writer.arch), self.h_cycles)
        self.gguf_writer.add_uint32(gguf.Keys.LLM.HRM_L_CYCLES.format(arch=self.gguf_writer.arch), self.l_cycles)
        self.gguf_writer.add_bool(gguf.Keys.LLM.HRM_PREFIX_LM.format(arch=self.gguf_writer.arch), bool(self.hparams.get("prefix_lm", False)))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embed_tokens.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch
            return
        if name == "lm_head.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch
            return
        if name == "model.z_L_init":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.HRM_Z_L_INIT, suffix=""), data_torch
            return

        match = re.fullmatch(r"model\.([LH])_module\.layers\.(\d+)\.(.+)", name)
        if match is None:
            raise ValueError(f"can not map tensor: {name}")

        stack, layer_s, tensor_name = match.groups()
        # the L stack occupies blocks [0, layers_per_stack), the H stack follows it
        layer_idx = int(layer_s) + (self.layers_per_stack if stack == "H" else 0)

        if tensor_name == "attn.gqkv_proj.weight":
            gate, q, k, v = data_torch.chunk(4, dim=0)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, layer_idx), gate.contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, layer_idx), q.contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, layer_idx), k.contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, layer_idx), v.contiguous()
        elif tensor_name == "mlp.gate_up_proj.weight":
            gate, up = data_torch.chunk(2, dim=0)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, layer_idx), gate.contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, layer_idx), up.contiguous()
        elif tensor_name in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                             "self_attn.gate_proj.weight", "self_attn.o_proj.weight",
                             "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            tensor = {
                "self_attn.q_proj.weight": gguf.MODEL_TENSOR.ATTN_Q,
                "self_attn.k_proj.weight": gguf.MODEL_TENSOR.ATTN_K,
                "self_attn.v_proj.weight": gguf.MODEL_TENSOR.ATTN_V,
                "self_attn.gate_proj.weight": gguf.MODEL_TENSOR.ATTN_GATE,
                "self_attn.o_proj.weight": gguf.MODEL_TENSOR.ATTN_OUT,
                "mlp.gate_proj.weight": gguf.MODEL_TENSOR.FFN_GATE,
                "mlp.up_proj.weight": gguf.MODEL_TENSOR.FFN_UP,
                "mlp.down_proj.weight": gguf.MODEL_TENSOR.FFN_DOWN,
            }[tensor_name]
            yield self.format_tensor_name(tensor, layer_idx), data_torch
        elif tensor_name == "attn.o_proj.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, layer_idx), data_torch
        else:
            raise ValueError(f"can not map tensor: {name}")
