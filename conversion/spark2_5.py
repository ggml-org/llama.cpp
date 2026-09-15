from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("Spark2_5ForCausalLM")
@ModelBase.example("XHToken/Spark-X2.5-1.7B")
class Spark2_5Model(TextModel):
    model_arch = gguf.MODEL_ARCH.SPARK2_5

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()

        hparams = self.hparams
        layer_types = hparams["layer_types"]
        if len(layer_types) != self.block_count:
            raise ValueError(
                f"Spark2_5 layer_types length {len(layer_types)} != num_hidden_layers {self.block_count}"
            )
        if any(layer_type not in ("sliding_attention", "full_attention") for layer_type in layer_types):
            raise ValueError(f"Spark2_5 has unsupported layer_types: {layer_types}")
        if hparams.get("gate_attn_act_mode") != "sigmoid" or hparams.get("headwise_attn_output_gate") is not True:
            raise ValueError("Spark2_5 conversion requires head-wise sigmoid attention gates")
        if hparams.get("hidden_act") != "gelu":
            raise ValueError(f"Spark2_5 conversion requires GELU, got {hparams.get('hidden_act')!r}")

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(
            [layer_type == "sliding_attention" for layer_type in layer_types]
        )

        head_dim = hparams["head_dim"]
        full_rope = self.rope_parameters["full_attention"]
        swa_rope = self.rope_parameters["sliding_attention"]
        self.gguf_writer.add_rope_dimension_count(
            int(head_dim * float(full_rope["partial_rotary_factor"]))
        )
        self.gguf_writer.add_rope_dimension_count_swa(
            int(head_dim * float(swa_rope["partial_rotary_factor"]))
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".self_attn.q_k_v_proj.weight"):
            if bid is None:
                raise ValueError(f"Spark2_5 fused QKV tensor has no block id: {name}")
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV, bid), data_torch
            return

        if name.endswith(".self_attn.g_proj.weight"):
            if bid is None:
                raise ValueError(f"Spark2_5 attention gate tensor has no block id: {name}")
            expected = self.hparams["num_attention_heads"]
            if data_torch.shape[0] != expected:
                raise ValueError(
                    f"Spark2_5 layer {bid} attention gate width {data_torch.shape[0]} != head count {expected}"
                )

        yield from super().modify_tensors(data_torch, name, bid)
