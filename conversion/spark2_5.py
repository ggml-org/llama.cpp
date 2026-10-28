from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
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


@ModelBase.register("SparkX25DesparkDraftModel")
class Spark2_5DSparkModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DFLASH

    def set_vocab(self) -> None:
        if self.target_model_dir is None:
            raise ValueError("Spark-X2.5 DSpark requires --target-model-dir")
        draft_dir = self.dir_model
        self.dir_model = Path(self.target_model_dir)
        try:
            super().set_vocab()
        finally:
            self.dir_model = draft_dir
        self.gguf_writer.add_mask_token_id(self.hparams["mask_token_id"])

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.hparams
        if hparams["num_hidden_layers"] <= 0 or hparams["markov_head_type"] != "vanilla":
            raise ValueError("Spark-X2.5 DSpark requires a positive layer count and a vanilla Markov head")
        if hparams.get("num_draft_layers", self.block_count) != self.block_count:
            raise ValueError("Spark-X2.5 DSpark draft layer counts do not match")
        if (hparams["hidden_act"] != "gelu" or not hparams["headwise_attn_output_gate"]
                or hparams["gate_attn_act_mode"] != "sigmoid"):
            raise ValueError("Spark-X2.5 DSpark requires GELU and head-wise attention gates")
        if not hparams["fc_norm"]:
            raise ValueError("Spark-X2.5 DSpark requires per-feature fc_norm")
        if not hparams["enable_confidence_head"] or not hparams["confidence_head_with_markov"]:
            raise ValueError("Spark-X2.5 DSpark requires the Markov-conditioned confidence head")
        if hparams["draft_vocab_size"] != hparams["vocab_size"]:
            raise ValueError("Spark-X2.5 DSpark currently requires the full target vocabulary")
        if len(hparams["target_layer_ids"]) != 5:
            raise ValueError("Spark-X2.5 DSpark requires five target feature layers")
        if hparams.get("sample_from_anchor", True) is not True:
            raise ValueError("Spark-X2.5 DSpark requires sample_from_anchor=true")
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_block_size(hparams.get("dspark_block_size", 8))
        self.gguf_writer.add_sample_from_anchor(True)
        self.gguf_writer.add_target_layers([i + 1 for i in hparams["target_layer_ids"]])
        attention_mode = os.getenv("SPARK_DSPARK_ATTENTION")
        if attention_mode is None:
            layer_types = hparams.get("layer_types", ["sliding_attention"] * self.block_count)
        elif attention_mode in {"full_attention", "sliding_attention"}:
            layer_types = [attention_mode] * self.block_count
        else:
            raise ValueError("SPARK_DSPARK_ATTENTION must be full_attention or sliding_attention")
        if len(layer_types) != self.block_count or len(set(layer_types)) != 1:
            raise ValueError("Spark-X2.5 DSpark currently requires one attention type for all draft layers")
        if layer_types[0] == "sliding_attention":
            self.gguf_writer.add_sliding_window(hparams["sliding_window"])
            self.gguf_writer.add_sliding_window_pattern([True] * self.block_count)
            self.gguf_writer.add_rope_dimension_count_swa(
                int(hparams["head_dim"] * hparams["rope_parameters"]["sliding_attention"]["partial_rotary_factor"])
            )
        elif layer_types[0] != "full_attention":
            raise ValueError(f"Unsupported Spark-X2.5 DSpark attention type: {layer_types[0]!r}")
        self.gguf_writer.add_rope_dimension_count(
            int(hparams["head_dim"] * hparams["rope_parameters"]["full_attention"]["partial_rotary_factor"])
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        name = name.removeprefix("draft_model.")
        roots = {
            "fc.weight": "fc.weight",
            "hidden_norm.weight": "enc.output_norm.weight",
            "norm.weight": "output_norm.weight",
            "confidence_head.proj.weight": "conf_proj.weight",
            "confidence_head.proj.bias": "conf_proj.bias",
            "markov_head.markov_w1.weight": "markov_w1.weight",
            "markov_head.markov_w2.weight": "markov_w2.weight",
        }
        if name == "embed_tokens.weight":
            return
        if name in roots:
            yield roots[name], data_torch
            return
        if name.startswith("fc_norm.") and name.endswith(".weight"):
            if bid is None or bid >= len(self.hparams["target_layer_ids"]):
                raise ValueError(f"Invalid feature norm tensor {name!r}")
            yield f"fc_norm.{bid}.weight", data_torch
            return
        if bid is None or not name.startswith(f"layers.{bid}."):
            raise ValueError(f"Unexpected Spark-X2.5 DSpark tensor {name!r}")
        suffix = name.removeprefix(f"layers.{bid}.")
        layer_map = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_k_v_proj.weight": "attn_qkv.weight",
            "self_attn.g_proj.weight": "attn_gate.weight",
            "self_attn.out_proj.weight": "attn_output.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        if suffix not in layer_map:
            raise ValueError(f"Unexpected Spark-X2.5 DSpark tensor {name!r}")
        yield f"blk.{bid}.{layer_map[suffix]}", data_torch
