from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, torch


@ModelBase.register("Spark2_5ForCausalLM")
@ModelBase.example("XHToken/Spark-X2.5-1.7B")
class Spark2_5Model(TextModel):
    model_arch = gguf.MODEL_ARCH.SPARK2_5

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0].startswith("draft_model."):
            return None
        return super().filter_tensors(item)

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


class Spark2_5DSparkModel(Spark2_5Model):
    model_arch = gguf.MODEL_ARCH.DFLASH

    def __init__(self, dir_model, *args, **kwargs):
        hparams = deepcopy(kwargs.pop("hparams", None) or ModelBase.load_hparams(dir_model, False))
        n_layers = hparams["num_draft_layers"]
        hparams["num_hidden_layers"] = n_layers
        hparams["layer_types"] = hparams.get("draft_layer_types", hparams["layer_types"][:n_layers])
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

        if self.hparams.get("markov_head_type", "vanilla") != "vanilla":
            raise ValueError("Spark DSpark requires a vanilla Markov head")
        if self.hparams.get("draft_vocab_size", self.hparams["vocab_size"]) != self.hparams["vocab_size"]:
            raise ValueError("Spark DSpark currently requires the full target vocabulary")
        if not self.hparams.get("confidence_head_with_markov", True):
            raise ValueError("Spark DSpark requires confidence_head_with_markov")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith("draft_model."):
            return None
        return TextModel.filter_tensors(("model." + name.removeprefix("draft_model."), gen))

    def set_vocab(self) -> None:
        super().set_vocab()
        self.gguf_writer.add_mask_token_id(self.hparams["mask_token_id"])

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        self.gguf_writer.add_block_size(self.hparams["dspark_block_size"])
        # Target layer outputs are extracted as the next layer's inputs.
        self.gguf_writer.add_target_layers([i + 1 for i in self.hparams["target_layer_ids"]])
        self.gguf_writer.add_sample_from_anchor(self.hparams.get("sample_from_anchor", True))
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_hidden_act(self.hparams["hidden_act"])
        # GGUF counts the anchor in the sliding window.
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"] + 1)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.fc_norm."):
            if name == "model.fc_norm.0.weight":
                norms = [self.model_tensors[f"model.fc_norm.{i}.weight"]().float()
                         for i in range(len(self.hparams["target_layer_ids"]))]
                yield self.format_tensor_name(gguf.MODEL_TENSOR.FC_NORM), torch.stack(norms)
            return

        if name.endswith(".self_attn.q_k_v_proj.weight"):
            head_dim = self.hparams["head_dim"]
            n_q = head_dim * self.hparams["num_attention_heads"]
            n_kv = head_dim * self.hparams["num_key_value_heads"]
            for tensor, value in zip(
                (gguf.MODEL_TENSOR.ATTN_Q, gguf.MODEL_TENSOR.ATTN_K, gguf.MODEL_TENSOR.ATTN_V),
                data_torch.split((n_q, n_kv, n_kv), dim=0),
                strict=True,
            ):
                yield self.format_tensor_name(tensor, bid), value
            return

        yield from super().modify_tensors(data_torch, name, bid)
