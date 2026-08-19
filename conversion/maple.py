from __future__ import annotations

from typing import Iterable, TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf


@ModelBase.register("MapleForCausalLM")
class MapleModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MAPLE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        assert hparams["hidden_act"] == "silu"
        assert hparams.get("num_shared_experts", 0) == 0
        assert hparams.get("norm_topk_prob", True)
        assert hparams.get("nope_on_global_attention", False)

        head_dim = hparams.get("head_dim", hparams["hidden_size"] // hparams["num_attention_heads"])
        partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", hparams.get("partial_rotary_factor", 1.0))

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(int(head_dim * partial_rotary_factor))
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([layer_type == "sliding_attention" for layer_type in hparams["layer_types"]])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_GATE_INP, bid):
            return gguf.GGMLQuantizationType.F32

        if any(self.match_model_tensor_name(new_name, key, bid) for key in (
            gguf.MODEL_TENSOR.TOKEN_EMBD,
            gguf.MODEL_TENSOR.OUTPUT,
        )):
            return gguf.GGMLQuantizationType.F16

        return super().tensor_force_quant(name, new_name, bid, n_dims)

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def _stack_experts(tensors: list[Tensor]) -> Tensor:
        shape = (len(tensors), *tensors[0].shape)
        dtype = tensors[0].dtype
        meta = LazyTorchTensor.meta_with_dtype_and_shape(dtype, shape)

        def stack() -> Tensor:
            result = torch.empty(shape, dtype=dtype)
            for expert_id, tensor in enumerate(tensors):
                result[expert_id].copy_(LazyTorchTensor.to_eager(tensor))
            tensors.clear()
            return result

        return cast(torch.Tensor, LazyTorchTensor(meta=meta, args=(), func=stack))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "mlp.experts" in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for weight_name in ("down_proj", "gate_proj", "up_proj"):
                    tensors = []

                    for expert_id in range(n_experts):
                        expert_name = f"model.layers.{bid}.mlp.experts.{expert_id}.{weight_name}.weight"
                        tensors.append(self._experts[bid].pop(expert_name))

                    merged_name = f"model.layers.{bid}.mlp.experts.{weight_name}.weight"
                    yield from super().modify_tensors(self._stack_experts(tensors), merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [name for layer in self._experts for name in layer]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")
