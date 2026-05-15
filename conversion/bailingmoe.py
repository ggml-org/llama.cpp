from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BailingMoeForCausalLM")
class BailingMoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        n_embd = self.hparams["hidden_size"]
        if (head_dim := self.hparams.get("head_dim")) is None:
            head_dim = n_embd // n_head

        output_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)

        if name.endswith("attention.dense.weight"):
            yield from super().modify_tensors(data_torch, self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), bid)
            return
        elif name.endswith("query_key_value.weight"):
            q, k, v = data_torch.split([n_head * head_dim, n_kv_head * head_dim, n_kv_head * head_dim], dim=-2)

            yield from super().modify_tensors(BailingMoeModel.permute(q, n_head, n_head), self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), bid)
            yield from super().modify_tensors(BailingMoeModel.permute(k, n_head, n_kv_head), self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), bid)
            yield from super().modify_tensors(v,self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), bid)
            return
        elif name.find("mlp.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    yield from super().modify_tensors(data_torch, new_name, bid)

            return

        new_name = self.map_tensor_name(name)

        if new_name == output_name and self.hparams.get("norm_head"):
            data_torch = data_torch.float()
            data_torch /= torch.norm(data_torch, p=2, dim=0, keepdim=True) + 1e-7

        yield from super().modify_tensors(data_torch, new_name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("BailingMoeV2ForCausalLM")
class BailingMoeV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if nextn_layers := self.hparams.get("num_nextn_predict_layers", 0):
            self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5)))
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams.get("moe_shared_expert_intermediate_size", hparams["moe_intermediate_size"] * hparams["num_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        if (nextn_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(nextn_layers)

    _experts: list[dict[str, Tensor]] | None = None

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "mlp.experts" in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("SarvamMoEForCausalLM", "modeling_sarvam_moe.SarvamMoEForCausalLM")
class SarvamMoEModel(BailingMoeV2Model):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE2
    # Sarvam-MoE shares the BailingMoeV2 architecture; only differences:
    #  - full rotary (no partial_rotary_factor)
    #  - expert bias is zero-mean normalized at load time

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        # Override the partial-rotary value written by BailingMoeV2 with the full rotary dim
        self.gguf_writer.add_rope_dimension_count(rope_dim)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.endswith(".expert_bias"):
            # Sarvam normalizes expert bias to zero mean
            inner = gen

            def gen():
                t = inner()
                return t - t.mean()
        return super().filter_tensors((name, gen))
