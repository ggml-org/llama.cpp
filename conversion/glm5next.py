from __future__ import annotations

import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf
from .qwen3vl import Glm4VVisionModel


@ModelBase.register("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
class Glm5NextModel(TextModel):
    """GLM-5.3-Flash text tower: hybrid KDA + DSA attention, nope-only MLA, mHC
    hyper-connections, and a NextN block with its own DSA attention and indexer.
    """

    model_arch = gguf.MODEL_ARCH.GLM5NEXT
    supports_mtp_export = True

    _experts: list[dict[str, Tensor]] | None = None
    _main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nextn_layers = 0 if self.no_mtp else (self.hparams.get("num_nextn_predict_layers", 0) or 0)
        self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        # two spellings of the same partition; disagreement means an odd config
        from_types = {il for il, t in enumerate(self.hparams["layer_types"]) if t == "deepseek_sparse_attention"}
        from_list = set(self.hparams["linear_attn_config"]["full_attn_layers"])
        if from_types != from_list:
            raise ValueError(f"layer_types picks DSA layers {sorted(from_types)} but full_attn_layers says {sorted(from_list)}")
        self._full_attn_layers = from_types

        dense_lead = self.hparams["first_k_dense_replace"]
        expected = ["dense"] * dense_lead + ["sparse"] * (self.hparams["num_hidden_layers"] - dense_lead)
        if self.hparams["mlp_layer_types"] != expected:
            raise ValueError("mlp_layer_types does not match first_k_dense_replace")

    def index_tensors(self, remote_hf_model_id: str | None = None):
        hp = self.hparams.get("text_config", self.hparams)
        type(self)._main_layers = hp["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def set_vocab(self):
        self._set_vocab_glm()

    def is_full_attention(self, bid: int) -> bool:
        return bid >= self.hparams["num_hidden_layers"] or bid in self._full_attn_layers

    def set_gguf_parameters(self):
        hp = self.hparams
        linear_cfg = hp["linear_attn_config"]

        # checked here, not in the loader: head_count_kv becomes the per-layer recurrence marker
        if hp["num_attention_heads"] != hp.get("num_key_value_heads"):
            raise ValueError("glm5next expects MHA-shaped head counts before MLA absorption")
        if hp["qk_rope_head_dim"] != 0 or not hp.get("mla_use_nope"):
            raise ValueError("glm5next is nope-only: qk_rope_head_dim must be 0 and mla_use_nope true")
        if linear_cfg["num_heads"] != hp["num_attention_heads"]:
            raise ValueError("glm5next KDA and full attention are expected to share a head count")
        if not hp.get("mhc"):
            raise ValueError("glm5next without mHC is not supported")
        if hp["index_topk"] % hp["index_kpool"] != 0:
            raise ValueError("glm5next index_topk must be a whole number of kpool pools")

        # no GGUF key carries these and the graph cannot express them off, so refuse to write
        if not hp.get("index_kpool_compress"):
            raise ValueError("glm5next without the indexer kpool compressor is not supported")
        if not hp.get("index_kpool_always_select_tail"):
            raise ValueError("glm5next without always-select-tail kpool is not supported")
        if not hp.get("indexer_rope_interleave"):
            raise ValueError("glm5next without interleaved indexer rope is not supported")
        if set(hp["indexer_types"]) != {"full"}:
            raise ValueError("glm5next expects every indexer to be full")

        hp.pop("head_dim", None)
        hp.pop("num_key_value_heads", None)

        super().set_gguf_parameters()

        self.gguf_writer.add_vocab_size(hp["vocab_size"])

        # n_head_kv == 0 marks a KDA (recurrent) layer, as in kimi-k3 and bailingmoe3
        self.gguf_writer.add_head_count_kv(
            [1 if self.is_full_attention(il) else 0 for il in range(self.block_count)])

        kv_lora_rank = hp["kv_lora_rank"]
        qk_rope_head_dim = hp["qk_rope_head_dim"]
        self.gguf_writer.add_q_lora_rank(hp["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        self.gguf_writer.add_value_length(kv_lora_rank)
        self.gguf_writer.add_key_length_mla(hp["qk_nope_head_dim"] + qk_rope_head_dim)
        self.gguf_writer.add_value_length_mla(hp["v_head_dim"])

        # indexer k_norm is a LayerNorm with bias at a fixed 1e-6, not the model's RMS eps
        self.gguf_writer.add_layer_norm_eps(1e-6)

        self.gguf_writer.add_ssm_conv_kernel(linear_cfg["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(linear_cfg["head_dim"])
        # not a clamp: scales the sigmoid decay gate. required, else the softplus branch is chosen
        self.gguf_writer.add_kda_gate_lower_bound(linear_cfg["gate_lower_bound"])

        self.gguf_writer.add_indexer_head_count(hp["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["index_topk"])
        self.gguf_writer.add_indexer_kpool(hp["index_kpool"])

        self.gguf_writer.add_hyper_connection_count(hp["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hp["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hp["hc_eps"])

        n_ff_exp = hp["moe_intermediate_size"]
        self.gguf_writer.add_expert_feed_forward_length(n_ff_exp)
        self.gguf_writer.add_expert_shared_feed_forward_length(n_ff_exp * hp["n_shared_experts"])
        self.gguf_writer.add_expert_shared_count(hp["n_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(hp["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hp["norm_topk_prob"])

        # no dense-FFN clamp key exists, so the expert arrays cover the leading dense layers too
        swiglu_limit = float(hp["swiglu_limit"])
        self.gguf_writer.add_swiglu_clamp_exp([swiglu_limit] * self.block_count)
        self.gguf_writer.add_swiglu_clamp_shexp([swiglu_limit] * self.block_count)

        if not self.no_mtp and (nextn_layers := hp.get("num_nextn_predict_layers", 0)):
            self.gguf_writer.add_nextn_predict_layers(nextn_layers)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        assert cls._main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._main_layers

        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # KDA conv1d: HF [d_inner, 1, d_conv] -> ggml ne [d_conv, 1, d_inner, 1]
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            d_inner = data_torch.shape[0]
            d_conv = data_torch.shape[-1]
            data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        # ssm_a holds -exp(A_log), the kimi-k3 convention (bailingmoe3 stores +exp(A_log));
        # the wrong sign turns decay into an unchecked growing state
        if name.endswith(".A_log"):
            decay = LazyTorchTensor.to_eager(torch.exp(data_torch.float()))
            if not bool(torch.isfinite(decay).all() and (decay > 0).all()):
                raise ValueError(f"{name}: exp(A_log) must be finite and positive")
            data_torch = -decay

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # bare in the checkpoint, but the GGUF names carry .weight
        if re.search(r"\.hc_(attn|ffn)_(fn|base|scale)$", name) or name.endswith(
                (".index_kpool_compress_gate", ".index_kpool_compress_ape")):
            name += ".weight"

        if ".mlp.experts." in name:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) < n_experts * 3:
                return

            for weight_name in ("down_proj", "gate_proj", "up_proj"):
                tensors = []
                for expert_id in range(n_experts):
                    expert_name = f"model.layers.{bid}.mlp.experts.{expert_id}.{weight_name}.weight"
                    tensors.append(self._experts[bid].pop(expert_name))
                merged_name = f"model.layers.{bid}.mlp.experts.{weight_name}.weight"
                yield from super().modify_tensors(torch.stack(tensors, dim=0), merged_name, bid)
            return

        if name.endswith(".kv_b_proj.weight"):
            assert bid is not None
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            yield from super().modify_tensors(k_b.transpose(1, 2), self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid), bid)
            yield from super().modify_tensors(v_b, self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # pinned as POS_EMBD is in base.py
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_APE, bid):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [name for layer in self._experts for name in layer]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Glm5NextForConditionalGeneration")
# [TAG_HF_EXAMPLE_MISSING]
class Glm5NextVisionModel(Glm4VVisionModel):
    """The vision tower is the GLM-OCR ViT under a `model.visual.` prefix.

    Every tensor already maps through the GLM-4V entries. The one structural
    difference is a clamp on the SwiGLU gate and up projections, applied in the
    per-block MLP and again in the merger, so it gets its own projector type
    rather than a flag on glm4v.
    """

    clip_projector_type = gguf.VisionProjectorType.GLM5NEXT

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        # no GLM4V-family mmproj carries this key and clip.cpp falls back to a hardcoded 2
        self.gguf_writer.add_vision_spatial_merge_size(int(self.hparams_vision["spatial_merge_size"]))

        self.gguf_writer.add_vision_swiglu_limit(float(self.hparams_vision["swiglu_limit"]))
