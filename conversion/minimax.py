from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf, logger


@ModelBase.register("MiniMaxM2ForCausalLM")
class MiniMaxM2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINIMAXM2
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # merge expert weights
        if 'experts' in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return

            for w_name in expert_weights:
                datas: list[Tensor] = []

                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                new_name = self.map_tensor_name(merged_name)
                yield from super().modify_tensors(data_torch, new_name, bid)

            del self._experts_cache[bid]
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")
class MiniMaxM3Model(TextModel):
    # Text-only MiniMax-M3 (minimax_m3_vl): MiniMax-M2 style GQA (per-head QK-norm, partial
    # rotary) + DeepSeek-V3 shared experts / leading dense, swigluoai activation. Sparse
    # attention, vision tower and MTP heads are dropped.
    model_arch = gguf.MODEL_ARCH.MINIMAXM3
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def set_gguf_parameters(self):
        # dense layers use dense_intermediate_size, experts use intermediate_size. Base
        # writes feed_forward_length from intermediate_size, so swap in the dense width
        # and emit the expert width separately.
        expert_ff = self.find_hparam(["intermediate_size"])
        self.hparams["intermediate_size"] = self.find_hparam(["dense_intermediate_size"])
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_feed_forward_length(expert_ff)
        self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))
        self.gguf_writer.add_expert_shared_count(self.find_hparam(["n_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(self.find_hparam(["routed_scaling_factor"]))
        self.gguf_writer.add_expert_weights_norm(True)

        # leading dense layers = count of leading zeros in moe_layer_freq
        moe_layer_freq = self.find_hparam(["moe_layer_freq"])
        n_dense = 0
        for v in moe_layer_freq:
            if v == 0:
                n_dense += 1
            else:
                break
        self.gguf_writer.add_leading_dense_block_count(n_dense)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # text-only: drop vision, projector, patch-merge and sparse-attention index tensors
        if name.startswith(("vision_tower", "multi_modal_projector", "patch_merge_mlp")) or ".index_" in name:
            return

        # strip VL wrapper prefix to match tensor_mapping names
        if name.startswith("language_model."):
            name = name[len("language_model."):]

        # Gemma-style (1 + w) RMSNorm: bake the +1 in so llama.cpp can use plain RMSNorm
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1.0

        # merge routed experts (w1=gate, w2=down, w3=up), like MiniMax-M2. shared_experts.*
        # does not match here and maps straight through to the *_shexp tensors.
        if "block_sparse_moe.experts." in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]

            # not enough expert weights to merge yet
            if len(expert_cache) < n_experts * len(expert_weights):
                return

            for w_name in expert_weights:
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]

                data_torch = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                yield from super().modify_tensors(data_torch, merged_name, bid)

            del self._experts_cache[bid]
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3VLForConditionalGeneration")
class MiniMaxM3VisionModel(MmprojModel):
    # Vision tower for MiniMax-M3 (minimax_m3_vl). CLIP-style ViT (separate biased
    # q/k/v/out, LayerNorm, gelu fc1/fc2, a pre_layrnorm and no post_layernorm / class
    # token / abs-pos table) with a Conv3d patch embed and 3D (T/H/W) RoPE. The text
    # tower is handled by MiniMaxM3Model; here we keep only the vision-side tensors.
    #
    # Projector is two on-disk modules:
    #   multi_modal_projector.linear_{1,2}  -> per-patch MLP   (V_MMPROJ -> mm.1, mm.2)
    #   patch_merge_mlp.linear_{1,2}        -> 2x2 merge MLP    (V_MM_MERGE_FC1/FC2)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        # keep only the vision-side tensors; text / mtp / sparse-index are dropped
        if not name.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")):
            return None
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MINIMAXM3)
        self.gguf_writer.add_vision_use_gelu(True)

        # the ViT carries its own LayerNorm eps (text tower uses a different one)
        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.hparams_vision.get("layer_norm_eps", 1e-5)
        )

        comp = self.hparams_vision.get("img_token_compression_config", {})
        merge_size = comp.get("spatial_merge_size", 2)
        self.gguf_writer.add_vision_spatial_merge_size(int(merge_size))

    def modify_tensors(self, data_torch, name, bid):
        assert self.hparams_vision is not None

        # Conv3d patch embed -> split into temporal_patch_size Conv2d slices, summed in C++.
        # MiniMax-M3 has no patch-embed bias.
        if name == "vision_tower.vision_model.embeddings.patch_embedding.weight":
            if data_torch.ndim != 5:
                raise ValueError(f"unexpected patch_embedding rank {data_torch.ndim} for {name}")
            kt = data_torch.shape[2]  # temporal_patch_size
            base = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH]
            for t in range(kt):
                suffix = ".weight" if t == 0 else f".weight.{t}"
                yield (base + suffix, data_torch[:, :, t, ...])
            return

        # everything else resolves through the precomputed MMPROJ name map:
        #   vision_tower.vision_model.*        -> v.* (auto, shared CLIP mapping)
        #   multi_modal_projector.linear_{bid} -> mm.{bid}
        #   patch_merge_mlp.linear_{1,2}       -> mm.merge.fc{1,2}
        yield from super().modify_tensors(data_torch, name, bid)
