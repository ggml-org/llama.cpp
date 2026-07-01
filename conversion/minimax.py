from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf


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
        if "block_sparse_moe.experts." in name:
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
class MiniMaxM3Model(MiniMaxM2Model):
    model_arch = gguf.MODEL_ARCH.MINIMAXM3
    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_expert_shared_count(self.find_hparam(["n_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(self.find_hparam(["routed_scaling_factor"]))
        self.gguf_writer.add_expert_weights_norm(True)

        sac = self.find_hparam(["sparse_attention_config"])
        self.gguf_writer.add_indexer_head_count(sac["sparse_num_index_heads"])
        self.gguf_writer.add_indexer_key_length(sac["sparse_index_dim"])
        self.gguf_writer.add_indexer_top_k(sac["sparse_topk_blocks"])
        self.gguf_writer.add_indexer_block_size(sac["sparse_block_size"])
        self.gguf_writer.add_indexer_local_blocks(sac["sparse_local_block"])

        moe_layer_freq = self.find_hparam(["moe_layer_freq"])
        n_dense = 0
        for v in moe_layer_freq:
            if v == 0:
                n_dense += 1
            else:
                break
        self.gguf_writer.add_leading_dense_block_count(n_dense)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        # Gemma-style (1 + w) RMSNorm: bake the +1 in so llama.cpp can use plain RMSNorm
        if name.endswith("norm.weight"):
            data_torch = data_torch + 1.0

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3VLForConditionalGeneration")
class MiniMaxM3VisionModel(MmprojModel):
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

        # Conv3d patch embed -> Conv2d slices
        if name == "vision_tower.vision_model.embeddings.patch_embedding.weight":
            if data_torch.ndim != 5:
                raise ValueError(f"unexpected patch_embedding rank {data_torch.ndim} for {name}")
            kt = data_torch.shape[2]
            base = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH]
            for t in range(kt):
                suffix = ".weight" if t == 0 else f".weight.{t}"
                yield (base + suffix, data_torch[:, :, t, ...])
            return

        # Permute ViT q/k. HF [Ta Ha Wa | Tb Hb Wb | pad] reorder to [Ta Tb | Ha Hb | Wa Wb | pad]. 
        for new_name, tensor in super().modify_tensors(data_torch, name, bid):
            if ".attn_q." in new_name or ".attn_k." in new_name:
                tensor = self._permute_vit_qk(tensor, new_name)
            yield new_name, tensor
            
    def _permute_vit_qk(self, t: "Tensor", new_name: str) -> "Tensor":
        n_head = self.hparams_vision["num_attention_heads"]
        d_head = t.shape[0] // n_head
        axis_dim = 2 * ((2 * (d_head // 2) // 3) // 2)
        ah   = axis_dim // 2
        half = 3 * ah
        perm = (list(range(0, ah))            + list(range(half,        half + ah)) +
                list(range(ah, 2 * ah))       + list(range(half + ah,   half + 2*ah)) +
                list(range(2 * ah, 3 * ah))   + list(range(half + 2*ah, half + 3*ah)) +
                list(range(2 * half, d_head)))
                
        assert axis_dim % 2 == 0
        assert 3 * axis_dim <= d_head
        assert len(perm) == d_head
        assert sorted(perm) == list(range(d_head)), "perm is not a bijection of d_head"
        assert t.shape[0] == n_head * d_head, f"{new_name}: {t.shape[0]} != {n_head}*{d_head}"
        assert d_head == 80
        
        idx = torch.tensor(perm, dtype=torch.long)
        if t.ndim == 2:
            return t.reshape(n_head, d_head, t.shape[1])[:, idx, :].reshape(t.shape)
        return t.reshape(n_head, d_head)[:, idx].reshape(t.shape)
