from __future__ import annotations

from typing import Iterable, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf, logger


@ModelBase.register("MiniMaxText01ForCausalLM")
@ModelBase.register("MiniMaxM1ForCausalLM")
@ModelBase.example("MiniMaxAI/MiniMax-Text-01", "MiniMaxAI/MiniMax-M1-40k")
class MiniMaxText01Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINIMAX01

    def _get_suppress_tokens(self) -> Sequence[int] | None:
        import json
        from transformers import AutoTokenizer
        from .base import LazyTorchTensor

        # check added tokens embeddings in embeddings tensor for zero-valued embeddings
        # they get in the way of the token sampling process and must be suppressed

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokenizer_vocab_size = tokenizer.vocab_size

        with open(self.dir_model / "model.safetensors.index.json", "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        embeddings_tensor_name = "model.embed_tokens.weight"
        embeddings_shard_name = weight_map[embeddings_tensor_name]
        with gguf.utility.SafetensorsLocal(self.dir_model / embeddings_shard_name) as model_shard:
            embeddings_data = model_shard[embeddings_tensor_name]

        embeddings_weights_dtype = LazyTorchTensor._dtype_str_map[embeddings_data.dtype]
        embeddings_weights = torch.from_numpy(embeddings_data.mmap_bytes()).view(embeddings_weights_dtype).reshape(embeddings_data.shape)
        embeddings_vocab_size = embeddings_weights.shape[0]

        embeddings_added_tokens = embeddings_weights[tokenizer_vocab_size:embeddings_vocab_size]
        embeddings_zero_rows = torch.all(embeddings_added_tokens == 0, dim=1)
        tokens_zero_embeddings_ids = (torch.nonzero(embeddings_zero_rows, as_tuple=False).flatten() + tokenizer_vocab_size).tolist()

        return tokens_zero_embeddings_ids

    def set_vocab(self) -> None:
        from pathlib import Path

        self._set_vocab_gpt2()

        for tmpl_file in [
            self.dir_model / "chat_template.jinja",
            Path(__file__).parent.parent / "models" / "templates" / "MiniMax-M1.jinja"
        ]:
            if tmpl_file.is_file():
                self.gguf_writer.add_chat_template(tmpl_file.read_text(encoding="utf-8"))
                logger.info(f"Chat template overridden with {tmpl_file}.")
                break

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        suppress_tokens = self._get_suppress_tokens()
        if suppress_tokens:
            logger.info(f"Suppressing tokens with zero embeddings {suppress_tokens}")
            self.gguf_writer.add_suppress_tokens(suppress_tokens)

        layernorm_full_attention_alpha = self.hparams["layernorm_full_attention_alpha"]
        layernorm_full_attention_beta = self.hparams["layernorm_full_attention_beta"]
        layernorm_linear_attention_alpha = self.hparams["layernorm_linear_attention_alpha"]
        layernorm_linear_attention_beta = self.hparams["layernorm_linear_attention_beta"]
        layernorm_mlp_alpha = self.hparams["layernorm_mlp_alpha"]
        layernorm_mlp_beta = self.hparams["layernorm_mlp_beta"]
        assert layernorm_full_attention_alpha == layernorm_linear_attention_alpha == layernorm_mlp_alpha
        assert layernorm_full_attention_beta == layernorm_linear_attention_beta == layernorm_mlp_beta == 1.0
        # we do not store the layernorm betas as they are all 1.0
        # layernorm alphas are stored as single residual_scale hparam
        self.gguf_writer.add_residual_scale(layernorm_full_attention_alpha)

        self.gguf_writer.add_rope_dimension_count(self.hparams["rotary_dim"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    yield from super().modify_tensors(data_torch, new_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MiniMaxM2ForCausalLM")
@ModelBase.example("MiniMaxAI/MiniMax-M2")
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
@ModelBase.example("MiniMaxAI/MiniMax-M3")
class MiniMaxM3Model(MiniMaxM2Model):
    model_arch = gguf.MODEL_ARCH.MINIMAXM3

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".indexer." in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

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
@ModelBase.example("MiniMaxAI/MiniMax-M3")
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
        assert self.hparams_vision is not None
        n_head = self.hparams_vision["num_attention_heads"]
        d_head = t.shape[0] // n_head
        axis_dim = 2 * ((2 * (d_head // 2) // 3) // 2)
        ah = axis_dim // 2
        half = 3 * ah
        perm = []
        perm += list(range(0, ah))
        perm += list(range(half, half + ah))
        perm += list(range(ah, 2 * ah))
        perm += list(range(half + ah, half + 2 * ah))
        perm += list(range(2 * ah, 3 * ah))
        perm += list(range(half + 2 * ah, half + 3 * ah))
        perm += list(range(2 * half, d_head))

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
