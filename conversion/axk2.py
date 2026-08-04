from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger


@ModelBase.register("AXK2ForCausalLM")
class AXK2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.AXK2

    merge_expert = True

    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        # A.X-K2 does not prepend BOS: the tokenizer has no post-processor and the chat template
        # never references bos_token
        self._set_vocab_gpt2()
        self.gguf_writer.add_add_bos_token(False)

    def set_gguf_parameters(self):
        # note: axk2 using MLA converts into MQA (ie: GQA with 1 group)
        self.hparams["num_key_value_heads"] = 1
        self.hparams["rms_norm_eps"] = self.hparams.get("rms_norm_eps", 1e-6)

        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])

        # note: axk2 using MLA converts into MQA with larger heads, then decompresses to MHA
        kv_lora_rank = hparams["kv_lora_rank"]
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_key_length(kv_lora_rank + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(kv_lora_rank)
        self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        if hparams.get("norm_topk_prob"):
            self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if (rope_mscale_all := self.rope_parameters.get("mscale_all_dim")) is not None:
            # ref https://github.com/ggml-org/llama.cpp/pull/17945
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * rope_mscale_all)

        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        # DSA (sparse attention) indexer parameters
        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])

        # bottleneck rank of the low-rank gate wrapping the norms
        self.gguf_writer.add_gated_norm_rank(hparams["gated_norm_rank"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the fused q + output-gate projection is block-diagonal: rows are per-head [q, gate] pairs,
        # the input is cat([post-norm, pre-norm]) of the query LoRA bottleneck, and the q rows only
        # read the post-norm half while the gate rows only read the pre-norm half. Split it into two
        # separate projections (lossless).
        if name.endswith("self_attn.q_b_proj.weight"):
            n_head       = self.hparams["num_attention_heads"]
            q_lora_rank  = self.hparams["q_lora_rank"]
            v_head_dim   = self.hparams["v_head_dim"]
            qk_head_dim  = self.hparams["qk_nope_head_dim"] + self.hparams["qk_rope_head_dim"]

            assert data_torch.shape == (n_head * (qk_head_dim + v_head_dim), 2 * q_lora_rank)

            qg = LazyTorchTensor.to_eager(data_torch).view(n_head, qk_head_dim + v_head_dim, 2 * q_lora_rank)
            q, gate = torch.split(qg, [qk_head_dim, v_head_dim], dim=1)

            # the two blocks dropped below must be zero, or the split would silently lose weights
            if q[:, :, q_lora_rank:].any() or gate[:, :, :q_lora_rank].any():
                raise ValueError(f"{name} is not block-diagonal, cannot split into q and gate")

            q = q[:, :, :q_lora_rank].reshape(n_head * qk_head_dim, q_lora_rank)
            gate = gate[:, :, q_lora_rank:].reshape(n_head * v_head_dim, q_lora_rank)

            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q_B, bid), q
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid), gate
            return

        # skip lm_head.weight if tie_word_embeddings is True
        if self.hparams.get("tie_word_embeddings", False):
            if name == "lm_head.weight" or name == "model.lm_head.weight":
                logger.info("Skipping tied output layer 'lm_head.weight' (will use token_embd.weight)")
                return

        # merge the per-expert tensors into a single 3d tensor
        if self.merge_expert and name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
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
            else:
                return

        # MLA with the absorption optimization needs kv_b_proj split and k_b_proj transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
