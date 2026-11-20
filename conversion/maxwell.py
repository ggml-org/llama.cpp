from __future__ import annotations

import re

from pathlib import Path

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("MaxwellMoeForCausalLM")
@ModelBase.example("maxwell-model/Maxwell-V1-Flash")
class MaxwellMoeModel(TextModel):
    """Maxwell-V1.0-Flash.

    MoE backbone (256 routed / 12 active + 2 shared experts) with manifold
    constrained hyper-connections (4 residual streams, Sinkhorn-Knopp) and
    block sparse GQA with a learned indexer (prefix + local + top-k remote
    128 token blocks).
    """

    model_arch = gguf.MODEL_ARCH.MAXWELL_MOE

    _experts: list[dict[str, Tensor]] | None = None
    _alphas: list[dict[str, Tensor]] | None = None
    _mtp_proj: dict[str, Tensor] | None = None


    def get_vocab_base_pre(self, tokenizer) -> str:
        # tokenizer.json splits newline, Han and SEA script runs before the o200k rule; llama.cpp has a
        # dedicated pre type for that sequence (LLAMA_VOCAB_PRE_TYPE_MAXWELL_MOE)
        return "maxwell-moe"

    def set_vocab(self):
        self._set_vocab_gpt2()
        # the reference tokenizer prepends BOS (<s>) to every prompt
        self.gguf_writer.add_add_bos_token(True)
        # turns end on <|maxwell_end|> (generation_config eos is [2, 200001]);
        # llama.cpp maps tokenizer.ggml.eot_token_id to a stop token
        self.gguf_writer.add_eot_token_id(200001)
        # the checkpoint ships a vision template (image/video tokens this model does not have)
        tpl = (Path(__file__).resolve().parents[1] / "models" / "templates" / "Maxwell-V1-Flash.jinja")
        self.gguf_writer.add_chat_template(tpl.read_text(encoding="utf-8"))

    def set_gguf_parameters(self):
        n_nextn = self.hparams.get("mtp_num_layers", 0) or 0
        self.block_count = self.hparams["num_hidden_layers"] + n_nextn
        super().set_gguf_parameters()
        if n_nextn:
            self.gguf_writer.add_nextn_predict_layers(n_nextn)

        hp = self.hparams

        self.gguf_writer.add_key_length(hp["attention_head_size"])
        self.gguf_writer.add_value_length(hp["attention_head_size"])
        self.gguf_writer.add_layer_norm_rms_eps(hp["rms_norm_eps"])
        self.gguf_writer.add_vocab_size(hp["vocab_size"])

        # MoE
        self.gguf_writer.add_expert_count(hp["num_experts"])
        self.gguf_writer.add_expert_used_count(hp["num_experts_per_tok"])
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hp["shared_expert_intermediate_size"])
        # the reference builds one shared expert (modeling_maxwell_moe.py, num_shared_experts = 1)
        self.gguf_writer.add_expert_shared_count(hp.get("num_shared_experts", 1))
        self.gguf_writer.add_leading_dense_block_count(len(hp["mlp_only_layers"]))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        # the reference renormalizes the selected sigmoid scores and applies no scale
        self.gguf_writer.add_expert_weights_norm(True)
        self.gguf_writer.add_expert_weights_scale(1.0)

        # block sparse GQA indexer
        self.gguf_writer.add_indexer_head_count(hp["block_sparse_gqa_indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["block_sparse_gqa_indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["block_sparse_gqa_topk_blocks"])
        self.gguf_writer.add_indexer_block_size(hp["block_sparse_gqa_block_size"])
        self.gguf_writer.add_indexer_local_blocks(
            hp["block_sparse_gqa_local_window_size"] // hp["block_sparse_gqa_block_size"])
        self.gguf_writer.add_indexer_prefix_blocks(
            hp["block_sparse_gqa_prefix_window_size"] // hp["block_sparse_gqa_block_size"])
        # query positions below this attend dense (model card: dense-GQA prefix); not in config.json
        self.gguf_writer.add_indexer_dense_prefix_tokens(hp.get("block_sparse_gqa_dense_prefix_tokens", 16384))
        self.gguf_writer.add_indexer_types([True] * hp["num_hidden_layers"])

        # mHC
        self.gguf_writer.add_hyper_connection_count(hp["num_residual_streams"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hp["mhc_sinkhorn_iterations"])
        self.gguf_writer.add_hyper_connection_epsilon(1e-6)

    def _hc_scale(self, bid: int, kind: str, alpha: str, data: Tensor) -> Iterable[tuple[str, Tensor]]:
        assert bid is not None
        if self._alphas is None:
            self._alphas = [{} for _ in range(self.hparams["num_hidden_layers"] + (self.hparams.get("mtp_num_layers", 0) or 0))]
        slot = f"{kind}.{alpha}"
        self._alphas[bid][slot] = data.reshape(1)
        if len([k for k in self._alphas[bid] if k.startswith(kind)]) == 3:
            parts = [self._alphas[bid].pop(f"{kind}.{a}") for a in ("alpha_pre", "alpha_post", "alpha_res")]
            scale = torch.cat(parts).to(torch.float32)
            t = gguf.MODEL_TENSOR.HC_ATTN_SCALE if kind == "attn" else gguf.MODEL_TENSOR.HC_FFN_SCALE
            yield (self.format_tensor_name(t, bid), scale)
        return

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # top level
        if name == "model.embed_tokens.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)
            return
        if name == "model.norm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch)
            return
        if name == "lm_head.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)
            return
        if name.startswith("model.hc_head."):
            tn = {
                "model.hc_head.hc_head_fn":    gguf.MODEL_TENSOR.HC_HEAD_FN,
                "model.hc_head.hc_head_base":  gguf.MODEL_TENSOR.HC_HEAD_BASE,
                "model.hc_head.hc_head_scale": gguf.MODEL_TENSOR.HC_HEAD_SCALE,
            }[name]
            yield (self.format_tensor_name(tn), data_torch.to(torch.float32))
            return

        if name.startswith("mtp_layers."):
            yield from self._map_mtp(name, data_torch)
            return

        m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
        if m is None:
            raise ValueError(f"unexpected tensor: {name}")
        il, rest = int(m.group(1)), m.group(2)
        assert bid == il
        yield from self._map_layer(rest, il, data_torch)

    def _map_mtp(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        n_layer = self.hparams["num_hidden_layers"]
        m = re.match(r"mtp_layers\.(\d+)\.(.+)", name)
        assert m is not None, name
        i, rest = int(m.group(1)), m.group(2)
        il = n_layer + i

        if rest.startswith("decoder_layer."):
            yield from self._map_layer(rest[len("decoder_layer."):], il, data_torch)
            return
        if rest == "enorm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_ENORM, il), data_torch)
            return
        if rest == "hnorm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_HNORM, il), data_torch)
            return
        if rest == "final_layernorm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.LAYER_OUT_NORM, il), data_torch)
            return
        if rest.startswith("hc_head."):
            tn = {
                "hc_head.hc_head_fn":    gguf.MODEL_TENSOR.NEXTN_HC_HEAD_FN,
                "hc_head.hc_head_base":  gguf.MODEL_TENSOR.NEXTN_HC_HEAD_BASE,
                "hc_head.hc_head_scale": gguf.MODEL_TENSOR.NEXTN_HC_HEAD_SCALE,
            }[rest]
            yield (self.format_tensor_name(tn, il), data_torch.to(torch.float32))
            return
        if rest in ("e_proj.weight", "h_proj.weight"):
            assert self.hparams.get("mtp_num_layers", 1) == 1, "MTP e/h projection fusion assumes one MTP layer"
            if self._mtp_proj is None:
                self._mtp_proj = {}
            self._mtp_proj[rest] = data_torch
            if len(self._mtp_proj) < 2:
                return
            # the MTP block adds e_proj(embed) and h_proj(hidden) per stream: one [2*in, out] matmul
            fused = torch.cat([self._mtp_proj.pop("e_proj.weight"), self._mtp_proj.pop("h_proj.weight")], dim=1)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, il), fused.contiguous())
            return
        raise ValueError(f"unhandled MTP tensor: {name}")

    def _map_layer(self, rest: str, il: int, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:

        hc = re.match(r"(self_attn_hyper_connection|mlp_hyper_connection)\.(alpha_pre|alpha_post|alpha_res|bias|mapping_proj\.weight)", rest)
        if hc is not None:
            module, field = hc.group(1), hc.group(2)
            kind = "attn" if module == "self_attn_hyper_connection" else "ffn"
            if field.startswith("alpha_"):
                yield from self._hc_scale(il, kind, field, data_torch)
                return
            if field == "bias":
                tn = gguf.MODEL_TENSOR.HC_ATTN_BASE if kind == "attn" else gguf.MODEL_TENSOR.HC_FFN_BASE
                yield (self.format_tensor_name(tn, il, ".bias" if tn == gguf.MODEL_TENSOR.FFN_EXP_PROBS_B else ".weight"), data_torch.to(torch.float32))
                return
            tn = gguf.MODEL_TENSOR.HC_ATTN_FN if kind == "attn" else gguf.MODEL_TENSOR.HC_FFN_FN
            yield (self.format_tensor_name(tn, il), data_torch)
            return

        simple = {
            "input_layernorm.weight":        gguf.MODEL_TENSOR.ATTN_NORM,
            "post_attention_layernorm.weight": gguf.MODEL_TENSOR.FFN_NORM,
            "self_attn.q_proj.weight":       gguf.MODEL_TENSOR.ATTN_Q,
            "self_attn.k_proj.weight":       gguf.MODEL_TENSOR.ATTN_K,
            "self_attn.v_proj.weight":       gguf.MODEL_TENSOR.ATTN_V,
            "self_attn.o_proj.weight":       gguf.MODEL_TENSOR.ATTN_OUT,
            "self_attn.q_layernorm.weight":  gguf.MODEL_TENSOR.ATTN_Q_NORM,
            "self_attn.k_layernorm.weight":  gguf.MODEL_TENSOR.ATTN_K_NORM,
            "self_attn.indexer.linear_q_proj.weight": gguf.MODEL_TENSOR.INDEXER_Q_PROJ,
            "self_attn.indexer.linear_k_proj.weight": gguf.MODEL_TENSOR.INDEXER_K_PROJ,
            "self_attn.indexer.q_layernorm.weight":   gguf.MODEL_TENSOR.INDEXER_Q_NORM,
            "self_attn.indexer.k_layernorm.weight":   gguf.MODEL_TENSOR.INDEXER_K_NORM,
            "block_sparse_moe.gate.weight":  gguf.MODEL_TENSOR.FFN_GATE_INP,
            "block_sparse_moe.expert_bias":  gguf.MODEL_TENSOR.FFN_EXP_PROBS_B,
            "block_sparse_moe.w1.weight":    gguf.MODEL_TENSOR.FFN_GATE,
            "block_sparse_moe.w2.weight":    gguf.MODEL_TENSOR.FFN_DOWN,
            "block_sparse_moe.w3.weight":    gguf.MODEL_TENSOR.FFN_UP,
            "block_sparse_moe.shared_expert.w1.weight": gguf.MODEL_TENSOR.FFN_GATE_SHEXP,
            "block_sparse_moe.shared_expert.w2.weight": gguf.MODEL_TENSOR.FFN_DOWN_SHEXP,
            "block_sparse_moe.shared_expert.w3.weight": gguf.MODEL_TENSOR.FFN_UP_SHEXP,
        }
        if rest in simple:
            tn = simple[rest]
            if tn == gguf.MODEL_TENSOR.FFN_EXP_PROBS_B:
                yield (self.format_tensor_name(tn, il, ".bias" if tn == gguf.MODEL_TENSOR.FFN_EXP_PROBS_B else ".weight"), data_torch.to(torch.float32))
            else:
                yield (self.format_tensor_name(tn, il), data_torch)
            return

        mexp = re.match(r"block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight", rest)
        if mexp is not None:
            n_experts = self.hparams["num_experts"]
            if self._experts is None:
                self._experts = [{} for _ in range(self.hparams["num_hidden_layers"] + (self.hparams.get("mtp_num_layers", 0) or 0))]
            self._experts[il][rest] = data_torch
            if len(self._experts[il]) < n_experts * 3:
                return
            for proj, tn in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                             ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP),
                             ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP)):
                stacked = torch.stack([
                    self._experts[il].pop(f"block_sparse_moe.experts.{e}.{proj}.weight")
                    for e in range(n_experts)
                ], dim=0)  # [n_expert, n_out, n_in]
                # llama.cpp layout is [n_in, n_out, n_expert]
                yield (self.format_tensor_name(tn, il), stacked.contiguous())
            return

        raise ValueError(f"unhandled tensor: {name}")

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            left = [n for layer in self._experts for n in layer]
            if left:
                raise ValueError(f"unprocessed experts: {left[:4]}")
        if self._alphas is not None:
            left = [n for layer in self._alphas for n in layer]
            if left:
                raise ValueError(f"unprocessed hyper-connection alphas: {left[:4]}")
