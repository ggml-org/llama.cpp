from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger


@ModelBase.register("ZayaModel", "ZayaForCausalLM")
class ZayaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ZAYA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._experts: dict[int, dict[str, Tensor]] | None = {}
        self._tokenizer_vocab_size: int | None = None
        try:
            from gguf.vocab import LlamaHfVocab
            self._tokenizer_vocab_size = LlamaHfVocab(self.dir_model).vocab_size
        except Exception:
            pass

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        vocab_size = self._tokenizer_vocab_size if self._tokenizer_vocab_size is not None else self.hparams["vocab_size"]
        self.gguf_writer.add_vocab_size(vocab_size)

        n_ff = self.hparams.get("ffn_hidden_size", 4096) // 2
        self.gguf_writer.add_feed_forward_length(n_ff)

        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("norm_epsilon", 1e-05))

        cca_time0 = self.hparams.get("cca_time0", 2)
        self.gguf_writer.add_ssm_conv_kernel(cca_time0)

        head_dim = self.hparams.get("head_dim", 128)
        partial_rotary = self.hparams.get("partial_rotary_factor", 0.5)
        self.gguf_writer.add_rope_dimension_count(int(partial_rotary * head_dim))

        n_expert_used = self.find_hparam(["moe_router_topk", "num_experts_per_tok"], optional=True) or 1
        self.gguf_writer.add_expert_used_count(n_expert_used)

        n_ff_exp = self.hparams.get("zaya_mlp_expansion", 256)
        self.gguf_writer.add_expert_feed_forward_length(n_ff_exp)

        zaya_high_prec = self.hparams.get("zaya_high_prec", True)
        self.gguf_writer.add_zaya_high_prec(zaya_high_prec)

    def _map_cca(self, name: str, data_torch: Tensor, bid: int) -> Iterable[tuple[str, Tensor]]:
        if "linear_q" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), data_torch
        elif "linear_k" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), data_torch
        elif "val_proj1" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_VAL_PROJ1, bid), data_torch
        elif "val_proj2" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_VAL_PROJ2, bid), data_torch
        elif "o_proj" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), data_torch
        elif "conv_qk.0" in name and name.endswith(".weight"):
            data_torch = data_torch.squeeze(1).contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid), data_torch
        elif "conv_qk.0" in name and name.endswith(".bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid, suffix=".bias"), data_torch
        elif "conv_qk.1" in name and name.endswith(".weight"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_CONV_GRP, bid), data_torch
        elif "conv_qk.1" in name and name.endswith(".bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_CONV_GRP, bid, suffix=".bias"), data_torch
        elif "temp" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_K_SCALE, bid), data_torch

    def _map_router(self, name: str, data_torch: Tensor, bid: int) -> Iterable[tuple[str, Tensor]]:
        if "down_proj.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid), data_torch
        elif "down_proj.bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid, suffix=".bias"), data_torch
        elif "rmsnorm_eda" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_NORM, bid), data_torch
        elif "router_mlp.0.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), data_torch
        elif "router_mlp.0.bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid, suffix=".bias"), data_torch
        elif "router_mlp.2.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2, bid), data_torch
        elif "router_mlp.2.bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2, bid, suffix=".bias"), data_torch
        elif "router_mlp.4.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP4, bid), data_torch
        elif "balancing_biases" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_BIASES, bid), data_torch
        elif "router_states_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_EDA_SCALE, bid), data_torch

    def _map_res_scale(self, name: str, data_torch: Tensor, bid: int) -> Iterable[tuple[str, Tensor]]:
        if "hidden_states_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS, bid), data_torch
        elif "hidden_states_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS, bid, suffix=".bias"), data_torch
        elif "residual_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES, bid), data_torch
        elif "residual_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES, bid, suffix=".bias"), data_torch

    def _map_final_res_scale(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        if "hidden_states_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS_FINAL), data_torch
        elif "hidden_states_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS_FINAL, suffix=".bias"), data_torch
        elif "residual_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES_FINAL), data_torch
        elif "residual_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES_FINAL, suffix=".bias"), data_torch

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embed_tokens.weight":
            if self._tokenizer_vocab_size is not None and data_torch.shape[0] > self._tokenizer_vocab_size:
                data_torch = data_torch[:self._tokenizer_vocab_size]
            yield self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch
            return
        if name == "model.final_norm.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch
            return
        if name.startswith("model.res_scale."):
            yield from self._map_final_res_scale(name, data_torch)
            return

        if bid is not None:
            if "self_attn" in name:
                yield from self._map_cca(name, data_torch, bid)
                return

            if "router" in name:
                yield from self._map_router(name, data_torch, bid)
                return

            if "input_norm" in name:
                yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid), data_torch
                return

            if "res_scale" in name:
                yield from self._map_res_scale(name, data_torch, bid)
                return

            if "zaya_block.experts" in name:
                assert bid is not None
                if self._experts is None:
                    self._experts = {}
                if bid not in self._experts:
                    self._experts[bid] = {}
                self._experts[bid][name] = data_torch

                n_expert = self.find_hparam(["num_experts"])
                if len(self._experts[bid]) >= n_expert * 2:
                    for w_name, gguf_tensor, permute_dims in [
                        ("linear_fc1", gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, None),
                        ("linear_fc2", gguf.MODEL_TENSOR.FFN_DOWN_EXP, None),
                    ]:
                        datas: list[Tensor] = []
                        for xid in range(n_expert):
                            ename = f"model.layers.{bid}.zaya_block.experts.local_experts.{xid}.{w_name}.weight"
                            datas.append(self._experts[bid][ename])
                            del self._experts[bid][ename]
                        data_torch_stacked = torch.stack(datas, dim=0)
                        if permute_dims is not None:
                            data_torch_stacked = data_torch_stacked.permute(*permute_dims)
                        yield self.format_tensor_name(gguf_tensor, bid), data_torch_stacked
                    del self._experts[bid]
                return

        try:
            yield from super().modify_tensors(data_torch, name, bid)
        except ValueError as e:
            if "Can not map tensor" in str(e):
                logger.warning(f"Skipping unmapped tensor: {name}")
            else:
                raise

    def set_vocab(self):
        from gguf.vocab import LlamaHfVocab

        vocab = LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("gemma4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)
        self.gguf_writer.add_add_bos_token(True)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts:
            unprocessed = [k for d in self._experts.values() for k in d.keys()]
            if unprocessed:
                raise ValueError(f"Unprocessed expert tensors: {unprocessed}")
