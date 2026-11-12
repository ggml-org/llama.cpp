from __future__ import annotations

import json

import torch

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("ZayaForCausalLM")
@ModelBase.example("Zyphra/ZAYA1-8B")
class ZayaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ZAYA1

    # checkpoint names without a .weight/.bias suffix
    _RENAME = {
        "model.input_hidden_states_scale": "model.input_hidden_states.weight",
        "model.input_hidden_states_bias":  "model.input_hidden_states.bias",
    }

    # the two kv heads of attn_v: head 0 = current token, head 1 = previous token
    _V_PARTS = (
        "self_attn.qkv_proj.v_proj_current.weight",
        "self_attn.qkv_proj.v_proj_delayed.weight",
    )

    # (hs + b_hs) * s_hs + (res + b_res) * s_res = hs * s_hs + res * s_res + (b_hs * s_hs + b_res * s_res)
    _RES_SCALES = {
        "post_attention_residual_scale.": (gguf.MODEL_TENSOR.ATTN_RES_SCALE_HS, gguf.MODEL_TENSOR.ATTN_RESID_SCALE),
        "post_mlp_residual_scale.":       (gguf.MODEL_TENSOR.FFN_RES_SCALE_HS,  gguf.MODEL_TENSOR.FFN_RESID_SCALE),
    }
    _RES_PARTS = ("hidden_states_scale", "hidden_states_bias", "residual_scale", "residual_bias")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # added tokens overlap the base vocab, so count the token list itself
        self._vocab = gguf.LlamaHfVocab(self.dir_model)
        self._n_vocab = self._vocab.vocab_size
        self._parts: dict[tuple[int, str], dict[str, Tensor]] = {}
        self.rope_parameters = self.hparams["rope_parameters"]["hybrid"]

    def set_vocab(self):
        # added special tokens overlap the base vocab, so mark them as control here
        with open(self.dir_model / "tokenizer.json", "r", encoding="utf-8") as f:
            added = {t["id"]: t["special"] for t in json.load(f)["added_tokens"]}

        tokens, scores, toktypes = [], [], []
        for i, (text, score, toktype) in enumerate(self._vocab.all_tokens()):
            tokens.append(text)
            scores.append(score)
            if added.get(i):
                toktype = gguf.TokenType.CONTROL
            toktypes.append(toktype)

        self.gguf_writer.add_tokenizer_model("gemma4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # the HF post-processor appends <eos>, a prompt must not end with it
        special_vocab.add_special_token["eos"] = False
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        hparams = self.hparams
        super().set_gguf_parameters()

        self.gguf_writer.add_rope_dimension_count(int(self.rope_parameters["partial_rotary_factor"] * hparams["head_dim"]))
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_ssm_conv_kernel(hparams["cca_time0"])
        self.gguf_writer.add_router_hidden_size(hparams["router_hidden_size"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # TiDAR draft head, not used for autoregressive decoding
        if name.startswith(("diffusion_output_layer.", "diffusion_markov_head.")):
            return None

        return super().filter_tensors(item)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # the router picks one expert, so its matmuls stay F32 like ffn_gate_inp
        if any(self.match_model_tensor_name(new_name, key, bid) for key in (
                gguf.MODEL_TENSOR.CCA_CONV_GRP,
                gguf.MODEL_TENSOR.ROUTER_FC1,
                gguf.MODEL_TENSOR.ROUTER_FC2,
                gguf.MODEL_TENSOR.ROUTER_OUT)):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if bid is not None:
            local = name.removeprefix(f"model.layers.{bid}.")
            if local in self._V_PARTS:
                p = self._parts.setdefault((bid, "v"), {})
                p[local] = data_torch
                if len(p) == len(self._V_PARTS):
                    del self._parts[(bid, "v")]
                    v = torch.cat([p[n] for n in self._V_PARTS], dim=0)
                    yield from super().modify_tensors(v, f"model.layers.{bid}.{self._V_PARTS[0]}", bid)
                return
            for prefix, (key_hs, key_res) in self._RES_SCALES.items():
                if local.startswith(prefix) and local[len(prefix):] in self._RES_PARTS:
                    p = self._parts.setdefault((bid, prefix), {})
                    p[local[len(prefix):]] = data_torch
                    if len(p) == len(self._RES_PARTS):
                        del self._parts[(bid, prefix)]
                        s_hs, b_hs, s_res, b_res = (p[n].float() for n in self._RES_PARTS)
                        yield self.format_tensor_name(key_hs,  bid, ".weight"), s_hs
                        yield self.format_tensor_name(key_res, bid, ".weight"), s_res
                        yield self.format_tensor_name(key_res, bid, ".bias"),   b_hs * s_hs + b_res * s_res
                    return

        name = self._RENAME.get(name, name)
        if name.endswith("balancing_biases"):
            name += ".bias"
        elif not name.endswith((".weight", ".bias")):
            name += ".weight"
        new_name = self.map_tensor_name(name)

        def is_tensor(key: gguf.MODEL_TENSOR) -> bool:
            return self.match_model_tensor_name(new_name, key, bid)

        if any(is_tensor(key) for key in (gguf.MODEL_TENSOR.ATTN_Q, gguf.MODEL_TENSOR.ATTN_K)):
            # through the base class, which fuses q, k and v with --fuse-qkv
            yield from super().modify_tensors(data_torch, name, bid)
            return
        if is_tensor(gguf.MODEL_TENSOR.TOKEN_EMBD):
            data_torch = data_torch[:self._n_vocab]
        elif is_tensor(gguf.MODEL_TENSOR.SSM_CONV1D):
            data_torch = data_torch.squeeze(1)
        elif is_tensor(gguf.MODEL_TENSOR.CCA_CONV_GRP):
            # [n_group*out, in, kernel] -> [kernel, n_group, out, in]
            n_group = self.hparams["num_attention_heads"] + self.hparams["num_key_value_heads"]
            n_out, n_in, n_k = data_torch.shape
            data_torch = data_torch.reshape(n_group, n_out // n_group, n_in, n_k).permute(3, 0, 1, 2).contiguous()

        yield new_name, data_torch
