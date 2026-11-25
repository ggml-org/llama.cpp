from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("LimiteForCausalLM")
@ModelBase.example("paradigma-inc/limite-1b-violetto")
class LimiteModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LIMITE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams

        expected = {
            "pos_mode": "rope",
            "qk_norm": "rms_pre_rope",
            "rope_style": "interleaved_pairs_odd_lane_sign_flip",
            "rope_per_layer": False,
            "rms_norm_has_weight": False,
            "rms_norm_eps_mode": "torch_finfo_default",
            "mlp_type": "swiglu",
            "attn_gate_applied": "per_head_before_o_proj",
            "ve_head_slice": "first_num_key_value_heads",
            "ve_applied_before_qk_norm": True,
            "global_nope": True,
            "global_window": -1,
            "mudd_r_site": "resid",
            "tie_word_embeddings": True,
            "final_softcap": 0.0,
        }
        for key, value in expected.items():
            if hp.get(key) != value:
                raise NotImplementedError(f"Limite: unsupported {key}={hp.get(key)!r}, expected {value!r}")
        if hp["softcap_logits"]["kind"] != "sigmoid":
            raise NotImplementedError(f"Limite: unsupported softcap_logits kind {hp['softcap_logits']['kind']!r}")

        self.n_head    = hp["num_attention_heads"]
        self.n_head_kv = hp["num_key_value_heads"]
        self.head_dim  = hp["head_dim"]
        self.n_pairs   = hp["rope_n_pairs"]

        self.mudd_taps: dict[int, list[int]] = {}
        if hp.get("mudd"):
            if not hp.get("mudd_mlp"):
                raise NotImplementedError("Limite: mudd without mudd_mlp is not supported")
            self.mudd_taps = {int(il): [int(t) for t in taps] for il, taps in hp["mudd_tap_idx"].items()}
            assert all(len(taps) == hp["mudd_taps"] for taps in self.mudd_taps.values())

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_key_length(self.head_dim)
        self.gguf_writer.add_value_length(self.head_dim)

        # torch rms_norm with eps=None uses the fp32 epsilon, even for bf16 inputs
        self.gguf_writer.add_layer_norm_rms_eps(torch.finfo(torch.float32).eps)
        self.gguf_writer.add_attention_scale(hp["attention_softmax_scale"])
        self.gguf_writer.add_xsa_eps(hp["xsa_normalize_eps"] if hp.get("xsa") else 0.0)

        # HF freq_i = base^(-i/(n_pairs-1)), ggml uses freq_base^(-i/n_pairs)
        self.gguf_writer.add_rope_dimension_count(2 * self.n_pairs)
        self.gguf_writer.add_rope_freq_base(hp["rope_base_local"] ** (self.n_pairs / (self.n_pairs - 1)))

        # HF keeps keys k >= q - window, so a window of W spans W + 1 tokens
        global_layers = set(hp["global_layers"])
        self.gguf_writer.add_sliding_window(hp["sliding_window"] + 1)
        self.gguf_writer.add_sliding_window_pattern([il not in global_layers for il in range(self.block_count)])

        cap = hp["softcap_logits"]
        self.gguf_writer.add_final_logit_sigmoid_capping(cap["a"], cap["b"], cap["c"])

        if self.mudd_taps:
            n_taps = hp["mudd_taps"]
            tap_indices = [-1] * (self.block_count * n_taps)
            for il, taps in self.mudd_taps.items():
                tap_indices[il * n_taps:(il + 1) * n_taps] = taps
            self.gguf_writer.add_mudd_feed_forward_length(hp["mudd_inter"])
            self.gguf_writer.add_mudd_tap_count(n_taps)
            self.gguf_writer.add_mudd_tap_indices(tap_indices)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # tiny mixing weights, kept at full precision as in the reference
        # the mudd tensors come from a global source tensor, so take the block id from the new name
        if new_name.startswith("blk."):
            bid = int(new_name.split(".")[1])
        for key in (gguf.MODEL_TENSOR.ATTN_GATE, gguf.MODEL_TENSOR.ATTN_VE_GATE, gguf.MODEL_TENSOR.MUDD_DOWN, gguf.MODEL_TENSOR.MUDD_UP):
            if self.match_model_tensor_name(new_name, key, bid):
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _swap_rope_pairs(self, w: Tensor, n_head: int) -> Tensor:
        # HF rotates each pair by -theta; swapping the pair lanes gives the ggml +theta rotation.
        # q and k are both swapped, so the attention scores do not change.
        idx = torch.arange(self.head_dim)
        idx[:2 * self.n_pairs] = idx[:2 * self.n_pairs].view(-1, 2).flip(-1).reshape(-1)
        w = w.reshape(n_head, self.head_dim, -1)
        return w[:, idx, :].reshape(n_head * self.head_dim, -1)

    def _pad_cols(self, w: Tensor) -> Tensor:
        # gates read only the first channels of the input, pad them to n_embd
        return torch.nn.functional.pad(w, (0, self.hparams["hidden_size"] - w.shape[-1]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hp = self.hparams
        T = gguf.MODEL_TENSOR

        def load(n: str) -> Tensor:
            return self.model_tensors[n]().float()

        if name == "model.value_embeds.weight":
            ve = data_torch.reshape(data_torch.shape[0], -1, hp["ve_dim"])[:, :self.n_head_kv]
            ve = torch.nn.functional.pad(ve, (0, self.head_dim - hp["ve_dim"]))
            yield self.format_tensor_name(T.VALUE_EMBD), ve.reshape(ve.shape[0], -1) * hp["ve_gate_scale"]
            return

        if name == "model.mudd.dense1":
            for il in self.mudd_taps:
                yield self.format_tensor_name(T.MUDD_DOWN, il), data_torch
            return
        if name == "model.mudd.dense2":
            # rows [0, n) mix the attention input, rows [n, 2n) mix the residual
            dense2_mlp = load("model.mudd.dense2_mlp")
            bias       = load("model.mudd.bias")
            bias_mlp   = load("model.mudd.bias_mlp")
            for il, taps in self.mudd_taps.items():
                n = len(taps)
                w = torch.cat([data_torch[il, :n], dense2_mlp[il, :n]], 0)
                b = torch.cat([bias[il, :n], bias_mlp[il, :n]], 0)
                yield self.format_tensor_name(T.MUDD_UP, il), w
                yield self.format_tensor_name(T.MUDD_UP, il, ".bias"), b
            return
        if name.startswith("model.mudd."):
            return

        if bid is None:
            yield from super().modify_tensors(data_torch, name, bid)
            return

        prefix = f"model.layers.{bid}."
        name = name.removeprefix(prefix)

        if name in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            data_torch = data_torch * load(prefix + "self_attn.qkv_scale")
            if name == "self_attn.q_proj.weight":
                data_torch = self._swap_rope_pairs(data_torch, self.n_head)
            elif name == "self_attn.k_proj.weight":
                data_torch = self._swap_rope_pairs(data_torch, self.n_head_kv)
        elif name == "self_attn.o_proj.weight":
            data_torch = data_torch * load(prefix + "self_attn.o_scale") * load(prefix + "post_lambda_attn") * hp["attn_gate_scale"]
        elif name == "mlp.down_proj.weight":
            data_torch = data_torch * load(prefix + "post_lambda_mlp")
        elif name == "self_attn.attn_gate":
            yield self.format_tensor_name(T.ATTN_GATE, bid), self._pad_cols(data_torch)
            return
        elif name == "self_attn.ve_gate":
            yield self.format_tensor_name(T.ATTN_VE_GATE, bid), self._pad_cols(data_torch[:self.n_head_kv])
            return
        elif name == "self_attn.xsa_alpha":
            if hp.get("xsa") and bid in hp["xsa_layers"]:
                yield self.format_tensor_name(T.ATTN_XSA_ALPHA, bid), torch.tanh(data_torch)
            return
        elif name == "resid_lambda_attn":
            yield self.format_tensor_name(T.ATTN_RESID_SCALE, bid), data_torch.reshape(1)
            return
        elif name == "resid_lambda_mlp":
            yield self.format_tensor_name(T.FFN_RESID_SCALE, bid), data_torch.reshape(1)
            return
        elif name in ("self_attn.qkv_scale", "self_attn.o_scale", "post_lambda_attn", "post_lambda_mlp"):
            # folded into the projections above
            return

        yield from super().modify_tensors(data_torch, prefix + name, bid)
