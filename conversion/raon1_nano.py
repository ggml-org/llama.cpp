from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger


@ModelBase.register("Raon1NanoForCausalLM")
@ModelBase.example("KRAFTON/Raon-1-nano")
class Raon1NanoModel(TextModel):
    # Raon-1 nano: a hybrid of Raon delta attention (RDA) and MLA layers with mHC residual streams,
    # dense SiLU FFNs, layout-2 per-layer embeddings and a tied head
    model_arch = gguf.MODEL_ARCH.RAON1_NANO

    # mHC sites: HF module name -> tensor prefix; each site has 10 parameters, folded into fn / base / chart
    _MHC_SITES = (("self_attn_raon_mhc", "HC_ATTN"), ("mlp_raon_mhc", "HC_FFN"))
    _MHC_PARTS = ("phi_pre", "phi_post", "res.phi_z", "b_pre", "b_post", "res.b_z",
                  "alpha_pre", "alpha_post", "res.alpha_z", "res.eta")
    # kept unquantized by llama-quantize (src/llama-quant.cpp); a quantized --outtype keeps them at F16 the same way
    _KEEP_UNQUANTIZED = ("hc_attn_fn", "hc_ffn_fn", "ssm_f_a", "ssm_f_b")
    _KEEP_UNQUANTIZED_Q8_0 = ("attn_kv_a_mqa", "attn_k_b", "attn_v_b")
    # RDA tensors folded into the fused layout (src/models/raon1nano.cpp build_rda_layer)
    _RDA_PARTS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "g_proj.weight", "q_conv1d.weight", "k_conv1d.weight",
                  "v_conv1d.weight", "f_a_proj.weight", "b_proj.weight", "rda_w_proj.weight", "f_b_proj.weight", "A_log", "dt_bias")
    # per_layer_input_gate is not a Linear: its parameters have no .weight suffix
    _PLE_LAYER = {"gate": gguf.MODEL_TENSOR.PER_LAYER_INP_GATE, "projection": gguf.MODEL_TENSOR.PER_LAYER_PROJ,
                  "output_norm.weight": gguf.MODEL_TENSOR.PER_LAYER_POST_NORM}
    _PLE_MODEL = {"model.per_layer_inputs.table": gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
                  "model.per_layer_inputs.embedding_projection": gguf.MODEL_TENSOR.PER_LAYER_MODEL_PROJ,
                  "model.per_layer_inputs.projection_norm.weight": gguf.MODEL_TENSOR.PER_LAYER_PROJ_NORM}
    # post = 2 * sigmoid(.) in the reference; the 2 is folded into o_proj and down_proj (exact, a power of two)
    _HC_POST_MAGNITUDE = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parts: dict[tuple[int, str], dict[str, Tensor]] = {}
        # 1-based layer ids, as in the HF config (is_rda_layer: layer_idx + 1 in rda_layers)
        self._rda_layers = set(self.hparams["rda_config"]["rda_layers"])

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)
        # MTP is not supported: drop the nextn layers (model.layers.{num_hidden_layers}..) before any data is read
        n_layer = self.hparams["num_hidden_layers"]
        mtp = [name for name in tensors if name.startswith("model.layers.") and int(name.split(".")[2]) >= n_layer]
        if mtp:
            logger.info(f"Skipping {len(mtp)} MTP tensors from model.layers.{n_layer}; MTP is not supported")
            for name in mtp:
                del tensors[name]
        return tensors

    def set_gguf_parameters(self):
        hp = self.hparams
        rda_config = hp["rda_config"]
        n_layer = hp["num_hidden_layers"]
        rda, mla = self._rda_layers, set(rda_config["mla_layers"])
        if rda & mla or rda | mla != set(range(1, n_layer + 1)) or not rda or not mla:
            raise ValueError("Raon-1 nano attention layer lists must partition all layers, with both kinds present")
        if (hp.get("num_experts") is not None or hp["hidden_act"] != "silu" or not hp["mla_use_nope"]
                or not hp["mla_use_output_gate"] or not rda_config["full_rank_gate"]
                or hp["per_layer_embedding_layout_version"] != 2 or not hp.get("tie_word_embeddings")
                or hp.get("attn_res_block_size") or hp.get("num_nextn_predict_layers")):
            raise ValueError("Unsupported Raon-1 nano architecture variant")

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        # RDA and MLA head counts differ: per-layer arrays
        self.gguf_writer.add_head_count([rda_config["num_heads"] if i + 1 in rda else hp["num_attention_heads"] for i in range(n_layer)])
        self.gguf_writer.add_head_count_kv([0 if i + 1 in rda else 1 for i in range(n_layer)])
        self.gguf_writer.add_ssm_conv_kernel(rda_config["conv_kernel_size"])
        self.gguf_writer.add_ssm_time_step_rank(hp.get("rda_decay_rank", 128))
        self.gguf_writer.add_kda_head_dim(rda_config["head_dim"])
        if (bound := rda_config["gate_lower_bound"]) is not None:
            self.gguf_writer.add_kda_gate_lower_bound(bound)
        self.gguf_writer.add_q_lora_rank(hp["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hp["kv_lora_rank"])
        self.gguf_writer.add_rope_dimension_count(hp["qk_rope_head_dim"])
        self.gguf_writer.add_key_length(hp["kv_lora_rank"] + hp["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hp["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(hp["qk_nope_head_dim"] + hp["qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hp["v_head_dim"])
        # q_a_layernorm / kv_a_layernorm use eps 1e-6 in the reference
        self.gguf_writer.add_latent_norm_rms_eps(1e-6)
        self.gguf_writer.add_embedding_length_per_layer_input(hp["per_layer_embedding_dim"])
        self.gguf_writer.add_hyper_connection_count(hp["raon_mhc_streams"])
        # the mHC selector RMS uses a fixed eps of 1e-5; the post factor is folded into the branch outputs,
        # so the file stores magnitude 1
        self.gguf_writer.add_hyper_connection_epsilon(1e-5)
        self.gguf_writer.add_hyper_connection_magnitude(1.0)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._parts:
            raise ValueError(f"Incomplete Raon-1 nano tensor groups: {list(self._parts)}")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".hc_" in new_name and not new_name.endswith("_fn.weight"):
            return gguf.GGMLQuantizationType.F32
        quantized = (gguf.LlamaFileType.MOSTLY_Q8_0, gguf.LlamaFileType.MOSTLY_TQ1_0, gguf.LlamaFileType.MOSTLY_TQ2_0)
        if self.ftype in quantized and new_name.startswith("blk."):
            keep = self._KEEP_UNQUANTIZED
            if self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                keep += self._KEEP_UNQUANTIZED_Q8_0
            if new_name.split(".")[2] in keep:
                return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _fold_mhc(self, parts: dict[str, Tensor], bid: int, prefix: str) -> Iterable[tuple[str, Tensor]]:
        # fn rows scaled by alpha; chart = [gamma * (H (x) H) acting on the mix vector | -1/(gamma*anchor) | (1-gamma) I + gamma*anchor]
        n = self.hparams["raon_mhc_streams"]
        nm = n - 1
        # small per-site parameters, folded in float64 below: evaluate them now
        f = {k: LazyTorchTensor.to_eager(v).float() for k, v in parts.items()}
        fn = torch.cat([f["phi_pre"] * f["alpha_pre"], f["phi_post"] * f["alpha_post"], f["res.phi_z"] * f["res.alpha_z"]])
        base = torch.cat([f["b_pre"], f["b_post"], f["res.b_z"]])
        gamma = torch.sigmoid(f["res.eta"].reshape(()).double())
        # Helmert basis H [n, n-1]
        q = torch.tensor([[(1. if i <= j else -(j + 1.) if i == j + 1 else 0.) / ((j + 1) * (j + 2)) ** .5
                           for j in range(nm)] for i in range(n)], dtype=torch.float64)
        # e[p + n*i] = sum_{a,c} H[p,a] H[i,c] z[a*nm + c], z = mixes[2n:]
        kron = torch.einsum("pa,ic->ipac", q, q).reshape(n * n, nm * nm)
        k = torch.cat([torch.zeros(n * n, 2 * n, dtype=torch.float64), gamma * kron], dim=1)
        anchor = .5 * (torch.eye(n, dtype=torch.float64) + 1. / n)
        chart = torch.cat([k.flatten(), (-1. / (gamma * anchor)).flatten(),
                           ((1 - gamma) * torch.eye(n, dtype=torch.float64) + gamma * anchor).flatten()]).float()
        for kind, tensor in (("FN", fn), ("BASE", base), ("CHART", chart)):
            yield self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"{prefix}_{kind}"), bid), tensor

    def _fold_rda(self, p: dict[str, Tensor], bid: int) -> Iterable[tuple[str, Tensor]]:
        heads = self.hparams["rda_config"]["num_heads"]
        exp_a = torch.exp(p["A_log"].float()).repeat_interleave(self.hparams["rda_config"]["head_dim"])
        b, d = p["b_proj.weight"].float(), p["rda_w_proj.weight"].float()
        assert b.shape[0] == heads
        yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV, bid), torch.cat([p[f"{x}_proj.weight"] for x in "qkvg"])
        yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid), torch.cat([p[f"{x}_conv1d.weight"] for x in "qkv"])
        yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_F_A, bid), torch.cat([p["f_a_proj.weight"].float(), b, b + d])
        yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_F_B, bid), p["f_b_proj.weight"].float() * exp_a[:, None]
        yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_DT, bid, ".bias"), p["dt_bias"].float() * exp_a

    def _collect(self, key: tuple[int, str], suffix: str, data_torch: Tensor) -> dict[str, Tensor]:
        parts = self._parts.setdefault(key, {})
        if suffix in parts:
            raise ValueError(f"Duplicate tensor {suffix} in group {key}")
        parts[suffix] = data_torch
        return parts

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight":
            return  # tied to model.embed_tokens
        for site, prefix in self._MHC_SITES:
            if f".{site}." in name:
                assert bid is not None
                parts = self._collect((bid, site), name.split(f".{site}.", 1)[1], data_torch)
                if len(parts) == len(self._MHC_PARTS):
                    yield from self._fold_mhc(parts, bid, prefix)
                    del self._parts[bid, site]
                return
        if bid is not None and bid + 1 in self._rda_layers and ".self_attn." in name:
            suffix = name.split(".self_attn.", 1)[1]
            if suffix in self._RDA_PARTS:
                parts = self._collect((bid, "rda"), suffix, data_torch)
                if len(parts) == len(self._RDA_PARTS):
                    yield from self._fold_rda(parts, bid)
                    del self._parts[bid, "rda"]
                return
        if name.endswith((".self_attn.o_proj.weight", ".mlp.down_proj.weight")):
            data_torch = data_torch.float() * self._HC_POST_MAGNITUDE
        if bid is not None and ".mlp." in name and name.endswith(("gate_proj.weight", "up_proj.weight")):
            # ffn_up rows [gate | up] for one swiglu
            parts = self._collect((bid, "ffn"), name.rsplit(".", 2)[1], data_torch)
            if len(parts) == 2:
                yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), torch.cat([parts["gate_proj"], parts["up_proj"]])
                del self._parts[bid, "ffn"]
            return
        if name.endswith(".per_layer_input_gate.gate"):
            # the graph feeds the stream sum, not the mean
            data_torch = data_torch.float() / self.hparams["raon_mhc_streams"]
        if name in self._PLE_MODEL:
            yield self.format_tensor_name(self._PLE_MODEL[name]), data_torch
            return
        if ".per_layer_input_gate." in name:
            assert bid is not None
            yield self.format_tensor_name(self._PLE_LAYER[name.split(".per_layer_input_gate.", 1)[1]], bid), data_torch
            return
        if name.endswith("kv_b_proj.weight"):
            # absorbed MLA: per-head k_b [kv_lora_rank, qk_nope] (transposed) and v_b [v_head_dim, kv_lora_rank]
            heads = self.hparams["num_attention_heads"]
            nk, nv = self.hparams["qk_nope_head_dim"], self.hparams["v_head_dim"]
            kv = data_torch.reshape(heads, nk + nv, self.hparams["kv_lora_rank"])
            k, v = torch.split(kv, [nk, nv], dim=1)
            yield self.map_tensor_name(name.replace("kv_b_proj", "k_b_proj")), k.transpose(1, 2).contiguous()
            yield self.map_tensor_name(name.replace("kv_b_proj", "v_b_proj")), v.contiguous()
            return
        if name.endswith(".self_attn.g_proj.weight"):
            # MLA output gate (the RDA g_proj is folded into attn_qkv above)
            assert bid is not None
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid), data_torch
            return
        yield from super().modify_tensors(data_torch, name, bid)
