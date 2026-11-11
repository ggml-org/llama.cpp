from __future__ import annotations

import math
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf


@ModelBase.register("Metis16ForCausalLM")
class Metis16Model(TextModel):
    model_arch = gguf.MODEL_ARCH.METIS16

    # read by generate_extra_tensors() or not used at inference: dropped in modify_tensors()
    _folded = {
        "pass_embeddings", "lm_head.weight", "depth_budget.multiplier",
        "depth_memory.pass_embeddings", "depth_memory.anchor_embeddings",
        "depth_memory.state_write.bias", "depth_memory.metadata_write.bias",
    }
    _folded_layer = {
        "mixer_connection.controller.weight", "mixer_connection.controller.bias", "mixer_connection.mix_logits",
        "mixer_connection.read_logits", "mixer_connection.write_logits",
        "moe_connection.controller.weight", "moe_connection.controller.bias", "moe_connection.mix_logits",
        "moe_connection.read_logits", "moe_connection.write_logits",
        "moe.k_router.weight", "moe.k_router.bias", "moe.k_budget.multiplier",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        self.n_pass = hp["num_passes"]
        self.attn_layers = list(hp["attention_layer_indices"])
        self._generated: set[str] = set()
        # one cache slot per pass and layer, while the tensors exist once per physical layer
        self.block_count = self.n_pass * hp["num_hidden_layers"]

    def set_vocab(self):
        self._set_vocab_gpt2()
        # turns end on the first eos id, so llama.cpp must stop on it too
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list) and eos:
            self.gguf_writer.add_eot_token_id(eos[0])

    def set_gguf_parameters(self):
        hp = self.hparams
        super().set_gguf_parameters()

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_head_count_kv([hp["num_key_value_heads"] if il % n_layer in self.attn_layers else 0 for il in range(self.block_count)])

        self.gguf_writer.add_expert_feed_forward_length(hp["expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hp["shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_moe_latent_size(hp["latent_dim"])

        d_inner = hp["mamba_expand"] * hp["hidden_size"]
        self.gguf_writer.add_ssm_conv_kernel(hp["mamba_d_conv"])
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(hp["mamba_d_state"])
        self.gguf_writer.add_ssm_time_step_rank(d_inner // hp["mamba_head_dim"])
        self.gguf_writer.add_ssm_group_count(hp["mamba_n_groups"])
        self.gguf_writer.add_ssm_norm_epsilon(hp["mamba_norm_eps"])

        self.gguf_writer.add_hyper_connection_count(hp["n_streams"])
        self.gguf_writer.add_metis_pass_count(self.n_pass)
        self.gguf_writer.add_metis_memory_length(hp["memory_dim"])
        self.gguf_writer.add_metis_route_length(hp["route_feature_dim"])
        self.gguf_writer.add_metis_continuation_logit_limit(hp["continuation_logit_limit"])
        self.gguf_writer.add_attention_pass_lora_rank(hp["attention_pass_lora_rank"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # the checkpoint leaves these names suffix-less; add the suffix of the GGUF tensor so the mapper keeps it
        if name.endswith((".mixer.impl.mixer.dt_bias", ".moe.selection_bias")):
            name += ".bias"
        elif (name.endswith((".moe.expert_embeddings", ".mixer.pass_gates"))
              or name in ("depth_memory.stream_gate", "stream_embeddings")):
            name += ".weight"
        elif name == "depth_memory.stream_gate_bias":
            name = "depth_memory.stream_gate.bias"
        return super().filter_tensors((name, gen))

    def _load(self, name: str) -> Tensor:
        return LazyTorchTensor.to_eager(self.model_tensors[name]())

    def _emit(self, name: str, data: Tensor) -> tuple[str, Tensor]:
        self._generated.add(name)
        return name, data

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        hp = self.hparams
        r = hp["route_feature_dim"]
        n_stream = hp["n_streams"]
        T = gguf.MODEL_TENSOR
        passes = self._load("pass_embeddings")[: self.n_pass]

        # the mixes depend only on the weights and the pass, so Sinkhorn runs once here
        # the controller runs in the checkpoint dtype, as the reference does
        for il in range(hp["num_hidden_layers"]):
            for src, (mix, read, write) in (
                ("mixer_connection", (T.HC_ATTN_MIX, T.HC_ATTN_READ, T.HC_ATTN_WRITE)),
                ("moe_connection", (T.HC_FFN_MIX, T.HC_FFN_READ, T.HC_FFN_WRITE)),
            ):
                prefix = f"layers.{il}.{src}."
                ctrl_w, ctrl_b = self._load(prefix + "controller.weight"), self._load(prefix + "controller.bias")
                mixes, reads, writes = [], [], []
                for p in range(self.n_pass):
                    delta = torch.nn.functional.linear(passes[p].to(ctrl_w.dtype), ctrl_w, ctrl_b).float()
                    m_delta, r_delta, w_delta = torch.split(delta, [n_stream * n_stream, n_stream, n_stream])
                    log_m = self._load(prefix + "mix_logits").float() + m_delta.view(n_stream, n_stream)
                    for _ in range(hp["mhc_sinkhorn_iterations"]):
                        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
                        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)
                    mixes.append(log_m.exp())
                    reads.append(torch.softmax(self._load(prefix + "read_logits").float() + r_delta, dim=-1))
                    writes.append(torch.softmax(self._load(prefix + "write_logits").float() + w_delta, dim=-1))
                act = ctrl_w.dtype
                yield self._emit(self.format_tensor_name(mix, il), torch.stack(mixes).to(act).float())
                yield self._emit(self.format_tensor_name(read, il), torch.stack(reads).to(act).float())
                yield self._emit(self.format_tensor_name(write, il), torch.stack(writes).to(act).float())

        # one bias per memory entry: write biases, pass and anchor embeddings, and the constant routed width
        meta_w = self._load("depth_memory.metadata_write.weight").float()
        mean_k = hp["num_experts_per_tok"] / hp["max_routed_k"]
        base = (self._load("depth_memory.state_write.bias").float() + self._load("depth_memory.metadata_write.bias").float()
                + mean_k * meta_w[:, r])
        pass_e = self._load("depth_memory.pass_embeddings").float()
        anchor_e = self._load("depth_memory.anchor_embeddings").float()
        n_anchor = len(self.attn_layers) + 1
        entry_bias = torch.stack([base + pass_e[p] + anchor_e[a] for p in range(self.n_pass) for a in range(n_anchor)])
        yield self._emit(self.format_tensor_name(T.DEPTH_MEM_ENTRY_BIAS), entry_bias)

    def _split_features(self, data_torch: Tensor, template: str, parts: tuple[str, ...], d: int, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the graph reads the concatenated features one block at a time; each block is its own tensor
        for i, part in enumerate(parts):
            piece = data_torch[:, i * d:(i + 1) * d] if i < 3 else data_torch[:, 3 * d:]
            yield from super().modify_tensors(piece, template.format(part), bid)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in self._generated:
            yield name, data_torch
            return
        if name in self._folded:
            return
        hp = self.hparams
        d = hp["hidden_size"]

        if bid is None:
            if name == "stream_embeddings.weight":
                # the graph keeps the streams innermost: [n_stream, n_embd, n_tokens]
                yield from super().modify_tensors(data_torch.T.contiguous(), name, bid)
                return
            if name == "depth_memory.metadata_write.weight":
                # metadata input becomes [summary, entropy, confidence, continuation]; entropy arrives unnormalized
                r = hp["route_feature_dim"]
                meta_w = data_torch.float()
                entropy_col = meta_w[:, r + 1:r + 2] / math.log(hp["n_routed_experts"])
                data_torch = torch.cat((meta_w[:, :r], entropy_col, meta_w[:, r + 2:]), dim=1)
            elif name == "depth_memory.stream_gate.weight":
                data_torch = data_torch.float() / math.sqrt(d)
            elif name == "depth_memory.route_projection.weight":
                yield from self._split_features(data_torch, "depth_memory.route_{}.weight", ("state", "retr", "diff", "hist"), d, bid)
                return
            elif name == "continuation.hidden.weight":
                yield from self._split_features(data_torch, "continuation.hidden_{}.weight", ("state", "retr", "diff", "route"), d, bid)
                return
            yield from super().modify_tensors(data_torch, name, bid)
            return

        local = name.removeprefix(f"layers.{bid}.")
        if local in self._folded_layer:
            return

        if local == "mixer.impl.mixer.A_log":
            data_torch = -torch.exp(data_torch).unsqueeze(-1)
        elif local == "mixer.impl.mixer.D":
            data_torch = data_torch.unsqueeze(-1)
        elif local == "mixer.impl.mixer.conv1d.weight":
            data_torch = data_torch.squeeze(1)
        elif local == "mixer.impl.mixer.norm.weight":
            data_torch = data_torch.reshape(hp["mamba_n_groups"], -1)
        elif local == "mixer.pass_gates.weight":
            # the reference applies tanh(gate) in the checkpoint dtype
            data_torch = torch.tanh(data_torch[: self.n_pass]).to(torch.bfloat16).float()
        elif local == "moe.shared_expert.gate_up.weight":
            gate, up = data_torch.chunk(2, dim=0)
            yield from super().modify_tensors(gate, name.replace("gate_up", "gate"), bid)
            yield from super().modify_tensors(up, name.replace("gate_up", "up"), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # FP32 in the checkpoint, tiny and read as rows, or computed here at full precision: keep FP32
        if (new_name.startswith(("cont_", "depth_mem_entry_bias", "depth_mem_stream_gate"))
                or new_name.endswith(("ffn_exp_embd.weight", "ffn_gate_inp.weight", "hc_attn_mix.weight", "hc_attn_read.weight",
                                      "hc_attn_write.weight", "hc_ffn_mix.weight", "hc_ffn_read.weight", "hc_ffn_write.weight"))):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
