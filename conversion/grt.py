from __future__ import annotations

import math
import re

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf


# block tensors read together with their chain's angles; the residual gates fold into its scales
_READ_WITH_ANGLES = re.compile(
    r"attention\.([qkvo]_(scales|coupling)|[kv]_angles|_attn(_inv)?_perms)"
    r"|mixer\.(scales|coupling|_(inv_)?perms_stack)"
    r"|ffn\.((expand|contract)_scales|_ffn(_inv)?_perms)"
    r"|res_gate_(attn|mixer|ffn)"
)


def _butterfly_perms(n_stages: int, dim: int) -> tuple[Tensor, Tensor]:
    # stage k pairs dimension j with j + stride, stride = 2**(k % n_levels), as in the reference implementation
    half = dim // 2
    n_levels = max(1, int(math.log2(half))) if half > 1 else 1
    perms = []
    for k in range(n_stages):
        stride = 1 << (k % n_levels)
        perms.append([b + i + h * stride for b in range(0, dim, 2 * stride) for i in range(stride) for h in (0, 1)])
    perm = torch.tensor(perms, dtype=torch.long)
    return perm, torch.argsort(perm, dim=-1)


@ModelBase.register("GRTForCausalLM")
class GRTModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GRT

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # PearlNorm divides by sqrt(mean of the squared pair radii + eps) = sqrt(2) * sqrt(mean(x^2) + eps/2),
        # so it is an RMS norm with eps/2 and the scales divided by sqrt(2)
        self.gguf_writer.add_layer_norm_rms_eps(1e-8 / 2)
        # the position rotates the whole residual stream, pair i by pos / rope_freqs[i]
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"])
        self.gguf_writer.add_rope_freq_base(1.0)
        self.gguf_writer.add_attention_rotation_count(self.hparams["n_qkv_rotations"])
        self.gguf_writer.add_mixer_rotation_count(self.hparams["n_mixer_rotations"])
        self.gguf_writer.add_feed_forward_rotation_count(self.hparams["ffn_n_rotations"])

    def _get(self, name: str) -> Tensor:
        return LazyTorchTensor.to_eager(self.model_tensors[name]()).to(torch.float64)

    def _perms(self, perm_name: str, n_stages: int, dim: int) -> tuple[Tensor, Tensor]:
        perm, inv = _butterfly_perms(n_stages, dim)
        # the graph pairs j with j + stride instead of gathering, so the stored routing has to be this butterfly
        if perm_name in self.model_tensors and not torch.equal(LazyTorchTensor.to_eager(self.model_tensors[perm_name]()).long(), perm):
            raise ValueError(f"{perm_name} is not the butterfly permutation")
        return perm, inv

    def _gate(self, bid: int, branch: str) -> Tensor | float:
        name = f"model.blocks.{bid}.res_gate_{branch}"
        if name not in self.model_tensors:
            return 1.0
        raw = self._get(name)
        mode = self.hparams["residual_gate_mode"]
        if mode == "bounded":
            return 1.0 + 0.5 * torch.tanh(raw)
        if mode == "bounded_depth":
            return 0.55 + 0.45 * torch.tanh(raw)
        return raw

    def _coupled_chain(self, key: gguf.MODEL_TENSOR, bid: int, prefix: str, names: list[str], perm_name: str,
                       gate: Tensor | float) -> Iterable[tuple[str, Tensor]]:
        angles   = torch.cat([self._get(prefix + n + "angles") for n in names], dim=-2)
        coupling = torch.cat([self._get(prefix + n + "coupling") for n in names], dim=-3)
        scales   = F.softplus(torch.cat([self._get(prefix + n + "scales") for n in names], dim=-2))
        n_stages, dim = scales.shape[0], scales.shape[-1]

        # a stage scales its pairs before the inverse permutation, the graph scales after it
        _, inv = self._perms(perm_name, n_stages, dim)
        scales = torch.stack([scales[k][..., inv[k]] for k in range(n_stages)])
        # the residual gate of the sublayer folds into the scales of the last stage
        scales[-1] *= gate

        if angles.dim() == 2:
            # mixer: the angles of a stage are grouped like the coupling
            angles = angles.reshape(n_stages, coupling.shape[1], coupling.shape[-1])

        # the coupling matrices of all stages and heads (or mixer groups) form one batch, the stage outermost
        coupling = coupling.transpose(-1, -2).reshape(-1, coupling.shape[-1], coupling.shape[-2])
        yield self.format_tensor_name(key, bid, ".weight"), coupling.contiguous().float()
        yield self.format_tensor_name(key, bid, ".bias"),   angles.float()
        yield self.format_tensor_name(key, bid, ".scale"),  scales.float()

    def _plain_chain(self, key: gguf.MODEL_TENSOR, bid: int, prefix: str, gate: Tensor | float) -> Iterable[tuple[str, Tensor]]:
        angles = self._get(prefix + "angles")
        scales = F.softplus(self._get(prefix + "scales"))
        n_stages, dim = scales.shape
        perm, _ = self._perms(f"model.blocks.{bid}.ffn._ffn_perms", n_stages, dim)

        # a stage is linear: out = x*a + partner(x)*b, where partner(x)[j] is the value that x[j] is paired with
        cos, sin = torch.cos(angles), torch.sin(angles)
        coef = torch.zeros(n_stages, 2, dim, dtype=torch.float64)
        for k in range(n_stages):
            j0, j1 = perm[k][0::2], perm[k][1::2]
            s0, s1 = scales[k][0::2], scales[k][1::2]
            coef[k, 0, j0] = cos[k] * s0
            coef[k, 1, j0] = -sin[k] * s0
            coef[k, 0, j1] = cos[k] * s1
            coef[k, 1, j1] = sin[k] * s1
        coef[-1] *= gate

        yield self.format_tensor_name(key, bid, ".scale"), coef.float()

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if "model.pos_encoding.log_freqs" in self.model_tensors:
            log_freqs = self._get("model.pos_encoding.log_freqs")
        else:
            # fixed frequencies are not saved in the checkpoint
            n_pairs = self.hparams["hidden_size"] // 2
            log_freqs = -torch.arange(n_pairs, dtype=torch.float64) / n_pairs * math.log(self.hparams["pos_freq_base"])
        yield self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.exp(-log_freqs).float()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS):
            yield name, data_torch
            return

        if name in ("model.embedding.log_magnitude", "lm_head.output_scale", "model.pos_encoding.log_freqs"):
            return

        if name == "model.embedding.phase_angles":
            angles = self._get(name)
            magnitude = torch.exp(self._get("model.embedding.log_magnitude").clamp(-5.0, 5.0))
            table = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).flatten(1) * magnitude[:, None]
            # the head is a cosine against the same table, times one learned temperature
            head = table / (table.norm(dim=-1, keepdim=True) + 1e-6) * F.softplus(self._get("lm_head.output_scale"))
            yield self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), table.float()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), head.float()
            return

        # PearlNorm scales
        if name.endswith(".scale"):
            weight = self._get(name) / math.sqrt(2)
            if bid is not None and name.endswith("post_norm_attn.scale"):
                weight = weight * self._gate(bid, "attn")
            if bid is not None and name.endswith("post_norm_ffn.scale"):
                weight = weight * self._gate(bid, "ffn")
            yield self.map_tensor_name(name.removesuffix(".scale") + ".weight"), weight.float()
            return

        if bid is None:
            raise ValueError(f"Can not map tensor {name!r}")

        p = f"model.blocks.{bid}."
        leaf = name.removeprefix(p)

        if leaf == "attention.q_angles":
            yield from self._coupled_chain(gguf.MODEL_TENSOR.ATTN_QKV_ROT, bid, p + "attention.", ["q_", "k_", "v_"],
                                           p + "attention._attn_perms", 1.0)
        elif leaf == "attention.o_angles":
            gate = 1.0 if p + "post_norm_attn.scale" in self.model_tensors else self._gate(bid, "attn")
            yield from self._coupled_chain(gguf.MODEL_TENSOR.ATTN_OUT_ROT, bid, p + "attention.", ["o_"],
                                           p + "attention._attn_perms", gate)
        elif leaf == "mixer.angles":
            yield from self._coupled_chain(gguf.MODEL_TENSOR.MIXER_ROT, bid, p + "mixer.", [""],
                                           p + "mixer._perms_stack", self._gate(bid, "mixer"))
        elif leaf == "ffn.expand_angles":
            yield from self._plain_chain(gguf.MODEL_TENSOR.FFN_UP_ROT, bid, p + "ffn.expand_", 1.0)
        elif leaf == "ffn.contract_angles":
            gate = 1.0 if p + "post_norm_ffn.scale" in self.model_tensors else self._gate(bid, "ffn")
            yield from self._plain_chain(gguf.MODEL_TENSOR.FFN_DOWN_ROT, bid, p + "ffn.contract_", gate)
        elif not _READ_WITH_ANGLES.fullmatch(leaf):
            raise ValueError(f"Can not map tensor {name!r}")
