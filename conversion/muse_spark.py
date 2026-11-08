from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, gguf
from .muse_glimmer import MuseGlimmerVisionModel


# The reference clamps the squared stream coefficient with this floor.
RESIDUAL_GATE_EPS = 1e-3

# Block norms hold a delta around one; the output norm holds an absolute gain.
OFFSET_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "pre_expert_norm.weight",
    "post_expert_norm.weight",
)


def _unpermute_for_rope(tensor: "Tensor", n_heads: int) -> "Tensor":
    """Invert transformers' `_permute_for_rope`: HF stores Q/K in rotate_half layout,
    llama.cpp consumes the interleaved (NORM) layout. Vision Q/K carry a bias too."""
    if tensor.ndim == 1:
        dim1, = tensor.shape
        return tensor.view(n_heads, 2, dim1 // n_heads // 2).transpose(1, 2).reshape(dim1)
    dim1, dim2 = tensor.shape
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


@ModelBase.register("MuseSparkForConditionalGeneration")
@ModelBase.example("meta-models/Muse-Spark-Open-816B-A42B")
class MuseSparkModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MUSE_SPARK

    def set_vocab(self):
        self._set_vocab_gpt2()

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.dir_model)
        eot_id = tok.convert_tokens_to_ids("<|eot|>")  # ty: ignore[unresolved-attribute]
        if isinstance(eot_id, int) and eot_id >= 0:
            self.gguf_writer.add_eot_token_id(eot_id)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # MetaP resolves to two scalars
        # The output multiplier lands here and the attention scale rides on the synthesized QK-norm weights
        self.gguf_writer.add_logit_scale(hparams["output_multiplier"])
        self.gguf_writer.add_final_logit_softcapping(hparams["final_logit_softcapping"])

        self.gguf_writer.add_post_norm_eps(hparams["post_norm_eps"])

        # Only the sliding layers carry rope, so the graph takes it from the pattern
        sliding = [t == "sliding_attention" for t in hparams["layer_types"]]
        theta = hparams["rope_parameters"]["rope_theta"]
        assert sliding == [bool(t) for t in hparams["layer_rope_theta"]]
        assert all(t in (0, theta) for t in hparams["layer_rope_theta"])
        assert all(t == "sparse" for t in hparams["mlp_layer_types"])

        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_rope_freq_base_swa(theta)
        self.gguf_writer.add_sliding_window_pattern(sliding)

        # Experts live in a latent space narrower than the residual stream.
        self.gguf_writer.add_moe_latent_size(hparams["moe_hidden_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(True)
        # the config carries no score function, the reference hardcodes sigmoid
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(OFFSET_NORM_SUFFIXES):
            data_torch = data_torch + 1.0

        if name.endswith(".self_attn.q_proj.weight"):
            assert bid is not None
            n_head = self.hparams["num_attention_heads"]
            head_dim = self.hparams["head_dim"]

            yield self.map_tensor_name(name), _unpermute_for_rope(data_torch, n_head)

            # Synthesize QK-norm to carry the MetaP scale so the graph keeps the plain 1 / sqrt(head_dim)
            ones = torch.ones(head_dim, dtype=torch.float32)
            q_norm = ones * self.hparams["qk_scale_factor"]
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q_NORM, bid), q_norm
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_NORM, bid), ones
            return

        if name.endswith(".self_attn.k_proj.weight"):
            n_head_kv = self.hparams["num_key_value_heads"]
            yield self.map_tensor_name(name), _unpermute_for_rope(data_torch, n_head_kv)
            return

        # gate_up_proj fuses both projections over the output dim, gate first.
        if name.endswith(".mlp.experts.gate_up_proj"):
            assert bid is not None
            n_ff = data_torch.shape[1] // 2
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid), data_torch[:, :n_ff, :]
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP, bid), data_torch[:, n_ff:, :]
            return

        if name.endswith(".mlp.experts.down_proj"):
            assert bid is not None
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid), data_torch
            return

        # The router scales its logits by the MetaP output multiplier before the sigmoid
        # Fold it into the weight to keep a plain matmul
        if name.endswith(".mlp.gate.weight"):
            data_torch = data_torch.float() * self.hparams["output_multiplier"]
            yield from super().modify_tensors(data_torch, name, bid)
            return

        # Norm preserving residual gate: alpha scales the stream, beta the branch, with alpha^2 + beta^2 = 1
        # Both follow from the weight, so resolve them here
        if name.endswith("_residual_gate.gate"):
            assert bid is not None
            attn = "post_attention" in name

            w = data_torch.float() / self.hparams["residual_gate_temperature"]
            beta = torch.sigmoid(w)
            alpha_sq = torch.sigmoid(-w) * (1.0 + beta)
            alpha = torch.sqrt(torch.clamp(alpha_sq, min=RESIDUAL_GATE_EPS))

            t_alpha = gguf.MODEL_TENSOR.ATTN_RESID_ALPHA if attn else gguf.MODEL_TENSOR.FFN_RESID_ALPHA
            t_beta = gguf.MODEL_TENSOR.ATTN_RESID_BETA if attn else gguf.MODEL_TENSOR.FFN_RESID_BETA
            yield self.format_tensor_name(t_alpha, bid), alpha
            yield self.format_tensor_name(t_beta, bid), beta
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MuseSparkForConditionalGeneration")
@ModelBase.example("meta-models/Muse-Spark-Open-816B-A42B")
class MuseSparkVisionModel(MuseGlimmerVisionModel):
    has_vision_encoder = True # same vision encoder as Muse Glimmer
    has_audio_encoder = True

    def get_audio_config(self) -> dict[str, Any]:
        c = self.global_config["audio_config"]
        # the base class wants one width and one layer count: use the last stage width and the total number of conv blocks
        return {
            **c,
            "hidden_size": c["hidden_sizes"][-1],
            "intermediate_size": int(c["hidden_sizes"][-1] * c["mlp_ratio"]),
            "num_attention_heads": 1,
            "num_hidden_layers": sum(c["depths"]),
        }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        assert self.hparams_audio
        a = self.hparams_audio

        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.MUSE_SPARK_AUDIO)
        # the input is the raw waveform, one channel takes the place of the mel bins
        self.gguf_writer.add_audio_num_mel_bins(1)
        self.gguf_writer.add_audio_attention_layernorm_eps(float(a["rms_norm_eps"]))
        self.gguf_writer.add_audio_conv_kernel_size(int(a["conv_kernel_size"]))
        self.gguf_writer.add_audio_hidden_sizes([int(x) for x in a["hidden_sizes"]])
        self.gguf_writer.add_audio_patch_sizes([int(x) for x in a["patch_sizes"]])
        self.gguf_writer.add_audio_depths([int(x) for x in a["depths"]])

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if name.startswith("model.audio_tower."):
            return MmprojModel.filter_tensors((name, gen))
        return super().filter_tensors((name, gen))

    _AUDIO_HEAD = {
        "model.audio_tower.norm": gguf.MODEL_TENSOR.A_PRE_NORM,
        "model.audio_tower.proj": gguf.MODEL_TENSOR.A_MMPROJ_FC,
        "model.audio_tower.post_proj_norm": gguf.MODEL_TENSOR.A_POST_NORM,
    }

    def _audio_tensor(self, data_torch, name, suffix):
        assert self.hparams_audio

        stem = name[: -len(suffix) - 1]

        if stem in self._AUDIO_HEAD:
            return self.format_tensor_name(self._AUDIO_HEAD[stem], suffix="." + suffix), data_torch

        parts = stem.split(".")
        stage = int(parts[3])

        if parts[4] == "patch_embedder":
            key = (gguf.MODEL_TENSOR.A_ENC_PATCH_PROJ if parts[5] == "proj"
                   else gguf.MODEL_TENSOR.A_ENC_PATCH_NORM)
            return self.format_tensor_name(key, bid=stage, suffix="." + suffix), data_torch

        # conv blocks are numbered per stage upstream and flat here
        depths = [int(d) for d in self.hparams_audio["depths"]]
        bid = sum(depths[:stage]) + int(parts[5])
        tail = ".".join(parts[6:])
        key = {
            "conv": gguf.MODEL_TENSOR.A_ENC_CONV1D,
            "norm": gguf.MODEL_TENSOR.A_ENC_CONV1D_NORM,
            "mlp.fc1": gguf.MODEL_TENSOR.A_ENC_FFN_UP,
            "mlp.fc2": gguf.MODEL_TENSOR.A_ENC_FFN_DOWN,
        }[tail]
        return self.format_tensor_name(key, bid=bid, suffix="." + suffix), data_torch

    def modify_tensors(self, data_torch, name, bid):
        if name.startswith("model.audio_tower."):
            yield self._audio_tensor(data_torch, name, name.rpartition(".")[2])
            return
        yield from super().modify_tensors(data_torch, name, bid)
