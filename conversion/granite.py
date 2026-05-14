from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger

from .llama import LlamaModel
from .mamba import Mamba2Model


@ModelBase.register("GraniteForCausalLM", "GraniteSpeechForConditionalGeneration")
class GraniteModel(LlamaModel):
    """Conversion for IBM's GraniteForCausalLM"""
    model_arch = gguf.MODEL_ARCH.GRANITE

    def set_gguf_parameters(self):
        """Granite uses standard llama parameters with the following differences:

        - No head_dim support
        - New multiplier params:
            - attention_scale
            - embedding_scale
            - residual_scale
        - logits_scaling
        """
        if head_dim := self.hparams.pop("head_dim", None):
            logger.warning("Ignoring head_dim (%s) from config for Granite", head_dim)
        super().set_gguf_parameters()
        # NOTE: Convert _multiplier params to _scale params for naming
        #   consistency
        if attention_scale := self.hparams.get("attention_multiplier"):
            self.gguf_writer.add_attention_scale(attention_scale)
            logger.info("gguf: (granite) attention_scale = %s", attention_scale)
        if embedding_scale := self.hparams.get("embedding_multiplier"):
            self.gguf_writer.add_embedding_scale(embedding_scale)
            logger.info("gguf: (granite) embedding_scale = %s", embedding_scale)
        if residual_scale := self.hparams.get("residual_multiplier"):
            self.gguf_writer.add_residual_scale(residual_scale)
            logger.info("gguf: (granite) residual_scale = %s", residual_scale)
        if logits_scale := self.hparams.get("logits_scaling"):
            self.gguf_writer.add_logit_scale(logits_scale)
            logger.info("gguf: (granite) logits_scale = %s", logits_scale)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith("encoder."):
            return None
        return super().filter_tensors(item)


@ModelBase.register("GraniteMoeForCausalLM", "GraniteMoeSharedForCausalLM")
class GraniteMoeModel(GraniteModel):
    """Conversion for IBM's GraniteMoeForCausalLM"""
    model_arch = gguf.MODEL_ARCH.GRANITE_MOE

    def set_gguf_parameters(self):
        """GraniteMoeShared uses GraniteMoe parameters plus the following:
        - shared_intermediate_size
        """
        super().set_gguf_parameters()
        if shared_feed_forward_length := self.hparams.get("shared_intermediate_size"):
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_feed_forward_length)
            logger.info("gguf: (granitemoeshared) shared_feed_forward_length = %s", shared_feed_forward_length)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """In modeling_granitemoe, the JetMoe implementation of parallel experts
        is used. This essentially merges w1 and w3 into a single tensor with 2x
        the hidden size that is then split during forward. To keep compatibility
        with existing mixtral support, we pull them apart here.
        """

        if name.endswith("block_sparse_moe.input_linear.weight"):
            ffn_dim = self.hparams["intermediate_size"]
            assert data_torch.shape[-2] == 2 * ffn_dim, "Merged FFN tensor size must be 2 * intermediate_size"
            gate, up = data_torch.split(ffn_dim, dim=-2)
            yield from ModelBase.modify_tensors(self, gate, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid), bid)
            yield from ModelBase.modify_tensors(self, up, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP, bid), bid)
            return

        has_experts = bool(self.hparams.get('num_local_experts'))

        if name.endswith("shared_mlp.input_linear.weight"):
            ffn_dim = self.hparams["shared_intermediate_size"]
            assert data_torch.shape[-2] == 2 * ffn_dim, "Merged FFN tensor size must be 2 * shared_intermediate_size"
            gate, up = data_torch.split(ffn_dim, dim=-2)
            if has_experts:
                yield from ModelBase.modify_tensors(self, gate,self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_SHEXP, bid), bid)
                yield from ModelBase.modify_tensors(self, up, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_SHEXP, bid), bid)
                return
            yield from ModelBase.modify_tensors(self, gate, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), bid)
            yield from ModelBase.modify_tensors(self, up, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), bid)
            return

        if not has_experts and name.endswith("shared_mlp.output_linear.weight"):
            yield from ModelBase.modify_tensors(self, data_torch, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN, bid), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("GraniteMoeHybridForCausalLM", "BambaForCausalLM")
class GraniteHybridModel(Mamba2Model, GraniteMoeModel):
    """GraniteHybrid is a hybrid SSM + Attention model that uses Mamba2 SSM
    layers and optionally uses MoE w/ a shared expert"""
    model_arch = gguf.MODEL_ARCH.GRANITE_HYBRID
    undo_permute = True

    def __init__(self, *args, **kwargs):

        # Hybrid mamba models use a prefix for the mamba-specific params.
        # TODO: Extend this if the prefix(es) need to be configurable
        self.hparam_prefixes = ["mamba"]

        super().__init__(*args, **kwargs)

        # Lists of which layers use ssm vs attention
        self._attn_layers = self.get_attn_layers()
        self._ssm_layers = [
            i for i in range(self.block_count)
            if i not in self._attn_layers
        ]

        # There are some models in this family that are non-hybrid, but keep the
        # same parent class by setting all layers to "attention." If this is the
        # case, the model architecture needs to be updated to a standard
        # "granite" or "granitemoe" model
        if not self._ssm_layers:
            has_experts = self.find_hparam(["num_experts_per_tok", "num_experts_per_token"], optional=True)
            new_arch = (
                gguf.MODEL_ARCH.GRANITE_MOE
                if has_experts else
                gguf.MODEL_ARCH.GRANITE
            )
            self.model_arch = new_arch
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[new_arch]
            self.gguf_writer.add_architecture()

        # n_group and d_inner are used during reshape_tensors for mamba2
        # NOTE: Explicitly include hparam prefix prefix for d_model to
        #   disambiguate with top-level head_dim
        # NOTE 2: If needed for future models, this can be isolated in a method
        #   to separate the prefix setting and the keys used
        self.d_model = self.find_hparam([f"{self.hparam_prefixes[0]}_head_dim", "hidden_size", "d_model"])
        self.n_group = self.find_hparam(["n_groups", "num_groups"])
        self.d_inner = self.find_hparam(["expand", "num_heads"]) * self.d_model

    def get_attn_layers(self):
        # Explicit list of layer type names
        if layer_types := self.hparams.get("layer_types"):
            return [
                i for i, typ in enumerate(layer_types)
                if typ == "attention"
            ]

        # Layer types indicated by index or period
        attn_layers = self.hparams.get("attn_layer_indices", [])
        if not attn_layers:
            attn_period = self.hparams.get("attn_layer_period")
            assert attn_period, "Didn't find attn_layer_indices or attn_layer_period"
            attn_offset = self.hparams.get("attn_layer_offset")
            assert attn_offset is not None, "No attention layer offset set with attn_layer_period"
            attn_layers = [
                i for i in range(self.block_count)
                if i % attn_period == attn_offset
            ]
        return attn_layers

    def find_hparam(self, keys: Iterable[str], *args, **kwargs) -> Any:
        prefixed = []
        for pfx in self.hparam_prefixes:
            prefixed.extend(
                "_".join([pfx, k])
                for k in keys
            )
        keys = list(keys) + prefixed
        return Mamba2Model.find_hparam(self, keys, *args, **kwargs)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if (
            name.endswith("block_sparse_moe.input_linear.weight")
            or "shared_mlp" in name
        ):
            yield from GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
            return

        # Determine whether this is a mamba layer or an attention layer
        if bid in self._ssm_layers:
            yield from Mamba2Model.modify_tensors(self, data_torch, name, bid)
            return
        elif bid in self._attn_layers:
            yield from GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
            return
        yield from ModelBase.modify_tensors(self, data_torch, name, bid)

    def set_gguf_parameters(self):
        """This method merges params from both parents and some that are
        specific to this model. The result is some duplication of how the params
        get set. The following warnings are expected during conversion:

        WARNING:Duplicated key name 'granitehybrid.attention.head_count_kv'
        WARNING:Duplicated key name 'granitehybrid.context_length'
        """
        GraniteMoeModel.set_gguf_parameters(self)

        ## Mamba mixer params ##
        self.gguf_writer.add_ssm_conv_kernel(self.find_hparam(["conv_kernel", "d_conv"]))
        self.gguf_writer.add_ssm_state_size(self.find_hparam(["state_size", "d_state", "state_dim", "ssm_state_size"]))
        self.gguf_writer.add_ssm_group_count(self.n_group)
        self.gguf_writer.add_ssm_inner_size(self.d_inner)
        # NOTE: The mamba_dt_rank is _not_ the right field for how this is used
        #   in llama.cpp
        self.gguf_writer.add_ssm_time_step_rank(self.find_hparam(["n_heads", "num_heads"]))

        ## Attention params ##
        head_count_kv = self.find_hparam(["num_key_value_heads", "n_head_kv"])
        head_count_kv_vec = [
            head_count_kv if i in self._attn_layers else 0 for i in range(self.block_count)
        ]
        if rope_dim := self.hparams.get("attn_rotary_emb"):
            self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_head_count_kv(head_count_kv_vec)

        ## If Bamba or non-hybrid, use rope, otherwise don't
        use_rope = (
            "BambaForCausalLM" in self.hparams["architectures"]
            or not self._ssm_layers
        )
        self.gguf_writer.add_rope_scaling_finetuned(use_rope)
        if not use_rope:
            self.gguf_writer.add_context_length(2**20)

        ## Validation ##
        d_head = self.find_hparam(["d_head"], optional=True) or 64
        assert self.hparams.get("hidden_act") in [None, "silu"], "Only SILU activation supported"
        assert self.d_inner % d_head == 0, f"SSM inner size {self.d_inner} not a multiple of head dim {d_head}"

    def set_vocab(self):
        self.hparams["pad_vocab_size_multiple"] = 8
        Mamba2Model.set_vocab(self)


@ModelBase.register("GraniteSpeechForConditionalGeneration")
class GraniteSpeechMmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    _batch_norm_tensors: list[dict[str, Tensor]] | None = None

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("encoder_config")

    def set_gguf_parameters(self):
        assert self.hparams_audio is not None
        a = self.hparams_audio
        a["hidden_size"] = a["hidden_dim"]
        a["intermediate_size"] = a["hidden_dim"] * a["feedforward_mult"]
        a["num_attention_heads"] = a["num_heads"]
        a["num_hidden_layers"] = a["num_layers"]

        super().set_gguf_parameters()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.GRANITE_SPEECH)
        self.gguf_writer.add_audio_num_mel_bins(a["input_dim"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)
        self.gguf_writer.add_audio_chunk_size(a["context_size"])
        self.gguf_writer.add_audio_conv_kernel_size(a["conv_kernel_size"])
        self.gguf_writer.add_audio_max_pos_emb(a["max_pos_emb"])

        p = self.global_config
        self.gguf_writer.add_audio_projector_window_size(p["window_size"])
        self.gguf_writer.add_audio_projector_downsample_rate(p["downsample_rate"])
        self.gguf_writer.add_audio_projector_head_count(p["projector_config"]["num_attention_heads"])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if "encoder" in name or "projector" in name:
            if ".conv" in name and ".weight" in name:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if "attention_dists" in name or "num_batches_tracked" in name:
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # fold running_mean, running_var and eps into weight and bias for batch_norm
        if "batch_norm" in name and "encoder.layers." in name:
            if self._batch_norm_tensors is None:
                self._batch_norm_tensors = [{} for _ in range(self.block_count)]
            assert bid is not None
            self._batch_norm_tensors[bid][name] = data_torch
            if len(self._batch_norm_tensors[bid]) < 4:
                return
            prefix = f"encoder.layers.{bid}.conv.batch_norm"
            weight = self._batch_norm_tensors[bid][f"{prefix}.weight"]
            bias = self._batch_norm_tensors[bid][f"{prefix}.bias"]
            running_mean = self._batch_norm_tensors[bid][f"{prefix}.running_mean"]
            running_var = self._batch_norm_tensors[bid][f"{prefix}.running_var"]
            eps = 1e-5
            a = weight / torch.sqrt(running_var + eps)
            b = bias - running_mean * a
            yield from super().modify_tensors(a, f"encoder.layers.{bid}.conv.batch_norm.weight", bid)
            yield from super().modify_tensors(b, f"encoder.layers.{bid}.conv.batch_norm.bias", bid)
            return

        if ".attn.to_kv.weight" in name:
            k_weight, v_weight = data_torch.chunk(2, dim=0)
            yield from super().modify_tensors(k_weight, name.replace("to_kv", "to_k"), bid)
            yield from super().modify_tensors(v_weight, name.replace("to_kv", "to_v"), bid)
            return

        if ("up_conv" in name or "down_conv" in name) and name.endswith(".weight"):
            if data_torch.ndim == 3 and data_torch.shape[2] == 1:
                data_torch = data_torch.squeeze(2)

        if "depth_conv" in name and name.endswith(".weight"):
            if data_torch.ndim == 3 and data_torch.shape[1] == 1:
                data_torch = data_torch.squeeze(1)

        yield from super().modify_tensors(data_torch, name, bid)
