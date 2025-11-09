from __future__ import annotations
from .base import (
    ModelBase, TextModel, MmprojModel, gguf, torch
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Lfm2ForCausalLM", "LFM2ForCausalLM")
class LFM2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.LFM2

    def _add_feed_forward_length(self):
        ff_dim = self.hparams["block_ff_dim"]
        auto_adjust_ff_dim = self.hparams["block_auto_adjust_ff_dim"]
        ff_dim = self.hparams["block_ff_dim"]
        ffn_dim_multiplier = self.hparams["block_ffn_dim_multiplier"]
        multiple_of = self.hparams["block_multiple_of"]
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        self.gguf_writer.add_feed_forward_length(ff_dim)

    def set_gguf_parameters(self):
        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type == "full_attention" else 0
            for layer_type in self.hparams["layer_types"]
        ]
        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["norm_eps"])
        self._add_feed_forward_length()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        is_vision_tensor = "vision_tower" in name or "multi_modal_projector" in name
        if is_vision_tensor:
            # skip vision tensors
            return []
        name = name.replace("language_model.", "")
        # conv op requires 2d tensor
        if 'conv.conv' in name:
            data_torch = data_torch.squeeze(1)
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("Lfm2MoeForCausalLM")
class LFM2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LFM2MOE

    def set_gguf_parameters(self):
        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type == "full_attention" else 0
            for layer_type in self.hparams["layer_types"]
        ]
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_count(self.hparams["num_experts"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["num_dense_layers"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])
    # cache for experts weights for merging
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # conv op requires 2d tensor
        if 'conv.conv' in name:
            data_torch = data_torch.squeeze(1)
        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")
        # merge expert weights
        if 'experts' in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]
            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return []
            tensors: list[tuple[str, Tensor]] = []
            for w_name in expert_weights:
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.feed_forward.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]
                data_torch = torch.stack(datas, dim=0)
                merged_name = f"layers.{bid}.feed_forward.experts.{w_name}.weight"
                new_name = self.map_tensor_name(merged_name)
                tensors.append((new_name, data_torch))
            del self._experts_cache[bid]
            return tensors
        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        assert not self._experts_cache


@ModelBase.register("Lfm2VlForConditionalGeneration")
class LFM2VLModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # TODO(tarek): for dynamic resolution image_size is not specified, setting here for compatibility
        self.hparams_vision["image_size"] = 256

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LFM2)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["layer_norm_eps"]))
        self.gguf_writer.add_vision_projector_scale_factor(self.global_config.get("downsample_factor", 2))
        self.gguf_writer.add_vision_use_gelu(True)
        # python notation, e.g. for vision_feature_layer == -1, we pick last layer -> vision_feature_layers_to_drop = 0
        vision_feature_layers_to_drop = -(self.global_config.get("vision_feature_layer", -1) + 1)
        self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys) - vision_feature_layers_to_drop)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        is_vision_tensor = "vision_tower" in name or "multi_modal_projector" in name
        if is_vision_tensor:
            # remove "model." prefix
            name = name.replace("model.vision_tower.", "vision_tower.")
            name = name.replace("model.multi_modal_projector.", "multi_modal_projector.")
            if "patch_embedding.weight" in name:
                data_torch = data_torch.view(data_torch.shape[0], 16, 16, 3).permute(0, 3, 1, 2)
            return [(self.map_tensor_name(name), data_torch)]
        return [] # skip other tensors
