from __future__ import annotations
from .base import (
    ModelBase, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Idefics3ForConditionalGeneration", "SmolVLMForConditionalGeneration")
class SmolVLMModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams["model_type"] == "smolvlm_vision":
            # fix for SmolVLM2, missing some keys in config.json
            # default values are taken from transformers code
            self.hparams["hidden_size"] = self.hparams.get("hidden_size", 1152)
            self.hparams["num_attention_heads"] = self.hparams.get("num_attention_heads", 16)
            self.hparams["intermediate_size"] = self.hparams.get("intermediate_size", 3072)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.IDEFICS3)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-5))
        self.gguf_writer.add_vision_projector_scale_factor(self.global_config.get("scale_factor", 2))
        self.gguf_writer.add_vision_use_gelu(True)
        # Add the preprocessor longest edge size
        preproc_image_size = self.preprocessor_config.get("size", {}).get("longest_edge", self.image_size)
        self.gguf_writer.add_vision_preproc_image_size(preproc_image_size)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".embeddings." in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        is_vision_tensor = "vision_tower" in name or "vision_model" in name or "model.connector" in name
        if is_vision_tensor:
            return [(self.map_tensor_name(name), data_torch)]
        return [] # skip other tensors
