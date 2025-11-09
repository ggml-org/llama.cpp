from __future__ import annotations
from .base import (
    gguf
)
from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    pass
from .llava import LlavaVisionModel


class PixtralModel(LlavaVisionModel):
    model_name = "Pixtral"
    hf_arch = ""
    is_mistral_format = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PIXTRAL)
        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.find_hparam(["norm_eps"])
        )
        self.gguf_writer.add_rope_freq_base(self.find_vparam(["rope_theta"]))
        self.gguf_writer.add_vision_use_silu(True)
        # spatial_merge_size
        if self.find_vparam(["mm_projector_id"]) == "patch_merge":
            self.gguf_writer.add_vision_spatial_merge_size(
                self.find_vparam(["spatial_merge_size"])
            )

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        if name == "vision_language_adapter.w_in.weight":
            return "mm.1.weight"
        elif name == "vision_language_adapter.w_out.weight":
            return "mm.2.weight"
        return super().map_tensor_name(name, try_suffixes)
