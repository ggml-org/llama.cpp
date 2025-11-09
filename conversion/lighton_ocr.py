from __future__ import annotations
from .base import (
    ModelBase, gguf
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor
from .llava import LlavaVisionModel


@ModelBase.register("LightOnOCRForConditionalGeneration")
class LightOnOCRVisionModel(LlavaVisionModel):
    is_mistral_format = False
    use_break_tok = False

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LIGHTONOCR)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        name = name.replace("model.vision_encoder.", "vision_tower.")
        name = name.replace("model.vision_projection.", "multi_modal_projector.")
        return super().modify_tensors(data_torch, name, bid)
