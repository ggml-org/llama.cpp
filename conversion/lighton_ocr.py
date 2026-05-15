from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .llava import LlavaVisionModel


@ModelBase.register("LightOnOCRForConditionalGeneration")
class LightOnOCRVisionModel(LlavaVisionModel):
    is_mistral_format = False
    use_break_tok = False

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LIGHTONOCR)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        name = name.replace("model.vision_encoder.", "vision_tower.")
        name = name.replace("model.vision_projection.", "multi_modal_projector.")

        return super().filter_tensors((name, gen))
