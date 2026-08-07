from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("Llama4ForConditionalGeneration")
class Llama4VisionModel(MmprojModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LLAMA4)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams["norm_eps"])
        self.gguf_writer.add_vision_projector_scale_factor(int(1.0 / self.hparams["pixel_shuffle_ratio"]))
        assert self.hparams["hidden_act"] == "gelu"
        self.gguf_writer.add_vision_use_gelu(True)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "multi_modal_projector" not in name and "vision_model" not in name:
            return None

        if "positional_embedding_vlm" in name and ".weight" not in name:
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "multi_modal_projector.linear_1" in name:
            # despite the name with number postfix, this is a single fully connected layer
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_MMPROJ_FC] + '.weight', data_torch)
        else:
            yield from super().modify_tensors(data_torch, name, bid)
