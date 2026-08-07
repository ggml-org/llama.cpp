from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("DotsOCRForCausalLM")
class DotsOCRVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = 0 # dynamic resolution

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.DOTSOCR)
        self.gguf_writer.add_vision_min_pixels(self.preprocessor_config["min_pixels"])
        self.gguf_writer.add_vision_max_pixels(self.preprocessor_config["max_pixels"])
        self.gguf_writer.add_vision_attention_layernorm_eps(self.find_vparam(["rms_norm_eps"]))
        self.gguf_writer.add_vision_projector_scale_factor(self.find_vparam(["spatial_merge_size"]))
        self.gguf_writer.add_vision_use_silu(True)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("vision_tower."):
            return None

        if "vision_tower.blocks." in name and ".mlp." in name:
            # note: to avoid naming conflicts in tensor_mapping.py, we need to handle FFN renaming here
            # x = F.silu(self.fc1(x)) * self.fc3(x)
            # x = self.fc2(x)
            # fc1 -> gate, fc2 -> down, fc3 -> up
            # mapping original names to Qwen2.5 naming scheme
            name = name.replace("vision_tower.blocks.", "visual.blocks.")
            name = name.replace(".fc1", ".gate_proj")
            name = name.replace(".fc2", ".down_proj")
            name = name.replace(".fc3", ".up_proj")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        yield from super().modify_tensors(data_torch, name, bid)
