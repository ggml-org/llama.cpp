from __future__ import annotations
from .base import (
    ModelBase, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("KimiVLForConditionalGeneration")
class KimiVLModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = 64 * 14 # for compatibility

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.KIMIVL)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_projector_scale_factor(2)
        # eps is the same as pytorch's default value
        assert self.hparams_vision is not None
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("layer_norm_eps", 1e-5))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        is_vision_tensor = "vision_tower" in name or "multi_modal_projector" in name
        if is_vision_tensor:
            if "pos_emb.weight" in name:
                data_torch = data_torch.view(data_torch.shape[0] * data_torch.shape[1], data_torch.shape[2])
            elif "wqkv" in name:
                split_dim = 0 if "weight" in name else -1
                wq, wk, wv = data_torch.chunk(3, dim=split_dim)
                return [
                    (self.map_tensor_name(name.replace("wqkv", "wq")), wq),
                    (self.map_tensor_name(name.replace("wqkv", "wk")), wk),
                    (self.map_tensor_name(name.replace("wqkv", "wv")), wv)
                ]
            return [(self.map_tensor_name(name), data_torch)]
        return [] # skip other tensors
