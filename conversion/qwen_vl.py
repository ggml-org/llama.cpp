from __future__ import annotations
from .base import (
    ModelBase, TextModel, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Qwen2VLModel", "Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen2_5OmniModel")
class Qwen2VLModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        mrope_section = self.hparams["rope_scaling"]["mrope_section"]
        mrope_section += [0] * max(0, 4 - len(mrope_section))
        self.gguf_writer.add_rope_dimension_sections(mrope_section)

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("thinker."):
            name = name.replace("thinker.", "")
        if name.startswith("visual") or name.startswith("audio") or name.startswith("talker") or name.startswith("token2wav"):
            # skip multimodal tensors
            return []
        if name.startswith("model.language_model.") or name.startswith("lm_head."):
            return []
        if name.startswith("model.visual."):
            return []
        # For other tensors, use the default mapping
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen2VLModel", "Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration")
class Qwen2VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = self.hparams_vision.get("image_size", 560)
        # rename config.json values
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get("num_heads")
        self.hparams_vision["num_hidden_layers"] = self.hparams_vision.get("depth")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN2VL)
        self.gguf_writer.add_vision_use_gelu(True)
        # Get image size
        image_size = self.hparams_vision.get("image_size", 560)
        self.gguf_writer.add_vision_image_size(image_size)
        # Get patch size
        patch_size = self.hparams_vision.get("patch_size", 14)
        self.gguf_writer.add_vision_patch_size(patch_size)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if name.startswith("model.language_model.") or name.startswith("lm_head."):
            return []
        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)
        if name == "visual.patch_embed.proj.weight":
            # split Conv3D into Conv2Ds
            c1, c2, kt, kh, kw = data_torch.shape
            del c1, c2, kh, kw  # unused
            assert kt == 2, "Current implmentation only support temporal_patch_size of 2"
            return [
                (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight"  , data_torch[:, :, 0, ...]),
                (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight.1", data_torch[:, :, 1, ...]),
            ]
        else:
            return [(self.map_tensor_name(name), data_torch)]
        return [] # skip other tensors


@ModelBase.register("Qwen2_5OmniModel")
class Qwen25OmniModel(Qwen2VLVisionModel):
    has_vision_encoder = True
    has_audio_encoder = True

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        if name.startswith("model.language_model.") or name.startswith("lm_head."):
            return []
        if name.startswith("visual") or name.startswith("audio") or name.startswith("talker") or name.startswith("token2wav"):
            return []
        return [(self.map_tensor_name(name), data_torch)]
