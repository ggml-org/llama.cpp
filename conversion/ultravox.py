from __future__ import annotations
from .base import (
    ModelBase, TextModel, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("UltravoxModel")
class UltravoxModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA # dummy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Ultravox does not have text decoder. Instead, it uses Llama or other models for text. If you want to get the audio encoder, please use --mmproj argument")


@ModelBase.register("Qwen2AudioForConditionalGeneration")
class WhisperEncoderModel(MmprojModel):
    has_vision_encoder = False # no vision encoder
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "hidden_size" not in self.hparams and "intermediate_size" not in self.hparams:
            self.hparams["hidden_size"] = self.hparams["d_model"]
            self.hparams["intermediate_size"] = self.hparams["encoder_ffn_dim"]
            self.hparams["num_attention_heads"] = self.hparams["encoder_attention_heads"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN2A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-5))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if name.startswith("language_model."):
            # skip language model tensors
            return []
        # prevent clash naming with vision tensors
        if name.startswith("multi_modal_projector"):
            name = "audio." + name
        if "conv1.bias" in name or "conv2.bias" in name:
            # transpose conv1 and conv2 bias
            data_torch = data_torch.unsqueeze(-1)
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("UltravoxModel")
class UltravoxWhisperEncoderModel(WhisperEncoderModel):
    has_vision_encoder = False # no vision encoder
    has_audio_encoder = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.ULTRAVOX)
        self.gguf_writer.add_audio_stack_factor(self.global_config["stack_factor"])


@ModelBase.register("VoxtralForConditionalGeneration")
class VoxtralWhisperEncoderModel(WhisperEncoderModel):
    has_vision_encoder = False # no vision encoder
    has_audio_encoder = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.VOXTRAL)
        self.gguf_writer.add_audio_stack_factor(4) # == intermediate_size // hidden_size
