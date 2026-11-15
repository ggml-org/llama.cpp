from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .qwen import Qwen3Model
from .ultravox import WhisperEncoderModel


# plain ChatML as built by transcribe.py upstream, the Qwen3 template renders past assistant turns differently from the last one
HIGGS_AUDIO_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}"
    "{{- '<think>\\n\\n</think>\\n\\n' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
)


@ModelBase.register("HiggsAudio3Model")
@ModelBase.example("bosonai/higgs-audio-v3-stt")
class HiggsAudioModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template(HIGGS_AUDIO_CHAT_TEMPLATE)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # audio input goes to the mmproj, audio output heads are for TTS only
        if name.startswith(("audio_tower.", "audio_encoder_proj.", "audio_codebook_embeddings.", "audio_decoder_proj.audio_lm_head.")):
            return None

        if name == "audio_decoder_proj.text_lm_head.weight":
            name = "lm_head.weight"
        else:
            name = "model." + name

        return super().filter_tensors((name, gen))


@ModelBase.register("HiggsAudio3Model")
class HiggsAudioWhisperEncoderModel(WhisperEncoderModel):
    has_vision_encoder = False
    has_audio_encoder = True

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config["audio_encoder_config"]

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # depthwise conv kernel, same as the whisper convs
        if name == "audio_encoder_proj.temporal.weight":
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_gguf_parameters(self):
        assert self.global_config["projector_type"] == "mlp" and self.global_config["projector_temporal_downsample"] == 2
        assert self.global_config["audio_adapter_type"] == "stack" and not self.global_config.get("duplicate_audio_triplet", False)
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.HIGGS_AUDIO)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if not name.startswith(("audio_tower.", "audio_encoder_proj.")):
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("audio_encoder_proj."):
            suffix = "." + name.rsplit(".", 1)[-1]
            if ".temporal." in name:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_ENC_DOWNSAMPLE_CONV, suffix=suffix), data_torch)
            elif ".linear1." in name:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 1, suffix=suffix), data_torch)
            elif ".linear2." in name:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 2, suffix=suffix), data_torch)
            else:
                raise ValueError(f"unexpected tensor {name}")
            return

        yield from super().modify_tensors(data_torch, name, bid)
