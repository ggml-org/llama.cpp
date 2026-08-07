from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf


@ModelBase.register("UltravoxModel")
class UltravoxModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA # dummy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Ultravox does not have text decoder. Instead, it uses Llama or other models for text. If you want to get the audio encoder, please use --mmproj argument")


@ModelBase.register("GlmasrModel")
class GlmASRWhisperEncoderModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "hidden_size" not in self.hparams and "intermediate_size" not in self.hparams:
            self.hparams["hidden_size"] = self.hparams["d_model"]
            self.hparams["intermediate_size"] = self.hparams["encoder_ffn_dim"]
            self.hparams["num_attention_heads"] = self.hparams["encoder_attention_heads"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.GLMA)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-5))
        self.gguf_writer.add_audio_stack_factor(self.global_config["merge_factor"])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(("model.", "lm_head.")):
            # skip language model tensors
            return None

        if name.startswith("audio_encoder.whisper."):
            name = name.replace("audio_encoder.whisper.","audio_tower.")
        if "audio_encoder.layer_norm." in name or "audio_encoder.proj." in name:
            name = name.replace("audio_encoder.", "audio_encoder.adapting.")
        if name.startswith("audio_encoder.adapting."):
            name = name.replace("audio_encoder.adapting.","audio.multi_modal_projector.")
            if ".layer_norm." in name:
                name = name.replace(".layer_norm.", ".ln_pre.")
            if ".0." in name:
                name = name.replace(".0.", ".linear_1.")
            if ".2." in name:
                name = name.replace(".2.", ".linear_2.")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("audio_encoder.audio_bos_eos_token."):
            yield from super().modify_tensors(data_torch[0], "model.vision.boi", bid)
            yield from super().modify_tensors(data_torch[1], "model.vision.eoi", bid)
            return

        if name.startswith("audio_encoder.adapting."):
            if ".proj." in name:
                return

        if "conv1.bias" in name or "conv2.bias" in name:
            # transpose conv1 and conv2 bias
            data_torch = data_torch.unsqueeze(-1)

        yield from super().modify_tensors(data_torch, name, bid)


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

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # prevent clash naming with vision tensors
        if name.startswith("multi_modal_projector"):
            name = "audio." + name

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "conv1.bias" in name or "conv2.bias" in name:
            # transpose conv1 and conv2 bias
            data_torch = data_torch.unsqueeze(-1)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("UltravoxModel")
class UltravoxWhisperEncoderModel(WhisperEncoderModel):
    has_vision_encoder = False # no vision encoder
    has_audio_encoder = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.ULTRAVOX)
        self.gguf_writer.add_audio_stack_factor(self.global_config["stack_factor"])


@ModelBase.register("MERaLiON2ForConditionalGeneration")
class MERaLiONWhisperEncoderModel(WhisperEncoderModel):
    has_vision_encoder = False
    has_audio_encoder = True

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("speech_config")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MERALION)
        self.gguf_writer.add_audio_stack_factor(self.global_config.get("speech_mlp_scale_factor", 15))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("text_decoder."):
            return None

        if name.startswith("speech_encoder."):
            name = name.replace("speech_encoder.", "audio_tower.")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        suffix = "." + name.rsplit(".", 1)[-1]

        if name.startswith("ln_speech."):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MM_NORM_PRE, suffix=suffix), data_torch)
            return

        if name.startswith("speech_audio_adapter."):
            if ".mlp_adapter.0." in name:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 0, suffix=suffix), data_torch)
            elif ".gate_proj." in name:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 1, suffix=suffix), data_torch)
            elif ".pool_proj." in name:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 2, suffix=suffix), data_torch)
            elif ".out_proj." in name:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_MMPROJ, 3, suffix=suffix), data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("VoxtralForConditionalGeneration")
class VoxtralWhisperEncoderModel(WhisperEncoderModel):
    has_vision_encoder = False # no vision encoder
    has_audio_encoder = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.VOXTRAL)
        self.gguf_writer.add_audio_stack_factor(4) # == intermediate_size // hidden_size


@ModelBase.register("AudioFlamingo3ForConditionalGeneration")
class AudioFlamingo3WhisperEncoderModel(WhisperEncoderModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MUSIC_FLAMINGO)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".conv" in name and ".weight" in name:
            # Was trained in BF16, being safe, avoiding quantizing to FP16
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
