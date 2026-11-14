from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("InklingForConditionalGeneration")
class InklingAudioModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    def get_audio_config(self) -> dict[str, Any] | None:
        config = self.global_config.get("audio_config")
        if not config:
            return None
        if config.get("audio_mode") != "dmel" or not config.get("use_audio_norm") or (config["dmel_min_value"], config["dmel_max_value"]) != (-7.0, 2.0):
            raise ValueError("Inkling audio requires normalized dMel input in the range [-7, 2]")
        # dMel has no transformer
        return {**config, "hidden_size": config["decoder_dmodel"], "intermediate_size": 0, "num_hidden_layers": 0, "num_attention_heads": 1}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_audio is not None
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.INKLINGA)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["n_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-6)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith("model.audio."):
            return None
        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the embedding lookup is a matmul on one-hot bins
        if name == "model.audio.encoder.weight":
            data_torch = data_torch.T
        yield from super().modify_tensors(data_torch, name, bid)
