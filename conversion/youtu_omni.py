from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


def _fusion_pattern(freq, n_layer: int) -> list[int]:
    """Match YoutuVITAOmniConfig.video_fusion_layer_freq.

    None is all fusion. An int N marks layer i when ``i % N == 0``.
    A list must be 0/1 and the same length as the encoder.
    """
    if freq is None:
        return [1] * n_layer
    if isinstance(freq, bool):
        raise ValueError(f"video_fusion_layer_freq must be int / list / None, got bool {freq!r}")
    if isinstance(freq, int):
        n = max(int(freq), 1)
        return [1 if i % n == 0 else 0 for i in range(n_layer)]
    if isinstance(freq, (list, tuple)):
        if len(freq) != n_layer:
            raise ValueError(
                f"video_fusion_layer_freq length {len(freq)} != num_hidden_layers {n_layer}"
            )
        pattern = []
        for value in freq:
            if value not in (0, 1, True, False):
                raise ValueError(f"video_fusion_layer_freq entries must be 0/1, got {value!r}")
            pattern.append(1 if value else 0)
        return pattern
    raise ValueError(
        f"video_fusion_layer_freq must be int / list / None, got {type(freq).__name__}"
    )


@ModelBase.register("YoutuVITAForCausalLM")
@ModelBase.example("tencent/Youtu-Parsing-Omni")
class YoutuOmniVisionModel(MmprojModel):
    """Vision and audio towers for Youtu-Parsing-Omni.

    Both modalities share one bidirectional transformer stored under
    ``omni_config``. The file keeps a single copy of that transformer under
    the vision tensor names. The audio context loads those same tensors and
    adds the conv stem plus the audio merger.
    """

    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # Native resolution is dynamic. image_size is only the metadata fallback.
        self.hparams_vision["image_size"] = self.preprocessor_config.get("image_size", 448)

    def get_vision_config(self):
        return self.global_config.get("omni_config")

    def get_audio_config(self):
        # audio_config is JSON null; the conv stem and the shared transformer
        # both read omni_config.
        return self.global_config.get("omni_config")

    def set_gguf_parameters(self):
        mean = self.preprocessor_config.get("mean")
        std = self.preprocessor_config.get("std")
        if mean is not None and "image_mean" not in self.preprocessor_config:
            self.preprocessor_config["image_mean"] = mean
        if std is not None and "image_std" not in self.preprocessor_config:
            self.preprocessor_config["image_std"] = std

        super().set_gguf_parameters()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.YOUTU_OMNI)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_vision_use_silu(True)
        self.gguf_writer.add_vision_spatial_merge_size(self.hparams.get("spatial_merge_size", 2))
        self.gguf_writer.add_vision_head_count_kv(self.hparams["num_key_value_heads"])
        self.gguf_writer.add_vision_head_dim(self.hparams.get("head_dim", 128))

        min_pixels = self.preprocessor_config.get("min_pixels")
        max_pixels = self.preprocessor_config.get("max_pixels")
        if min_pixels is not None:
            self.gguf_writer.add_vision_min_pixels(int(min_pixels))
        if max_pixels is not None:
            self.gguf_writer.add_vision_max_pixels(int(max_pixels))
        self.gguf_writer.add_vision_image_resize_algo("bicubic")

        self.gguf_writer.add_audio_num_mel_bins(int(self.hparams.get("num_mel_bins", 128)))
        self.gguf_writer.add_audio_head_dim(int(self.hparams.get("head_dim", 128)))
        self.gguf_writer.add_audio_attention_layernorm_eps(float(self.hparams.get("rms_norm_eps", 1e-6)))
        self.gguf_writer.add_audio_head_count_kv(int(self.hparams["num_key_value_heads"]))

        n_layer = int(self.hparams["num_hidden_layers"])
        freq = self.hparams.get("video_fusion_layer_freq")
        self.gguf_writer.add_array(gguf.Keys.ClipVision.FUSION_LAYER, _fusion_pattern(freq, n_layer))
        self.gguf_writer.add_bool(
            gguf.Keys.ClipVision.VIDEO_GROUP_ATTENTION,
            bool(self.hparams.get("video_group_attention", True)),
        )

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]
        if not (
            name.startswith("model.omni_model.vision_embeddings.")
            or name.startswith("model.omni_model.encoder.")
            or name.startswith("model.omni_model.vision_merger.")
            or name.startswith("model.omni_model.audio_embeddings.")
            or name.startswith("model.omni_model.audio_merger.")
        ):
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Linear(C*P*P, hidden) stored as a conv kernel. ggml reads ne[0] as KW.
        if name.endswith("vision_embeddings.patch_embedding.weight"):
            out_features, in_features = data_torch.shape
            patch_size = int(self.hparams["patch_size"])
            channels = int(self.hparams.get("num_channels", 3))
            expected = channels * patch_size * patch_size
            if in_features != expected:
                raise ValueError(f"Unexpected patch embedding shape {tuple(data_torch.shape)}, expected in_features={expected}")
            data_torch = data_torch.reshape(out_features, channels, patch_size, patch_size).contiguous()

        if "audio_embeddings.conv2d" in name and name.endswith(".bias"):
            # transform conv2d bias [n_embd] --> [1, 1, n_embd]
            data_torch = data_torch.unsqueeze(-1).unsqueeze(-1)

        yield from super().modify_tensors(data_torch, name, bid)
