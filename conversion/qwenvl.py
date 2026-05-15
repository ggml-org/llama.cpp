from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf


@ModelBase.register(
    "Qwen2VLModel",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5OmniModel",
)
class Qwen2VLModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2VL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("thinker."):
            name = name.replace("thinker.", "")

        return super().filter_tensors((name, gen))


@ModelBase.register("Qwen2VLModel", "Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration")
class Qwen2VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = self.hparams_vision.get("image_size", 560)
        # rename config.json values
        self.hparams_vision["num_attention_heads"] = self.hparams_vision.get("num_heads")
        self.hparams_vision["num_hidden_layers"] = self.hparams_vision.get("depth")
        if "embed_dim" in self.hparams_vision: # qwen2vl
            self.hparams_vision["intermediate_size"] = self.hparams_vision.get("hidden_size")
            self.hparams_vision["hidden_size"] = self.hparams_vision.get("embed_dim")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        hparams = self.hparams_vision
        model_type = self.global_config['model_type']
        if model_type == 'qwen2_vl':
            self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN2VL)
        elif model_type == 'qwen2_5_vl' or model_type == 'qwen2_5_omni':
            if model_type == 'qwen2_5_omni':
                self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN25O)
            else:
                self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN25VL)
            self.gguf_writer.add_vision_use_silu(True)
            # find n_wa_pattern (window attention pattern)
            fullatt_block_indexes = hparams.get("fullatt_block_indexes")
            assert fullatt_block_indexes is not None, "fullatt_block_indexes is required for qwen2_5_vl"
            n_wa_pattern = fullatt_block_indexes[0] + 1
            # validate n_wa_pattern
            for i in range(1, len(fullatt_block_indexes)):
                if fullatt_block_indexes[i] - fullatt_block_indexes[i - 1] != n_wa_pattern:
                    raise ValueError(f"Invalid fullatt_block_indexes: {fullatt_block_indexes}")
            self.gguf_writer.add_vision_n_wa_pattern(n_wa_pattern)
        else:
            raise ValueError(f"Unknown QwenVL model type: {self.global_config['model_type']}")
        # default values below are taken from HF tranformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(self.global_config.get("rms_norm_eps", 1e-6))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".position_embd." in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("visual."):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # split QKV tensors if needed
        if ".qkv." in name:
            if data_torch.ndim == 2: # weight
                c3, _ = data_torch.shape
            else: # bias
                c3 = data_torch.shape[0]
            assert c3 % 3 == 0
            c = c3 // 3
            wq = data_torch[:c]
            wk = data_torch[c: c * 2]
            wv = data_torch[c * 2:]
            yield from super().modify_tensors(wq, name.replace("qkv", "q"), bid)
            yield from super().modify_tensors(wk, name.replace("qkv", "k"), bid)
            yield from super().modify_tensors(wv, name.replace("qkv", "v"), bid)
        elif 'patch_embed.proj.weight' in name:
            # split Conv3D into Conv2Ds
            c1, c2, kt, kh, kw = data_torch.shape
            del c1, c2, kh, kw  # unused
            assert kt == 2, "Current implementation only support temporal_patch_size of 2"
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight"  , data_torch[:, :, 0, ...])
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight.1", data_torch[:, :, 1, ...])
        else:
            yield from super().modify_tensors(data_torch, name, bid)


class Qwen25AudioModel(MmprojModel):
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_audio is not None
        self.hparams_audio["hidden_size"] = self.hparams_audio["d_model"]
        self.hparams_audio["intermediate_size"] = self.hparams_audio["encoder_ffn_dim"]
        self.hparams_audio["num_attention_heads"] = self.hparams_audio["encoder_attention_heads"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_audio is not None
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams_audio.get("layer_norm_eps", 1e-5))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # SinusoidsPositionEmbedding
        assert self.hparams_audio is not None
        max_timescale = 10000
        length = 1500
        channels = self.hparams_audio["hidden_size"]
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        pos_embd = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).to(dtype=torch.float32)
        yield ("audio_tower.embed_positions.weight", pos_embd)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "conv1.bias" in name or "conv2.bias" in name:
            # transpose conv1 and conv2 bias
            data_torch = data_torch.unsqueeze(-1)

        yield from MmprojModel.modify_tensors(self, data_torch, name, bid)


@ModelBase.register("Qwen2_5OmniModel")
class Qwen25OmniModel(Qwen2VLVisionModel, Qwen25AudioModel):
    has_audio_encoder = True
    has_vision_encoder = True

    def get_vision_config(self) -> dict[str, Any] | None:
        return self.global_config["thinker_config"].get("vision_config")

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config["thinker_config"].get("audio_config")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.QWEN25O)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("visual.") and not name.startswith("audio_tower."):
            return None

        if name.startswith("thinker."):
            name = name.replace("thinker.", "")

        if "audio_bos_eos_token" in name:
            # this tensor is left unused in transformers code
            # https://github.com/huggingface/transformers/blob/6e3063422c4b1c014aa60c32b9254fd2902f0f28/src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py#L1809
            return None

        return MmprojModel.filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "visual." in name:
            yield from Qwen2VLVisionModel.modify_tensors(self, data_torch, name, bid)
        elif "audio_tower." in name:
            yield from Qwen25AudioModel.modify_tensors(self, data_torch, name, bid)
        return  # skip other tensors
