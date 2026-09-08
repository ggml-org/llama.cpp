# Copyright (c) 2026 codec.cpp contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
from pathlib import Path
import re
import tarfile

import torch

from .base import ModelBase, MmprojModel, gguf


_ARCHIVE = "nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo"


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    for member in archive.getmembers():
        if member.name.removeprefix("./") == name and member.isfile():
            with archive.extractfile(member) as f:
                return f.read()
    raise ValueError(f"NeMo archive is missing {name}")


@ModelBase.register_hparams_loader(lambda path: (path / _ARCHIVE).is_file())
def _load_hparams(path: Path) -> dict:
    import yaml

    with tarfile.open(path / _ARCHIVE) as archive:
        config = yaml.safe_load(_read_member(archive, "model_config.yaml"))
    decoder = config["audio_decoder"]
    quantizer = config["vector_quantizer"]
    expected = {
        "up_sample_rates": [7, 7, 6, 3, 2], "input_dim": 16, "base_channels": 864,
        "activation": "half_snake", "output_activation": "clamp", "pad_mode": "zeros",
        "n_groups_equal_to_out_channels": True,
    }
    defaults = {"in_kernel_size": 7, "out_kernel_size": 3,
                "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [1, 3, 5]}
    if (not decoder.get("_target_", "").endswith(".CausalHiFiGANDecoder")
            or any(decoder.get(k) != v for k, v in expected.items())
            or any(decoder.get(k, v) != v for k, v in defaults.items())
            or not quantizer.get("_target_", "").endswith(".GroupFiniteScalarQuantizer")
            or config.get("sample_rate") != 22050 or config.get("samples_per_frame") != 1764
            or quantizer.get("num_groups") != 4 or quantizer.get("num_levels_per_group") != [9, 8, 8, 7]):
        raise ValueError("Only NeMo Nano Codec 22 kHz / 12.5 fps is supported")
    return {
        "architectures": ["NemoNanoCodecModel"], "hidden_size": 16,
        "audio_config": {"num_hidden_layers": 5},
    }


@ModelBase.register("NemoNanoCodecModel")
@ModelBase.example("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")
class NemoNanoCodecModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.NEMO_NANO_CODEC)
        self.gguf_writer.add_gen_audio_projection_dim(16)
        self.gguf_writer.add_gen_audio_embedding_length(864)
        self.gguf_writer.add_gen_audio_feed_forward_length(864)
        self.gguf_writer.add_gen_audio_block_count(5)
        self.gguf_writer.add_gen_audio_head_count(1)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-5)

    def get_tensors(self):
        with tarfile.open(self.dir_model / _ARCHIVE) as archive:
            state = torch.load(io.BytesIO(_read_member(archive, "model_weights.ckpt")), map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        for name, value in state.items():
            if not name.startswith("audio_decoder.") or name.endswith(".weight_v"):
                continue
            if name.endswith(".weight_g"):
                weight = state[name.removesuffix("weight_g") + "weight_v"].float()
                norm = torch.linalg.vector_norm(weight.flatten(1), dim=1).reshape(-1, 1, 1)
                value = weight * (value.float().reshape(-1, 1, 1) / norm)
                name = name.removesuffix("weight_g") + "weight"
            yield name, value

        # All four FSQ groups use the same fixed codebook.
        levels = torch.tensor([9, 8, 8, 7])
        bases = torch.tensor([1, 9, 72, 576])
        scale = levels // 2
        codes = (torch.arange(4032)[:, None] // bases) % levels
        yield "fsq_codebook", (codes - scale).float() / scale

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if new_name.endswith(".alpha") or name == "fsq_codebook":
            return gguf.GGMLQuantizationType.F32
        # Convolution uses the standard ggml F16 im2col path.
        if new_name.endswith(".weight") and n_dims == 3:
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch, name, bid):
        T = gguf.MODEL_TENSOR
        suffix = "." + name.rsplit(".", 1)[-1]
        name = name.removeprefix("audio_decoder.")
        if name == "fsq_codebook":
            yield self.format_tensor_name(T.A_GEN_WAV_HIFIGAN_CODEBOOK), data_torch
            return
        if name.startswith("pre_conv."):
            tensor = T.A_GEN_WAV_HIFIGAN_PRE
        elif name.startswith("post_conv."):
            tensor = T.A_GEN_WAV_HIFIGAN_POST
        elif name.startswith("post_activation."):
            tensor = T.A_GEN_WAV_HIFIGAN_POST_ACT
        elif match := re.fullmatch(r"activations\.(\d+)\.activation\.snake_act\.alpha", name):
            tensor, bid = T.A_GEN_WAV_HIFIGAN_UP_ACT, int(match[1])
        elif match := re.fullmatch(r"up_sample_conv_layers\.(\d+)\.conv\.(weight|bias)", name):
            tensor, bid = T.A_GEN_WAV_HIFIGAN_UP, int(match[1])
            if suffix == ".weight":
                # Grouped transposed convolution: [IC, 1, K] -> [OC, K, 2].
                data_torch = data_torch.reshape(-1, 2, data_torch.shape[-1]).transpose(1, 2).contiguous()
        elif match := re.fullmatch(r"res_layers\.(\d+)\.res_blocks\.(\d+)\.res_blocks\.(\d+)\.(input_conv|skip_conv|input_activation|skip_activation)\..+", name):
            bid = int(match[1]) * 9 + int(match[2]) * 3 + int(match[3])
            tensor = {
                "input_conv": T.A_GEN_WAV_HIFIGAN_RES_CONV1,
                "skip_conv": T.A_GEN_WAV_HIFIGAN_RES_CONV2,
                "input_activation": T.A_GEN_WAV_HIFIGAN_RES_ACT1,
                "skip_activation": T.A_GEN_WAV_HIFIGAN_RES_ACT2,
            }[match[4]]
        else:
            raise ValueError(f"Unexpected NeMo decoder tensor: {name}")
        if suffix == ".alpha":
            data_torch = data_torch.flatten()
        yield self.format_tensor_name(tensor, bid, suffix), data_torch
