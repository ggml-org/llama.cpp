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


@ModelBase.register("KaniTTS2ForCausalLM")
class KaniTTS2MmprojModel(NemoNanoCodecModel):
    has_audio_encoder = True

    def get_audio_config(self):
        import json
        path = self.dir_model / "speaker_encoder" / "config.json"
        with path.open() as f:
            config = json.load(f)
        expected = {"model_type": "wavlm", "hidden_size": 1024, "num_hidden_layers": 24,
                    "num_attention_heads": 16, "intermediate_size": 4096, "do_stable_layer_norm": True,
                    "feat_extract_norm": "layer", "layer_norm_eps": 1e-5, "conv_bias": False, "embd_size": 128,
                    "top_interm_size": 512, "num_conv_pos_embeddings": 128,
                    "num_conv_pos_embedding_groups": 16, "num_buckets": 320, "max_bucket_distance": 800,
                    "conv_dim": [512] * 7, "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                    "conv_stride": [5, 2, 2, 2, 2, 2, 2], "hidden_act": "gelu", "feat_extract_activation": "gelu"}
        if any(config.get(k) != v for k, v in expected.items()) or config.get("add_adapter"):
            raise ValueError("Unsupported Kani WavLM speaker encoder configuration")
        return config

    def set_gguf_parameters(self):
        _load_hparams(self.dir_model)  # validate the matching NeMo archive
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.KANI_SPKENC)
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_audio_embedding_length(1024)
        self.gguf_writer.add_audio_feed_forward_length(4096)
        self.gguf_writer.add_audio_block_count(24)
        self.gguf_writer.add_audio_head_count(16)
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams_audio["layer_norm_eps"])
        self.gguf_writer.add_audio_num_mel_bins(1)

    def get_tensors(self):
        from safetensors.torch import load_file
        yield from super().get_tensors()
        speaker = self.dir_model / "speaker_encoder"
        if (speaker / "model.safetensors").is_file():
            state = load_file(speaker / "model.safetensors")
        else:
            state = torch.load(speaker / "pytorch_model.bin", map_location="cpu", weights_only=True)
        pos = "wavlm.encoder.pos_conv_embed.conv."
        wv = state.get(pos + "parametrizations.weight.original1", state.get(pos + "weight_v"))
        wg = state.get(pos + "parametrizations.weight.original0", state.get(pos + "weight_g"))
        if wv is None or wg is None:
            raise ValueError("Missing WavLM positional convolution weight normalization")
        weight = wv.float() * wg.float() / torch.linalg.vector_norm(wv.float(), dim=(0, 1), keepdim=True)
        yield "a.spk.pos.weight", weight.reshape(16, 64, 64, 128).contiguous()
        yield "a.spk.pos.bias", state[pos + "bias"]
        for name, value in state.items():
            if name.startswith(pos) or name == "wavlm.masked_spec_embed" or name.endswith("num_batches_tracked"):
                continue
            name = name.replace("wavlm.feature_extractor.conv_layers.", "a.spk.conv.")
            name = name.replace("wavlm.feature_projection.", "a.spk.feature.")
            name = name.replace("wavlm.encoder.layers.", "a.spk.blk.")
            name = name.replace("wavlm.encoder.layer_norm.", "a.spk.output_norm.")
            name = name.replace("top_layers.", "a.spk.top.")
            if not name.startswith("a.spk."):
                raise ValueError(f"Unexpected speaker encoder tensor: {name}")
            if "gru_rel_pos_linear" in name:
                value = value.float().reshape(2, 4, -1).sum(1)
                if name.endswith("bias"):
                    value = value.flatten()
            elif "gru_rel_pos_const" in name:
                value = value.reshape(16, 1, 1)
            elif ".affine" in name and name.endswith("weight"):
                value = value.squeeze(-1)
            if ".batchnorm" in name:
                if name.endswith("running_var"):
                    continue
                if not name.endswith("running_mean"):
                    raise ValueError(f"Unexpected speaker batch norm tensor: {name}")
                key = name.replace("a.spk.top.", "top_layers.").replace("running_mean", "running_var")
                scale = torch.rsqrt(state[key].float() + 1e-3)
                yield name.replace("running_mean", "weight"), scale
                yield name.replace("running_mean", "bias"), -value.float() * scale
                continue
            yield name, value
        from safetensors import safe_open
        with safe_open(self.dir_model / "model.safetensors", framework="pt", device="cpu") as backbone:
            yield "a.spk.projection.weight", backbone.get_tensor("model.speaker_emb_projection.weight")

    def modify_tensors(self, data_torch, name, bid):
        if not name.startswith("a.spk."):
            yield from super().modify_tensors(data_torch, name, bid)
            return
        T = gguf.MODEL_TENSOR
        name = name.removeprefix("a.spk.")
        suffix = name.rsplit(".", 1)[-1]
        stem = name.rsplit(".", 1)[0]
        direct = {"pos": T.A_ENC_POSITION_CONV, "feature.layer_norm": T.A_PRE_NORM,
                  "feature.projection": T.A_ENC_INP_PROJ, "output_norm": T.A_POST_NORM,
                  "projection": T.A_ENC_SPEAKER_PROJ}
        if stem in direct:
            tensor = direct[stem]
        elif match := re.fullmatch(r"conv\.(\d+)\.(conv|layer_norm)", stem):
            bid = int(match[1])
            tensor = T.A_ENC_CONV1D if match[2] == "conv" else T.A_ENC_CONV1D_NORM
        elif match := re.fullmatch(r"top\.(affine|batchnorm)([12])", stem):
            bid = int(match[2]) - 1
            tensor = T.A_ENC_SPK_FC if match[1] == "affine" else T.A_ENC_SPK_FC_NORM
        elif match := re.fullmatch(r"blk\.(\d+)\.(.+)", name):
            bid = int(match[1])
            tail = match[2]
            if tail == "attention.gru_rel_pos_const":
                tensor, suffix = T.A_ENC_ATTN_REL_GATE_CONST, "weight"
            else:
                tensor = {"layer_norm": T.A_ENC_INPUT_NORM, "final_layer_norm": T.A_ENC_FFN_NORM,
                          "attention.q_proj": T.A_ENC_ATTN_Q, "attention.k_proj": T.A_ENC_ATTN_K,
                          "attention.v_proj": T.A_ENC_ATTN_V, "attention.out_proj": T.A_ENC_OUTPUT,
                          "attention.gru_rel_pos_linear": T.A_ENC_ATTN_REL_GATE,
                          "attention.rel_attn_embed": T.A_ENC_ATTN_REL_POS_EMB,
                          "feed_forward.intermediate_dense": T.A_ENC_FFN_UP,
                          "feed_forward.output_dense": T.A_ENC_FFN_DOWN}[tail.rsplit(".", 1)[0]]
        else:
            raise ValueError(f"Unexpected speaker tensor: {name}")
        yield self.format_tensor_name(tensor, bid, "." + suffix), data_torch

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if name.startswith("a.spk."):
            if ".conv.weight" in name or name == "a.spk.pos.weight":
                return gguf.GGMLQuantizationType.F16
            if "gru_rel_pos" in name or "rel_attn_embed" in name:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
