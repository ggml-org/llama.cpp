from __future__ import annotations

import json
import math

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger
from .lfm2 import D1Model, lfm2_feed_forward_length


@ModelBase.register("Lfm3ForCausalLM", "Lfm3VlForConditionalGeneration", "Lfm3OmniForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM3-1.7B", "LiquidAI/LFM3-1.7B-Base")
class LFM3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.LFM3

    # LFM3-Audio/Omni: encoder, projector and audio-output heads live next to the text model
    _media_prefixes = ("encoder.", "mlp.", "codebook_embedding.", "depth_embeddings.", "depth_linear.", "depthformer.",
                       "mask_embedding", "model.codebook_embedding.", "model.depth_decoder.")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(cls._media_prefixes):
            return None

        name = name.removeprefix("lfm.")

        return super().filter_tensors((name, gen))

    # LFM3-Audio live ASR prompt of the model card, the audio follows in the assistant turn
    _asr_template = (
        "{{- bos_token -}}"
        "{{- '<|im_start|>system\\nPerform live ASR.<|im_end|>\\n' -}}"
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
    )

    # TTS prompt of the model card: a language code is looked up, anything else is taken as the language name
    _tts_langs = {
        "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "American English", "en-us": "American English",
        "en-gb": "British English", "fi": "Finnish", "fr": "French", "he": "Hebrew", "it": "Italian", "ja": "Japanese",
        "ko": "Korean", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "es": "Spanish", "sv": "Swedish", "tr": "Turkish",
        "zh": "Chinese",
    }

    # "language" and "voice" are optional, the text to speak is the first message
    def _tts_template(self, think: bool) -> str:
        return (
            "{%- set names = " + json.dumps(self._tts_langs) + " -%}"
            "{%- set code = (language | lower) if language is defined and language else 'en' -%}"
            "{%- set lang = names[code] if code in names else language -%}"
            "{%- set v = (voice | lower) if voice is defined and voice and (voice | lower) in ['female', 'male'] else 'female' -%}"
            "{{- bos_token -}}"
            "{{- '<|im_start|>system\\nPerform TTS in ' + lang + ' using a ' + v + ' voice.<|im_end|>\\n' -}}"
            "{{- '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}"
            "{{- '<|im_start|>assistant\\n" + ("<think></think>" if think else "") + "<|audio_start|>' -}}"
        )

    def set_vocab(self):
        super().set_vocab()
        # hparams hold the merged text config, the checkpoint's own model_type is in config.json
        with open(self.dir_model / "config.json", encoding="utf-8") as f:
            model_type = json.load(f).get("model_type")
        is_audio = model_type == "lfm3_audio"
        is_omni  = model_type == "lfm3_omni"
        templates = []
        if is_audio:
            templates.append({"name": "asr", "template": self._asr_template})
        if is_audio or is_omni:
            # LFM3-Omni opens every assistant turn with an empty think block
            templates.append({"name": "tts", "template": self._tts_template(think=is_omni)})
        if templates:
            self.gguf_writer.add_chat_template(templates)

    def set_gguf_parameters(self):
        layer_types = self.hparams["layer_types"]
        n_kv_heads = self.hparams["num_key_value_heads"]

        # set num_key_value_heads only for attention layers
        self.hparams["num_key_value_heads"] = [
            n_kv_heads if lt != "conv" else 0
            for lt in layer_types
        ]

        super().set_gguf_parameters()

        self.gguf_writer.add_rope_dimension_count(self.hparams["head_dim"])

        self.gguf_writer.add_sliding_window_pattern([lt == "sliding_attention" for lt in layer_types])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])

        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # Partial RoPE is represented by masking trailing frequency pairs.
        # rope_parameters is keyed by layer type; only full_attention layers use partial RoPE.
        rope_parameters = self.hparams["rope_parameters"]["full_attention"]
        partial = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = self.hparams["head_dim"]
        rope_angles = int(partial * head_dim) // 2
        n_pairs = head_dim // 2
        assert 0 < rope_angles <= n_pairs
        values = [1.0] * rope_angles + [1e30] * (n_pairs - rope_angles)
        rope_freqs = torch.tensor(values, dtype=torch.float32)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), rope_freqs)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # conv op requires 2d tensor
        if "conv.conv" in name:
            data_torch = data_torch.squeeze(1)

        yield from super().modify_tensors(data_torch, name, bid)


# d1 on LFM3-VL / LFM3-Omni: the stock weights, the d1 prompt read at its last token
_D1_TYPES = {"lfm3_vl": gguf.DecisionType.LFM3_D1, "lfm3_omni": gguf.DecisionType.LFM3_D1_OMNI}


def _is_lfm3_d1_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("architectures") == ["D1Model"] and config.get("model_type") in _D1_TYPES


@ModelBase.register_hparams_loader(_is_lfm3_d1_checkpoint)
def _load_lfm3_d1_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected LFM3 d1 checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["d1_family"] = hparams["model_type"]
    # the mmproj is the base model's, the text model writes the decision
    hparams["architectures"] = ["Lfm3VlForConditionalGeneration" if hparams["model_type"] == "lfm3_vl" else "Lfm3OmniForConditionalGeneration"]
    hparams["text_config"]["architectures"] = ["Lfm3D1Model"]
    return hparams


@ModelBase.register("Lfm3D1Model")
@ModelBase.example("LiquidAI/d1-2B", "LiquidAI/d1-omni-2B")
class LFM3D1Model(LFM3Model):
    model_arch = gguf.MODEL_ARCH.LFM3

    # how the checkpoint serves, hard-coded in the server: assert the checkpoint records the same
    _served = {"d1_codes": 2, "d1_attention": "set"}

    def set_vocab(self):
        # the vocab without the speech task templates of LFM3-Audio/Omni, d1 only decides
        TextModel.set_vocab(self)
        # the d1 prompt of LFM2, the LFM3 template writes an empty thought before the answer
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": D1Model._systemone_template() + "<think></think>"}])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        for key, value in self._served.items():
            if self.hparams.get(key) != value:
                raise ValueError(f"unsupported d1 checkpoint: {key} = {self.hparams.get(key)!r}, expected {value!r}")

        # a lone question reads the whole prompt both ways on the full attention layers
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_decision_type(_D1_TYPES[self.hparams["d1_family"]])

        # the option answers share one temperature
        with open(self.dir_model / "calibration.json", encoding="utf-8") as f:
            calibration = json.load(f)
        if calibration.get("temperatures"):
            raise ValueError("unsupported d1 calibration: per-bucket temperatures")
        for name in ("noul", "choice", "score"):
            self.gguf_writer.add_decision_temperature(name, calibration["default"])


class LFM3VLVisionMixin(MmprojModel):
    """LFM3-VL vision tower and projector, shared with LFM3-Omni."""

    def _set_lfm3vl_gguf_parameters(self):
        assert self.hparams_vision is not None
        # clip.cpp hardcodes the GELU projector and its input norm eps
        if (self.global_config["projector_hidden_act"] != "gelu" or not self.global_config["projector_bias"]
                or self.global_config["projector_layer_norm_eps"] != 1e-6):
            raise ValueError("unsupported projector_hidden_act / projector_bias / projector_layer_norm_eps")
        # clip.cpp hardcodes the vision rope base, the RGB patchifier and the 3x3 spatial conv
        if (self.hparams_vision["rope_parameters"]["rope_theta"] != 1e6 or self.hparams_vision["in_channels"] != 3
                or self.hparams_vision["conv_kernel_size"] != 3):
            raise ValueError("unsupported vision rope_theta / in_channels / conv_kernel_size")
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.LFM3VL)
        self.gguf_writer.add_vision_head_count_kv(self.hparams_vision["num_key_value_heads"])
        self.gguf_writer.add_vision_spatial_merge_size(self.global_config["downsample_factor"])
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision["norm_eps"])
        self.gguf_writer.add_vision_use_silu(True)
        size = self.preprocessor_config["size"]
        self.gguf_writer.add_vision_min_pixels(size["shortest_edge"])
        self.gguf_writer.add_vision_max_pixels(size["longest_edge"])
        # PIL resample enum, the torchvision backend has no lanczos
        resample = self.preprocessor_config["resample"]
        resize_algo = {2: "bilinear", 3: "bicubic"}.get(resample)
        if resize_algo is None:
            raise ValueError(f"unsupported resample: {resample}")
        self.gguf_writer.add_vision_image_resize_algo(resize_algo)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if "patch_embd" in new_name:
            return gguf.GGMLQuantizationType.F16
        if "conv2d_spatial" in new_name:
            # CUDA CONV_2D_DW requires an f32 kernel; the tensor is tiny
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _lfm3vl_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.vision_tower.patchifier.weight":
            # qwen2vl-style layout [channel, temporal, ph, pw], one conv kernel per temporal slice
            assert self.hparams_vision is not None
            t = self.hparams_vision["temporal_patch_size"]
            p = self.hparams_vision["patch_size"]
            n_embd = data_torch.shape[0]
            w = data_torch.view(n_embd, 3, t, p, p)
            yield ("v.patch_embd.weight", w[:, :, 0].contiguous())
            if t == 2:
                yield ("v.patch_embd.weight.1", w[:, :, 1].contiguous())
            else:
                assert t == 1, f"unsupported vision_temporal_patch_size {t}"
            return
        if name == "model.vision_tower.embedding_norm.weight":
            yield ("v.embedding_norm.weight", data_torch)
            return
        if "conv.in_proj.weight" in name:
            yield (f"v.blk.{bid}.conv2d_in_proj.weight", data_torch)
            return
        if "conv.spatial_conv.weight" in name:
            yield (f"v.blk.{bid}.conv2d_spatial.weight", data_torch)
            return
        if "conv.out_proj.weight" in name:
            yield (f"v.blk.{bid}.conv2d_out_proj.weight", data_torch)
            return

        # projector: input LayerNorm -> linear_1 -> GELU -> linear_2
        if name.startswith("model.multi_modal_projector."):
            layer, suffix = name.removeprefix("model.multi_modal_projector.").split(".")
            yield ({"layer_norm": "mm.input_norm", "linear_1": "mm.0", "linear_2": "mm.2"}[layer] + "." + suffix, data_torch)
            return

        yield (self.map_tensor_name(name), data_torch)


@ModelBase.register("Lfm3VlForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM3-VL-2B")
class LFM3VLMmprojModel(LFM3VLVisionMixin):
    has_vision_encoder = True
    has_audio_encoder = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # dynamic resolution, image_size is only set for MmprojModel compatibility
        self.hparams_vision["image_size"] = 512

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._set_lfm3vl_gguf_parameters()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith(("model.vision_tower.", "model.multi_modal_projector.")):
            yield from self._lfm3vl_tensors(data_torch, name, bid)


def _is_lfm3_audio_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("model_type") == "lfm3_audio" and "lfm" in config


# read the release config.json as is: a transformers that knows lfm3_audio rewrites it into another layout
@ModelBase.register_hparams_loader(_is_lfm3_audio_checkpoint)
def _load_lfm3_audio_hparams(dir_model: Path) -> dict[str, Any]:
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    config["text_config"] = config["lfm"]
    return config


@ModelBase.register("Lfm3AudioForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM3-Audio-2B")
class LFM3AudioMmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    # the downsampler has the LFM2-Audio pre-encoder layout: conv, dw conv, pw conv, dw conv, pw conv
    _downsample_idx = {"conv_in": 0, "conv_mid": 2, "lin_mid": 3, "conv_end": 5, "lin_end": 6}

    # audio output: MaskGIT depthformer (a.gen.code.*) and the codec decoder (a.gen.wav.*)
    _depth_layer_map = {
        "operator_norm":      gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM,
        "self_attn.q_proj":   gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q,
        "self_attn.q_norm":   gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q_NORM,
        "self_attn.k_proj":   gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K,
        "self_attn.k_norm":   gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K_NORM,
        "self_attn.v_proj":   gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_V,
        "self_attn.o_proj":   gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT,
        "self_attn.gate_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_GATE,
        "ffn_norm":           gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM,
        "feed_forward.w1":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE,
        "feed_forward.w3":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP,
        "feed_forward.w2":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN,
    }
    _codec_layer_map = {
        "operator_norm":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_NORM,
        "post_operator_norm":    gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_POST_NORM,
        "self_attn.q_proj":      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_Q,
        "self_attn.q_norm":      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_Q_NORM,
        "self_attn.k_proj":      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_K,
        "self_attn.k_norm":      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_K_NORM,
        "self_attn.v_proj":      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_V,
        "self_attn.o_proj":      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_OUT,
        "self_attn.gate_proj":   gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_GATE,
        "conv.in_proj":          gguf.MODEL_TENSOR.A_GEN_WAV_TFM_CONV_PW1,
        "conv.conv":             gguf.MODEL_TENSOR.A_GEN_WAV_TFM_CONV_DW,
        "conv.out_proj":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_CONV_PW2,
        "ffn_norm":              gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_NORM,
        "post_ffn_norm":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_POST_NORM,
        "feed_forward.w1":       gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_GATE,
        "feed_forward.w3":       gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_UP,
        "feed_forward.w2":       gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_DOWN,
    }
    # the codec graph in clip.cpp hardcodes the liquid-codec FSQ front end and decoder shape, conversion fails on a mismatch
    _codec = {
        "model_type": "liquid_codec", "fsq_levels": [8, 7, 6, 6], "n_q": 12, "quantizer_dim": 256,
        "encoder_dim": 512, "stft_n_fft": 1280, "stft_hop_length": 320, "stft_win_length": 1280, "stft_frame_stack": 6,
        "decoder_window": 30, "sampling_rate": 24000,
    }
    # the lfm3a pipeline in mtmd-helper-gen.cpp hardcodes these
    _generation = {"codebooks": 12, "maskgit_schedule": [4, 4, 4], "interleaved_n_text": 6, "interleaved_n_audio": 12}

    # speech-code sampling of the model card, generation_config.json overrides it
    _audio_sampling = {
        "audio_temperature": 1.0, "audio_top_k": 128, "audio_repetition_penalty": 1.1, "audio_repetition_window": 4096,
    }

    # mtmd_audio_preprocessor_lfm3a hardcodes this front end, conversion fails on a mismatch
    _feature_extractor = {
        "sampling_rate": 16000, "feature_size": 128, "n_fft": 400, "win_length": 400, "hop_length": 160,
        "log_floor": -12.0, "normalization_offset": 4.0, "normalization_divisor": 4.0, "encoder_stride": 8,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # audio output is exported when the checkpoint ships the codec
        self._audio_output = (self.dir_model / "audio_tokenizer").is_dir()
        gen = self.get_generation_config()
        self._n_codebooks: int = gen["codebooks"]
        self._audio_vocab: int = gen["audio_vocab_size"]
        self._depth_embd: dict[int, Tensor] = {}
        self._depth_norm: dict[int, Tensor] = {}

    def get_audio_config(self) -> dict[str, Any] | None:
        enc = self.global_config["encoder"]
        cfg = dict(enc["lfm"])
        cfg["num_mel_bins"] = enc["dim_in"]
        cfg["sliding_window"] = enc["sliding_window"]
        return cfg

    def get_generation_config(self) -> dict[str, Any]:
        c = self.global_config
        steps = c["maskgit_steps"]
        return {
            "codebooks": c["codebooks"], "audio_vocab_size": c["audio_vocab_size"],
            "maskgit_schedule": [c["codebooks"] // steps] * steps,
            "interleaved_n_text": c["interleaved_n_text"], "interleaved_n_audio": c["interleaved_n_audio"],
        }

    def get_depth_config(self) -> dict[str, Any]:
        return self.global_config["depthformer"]

    def set_gguf_parameters(self):
        assert self.hparams_audio is not None
        fe = self.preprocessor_config["feature_extractor"]
        if fe["feature_extractor_type"] not in ("Lfm3AudioFeatureExtractor", "Lfm3OmniFeatureExtractor"):
            raise ValueError(f"unsupported feature_extractor_type {fe['feature_extractor_type']!r}")
        for k, v in self._feature_extractor.items():
            if fe[k] != v:
                raise ValueError(f"unsupported feature_extractor.{k}={fe[k]!r}, expected {v!r}")

        # clip.cpp hardcodes the encoder rope base
        if self.hparams_audio["rope_parameters"]["rope_theta"] != 1e6:
            raise ValueError("unsupported audio encoder rope_theta")

        self.hparams_audio["intermediate_size"] = lfm2_feed_forward_length(self.hparams_audio)
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.LFM3A)
        self.gguf_writer.add_audio_head_count_kv(self.hparams_audio["num_key_value_heads"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams_audio["norm_eps"])
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["num_mel_bins"])
        self.gguf_writer.add_audio_window_size(self.hparams_audio["sliding_window"])

        if not self._audio_output:
            return

        # audio generation: depthformer hparams, the codec ones are constant and live in clip.cpp
        gen = self.get_generation_config()
        for k, v in self._generation.items():
            if gen[k] != v:
                raise ValueError(f"unsupported {k}={gen[k]!r}, expected {v!r}")
        depth = self.get_depth_config()
        if depth.get("is_causal", False) or depth["use_sandwich_norm"] or set(depth["layer_types"]) != {"full_attention"}:
            raise ValueError("unsupported depthformer, expected a bidirectional all-attention stack")
        if depth["rope_parameters"]["full_attention"]["rope_theta"] != 1e6:
            raise ValueError("unsupported depthformer rope_theta")
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.LFM3A_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(depth["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(depth["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(depth["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(depth["num_attention_heads"])
        self.gguf_writer.add_gen_audio_head_count_kv(depth["num_key_value_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(depth["norm_eps"])

        # sampling defaults of the speech codes, next to the text ones that become general.sampling.*
        gen = gguf.Metadata.load_generation_config(self.dir_model)
        for key, add in (("audio_temperature",        self.gguf_writer.add_gen_audio_sampling_temp),
                         ("audio_top_k",              self.gguf_writer.add_gen_audio_sampling_top_k),
                         ("audio_repetition_penalty", self.gguf_writer.add_gen_audio_sampling_penalty_repeat),
                         ("audio_repetition_window",  self.gguf_writer.add_gen_audio_sampling_penalty_last_n)):
            value = gen.get(key, self._audio_sampling.get(key))
            if value is not None:
                add(value)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if new_name.startswith("a.conv1d.") or "conv_dw" in new_name or new_name.startswith("a.gen.wav.quant."):
            return gguf.GGMLQuantizationType.F32
        # the codec decoder is small and its log-magnitude/phase output does not survive quantization
        if new_name.startswith("a.gen.wav."):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self._audio_output:
            yield from self._codec_tensors()

    def _codec_tensors(self) -> Iterable[tuple[str, Tensor]]:
        import json
        from safetensors.torch import load_file

        T = gguf.MODEL_TENSOR
        codec_dir = self.dir_model / "audio_tokenizer"
        with open(codec_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in self._codec.items():
            if cfg.get(k) != v:
                raise ValueError(f"unsupported audio_tokenizer {k}={cfg.get(k)!r}, expected {v!r}")
        if math.prod(cfg["fsq_levels"]) != self._audio_vocab - 1:
            raise ValueError(f"audio_vocab_size {self._audio_vocab} does not match the codec, expected codebook size + 1")

        state_dict = load_file(codec_dir / "model.safetensors")

        # FSQ code table: index -> the len(levels) quantized values, see FiniteScalarQuantization._indices_to_codes
        levels = torch.tensor(cfg["fsq_levels"], dtype=torch.int64)
        basis = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.int64), levels[:-1]]), dim=0)
        half = levels // 2
        idx = torch.arange(int(levels.prod()), dtype=torch.int64)[:, None]
        codes = ((idx // basis) % levels - half) / half
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_CB), codes.to(torch.float32))

        # one projection over the concatenated code vectors of all codebooks, under quantizer.fsq. in newer exports
        quantizer = "quantizer.fsq." if "quantizer.fsq.project_out.weight" in state_dict else "quantizer."
        for suffix in ("weight", "bias"):
            yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_OUT, suffix=f".{suffix}"), state_dict[f"{quantizer}project_out.{suffix}"])
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_IN_PROJ, suffix=f".{suffix}"), state_dict[f"from_quantizer.{suffix}"])
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUT_PROJ, suffix=f".{suffix}"), state_dict[f"decoder.lin.{suffix}"])
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUTPUT_NORM), state_dict["decoder.lfm.embedding_norm.weight"])

        for name, data_torch in state_dict.items():
            if not name.startswith("decoder.lfm.layers."):
                continue
            bid_str, key = name.removeprefix("decoder.lfm.layers.").rsplit(".weight", 1)[0].split(".", 1)
            if key == "conv.conv":
                data_torch = data_torch.squeeze(1)
            yield (self.format_tensor_name(self._codec_layer_map[key], int(bid_str)), data_torch)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("a.gen.wav."):
            yield (name, data_torch)
            return

        if name.startswith(("depthformer.", "depth_", "codebook_embedding.", "mask_embedding")):
            if self._audio_output:
                yield from self._depthformer_tensor(data_torch, name, bid)
            return

        if not name.startswith(("encoder.", "mlp.")):
            return

        if name.startswith("encoder.downsample."):
            layer, suffix = name.removeprefix("encoder.downsample.").split(".")
            if layer == "lin_out":
                if suffix == "weight":
                    data_torch = data_torch.squeeze(-1)
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_ENC_OUT, suffix=f".{suffix}"), data_torch)
                return
            if suffix == "bias":
                data_torch = data_torch[:, None, None]
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_ENC_CONV1D, self._downsample_idx[layer], f".{suffix}"), data_torch)
            return

        # conv op requires 2d tensor
        if name.endswith("conv.conv.weight"):
            data_torch = data_torch.squeeze(1)

        yield (self.map_tensor_name(name), data_torch)

    def _depthformer_tensor(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR

        if name.startswith("depthformer.model.layers."):
            key = name.removeprefix("depthformer.model.layers.").split(".", 1)[1].rsplit(".weight", 1)[0]
            yield (self.format_tensor_name(self._depth_layer_map[key], bid), data_torch)
            return
        if name == "depthformer.model.embedding_norm.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return
        if name.startswith("depth_linear."):
            yield (self.format_tensor_name(T.A_GEN_CODE_PROJ_IN, suffix="." + name.rsplit(".", 1)[1]), data_torch)
            return
        if name == "mask_embedding":
            yield (self.format_tensor_name(T.A_GEN_CODE_MASK_EMBD, suffix=""), data_torch)
            return
        if name == "codebook_embedding.weight":
            assert data_torch.shape[0] == self._n_codebooks * self._audio_vocab
            yield (self.format_tensor_name(T.A_GEN_CODE_OUT_EMBD), data_torch)
            return

        # per-codebook heads: the embedding table is tied to to_logits, merge the 12 tables into one tensor
        assert name.startswith("depth_embeddings.")
        idx = int(name.split(".")[1])
        key = name.split(".", 2)[2]
        if key == "embedding.weight":
            self._depth_embd[idx] = data_torch
            if len(self._depth_embd) == self._n_codebooks:
                yield (self.format_tensor_name(T.A_GEN_CODE_EMBD), torch.stack([self._depth_embd[i] for i in range(self._n_codebooks)], dim=0))
        elif key == "embedding_norm.weight":
            self._depth_norm[idx] = data_torch
            if len(self._depth_norm) == self._n_codebooks:
                yield (self.format_tensor_name(T.A_GEN_CODE_HEAD_NORM), torch.stack([self._depth_norm[i] for i in range(self._n_codebooks)], dim=0))
        elif key == "to_logits.weight":
            if not torch.equal(data_torch, self._depth_embd[idx]):
                raise ValueError(f"depth_embeddings.{idx} is not tied")
        else:
            raise ValueError(f"unexpected tensor {name}")


@ModelBase.register("Lfm3OmniForConditionalGeneration")
@ModelBase.example("LiquidAI/LFM3-Omni-2B")
class LFM3OmniMmprojModel(LFM3VLVisionMixin, LFM3AudioMmprojModel):
    has_vision_encoder = True
    has_audio_encoder = True

    _audio_sampling = {"audio_temperature": 0.4, "audio_repetition_penalty": 1.2, "audio_repetition_window": 64}

    # the audio side is the LFM3-Audio model under the HF-native names
    _audio_renames = (
        ("model.audio_encoder.lfm.", "encoder.lfm.lfm."),
        ("model.audio_encoder.", "encoder."),
        ("model.audio_projector.layer_norm.", "mlp.model.0."),
        ("model.audio_projector.linear_1.", "mlp.model.1."),
        ("model.audio_projector.linear_2.", "mlp.model.3."),
        ("model.depth_decoder.model.", "depthformer.model."),
        ("model.depth_decoder.", ""),
        ("model.codebook_embedding.", "codebook_embedding."),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # dynamic resolution, image_size is only set for MmprojModel compatibility
        self.hparams_vision["image_size"] = 512

    def get_audio_config(self) -> dict[str, Any] | None:
        enc = self.global_config["audio_encoder_config"]
        cfg = dict(enc["text_config"])
        cfg["num_mel_bins"] = enc["num_mel_bins"]
        cfg["sliding_window"] = enc["sliding_window"]
        return cfg

    def get_generation_config(self) -> dict[str, Any]:
        c = self.global_config
        return {
            "codebooks": c["num_codebooks"], "audio_vocab_size": c["codebook_size"], "maskgit_schedule": c["maskgit_schedule"],
            "interleaved_n_text": c["interleaved_n_text"], "interleaved_n_audio": c["interleaved_n_audio"],
        }

    def get_depth_config(self) -> dict[str, Any]:
        return self.global_config["depth_config"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._set_lfm3vl_gguf_parameters()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith(("model.vision_tower.", "model.multi_modal_projector.")):
            yield from self._lfm3vl_tensors(data_torch, name, bid)
            return
        for old, new in self._audio_renames:
            if name.startswith(old):
                yield from super().modify_tensors(data_torch, new + name.removeprefix(old), bid)
                return
        yield from super().modify_tensors(data_torch, name, bid)
