from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger

# VibeVoice is a next-token diffusion TTS model:
#   language_model (Qwen2) --> hidden state per speech frame
#   prediction_head        --> diffusion head, denoises one acoustic latent per frame, conditioned on the hidden state
#   acoustic_tokenizer     --> sigma-VAE, the decoder turns acoustic latents into 24kHz audio
#   semantic_tokenizer     --> encoder only, re-encodes generated audio for the next LM step
#   acoustic/semantic_connector --> project the latents back into the LM embedding space
#
# everything goes into one GGUF; the backbone uses the normal blk.* names.
# the checkpoint ships no tokenizer, it uses the one of the Qwen2.5 base model.


@ModelBase.register("VibeVoiceForConditionalGeneration")
@ModelBase.example("microsoft/VibeVoice-1.5B")
class VibeVoiceModel(TextModel):
    model_arch = gguf.MODEL_ARCH.VIBEVOICE

    # <tokenizer>.<encoder|decoder>.stages.<stage>.<block>.<key>
    _STAGE_RE = re.compile(r"^model\.(acoustic|semantic)_tokenizer\.(encoder|decoder)\.stages\.(\d+)\.(\d+)\.(.+)$")

    # Block1D keys of each tokenizer stage
    _STAGE_TENSORS = {
        ("acoustic", "encoder"): {
            "norm":                 gguf.MODEL_TENSOR.VV_ACOUSTIC_ENC_NORM,
            "mixer.conv.conv.conv": gguf.MODEL_TENSOR.VV_ACOUSTIC_ENC_DWCONV,
            "gamma":                gguf.MODEL_TENSOR.VV_ACOUSTIC_ENC_GAMMA,
            "ffn_norm":             gguf.MODEL_TENSOR.VV_ACOUSTIC_ENC_FFN_NORM,
            "ffn.linear1":          gguf.MODEL_TENSOR.VV_ACOUSTIC_ENC_FFN_UP,
            "ffn.linear2":          gguf.MODEL_TENSOR.VV_ACOUSTIC_ENC_FFN_DOWN,
            "ffn_gamma":            gguf.MODEL_TENSOR.VV_ACOUSTIC_ENC_FFN_GAMMA,
        },
        ("acoustic", "decoder"): {
            "norm":                 gguf.MODEL_TENSOR.VV_ACOUSTIC_DEC_NORM,
            "mixer.conv.conv.conv": gguf.MODEL_TENSOR.VV_ACOUSTIC_DEC_DWCONV,
            "gamma":                gguf.MODEL_TENSOR.VV_ACOUSTIC_DEC_GAMMA,
            "ffn_norm":             gguf.MODEL_TENSOR.VV_ACOUSTIC_DEC_FFN_NORM,
            "ffn.linear1":          gguf.MODEL_TENSOR.VV_ACOUSTIC_DEC_FFN_UP,
            "ffn.linear2":          gguf.MODEL_TENSOR.VV_ACOUSTIC_DEC_FFN_DOWN,
            "ffn_gamma":            gguf.MODEL_TENSOR.VV_ACOUSTIC_DEC_FFN_GAMMA,
        },
        ("semantic", "encoder"): {
            "norm":                 gguf.MODEL_TENSOR.VV_SEMANTIC_ENC_NORM,
            "mixer.conv.conv.conv": gguf.MODEL_TENSOR.VV_SEMANTIC_ENC_DWCONV,
            "gamma":                gguf.MODEL_TENSOR.VV_SEMANTIC_ENC_GAMMA,
            "ffn_norm":             gguf.MODEL_TENSOR.VV_SEMANTIC_ENC_FFN_NORM,
            "ffn.linear1":          gguf.MODEL_TENSOR.VV_SEMANTIC_ENC_FFN_UP,
            "ffn.linear2":          gguf.MODEL_TENSOR.VV_SEMANTIC_ENC_FFN_DOWN,
            "ffn_gamma":            gguf.MODEL_TENSOR.VV_SEMANTIC_ENC_FFN_GAMMA,
        },
    }

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        if "decoder_config" not in hparams:
            # transformers has its own VibeVoice config layout, read the original one
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        hparams["text_config"] = hparams["decoder_config"]
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

        # tokenizer convs and diffusion head layers are indexed by {bid} too, and can outnumber the LM layers
        n_blocks = max(
            self.block_count,
            self.hparams["diffusion_head_config"]["head_layers"],
            len(self.hparams["acoustic_tokenizer_config"]["encoder_ratios"]) + 1,
            len(self.hparams["acoustic_tokenizer_config"].get("decoder_ratios") or []) + 1,
            len(self.hparams["semantic_tokenizer_config"]["encoder_ratios"]) + 1,
        )
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, n_blocks)

    def _key(self, key: str) -> str:
        return key.format(arch=self.gguf_writer.arch)

    def set_vocab(self):
        if not (self.dir_model / "tokenizer.json").is_file() and not (self.dir_model / "vocab.json").is_file():
            raise FileNotFoundError(
                "VibeVoice ships no tokenizer files, copy them from the Qwen2.5 base model "
                "(e.g. Qwen/Qwen2.5-1.5B for VibeVoice-1.5B) into the model directory"
            )
        self._set_vocab_gpt2()

    @staticmethod
    def _parse_depths(depths: str | list[int]) -> list[int]:
        if isinstance(depths, str):
            return [int(d) for d in depths.split("-")]
        return list(depths)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        K = gguf.Keys.VibeVoice
        w = self.gguf_writer

        # ratios are stored in the HF config order
        at: dict[str, Any] = self.hparams["acoustic_tokenizer_config"]
        at_enc_depths = self._parse_depths(at["encoder_depths"])
        at_dec_depths = self._parse_depths(at["decoder_depths"]) if at.get("decoder_depths") else list(reversed(at_enc_depths))
        w.add_uint32(self._key(K.ACOUSTIC_VAE_DIM), at["vae_dim"])
        w.add_uint32(self._key(K.ACOUSTIC_N_FILTERS), at["encoder_n_filters"])
        w.add_array(self._key(K.ACOUSTIC_ENCODER_RATIOS), at["encoder_ratios"])
        w.add_array(self._key(K.ACOUSTIC_ENCODER_DEPTHS), at_enc_depths)
        w.add_array(self._key(K.ACOUSTIC_DECODER_RATIOS), at.get("decoder_ratios") or at["encoder_ratios"])
        w.add_array(self._key(K.ACOUSTIC_DECODER_DEPTHS), at_dec_depths)
        w.add_float32(self._key(K.ACOUSTIC_FIX_STD), at["fix_std"])
        w.add_float32(self._key(K.ACOUSTIC_LAYERNORM_EPS), at["layernorm_eps"])

        st: dict[str, Any] = self.hparams["semantic_tokenizer_config"]
        w.add_uint32(self._key(K.SEMANTIC_VAE_DIM), st["vae_dim"])
        w.add_uint32(self._key(K.SEMANTIC_N_FILTERS), st["encoder_n_filters"])
        w.add_array(self._key(K.SEMANTIC_ENCODER_RATIOS), st["encoder_ratios"])
        w.add_array(self._key(K.SEMANTIC_ENCODER_DEPTHS), self._parse_depths(st["encoder_depths"]))
        w.add_float32(self._key(K.SEMANTIC_LAYERNORM_EPS), st["layernorm_eps"])

        dh: dict[str, Any] = self.hparams["diffusion_head_config"]
        w.add_uint32(self._key(K.DIFFUSION_BLOCK_COUNT), dh["head_layers"])
        w.add_uint32(self._key(K.DIFFUSION_FEED_FORWARD_LEN), int(dh["hidden_size"] * dh["head_ffn_ratio"]))
        w.add_uint32(self._key(K.DIFFUSION_LATENT_SIZE), dh["latent_size"])
        w.add_float32(self._key(K.DIFFUSION_LAYERNORM_EPS), dh["rms_norm_eps"])
        w.add_string(self._key(K.DIFFUSION_TYPE), dh["diffusion_type"])
        w.add_string(self._key(K.DIFFUSION_PREDICTION_TYPE), dh["prediction_type"])
        w.add_string(self._key(K.DIFFUSION_BETA_SCHEDULE), dh["ddpm_beta_schedule"])
        w.add_uint32(self._key(K.DIFFUSION_TRAIN_STEPS), dh["ddpm_num_steps"])
        w.add_uint32(self._key(K.DIFFUSION_INFERENCE_STEPS), dh["ddpm_num_inference_steps"])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # conv1d/conv_transpose_1d kernels must be F16, ggml has no BF16 path for them
        if n_dims == 3 and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # scalar buffers that normalize the acoustic latents, stored as KV
        if name in ("model.speech_scaling_factor", "model.speech_bias_factor"):
            key = gguf.Keys.VibeVoice.SPEECH_SCALING_FACTOR if name.endswith("scaling_factor") else gguf.Keys.VibeVoice.SPEECH_BIAS_FACTOR
            value = LazyTorchTensor.to_eager(data_torch).float().item()
            logger.info(f"gguf: {name} = {value}")
            self.gguf_writer.add_float32(self._key(key), value)
            return

        if (m := self._STAGE_RE.match(name)) is not None:
            tok, part, stage, block, rest = m.groups()
            # gamma and ffn_gamma are bare parameters, give them a .weight suffix
            if rest in ("gamma", "ffn_gamma"):
                key, suffix = rest, ".weight"
            else:
                key, suffix = rest.rsplit(".", 1)
                suffix = "." + suffix
            tensor = self._STAGE_TENSORS.get((tok, part), {}).get(key)
            if tensor is None:
                raise ValueError(f"Can not map tensor {name!r}")
            new_name = gguf.TENSOR_NAMES[tensor].format(bid=int(stage), xid=int(block)) + suffix
            yield (new_name, data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)
