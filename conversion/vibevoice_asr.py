from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf
from .qwen import Qwen2Model


# the upstream chat_template.jinja is a processor template (list content, audio parts), use plain ChatML with its default system prompt
VIBEVOICE_ASR_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' -%}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\\n' -}}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)


@ModelBase.register("VibeVoiceAsrForConditionalGeneration")
@ModelBase.example("microsoft/VibeVoice-ASR-HF")
class VibeVoiceAsrModel(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template(VIBEVOICE_ASR_CHAT_TEMPLATE)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if not name.startswith("language_model."):
            return None
        return super().filter_tensors(item)


@ModelBase.register("VibeVoiceAsrForConditionalGeneration")
class VibeVoiceAsrAudioModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    # values hard-coded in tools/mtmd/models/vibevoice-asr.cpp
    _encoder_consts = {"channels": 1, "kernel_size": 7, "num_filters": 32, "ffn_expansion": 4, "hidden_act": "gelu", "rms_norm_eps": 1e-5}
    _encoders = ("acoustic_tokenizer_encoder", "semantic_tokenizer_encoder")

    def get_audio_config(self) -> dict[str, Any] | None:
        c = self.global_config["acoustic_tokenizer_encoder_config"]
        return {
            **c,
            "intermediate_size": c["ffn_expansion"] * c["hidden_size"],
            "num_attention_heads": 1,
            "num_hidden_layers": sum(c["depths"]),
        }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        for enc in self._encoders:
            c = self.global_config[f"{enc}_config"]
            for k, v in self._encoder_consts.items():
                assert c[k] == v, f"{enc}.{k} = {c[k]}, expected {v}"
            assert c["depths"] == self.hparams["depths"] and c["downsampling_ratios"] == self.hparams["downsampling_ratios"]

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.VIBEVOICE_ASR)
        # the input is the raw waveform, one channel takes the place of the mel bins
        self.gguf_writer.add_audio_num_mel_bins(1)
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_audio_conv_kernel_size(self.hparams["kernel_size"])
        # stride of each stage, the stem has none
        self.gguf_writer.add_audio_patch_sizes([1] + [int(r) for r in self.hparams["downsampling_ratios"]])
        self.gguf_writer.add_audio_depths([int(d) for d in self.hparams["depths"]])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".dw." in new_name:
            return gguf.GGMLQuantizationType.F32 # ggml_ssm_conv
        if ".conv." in new_name and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if not name.startswith(cls._encoders + ("multi_modal_projector.",)):
            return None
        return super().filter_tensors(item)

    _block_tensors = {
        "norm.weight":         gguf.MODEL_TENSOR.A_ENC_CNX_NORM,
        "mixer.conv.weight":   gguf.MODEL_TENSOR.A_ENC_CNX_DW,
        "mixer.conv.bias":     gguf.MODEL_TENSOR.A_ENC_CNX_DW,
        "gamma":               gguf.MODEL_TENSOR.A_ENC_CNX_LS1,
        "ffn_norm.weight":     gguf.MODEL_TENSOR.A_ENC_CNX_FFN_NORM,
        "ffn.linear1.weight":  gguf.MODEL_TENSOR.A_ENC_CNX_FFN_UP,
        "ffn.linear1.bias":    gguf.MODEL_TENSOR.A_ENC_CNX_FFN_UP,
        "ffn.linear2.weight":  gguf.MODEL_TENSOR.A_ENC_CNX_FFN_DOWN,
        "ffn.linear2.bias":    gguf.MODEL_TENSOR.A_ENC_CNX_FFN_DOWN,
        "ffn_gamma":           gguf.MODEL_TENSOR.A_ENC_CNX_LS2,
    }

    # mm.a.mlp.{0,1} + mm.a.mlp_norm.0 for the acoustic connector, {2,3} + 2 for the semantic one
    _proj_tensors = {
        "acoustic_linear_1": (gguf.MODEL_TENSOR.A_MMPROJ, 0),
        "acoustic_norm":     (gguf.MODEL_TENSOR.A_MMPROJ_NORM, 0),
        "acoustic_linear_2": (gguf.MODEL_TENSOR.A_MMPROJ, 1),
        "semantic_linear_1": (gguf.MODEL_TENSOR.A_MMPROJ, 2),
        "semantic_norm":     (gguf.MODEL_TENSOR.A_MMPROJ_NORM, 2),
        "semantic_linear_2": (gguf.MODEL_TENSOR.A_MMPROJ, 3),
    }

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the layer scales (gamma) have no suffix upstream
        suffix = "." + name.rsplit(".", 1)[-1] if name.endswith((".weight", ".bias")) else ".weight"

        if name.startswith("multi_modal_projector."):
            key, idx = self._proj_tensors[name.split(".")[1]]
            yield (self.format_tensor_name(key, idx, suffix=suffix), data_torch)
            return

        enc, _, rest = name.partition(".")
        enc_id = self._encoders.index(enc)
        parts = rest.split(".")
        depths = [int(d) for d in self.hparams["depths"]]

        # causal convs of the stages: 0 = stem, 1..6 = downsample, 7 = head
        if rest.startswith("stem.conv."):
            conv_id = 0
        elif parts[0] == "conv_layers" and parts[2] == "conv":
            conv_id = int(parts[1]) + 1
        elif parts[0] == "head":
            conv_id = len(depths)
        else:
            conv_id = None
        if conv_id is not None:
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.A_ENC_CNX_CONV].format(bid=enc_id, xid=conv_id) + suffix, data_torch)
            return

        # ConvNeXt blocks are numbered per stage upstream and flat here
        if parts[0] == "stem":
            stage, blk, tail = 0, int(parts[2]), ".".join(parts[3:])
        else:
            assert parts[0] == "conv_layers" and parts[2] == "stage", name
            stage, blk, tail = int(parts[1]) + 1, int(parts[3]), ".".join(parts[4:])
        xid = sum(depths[:stage]) + blk

        key = self._block_tensors[tail]
        if key == gguf.MODEL_TENSOR.A_ENC_CNX_DW and tail.endswith(".weight"):
            data_torch = data_torch.squeeze(1) # [C, 1, K] -> [C, K] for ggml_ssm_conv
        yield (gguf.TENSOR_NAMES[key].format(bid=enc_id, xid=xid) + suffix, data_torch)
