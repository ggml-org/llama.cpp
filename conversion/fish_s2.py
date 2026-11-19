from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .drama3 import Drama3MmprojModel, fold_conv, fold_snake, fold_wn, load_codec
from .qwen import Qwen3Model

# Fish Audio S2 Pro (dual-AR)
# text model: Qwen3 backbone (slow AR), samples the semantic code of each frame
# mmproj: fast AR code predictor (other codebooks) and Fish DAC decoder (codec.pth, same as drama3)

# residual codebook size of codec.pth, the fast AR head is cut to it
_N_RESIDUAL_CODES = 1024

# codec encoder strides (encoder_rates of the codec config), the quantizer then downsamples 2x2
_CODEC_ENC_STRIDES = (2, 4, 8, 8)

# codec encoder transformers, both 1024 wide with 16 heads of 64
# generic audio layers a.blk.0..3: last encoder block, a.blk.4..11: quantizer pre_module
_N_ENC_TFM_LAYERS = 4
_N_PRE_TFM_LAYERS = 8
_CODEC_HEAD_DIM = 64


def _load_hparams(dir_model: Path) -> dict[str, Any]:
    hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
    cfg = hparams["text_config"]
    hparams["text_config"] = {
        "hidden_size":             cfg["dim"],
        "num_hidden_layers":       cfg["n_layer"],
        "num_attention_heads":     cfg["n_head"],
        "num_key_value_heads":     cfg["n_local_heads"],
        "head_dim":                cfg["head_dim"],
        "intermediate_size":       cfg["intermediate_size"],
        "rms_norm_eps":            cfg["norm_eps"],
        "rope_theta":              cfg["rope_base"],
        "max_position_embeddings": cfg["max_seq_len"],
        "vocab_size":              cfg["vocab_size"],
        "tie_word_embeddings":     cfg["tie_word_embeddings"],
    }
    return hparams


def _neox_rows(w: Tensor, head_dim: int) -> Tensor:
    # checkpoint uses interleaved RoPE, reorder the rows of each head so NEOX RoPE gives the same result
    order = torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])
    return w.reshape(-1, head_dim, *w.shape[1:])[:, order].reshape(w.shape)


@ModelBase.register("fish_qwen3_omni")
@ModelBase.example("fishaudio/s2-pro")
class FishS2TextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    def __init__(self, dir_model: Path, *args, **kwargs):
        kwargs.pop("hparams", None)
        super().__init__(dir_model, *args, hparams=_load_hparams(dir_model), **kwargs)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the rest is mapped by tensor_mapping.py
        name = name.removeprefix("text_model.")
        T = gguf.MODEL_TENSOR
        head_dim = self.hparams["head_dim"]
        if name.endswith(".attention.wqkv.weight"):
            n_q  = self.hparams["num_attention_heads"] * head_dim
            n_kv = self.hparams["num_key_value_heads"] * head_dim
            q, k, v = data_torch.split([n_q, n_kv, n_kv], dim=0)
            yield (self.format_tensor_name(T.ATTN_Q, bid), _neox_rows(q, head_dim))
            yield (self.format_tensor_name(T.ATTN_K, bid), _neox_rows(k, head_dim))
            yield (self.format_tensor_name(T.ATTN_V, bid), v)
            return
        if name.endswith(".attention.q_norm.weight"):
            yield (self.format_tensor_name(T.ATTN_Q_NORM, bid), _neox_rows(data_torch, head_dim))
            return
        if name.endswith(".attention.k_norm.weight"):
            yield (self.format_tensor_name(T.ATTN_K_NORM, bid), _neox_rows(data_torch, head_dim))
            return
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("fish_qwen3_omni")
@ModelBase.example("fishaudio/s2-pro")
class FishS2MmprojModel(Drama3MmprojModel):
    has_audio_encoder = True  # codec encoder, for voice cloning
    gen_projector_type = gguf.VisionProjectorType.FISH_S2_GEN

    _FAST_AR_MAP = {
        "attention_norm.weight":     gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM,
        "attention.wo.weight":       gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT,
        "ffn_norm.weight":           gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM,
        "feed_forward.w1.weight":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE,
        "feed_forward.w3.weight":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP,
        "feed_forward.w2.weight":    gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN,
    }

    def __init__(self, dir_model: Path, *args, **kwargs):
        kwargs.pop("hparams", None)
        super().__init__(dir_model, *args, hparams=_load_hparams(dir_model), **kwargs)

    def get_audio_config(self) -> dict[str, Any] | None:
        cfg = dict(self.global_config["audio_decoder_config"])
        cfg["num_hidden_layers"] = cfg["n_layer"]
        return cfg

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # codec encoder: raw 44.1 kHz waveform -> backbone input rows, other hparams are hardcoded in tools/mtmd
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.FISH_S2_SPKENC)
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_audio_num_mel_bins(1)
        self.gguf_writer.add_audio_block_count(_N_ENC_TFM_LAYERS + _N_PRE_TFM_LAYERS)
        self.gguf_writer.add_audio_embedding_length(1024)
        self.gguf_writer.add_audio_feed_forward_length(3072)
        self.gguf_writer.add_audio_head_count(1024 // _CODEC_HEAD_DIM)
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the VQ picks the nearest codebook entry, small errors in the encoder change the codes
        if new_name.startswith("a.codec."):
            return gguf.GGMLQuantizationType.F32
        # codec transformers: F16 gives codes closer to the reference than BF16
        if new_name.startswith("a.blk.") and n_dims == 2:
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]
        if name.startswith("audio_decoder.") or name == "text_model.model.embeddings.weight":
            return item
        return None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        cfg = self.hparams

        # already named by generate_extra_tensors()
        if name.startswith("a."):
            yield (name, data_torch)
            return

        if name == "text_model.model.embeddings.weight":
            # semantic band of the backbone embeddings
            begin = self.global_config["semantic_start_token_id"]
            yield (self.format_tensor_name(T.A_GEN_CODE_OUT_EMBD), data_torch[begin:begin + cfg["vocab_size"]])
            return

        if name == "audio_decoder.codebook_embeddings.weight":
            # backbone input of a speech row, one table per codebook
            yield (self.format_tensor_name(T.A_GEN_CODE_EMBD), data_torch.reshape(cfg["num_codebooks"], -1, data_torch.shape[-1]))
            return

        if name == "audio_decoder.embeddings.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_TOK_EMBD), data_torch)
            return

        if name == "audio_decoder.output.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), data_torch[:_N_RESIDUAL_CODES])
            return

        if name == "audio_decoder.norm.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return

        assert bid is not None and name.startswith(f"audio_decoder.layers.{bid}.")
        key = name.removeprefix(f"audio_decoder.layers.{bid}.")
        if key == "attention.wqkv.weight":
            head_dim = cfg["head_dim"]
            n_q  = cfg["n_head"] * head_dim
            n_kv = cfg["n_local_heads"] * head_dim
            q, k, v = data_torch.split([n_q, n_kv, n_kv], dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_Q, bid), _neox_rows(q, head_dim))
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_K, bid), _neox_rows(k, head_dim))
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_V, bid), v)
            return
        yield (self.format_tensor_name(self._FAST_AR_MAP[key], bid), data_torch)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()

        sd = load_codec(self.dir_model)
        T = gguf.MODEL_TENSOR

        def get(name: str) -> Tensor:
            return sd[name].float()

        def tn(tensor: gguf.MODEL_TENSOR, bid: int | None = None, xid: int | None = None) -> str:
            return gguf.TENSOR_NAMES[tensor].format(bid=bid, xid=xid)

        def conv_wb(tensor: gguf.MODEL_TENSOR, prefix: str, bid: int | None = None, xid: int | None = None, stride: int = 1):
            # taps-major [K / stride, OC, stride * IC], see conv() in tools/mtmd/models/fish-s2-spkenc.cpp
            w = fold_conv(sd, prefix)
            n_out, n_in, k = w.shape
            w = w.reshape(n_out, n_in, k // stride, stride).permute(2, 0, 3, 1).reshape(k // stride, n_out, stride * n_in)
            yield (tn(tensor, bid, xid) + ".weight", w)
            yield (tn(tensor, bid, xid) + ".bias", get(prefix + ".conv.bias"))

        def snake_ab(tensor: gguf.MODEL_TENSOR, name: str, bid: int | None = None, xid: int | None = None):
            a, b = fold_snake(get(name))
            yield (tn(tensor, bid, xid) + ".alpha", a)
            yield (tn(tensor, bid, xid) + ".beta", b)

        def tfm_layers(prefix: str, n_layer: int, offset: int):
            for il in range(n_layer):
                p = f"{prefix}layers.{il}."
                bid = offset + il
                wq, wk, wv = get(p + "attention.wqkv.weight").chunk(3, dim=0)
                yield (self.format_tensor_name(T.A_ENC_ATTN_Q,       bid), _neox_rows(wq, _CODEC_HEAD_DIM))
                yield (self.format_tensor_name(T.A_ENC_ATTN_K,       bid), _neox_rows(wk, _CODEC_HEAD_DIM))
                yield (self.format_tensor_name(T.A_ENC_ATTN_V,       bid), wv)
                yield (self.format_tensor_name(T.A_ENC_OUTPUT,       bid), get(p + "attention.wo.weight"))
                yield (self.format_tensor_name(T.A_ENC_INPUT_NORM,   bid), get(p + "attention_norm.weight"))
                yield (self.format_tensor_name(T.A_ENC_ATTN_SCALE,   bid), get(p + "attention_layer_scale.gamma"))
                yield (self.format_tensor_name(T.A_ENC_OUTPUT_NORM,  bid), get(p + "ffn_norm.weight"))
                yield (self.format_tensor_name(T.A_ENC_FFN_GATE,     bid), get(p + "feed_forward.w1.weight"))
                yield (self.format_tensor_name(T.A_ENC_FFN_UP,       bid), get(p + "feed_forward.w3.weight"))
                yield (self.format_tensor_name(T.A_ENC_FFN_DOWN,     bid), get(p + "feed_forward.w2.weight"))
                yield (self.format_tensor_name(T.A_ENC_FFN_SCALE_LS, bid), get(p + "ffn_layer_scale.gamma"))

        # encoder: entry conv, 4x (3 residual units, Snake, strided conv), Snake, post conv
        e = "encoder.block."
        yield from conv_wb(T.A_CODEC_ENC_ENTRY, e + "0")
        for bid, stride in enumerate(_CODEC_ENC_STRIDES):
            p = f"{e}{bid + 1}.block."
            for xid in range(3):
                r = f"{p}{xid}.block."
                yield from snake_ab(T.A_CODEC_ENC_RES_ACT1, r + "0.alpha", bid, xid)
                yield from conv_wb(T.A_CODEC_ENC_RES_CONV1, r + "1", bid, xid)
                yield from snake_ab(T.A_CODEC_ENC_RES_ACT2, r + "2.alpha", bid, xid)
                yield from conv_wb(T.A_CODEC_ENC_RES_CONV2, r + "3", bid, xid)
            yield from snake_ab(T.A_CODEC_ENC_SNAKE, p + "3.alpha", bid)
            yield from conv_wb(T.A_CODEC_ENC_CONV, p + "4", bid, stride=stride)
        # the last block ends with a windowed transformer
        yield from tfm_layers(f"{e}4.block.5.", _N_ENC_TFM_LAYERS, 0)
        yield (self.format_tensor_name(T.A_POST_NORM), get(f"{e}4.block.5.norm.weight"))
        yield from snake_ab(T.A_CODEC_ENC_POST_SNAKE, e + "5.alpha")
        yield from conv_wb(T.A_CODEC_ENC_POST_CONV, e + "6")

        # quantizer: 2x (strided conv + ConvNeXt), windowed transformer, then semantic + residual VQ
        q = "quantizer."
        for bid in range(2):
            p = f"{q}downsample.{bid}."
            yield from conv_wb(T.A_CODEC_DOWN_CONV, p + "0", bid, stride=2)
            # depthwise: one [C] row per tap
            yield (tn(T.A_CODEC_DOWN_DWCONV, bid) + ".weight", fold_conv(sd, p + "1.dwconv").squeeze(1).T.contiguous())
            yield (tn(T.A_CODEC_DOWN_DWCONV, bid) + ".bias", get(p + "1.dwconv.conv.bias"))
            for key, tensor in (("norm", T.A_CODEC_DOWN_NORM), ("pwconv1", T.A_CODEC_DOWN_PW1), ("pwconv2", T.A_CODEC_DOWN_PW2)):
                yield (tn(tensor, bid) + ".weight", get(f"{p}1.{key}.weight"))
                yield (tn(tensor, bid) + ".bias", get(f"{p}1.{key}.bias"))
            yield (tn(T.A_CODEC_DOWN_GAMMA, bid), get(p + "1.gamma"))
        yield from tfm_layers(q + "pre_module.", _N_PRE_TFM_LAYERS, _N_ENC_TFM_LAYERS)
        yield (self.format_tensor_name(T.A_CODEC_PRE_NORM), get(q + "pre_module.norm.weight"))

        # nearest codebook entry by cosine similarity, see VectorQuantize.decode_latents()
        def vq(prefix: str) -> tuple[Tensor, Tensor, Tensor]:
            w = fold_wn(sd[prefix + ".in_proj.weight_g"], sd[prefix + ".in_proj.weight_v"])
            cb = torch.nn.functional.normalize(get(prefix + ".codebook.weight"), dim=-1)
            return w, get(prefix + ".in_proj.bias"), cb

        w, b, cb = vq(q + "semantic_quantizer.quantizers.0")
        yield (self.format_tensor_name(T.A_CODEC_VQ_FIRST_IN, suffix=".weight"), w)
        yield (self.format_tensor_name(T.A_CODEC_VQ_FIRST_IN, suffix=".bias"), b)
        yield (self.format_tensor_name(T.A_CODEC_VQ_FIRST_CB), cb)
        n_rest = self.hparams["num_codebooks"] - 1
        rest = [vq(f"{q}quantizer.quantizers.{i}") for i in range(n_rest)]
        yield (self.format_tensor_name(T.A_CODEC_VQ_REST_IN, suffix=".weight"), torch.stack([r[0] for r in rest]))
        yield (self.format_tensor_name(T.A_CODEC_VQ_REST_IN, suffix=".bias"), torch.stack([r[1] for r in rest]))
        yield (self.format_tensor_name(T.A_CODEC_VQ_REST_CB), torch.stack([r[2] for r in rest]))
