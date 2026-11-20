from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, gguf
from .qwen import Qwen3_5MoeTextModel

# Fish Audio Drama 3 (FishOmniModel)
# text model: Qwen3.5-MoE backbone, samples the main token of each frame
# mmproj: codebook tower and Fish DAC decoder (codec.pth)

_TOWER_SHARED = "audio_decoder_mtp_shared.layers."
_TOWER_RESIDUAL = "audio_decoder_mtp.layers."


def load_codec(dir_model) -> dict[str, Tensor]:
    sd = torch.load(dir_model / "codec.pth", map_location="cpu", mmap=True, weights_only=True)
    return sd.get("state_dict", sd)


def fold_conv(sd: dict[str, Tensor], prefix: str) -> Tensor:
    # fold weight norm
    if prefix + ".conv.weight" in sd:
        return sd[prefix + ".conv.weight"].float()
    g = sd[prefix + ".conv.parametrizations.weight.original0"].float()
    v = sd[prefix + ".conv.parametrizations.weight.original1"].float()
    return g * v / v.norm(dim=(1, 2), keepdim=True)


def fold_wn(g: Tensor, v: Tensor) -> Tensor:
    # fold old-style weight norm of a 1x1 conv, returns [out, in]
    return (g.float() * v.float() / v.float().norm(dim=(1, 2), keepdim=True)).squeeze(-1)


def fold_snake(alpha: Tensor) -> tuple[Tensor, Tensor]:
    alpha = alpha.float().flatten()
    return alpha, 1.0 / (alpha + 1e-9)


# codec encoder strides (encoder_rates of the codec config), the quantizer then downsamples 2x2
_CODEC_ENC_STRIDES = (2, 4, 8, 8)

# codec encoder transformers, both 1024 wide with 16 heads of 64
# generic audio layers a.blk.0..3: last encoder block, a.blk.4..11: quantizer pre_module
_N_ENC_TFM_LAYERS = 4
_N_PRE_TFM_LAYERS = 8
_CODEC_HEAD_DIM = 64


def _neox_rows(w: Tensor, head_dim: int) -> Tensor:
    # checkpoint uses interleaved RoPE, reorder the rows of each head so NEOX RoPE gives the same result
    order = torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])
    return w.reshape(-1, head_dim, *w.shape[1:])[:, order].reshape(w.shape)


def codec_encoder_tensors(sd: dict[str, Tensor], num_codebooks: int) -> Iterable[tuple[str, Tensor]]:
    T = gguf.MODEL_TENSOR

    def get(name: str) -> Tensor:
        return sd[name].float()

    def tn(tensor: gguf.MODEL_TENSOR, bid: int | None = None, xid: int | None = None) -> str:
        return gguf.TENSOR_NAMES[tensor].format(bid=bid, xid=xid)

    def fmt(tensor: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        return tn(tensor, bid) + suffix

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
            yield (fmt(T.A_ENC_ATTN_Q,       bid), _neox_rows(wq, _CODEC_HEAD_DIM))
            yield (fmt(T.A_ENC_ATTN_K,       bid), _neox_rows(wk, _CODEC_HEAD_DIM))
            yield (fmt(T.A_ENC_ATTN_V,       bid), wv)
            yield (fmt(T.A_ENC_OUTPUT,       bid), get(p + "attention.wo.weight"))
            yield (fmt(T.A_ENC_INPUT_NORM,   bid), get(p + "attention_norm.weight"))
            yield (fmt(T.A_ENC_ATTN_SCALE,   bid), get(p + "attention_layer_scale.gamma"))
            yield (fmt(T.A_ENC_OUTPUT_NORM,  bid), get(p + "ffn_norm.weight"))
            yield (fmt(T.A_ENC_FFN_GATE,     bid), get(p + "feed_forward.w1.weight"))
            yield (fmt(T.A_ENC_FFN_UP,       bid), get(p + "feed_forward.w3.weight"))
            yield (fmt(T.A_ENC_FFN_DOWN,     bid), get(p + "feed_forward.w2.weight"))
            yield (fmt(T.A_ENC_FFN_SCALE_LS, bid), get(p + "ffn_layer_scale.gamma"))

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
    yield (fmt(T.A_POST_NORM), get(f"{e}4.block.5.norm.weight"))
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
    yield (fmt(T.A_CODEC_PRE_NORM), get(q + "pre_module.norm.weight"))

    # nearest codebook entry by cosine similarity, see VectorQuantize.decode_latents()
    def vq(prefix: str) -> tuple[Tensor, Tensor, Tensor]:
        w = fold_wn(sd[prefix + ".in_proj.weight_g"], sd[prefix + ".in_proj.weight_v"])
        cb = torch.nn.functional.normalize(get(prefix + ".codebook.weight"), dim=-1)
        return w, get(prefix + ".in_proj.bias"), cb

    w, b, cb = vq(q + "semantic_quantizer.quantizers.0")
    yield (fmt(T.A_CODEC_VQ_FIRST_IN, suffix=".weight"), w)
    yield (fmt(T.A_CODEC_VQ_FIRST_IN, suffix=".bias"), b)
    yield (fmt(T.A_CODEC_VQ_FIRST_CB), cb)
    n_rest = num_codebooks - 1
    rest = [vq(f"{q}quantizer.quantizers.{i}") for i in range(n_rest)]
    yield (fmt(T.A_CODEC_VQ_REST_IN, suffix=".weight"), torch.stack([r[0] for r in rest]))
    yield (fmt(T.A_CODEC_VQ_REST_IN, suffix=".bias"), torch.stack([r[1] for r in rest]))
    yield (fmt(T.A_CODEC_VQ_REST_CB), torch.stack([r[2] for r in rest]))


@ModelBase.register("FishOmniModel")
@ModelBase.example("fishaudio/drama-3")
class Drama3TextModel(Qwen3_5MoeTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE
    no_mtp = True  # the checkpoint has no MTP head

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0].startswith("audio_decoder"):
            return None
        return super().filter_tensors(item)


@ModelBase.register("FishOmniModel")
@ModelBase.example("fishaudio/drama-3")
class Drama3MmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True  # codec encoder, for voice cloning
    gen_projector_type = gguf.VisionProjectorType.DRAMA3_GEN

    _LAYER_MAP = {
        "enorm.weight":                                 (gguf.MODEL_TENSOR.A_GEN_CODE_ENORM, True),
        "hnorm.weight":                                 (gguf.MODEL_TENSOR.A_GEN_CODE_HNORM, True),
        "eh_proj.weight":                               (gguf.MODEL_TENSOR.A_GEN_CODE_EH_PROJ, False),
        "semantic_embedding.weight":                    (gguf.MODEL_TENSOR.A_GEN_CODE_BLK_EMBD, False),
        "codebook_embedding.weight":                    (gguf.MODEL_TENSOR.A_GEN_CODE_BLK_EMBD, False),
        "final_layernorm.weight":                       (gguf.MODEL_TENSOR.A_GEN_CODE_BLK_OUT_NORM, True),
        "transformer_layer.attention_norm.weight":      (gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM, True),
        "transformer_layer.ffn_norm.weight":            (gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM, True),
        "transformer_layer.attention.q_norm.weight":    (gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q_NORM, True),
        "transformer_layer.attention.k_norm.weight":    (gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K_NORM, True),
        "transformer_layer.attention.wo.weight":        (gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT, False),
        "transformer_layer.feed_forward.w1.weight":     (gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE, False),
        "transformer_layer.feed_forward.w3.weight":     (gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP, False),
        "transformer_layer.feed_forward.w2.weight":     (gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN, False),
    }

    _heads: dict[int, Tensor]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heads = {}

    def get_audio_config(self) -> dict[str, Any] | None:
        cfg = dict(self.global_config["audio_decoder_config"])
        cfg["num_hidden_layers"] = cfg["shared_enrichment_layers"] + cfg["num_codebooks"] - 1
        return cfg

    def set_gguf_parameters(self):
        cfg = self.hparams
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(self.gen_projector_type)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(cfg["dim"])
        self.gguf_writer.add_gen_audio_feed_forward_length(cfg["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(cfg["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(cfg["n_head"])
        self.gguf_writer.add_gen_audio_head_count_kv(cfg["n_local_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(cfg["norm_eps"])
        # other tower and codec hparams are hardcoded in tools/mtmd
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

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]
        if name.startswith("audio_decoder") or name == "model.language_model.embed_tokens.weight":
            return item
        return None

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the VQ picks the nearest codebook entry, small errors in the encoder change the codes
        if new_name.startswith("a.codec."):
            return gguf.GGMLQuantizationType.F32
        # codec transformers: F16 gives codes closer to the reference than BF16
        if new_name.startswith("a.blk.") and n_dims == 2:
            return gguf.GGMLQuantizationType.F16
        # conv1d/conv1d_dw kernels must be F16, ggml_conv_1d(_dw) has no BF16 path
        if new_name.endswith(".weight") and (
            new_name in ("a.gen.wav.dac.entry.weight", "a.gen.wav.dac.post_conv.weight")
            or (".up.blk." in new_name and new_name.endswith(".dwconv.weight"))
            or (".dac.blk." in new_name and (new_name.endswith(".conv1.weight") or new_name.endswith(".conv2.weight")))
        ):
            return gguf.GGMLQuantizationType.F16
        # keep ConvTranspose1d kernels and folded codebook tables in F32
        if new_name.endswith(".conv.weight") and (".up.blk." in new_name or ".dac.blk." in new_name):
            return gguf.GGMLQuantizationType.F32
        if new_name.startswith("a.gen.wav.quant."):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        cfg = self.hparams

        if name.startswith("a."):
            yield (name, data_torch)
            return

        if name == "model.language_model.embed_tokens.weight":
            # semantic band of the backbone embeddings
            begin = self.global_config["semantic_start_token_id"]
            yield (self.format_tensor_name(T.A_GEN_CODE_OUT_EMBD), data_torch[begin:begin + cfg["vocab_size"]])
            return

        if name == "audio_decoder.codebook_embeddings.weight":
            n_cb = cfg["num_codebooks"]
            yield (self.format_tensor_name(T.A_GEN_CODE_EMBD), data_torch.reshape(n_cb, -1, data_torch.shape[-1]))
            return

        if name.startswith("audio_decoder_mtp.heads."):
            n_heads = cfg["num_codebooks"] - 1
            self._heads[int(name.split(".")[2])] = data_torch
            if len(self._heads) < n_heads:
                return
            yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), torch.stack([self._heads.pop(i) for i in range(n_heads)]))
            return

        for prefix, offset in ((_TOWER_SHARED, 0), (_TOWER_RESIDUAL, cfg["shared_enrichment_layers"])):
            if not name.startswith(prefix):
                continue
            idx, key = name[len(prefix):].split(".", 1)
            il = int(idx) + offset
            if key == "transformer_layer.attention.wqkv.weight":
                # rows are [query | gate | key | value]
                n_q = cfg["n_head"] * cfg["head_dim"]
                n_kv = cfg["n_local_heads"] * cfg["head_dim"]
                q, gate, k, v = data_torch.split([n_q, n_q, n_kv, n_kv], dim=0)
                yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_Q, il), q)
                yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_GATE, il), gate)
                yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_K, il), k)
                yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_V, il), v)
                return
            tensor, zero_centered = self._LAYER_MAP[key]
            if zero_centered:
                data_torch = data_torch.float() + 1
            yield (self.format_tensor_name(tensor, il), data_torch)
            return

        raise ValueError(f"unexpected tensor {name}")

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # Fish DAC decoder, then the shared codec encoder for voice cloning
        sd = load_codec(self.dir_model)
        T = gguf.MODEL_TENSOR

        def get(name: str) -> Tensor:
            return sd[name].float()

        def conv(prefix: str) -> Tensor:
            return fold_conv(sd, prefix)

        def quant_table(prefix: str) -> Tensor:
            # fold out_proj into the codebook, so decoding is a row lookup
            w = fold_wn(sd[prefix + ".out_proj.weight_g"], sd[prefix + ".out_proj.weight_v"])
            return get(prefix + ".codebook.weight") @ w.T + get(prefix + ".out_proj.bias")

        q = "quantizer."
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_CB), quant_table(q + "semantic_quantizer.quantizers.0"))
        n_rest = sum(1 for k in sd if k.startswith(q + "quantizer.quantizers.") and k.endswith(".codebook.weight"))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_CB),
               torch.stack([quant_table(f"{q}quantizer.quantizers.{i}") for i in range(n_rest)]))

        # post_module uses interleaved RoPE, permute q/k rows so NEOX RoPE gives the same result
        tfm = q + "post_module."
        n_head, d_head = 16, 64
        order = torch.cat([torch.arange(0, d_head, 2), torch.arange(1, d_head, 2)])

        def neox_rows(w: Tensor) -> Tensor:
            return w.reshape(n_head, d_head, -1)[:, order].reshape(w.shape)

        n_tfm = sum(1 for k in sd if k.startswith(tfm + "layers.") and k.endswith(".ffn_norm.weight"))
        for il in range(n_tfm):
            p = f"{tfm}layers.{il}."
            wq, wk, wv = get(p + "attention.wqkv.weight").chunk(3, dim=0)
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_Q, il), neox_rows(wq))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_K, il), neox_rows(wk))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_V, il), wv)
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_OUT, il), get(p + "attention.wo.weight"))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_NORM, il), get(p + "attention_norm.weight"))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_ATTN_SCALE, il), get(p + "attention_layer_scale.gamma"))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_FFN_NORM, il), get(p + "ffn_norm.weight"))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_FFN_GATE, il), get(p + "feed_forward.w1.weight"))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_FFN_UP, il), get(p + "feed_forward.w3.weight"))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_FFN_DOWN, il), get(p + "feed_forward.w2.weight"))
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_FFN_SCALE, il), get(p + "ffn_layer_scale.gamma"))
        yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUTPUT_NORM), get(tfm + "norm.weight"))

        # upsample: 2x (ConvTranspose1d + ConvNeXt)
        for il in range(2):
            p = f"{q}upsample.{il}."
            yield (self.format_tensor_name(T.A_GEN_WAV_UP_CONV, il, suffix=".weight"), conv(p + "0"))
            yield (self.format_tensor_name(T.A_GEN_WAV_UP_CONV, il, suffix=".bias"), get(p + "0.conv.bias"))
            yield (self.format_tensor_name(T.A_GEN_WAV_UP_DWCONV, il, suffix=".weight"), conv(p + "1.dwconv"))
            yield (self.format_tensor_name(T.A_GEN_WAV_UP_DWCONV, il, suffix=".bias"), get(p + "1.dwconv.conv.bias"))
            for key, tensor in (("norm", T.A_GEN_WAV_UP_NORM), ("pwconv1", T.A_GEN_WAV_UP_PW1), ("pwconv2", T.A_GEN_WAV_UP_PW2)):
                yield (self.format_tensor_name(tensor, il, suffix=".weight"), get(f"{p}1.{key}.weight"))
                yield (self.format_tensor_name(tensor, il, suffix=".bias"), get(f"{p}1.{key}.bias"))
            yield (self.format_tensor_name(T.A_GEN_WAV_UP_GAMMA, il, suffix=""), get(p + "1.gamma"))

        # DAC: entry conv, 4x (Snake, ConvTranspose1d, 3 residual units), Snake, post conv
        d = "decoder.model."
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix=".weight"), conv(d + "0"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix=".bias"), get(d + "0.conv.bias"))
        for bid in range(4):
            p = f"{d}{bid + 1}.block."
            a, b = fold_snake(get(p + "0.alpha"))
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".alpha"), a)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".beta"), b)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".weight"), conv(p + "1"))
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".bias"), get(p + "1.conv.bias"))
            for xid in range(3):
                r = f"{p}{xid + 2}.block."
                for act, tensor in (("0", T.A_GEN_WAV_DAC_RES_ACT1), ("2", T.A_GEN_WAV_DAC_RES_ACT2)):
                    a, b = fold_snake(get(r + act + ".alpha"))
                    name = gguf.TENSOR_NAMES[tensor].format(bid=bid, xid=xid)
                    yield (name + ".alpha", a)
                    yield (name + ".beta", b)
                for cv, tensor in (("1", T.A_GEN_WAV_DAC_RES_CONV1), ("3", T.A_GEN_WAV_DAC_RES_CONV2)):
                    name = gguf.TENSOR_NAMES[tensor].format(bid=bid, xid=xid)
                    yield (name + ".weight", conv(r + cv))
                    yield (name + ".bias", get(r + cv + ".conv.bias"))
        a, b = fold_snake(get(d + "5.alpha"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".alpha"), a)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".beta"), b)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".weight"), conv(d + "6"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".bias"), get(d + "6.conv.bias"))

        yield from codec_encoder_tensors(sd, self.hparams["num_codebooks"])
