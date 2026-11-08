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
    has_audio_encoder = False

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
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.DRAMA3_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(cfg["dim"])
        self.gguf_writer.add_gen_audio_feed_forward_length(cfg["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(cfg["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(cfg["n_head"])
        self.gguf_writer.add_gen_audio_head_count_kv(cfg["n_local_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(cfg["norm_eps"])
        # other tower and codec hparams are hardcoded in clip.cpp

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]
        if name.startswith("audio_decoder") or name == "model.language_model.embed_tokens.weight":
            return item
        return None

    def tensor_force_quant(self, name, new_name, bid, n_dims):
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

        if name.startswith("a.gen.wav."):
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
        # only the DAC decoder, the encoder is for voice cloning (not supported yet)
        sd = torch.load(self.dir_model / "codec.pth", map_location="cpu", mmap=True, weights_only=True)
        sd = sd.get("state_dict", sd)
        T = gguf.MODEL_TENSOR

        def get(name: str) -> Tensor:
            return sd[name].float()

        def conv(prefix: str) -> Tensor:
            # fold weight norm
            if prefix + ".conv.weight" in sd:
                return get(prefix + ".conv.weight")
            g = get(prefix + ".conv.parametrizations.weight.original0")
            v = get(prefix + ".conv.parametrizations.weight.original1")
            return g * v / v.norm(dim=(1, 2), keepdim=True)

        def quant_table(prefix: str) -> Tensor:
            # fold out_proj into the codebook, so decoding is a row lookup
            g = get(prefix + ".out_proj.weight_g")
            v = get(prefix + ".out_proj.weight_v")
            w = (g * v / v.norm(dim=(1, 2), keepdim=True)).squeeze(-1)
            return get(prefix + ".codebook.weight") @ w.T + get(prefix + ".out_proj.bias")

        def snake(alpha: Tensor) -> tuple[Tensor, Tensor]:
            alpha = alpha.flatten()
            return alpha, 1.0 / (alpha + 1e-9)

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
            a, b = snake(get(p + "0.alpha"))
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".alpha"), a)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".beta"), b)
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".weight"), conv(p + "1"))
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=".bias"), get(p + "1.conv.bias"))
            for xid in range(3):
                r = f"{p}{xid + 2}.block."
                for act, tensor in (("0", T.A_GEN_WAV_DAC_RES_ACT1), ("2", T.A_GEN_WAV_DAC_RES_ACT2)):
                    a, b = snake(get(r + act + ".alpha"))
                    name = gguf.TENSOR_NAMES[tensor].format(bid=bid, xid=xid)
                    yield (name + ".alpha", a)
                    yield (name + ".beta", b)
                for cv, tensor in (("1", T.A_GEN_WAV_DAC_RES_CONV1), ("3", T.A_GEN_WAV_DAC_RES_CONV2)):
                    name = gguf.TENSOR_NAMES[tensor].format(bid=bid, xid=xid)
                    yield (name + ".weight", conv(r + cv))
                    yield (name + ".bias", get(r + cv + ".conv.bias"))
        a, b = snake(get(d + "5.alpha"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".alpha"), a)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".beta"), b)
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".weight"), conv(d + "6"))
        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=".bias"), get(d + "6.conv.bias"))
