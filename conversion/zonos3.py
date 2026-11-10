from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, LazyTorchTensor, gguf, logger

LM_DIR = Path("lm") / "zonos3_mini"

# codec decoder: quantizers and semantic head from the generator, the rest from the head checkpoint
CODEC_GENERATOR = Path("autoencoder") / "generator.pt"
CODEC_VARIANT   = "keep2_plain"
CODEC_HEAD      = Path("autoencoder") / "heads" / CODEC_VARIANT / "step_0050000.pt"
CODEC_STAGES    = [0, 2, 4, 6, 8]  # transformer stages of the decoder

DEFAULT_VOICE = "Savannah_en-American_F_1"

# speaker encoder (ownmect_s2), MECT_B2 blocks: (cnn depth, out channels, (stride f, stride t), expansion)
SPK_CKPT   = Path("speaker_encoders") / "ownmect_s2" / "ckpt_0037.pth"
SPK_STAGES = [
    [(3, 32, (1, 1), 4)] * 2,
    [(3, 64, (2, 1), 2)] + [(3, 64, (1, 2), 2)] * 3,
    [(3, 128, (2, 1), 2)] + [(3, 128, (1, 2), 2)] * 4,
    [(3, 256, (2, 1), 1)] + [(2, 256, (1, 2), 1)] * 2,
]
VOICE_MAX_FRAMES = 900  # reference_max_frames of the reference

# GDN geometry, hard-coded in the reference (zonos3.reference.lm)
GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL = 4, 128, 4

# the text table keeps the tokenizer ids shifted by this offset, ids 2 and 3 are BOS and EOS
TEXT_OFFSET = 4
TEXT_BOS, TEXT_EOS = 2, 3


def _round_up(value: float, multiple: int) -> int:
    return multiple * ((int(value) + multiple - 1) // multiple)


def _load_params(dir_model: Path) -> dict[str, Any] | None:
    path = dir_model / LM_DIR / "params.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        params = json.load(f).get("model", {})
    return params if params.get("model_type") == "zonos2" else None


def _is_zonos3_checkpoint(dir_model: Path) -> bool:
    return _load_params(dir_model) is not None


@ModelBase.register_hparams_loader(_is_zonos3_checkpoint)
def _load_zonos3_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected ZONOS3 checkpoint")
    params = _load_params(dir_model)
    assert params is not None
    for key, want in (("attention_pattern", "gdn3_global1"), ("residual_type", "full_attnres"),
                      ("moe_router_type", "linear"), ("moe_router_use_eda", False), ("use_xsa", False),
                      ("moe_end_from_layer", 0), ("speaker_enabled", True)):
        if params.get(key) != want:
            raise NotImplementedError(f"ZONOS3: unsupported {key} = {params.get(key)!r}, expected {want!r}")
    text_config = dict(params, hidden_size=params["dim"])
    return {"architectures": ["Zonos3Model"], "text_config": text_config, "audio_config": {"num_hidden_layers": 0}}


def _index_lm_tensors(model: ModelBase) -> dict[str, Callable[[], Tensor]]:
    path = model.dir_model / LM_DIR / "model.pth"
    logger.info(f"gguf: indexing model part '{LM_DIR / 'model.pth'}'")
    state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    tensors: dict[str, Callable[[], Tensor]] = {}
    for name, data in state.items():
        gen = (lambda data=data: LazyTorchTensor.from_eager(data)) if model.lazy else (lambda data=data: data)
        titem = model.filter_tensors((name, gen))
        model._source_tensor_names[name] = titem[0] if titem else None
        if titem:
            tensors[titem[0]] = titem[1]
    return tensors


@ModelBase.register("Zonos3Model")
class Zonos3TextModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ZONOS3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        self.n_embd       = hp["dim"]
        self.n_codebooks  = hp["n_codebooks"]
        self.audio_vocab  = hp["codebook_size"] + 2  # codes, EOA, pad
        self.audio_pad_id = hp["audio_pad_id"]
        self.text_rows    = hp["text_vocab"] + 1     # + the padding row, used by the generated rows
        self.cond_counts  = [n + 1 for n in hp["condition_bucket_counts"]]  # + the unknown bucket
        self.gdn_period   = int(hp["attention_pattern"][len("gdn"):].split("_")[0]) + 1
        self.n_ff         = _round_up(hp["ffn_dim_multiplier"] * hp["dim"], hp["multiple_of"])
        self.n_ff_exp     = _round_up(hp["moe_ffn_dim_multiplier"] * hp["dim"], hp["moe_multiple_of"])
        self.n_latent     = int(hp["dim"] * hp["moe_alpha"])
        self._embd: dict[str, Tensor] = {}
        self._router: dict[int, dict[str, Tensor]] = {}

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        return _index_lm_tensors(self)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # audio heads and speaker projection go to the mmproj
        if name.startswith(("multi_output.", "speaker_projection.")):
            return None
        return item

    def set_vocab(self):
        with open(self.dir_model / LM_DIR / "tokenizer" / "superbpe8192.json", encoding="utf-8") as f:
            tok = json.load(f)
        model = tok["model"]
        if model["type"] != "BPE" or model.get("byte_fallback") or tok.get("normalizer") is not None:
            raise NotImplementedError("ZONOS3: unexpected tokenizer type")

        vocab: dict[str, int] = model["vocab"]
        n_bpe = len(vocab)
        if n_bpe + TEXT_OFFSET != self.hparams["text_vocab"]:
            raise ValueError(f"ZONOS3: tokenizer has {n_bpe} tokens, text_vocab is {self.hparams['text_vocab']}")
        added = {t["id"] for t in tok.get("added_tokens", [])}

        tokens: list[str] = ["<|unused_0|>", "<|unused_1|>", "<|bos|>", "<|eos|>"]
        toktypes: list[int] = [gguf.TokenType.CONTROL] * TEXT_OFFSET
        inv = {i: t for t, i in vocab.items()}
        for i in range(n_bpe):
            tokens.append(inv[i])
            toktypes.append(gguf.TokenType.CONTROL if i in added else gguf.TokenType.NORMAL)

        # padding row of the text table (the text column of every generated row)
        tokens.append("<|text_pad|>")
        toktypes.append(gguf.TokenType.CONTROL)

        # condition rows, the last bucket of each table is "unknown"
        for cond, n in zip(self.hparams["condition_names"], self.cond_counts):
            tokens += [f"<|{cond}_{i}|>" for i in range(n - 1)] + [f"<|{cond}_unk|>"]
            toktypes += [gguf.TokenType.CONTROL] * n

        merges = [" ".join(m) if isinstance(m, list) else m for m in model["merges"]]

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("zonos3")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)
        self.gguf_writer.add_bos_token_id(TEXT_BOS)
        self.gguf_writer.add_eos_token_id(TEXT_EOS)
        self.gguf_writer.add_pad_token_id(TEXT_OFFSET + 1)  # <|pad|> of the tokenizer
        # the reference wraps the text as [BOS] + ids + [EOS]
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)

    def set_gguf_parameters(self):
        hp = self.hparams
        n_layer = self.block_count
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_block_count(n_layer)
        self.gguf_writer.add_context_length(hp["max_seqlen"])
        self.gguf_writer.add_embedding_length(self.n_embd)
        self.gguf_writer.add_feed_forward_length(self.n_ff)
        self.gguf_writer.add_head_count(hp["n_heads"])
        self.gguf_writer.add_head_count_kv(hp["n_kv_heads"])
        self.gguf_writer.add_key_length(hp["head_dim"])
        self.gguf_writer.add_value_length(hp["head_dim"])
        self.gguf_writer.add_rope_dimension_count(hp["head_dim"])
        self.gguf_writer.add_rope_freq_base(hp["rope_theta"])
        self.gguf_writer.add_layer_norm_rms_eps(hp["norm_eps"])
        self.gguf_writer.add_full_attention_interval(self.gdn_period)

        # gated delta net
        self.gguf_writer.add_ssm_conv_kernel(GDN_KERNEL)
        self.gguf_writer.add_ssm_inner_size(GDN_HEADS * GDN_HEAD_DIM)
        self.gguf_writer.add_ssm_state_size(GDN_HEAD_DIM)
        self.gguf_writer.add_ssm_time_step_rank(GDN_HEADS)
        self.gguf_writer.add_ssm_group_count(GDN_HEADS)

        # latent MoE, the gating (softmax over the selected logits) is hard-coded in libllama
        self.gguf_writer.add_leading_dense_block_count(hp["moe_start_from_layer"])
        self.gguf_writer.add_expert_count(hp["moe_n_experts"])
        self.gguf_writer.add_expert_used_count(hp["moe_router_topk"])
        self.gguf_writer.add_expert_feed_forward_length(self.n_ff_exp)
        self.gguf_writer.add_moe_latent_size(self.n_latent)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # router logits pick the experts, keep them exact
        if new_name.endswith(("ffn_gate_inp.weight", "exp_probs_b.bias")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _token_embd(self) -> Tensor:
        n_cb = self.n_codebooks
        pad = sum(self._embd[f"multi_embedder.embedders.{cb}.weight"][self.audio_pad_id].float() for cb in range(n_cb))
        text = self._embd[f"multi_embedder.embedders.{n_cb}.weight"].float()
        if text.shape[0] != self.text_rows:
            raise ValueError(f"ZONOS3: text table has {text.shape[0]} rows, expected {self.text_rows}")
        conds = [self._embd[f"condition_embeddings.{i}.weight"].float() for i in range(len(self.cond_counts))]
        for t, n in zip(conds, self.cond_counts):
            assert t.shape[0] == n
        # a text row is [pad x n_codebooks, text_id], condition rows replace the summed row
        return torch.cat([text + pad] + conds, dim=0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        hp = self.hparams
        n_layer = self.block_count

        if name.startswith(("multi_embedder.", "condition_embeddings.")):
            self._embd[name] = data_torch
            if len(self._embd) == self.n_codebooks + 1 + len(self.cond_counts):
                yield (self.format_tensor_name(T.TOKEN_EMBD), self._token_embd())
                self._embd.clear()
            return

        if name == "block_attnres_queries":
            # one query per routing point: 2*il before the mixer, 2*il+1 before the FFN, the last one before out_norm
            assert data_torch.shape[0] == 2 * n_layer + 1
            for il in range(n_layer):
                yield (self.format_tensor_name(T.ATTN_RES_SCORE, il), data_torch[2 * il])
                yield (self.format_tensor_name(T.FFN_RES_SCORE, il), data_torch[2 * il + 1])
            yield (self.format_tensor_name(T.OUTPUT_RES_SCORE), data_torch[2 * n_layer])
            return

        if name == "out_norm.weight":
            yield (self.format_tensor_name(T.OUTPUT_NORM), data_torch)
            return

        assert bid is not None, name
        key = name.split(".", 2)[2]  # strip "layers.{bid}."

        if key == "attention_norm.weight":
            yield (self.format_tensor_name(T.ATTN_NORM, bid), data_torch)
            return
        if key == "ffn_norm.weight":
            yield (self.format_tensor_name(T.FFN_NORM, bid), data_torch)
            return

        # gated delta net
        if key == "attention.in_proj.weight":
            # split the fused [qkv | z | a | b] weight into the qwen35 GDN tensors
            # attn_gate holds z, per-head output gate in the attention layers
            inner = GDN_HEADS * GDN_HEAD_DIM
            qkv, z, a, b = data_torch.split([3 * inner, inner, GDN_HEADS, GDN_HEADS], dim=0)
            yield (self.format_tensor_name(T.ATTN_QKV, bid), qkv)
            yield (self.format_tensor_name(T.ATTN_GATE, bid), z)
            yield (self.format_tensor_name(T.SSM_ALPHA, bid), a)
            yield (self.format_tensor_name(T.SSM_BETA, bid), b)
            return
        if key == "attention.conv.weight":
            yield (self.format_tensor_name(T.SSM_CONV1D, bid), data_torch.squeeze(1))
            return
        if key == "attention.A_log":
            yield (self.format_tensor_name(T.SSM_A, bid, suffix=""), -torch.exp(data_torch.float()))
            return
        if key == "attention.dt_bias":
            yield (self.format_tensor_name(T.SSM_DT, bid, suffix=".bias"), data_torch)
            return
        if key == "attention.out_weight":
            yield (self.format_tensor_name(T.SSM_NORM, bid), data_torch)
            return
        if key == "attention.wo.weight" and (bid + 1) % self.gdn_period != 0:
            yield (self.format_tensor_name(T.SSM_OUT, bid), data_torch)
            return

        # global attention
        if key == "attention.wq.weight":
            yield (self.format_tensor_name(T.ATTN_Q, bid), data_torch)
            return
        if key == "attention.wkv.weight":
            yield (self.format_tensor_name(T.ATTN_K, bid), data_torch[0])
            yield (self.format_tensor_name(T.ATTN_V, bid), data_torch[1])
            return
        if key == "attention.wo.weight":
            yield (self.format_tensor_name(T.ATTN_OUT, bid), data_torch)
            return
        if key == "attention.gater.weight":
            # one sigmoid gate per head, same tensor name as the GDN z gate
            yield (self.format_tensor_name(T.ATTN_GATE, bid), data_torch)
            return
        if key == "attention.temp":
            # Q = rms_norm(q) * |temp|, the per-head scale is stored as the Q norm weight
            temp = data_torch.float().abs().reshape(hp["n_heads"], 1).expand(hp["n_heads"], hp["head_dim"])
            yield (self.format_tensor_name(T.ATTN_Q_NORM, bid), temp.contiguous())
            return

        # dense FFN (layer 0): out = w_out(w_in[0](x) * silu(w_in[1](x)))
        if key == "feed_forward.w_in.weight":
            yield (self.format_tensor_name(T.FFN_UP, bid), data_torch[0])
            yield (self.format_tensor_name(T.FFN_GATE, bid), data_torch[1])
            return
        if key == "feed_forward.w_out.weight":
            yield (self.format_tensor_name(T.FFN_DOWN, bid), data_torch)
            return

        # latent MoE
        if key in ("feed_forward.router.router_linear.weight", "feed_forward.router.rmsnorm_eda.weight"):
            # logits = router_linear(rms_norm(x) * w), the norm weight is folded into the router columns
            r = self._router.setdefault(bid, {})
            r[key] = data_torch
            if len(r) == 2:
                w = r["feed_forward.router.router_linear.weight"].float() * r["feed_forward.router.rmsnorm_eda.weight"].float()[None, :]
                yield (self.format_tensor_name(T.FFN_GATE_INP, bid), w)
                del self._router[bid]
            return
        if key == "feed_forward.router.balancing_biases":
            # the reference selects on logits - bias
            yield (self.format_tensor_name(T.FFN_EXP_PROBS_B, bid, suffix=".bias"), -data_torch.float())
            return
        if key == "feed_forward.latent_down.weight":
            yield (self.format_tensor_name(T.MOE_LATENT_DOWN, bid), data_torch)
            return
        if key == "feed_forward.latent_up.weight":
            yield (self.format_tensor_name(T.MOE_LATENT_UP, bid), data_torch)
            return
        if key == "feed_forward.experts.gate_up_proj":
            gate, up = data_torch.chunk(2, dim=1)
            yield (self.format_tensor_name(T.FFN_GATE_EXP, bid), gate)
            yield (self.format_tensor_name(T.FFN_UP_EXP, bid), up)
            return
        if key == "feed_forward.experts.down_proj":
            yield (self.format_tensor_name(T.FFN_DOWN_EXP, bid), data_torch)
            return

        raise ValueError(f"ZONOS3: unexpected tensor {name}")


@ModelBase.register("Zonos3Model")
class Zonos3MmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.global_config["text_config"]
        self.n_codebooks = hp["n_codebooks"]
        self.text_vocab  = hp["text_vocab"]
        self._embd: dict[str, Tensor] = {}

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        return _index_lm_tensors(self)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.startswith(("multi_output.", "multi_embedder.", "speaker_projection.")):
            return item
        return None

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        # speaker encoder for voice cloning, the front-end hparams are hard-coded in tools/mtmd
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.ZONOS3_SPKENC)
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_audio_num_mel_bins(80)
        self.gguf_writer.add_audio_block_count(0)
        self.gguf_writer.add_audio_embedding_length(192)
        self.gguf_writer.add_audio_head_count(1)
        self.gguf_writer.add_audio_feed_forward_length(192)
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.ZONOS3_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(self.n_embd_text)
        self.gguf_writer.add_gen_audio_feed_forward_length(0)
        self.gguf_writer.add_gen_audio_block_count(0)
        self.gguf_writer.add_gen_audio_head_count(0)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(self.global_config["text_config"]["norm_eps"])
        self.gguf_writer.add_gen_audio_model_variant(CODEC_VARIANT)
        if any(self.dir_model.glob(f"gguf/voices/*/{DEFAULT_VOICE}.codes.npy")):
            self.gguf_writer.add_gen_audio_default_voice(DEFAULT_VOICE)
        # softcap and the codec hparams are hard-coded in tools/mtmd

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # read on the host, or used by get_rows / ggml_ssm_conv / the conv ops of the speaker encoder
        if new_name.startswith(("voice.", "a.spk.")) or new_name.endswith((".table", ".dw.weight", ".mixer.weight", ".resample.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @staticmethod
    def _wn(sd: dict[str, Tensor], prefix: str) -> Tensor:
        # torch weight norm, over every dim but 0
        g, v = sd[prefix + ".weight_g"].float(), sd[prefix + ".weight_v"].float()
        return (g * v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))).squeeze(-1)

    def _speaker_encoder(self) -> Iterable[tuple[str, Tensor]]:
        # same layout as the reference ggml runtime (tools/convert_gguf.py speaker-encoder)
        sd = torch.load(self.dir_model / SPK_CKPT, map_location="cpu", weights_only=True)["modules"]["spk_model"]
        sd = {k[len("model."):]: v.double() for k, v in sd.items() if not k.endswith("num_batches_tracked")}
        p = "a.spk"

        def bn(dst: str, src: str) -> Iterable[tuple[str, Tensor]]:
            scale = sd[src + ".weight"] / torch.sqrt(sd[src + ".running_var"] + 1e-5)
            yield (f"{p}.{dst}.scale", scale)
            yield (f"{p}.{dst}.shift", sd[src + ".bias"] - sd[src + ".running_mean"] * scale)

        def put(dst: str, t: Tensor) -> tuple[str, Tensor]:
            return (f"{p}.{dst}", t)

        fe = "feature_extractor."
        yield put("fe.head.conv.weight", sd[fe + "head.0.weight"])
        yield put("fe.head.conv.bias",   sd[fe + "head.0.bias"])
        yield from bn("fe.head.bn", fe + "head.1")
        c_in, f_in = 32, 80
        for si, stage in enumerate(SPK_STAGES):
            for k, (depth, out_ch, (s_f, s_t), expansion) in enumerate(stage):
                src, dst = f"{fe}mect_block{si + 1}.{k}.", f"fe.b{si + 1}.{k}."
                for r in range(depth):
                    cs, cd = f"{src}cnn_block.{r}.", f"{dst}cnn.{r}."
                    # grouped 3x3 convs, one depthwise kernel per input slot of a group
                    for conv in ("conv1", "conv2"):
                        w = sd[cs + conv + ".weight"]
                        for j in range(w.shape[1]):
                            yield put(f"{cd}{conv}.w{j}", w[:, j:j + 1])
                    for i in (1, 2):
                        yield put(f"{cd}pw{i}.weight", sd[f"{cs}conv_pw{i}.weight"])
                        yield put(f"{cd}pw{i}.bias",   sd[f"{cs}conv_pw{i}.bias"])
                        yield from bn(f"{cd}bn{i}", f"{cs}bn{i}")
                    if cs + "shortcut.0.weight" in sd:
                        yield put(cd + "sc.weight", sd[cs + "shortcut.0.weight"])
                        yield from bn(cd + "sc.bn", cs + "shortcut.1")
                if expansion > 1:
                    yield put(dst + "reduce.weight", sd[src + "reduce_conv.0.weight"])
                    yield put(dst + "reduce.bias",   sd[src + "reduce_conv.0.bias"])
                    yield from bn(dst + "reduce.bn", src + "reduce_conv.1")
                yield put(dst + "rproj.weight",    sd[src + "reduce_project.0.weight"].squeeze(-1))
                yield put(dst + "rproj.bias",      sd[src + "reduce_project.0.bias"])
                yield put(dst + "rproj.ln.weight", sd[src + "reduce_project.1.weight"])
                yield put(dst + "rproj.ln.bias",   sd[src + "reduce_project.1.bias"])
                yield put(dst + "pos.dw.weight",   sd[src + "pos_conv.pos_conv.0.weight"].squeeze(1))
                yield put(dst + "pos.dw.bias",     sd[src + "pos_conv.pos_conv.0.bias"])
                yield from bn(dst + "pos.bn", src + "pos_conv.pos_conv.1")
                yield put(dst + "pos.pw.weight",   sd[src + "pos_conv.pos_conv.3.weight"].squeeze(-1))
                yield put(dst + "pos.pw.bias",     sd[src + "pos_conv.pos_conv.3.bias"])
                tr = src + "transformer."
                for a, b in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")):
                    yield put(f"{dst}attn.{a}.weight", sd[f"{tr}attention.{b}.weight"])
                    yield put(f"{dst}attn.{a}.bias",   sd[f"{tr}attention.{b}.bias"])
                yield put(dst + "ln1.weight", sd[tr + "layer_norm.weight"])
                yield put(dst + "ln1.bias",   sd[tr + "layer_norm.bias"])
                moe = tr + "feed_forward.intermediate_dense.token_moe."
                yield put(dst + "moe.gate.weight",    sd[moe + "gate.weight"])
                yield put(dst + "moe.experts.weight", sd[moe + "experts.weight"])
                yield put(dst + "ffn_out.weight", sd[tr + "feed_forward.output_dense.weight"])
                yield put(dst + "ffn_out.bias",   sd[tr + "feed_forward.output_dense.bias"])
                yield put(dst + "ln2.weight", sd[tr + "final_layer_norm.weight"])
                yield put(dst + "ln2.bias",   sd[tr + "final_layer_norm.bias"])
                yield put(dst + "eproj.weight", sd[src + "expansion_project.0.weight"].squeeze(-1))
                yield put(dst + "eproj.bias",   sd[src + "expansion_project.0.bias"])
                c_in, f_in = c_in * s_f, f_in // s_f
        # the stage outputs are mixed by softmax weights
        yield put("fe.dense_w", torch.softmax(sd[fe + "dense3.inputs_weights"], dim=1)[0, :, :, 0])
        yield put("pool.l1.weight", sd["encoding_layers.linear1.weight"].squeeze(-1))
        yield put("pool.l1.bias",   sd["encoding_layers.linear1.bias"])
        yield put("pool.l2.weight", sd["encoding_layers.linear2.weight"].squeeze(-1))
        yield put("pool.l2.bias",   sd["encoding_layers.linear2.bias"])
        yield from bn("bn", "bn")
        yield put("linear.weight", sd["linear.weight"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from self._speaker_encoder()
        logger.info(f"gguf: loading codec decoder from '{CODEC_GENERATOR}' and '{CODEC_HEAD}'")
        gen  = torch.load(self.dir_model / CODEC_GENERATOR, map_location="cpu", mmap=True, weights_only=True)["generator"]
        head = torch.load(self.dir_model / CODEC_HEAD,      map_location="cpu", mmap=True, weights_only=True)["trainable"]
        p = "a.gen.wav"

        def convnext(dst: str, src: dict[str, Tensor], prefix: str) -> Iterable[tuple[str, Tensor]]:
            yield (f"{dst}.dw.weight",   src[f"{prefix}.dwconv.weight"].squeeze(1))
            yield (f"{dst}.dw.bias",     src[f"{prefix}.dwconv.bias"])
            yield (f"{dst}.norm.weight", src[f"{prefix}.norm.weight"])
            yield (f"{dst}.norm.bias",   src[f"{prefix}.norm.bias"])
            yield (f"{dst}.pw1.weight",  src[f"{prefix}.pwconv1.weight"])
            yield (f"{dst}.pw1.bias",    src[f"{prefix}.pwconv1.bias"])
            yield (f"{dst}.pw2.weight",  src[f"{prefix}.pwconv2.weight"])
            yield (f"{dst}.pw2.bias",    src[f"{prefix}.pwconv2.bias"])

        # RVQ codebooks folded through their out_proj, the biases are added once
        def rvq(prefix: str) -> tuple[Tensor, Tensor]:
            return gen[prefix + ".codebook.weight"].float() @ self._wn(gen, prefix + ".out_proj").T, gen[prefix + ".out_proj.bias"].float()

        t, b = rvq("quantizer.semantic_quantizer.quantizers.0")
        yield (f"{p}.sem.table", t)
        yield (f"{p}.sem.bias",  b)
        res = [rvq(f"quantizer.quantizer.quantizers.{i}") for i in range(self.n_codebooks - 1)]
        yield (f"{p}.res.table", torch.cat([t for t, _ in res], dim=0))
        # one bias per residual codebook, a frame decoded before all its codes are known leaves the missing ones out
        yield (f"{p}.res.bias",  torch.stack([b for _, b in res]))

        n_sem = sum(1 for k in gen if k.startswith("semantic_head.") and k.endswith(".dwconv.weight"))
        for i in range(n_sem):
            yield from convnext(f"{p}.sem.{i}", gen, f"semantic_head.{i}")
        yield (f"{p}.sem.out.weight", self._wn(gen, f"semantic_head.{n_sem}"))
        yield (f"{p}.sem.out.bias",   gen[f"semantic_head.{n_sem}.bias"])

        names = {"norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias", "qkv.weight": "qkv.weight",
                 "attention_out.weight": "wo.weight", "attention_scale": "attn_scale", "norm2.weight": "norm2.weight",
                 "norm2.bias": "norm2.bias", "ffn.0.weight": "ffn1.weight", "ffn.3.weight": "ffn2.weight", "ffn_scale": "ffn_scale"}
        for si, s in enumerate(CODEC_STAGES):
            yield (f"{p}.stage.{si}.in.weight", head[f"stages.{s}.input_proj.weight"])
            if f"stages.{s}.output_proj.weight" in head:
                yield (f"{p}.stage.{si}.out.weight", head[f"stages.{s}.output_proj.weight"])
            n_layer = sum(1 for k in head if k.startswith(f"stages.{s}.layers.") and k.endswith(".qkv.weight"))
            for il in range(n_layer):
                for src, dst in names.items():
                    yield (f"{p}.stage.{si}.blk.{il}.{dst}", head[f"stages.{s}.layers.{il}.{src}"])

        n_head = sum(1 for k in head if k.startswith("head.blocks.") and k.endswith(".dwconv.weight"))
        for i in range(n_head):
            yield from convnext(f"{p}.head.{i}", head, f"head.blocks.{i}")
        yield (f"{p}.head.norm.weight", head["head.norm.weight"])
        yield (f"{p}.head.norm.bias",   head["head.norm.bias"])
        # both iSTFT branches in one projection
        yield (f"{p}.head.out.weight", torch.cat([head["head.outs.0.weight"], head["head.outs.1.weight"]], dim=0))
        yield (f"{p}.head.out.bias",   torch.cat([head["head.outs.0.bias"],   head["head.outs.1.bias"]],   dim=0))
        # read on the host
        yield (f"{p}.mixer.weight",   head["head.mixer.weight"].squeeze(0))
        yield (f"{p}.resample.weight", gen["to_external.kernel"].squeeze(1))

    # "reference" speaker mode, the default of the reference: the voice is a prefix of delayed code rows
    def _voices(self, tables: list[Tensor]) -> Iterable[tuple[str, Tensor]]:
        n_cb, pad = self.n_codebooks, self.global_config["text_config"]["audio_pad_id"]
        for path in sorted((self.dir_model / "gguf" / "voices").glob("*/*.codes.npy")):
            codes = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.int64))
            if codes.ndim != 2 or codes.shape[1] != n_cb or codes.min() < 0 or codes.max() >= pad - 1:
                logger.warning(f"voice {path.name}: skipped, expected [frames, {n_cb}] codes")
                continue
            codes = codes[:VOICE_MAX_FRAMES]
            n_frames = codes.shape[0]
            # codebook cb of frame t is in row t + cb, the other cells are pad
            rows = torch.full((n_frames + n_cb - 1, n_cb), pad, dtype=torch.int64)
            for cb in range(n_cb):
                rows[cb:cb + n_frames, cb] = codes[:, cb]
            embd = sum(tables[cb][rows[:, cb]] for cb in range(n_cb))
            yield (f"voice.{path.name[:-len('.codes.npy')]}", embd)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR

        if name.startswith(("a.gen.wav.", "a.spk.")):
            yield (name, data_torch)
            return

        # the speaker encoder ends with the speaker projection of the backbone
        if name.startswith("speaker_projection."):
            yield ("a.spk.proj." + name.split(".")[-1], data_torch)
            return

        if name == "multi_output.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), data_torch)
            return

        assert name.startswith("multi_embedder.")
        self._embd[name] = data_torch
        if len(self._embd) < self.n_codebooks + 1:
            return
        # every row built by the mmproj has text id = text_vocab, so that row is folded into codebook 0
        tables = [self._embd[f"multi_embedder.embedders.{cb}.weight"].float() for cb in range(self.n_codebooks)]
        tables[0] = tables[0] + self._embd[f"multi_embedder.embedders.{self.n_codebooks}.weight"][self.text_vocab].float()
        yield (self.format_tensor_name(T.A_GEN_CODE_OUT_EMBD), torch.cat(tables, dim=0))
        yield from self._voices([LazyTorchTensor.to_eager(t) for t in tables])
        self._embd.clear()
