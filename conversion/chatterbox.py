# Chatterbox (ResembleAI) turbo and multilingual v3, from the raw checkpoints of the official repos
# the checkpoints ship no config.json, the variant is detected from the talker file and the hparams come from the reference code
# talker: text_emb and speech_emb are fused into one [text | speech] vocab, speech token i is <|speech_i|>
# mmproj: the s3gen sidecar, the voice and conditioning encoders, the position and speech tables, and the conds.pt voice

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, LazyTorchTensor, gguf, logger
from .llama import LlamaModel

# checkpoint file prefixes, the files are "<prefix>.safetensors"
TURBO_TALKER = "t3_turbo_v1"
MTL_TALKER   = "t3_mtl23ls_v3"
TURBO_S3GEN  = "s3gen_meanflow"
MTL_S3GEN    = "s3gen_v3"
VE           = "ve"

# relative speech ids shared by both variants (start/stop_speech_token in the reference)
SPEECH_BOS = 6561
SPEECH_EOS = 6562

# sampling defaults of the reference, samplers it does not use are disabled so the common defaults do not apply
TURBO_TOP_K = 1000
MTL_TOP_K   = 0
TURBO_SAMPLING = {"min_p": 0.0,  "top_p": 0.95, "temp": 0.8, "penalty_repeat": 1.2}
MTL_SAMPLING   = {"min_p": 0.05, "top_p": 1.0,  "temp": 0.8, "penalty_repeat": 1.2}
# the reference repetition penalty covers the whole generated history
PENALTY_LAST_N = -1


def _has_file(dir_model: Path, prefix: str) -> bool:
    return (dir_model / f"{prefix}.safetensors").is_file()


def _is_turbo(dir_model: Path) -> bool:
    if _has_file(dir_model, TURBO_TALKER):
        return True
    if _has_file(dir_model, MTL_TALKER):
        return False
    raise FileNotFoundError(f"no chatterbox talker checkpoint in {dir_model}")


# talker hparams of the reference (GPT2_MEDIUM_CONFIG / LLAMA_520M_CONFIG_DICT, T3Config)
TURBO_TEXT_CONFIG = {
    "n_layer": 24, "n_embd": 1024, "n_inner": 4096, "n_head": 16, "n_ctx": 8196, "layer_norm_epsilon": 1e-05,
    "vocab_size": 50276, "speech_vocab_size": 6563,
}
MTL_TEXT_CONFIG = {
    "num_hidden_layers": 30, "hidden_size": 1024, "intermediate_size": 4096, "num_attention_heads": 16,
    "num_key_value_heads": 16, "head_dim": 64, "max_position_embeddings": 131072, "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "rope_scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192},
    "vocab_size": 2454, "speech_vocab_size": 8194,
}


def _is_chatterbox_checkpoint(dir_model: Path) -> bool:
    return _has_file(dir_model, VE) and (_has_file(dir_model, TURBO_TALKER) or _has_file(dir_model, MTL_TALKER))


@ModelBase.register_hparams_loader(_is_chatterbox_checkpoint)
def _load_chatterbox_hparams(dir_model: Path) -> dict[str, Any]:
    turbo = _is_turbo(dir_model)
    logger.info(f"gguf: detected Chatterbox {'turbo' if turbo else 'multilingual v3'} checkpoint")
    text_config = dict(TURBO_TEXT_CONFIG if turbo else MTL_TEXT_CONFIG)
    # the mmproj tensors are named explicitly, they do not go through the block tensor map
    return {"architectures": ["ChatterboxModel"], "text_config": text_config, "audio_config": {"num_hidden_layers": 0}}


@ModelBase.register("ChatterboxModel")
class ChatterboxTalkerModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA  # multilingual; the turbo constructor switches to GPT2

    def __init__(self, dir_model: Path, *args, **kwargs):
        self.is_turbo = _is_turbo(dir_model)
        if self.is_turbo:
            self.model_arch = gguf.MODEL_ARCH.GPT2
        super().__init__(dir_model, *args, **kwargs)
        self._text_embd: Tensor | None = None
        self._speech_embd: Tensor | None = None
        self._text_head: Tensor | None = None
        self._speech_head: Tensor | None = None

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        return self.index_local_tensors(TURBO_TALKER if self.is_turbo else MTL_TALKER)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # the reference feeds inputs_embeds, the transformer input embeddings are unused
        if name in ("tfmr.wte.weight", "tfmr.embed_tokens.weight"):
            return None
        if name.startswith(("cond_enc.", "text_pos_emb.", "speech_pos_emb.")):
            return None
        return super().filter_tensors(item)

    def set_vocab(self):
        if self.is_turbo:
            self._set_vocab_turbo()
        else:
            self._set_vocab_mtl()

    def _speech_token_names(self, n_speech: int) -> list[str]:
        return [f"<|speech_{i}|>" for i in range(n_speech)]

    def _set_vocab_turbo(self):
        # stock GPT-2 BPE from the checkpoint dir, extended with the speech tokens
        tokens, toktypes, tokpre = self.get_vocab_base()
        n_text = len(tokens)
        n_speech = self.hparams["speech_vocab_size"]
        speech = self._speech_token_names(n_speech)
        tokens += speech
        toktypes += [gguf.TokenType.CONTROL] * n_speech

        with open(self.dir_model / "merges.txt", "r", encoding="utf-8") as f:
            merges = [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#version")]

        self._write_vocab(tokens, toktypes, merges, tokpre, n_text)

    def _write_vocab(self, tokens: list[str], toktypes: list[int], merges: list[str], tokpre: str, n_text: int):
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)

        self.gguf_writer.add_bos_token_id(n_text + SPEECH_BOS)
        self.gguf_writer.add_eos_token_id(n_text + SPEECH_EOS)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)

        # the reference samples the speech head only
        self.gguf_writer.add_suppress_tokens(list(range(n_text)))

    def _set_vocab_mtl(self):
        # the reference BPE is char-level, it is re-encoded to byte-level with extra merges that rebuild each multi-byte char
        # its lowercasing is folded into the embedding table, so the prompt needs no text preprocessing
        with open(self.dir_model / "mtl_tokenizer.json", "r", encoding="utf-8") as f:
            tok = json.load(f)

        byte_map = gguf.vocab.bytes_to_unicode()

        def enc(s: str) -> str:
            return "".join(byte_map[b] for b in s.encode("utf-8"))

        n_text = self.hparams["vocab_size"]
        tokens: list[str] = [f"[unused_{i}]" for i in range(n_text)]
        toktypes = [int(gguf.TokenType.UNUSED)] * n_text
        char_merges: list[str] = []
        for t, i in tok["model"]["vocab"].items():
            if t == " ":
                # the reference substitutes [SPACE] before encoding, the raw space token is never used
                continue
            tokens[i] = enc(t)
            toktypes[i] = int(gguf.TokenType.NORMAL)
            if len(t) == 1 and len(t.encode("utf-8")) > 1:
                parts = [byte_map[b] for b in t.encode("utf-8")]
                for k in range(1, len(parts)):
                    char_merges.append("".join(parts[:k]) + " " + parts[k])
        for entry in tok.get("added_tokens", []):
            tokens[entry["id"]] = entry["content"]
            toktypes[entry["id"]] = int(gguf.TokenType.CONTROL)

        # raw spaces tokenize to [SPACE] directly
        space_id = tok["model"]["vocab"]["[SPACE]"]
        tokens[space_id] = enc(" ")
        toktypes[space_id] = int(gguf.TokenType.NORMAL)

        n_speech = self.hparams["speech_vocab_size"]
        tokens += self._speech_token_names(n_speech)
        toktypes += [int(gguf.TokenType.CONTROL)] * n_speech

        # chars are atomic in the reference, they must form before any of its merges
        merges = char_merges
        for m in tok["model"].get("merges", []):
            a, b = m if isinstance(m, list) else m.split(" ")
            merges.append(enc(a) + " " + enc(b))

        self._write_vocab(tokens, toktypes, merges, "default", n_text)

    def _mtl_case_fold_pairs(self) -> list[tuple[int, int]]:
        # (upper_id, lower_id) of the single-char vocab entries, for every script in the vocab
        with open(self.dir_model / "mtl_tokenizer.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)["model"]["vocab"]
        pairs: list[tuple[int, int]] = []
        for t, i in vocab.items():
            if len(t) != 1:
                continue
            low = t.lower()
            if low != t and len(low) == 1 and low in vocab:
                pairs.append((i, vocab[low]))
        return pairs

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        sampling = TURBO_SAMPLING if self.is_turbo else MTL_SAMPLING
        self.gguf_writer.add_sampling_top_k(TURBO_TOP_K if self.is_turbo else MTL_TOP_K)
        self.gguf_writer.add_sampling_min_p(sampling["min_p"])
        self.gguf_writer.add_sampling_top_p(sampling["top_p"])
        self.gguf_writer.add_sampling_temp(sampling["temp"])
        self.gguf_writer.add_sampling_penalty_repeat(sampling["penalty_repeat"])
        self.gguf_writer.add_sampling_penalty_last_n(PENALTY_LAST_N)
        if not self.is_turbo:
            self.gguf_writer.add_classifier_free_guidance(True)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self.is_turbo:
            return
        # llama3 rope scaling baked into the rope_freqs factors tensor
        rp = self.hparams["rope_scaling"]
        dim = self.hparams["head_dim"]
        base = self.hparams["rope_theta"]
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        factor = rp["factor"]
        low_freq_wavelen = rp["original_max_position_embeddings"] / rp["low_freq_factor"]
        high_freq_wavelen = rp["original_max_position_embeddings"] / rp["high_freq_factor"]
        rope_factors = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                rope_factors.append(1)
            elif wavelen > low_freq_wavelen:
                rope_factors.append(factor)
            else:
                smooth = (rp["original_max_position_embeddings"] / wavelen - rp["low_freq_factor"]) / (rp["high_freq_factor"] - rp["low_freq_factor"])
                rope_factors.append(1 / ((1 - smooth) / factor + smooth))
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))

    def _maybe_emit_fused(self) -> Iterable[tuple[str, Tensor]]:
        if self._text_embd is not None and self._speech_embd is not None:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD),
                   torch.cat([self._text_embd, self._speech_embd], dim=0))
            self._text_embd = None
            self._speech_embd = None
        if self._text_head is not None and self._speech_head is not None:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT),
                   torch.cat([self._text_head, self._speech_head], dim=0))
            self._text_head = None
            self._speech_head = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # generate_extra_tensors output comes back through here with its final name
        if name.startswith("rope_freqs"):
            yield (name, data_torch)
            return
        # fused [text | speech] vocab: embeddings and output head
        if name == "text_emb.weight":
            if not self.is_turbo:
                # the reference lowercases its input, the uppercase rows take the lowercase ones
                fold = list(range(data_torch.shape[0]))
                for upper, lower in self._mtl_case_fold_pairs():
                    fold[upper] = lower
                data_torch = data_torch[torch.tensor(fold)]
            self._text_embd = data_torch
            yield from self._maybe_emit_fused()
            return
        if name == "speech_emb.weight":
            self._speech_embd = data_torch
            yield from self._maybe_emit_fused()
            return
        if name == "text_head.weight":
            self._text_head = data_torch
            yield from self._maybe_emit_fused()
            return
        if name == "speech_head.weight":
            self._speech_head = data_torch
            yield from self._maybe_emit_fused()
            return
        if name == "speech_head.bias":
            # the text head has no bias, its zone is suppressed at sampling anyway
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT, suffix=".bias"), F.pad(data_torch, (self.hparams["vocab_size"], 0)))
            return

        assert name.startswith("tfmr.")
        name = name[len("tfmr."):]

        if self.is_turbo:
            # HF GPT-2 layout: Conv1D style weights are stored transposed
            if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight")):
                data_torch = data_torch.transpose(1, 0)
            yield (self.map_tensor_name(name), data_torch)
            return

        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, self.hparams["num_attention_heads"], None)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, self.hparams["num_attention_heads"], self.hparams["num_key_value_heads"])
        yield (self.map_tensor_name("model." + name), data_torch)


@ModelBase.register("ChatterboxModel")
class ChatterboxMmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    def __init__(self, dir_model: Path, *args, **kwargs):
        self.is_turbo = _is_turbo(dir_model)
        super().__init__(dir_model, *args, **kwargs)
        self._wnorm_g: dict[str, Tensor] = {}
        self._wnorm_v: dict[str, Tensor] = {}

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("audio_config")

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors: dict[str, Callable[[], Tensor]] = {}
        for prefix in (TURBO_S3GEN if self.is_turbo else MTL_S3GEN, VE, TURBO_TALKER if self.is_turbo else MTL_TALKER):
            tensors.update(self.index_local_tensors(prefix))
        return tensors

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # the s3gen, ve and talker checkpoints have disjoint top-level names
        name, gen = item
        new_name: str | None = None
        # batchnorm bookkeeping (over the gguf name length cap) and the tokenizer dsp buffers are unused
        if name.endswith("num_batches_tracked") or name in ("tokenizer.window", "tokenizer._mel_filters"):
            return None
        for src, dst in (
            ("flow.encoder.",           "a.gen.fenc."),
            ("flow.decoder.estimator.", "a.gen.est."),
            ("mel2wav.",                "a.gen.hift."),
            ("speaker_encoder.",        "a.spk."),
            ("tokenizer.",              "a.s3tok."),
            ("flow.",                   "a.gen.flow."),  # input_embedding, encoder_proj, spk_embed_affine_layer
            ("lstm.",                   "a.ve.lstm."),
            ("proj.",                   "a.ve.proj."),
            ("cond_enc.",               "a.cenc."),
        ):
            if name.startswith(src):
                new_name = dst + name[len(src):]
                break
        if name == "text_pos_emb.emb.weight":
            new_name = "a.gen.t3.text_pos_emb"
        elif name == "speech_pos_emb.emb.weight":
            new_name = "a.gen.t3.speech_pos_emb"
        elif name == "speech_emb.weight":
            new_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.A_GEN_CODE_OUT_EMBD] + ".weight"
        if new_name is None:
            return None
        return super().filter_tensors((new_name, gen))

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        # speaker encoder, the front-end hparams are hard-coded in clip.cpp
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.CHATTERBOX_SPKENC)
        self.gguf_writer.add_audio_projection_dim(80)
        self.gguf_writer.add_audio_num_mel_bins(80)
        self.gguf_writer.add_audio_block_count(0)
        self.gguf_writer.add_audio_embedding_length(192)
        self.gguf_writer.add_audio_head_count(1)
        self.gguf_writer.add_audio_feed_forward_length(192)
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

        # audio generator (s3gen, chatterbox projector); the flow encoder shape
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.CHATTERBOX)
        self.gguf_writer.add_gen_audio_projection_dim(80)
        self.gguf_writer.add_gen_audio_embedding_length(512)
        self.gguf_writer.add_gen_audio_feed_forward_length(2048)
        self.gguf_writer.add_gen_audio_block_count(6)
        self.gguf_writer.add_gen_audio_head_count(8)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-5)

    def _fuse_weight_norm(self, name: str, data_torch: Tensor) -> tuple[str, Tensor] | None:
        # torch weight_norm parametrization: weight = g * v / |v| over dims 1..n
        base = name.split(".parametrizations.weight.original")[0]
        if name.endswith("original0"):
            self._wnorm_g[base] = data_torch
        else:
            self._wnorm_v[base] = data_torch
        if base in self._wnorm_g and base in self._wnorm_v:
            g = self._wnorm_g.pop(base)
            v = self._wnorm_v.pop(base)
            norm = v.float().norm(dim=tuple(range(1, v.dim())), keepdim=True)
            return (base + ".weight", g.float() * v.float() / norm)
        return None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if ".parametrizations.weight.original" in name:
            fused = self._fuse_weight_norm(name, data_torch)
            if fused is not None:
                yield fused
            return
        yield (name, data_torch)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid
        # tensors read on the host stay F32
        if new_name.startswith(("a.ve.", "a.cenc.", "voice.", "a.gen.t3.", "a.gen.hift.m_source.")):
            return gguf.GGMLQuantizationType.F32
        # the conv ops have no BF16 kernels
        if n_dims >= 2 and new_name.endswith((".weight", ".weight_v")):
            return gguf.GGMLQuantizationType.F16
        return gguf.GGMLQuantizationType.F32

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        conds = torch.load(self.dir_model / "conds.pt", map_location="cpu", weights_only=True)
        t3c = conds["t3"] if isinstance(conds, dict) else conds.t3
        genc = conds["gen"] if isinstance(conds, dict) else conds.gen
        t3c = vars(t3c) if not isinstance(t3c, dict) else t3c
        genc = vars(genc) if not isinstance(genc, dict) else genc

        def talker_tensor(name: str) -> Tensor:
            return LazyTorchTensor.to_eager(self.model_tensors[name]())

        # the conds.pt voice: voice.default holds the talker conditioning rows, voice.default.ref_* the flow decoder reference
        # 1D tensors are written as F32, the speech ids stay exact
        yield ("voice.default.ref_tokens", genc["prompt_token"][0].float())
        yield ("voice.default.ref_feat", genc["prompt_feat"][0].float())
        yield ("voice.default.ref_spk", genc["embedding"][0].float())

        spkr_w = talker_tensor("a.cenc.spkr_enc.weight").float()
        spkr_b = talker_tensor("a.cenc.spkr_enc.bias").float()
        spkr_row = spkr_w @ t3c["speaker_emb"][0].float() + spkr_b
        speech_emb = talker_tensor(self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_OUT_EMBD)).float()
        cond_tokens = t3c["cond_prompt_speech_tokens"][0]

        if self.is_turbo:
            # turbo talker conditioning: [spkr, speech rows of the cond tokens]
            yield ("voice.default", torch.cat([spkr_row[None], speech_emb[cond_tokens]], dim=0))
            return

        # multilingual talker conditioning [spkr, perceiver x32, emotion], from the reference perceiver over the cond speech tokens
        with torch.no_grad():
            speech_pos = talker_tensor("a.gen.t3.speech_pos_emb").float()
            pse = speech_emb[cond_tokens] + speech_pos[: cond_tokens.shape[0]]

            ln_w = talker_tensor("a.cenc.perceiver.attn.norm.weight").float()
            ln_b = talker_tensor("a.cenc.perceiver.attn.norm.bias").float()
            wq = talker_tensor("a.cenc.perceiver.attn.to_q.weight").float()
            bq = talker_tensor("a.cenc.perceiver.attn.to_q.bias").float()
            wk = talker_tensor("a.cenc.perceiver.attn.to_k.weight").float()
            bk = talker_tensor("a.cenc.perceiver.attn.to_k.bias").float()
            wv = talker_tensor("a.cenc.perceiver.attn.to_v.weight").float()
            bv = talker_tensor("a.cenc.perceiver.attn.to_v.bias").float()
            wo = talker_tensor("a.cenc.perceiver.attn.proj_out.weight").float()
            bo = talker_tensor("a.cenc.perceiver.attn.proj_out.bias").float()
            query = talker_tensor("a.cenc.perceiver.pre_attention_query")[0].float()

            n_head = 4
            n_e = query.shape[1]

            def attn_block(x1: Tensor, x2: Tensor) -> Tensor:
                nx1 = F.layer_norm(x1, (n_e,), ln_w, ln_b)
                nx2 = F.layer_norm(x2, (n_e,), ln_w, ln_b)
                q = (nx1 @ wq.T + bq).view(-1, n_head, n_e // n_head).transpose(0, 1)
                k = (nx2 @ wk.T + bk).view(-1, n_head, n_e // n_head).transpose(0, 1)
                v = (nx2 @ wv.T + bv).view(-1, n_head, n_e // n_head).transpose(0, 1)
                ctx = F.scaled_dot_product_attention(q, k, v)
                ctx = ctx.transpose(0, 1).reshape(-1, n_e)
                return ctx @ wo.T + bo + x1

            pre = attn_block(query, pse)
            p32 = attn_block(pre, pre)

            emo = talker_tensor("a.cenc.emotion_adv_fc.weight").float()
            emo_row = emo[:, 0] * t3c["emotion_adv"].reshape(-1)[0]
            yield ("voice.default", torch.cat([spkr_row[None], p32, emo_row[None]], dim=0))
