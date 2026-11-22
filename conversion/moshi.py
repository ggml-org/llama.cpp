from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, SentencePieceTokenTypes, TextModel, gguf, logger
from .pockettts import mimi_seanet_tensors, mimi_tfm_tensors

# Moshi family (Kyutai): a backbone over parallel streams (text + generated audio codes + input audio codes),
# one position per 80 ms frame. Only Hibiki (simultaneous speech translation) is supported for now.
# The checkpoint is not in the HF layout: a moshi-style config.json and two safetensors (LM + mimi).
#
# pipeline stage mapping:
#   transformer + text_emb/text_linear  --> normal libllama text model (arch moshi)
#   mimi encoder + RVQ + emb[dep_q:]    --> mtmd audio stream (projector moshi), rows of the input audio
#   depformer + emb[:dep_q]             --> MTMD_GEN_PROCESS_TYPE_GEN_CODE, codes of a frame + their backbone embedding
#   RVQ decode + mimi decoder           --> MTMD_GEN_PROCESS_TYPE_GEN_WAV
#
# Tricks:
# - the initial text token (text_card) is appended to the vocab as <|text_start|>, the head has no logit for it;
#   it only opens the stream, the embeddings of the initial codes of the generated audio are folded into it
# - the condition of Hibiki ("very_good" description, added to every row) is folded into the text embeddings,
#   every row has exactly one text embedding
# - Hibiki ignores an early EOS: the reference gives it the embedding of PAD, done here too
# - low-rank depformer embeddings are folded into full tables

_MIMI_PREFIX = "mimi."
_COND_VALUE = "very_good"


def _read_config(dir_model: Path) -> dict[str, Any] | None:
    path = dir_model / "config.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config if isinstance(config, dict) and "moshi_name" in config else None


def _tensor_shape(path: Path, name: str) -> tuple[int, ...]:
    with gguf.utility.SafetensorsLocal(path) as part:
        return tuple(part[name].shape)


@ModelBase.register_hparams_loader(lambda dir_model: _read_config(dir_model) is not None)
def _load_hparams(dir_model: Path) -> dict[str, Any]:
    config = _read_config(dir_model)
    assert config is not None
    model_type = config.get("model_type", "moshi")
    if model_type != "hibiki":
        raise NotImplementedError(f"moshi checkpoint of type {model_type!r} is not supported yet")
    logger.info(f"gguf: detected a moshi checkpoint ({model_type})")

    # mtmd has the layout of Hibiki: the acoustic codes lag by 2 frames, the depformer has no positional embedding,
    # its steps past the last weight set reuse it, and the initial code is the extra entry after the mimi codebook
    n_q = config["dep_q"]
    schedule = config.get("depformer_weights_per_step_schedule") or list(range(n_q))
    # text, then the generated and the input codes
    if config["delays"] != [0] + ([0] + [2] * (n_q - 1)) * 2:
        raise NotImplementedError(f"unsupported delays {config['delays']}")
    if schedule != [min(i, max(schedule)) for i in range(n_q)] or config.get("depformer_pos_emb", "none") != "none":
        raise NotImplementedError("unsupported depformer layout")
    if config["card"] != 2048 or config["n_q"] != 2 * n_q:
        raise NotImplementedError("unsupported audio codebooks")

    lm_path = dir_model / config["moshi_name"]
    # the gating hidden size is derived from dim_feedforward, read it back from the tensors
    n_ff     = _tensor_shape(lm_path, "transformer.layers.0.gating.linear_out.weight")[1]
    n_ff_dep = _tensor_shape(lm_path, "depformer.layers.0.gating.0.linear_out.weight")[1]
    return {
        "architectures": ["MoshiModel"],
        "model_type": "moshi",
        "moshi_config": config,
        "num_hidden_layers": config["num_layers"],
        "hidden_size": config["dim"],
        "intermediate_size": n_ff,
        "num_attention_heads": config["num_heads"],
        "num_key_value_heads": config["num_heads"],
        "max_position_embeddings": config["context"],
        "sliding_window": config["context"],
        "rope_theta": config["max_period"],
        "rms_norm_eps": 1e-8, # rms_norm_f32
        "vocab_size": config["text_card"] + 1,
        "audio_config": {
            # mimi encoder transformer, constant across the family
            "num_hidden_layers": 8,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
        },
        "depformer_config": {
            "num_hidden_layers": config["depformer_num_layers"],
            "hidden_size": config["depformer_dim"],
            "intermediate_size": n_ff_dep,
            "num_attention_heads": config["depformer_num_heads"],
        },
    }


# the LM and mimi weights are in two files named by the config, mimi tensors get the "mimi." prefix
def _moshi_config(model: ModelBase) -> dict[str, Any]:
    config = model.hparams.get("moshi_config")
    if config is None and isinstance(model, MmprojModel):
        config = model.global_config["moshi_config"] # the mmproj moves the hparams aside
    assert config is not None
    return config


def _index_moshi_tensors(model: ModelBase) -> dict[str, Callable[[], Tensor]]:
    config = _moshi_config(model)
    tensors = model.index_local_tensors(Path(config["moshi_name"]).stem)
    for name, gen in model.index_local_tensors(Path(config["mimi_name"]).stem).items():
        tensors[_MIMI_PREFIX + name] = gen
    return tensors


def _low_rank_embd(model: ModelBase, prefix: str) -> Tensor:
    embd = model.model_tensors[prefix + ".weight"]().float()
    proj = model.model_tensors.get(prefix + ".low_rank.weight")
    return embd @ proj().float().T if proj is not None else embd


def _condition(model: ModelBase, value: str) -> Tensor:
    # LUT conditioner: output_proj(embed[index of value])
    lut = _moshi_config(model)["conditioners"]["description"]["lut"]
    prefix = "condition_provider.conditioners.description."
    embed = model.model_tensors[prefix + "embed.weight"]().float()
    proj  = model.model_tensors[prefix + "output_proj.weight"]().float()
    return embed[lut["possible_values"].index(value)] @ proj.T


@ModelBase.register("MoshiModel")
@ModelBase.example("kyutai/hibiki-1b-pytorch-bf16", "kyutai/hibiki-2b-pytorch-bf16")
class MoshiModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MOSHI

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        assert remote_hf_model_id is None
        return _index_moshi_tensors(self)

    def set_vocab(self):
        from sentencepiece import SentencePieceProcessor

        config = _moshi_config(self)
        sp = SentencePieceProcessor()
        sp.LoadFromFile(str(self.dir_model / config["tokenizer_name"]))
        assert sp.vocab_size() == config["text_card"]

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []
        for i in range(sp.vocab_size()):
            toktype = SentencePieceTokenTypes.NORMAL
            if sp.IsControl(i) or sp.IsUnknown(i):
                # <unk> (0) is the end-of-padding marker of the text stream, it has no text
                toktype = SentencePieceTokenTypes.CONTROL
            elif sp.IsUnused(i):
                toktype = SentencePieceTokenTypes.UNUSED
            elif sp.IsByte(i):
                toktype = SentencePieceTokenTypes.BYTE
            tokens.append(sp.IdToPiece(i).encode("utf-8"))
            scores.append(sp.GetScore(i))
            toktypes.append(toktype)

        # the initial token of the text stream
        tokens.append(b"<|text_start|>")
        scores.append(-1000.0)
        toktypes.append(SentencePieceTokenTypes.CONTROL)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_unk_token_id(sp.unk_id())
        self.gguf_writer.add_bos_token_id(sp.bos_id())
        self.gguf_writer.add_eos_token_id(sp.eos_id())
        self.gguf_writer.add_pad_token_id(sp.pad_id())
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])

        # sampling of the text tokens, the audio codes are sampled by mtmd
        gen = _moshi_config(self).get("lm_gen_config", {})
        self.gguf_writer.add_sampling_temp(gen.get("temp_text", 0.7))
        self.gguf_writer.add_sampling_top_k(gen.get("top_k_text", 25))
        self.gguf_writer.add_sampling_top_p(1.0)
        self.gguf_writer.add_sampling_min_p(0.0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR

        if name == "text_emb.weight":
            config = _moshi_config(self)
            embd = data_torch.float().clone()
            embd[2] = embd[3] # EOS -> PAD, see get_moshi() of the reference
            embd += _condition(self, _COND_VALUE)
            assert embd.shape[0] == config["text_card"] + 1
            # <|text_start|> only opens the stream, with the initial codes of the generated audio
            for k in range(config["dep_q"]):
                embd[-1] += self.model_tensors[f"emb.{k}.weight"]()[config["card"]].float()
            yield (self.format_tensor_name(T.TOKEN_EMBD), embd)
            return
        if name == "text_linear.weight":
            yield (self.format_tensor_name(T.OUTPUT), data_torch)
            return
        if name == "out_norm.alpha":
            yield (self.format_tensor_name(T.OUTPUT_NORM), data_torch.reshape(-1))
            return

        if not name.startswith("transformer.layers."):
            return # mimi, depformer and the audio embeddings go to the mmproj
        assert bid is not None
        key = name.split(f"layers.{bid}.", 1)[1]

        if key == "self_attn.in_proj_weight":
            q, k, v = data_torch.chunk(3, dim=0)
            yield (self.format_tensor_name(T.ATTN_Q, bid), q)
            yield (self.format_tensor_name(T.ATTN_K, bid), k)
            yield (self.format_tensor_name(T.ATTN_V, bid), v)
        elif key == "self_attn.out_proj.weight":
            yield (self.format_tensor_name(T.ATTN_OUT, bid), data_torch)
        elif key == "norm1.alpha":
            yield (self.format_tensor_name(T.ATTN_NORM, bid), data_torch.reshape(-1))
        elif key == "norm2.alpha":
            yield (self.format_tensor_name(T.FFN_NORM, bid), data_torch.reshape(-1))
        elif key == "gating.linear_in.weight":
            # silu(first half) * second half
            gate, up = data_torch.chunk(2, dim=0)
            yield (self.format_tensor_name(T.FFN_GATE, bid), gate)
            yield (self.format_tensor_name(T.FFN_UP, bid), up)
        elif key == "gating.linear_out.weight":
            yield (self.format_tensor_name(T.FFN_DOWN, bid), data_torch)
        else:
            raise ValueError(f"unexpected tensor {name}")


@ModelBase.register("MoshiModel")
@ModelBase.example("kyutai/hibiki-1b-pytorch-bf16", "kyutai/hibiki-2b-pytorch-bf16")
class MoshiMmprojModel(MmprojModel):
    has_audio_encoder = True
    has_vision_encoder = False

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        assert remote_hf_model_id is None
        return _index_moshi_tensors(self)

    def _n_codebooks(self) -> int:
        return _moshi_config(self)["dep_q"]

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        assert self.hparams_audio is not None
        dep = self.global_config["depformer_config"]

        # input audio: mimi encoder + RVQ + embeddings of the input codes
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.MOSHI)
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_audio_block_count(self.hparams_audio["num_hidden_layers"])
        self.gguf_writer.add_audio_embedding_length(self.hparams_audio["hidden_size"])
        self.gguf_writer.add_audio_feed_forward_length(self.hparams_audio["intermediate_size"])
        self.gguf_writer.add_audio_head_count(self.hparams_audio["num_attention_heads"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)
        # mimi convolves the waveform directly, it is passed around as a 1-row "mel"
        self.gguf_writer.add_audio_num_mel_bins(1)

        # generation: depformer + mimi decoder, the mimi hparams are constant and held by mtmd
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.MOSHI_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(dep["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(dep["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(dep["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(dep["num_attention_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-8)
        self.gguf_writer.add_gen_audio_model_variant(_moshi_config(self)["model_type"])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims
        # conv1d/conv1d_dw kernels must be F16, ggml_conv_1d(_dw) has no BF16 path
        if ".seanet." in new_name or new_name in ("a.downsample.conv.weight", "a.gen.wav.upsample.weight"):
            return gguf.GGMLQuantizationType.F16
        # nearest-code search and RVQ decode stay exact
        if ".codebook" in new_name or new_name.startswith("a.codec.vq."):
            return gguf.GGMLQuantizationType.F32
        return False

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        T = gguf.MODEL_TENSOR
        if name.startswith(("a.", "mm.a.")):
            yield (name, data_torch) # already named by generate_extra_tensors()
            return
        if not name.startswith(_MIMI_PREFIX):
            return # the LM tensors are stacked in generate_extra_tensors()
        if name == "mimi.downsample.conv.conv.conv.weight":
            yield (self.format_tensor_name(T.A_ENC_DOWNSAMPLE_CONV), data_torch)
        elif name == "mimi.upsample.convtr.convtr.convtr.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_UPSAMPLE), data_torch)
        elif "_transformer.transformer.layers." in name:
            # pocket-tts names it in_proj.weight
            yield from mimi_tfm_tensors(self, name.replace(".in_proj_weight", ".in_proj.weight"), data_torch)
        elif name.startswith(("mimi.encoder.model.", "mimi.decoder.model.")):
            yield from mimi_seanet_tensors(self, name, data_torch, 4)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        config = _moshi_config(self)
        n_q = self._n_codebooks()

        def get(name: str) -> Tensor:
            return self.model_tensors[name]()

        def stack(names: Iterable[str]) -> Tensor:
            return torch.stack([get(n) for n in names], dim=0)

        # mimi RVQ: one semantic codebook, then n_q - 1 acoustic ones
        def codebooks(prefix: str, n: int) -> Tensor:
            books = []
            for i in range(n):
                embedding_sum = get(f"{prefix}.vq.layers.{i}._codebook.embedding_sum").float()
                cluster_usage = get(f"{prefix}.vq.layers.{i}._codebook.cluster_usage").float()
                books.append(embedding_sum / cluster_usage.clamp_min(1e-5).unsqueeze(-1))
            return torch.stack(books, dim=0) if n > 1 else books[0]

        q = "mimi.quantizer."
        cb_first = codebooks(q + "rvq_first", 1)
        cb_rest  = codebooks(q + "rvq_rest", n_q - 1)
        yield (self.format_tensor_name(T.A_CODEC_VQ_FIRST_IN), get(q + "rvq_first.input_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_CODEC_VQ_FIRST_CB), cb_first)
        yield (self.format_tensor_name(T.A_CODEC_VQ_REST_IN), get(q + "rvq_rest.input_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_CODEC_VQ_REST_CB), cb_rest)
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_OUT), get(q + "rvq_first.output_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_FIRST_CB), cb_first)
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_OUT), get(q + "rvq_rest.output_proj.weight").squeeze(-1))
        yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_REST_CB), cb_rest)

        # backbone embeddings of the audio codes: the generated stream first, then the input stream
        assert config["n_q"] == 2 * n_q
        yield (self.format_tensor_name(T.A_GEN_CODE_OUT_EMBD), stack(f"emb.{k}.weight" for k in range(n_q)))
        yield (self.format_tensor_name(T.A_MM_CODE_EMBD), stack(f"emb.{k}.weight" for k in range(n_q, 2 * n_q)))

        # depformer: per-step weights are stacked on the last ggml dim, steps past the last weight set reuse it
        dep = self.global_config["depformer_config"]
        n_dim = dep["hidden_size"]
        n_in = sum(1 for name in self.model_tensors if name.startswith("depformer_in."))
        yield (self.format_tensor_name(T.A_GEN_CODE_PROJ_IN), stack(f"depformer_in.{i}.weight" for i in range(n_in)))
        yield (self.format_tensor_name(T.A_GEN_CODE_TOK_EMBD), _low_rank_embd(self, "depformer_text_emb"))
        yield (self.format_tensor_name(T.A_GEN_CODE_EMBD),
               torch.stack([_low_rank_embd(self, f"depformer_emb.{i}") for i in range(n_q - 1)], dim=0))
        yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), stack(f"linears.{k}.weight" for k in range(n_q)))

        for il in range(dep["num_hidden_layers"]):
            p = f"depformer.layers.{il}."
            n_w = sum(1 for name in self.model_tensors if name.startswith(p + "gating.") and name.endswith(".linear_in.weight"))
            assert n_w == n_in, "depformer_in and the per-step weights use the same schedule"
            qkv = get(p + "self_attn.in_proj_weight").reshape(n_w, 3, n_dim, n_dim)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_Q, il), qkv[:, 0])
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_K, il), qkv[:, 1])
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_V, il), qkv[:, 2])
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_OUT, il), get(p + "self_attn.out_proj.weight").reshape(n_w, n_dim, n_dim))
            gate_up = stack(f"{p}gating.{i}.linear_in.weight" for i in range(n_w))
            gate, up = gate_up.chunk(2, dim=1)
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_GATE, il), gate)
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_UP, il), up)
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_DOWN, il), stack(f"{p}gating.{i}.linear_out.weight" for i in range(n_w)))
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_NORM, il), get(p + "norm1.alpha").reshape(-1))
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_NORM, il), get(p + "norm2.alpha").reshape(-1))
