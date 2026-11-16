from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf, logger

# tiny-voca, https://huggingface.co/n4ze3m/tiny-voca
#
# text model: phoneme encoder (4 blocks), the input rows are built by the audio pipeline (token + stress + voice)
# mmproj:     input tables, voices, duration predictor, prosody and acoustic stacks, source-filter heads
#
# the text to phonemes step (dizione) is the "tts" chat template, see models/templates/n4ze3m-tiny-voca-tts.jinja
# its lexicon comes from the phonemizer/ directory of the model, the other tables from the dizione python package
#
# the prosody and acoustic stacks have the same blocks as the encoder, they are stored with the libllama names
# of the tiny-voca arch under a.gen.prosody. and a.gen.acoustic., mtmd runs them on llama contexts of its own

CHECKPOINT = "tiny-voca.pt"
DEFAULT_VOICE = "M2"

# phone classes of the reference (text.py): the phones get at least one frame, the vowels are always voiced
VOWELS = ("ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
          "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
          "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩")
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
              "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ")

# block tensors: reference name -> libllama name
BLOCK_MAP = {
    "norm1":          "attn_norm",
    "attention.qkv":  "attn_qkv",
    "attention.out":  "attn_output",
    "norm2":          "ffn_norm",
    "depthwise":      "ssm_conv1d",
    "expand":         "ffn_up",
    "project":        "ffn_down",
}

# mmproj stacks: reference module -> prefix
STACK_PREFIX = {
    "prosody_stack":  "a.gen.prosody.",
    "acoustic_stack": "a.gen.acoustic.",
}

# other mmproj tensors: reference module -> name
MMPROJ_MAP = {
    "tokens":             "a.gen.tok_embd",
    "stress":             "a.gen.stress_embd",
    "duration.conv1":     "a.gen.dur.conv1",
    "duration.conv2":     "a.gen.dur.conv2",
    "duration.norm1":     "a.gen.dur.norm1",
    "duration.norm2":     "a.gen.dur.norm2",
    "duration.out":       "a.gen.dur.out",
    "frame_position":     "a.gen.frame_pos",
    "prosody_head":       "a.gen.prosody_head",
    "prosody_projection": "a.gen.prosody_proj",
    "vocoder.harmonic":   "a.gen.voc.harmonic",
    "vocoder.noise":      "a.gen.voc.noise",
}


def _is_tiny_voca_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / CHECKPOINT).is_file() or not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("phonemizer", {}).get("name") == "dizione" and "vocab" in config and "model" in config


@ModelBase.register_hparams_loader(_is_tiny_voca_checkpoint)
def _load_tiny_voca_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected tiny-voca checkpoint")
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    c = config["model"]
    return {
        "architectures": ["TinyVocaModel"],
        "model_type": "tiny-voca",
        "num_hidden_layers": c["encoder_layers"],
        "hidden_size": c["channels"],
        "intermediate_size": c["feedforward"],
        "num_attention_heads": c["heads"],
        "num_key_value_heads": c["heads"],
        "max_position_embeddings": c["max_tokens"],
        "vocab_size": c["vocab_size"],
        "layer_norm_eps": 1e-5,  # torch LayerNorm default
        "rope_theta": 10000.0,
        "tiny_voca": config,
        # the mmproj tensors are named explicitly, they do not go through the block tensor map
        "audio_config": {"num_hidden_layers": 0},
    }


# CMUdict notation -> phones, for the entries of exceptions.txt written that way
# a simplified form of convert.py of dizione: the spelling-dependent rules are not applied
ARPA_CONSONANTS = {"B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
                   "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
                   "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
ARPA_VOWELS = {"AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɜː", "EY": "eɪ",
               "IH": "ɪ", "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː"}
ARPA_REDUCED = {"AH": "ə", "ER": "ɚ", "IH": "ɪ", "IY": "i"}
ARPA_RHOTIC = {"AA": "ɑːɹ", "AO": "ɔːɹ", "IH": "ɪɹ", "IY": "ɪɹ", "EH": "ɛɹ", "EY": "ɛɹ", "UH": "ʊɹ", "UW": "ʊɹ"}
STRESS_MARK = {"1": "ˈ", "2": "ˌ", "0": ""}


def _arpa_to_phones(arpa: str) -> str:
    syms = arpa.split()
    out: list[str] = []
    i = 0
    while i < len(syms):
        sym = syms[i]
        base, digit = (sym[:-1], sym[-1]) if sym[-1].isdigit() else (sym, "")
        nxt = syms[i + 1] if i + 1 < len(syms) else ""
        after = syms[i + 2] if i + 2 < len(syms) else ""
        if digit:
            mark = STRESS_MARK[digit]
            # r-coloured vowel: the r is not followed by a vowel
            if nxt == "R" and base in ARPA_RHOTIC and not after[-1:].isdigit():
                out.append(mark + ARPA_RHOTIC[base])
                i += 2
                continue
            if digit == "0" and base == "AH" and nxt == "L" and not after[-1:].isdigit():
                out.append("əl")
                i += 2
                continue
            out.append(mark + (ARPA_REDUCED.get(base, ARPA_VOWELS[base]) if digit == "0" else ARPA_VOWELS[base]))
        else:
            out.append(ARPA_CONSONANTS[base])
        i += 1
    return " ".join(out)


def _table_rows(path: Path) -> Iterable[list[str]]:
    # dizione tables: columns separated by tabs or two or more spaces, "#" comments
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].rstrip() if line.lstrip().startswith("#") else line.rstrip()
        if line.strip():
            yield [c.strip() for c in re.split(r"\t+| {2,}", line.strip())]


def _g2p_dicts(dir_model: Path) -> dict[str, dict[str, Any]] | None:
    lexicon_path = dir_model / "phonemizer" / "lexicon.tsv"
    spec = importlib.util.find_spec("dizione")
    if not lexicon_path.is_file() or spec is None or spec.origin is None:
        logger.warning("dizione or phonemizer/lexicon.tsv not found, the model will take phonemes as prompt "
                       "(pip install git+https://github.com/n4ze3m/dizione)")
        return None
    data = Path(spec.origin).parent / "data"
    lexicon: dict[str, str] = {}
    for line in lexicon_path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            word, phones = line.split("\t", 1)
            lexicon[word] = phones.strip()

    def phones(text: str) -> str:
        return text if re.search(r"[a-zɐ-˿]", text) else _arpa_to_phones(text)

    # word -> [stress class, form, final form or ""], see function_words.txt
    functions = {r[0]: [r[1], r[2], r[3] if len(r) > 3 else ""] for r in _table_rows(data / "function_words.txt")}
    # words -> [class, forms (one, or one per word), final form or ""], see phrases.txt
    phrases = {r[0]: [r[1], [f.strip() for f in r[2].split("|")], r[3] if len(r) > 3 else ""] for r in _table_rows(data / "phrases.txt")}
    letters = {r[0]: r[1] for r in _table_rows(data / "letters.txt")}
    # single words only, the names of several words are not supported yet
    exceptions = {r[0]: phones(r[1]) for r in _table_rows(data / "exceptions.txt") if " " not in r[0] and "|" not in r[1]}
    # abbreviation -> [when, words]
    abbreviations = {r[0]: [r[1], r[2]] for r in _table_rows(data / "abbreviations.txt")}
    return {
        "lexicon":       lexicon,
        "functions":     functions,
        "phrases":       phrases,
        "letters":       letters,
        "exceptions":    exceptions,
        "abbreviations": abbreviations,
    }


def _load_state(dir_model: Path) -> dict[str, Tensor]:
    ckpt = torch.load(dir_model / CHECKPOINT, map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != 4:
        raise ValueError(f"unsupported tiny-voca checkpoint format {ckpt.get('format_version')}")
    return ckpt["model"]


def _block_name(name: str) -> tuple[int, str] | None:
    # "blocks.<i>.<module>.<weight|bias>" -> (i, "blk.<i>.<libllama name>.<weight|bias>")
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return None
    bid = int(parts[1])
    module, suffix = ".".join(parts[2:-1]), parts[-1]
    return bid, f"blk.{bid}.{BLOCK_MAP[module]}.{suffix}"


def _block_tensors(new_name: str, t: Tensor) -> Iterable[tuple[str, Tensor]]:
    # depthwise conv [C, 1, K] -> [C, K], the layout of ggml_ssm_conv
    if ".ssm_conv1d.weight" in new_name:
        yield (new_name, t.squeeze(1))
        return
    # separate q, k, v: a model that libllama builds from metadata (the mmproj stacks) has no fused qkv
    if ".attn_qkv." in new_name:
        for part, chunk in zip(("attn_q", "attn_k", "attn_v"), t.chunk(3, dim=0)):
            yield (new_name.replace("attn_qkv", part), chunk.contiguous())
        return
    yield (new_name, t)


@ModelBase.register("TinyVocaModel")
@ModelBase.example("n4ze3m/tiny-voca")
class TinyVocaTextModel(TextModel):
    model_arch = gguf.MODEL_ARCH.TINY_VOCA

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        del remote_hf_model_id
        state = _load_state(self.dir_model)
        out: dict[str, Callable[[], Tensor]] = {}
        for name, t in state.items():
            if name == "tokens.weight" or name.startswith("encoder."):
                out[name] = (lambda t=t: t)
        return out

    def set_vocab(self):
        config = self.hparams["tiny_voca"]
        n_vocab = config["model"]["vocab_size"]
        tokens: list[str] = [""] * n_vocab
        toktypes = [gguf.TokenType.NORMAL] * n_vocab
        for text, i in config["vocab"].items():
            tokens[i] = text
            if text.startswith("<") and text.endswith(">"):
                toktypes[i] = gguf.TokenType.CONTROL
        assert all(tokens)

        # the prompt is split into phonemes by the audio pipeline, see mtmd-helper-gen.cpp
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores([0.0] * n_vocab)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_bos_token_id(config["vocab"]["<bos>"])
        self.gguf_writer.add_eos_token_id(config["vocab"]["<eos>"])
        self.gguf_writer.add_pad_token_id(config["vocab"]["<pad>"])
        self.gguf_writer.add_unk_token_id(config["vocab"]["<unk>"])
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)

        dicts = _g2p_dicts(self.dir_model)
        if dicts is not None:
            template_path = Path(__file__).parent.parent / "models" / "templates" / "n4ze3m-tiny-voca-tts.jinja"
            with open(template_path, encoding="utf-8") as f:
                self.gguf_writer.add_chat_template([{"name": "tts", "template": f.read()}])
            self.gguf_writer.add_tts_dicts(dicts)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_pooling_type(gguf.PoolingType.NONE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        if name == "tokens.weight":
            yield ("token_embd.weight", data_torch)
            return
        if name.startswith("encoder.norm."):
            yield ("output_norm." + name.rsplit(".", 1)[1], data_torch)
            return
        res = _block_name(name.removeprefix("encoder."))
        if res is None:
            raise ValueError(f"unexpected tensor {name}")
        yield from _block_tensors(res[1], data_torch)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims
        # ggml_ssm_conv takes F32 weights; token_embd is also read row by row on the host side
        if ".ssm_conv1d." in new_name or new_name == "token_embd.weight":
            return gguf.GGMLQuantizationType.F32
        return False


@ModelBase.register("TinyVocaModel")
@ModelBase.example("n4ze3m/tiny-voca")
class TinyVocaMmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("audio_config")

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        del remote_hf_model_id
        state = _load_state(self.dir_model)
        out: dict[str, Callable[[], Tensor]] = {}
        for name, t in state.items():
            if name.startswith("encoder.") or name in ("voices.weight", "frame_voices.weight"):
                continue
            out[name] = (lambda t=t: t)
        return out

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        return item

    def set_gguf_parameters(self):
        c = self.global_config["tiny_voca"]["model"]
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.TINY_VOCA)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(c["channels"])
        self.gguf_writer.add_gen_audio_feed_forward_length(c["feedforward"])
        self.gguf_writer.add_gen_audio_block_count(c["acoustic_layers"])
        self.gguf_writer.add_gen_audio_head_count(c["heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-5)
        self.gguf_writer.add_gen_audio_default_voice(DEFAULT_VOICE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        # from generate_extra_tensors(), named already
        if name.startswith(("voice.", "a.gen.")):
            yield (name, data_torch)
            return
        stack, _, rest = name.partition(".")
        if stack in STACK_PREFIX:
            prefix = STACK_PREFIX[stack]
            if rest.startswith("norm."):
                yield (prefix + "output_norm." + rest.rsplit(".", 1)[1], data_torch)
                return
            res = _block_name(rest)
            if res is None:
                raise ValueError(f"unexpected tensor {name}")
            yield from _block_tensors(prefix + res[1], data_torch)
            return
        module, suffix = name.rsplit(".", 1)
        if module not in MMPROJ_MAP:
            raise ValueError(f"unexpected tensor {name}")
        yield (MMPROJ_MAP[module] + "." + suffix, data_torch)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid
        # the blocks of the stacks are loaded by libllama, like the text model
        if new_name.startswith(tuple(STACK_PREFIX.values())) and ".ssm_conv1d." not in new_name and n_dims >= 2:
            return False
        # the rest is small and partly read on the host
        return gguf.GGMLQuantizationType.F32

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        config = self.global_config["tiny_voca"]
        state = _load_state(self.dir_model)
        # a voice row is the embedding added to the encoder input, then the one added to the frames
        for i, voice in enumerate(config["voices"]):
            yield (f"voice.{voice}", torch.cat([state["voices.weight"][i], state["frame_voices.weight"][i]]).float())
        # class of each token: 0 = pause (special or punctuation), 1 = consonant or <unk>, 2 = vowel
        vocab = config["vocab"]
        cls = torch.zeros(len(vocab), dtype=torch.float32)
        for p in CONSONANTS + ("<unk>",):
            cls[vocab[p]] = 1.0
        for p in VOWELS:
            cls[vocab[p]] = 2.0
        yield ("a.gen.token_class", cls)
