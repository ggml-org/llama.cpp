from __future__ import annotations

import importlib.util
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
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

# input letters and output phonemes of the unknown-word model, fixed in mtmd (see mtmd_graph_tiny_voca)
G2P_LETTERS = "abcdefghijklmnopqrstuvwxyz'"
G2P_SYMBOLS = [v + d for v in ("AA", "AE", "AH", "AO", "AW", "AY") for d in "012"] + ["B", "CH", "D", "DH"] + \
    [v + d for v in ("EH", "ER", "EY") for d in "012"] + ["F", "G", "HH"] + [v + d for v in ("IH", "IY") for d in "012"] + \
    ["JH", "K", "L", "M", "N", "NG"] + [v + d for v in ("OW", "OY") for d in "012"] + ["P", "R", "S", "SH", "T", "TH"] + \
    [v + d for v in ("UH", "UW") for d in "012"] + ["V", "W", "Y", "Z", "ZH"]

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

    # word -> [stress class, form, final form or ""], see function_words.txt
    functions = {r[0]: [r[1], r[2], r[3] if len(r) > 3 else ""] for r in _table_rows(data / "function_words.txt")}
    # words -> [class, forms (one, or one per word), final form or ""], see phrases.txt
    phrases = {r[0]: [r[1], [f.strip() for f in r[2].split("|")], r[3] if len(r) > 3 else ""] for r in _table_rows(data / "phrases.txt")}
    letters = {r[0]: r[1] for r in _table_rows(data / "letters.txt")}
    # a value is phones, or CMUdict notation (converted by the template, like the output of the unknown-word model)
    exceptions = {r[0].lower(): r[1] for r in _table_rows(data / "exceptions.txt") if " " not in r[0]}
    # a name of several words: "w1 w2 ..." -> one pronunciation per word; first word -> the longest name it starts
    names: dict[str, Any] = {}
    name_starts: dict[str, int] = {}
    for r in _table_rows(data / "exceptions.txt"):
        words = r[0].lower().split()
        if len(words) > 1:
            names[" ".join(words)] = [part.strip() for part in r[1].split("|")]
            name_starts[words[0]] = max(name_starts.get(words[0], 0), len(words))
    # phoneme (CMUdict, no stress) -> {letters: log probability}: aligns the letters of a word to its phonemes
    with open(dir_model / "phonemizer" / "align.json", encoding="utf-8") as f:
        align = json.load(f)
    # abbreviation -> [when, words]
    abbreviations = {r[0]: [r[1], r[2]] for r in _table_rows(data / "abbreviations.txt")}
    # chat spelling -> [words to say ("*LOL": letter by letter, "-": nothing), only as a reply by itself], see chat.txt
    chat = {r[0].lower(): [r[1] if len(r) > 1 else "-", len(r) > 2] for r in _table_rows(data / "chat.txt")}
    # accented latin letter -> plain letter(s): "café" is in the dictionary as "cafe" (dizione: _plain_letter)
    accents = {"ß": "ss", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ø": "o", "Ø": "O", "đ": "d", "ł": "l", "Ł": "L"}
    for cp in range(0x80, 0x2000):
        char = chr(cp)
        if char.isalpha() and char not in accents:
            plain = "".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c))
            if plain.isascii() and plain.isalpha():
                accents[char] = plain
    # compatibility forms of the text (NFKC), a character at a time: "…" is "...", "ﬁ" is "fi", "№" is "No", full
    # width letters. Sequences that NFKC would compose are left as they are
    nfkc: dict[str, str] = {}
    for cp in list(range(0x80, 0x3000)) + list(range(0xFB00, 0xFB07)) + list(range(0xFF01, 0xFF5F)):
        if unicodedata.normalize("NFKC", chr(cp)) != chr(cp):
            nfkc[chr(cp)] = unicodedata.normalize("NFKC", chr(cp))
    # units after a number: name -> [one, many]; "exact" ones are written in one letter case only ("m", not "M")
    units: dict[str, Any] = {}
    exact_units: dict[str, Any] = {}
    for r in _table_rows(data / "units.txt"):
        if len(r) > 3 and r[3] == "exact":
            exact_units[r[0]] = [r[1], r[2]]
        else:
            units[r[0].lower()] = [r[1], r[2]]
    # the unit names as a pattern of tts_norm_scan(), the longest first; a unit does not stop inside a word
    unit_names = sorted(set(units) | set(exact_units), key=lambda name: (-len(name), name))
    # (the pattern ignores the case of ASCII letters only, the greek letter is written in both)
    unit_pattern = "|".join("".join("[μΜ]" if c == "μ" else c if c.isalnum() else "%" + c for c in name) + "(?![A-Za-z])" for name in unit_names)
    # endings of file names and addresses: "word" (dot com) or "letters" (dot P D F)
    endings = {r[0]: (r[1] if len(r) > 1 else "") for r in _table_rows(data / "endings.txt")}
    out = {
        "lexicon":       lexicon,
        "functions":     functions,
        "phrases":       phrases,
        "letters":       letters,
        "exceptions":    exceptions,
        "abbreviations": abbreviations,
        "align":         align,
        "names":         names,
        "chat":          chat,
        "accents":       accents,
        "nfkc":          nfkc,
        "units":         units,
        "exact_units":   exact_units,
        "endings":       endings,
        "patterns":      {"units": unit_pattern},
        "name_starts":   name_starts,
    }
    out.update(_homograph_dicts(dir_model, data))
    return out


def _homograph_dicts(dir_model: Path, data: Path) -> dict[str, dict[str, Any]]:
    # words spelled alike and pronounced differently (dizione homographs.py): a linear classifier on the neighbouring
    # words, and hand-written rules that are tried first. Weights are stored x10000 (they have four decimals), so the
    # template adds integers.
    path = dir_model / "phonemizer" / "homographs.json"
    if not path.is_file():
        return {"homographs": {}, "homograph_shared": {}, "homograph_plurals": {}, "homograph_rules": {}}
    with open(path, encoding="utf-8") as f:
        source = json.load(f)

    def fixed(x: float) -> int:
        value = round(x * 10000)
        if abs(value - x * 10000) > 1e-6:
            raise ValueError(f"homographs.json: weight {x} has more than four decimals")
        return value

    words: dict[str, Any] = {}
    for word, entry in source["words"].items():
        model = entry["model"]
        noun_verb = model["kind"] == "noun-verb"
        if not noun_verb and model["kind"] != "senses":
            raise ValueError(f"homographs.json: unknown model kind {model['kind']!r}")
        words[word] = {
            # sense: [id, role, phones]
            "senses": [[s["id"], s["role"], s["phones"]] for s in entry["senses"]],
            # noun-verb: one score for the verb reading; else one score per sense
            "noun_verb": noun_verb,
            "bias":    [fixed(model["bias"])] if noun_verb else [fixed(b) for b in model["bias"]],
            "weights": [{k: fixed(v) for k, v in w.items()} for w in ([model["weights"]] if noun_verb else model["weights"])],
        }
    # the weights all noun-or-verb homographs share; "" is the bias
    shared = {k: fixed(v) for k, v in source["pooled"]["weights"].items()}
    shared[""] = fixed(source["pooled"]["bias"])
    # word -> [[sense id, [[condition, [words]], ...]], ...], see homograph_rules.txt
    rules: dict[str, Any] = {}
    for r in _table_rows(data / "homograph_rules.txt"):
        conditions = []
        for part in r[2].split(";"):
            name, _, listed = part.strip().partition(":")
            conditions.append([name, listed.split()])
        rules.setdefault(r[0], []).append([r[1], conditions])
    return {
        "homographs":        words,
        "homograph_shared":  shared,
        "homograph_plurals": {base: 1 for base in source.get("plural_bases", ())},
        "homograph_rules":   rules,
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
        # the unknown-word model of dizione (letters -> CMUdict phonemes), run on the host by mtmd_gen_audio_g2p()
        g2p_dir = self.dir_model / "phonemizer"
        if (g2p_dir / "g2p.npz").is_file():
            with open(g2p_dir / "g2p.json", encoding="utf-8") as f:
                settings = json.load(f)
            if settings["letters"] != G2P_LETTERS or settings["symbols"] != G2P_SYMBOLS or settings["convs"] != 0 or settings["slots"] != 2:
                raise ValueError("unsupported g2p model in phonemizer/g2p.json")
            with np.load(g2p_dir / "g2p.npz") as arrays:
                for name in arrays.files:
                    yield ("a.gen.g2p." + name.replace("__", "."), torch.from_numpy(arrays[name].astype(np.float32)))
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
