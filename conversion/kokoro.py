from __future__ import annotations

import importlib
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf, logger

# Kokoro (StyleTTS2 with an iSTFTNet decoder), https://huggingface.co/hexgrad/Kokoro-82M
# text model: the PL-BERT phoneme encoder (ALBERT), one token per phoneme character
# mmproj: bert_encoder, prosody predictor, text encoder, decoder and the voices
# the text to phonemes step is the "tts" chat template, its dictionaries come from the misaki and nltk python packages

DEFAULT_VOICE = "af_heart"

MODULE_PREFIX = {
    "bert_encoder": "a.gen.bert_enc.",
    "predictor":    "a.gen.pred.",
    "text_encoder": "a.gen.text.",
    "decoder":      "a.gen.dec.",
}


def _checkpoint(dir_model: Path) -> Path | None:
    paths = sorted(dir_model.glob("kokoro-*.pth"))
    return paths[0] if paths else None


def _is_kokoro_checkpoint(dir_model: Path) -> bool:
    if _checkpoint(dir_model) is None or not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return "plbert" in config and "istftnet" in config and "vocab" in config


@ModelBase.register_hparams_loader(_is_kokoro_checkpoint)
def _load_kokoro_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Kokoro checkpoint")
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    plbert = config["plbert"]
    return {
        "architectures": ["KokoroModel"],
        "model_type": "kokoro",
        "num_hidden_layers": plbert["num_hidden_layers"],
        "hidden_size": plbert["hidden_size"],
        "intermediate_size": plbert["intermediate_size"],
        "num_attention_heads": plbert["num_attention_heads"],
        "max_position_embeddings": plbert["max_position_embeddings"],
        "vocab_size": config["n_token"],
        "layer_norm_eps": 1e-12,  # AlbertConfig default
        "kokoro": config,
        # the mmproj tensors are named explicitly, they do not go through the block tensor map
        "audio_config": {"num_hidden_layers": 0},
    }


def _load_modules(dir_model: Path, modules: Iterable[str]) -> dict[str, dict[str, Tensor]]:
    path = _checkpoint(dir_model)
    assert path is not None
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    # every module was saved from a DataParallel wrapper
    return {m: {k.removeprefix("module."): v for k, v in state[m].items()} for m in modules}


def _g2p_fallback(lexicons: list[dict[str, Any]]) -> dict[str, Any]:
    # model for the words missing from the lexicon: n-grams over units "letter:sounds", learned from the lexicon
    # key "<0 to 4 units before> <letter>" -> {sounds: probability}; "<s>" pads the start of a word, the letter "$" is its end
    # the "tts" template decodes it with a beam search
    order, min_count, stress = 5, 3, "\u02c8\u02cc"
    vowels = set("AIOQWYaiu\u00e6\u0251\u0252\u0254\u0259\u025b\u025c\u026a\u028a\u028c\u1d7b\u1d4a")

    def sounds(ps: str) -> list[str]:
        # a stress mark goes with the sound after it, a length mark with the one before
        out: list[str] = []
        pre = ""
        for c in ps:
            if c in stress:
                pre += c
            elif c == "\u02d0" and out:
                out[-1] += c
            else:
                out.append(pre + c)
                pre = ""
        return out

    def penalty(letter: str, chunk: str) -> float:
        # a consonant letter says consonants only; a vowel letter says a vowel, with at most a glide
        flat = [c for c in chunk if c not in stress and c != "\u02d0"]
        n_vowel = sum(c in vowels for c in flat)
        if letter in "bcdfgjkmnpqstvxz" and n_vowel > 0:
            return -8.0
        if letter in "aeiou" and flat and (n_vowel == 0 or any(c not in vowels and c not in "jw" for c in flat)):
            return -8.0
        return 0.0

    words: dict[str, list[str]] = {}
    for lexicon in lexicons:
        for word, v in lexicon.items():
            ps = v["DEFAULT"] if isinstance(v, dict) else v
            # same spelling normalization as the template: doubled consonants and "ck" say one sound
            w = re.sub(r"([b-df-hj-np-tv-z])\1", r"\1", word.lower()).replace("ck", "k")
            if ps and " " not in ps and len(w) >= 2 and re.fullmatch(r"[a-z']+", w) and w not in words:
                t = sounds(ps)
                if len(t) <= 2 * len(w):
                    words[w] = t

    def align(w: str, t: list[str], cost: dict[str, dict[str, float]]) -> list[str] | None:
        # best split of the sounds over the letters, each letter takes 0, 1 or 2 of them
        n_l, n_t = len(w), len(t)
        two = [t[j] + t[j + 1] for j in range(n_t - 1)]
        best = [[-1e18] * (n_t + 1) for _ in range(n_l + 1)]
        back = [[0] * (n_t + 1) for _ in range(n_l + 1)]
        best[0][0] = 0.0
        for i in range(n_l):
            letter = w[i]
            row = cost.setdefault(letter, {})
            cur, nxt, bk = best[i], best[i + 1], back[i + 1]
            # the sounds left must fit in the letters left
            for j in range(max(0, n_t - 2 * (n_l - i)), n_t + 1):
                base = cur[j]
                if base < -1e17:
                    continue
                for k, chunk in ((0, ""), (1, t[j] if j < n_t else None), (2, two[j] if j + 1 < n_t else None)):
                    if chunk is None:
                        break
                    c = row.get(chunk)
                    if c is None:
                        # not seen yet for this letter
                        c = row[chunk] = -12.0 + penalty(letter, chunk)
                    if base + c > nxt[j + k]:
                        nxt[j + k] = base + c
                        bk[j + k] = k
        if best[n_l][n_t] < -1e17:
            return None
        out: list[str] = []
        j = n_t
        for i in range(n_l, 0, -1):
            k = back[i][j]
            out.append("".join(t[j - k:j]))
            j -= k
        return out[::-1]

    # hard EM: align with the current letter -> sounds costs, count, repeat
    # the costs are learned on a sample of the words, then every word is aligned once
    def step(items: Iterable[tuple[str, list[str]]], cost: dict[str, dict[str, float]]) -> list[tuple[str, list[str]]]:
        return [(w, a) for w, a in ((w, align(w, t, cost)) for w, t in items) if a is not None]

    sample = list(words.items())[::max(1, len(words) // 20000)]
    cost: dict[str, dict[str, float]] = {}
    for _ in range(5):
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for w, a in step(sample, cost):
            for c, o in zip(w, a):
                counts[c][o] += 1
        # small cost for a silent letter, so that a sound stays on its own letter
        cost = {c: {o: math.log(n / sum(cnt.values())) - (0.5 if o == "" else 0.0) + penalty(c, o) for o, n in cnt.items()}
                for c, cnt in counts.items()}
    aligned = step(words.items(), cost)

    # counts of a unit after each of its contexts, from 4 units down to none
    ngram: dict[str, Counter[str]] = defaultdict(Counter)
    n_ctx: Counter[str] = Counter()
    for w, a in aligned:
        units = ["<s>"] * (order - 1) + [c + ":" + o for c, o in zip(w, a)] + ["$:"]
        for i in range(order - 1, len(units)):
            letter, _, o = units[i].partition(":")
            for n in range(order):
                ctx = " ".join(units[i - n:i])
                ngram[(ctx + " " if ctx else "") + letter][o] += 1
                n_ctx[ctx] += 1

    # the probability is over all the units after the context, not only the ones of this letter
    # this tells the search how likely the letter is after the sounds it chose, so it can drop a bad choice
    model: dict[str, Any] = {}
    for key, cnt in ngram.items():
        ctx, _, letter = key.rpartition(" ")
        # the two shortest contexts keep all the readings that the letter has min_count times or more
        kept = {o: float(f"{n / n_ctx[ctx]:.4g}") for o, n in cnt.most_common()
                if n >= min_count or (key.count(" ") < 2 and ngram[letter][o] >= min_count)}
        if kept:
            model[key] = kept
    return model


def _g2p_dicts() -> dict[str, dict[str, Any]] | None:
    # dictionaries of the "tts" template: the misaki lexicon and the nltk averaged perceptron tagger
    spec = importlib.util.find_spec("misaki")
    if spec is None or spec.origin is None:
        logger.warning("misaki is not installed, the model will take phonemes as prompt (pip install misaki nltk)")
        return None
    data = Path(spec.origin).parent / "data"
    try:
        nltk = importlib.import_module("nltk")
        base = Path(str(nltk.data.find("taggers/averaged_perceptron_tagger_eng"))) / "averaged_perceptron_tagger_eng"
    except (ImportError, LookupError):
        logger.warning("the nltk tagger is not available, the model will take phonemes as prompt "
                       "(pip install nltk && python -m nltk.downloader averaged_perceptron_tagger_eng)")
        return None

    def load(path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # tagger: classes sorted by label, feature -> {"c<class index>": weight}
    # (index keys: jinja2's map(attribute=...) splits on "." and some labels are punctuation)
    classes = sorted(load(Path(f"{base}.classes.json")))
    cid = {c: f"c{i}" for i, c in enumerate(classes)}
    weights = load(Path(f"{base}.weights.json"))
    lex = {
        "gold":      load(data / "us_gold.json"),
        "silver":    load(data / "us_silver.json"),
        "gold_gb":   load(data / "gb_gold.json"),
        "silver_gb": load(data / "gb_silver.json"),
    }
    logger.info("gguf: learning the pronunciation of unknown words from the lexicon, this takes a few minutes")
    # one process per variant
    with ProcessPoolExecutor(max_workers=2) as pool:
        lts, lts_gb = pool.map(_g2p_fallback, [[lex["gold"], lex["silver"]], [lex["gold_gb"], lex["silver_gb"]]])
    return {
        **lex,
        "lts":         lts,
        "lts_gb":      lts_gb,
        "pos_weights": {f: {cid[t]: w for t, w in tw.items()} for f, tw in weights.items()},
        "pos_tagdict": load(Path(f"{base}.tagdict.json")),
        "pos_classes": {"": classes},
    }


@ModelBase.register("KokoroModel")
@ModelBase.example("hexgrad/Kokoro-82M")
class KokoroTextModel(TextModel):
    model_arch = gguf.MODEL_ARCH.KOKORO

    _LAYER = "encoder.albert_layer_groups.0.albert_layers.0."
    _TENSOR_MAP = {
        "embeddings.LayerNorm":                gguf.MODEL_TENSOR.TOKEN_EMBD_NORM,
        "encoder.embedding_hidden_mapping_in": gguf.MODEL_TENSOR.TOKEN_EMBD_PROJ,
    }
    # the layers share their weights, stored as block 0
    _LAYER_TENSOR_MAP = {
        "attention.query":       gguf.MODEL_TENSOR.ATTN_Q,
        "attention.key":         gguf.MODEL_TENSOR.ATTN_K,
        "attention.value":       gguf.MODEL_TENSOR.ATTN_V,
        "attention.dense":       gguf.MODEL_TENSOR.ATTN_OUT,
        "attention.LayerNorm":   gguf.MODEL_TENSOR.ATTN_OUT_NORM,
        "ffn":                   gguf.MODEL_TENSOR.FFN_UP,
        "ffn_output":            gguf.MODEL_TENSOR.FFN_DOWN,
        "full_layer_layer_norm": gguf.MODEL_TENSOR.LAYER_OUT_NORM,
    }

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        del remote_hf_model_id
        bert = _load_modules(self.dir_model, ["bert"])["bert"]
        return {name: (lambda t=t: t) for name, t in bert.items()}

    def set_vocab(self):
        config = self.hparams["kokoro"]
        n_vocab = config["n_token"]
        tokens: list[str] = [f"[UNUSED_{i}]" for i in range(n_vocab)]
        toktypes = [gguf.TokenType.UNUSED] * n_vocab
        for text, i in config["vocab"].items():
            tokens[i] = text
            toktypes[i] = gguf.TokenType.NORMAL
        # id 0 pads both ends of the phoneme sequence
        tokens[0] = "<pad>"
        toktypes[0] = gguf.TokenType.CONTROL

        # the phonemes are mapped to tokens one character at a time by the audio pipeline
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores([0.0] * n_vocab)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_bos_token_id(0)
        self.gguf_writer.add_eos_token_id(0)
        self.gguf_writer.add_pad_token_id(0)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)

        dicts = _g2p_dicts()
        if dicts is not None:
            template_path = Path(__file__).parent.parent / "models" / "templates" / "hexgrad-Kokoro-82M-tts.jinja"
            with open(template_path, encoding="utf-8") as f:
                self.gguf_writer.add_chat_template([{"name": "tts", "template": f.read()}])
            self.gguf_writer.add_tts_dicts(dicts)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_pooling_type(gguf.PoolingType.NONE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        n_embd = self.hparams["hidden_size"]

        if name == "embeddings.word_embeddings.weight":
            # zero-pad the rows to n_embd, llama.cpp wants token and embedding inputs of the same width
            data_torch = torch.nn.functional.pad(data_torch, (0, n_embd - data_torch.shape[1]))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)
            return
        if name == "embeddings.position_embeddings.weight":
            # every token has type 0, fold its embedding into the positions
            type0 = self.model_tensors["embeddings.token_type_embeddings.weight"]()[0]
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.POS_EMBD), data_torch + type0)
            return
        if name.startswith(("embeddings.token_type_embeddings.", "pooler.")):
            return

        key, suffix = name.rsplit(".", 1)
        if key in self._TENSOR_MAP:
            yield (self.format_tensor_name(self._TENSOR_MAP[key], suffix="." + suffix), data_torch)
            return
        if key.startswith(self._LAYER):
            tensor = self._LAYER_TENSOR_MAP[key.removeprefix(self._LAYER)]
            yield (self.format_tensor_name(tensor, 0, suffix="." + suffix), data_torch)
            return
        raise ValueError(f"unexpected tensor {name}")


@ModelBase.register("KokoroModel")
@ModelBase.example("hexgrad/Kokoro-82M")
class KokoroMmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wnorm_g: dict[str, Tensor] = {}
        self._wnorm_v: dict[str, Tensor] = {}
        self._lstm_bias: dict[str, Tensor] = {}

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("audio_config")

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        del remote_hf_model_id
        tensors: dict[str, Callable[[], Tensor]] = {}
        for module, state in _load_modules(self.dir_model, MODULE_PREFIX.keys()).items():
            for name, t in state.items():
                tensors[MODULE_PREFIX[module] + name] = (lambda t=t: t)
        return tensors

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # the names are final already
        return item

    def set_gguf_parameters(self):
        config = self.global_config["kokoro"]
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.KOKORO)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(config["hidden_dim"])
        self.gguf_writer.add_gen_audio_feed_forward_length(config["hidden_dim"])
        self.gguf_writer.add_gen_audio_block_count(config["n_layer"])
        self.gguf_writer.add_gen_audio_head_count(1)
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-5)
        if (self.dir_model / "voices" / f"{DEFAULT_VOICE}.pt").is_file():
            self.gguf_writer.add_gen_audio_default_voice(DEFAULT_VOICE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        # torch weight_norm: weight = g * v / |v| over dims 1..n
        if name.endswith((".weight_g", ".weight_v")):
            base = name[:-len(".weight_g")]
            (self._wnorm_g if name.endswith("_g") else self._wnorm_v)[base] = data_torch
            if base in self._wnorm_g and base in self._wnorm_v:
                g = self._wnorm_g.pop(base).float()
                v = self._wnorm_v.pop(base).float()
                w = g * v / v.norm(dim=tuple(range(1, v.dim())), keepdim=True)
                if base.endswith(".pool"):
                    # depthwise transposed conv [C, 1, 3], stored as one row per tap
                    w = w.squeeze(1).T.contiguous()
                yield (base + ".weight", w)
            return
        # the two LSTM biases are always added
        if ".bias_ih_l0" in name or ".bias_hh_l0" in name:
            base = name.replace(".bias_ih_l0", ".bias_l0").replace(".bias_hh_l0", ".bias_l0")
            if base in self._lstm_bias:
                yield (base, self._lstm_bias.pop(base).float() + data_torch.float())
            else:
                self._lstm_bias[base] = data_torch
            return
        # snake alphas are per channel
        if ".alpha1." in name or ".alpha2." in name:
            data_torch = data_torch.reshape(-1)
        yield (name, data_torch)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid
        # tensors read on the host stay F32
        if "lstm" in new_name or ".shared." in new_name or new_name.startswith(("a.gen.bert_enc.", "voice.")) \
                or ".duration_proj." in new_name or ".m_source." in new_name or ".pool." in new_name:
            return gguf.GGMLQuantizationType.F32
        # CONV_TRANSPOSE_1D takes only F32 weights on some backends
        if ".ups." in new_name:
            return gguf.GGMLQuantizationType.F32
        # the conv ops have no BF16 kernels
        if n_dims >= 2:
            return gguf.GGMLQuantizationType.F16
        return gguf.GGMLQuantizationType.F32

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # a voice is a table of style vectors, the row is picked by the number of phonemes
        for path in sorted((self.dir_model / "voices").glob("*.pt")):
            pack = torch.load(path, map_location="cpu", weights_only=True)
            yield (f"voice.{path.stem}", pack.squeeze(1).float())
