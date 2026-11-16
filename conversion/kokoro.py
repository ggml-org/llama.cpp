from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf, logger

# Kokoro (StyleTTS2 with an iSTFTNet decoder), https://huggingface.co/hexgrad/Kokoro-82M
#
# text model: the PL-BERT phoneme encoder (ALBERT), one token per phoneme character
# mmproj:     bert_encoder, prosody predictor, text encoder, decoder and the voices
#
# the text to phonemes step (misaki) is the "tts" chat template, see models/templates/hexgrad-Kokoro-82M-tts.jinja
# its dictionaries come from the misaki and nltk python packages, when they are installed

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
    return {
        "gold":        load(data / "us_gold.json"),
        "silver":      load(data / "us_silver.json"),
        "gold_gb":     load(data / "gb_gold.json"),
        "silver_gb":   load(data / "gb_silver.json"),
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
        # tensors read on the host stay F32: LSTMs, linear layers around them, the harmonic source and the voices
        if "lstm" in new_name or ".shared." in new_name or new_name.startswith(("a.gen.bert_enc.", "voice.")) \
                or ".duration_proj." in new_name or ".m_source." in new_name or ".pool." in new_name:
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
