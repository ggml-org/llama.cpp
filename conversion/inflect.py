from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from .base import ModelBase, gguf, logger
from .kokoro import MODULE_PREFIX, KokoroMmprojModel, KokoroTextModel

# Inflect Mini v3 (https://huggingface.co/owensong/Inflect-Mini-v3) is a smaller Kokoro, converted to the same layout:
# - the decoder blocks before the generator ("trunk") are narrow, a 1x1 conv brings their output to the generator width
# - a voice has 3 styles: predictor, trunk, generator

# voice -> Kokoro voice that gives its generator styles, in the order of the speaker rows (native_mini_inference.py)
VOICES = {
    "hope":    "af_heart",
    "sophia":  "af_heart",
    "cassidy": "af_sarah",
    "james":   "am_michael",
    "jasper":  "am_michael",
    "mark":    "am_michael",
}


def _checkpoint(dir_model: Path) -> Path | None:
    if not (dir_model / "config.json").is_file():
        return None
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("schema") != "inflect-package-descriptor" or "core_asset" not in config:
        return None
    path = dir_model / config["core_asset"]
    return path if path.is_file() else None


def _load_checkpoint(dir_model: Path) -> dict[str, Any]:
    path = _checkpoint(dir_model)
    assert path is not None
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    # Micro and Atom have another vocoder
    if payload.get("schema") != "inflect-native-mini-phonemes-v1":
        raise ValueError(f"unsupported Inflect model: {payload.get('schema')}")
    return payload


@ModelBase.register_hparams_loader(lambda dir_model: _checkpoint(dir_model) is not None)
def _load_inflect_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Inflect checkpoint")
    payload = _load_checkpoint(dir_model)
    config = payload["text_config"]
    n_embd, n_layer, n_head, hidden_dim, _, n_lstm = config["widths"]
    return {
        "architectures": ["InflectModel"],
        "model_type": "kokoro",
        "num_hidden_layers": n_layer,
        "hidden_size": n_embd,
        "intermediate_size": n_embd * 3,
        "num_attention_heads": n_head,
        "max_position_embeddings": payload["text_state"]["bert.embeddings.position_embeddings.weight"].shape[0],
        "vocab_size": config["n_token"],
        "layer_norm_eps": 1e-12,  # AlbertConfig default
        "kokoro": {"n_token": config["n_token"], "vocab": payload["vocab"], "hidden_dim": hidden_dim, "n_layer": n_lstm},
        # the mmproj tensors are named explicitly, they do not go through the block tensor map
        "audio_config": {"num_hidden_layers": 0},
    }


def _biased_voices(payload: dict[str, Any]) -> list[str]:
    # voices with a bias on the output of the generator AdaIN layers
    return [voice for voice in VOICES if payload["styles"]["affines"][voice]["required"]]


def _load_modules(dir_model: Path) -> dict[str, dict[str, Tensor]]:
    # same modules and tensor names as a Kokoro checkpoint
    payload = _load_checkpoint(dir_model)

    def wnorm(name: str) -> str:
        return name.replace(".parametrizations.weight.original0", ".weight_g").replace(".parametrizations.weight.original1", ".weight_v")

    modules: dict[str, dict[str, Tensor]] = {m: {} for m in ("bert", *MODULE_PREFIX)}
    for name, t in payload["text_state"].items():
        module, _, rest = name.partition(".")
        if module == "speaker_embedding":
            continue
        if module == "asr_proj":
            module, rest = "text_encoder", name
        modules[module][wnorm(rest)] = t
    for name, t in payload["bridge_state"].items():
        if name.startswith("trunk."):
            modules["decoder"][wnorm(name.removeprefix("trunk."))] = t
        elif name.startswith("projection."):
            modules["decoder"]["proj." + name.removeprefix("projection.")] = t

    # the bias of a voice becomes one more style input: its value in the voice row is the strength of the bias
    affines = payload["styles"]["affines"]
    biased = _biased_voices(payload)
    for name, t in payload["generator_state"].items():
        if name.endswith(".fc.weight"):
            layer = name.removesuffix(".fc.weight")
            t = torch.cat([t, *(affines[voice]["biases_by_module"][layer][:, None] for voice in biased)], dim=1)
        modules["decoder"]["generator." + wnorm(name)] = t
    return modules


@ModelBase.register("InflectModel")
@ModelBase.example("owensong/Inflect-Mini-v3")
class InflectTextModel(KokoroTextModel):
    model_arch = gguf.MODEL_ARCH.KOKORO

    # English only, and the voice names do not give the language
    tts_template_prefix = "{%- set language = language if language is defined and language else 'en-us' -%}\n"

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        del remote_hf_model_id
        return {name: (lambda t=t: t) for name, t in _load_modules(self.dir_model)["bert"].items()}


@ModelBase.register("InflectModel")
@ModelBase.example("owensong/Inflect-Mini-v3")
class InflectMmprojModel(KokoroMmprojModel):
    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        del remote_hf_model_id
        modules = _load_modules(self.dir_model)
        return {prefix + name: (lambda t=t: t) for module, prefix in MODULE_PREFIX.items() for name, t in modules[module].items()}

    def default_voice(self) -> str | None:
        return next(iter(VOICES))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # row: generator style (with the bias inputs), predictor style, trunk style; only the first one depends on the number of phonemes
        payload = _load_checkpoint(self.dir_model)
        styles = payload["styles"]
        biased = _biased_voices(payload)
        style_pred = payload["text_state"]["speaker_embedding.weight"]
        style_trunk = payload["bridge_state"]["speaker_embedding.weight"]
        for i, (voice, donor) in enumerate(VOICES.items()):
            style_gen = styles["banks"][donor] + styles["offsets"][i] * styles["strengths"][i]
            bias = torch.zeros(len(biased))
            if voice in biased:
                bias[biased.index(voice)] = styles["affines"][voice]["strength"]
            fixed = torch.cat([bias, style_pred[i], style_trunk[i]])
            yield (f"voice.{voice}", torch.cat([style_gen, fixed.expand(style_gen.shape[0], -1)], dim=1).float())
