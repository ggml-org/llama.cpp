from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, gguf, jinja_str_or_json, logger
from .qwen import Qwen3_5TextModel


def _decision_lora_base(dir_model: Path) -> tuple[str, str | None]:
    # the base model of a LoRA adapter: (repo id, revision)
    with open(dir_model / "adapter_config.json", encoding="utf-8") as f:
        lora_config = json.load(f)
    revision = lora_config.get("revision")
    if revision is None and (dir_model / "training_config.json").is_file():
        with open(dir_model / "training_config.json", encoding="utf-8") as f:
            revision = json.load(f).get("base_revision")
    return lora_config["base_model_name_or_path"], revision


def _load_decision_lora_hparams(dir_model: Path, arch: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    repo_id, revision = _decision_lora_base(dir_model)
    with open(hf_hub_download(repo_id, "config.json", revision=revision), encoding="utf-8") as f:
        hparams = json.load(f)
    hparams["architectures"] = [arch]
    return hparams


class _DecisionLoraMixin:
    # decision model released as a LoRA adapter: the base model is downloaded and the adapter is merged into it
    no_mtp = True

    def __init__(self, dir_model: Path, *args, **kwargs):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        repo_id, revision = _decision_lora_base(dir_model)
        logger.info(f"gguf: downloading the base model {repo_id}")
        dir_base = Path(snapshot_download(repo_id, revision=revision, allow_patterns=["*.json", "*.jinja", "*.safetensors"]))
        super().__init__(dir_base, *args, **kwargs)  # ty: ignore[too-many-positional-arguments]
        self.dir_adapter = dir_model
        self.dir_model_card = dir_model

        with open(dir_model / "adapter_config.json", encoding="utf-8") as f:
            lora_config = json.load(f)
        # only a plain LoRA can be merged as scale * B @ A
        assert lora_config["peft_type"] == "LORA"
        assert lora_config.get("bias", "none") == "none"
        assert not lora_config.get("use_dora") and not lora_config.get("use_rslora") and not lora_config.get("lora_bias")
        assert not lora_config.get("rank_pattern") and not lora_config.get("alpha_pattern")
        assert not lora_config.get("modules_to_save")
        self.lora_scale = lora_config["lora_alpha"] / lora_config["r"]

        # "layers.0.mlp.up_proj.weight" -> {"A": tensor, "B": tensor}
        self.lora: dict[str, dict[str, Tensor]] = {}
        for name, tensor in load_file(dir_model / "adapter_model.safetensors").items():
            base_name, _, part = name[name.index("layers."):].partition(".lora_")
            assert part in ("A.weight", "B.weight"), f"unexpected LoRA tensor: {name}"
            self.lora.setdefault(base_name + ".weight", {})[part[0]] = tensor.float()
        self.lora_merged: set[str] = set()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        lora = self.lora.get(name[name.index("layers."):]) if "layers." in name else None
        if lora is not None:
            assert set(lora) == {"A", "B"} and data_torch.shape == (lora["B"].shape[0], lora["A"].shape[1])
            delta = self.lora_scale * (lora["B"] @ lora["A"])
            data_torch = data_torch.float() + LazyTorchTensor.from_eager(delta)
            self.lora_merged.add(name[name.index("layers."):])
        yield from super().modify_tensors(data_torch, name, bid)  # ty: ignore[unresolved-attribute]

    def prepare_tensors(self):
        super().prepare_tensors()  # ty: ignore[unresolved-attribute]
        if len(self.lora_merged) != len(self.lora):
            raise ValueError(f"only {len(self.lora_merged)} of {len(self.lora)} LoRA tensors were merged into the base model")


@ModelBase.register_hparams_loader(lambda dir_model: (dir_model / "lev_release.json").is_file())
def _load_lev_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Lev checkpoint")
    return _load_decision_lora_hparams(dir_model, "LevModel")


@ModelBase.register("LevModel")
@ModelBase.example("interfaze-ai/lev")
class LevModel(_DecisionLoraMixin, Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    # TODO: the head for large option sets (mode B, mode_b_head.pt) is not converted, only the label readout is supported
    # TODO: a description that is not text is given as JSON without the escaping of non-ASCII characters used in training

    # prompt follows packages/lev/src/lev/prompt.py of https://github.com/Abhinavexists/lev (chat style, state first)
    _SYSTEM_PROMPT = (
        "You are a System One decision model. You read the Evidence and answer each "
        "Criterion by choosing exactly one of the listed options. You never explain. "
        "You answer with the single option label only."
    )

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    def _systemone_template(self) -> str:
        description = jinja_str_or_json("o.description")
        options = (
            "{{ '# Options\\n' }}{% for o in options %}{{ o.label }}. "
            "{% if type == 'score' %}(level {{ o.key }} of {{ options | length - 1 }}) " + description
            + "{% else %}{{ o.key }}{% if o.description %}: " + description + "{% endif %}{% endif %}"
            "{{ '\\n' }}{% endfor %}"
            "{{ '\\nRespond with only the letter of ' }}"
            "{% if type == 'score' %}the level that best matches.{% else %}the best option.{% endif %}"
        )
        # noul is answered on a rating scale, its 2 options are only used for their description
        scale = "{{ '# Scale\\n0 = certainly no ... 8 = certainly yes\\n' }}"
        for key, name in (("true", "yes"), ("false", "no")):
            scale += (
                "{% for o in options %}{% if o.key == '" + key + "' and o.description %}"
                + name + ": " + description + "{{ '\\n' }}{% endif %}{% endfor %}"
            )
        scale += "{{ '\\nRespond with only a digit from 0 to 8.' }}"
        return (
            "<|im_start|>system\n" + self._SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n# Evidence\n" + jinja_str_or_json("state") + "\n\n# Criterion\n"
            "{% if instructions %}" + jinja_str_or_json("instructions") + "{% else %}{{ id }}{% endif %}"
            "{{ '\\n\\n' }}{% if type == 'noul' %}" + scale + "{% else %}" + options + "{% endif %}"
            "{{ '\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.LEV)
        with open(self.dir_adapter / "calibration.json", encoding="utf-8") as f:
            temperatures = json.load(f)["temperatures"]
        # "choice:A:small" -> "choice.small", only the label readout (mode A) is supported
        for name, value in temperatures.items():
            qtype, mode, *band = name.split(":")
            if mode == "A":
                self.gguf_writer.add_decision_temperature(".".join([qtype] + band), value)


def _is_kev_checkpoint(dir_model: Path) -> bool:
    # a LoRA adapter with the pointer head and the config of the kev training code
    if not all((dir_model / name).is_file() for name in ("adapter_config.json", "head.pt", "training_config.json")):
        return False
    with open(dir_model / "training_config.json", encoding="utf-8") as f:
        return "head_dim" in json.load(f).get("args", {})


@ModelBase.register_hparams_loader(_is_kev_checkpoint)
def _load_kev_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Kev checkpoint")
    return _load_decision_lora_hparams(dir_model, "KevModel")


@ModelBase.register("KevModel")
@ModelBase.example("jaredpalmer/kev-4b")
class KevModel(_DecisionLoraMixin, Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    # TODO: the server needs a question and its options in one batch, the state can be in previous batches
    # note: no plan to support date_facts (kev/api.py), its regex matching is fragile, a more generic impl is needed

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.load(self.dir_adapter / "head.pt", map_location="cpu", weights_only=True)
        assert set(self.head["head"]) == {"q.weight", "q.bias", "k.weight", "k.bias"}
        assert self.head["head"]["q.weight"].shape[0] == self.head["head_dim"]

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    def _systemone_template(self) -> str:
        # prompt follows kev/model.py and kev/api.py of https://github.com/jaredpalmer/kev
        # state, instructions and descriptions are given as text
        name = "{% if type != 'noul' %}{{ o.key }}{% elif o.key == 'true' %}yes{% else %}no{% endif %}"
        option = (
            "{% if type == 'score' %}{% if o.description %}{{ o.description }}{% endif %}"
            "{% else %}" + name + "{% if o.description %}: {{ o.description }}{% endif %}{% endif %}"
        )
        return (
            "<|fim_prefix|>{{ state }}<|fim_middle|>{{ instructions }}"
            "{% for o in options %}<|box_start|>" + option + "<|box_end|>{% endfor %}<|fim_suffix|>"
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.KEV)
        self.gguf_writer.add_embedding_length_out(2 * self.head["head_dim"])
        for name in ("choice", "score", "noul"):
            self.gguf_writer.add_decision_temperature(name, self.head["temperature"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()
        # pointer head: the output of a token is [q | k]
        head = self.head["head"]
        yield "classifier.out_proj.weight", torch.cat([head["q.weight"], head["k.weight"]], dim=0)
        yield "classifier.out_proj.bias",   torch.cat([head["q.bias"],   head["k.bias"]],   dim=0)
