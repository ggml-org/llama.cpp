from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen import Qwen3Model


def _is_vllm_decision2_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("decision_format") == "vllm-sr-decision"


@ModelBase.register_hparams_loader(_is_vllm_decision2_checkpoint)
def _load_vllm_decision2_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected vllm-sr Decision 2.0 checkpoint")
    with open(dir_model / "config.json", encoding="utf-8") as f:
        package = json.load(f)
    hparams = ModelBase.load_hparams((dir_model / package["backbone"]["config"]).parent, False, guess=False)
    hparams["architectures"] = ["VllmDecision2Model"]
    with open(dir_model / package["model_config"], encoding="utf-8") as f:
        hparams["decision"] = json.load(f)
    with open(dir_model / "score_bias.json", encoding="utf-8") as f:
        hparams["decision"]["score_bias"] = json.load(f)["offsets"]
    return hparams


@ModelBase.register("VllmDecision2Model")
@ModelBase.example("vllm-sr/Decision-2.0-Kai-0.6B")
class VllmDecision2Model(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.VLLM_DECISION2

    # torch.nn.LayerNorm default, used by the head
    _HEAD_NORM_EPS = 1e-5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head = self.hparams["decision"]
        if head["architecture"] != "qwen3-text-endpoints-global-query-shared-bilinear-mlp" or head["head_variant"] != "shared":
            raise ValueError(f"unsupported decision head: {head['architecture']}")

    def index_tensors(self, *args, **kwargs) -> dict[str, Callable[[], Tensor]]:
        # the backbone weights are in their own directory
        dir_model, self.dir_model = self.dir_model, self.dir_model / "backbone"
        try:
            return super().index_tensors(*args, **kwargs)
        finally:
            self.dir_model = dir_model

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        yield from super().get_tensors()
        from safetensors.torch import load_file
        for name, data in load_file(self.dir_model / "decision_head.safetensors").items():
            yield "decision_head." + name, data

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # the head is computed in F32
        if new_name.startswith("decision."):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    @classmethod
    def _systemone_template(cls) -> str:
        # prompt follows decision2/_vendor/dev2model/decision_model.py (segments) of the model repo
        # the pieces of the prompt are tokenized one by one, the server gives the text that separates them (sep)
        # and the text that starts the span of a question or of an option (mark_question, mark_option)
        # the head reads the last token of each option and the last token of the question
        # the start of an option is a piece of its own, so that the spans of two options are not joined
        # the keys of JSON objects are given in sorted order
        def text(value: str) -> str:
            return "{{ " + json.dumps(value) + " }}"

        def render(name: str) -> str:
            return "{{ " + name + " if " + name + " is string else " + name + " | tojson(separators=[',', ':']) }}"

        option = (
            "{% set d = o.description %}"
            "{% if q.type == 'noul' and d is none %}{% set d = 'Yes' if o.key == 'true' else 'No' %}{% endif %}"
            "{{ {'description': d, 'key': o.key} | tojson(separators=[',', ':']) }}"
        )
        return (
            "{% for q in questions %}"
            + text("Context:\n") + render("state")
            + text("\n\nTask type: ") + "{{ q.type }}" + text("\nQuestion:\n") + render("q.instructions") + text("\nOptions:")
            + "{% for o in q.options %}{{ sep }}" + text("\n<option>\n") + "{{ sep }}{{ mark_option }}" + option + text("\n</option>") + "{% endfor %}"
            + "{{ sep }}{{ mark_question }}" + text("\n\nSelect the single option best supported by the context and instructions.\nDecision:")
            + "{% endfor %}"
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.VLLM_DECISION2)
        self.gguf_writer.add_layer_norm_eps(self._HEAD_NORM_EPS)
        for offsets in self.hparams["decision"]["score_bias"].values():
            self.gguf_writer.add_decision_score_bias(offsets)
