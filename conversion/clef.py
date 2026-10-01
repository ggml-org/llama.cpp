from __future__ import annotations

import json
import math

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen import Qwen3_5TextModel


def _is_clef_checkpoint(dir_model: Path) -> bool:
    return (dir_model / "joint_head_config.json").is_file() and (dir_model / "config.json").is_file()


@ModelBase.register_hparams_loader(_is_clef_checkpoint)
def _load_clef_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Clef checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["architectures"] = ["ClefModel"]
    with open(dir_model / "joint_head_config.json", encoding="utf-8") as f:
        hparams["decision"] = json.load(f)
    return hparams


# TODO: image input needs token and embedding entries in the same batch, see https://github.com/ggml-org/llama.cpp/pull/29622
@ModelBase.register("ClefModel")
class ClefModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.CLEF
    no_mtp = True  # the checkpoint has no MTP head

    # prompt follows joint_schema_model.py of the model repo
    _SYSTEM_PROMPT = (
        "Read the complete state and schema. Decide every field jointly. Each answer "
        "must be exactly one of that field's allowed options."
    )
    # the pieces of the prompt are tokenized one by one, this separates them
    _PIECE_SEP = "<<clef:sep>>"
    # start of a piece: state, question span, option span
    _PIECE_STATE, _PIECE_QUESTION, _PIECE_OPTION = "<<clef:state>>", "<<clef:question>>", "<<clef:option>>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head = self.hparams["decision"]
        self._n_routing = head["routing_layers"]
        # the head blocks are named dec.blk.N, routing blocks first
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, max(self.block_count, self._n_routing + head["layers"]))
        self._scales: dict[str, float] = {}

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    @classmethod
    def _systemone_template(cls) -> str:
        def text(value: str) -> str:
            return "{{ " + json.dumps(value) + " }}"

        sep = text(cls._PIECE_SEP)
        # state, instructions and option text are given as strings
        return (
            text(f"<|im_start|>system\n{cls._SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
            + sep + text(cls._PIECE_STATE) + "{{ state }}"
            + sep + text("\n\nSCHEMA FIELDS:\n")
            + "{% for q in questions %}"
            + sep + text("\nFIELD ") + "{{ loop.index }}" + text("\nID: ") + "{{ q.id }}"
            + text("\nTYPE: ") + "{{ q.type }}" + text("\nINSTRUCTION: ")
            + sep + text(cls._PIECE_QUESTION) + "{{ q.instructions }}"
            + sep + text("\nALLOWED OPTIONS:\n")
            + "{% for o in q.options %}"
            + sep + text("OPTION ") + "{{ loop.index }}" + text(": ")
            + sep + text(cls._PIECE_OPTION) + "{{ o.text }}"
            + sep + text("\n")
            + "{% endfor %}"
            + sep + text("END FIELD\n")
            + "{% endfor %}"
            + sep + text("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        head = self.hparams["decision"]
        self.gguf_writer.add_decision_type(gguf.DecisionType.CLEF)
        self.gguf_writer.add_decision_routing_block_count(head["routing_layers"])
        self.gguf_writer.add_decision_block_count(head["layers"])
        self.gguf_writer.add_decision_head_count(head["heads"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()
        from safetensors.torch import load_file
        for name, data in load_file(self.dir_model / "joint_head.safetensors").items():
            yield "joint_head." + name, data

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if not name.startswith("joint_head."):
            yield from super().modify_tensors(data_torch, name, bid)
            return

        parts = name.split(".")

        # learned scalars, stored as the values used at inference
        if len(parts) == 2 and data_torch.ndim == 0:
            value = float(data_torch)
            if parts[1] == "residual_gate":
                self._scales[parts[1]] = 1.0 / (1.0 + math.exp(-value))
            else:
                self._scales[parts[1]] = math.exp(min(value, math.log(100.0)))
            if len(self._scales) == 3:
                scales = [self._scales[k] for k in ("prior_logit_scale", "joint_logit_scale", "residual_gate")]
                yield self.format_tensor_name(gguf.MODEL_TENSOR.DECISION_SCALES, suffix=""), torch.tensor(scales, dtype=torch.float32)
            return

        # routing blocks come first
        if parts[1] == "layers":
            parts[2] = str(int(parts[2]) + self._n_routing)
        name = ".".join(parts)

        # nn.MultiheadAttention keeps q, k, v in one tensor
        for suffix in ("weight", "bias"):
            if name.endswith(".in_proj_" + suffix):
                prefix = name[:-len("in_proj_" + suffix)]
                for x, data in zip("qkv", data_torch.chunk(3, dim=0)):
                    yield self.map_tensor_name(prefix + x + "." + suffix), data
                return

        yield self.map_tensor_name(name), data_torch
