from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Callable, Iterable

from .base import ModelBase, gguf
from .qwen import Qwen3_5TextModel
from .qwen3vl import Qwen3VLVisionModel

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("SenseNovaLoopForConditionalGeneration", "SenseNovaLoopModel", "Qwen3_5LoopForConditionalGeneration")
@ModelBase.example("sensenova/SenseNova-Loop-9B-Preview")
class SenseNovaLoopModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.SENSENOVA_LOOP
    supports_mtp_export = True

    def __init__(self, dir_model, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, self.is_mistral_format)
        n_mtp = hparams.get("text_config", hparams).get("mtp_num_hidden_layers", 0)
        if type(n_mtp) is not int or not 0 <= n_mtp <= 1:
            raise ValueError("SenseNova Loop supports at most one MTP layer")
        if n_mtp == 0:
            if self.mtp_only:
                raise ValueError("SenseNova Loop checkpoint has no MTP layer")
            self.no_mtp = True
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        hp = self.hparams
        self.loop = hp["loop_settings"]
        self.mhc = hp["mhc"]
        if self.loop.get("use_gate", False):
            raise ValueError("SenseNova Loop gate is not supported")
        start, end, count = (self.loop[k] for k in ("start", "end", "eval_count"))
        n_layer = hp["num_hidden_layers"]
        if any(type(v) is not int for v in (start, end, count, n_layer)) or not (0 <= start < end <= n_layer and count >= 1):
            raise ValueError("Invalid SenseNova Loop execution plan")
        if self.loop["counts"] != [count]:
            raise ValueError("SenseNova Loop requires a fixed loop count")
        n_mtp = self.block_count - n_layer
        if n_layer + (end - start) * (count - 1) + n_mtp > 512:
            raise ValueError("SenseNova Loop exceeds the maximum of 512 execution slots")
        layer_types = hp["layer_types"]
        if len(layer_types) != n_layer:
            raise ValueError("SenseNova Loop layer_types must describe every physical layer")
        self.plan = list(range(start)) + list(range(start, end)) * count + list(range(end, n_layer))
        self.first_visit = {p: self.plan.index(p) for p in range(n_layer)}
        self.block_count = len(self.plan) + n_mtp
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self.hparams = {**hp, "num_hidden_layers": len(self.plan), "layer_types": [layer_types[p] for p in self.plan]}

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id)
        if remote_hf_model_id is None and not self.no_mtp and self.opt_num_mtp_layers == 0 and (self.dir_model / "mtp.safetensors").is_file():
            tensors.update(self.index_local_tensors("mtp"))
        return tensors

    @classmethod
    def filter_tensors(cls, item):
        name, tensor = item
        if name.startswith("model.language_model."):
            name = "model." + name.removeprefix("model.language_model.")
        return super().filter_tensors((name, tensor))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        arch = self.gguf_writer.arch
        for key, value in ((gguf.Keys.Loop.START, self.loop["start"]),
                           (gguf.Keys.Loop.END, self.loop["end"]),
                           (gguf.Keys.Loop.COUNT, self.loop["eval_count"]),
                           (gguf.Keys.MHC.START, self.mhc["start"]),
                           (gguf.Keys.MHC.END, self.mhc["end"]),
                           (gguf.Keys.MHC.STREAMS, self.mhc["streams"]),
                           (gguf.Keys.MHC.ITERATIONS, self.mhc["iterations"])):
            self.gguf_writer.add_uint32(key.format(arch=arch), value)
        self.gguf_writer.add_bool(gguf.Keys.MHC.FIXED_IDENTITY.format(arch=arch), self.mhc["fixed_identity"])
        scale = self.mhc.get("pre_scale", 1.0)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid SenseNova mHC pre_scale")
        self.gguf_writer.add_float32(gguf.Keys.MHC.PRE_SCALE.format(arch=arch), scale)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.visual."):
            return
        name = name.removeprefix("model.language_model.") if name.startswith("model.language_model.") else name
        if not name.startswith(("model.", "lm_head.")):
            name = "model." + name
        if name == "model.loop_gate.logits":
            return
        match = re.match(r"model\.layers\.(\d+)\.(.+)", name)
        if match:
            physical = int(match[1])
            if physical < len(self.first_visit):
                bid = self.first_visit[physical]
            elif self.no_mtp:
                return
            else:
                bid = len(self.plan) + physical - len(self.first_visit)
            name = f"model.layers.{bid}.{match[2]}"
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".mhc_" in name or ".ssm_conv1d." in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


@ModelBase.register("SenseNovaLoopForConditionalGeneration", "SenseNovaLoopModel", "Qwen3_5LoopForConditionalGeneration")
class SenseNovaLoopVisionModel(Qwen3VLVisionModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        size = self.preprocessor_config["size"]
        self.gguf_writer.add_vision_min_pixels(int(size["shortest_edge"]))
        self.gguf_writer.add_vision_max_pixels(int(size["longest_edge"]))
