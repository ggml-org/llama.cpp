from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Iterable

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

    def prepare_tensors(self):
        if not self.no_mtp and not any(name.startswith(f"model.layers.{len(self.first_visit)}.") for name in self.model_tensors):
            raise ValueError("MTP weights are missing from the checkpoint index; include them in model.safetensors.index.json or use --no-mtp")
        super().prepare_tensors()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_loop_start(self.loop["start"])
        self.gguf_writer.add_loop_end(self.loop["end"])
        self.gguf_writer.add_loop_count(self.loop["eval_count"])
        self.gguf_writer.add_hyper_connection_start(self.mhc["start"])
        self.gguf_writer.add_hyper_connection_end(self.mhc["end"])
        self.gguf_writer.add_hyper_connection_count(self.mhc["streams"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(self.mhc["iterations"])
        self.gguf_writer.add_hyper_connection_fixed_identity(self.mhc["fixed_identity"])
        scale = self.mhc.get("pre_scale", 1.0)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid SenseNova mHC pre_scale")
        self.gguf_writer.add_hyper_connection_pre_scale(scale)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
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
        if ".mhc_" in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


@ModelBase.register("SenseNovaLoopForConditionalGeneration", "SenseNovaLoopModel", "Qwen3_5LoopForConditionalGeneration")
class SenseNovaLoopVisionModel(Qwen3VLVisionModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        size = self.preprocessor_config["size"]
        self.gguf_writer.add_vision_min_pixels(int(size["shortest_edge"]))
        self.gguf_writer.add_vision_max_pixels(int(size["longest_edge"]))
