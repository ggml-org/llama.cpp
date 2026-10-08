from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, gguf, logger
from .qwen import Qwen3Model, Qwen3_5TextModel


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _package_file(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid Decision 2.0 package path: {name}")
    return root / path


def _is_decision2_checkpoint(dir_model: Path) -> bool:
    config = dir_model / "config.json"
    return config.is_file() and _read_json(config).get("model_type") == "decision2"


def _decision2_base(dir_model: Path, config: dict[str, Any]) -> tuple[str, str]:
    manifest = _read_json(_package_file(dir_model, config["manifest"]))
    base = manifest.get("base") or {}
    backbone = config["backbone"]
    if base.get("repo_id") != backbone.get("repository") or base.get("revision") != backbone.get("revision"):
        raise ValueError("Decision 2.0 base differs from the pinned manifest")
    revision = base.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Decision 2.0 requires a pinned base commit")
    return base["repo_id"], revision


@ModelBase.register_hparams_loader(_is_decision2_checkpoint)
def _load_decision2_hparams(dir_model: Path) -> dict[str, Any]:
    config = _read_json(dir_model / "config.json")
    if "config" in config["backbone"]:
        hparams = _read_json(_package_file(dir_model, config["backbone"]["config"]))
    else:
        from huggingface_hub import hf_hub_download
        repo_id, revision = _decision2_base(dir_model, config)
        path = Path(hf_hub_download(repo_id, "config.json", revision=revision))
        manifest = _read_json(_package_file(dir_model, config["manifest"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["base"]["files_sha256"]["config.json"]:
            raise ValueError("Decision 2.0 base config differs from the pinned manifest")
        hparams = _read_json(path)
    text = hparams.get("text_config", hparams)
    if text["model_type"] == "qwen3":
        arch = "Decision2Qwen3Model"
    elif text["model_type"] in ("qwen3_5", "qwen3_5_text"):
        arch = "Decision2Qwen35Model"
    else:
        raise ValueError(f"Unsupported Decision 2.0 backbone: {text['model_type']}")
    hparams["architectures"] = [arch]
    text["architectures"] = [arch]
    text["max_position_embeddings"] = min(text["max_position_embeddings"], config["max_input_tokens"])
    return hparams


class _Decision2Mixin(ModelBase if TYPE_CHECKING else object):
    no_mtp = True

    def __init__(self, dir_model: Path, *args, **kwargs):
        from safetensors.torch import load_file

        config = _read_json(dir_model / "config.json")
        self.decision_config = _read_json(_package_file(dir_model, config["model_config"]))
        if self.decision_config.get("prompt_version") != "decision2-segmented-options-global-query-v1":
            raise ValueError("Unsupported Decision 2.0 prompt version")
        if self.decision_config.get("head_variant", "shared") != "shared":
            raise ValueError("Only the shared Decision 2.0 candidate head is supported")
        if config.get("calibration") is not None:
            raise ValueError("Decision 2.0 calibration is not supported")
        self.head_dim = int(self.decision_config["head_dim"])
        if self.head_dim <= 0 or self.decision_config["max_options"] != 255:
            raise ValueError("Invalid Decision 2.0 head configuration")
        self.head = load_file(_package_file(dir_model, config["decision_weights"]["decision_head"]))
        self.score_bias = self._load_score_bias(dir_model)
        self.lora: dict[str, dict[str, gguf.utility.LocalTensor]] = {}
        self.lora_merged: set[str] = set()
        self.lora_scale = 1.0
        self.lora_rank = 0

        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = _load_decision2_hparams(dir_model)
        if "adapter" in config:
            from huggingface_hub import snapshot_download
            repo_id, revision = _decision2_base(dir_model, config)
            logger.info(f"gguf: downloading Decision 2.0 base {repo_id} at {revision}")
            dir_backbone = Path(snapshot_download(repo_id, revision=revision, allow_patterns=["config.json", "model*.safetensors", "model.safetensors.index.json"]))
            self._load_adapter(dir_model, config["adapter"])
        else:
            dir_backbone = _package_file(dir_model, config["backbone"]["config"]).parent
        super().__init__(dir_backbone, *args, hparams=hparams, **kwargs)
        self.dir_model = dir_model
        self.dir_model_card = dir_model
        self._validate_head()

    def _validate_head(self):
        hidden = self.hparams["hidden_size"]
        shapes = {
            "candidate_norm.weight": (hidden,),
            "candidate_norm.bias": (hidden,),
            "query_norm.weight": (hidden,),
            "query_norm.bias": (hidden,),
            "key.weight": (self.head_dim, hidden),
            "query.weight": (self.head_dim, hidden),
            "candidate_mlp.weight": (self.head_dim, hidden),
            "candidate_mlp.bias": (self.head_dim,),
            "query_mlp.weight": (self.head_dim, hidden),
            "scalar.weight": (1, self.head_dim),
        }
        if set(self.head) != set(shapes):
            raise ValueError("Unexpected Decision 2.0 head tensors")
        for name, shape in shapes.items():
            if tuple(self.head[name].shape) != shape or not torch.isfinite(self.head[name]).all():
                raise ValueError(f"Invalid Decision 2.0 head tensor: {name}")

    @staticmethod
    def _load_score_bias(dir_model: Path) -> dict[int, list[float]]:
        path = dir_model / "score_bias.json"
        if not path.is_file():
            return {}
        report = _read_json(path)
        if report.get("format") != "dev2-score-bias-v1":
            raise ValueError("Unknown Decision 2.0 score bias format")
        offsets = {}
        for key, values in report["offsets"].items():
            if not key.isdigit() or str(int(key)) != key or not 2 <= int(key) <= 255:
                raise ValueError("Invalid Decision 2.0 score level count")
            if len(values) != int(key) or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError("Invalid Decision 2.0 score offsets")
            offsets[int(key)] = values
        return offsets

    def _load_adapter(self, dir_model: Path, adapter: dict[str, Any]):
        config = _read_json(_package_file(dir_model, adapter["config"]))
        unsupported = ("use_dora", "use_rslora", "lora_bias", "rank_pattern", "alpha_pattern", "modules_to_save", "fan_in_fan_out", "layer_replication")
        if config["peft_type"] != "LORA" or config.get("bias", "none") != "none" or any(config.get(key) for key in unsupported):
            raise ValueError("Only plain Decision 2.0 LoRA adapters can be merged")
        self.lora_rank = int(config["r"])
        if self.lora_rank <= 0:
            raise ValueError("Invalid Decision 2.0 LoRA rank")
        self.lora_scale = float(config["lora_alpha"]) / self.lora_rank
        if not math.isfinite(self.lora_scale):
            raise ValueError("Invalid Decision 2.0 LoRA scale")
        for path in adapter["weights"]:
            with gguf.utility.SafetensorsLocal(_package_file(dir_model, path)) as tensors:
                for name in tensors.keys():
                    if "layers." not in name:
                        raise ValueError(f"Unexpected Decision 2.0 LoRA tensor: {name}")
                    base_name, _, part = name[name.index("layers."):].partition(".lora_")
                    if part not in ("A.weight", "B.weight"):
                        raise ValueError(f"Unexpected Decision 2.0 LoRA tensor: {name}")
                    pair = self.lora.setdefault(base_name + ".weight", {})
                    if part[0] in pair:
                        raise ValueError(f"Duplicate Decision 2.0 LoRA tensor: {name}")
                    pair[part[0]] = tensors[name]
        if not self.lora or any(set(pair) != {"A", "B"} for pair in self.lora.values()):
            raise ValueError("Incomplete Decision 2.0 LoRA adapter")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0] == "lm_head.weight":
            return None
        return super().filter_tensors(item)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()
        for name, tensor in self.head.items():
            yield "decision2." + name, tensor.float()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("decision2."):
            yield self.map_tensor_name(name), data_torch.float()
            return
        key = name[name.index("layers."):] if "layers." in name else name
        if pair := self.lora.get(key):
            a = LazyTorchTensor.from_local_tensor(pair["A"]).float()
            b = LazyTorchTensor.from_local_tensor(pair["B"]).float()
            if a.shape[0] != self.lora_rank or b.shape[1] != self.lora_rank or data_torch.shape != (b.shape[0], a.shape[1]):
                raise ValueError(f"Decision 2.0 LoRA shape mismatch: {name}")
            if not isinstance(data_torch, LazyTorchTensor):
                data_torch = LazyTorchTensor.from_eager(data_torch)
            data_torch = data_torch.float() + self.lora_scale * (b @ a)
            self.lora_merged.add(key)
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.startswith("decision2."):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self.lora_merged != set(self.lora):
            raise ValueError(f"Unmerged Decision 2.0 LoRA tensors: {sorted(set(self.lora) - self.lora_merged)}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.DECISION2)
        self.gguf_writer.add_decision_head_dim(self.head_dim)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        self.gguf_writer.add_embedding_length_out(1)
        self.gguf_writer.add_pooling_type(gguf.PoolingType.NONE)
        for levels, offsets in self.score_bias.items():
            for index, value in enumerate(offsets):
                self.gguf_writer.add_decision_score_bias(levels, index, value)


@ModelBase.register("Decision2Qwen3Model")
@ModelBase.example("vllm-sr/Decision-2.0-Kai-0.6B")
class Decision2Qwen3Model(_Decision2Mixin, Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3


@ModelBase.register("Decision2Model", "Decision2Qwen35Model")
@ModelBase.example("vllm-sr/Decision-2.0-Eos-0.8B")
class Decision2Qwen35Model(_Decision2Mixin, Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
