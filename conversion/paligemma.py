from __future__ import annotations

import json
import math
import pathlib
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger
from .gemma import GemmaModel


@ModelBase.register("PaliGemmaForConditionalGeneration")
class PaliGemmaTextModel(GemmaModel):
    model_arch = gguf.MODEL_ARCH.GEMMA

    def __init__(self, dir_model, *args, **kwargs):
        cfg = json.load(open(pathlib.Path(dir_model) / "config.json"))
        if cfg.get("text_config", {}).get("model_type") == "gemma2":
            raise NotImplementedError(
                "PaliGemma 2 (Gemma 2 backbone) is not supported by this converter. "
                "Only PaliGemma 1 (Gemma 1 backbone) is implemented."
            )
        super().__init__(dir_model, *args, **kwargs)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith("language_model."):
            # Strip prefix so downstream mapping sees standard Gemma tensor names
            return super().filter_tensors((name[len("language_model."):], gen))
        return None  # skip vision_tower, multi_modal_projector, etc.

    def set_vocab(self):
        # PaliGemma 1 ships tokenizer.model (SentencePiece)
        self._set_vocab_sentencepiece()
        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        # PaliGemma 1 omits several fields from text_config that GemmaModel
        # expects. Fill in the canonical GemmaConfig defaults before delegating.
        self.hparams.setdefault("max_position_embeddings", 8192)
        self.hparams.setdefault("rms_norm_eps", 1e-6)
        self.hparams.setdefault(
            "head_dim",
            self.hparams["hidden_size"] // self.hparams["num_attention_heads"],
        )
        super().set_gguf_parameters()


@ModelBase.register("PaliGemmaForConditionalGeneration")
class PaliGemmaVisionModel(MmprojModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SigLIP vision_config in PaliGemma HF configs omits image_size.
        # Infer it from preprocessor_config or num_image_tokens * patch_size.
        if self.hparams_vision and "image_size" not in self.hparams_vision:
            size = self.preprocessor_config.get("size", {})
            if "height" in size:
                self.hparams_vision["image_size"] = size["height"]
            else:
                patch_size = self.hparams_vision["patch_size"]
                n_tokens = self.hparams_vision.get("num_image_tokens", 256)
                self.hparams_vision["image_size"] = int(math.sqrt(n_tokens)) * patch_size

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith(("vision_tower.", "multi_modal_projector.")):
            return super().filter_tensors(item)
        return None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PALIGEMMA)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # The single linear projector has no block id; map manually to mm.0.
        if name == "multi_modal_projector.linear.weight":
            yield ("mm.0.weight", data_torch.to(torch.float16))
            return
        if name == "multi_modal_projector.linear.bias":
            yield ("mm.0.bias", data_torch.to(torch.float32))
            return
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # position_embd must be F32: ggml_add in clip.cpp requires matching dtypes
        if new_name == "v.position_embd.weight":
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
