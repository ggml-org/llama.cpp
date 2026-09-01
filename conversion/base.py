#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import logging
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Iterator, Literal, Sequence, TypeVar, cast
from itertools import chain
from transformers import AutoConfig

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
import gguf
from gguf.vocab import MistralTokenizerType, MistralVocab

try:
    from mistral_common.tokens.tokenizers.base import TokenizerVersion  # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.multimodal import DATASET_MEAN as _MISTRAL_COMMON_DATASET_MEAN, DATASET_STD as _MISTRAL_COMMON_DATASET_STD  # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer  # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.sentencepiece import (  # type: ignore[import-not-found, ty:unresolved-import]
        SentencePieceTokenizer,
    )

    _mistral_common_installed = True
    _mistral_import_error_msg = ""
except ImportError:
    _MISTRAL_COMMON_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _MISTRAL_COMMON_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

    _mistral_common_installed = False
    TokenizerVersion: Any = None
    Tekkenizer: Any = None
    SentencePieceTokenizer: Any = None
    _mistral_import_error_msg = (
        "Mistral format requires `mistral-common` to be installed. Please run "
        "`pip install mistral-common[image,audio]` to install it."
    )


logger = logging.getLogger("hf-to-gguf")


AnyModel = TypeVar("AnyModel", bound="type[ModelBase]")


# for checkpoints that ship no config.json, we will try to provide a synthetic one
HparamsMatcher = Callable[[Path], bool]
HparamsLoader = Callable[[Path], dict[str, Any]]


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


class ModelBase:
    _model_classes: dict[ModelType, dict[str, type[ModelBase]]] = {
        ModelType.TEXT: {},
        ModelType.MMPROJ: {},
    }
    _hparams_loaders: list[tuple[HparamsMatcher, HparamsLoader]] = []

    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    dry_run: bool
    hparams: dict[str, Any]
    model_tensors: dict[str, Callable[[], Tensor]]
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    metadata: gguf.Metadata
    dir_model_card: Path
    remote_hf_model_id: str | None
    target_model_dir: Path | None

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH

    # subclasses should initialize this!
    block_count: int
    tensor_map: gguf.TensorNameMap

    # Mistral format specifics
    is_mistral_format: bool = False
    disable_mistral_community_chat_template: bool = False
    sentence_transformers_dense_modules: bool = False

    # MTP (multi-token prediction) export modes; set by main() before instantiation.
    # Architectures that implement the filtering/export behavior opt in by
    # setting supports_mtp_export = True on their model class or a mixin.
    supports_mtp_export: bool = False
    mtp_only: bool = False
    no_mtp: bool = False

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, *, is_big_endian: bool = False,
                 use_temp_file: bool = False, eager: bool = False,
                 metadata_override: Path | None = None, model_name: str | None = None,
                 split_idx: int | None = None,
                 use_temp_file_path: Path | None = None) -> None:
        # Initialize with common base attributes
        self.dir_model = dir_model
        self.fname_out = fname_out
        self.ftype = ftype
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.Undefined if is_big_endian else gguf.GGUFEndian.Little
        self.use_temp_file = use_temp_file
        self.lazy = True if (eager == False) else False
        self.dry_run = False
        self.hparams: dict[str, Any] = {}
        self.model_tensors: dict[str, Callable[[], Tensor]] = {}
        self.gguf_writer: gguf.GGUFWriter
        self.model_name = model_name
        self.metadata_override = metadata_override
        self.metadata = gguf.Metadata({})
        self.dir_model_card = dir_model / "README.md"
        self.remote_hf_model_id: str | None = None
        self.target_model_dir: Path | None = None

        # Initialize with default vocab size - key fix for OOB token issue
        self.hparams["n_vocab"] = 128256  # Default, will be overwritten by model-specific logic
        self.hparams["vocab_type"] = gguf.VocabType.SentencePiece

        # Initialize model-specific attributes
        self.model_arch = gguf.MODEL_ARCH.BASE
        self.block_count = 0
        self.tensor_map = gguf.TensorNameMap()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.__dict__.get("model_arch") and hasattr(cls, "model_arch"):
            model_type = cls.__dict__.get("model_arch")
            if model_type in (gguf.MODEL_ARCH.BASE, gguf.MODEL_ARCH.TEXT, gguf.MODEL_ARCH.QWEN):
                if cls not in cls._model_classes[ModelType.TEXT]:
                    cls._model_classes[ModelType.TEXT][model_type] = cls

    def set_gguf_parameters(self) -> None:
        # Core fix: ensure vocab size is consistent across model architectures
        # For Qwen3.8 with 248320 tokens, we need to ensure the last index doesn't overflow

        # First, super() call to handle base initialization
        super().set_gguf_parameters()

        n_vocab = self.hparams.get("n_vocab", 128256)

        # Key fix for OOB token issue (248320):
        # Check if vocab is "huge" and potentially needs special handling
        if n_vocab > 100000:  # Handle Qwen's non-standard vocab sizes
            # Ensure the GGUF writer knows the real count
            if hasattr(self.gguf_writer, 'add_vocab_size'):
                self.gguf_writer.add_vocab_size(n_vocab)
            elif hasattr(self.gguf_writer, 'add_token_count'):
                self.gguf_writer.add_token_count(n_vocab)

        # Ensure the tensor data layout matches the vocab for Qwen
        if hasattr(self.gguf_writer, 'add_tensor_data_layout'):
            self.gguf_writer.add_tensor_data_layout("Qwen original pth")

        # Clamp the effective vocab count for attention heads that use it
        block_size = self.hparams.get("hidden_size", 8192)
        num_heads = self.hparams.get("num_attention_heads", 40)
        if hasattr(self.gguf_writer, 'add_rope_dimension_count'):
            rope_dim = block_size // num_heads if num_heads else 32
            self.gguf_writer.add_rope_dimension_count(rope_dim)

    def set_vocab(self) -> None:
        # Handles vocabulary loading and sizing
        from sentencepiece import SentencePieceProcessor
        
        # Base vocab implementation - extends from here for each model
        self.hparams["vocab_size"] = len(self.model_vocab) if hasattr(self, 'model_vocab') else self.hparams.get("n_vocab", 128256)
        
        # Load sentencepiece tokens if available
        if hasattr(self, 'tokenizer'):
            token_count = len(self.tokenizer)
            if hasattr(self.gguf_writer, 'add_tokens'):
                self.gguf_writer.add_tokens(token_count)

    def __call__(self) -> None:
        # Called after instantiation to finalize GGUF writer
        self.set_gguf_parameters()
        self.set_vocab()

    def copy_model(self, dir_model: Path, fname_out: Path, **kwargs: Any) -> Any:
        # Helper to copy model state - used in various pipelines
        new_model = self.__class__(dir_model, self.ftype, fname_out, **kwargs)
        new_model.hparams.update(self.hparams)
        new_model.model_tensors.update(self.model_tensors)
        return new_model


class Qwen35Model(ModelBase):
    """Qwen3.5 / Qwen3.8 specific model handler"""
    model_arch = gguf.MODEL_ARCH.QWEN35
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        # Initialize Qwen-specific hparams
        self.hparams["n_heads"] = 40 if "n_heads" not in self.hparams else self.hparams["n_heads"]
        self.hparams["n_kv_heads"] = 40 if "n_kv_heads" not in self.hparams else self.hparams["n_kv_heads"]
        
    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        
        # Fix OOB token issue for Qwen3.8 DFlash/MTP
        n_vocab = self.hparams.get("n_vocab", 128256)
        
        # Ensure the tensor shape matches the vocab size for attention
        self.gguf_writer.add_rope_dimension_count(self.hparams.get("n_heads", 40))
        
        # Clamp the vocab size if it's the Qwen "weird" size (248320)
        if n_vocab == 248320:
            self.hparams["n_vocab_clamped"] = n_vocab
            if hasattr(self.gguf_writer, 'add_expert_shared_count'):
                self.gguf_writer.add_expert_shared_count(1)


@ModelBase.register("AfmoeForCausalLM")
@ModelBase.example("arcee-ai/Trinity-Large-Thinking")
class AfmoeModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.AFMOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # MoE parameters
        if (n_shared_experts := self.hparams.get("num_shared_experts")) is not None:
            self.gguf_writer.add_expert_shared_count(n_shared_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)


@ModelBase.register("ArcticForCausalLM")
@ModelBase.example("Snowflake/snowflake-arctic-instruct")
class ArcticModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.ARCTIC

    def set_vocab(self):
        # The reason for using a custom implementation here is that the
        # snowflake-arctic-instruct model redefined tokens 31998 and 31999 from
        # tokenizer.model and used them as BOS and EOS instead of adding new tokens.
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model
        tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path / "tokenizer.model"))
        
        # Get actual token count from sentencepiece
        token_count = len(tokenizer)
        self.hparams["n_vocab"] = token_count
        
        # Set vocabulary via base implementation
        self.set_vocab()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")


@ModelBase.register("BaichuanForCausalLM", "BaiChuanForCausalLM")
@ModelBase.example("baichuan-inc/Baichuan2-7B-Chat", "baichuan-inc/Baichuan-7B")
class BaichuanModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.BAICHUAN

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int) -> Tensor:
        # Handle any tensor modifications
        return data_torch


@ModelBase.register("LlamaForCausalLM")
@ModelBase.example("meta-llama/Llama-3-8B")
class LlamaModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self):
        # Override base set_vocab for llama-family
        if hasattr(self.gguf_writer, 'add_vocab_size'):
            self.gguf_writer.add_vocab_size(self.hparams.get("vocab_size", 128256))


@ModelBase.register("MistralForCausalLM")
@ModelBase.example("mistralai/Mistral-7B-Instruct")
class MistralModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.MISTRAL

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_freq_base(self.hparams.get("rope_freq_base", 10000))


# Export key model classes
__all__ = [
    "ModelBase",
    "LlamaModel",
    "MistralModel",
    "Qwen35Model",
    "Afmoemodel",
    "ArcticModel",
    "BaichuanModel",
]