#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import argparse
import json
import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Iterable,
    Iterator,
    Sequence,
    Type,
    cast,
)

import numpy as np
import torch

from gguf.constants import MODEL_ARCH, MODEL_ARCH_NAMES
from gguf.vocab import MistralTokenizerType, MistralVocab
from mistral_common.tokens.tokenizers.multimodal import DATASET_MEAN, DATASET_STD

if TYPE_CHECKING:
    from torch import Tensor

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf

logger = logging.getLogger("mistral-to-gguf")


###### MODEL DEFINITIONS ######


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
    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    hparams: dict[str, Any]
    tensor_names: set[str] | None
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    dir_model_card: Path
    remote_hf_model_id: str | None
    model_arch: MODEL_ARCH
    model_type: ModelType

    # subclasses should initialize this!
    block_count: int
    tensor_map: gguf.TensorNameMap

    def __init__(
        self,
        dir_model: Path,
        ftype: gguf.LlamaFileType,
        fname_out: Path,
        *,
        is_big_endian: bool = False,
        use_temp_file: bool = False,
        eager: bool = False,
        metadata_override: Path | None = None,
        model_name: str | None = None,
        split_max_tensors: int = 0,
        split_max_size: int = 0,
        dry_run: bool = False,
        small_first_shard: bool = False,
        hparams: dict[str, Any] | None = None,
        remote_hf_model_id: str | None = None,
        ctx: int = 0,
    ):
        if (
            type(self) is ModelBase
            or type(self) is TextModel
            or type(self) is MmprojModel
        ):
            raise TypeError(
                f"{type(self).__name__!r} should not be directly instantiated"
            )

        self.ctx = ctx
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = (
            gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        )
        self.use_temp_file = use_temp_file
        self.lazy = not eager or (remote_hf_model_id is not None)
        self.remote_hf_model_id = remote_hf_model_id
        self.vocab = MistralVocab(self.dir_model)
        if remote_hf_model_id is not None:

            def get_remote_tensors() -> Iterator[tuple[str, Tensor]]:
                logger.info(
                    f"Using remote model with HuggingFace id: {remote_hf_model_id}"
                )
                remote_tensors = gguf.utility.SafetensorRemote.get_list_tensors_model(
                    remote_hf_model_id
                )
                self.tensor_names = set(name for name in remote_tensors.keys())
                for (
                    name,
                    remote_tensor,
                ) in gguf.utility.SafetensorRemote.get_list_tensors_model(
                    remote_hf_model_id
                ).items():
                    yield (name, LazyTorchTensor.from_remote_tensor(remote_tensor))

            self.get_tensors = get_remote_tensors

        self.hparams = (
            ModelBase.load_hparams(self.dir_model) if hparams is None else hparams
        )
        self.tensor_names = None
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py

        # Apply heuristics to figure out typical tensor encoding based on first layer tensor encoding type
        if self.ftype == gguf.LlamaFileType.GUESSED:
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(
                    f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(
                    f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(
            path=None,
            arch=MODEL_ARCH_NAMES[self.model_arch],
            endianess=self.endianess,
            use_temp_file=self.use_temp_file,
            split_max_tensors=split_max_tensors,
            split_max_size=split_max_size,
            dry_run=dry_run,
            small_first_shard=small_first_shard,
        )

    @classmethod
    def add_prefix_to_filename(cls, path: Path, prefix: str) -> Path:
        stem, suffix = path.stem, path.suffix
        new_name = f"{prefix}{stem}{suffix}"
        return path.with_name(new_name)

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        tensor_names_from_parts: set[str] = set()

        self.tensor_names = tensor_names_from_parts
        weight_map: dict[str, str] = {}

        logger.info("gguf: loading 'consolidated.satensors'")
        ctx: ContextManager[Any]
        from safetensors import safe_open

        ctx = cast(
            ContextManager[Any],
            safe_open(
                self.dir_model / "consolidated.safetensors",
                framework="pt",
                device="cpu",
            ),
        )

        with ctx as model_part:
            tensor_names_from_parts.update(model_part.keys())

            for name in model_part.keys():
                if self.lazy:
                    data = model_part.get_slice(name)
                    data = LazyTorchTensor.from_safetensors_slice(data)
                else:
                    data = model_part.get_tensor(name)
                yield name, data

        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_parts.symmetric_difference(self.tensor_names)) > 0:
            missing = sorted(self.tensor_names.difference(tensor_names_from_parts))
            extra = sorted(tensor_names_from_parts.difference(self.tensor_names))
            missing_files = sorted(
                set(weight_map[n] for n in missing if n in weight_map)
            )
            if len(extra) == 0 and len(missing_files) > 0:
                raise ValueError(
                    f"Missing or incomplete model files: {missing_files}\n"
                    f"Missing tensors: {missing}"
                )
            else:
                raise ValueError(
                    "Mismatch between weight map and model parts for tensor names:\n"
                    f"Missing tensors: {missing}\n"
                    f"Extra tensors: {extra}"
                )

    def format_tensor_name(
        self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight"
    ) -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(
                f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}"
            )
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(
        self,
        name: str,
        key: gguf.MODEL_TENSOR,
        bid: int | None,
        suffix: str = ".weight",
    ) -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(
        self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")
    ) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        raise NotImplementedError(
            "set_gguf_parameters() must be implemented in subclasses"
        )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(
            ".weight,"
        )

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith(
                (".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")
            ):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data_torch in self.modify_tensors(data_torch, name, bid):
                # hard coded for pixtral
                if name == "vision_language_adapter.w_in.weight":
                    assert new_name == "mm.23.weight", new_name
                    new_name = "mm.1.weight"
                elif name == "vision_language_adapter.w_out.weight":
                    assert new_name == "mm.23.weight", new_name
                    new_name = "mm.2.weight"

                data = data_torch.numpy()

                # if data ends up empty, it means data_torch was a scalar tensor -> restore
                if len(data.shape) == 0:
                    data = data_torch.numpy()

                n_dims = len(data.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = False

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                if n_dims <= 1 or new_name.endswith("_norm.weight"):
                    data_qtype = gguf.GGMLQuantizationType.F32

                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                # Some tensor types are always in float32
                if data_qtype is False and (
                    any(
                        self.match_model_tensor_name(new_name, key, bid)
                        for key in (
                            gguf.MODEL_TENSOR.FFN_GATE_INP,
                            gguf.MODEL_TENSOR.POS_EMBD,
                            gguf.MODEL_TENSOR.TOKEN_TYPES,
                            gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                        )
                    )
                    or not new_name.endswith(".weight")
                ):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and any(
                    self.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.TOKEN_EMBD,
                        gguf.MODEL_TENSOR.OUTPUT,
                    )
                ):
                    if self.ftype in (
                        gguf.LlamaFileType.MOSTLY_TQ1_0,
                        gguf.LlamaFileType.MOSTLY_TQ2_0,
                    ):
                        # TODO: use Q4_K and Q6_K
                        data_qtype = gguf.GGMLQuantizationType.F16

                # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
                if isinstance(data_qtype, bool):
                    if self.ftype == gguf.LlamaFileType.ALL_F32:
                        data_qtype = gguf.GGMLQuantizationType.F32
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                        data_qtype = gguf.GGMLQuantizationType.F16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data_qtype = gguf.GGMLQuantizationType.BF16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                        data_qtype = gguf.GGMLQuantizationType.Q8_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ1_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ1_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ2_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ2_0
                    else:
                        raise ValueError(f"Unknown file type: {self.ftype.name}")

                try:
                    data = gguf.quants.quantize(data, data_qtype)
                except gguf.QuantError as e:
                    logger.warning("%s, %s", e, "falling back to F16")
                    data_qtype = gguf.GGMLQuantizationType.F16
                    data = gguf.quants.quantize(data, data_qtype)

                shape = (
                    gguf.quant_shape_from_byte_shape(data.shape, data_qtype)
                    if data.dtype == np.uint8
                    else data.shape
                )

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(
                    f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}"
                )

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self):
        total_params, shared_params, expert_params, expert_count = (
            self.gguf_writer.get_total_parameter_count()
        )

        self.metadata = gguf.Metadata.load(
            self.metadata_override, self.dir_model_card, self.model_name, total_params
        )

        # If we are using HF model id, set the metadata name to the model id
        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = self.dir_model.name

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(
                total_params, shared_params, expert_params, expert_count
            )

        self.set_type()

        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)

        logger.info("Set model parameters")
        self.set_gguf_parameters()

        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata()
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "params.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config


class TextModel(ModelBase):
    model_type = ModelType.TEXT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "text_config" in self.hparams:
            # move the text_config to the root level
            self.hparams = {**self.hparams, **self.hparams["text_config"]}

        self.block_count = self.find_hparam(
            ["n_layers", "num_hidden_layers", "n_layer", "num_layers"]
        )
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        logger.info(
            f"Converting tokenizer {self.vocab.tokenizer_type} of size {self.vocab.vocab_size}."
        )

        self.gguf_writer.add_tokenizer_model(self.vocab.gguf_tokenizer_model)

        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in self.vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == self.vocab.vocab_size, (
            f"token count ({len(tokens)}) != vocab size ({self.vocab.vocab_size})"
        )

        if self.vocab.tokenizer_type == MistralTokenizerType.tekken:
            self.gguf_writer.add_tokenizer_pre("tekken")
            self.gguf_writer.add_token_merges(
                self.vocab.extract_vocab_merges_from_model()
            )

        logger.info(
            f"Setting bos, eos, unk and pad token IDs to {self.vocab.bos_id}, {self.vocab.eos_id}, {self.vocab.unk_id}, {self.vocab.pad_id}."
        )

        self.gguf_writer.add_bos_token_id(self.vocab.bos_id)
        self.gguf_writer.add_eos_token_id(self.vocab.eos_id)
        self.gguf_writer.add_unk_token_id(self.vocab.unk_id)
        self.gguf_writer.add_pad_token_id(self.vocab.pad_id)

        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_vocab_size(self.vocab.vocab_size)

        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(False)

    def set_vocab_none(self):
        logger.info("Skipping tokenizer conversion.")
        logger.info("Setting tokenizer to 'none'.")
        self.gguf_writer.add_tokenizer_model("none")

        logger.info(
            f"Setting bos, eos, unk and pad token IDs to {self.vocab.bos_id}, {self.vocab.eos_id}, {self.vocab.unk_id}, {self.vocab.pad_id}."
        )
        self.gguf_writer.add_bos_token_id(self.vocab.bos_id)
        self.gguf_writer.add_eos_token_id(self.vocab.eos_id)
        self.gguf_writer.add_unk_token_id(self.vocab.unk_id)
        self.gguf_writer.add_pad_token_id(self.vocab.pad_id)

        logger.info(f"Setting vocab size to {self.vocab.vocab_size}.")
        self.gguf_writer.add_vocab_size(self.vocab.vocab_size)

        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)

    def prepare_metadata(self):
        super().prepare_metadata()

        total_params = self.gguf_writer.get_total_parameter_count()[0]
        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = self.ftype.name.partition("_")[2]

        # Filename Output
        if self.fname_out.is_dir():
            # Generate default filename based on model specification and available metadata
            fname_default: str = gguf.naming_convention(
                self.metadata.name,
                self.metadata.basename,
                self.metadata.finetune,
                self.metadata.version,
                self.metadata.size_label,
                output_type,
                model_type="LoRA" if total_params < 0 else None,
            )

            # Use the default filename
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
        else:
            # Output path is a custom defined templated filename
            # Note: `not is_dir()` is used because `.is_file()` will not detect
            #       file template strings as it doesn't actually exist as a file

            # Process templated file name with the output ftype, useful with the "auto" ftype
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(
                self.fname_out.name, output_type
            )

        logger.info("Set model tokenizer")
        self.set_vocab()

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)

        if self.ctx == 0:
            raise ValueError("ctx not passed as argument")
        self.gguf_writer.add_context_length(self.ctx)
        logger.info(f"gguf: training context length = {self.ctx}")

        if (n_embd := self.find_hparam(["dim"], optional=True)) is not None:
            self.gguf_writer.add_embedding_length(n_embd)
            logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["hidden_dim"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        if (n_head := self.find_hparam(["n_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count(n_head)
            logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("n_kv_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")

        if (f_norm_eps := self.find_hparam(["norm_eps"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")


class MmprojModel(ModelBase):
    model_type = ModelType.MMPROJ
    model_arch = gguf.MODEL_ARCH.MMPROJ
    preprocessor_config: dict[str, Any]
    global_config: dict[str, Any]

    n_block_keys = ["num_hidden_layers"]

    has_vision_encoder: bool = True

    hparams_vision: dict[str, Any]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        text_config = {
            k: v for k, v in self.hparams.items() if k not in ["vision_encoder"]
        }
        self.n_embd_text = text_config.get("hidden_dim", 0)
        assert self.n_embd_text > 0, "n_embd not found in hparams"

        # move vision config to the top level, while preserving the original hparams in global_config
        import copy

        self.global_config = copy.deepcopy(self.hparams)
        self.hparams_vision = self.get_vision_config()

        self.block_count = self.hparams_vision.get("num_hidden_layers", 0)
        assert self.block_count > 0, "num_hidden_layers not found in vision_config"
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def get_vision_config(self) -> dict[str, Any]:
        vision_config = self.global_config.get("vision_encoder")
        assert vision_config is not None, "vision_config not found in hparams"
        return vision_config

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MMPROJ)

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        if not self.has_vision_encoder:
            raise ValueError("MmprojModel must have a vision encoder")

    def find_vparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_vision is not None
        return self._find_param(self.hparams_vision, keys, optional)

    def _find_param(
        self, obj: dict[str, Any], keys: Iterable[str], optional: bool = False
    ) -> Any:
        key = next((k for k in keys if k in obj), None)
        if key is not None:
            return obj[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")


class MistralModel(TextModel):
    model_name = "mistral"
    model_arch = MODEL_ARCH.MISTRAL
    undo_permute = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        if "head_dim" in hparams:
            rope_dim = hparams["head_dim"]
        else:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        rope_scaling = self.hparams.get("rope_scaling") or {}
        if (
            rope_scaling.get("rope_type", rope_scaling.get("type")) == "linear"
            and "factor" in rope_scaling
        ):
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (
            weights.reshape(
                n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["n_heads"]
        n_kv_head = self.hparams.get("n_kv_heads")
        is_vision_tensor = any(
            name.startswith(prefix)
            for prefix in [
                "vision_encoder.",
                "vision_language_adapter.",
                "patch_merger.",
                "pre_mm_projector_norm",
            ]
        )

        if is_vision_tensor:
            return []  # skip vision tensors

        if self.undo_permute:
            if name.endswith(("q_proj.weight", "q_proj.bias")):
                data_torch = self.permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight", "k_proj.bias")):
                data_torch = self.permute(data_torch, n_head, n_kv_head)

        return [(self.map_tensor_name(name), data_torch)]


class PixtralModel(MmprojModel):
    model_name = "mistral"
    img_break_tok_id = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # layer_norm_eps is not in config.json, it is hard-coded in modeling_pixtral.py
        self.hparams["layer_norm_eps"] = self.hparams.get("norm_eps", 1e-5)
        self.img_break_tok_id = self.hparams_vision.get("image_break_token_id", -1)
        assert self.img_break_tok_id >= 0, (
            "image_break_token_id not found in vision_config"
        )
        logger.info(f"Image break token id: {self.img_break_tok_id}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PIXTRAL)

        self.gguf_writer.add_clip_has_vision_encoder(True)
        self.gguf_writer.add_vision_projection_dim(self.n_embd_text)

        # vision config
        self.gguf_writer.add_vision_image_size(self.find_vparam(["image_size"]))
        self.gguf_writer.add_vision_patch_size(self.find_vparam(["patch_size"]))
        self.gguf_writer.add_vision_embedding_length(self.find_vparam(["hidden_size"]))
        self.gguf_writer.add_vision_feed_forward_length(
            self.find_vparam(["intermediate_size"])
        )
        self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys))
        self.gguf_writer.add_vision_head_count(
            self.find_vparam(["num_attention_heads"])
        )

        # preprocessor config
        self.gguf_writer.add_vision_image_mean(
            self.hparams_vision.get("image_mean", DATASET_MEAN)
        )
        self.gguf_writer.add_vision_image_std(
            self.hparams_vision.get("image_std", DATASET_STD)
        )

        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.find_hparam(["layer_norm_eps"])
        )
        self.gguf_writer.add_rope_freq_base(self.find_vparam(["rope_theta"]))

        self.gguf_writer.add_vision_use_silu(True)

        # spatial_merge_size
        if self.hparams_vision["mm_projector_id"] == "patch_merge":
            self.gguf_writer.add_vision_spatial_merge_size(
                self.find_vparam(["spatial_merge_size"])
            )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        n_head = self.hparams_vision["num_attention_heads"]
        n_kv_head = n_head

        if any(
            name.startswith(prefix)
            for prefix in [
                "vision_encoder.",
                "vision_language_adapter.",
                "patch_merger.",
                "pre_mm_projector_norm",
            ]
        ):
            # process vision tensors
            if name.endswith(("q_proj.weight", "q_proj.bias")):
                data_torch = MistralModel.permute(data_torch, n_head, n_head)
            if name.endswith(("k_proj.weight", "k_proj.bias")):
                data_torch = MistralModel.permute(data_torch, n_head, n_kv_head)
            return [(self.map_tensor_name(name), data_torch)]

        if self.img_break_tok_id > 0 and "tok_embeddings.weight" in name:
            logger.info(f"Extracting [IMG_BREAK] token embedding from {name}")
            # for pixtral model, we need to extract the [IMG_BREAK] token embedding
            img_break_embd = data_torch[self.img_break_tok_id]
            name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_TOK_EMBD_IMG_BREAK]
            return [(self.map_tensor_name(name), img_break_embd)]

        return []  # skip other tensors


# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy()),
        )

    @classmethod
    def meta_with_dtype_and_shape(
        cls, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(
            meta=cls.meta_with_dtype_and_shape(dtype, shape),
            args=(st_slice,),
            func=lambda s: s[:],
        )
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_remote_tensor(cls, remote_tensor: gguf.utility.RemoteTensor):
        dtype = cls._dtype_str_map[remote_tensor.dtype]
        shape = remote_tensor.shape
        meta = cls.meta_with_dtype_and_shape(dtype, shape)
        lazy = cls(
            meta=meta,
            args=(remote_tensor,),
            func=lambda r: torch.frombuffer(r.data(), dtype=dtype).reshape(shape),
        )
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file"
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
        default="bf16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian",
        action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="directory containing model file",
        nargs="?",
    )
    parser.add_argument(
        "--ctx-train",
        type=int,
        help="Training context size",
        required=False,
    )
    parser.add_argument(
        "--use-temp-file",
        action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy",
        action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors",
        type=int,
        default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size",
        type=str,
        default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print out a split plan and exit, without writing any new files",
    )
    parser.add_argument(
        "--no-tensor-first-split",
        action="store_true",
        help="do not add tensors to the first split (disabled by default)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Specify the path for an authorship metadata override file",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="(Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model repo name instead of a local directory. For example: 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.",
    )
    parser.add_argument(
        "--mmproj",
        action="store_true",
        help="(Experimental) Export multimodal projector (mmproj) for vision models. This will only work on some vision models. A prefix 'mmproj-' will be added to the output file name.",
    )

    args = parser.parse_args()
    return args


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(
            f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G"
        )

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dir_model = args.model

    if args.remote:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=str(dir_model),
            allow_patterns=[
                "LICENSE",
                "params.json",
                "tekken.json",
                "*.md",
                "tokenizer.model",
            ],
        )
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")

    if not dir_model.is_dir():
        logger.error(f"Error: {args.model} is not a directory")
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    elif args.remote:
        # if remote, use the model ID as the output file name
        fname_out = Path("./" + str(args.model).replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        hparams = ModelBase.load_hparams(dir_model)
        model_class: Type[ModelBase]
        if args.mmproj and hparams.get("vision_encoder") is not None:
            model_class = PixtralModel
        elif args.mmproj:
            raise ValueError(
                "Multimodal projector export is only supported for vision models"
            )
        else:
            model_class = MistralModel
        logger.info(f"Model architecture: {model_class.__name__}")

        model_instance = model_class(
            dir_model,
            output_type,
            fname_out,
            is_big_endian=args.bigendian,
            use_temp_file=args.use_temp_file,
            eager=args.no_lazy,
            metadata_override=args.metadata,
            model_name=args.model_name,
            split_max_tensors=args.split_max_tensors,
            split_max_size=split_str_to_n_bytes(args.split_max_size),
            dry_run=args.dry_run,
            small_first_shard=args.no_tensor_first_split,
            remote_hf_model_id=str(args.model) if args.remote else None,
            ctx=args.ctx_train,
        )

        logger.info("Exporting model...")
        model_instance.write()
        out_path = (
            f"{model_instance.fname_out.parent}{os.sep}"
            if is_split
            else model_instance.fname_out
        )
        logger.info(f"Model successfully exported to {out_path}")


if __name__ == "__main__":
    main()
