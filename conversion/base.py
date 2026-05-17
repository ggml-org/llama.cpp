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
    from mistral_common.tokens.tokenizers.base import TokenizerVersion # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.multimodal import DATASET_MEAN as _MISTRAL_COMMON_DATASET_MEAN, DATASET_STD as _MISTRAL_COMMON_DATASET_STD # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer # type: ignore[import-not-found, ty:unresolved-import]
    from mistral_common.tokens.tokenizers.sentencepiece import ( # type: ignore[import-not-found, ty:unresolved-import]
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
    # Architectures opt in by overriding the handling (see _Qwen35MtpMixin).
    mtp_only: bool = False
    no_mtp: bool = False

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, *, is_big_endian: bool = False,
                 use_temp_file: bool = False, eager: bool = False,
                 metadata_override: Path | None = None, model_name: str | None = None,
                 split_max_tensors: int = 0, split_max_size: int = 0, dry_run: bool = False,
                 small_first_shard: bool = False, hparams: dict[str, Any] | None = None, remote_hf_model_id: str | None = None,
                 disable_mistral_community_chat_template: bool = False,
                 sentence_transformers_dense_modules: bool = False,
                 fuse_gate_up_exps: bool = False):
        if type(self) is ModelBase or \
                type(self) is TextModel or \
                type(self) is MmprojModel:
            raise TypeError(f"{type(self).__name__!r} should not be directly instantiated")

        if self.is_mistral_format and not _mistral_common_installed:
            raise ImportError(_mistral_import_error_msg)

        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.lazy = not eager or (remote_hf_model_id is not None)
        self.dry_run = dry_run
        self.remote_hf_model_id = remote_hf_model_id
        self.sentence_transformers_dense_modules = sentence_transformers_dense_modules
        self.fuse_gate_up_exps = fuse_gate_up_exps
        self._gate_exp_buffer: dict[int, Tensor] = {}
        self._up_exp_buffer: dict[int, Tensor] = {}
        self.hparams = ModelBase.load_hparams(self.dir_model, self.is_mistral_format) if hparams is None else hparams
        self.model_tensors = self.index_tensors(remote_hf_model_id=remote_hf_model_id)
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py
        self._is_nvfp4 = False
        self._is_mxfp4 = False

        # Apply heuristics to figure out typical tensor encoding based on first tensor's dtype
        # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
        if self.ftype == gguf.LlamaFileType.GUESSED:
            for _, tensor in self.get_tensors():
                if tensor.dim() < 2:
                    continue

                if tensor.dtype == torch.bfloat16:
                    self.ftype = gguf.LlamaFileType.MOSTLY_BF16
                    logger.info("heuristics detected bfloat16 tensor dtype, setting --outtype bf16")
                    break
                elif tensor.dtype == torch.float16:
                    self.ftype = gguf.LlamaFileType.MOSTLY_F16
                    logger.info("heuristics detected float16 tensor dtype, setting --outtype f16")
                    break
            else:
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
                logger.info("heuristics unable to detect tensor dtype, defaulting to --outtype f16")

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(path=None, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file,
                                           split_max_tensors=split_max_tensors, split_max_size=split_max_size, dry_run=dry_run, small_first_shard=small_first_shard)

        # Mistral specific
        self.disable_mistral_community_chat_template = disable_mistral_community_chat_template

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

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors: dict[str, Callable[[], Tensor]] = {}

        if remote_hf_model_id is not None:
            is_safetensors = True

            logger.info(f"Using remote model with HuggingFace id: {remote_hf_model_id}")
            remote_tensors = gguf.utility.SafetensorRemote.get_list_tensors_hf_model(remote_hf_model_id)
            for name, remote_tensor in remote_tensors.items():
                data_gen = lambda r=remote_tensor: LazyTorchTensor.from_remote_tensor(r)  # noqa: E731
                if titem := self.filter_tensors((name, data_gen)):
                    tname, tgen = titem
                    tensors[tname] = tgen

            return tensors

        prefix = "model" if not self.is_mistral_format else "consolidated"
        part_names: list[str] = ModelBase.get_model_part_names(self.dir_model, prefix, ".safetensors")
        is_safetensors: bool = len(part_names) > 0
        if not is_safetensors:
            part_names = ModelBase.get_model_part_names(self.dir_model, "pytorch_model", ".bin")

        tensor_names_from_index: set[str] = set()
        tensor_names_from_parts: set[str] = set()

        if not self.is_mistral_format:
            index_name = "model.safetensors" if is_safetensors else "pytorch_model.bin"
            index_name += ".index.json"
            index_file = self.dir_model / index_name

            if index_file.is_file():
                logger.info(f"gguf: loading model weight map from '{index_name}'")
                with open(index_file, "r", encoding="utf-8") as f:
                    index: dict[str, Any] = json.load(f)
                    weight_map = index.get("weight_map")
                    if weight_map is None or not isinstance(weight_map, dict):
                        raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                    tensor_names_from_index.update(weight_map.keys())
                    part_dict: dict[str, None] = dict.fromkeys(weight_map.values(), None) # ty: ignore[invalid-assignment]
                    part_names = sorted(part_dict.keys())
            else:
                weight_map = {}
        else:
            weight_map = {}

        for part_name in part_names:
            logger.info(f"gguf: indexing model part '{part_name}'")
            ctx: ContextManager[Any]
            if is_safetensors:
                ctx = cast(ContextManager[Any], gguf.utility.SafetensorsLocal(self.dir_model / part_name))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                assert model_part is not None

                for name in model_part.keys():
                    tensor_names_from_parts.add(name)
                    if is_safetensors:
                        data: gguf.utility.LocalTensor = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                        else:
                            dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                            data_gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731
                    else:
                        data_torch: Tensor = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data_torch: LazyTorchTensor.from_eager(data)  # noqa: E731
                        else:
                            data_gen = lambda data=data_torch: data  # noqa: E731
                    if titem := self.filter_tensors((name, data_gen)):
                        tname, tgen = titem
                        tensors[tname] = tgen

        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_index) > 0:
            if len(tensor_names_from_parts.symmetric_difference(tensor_names_from_index)) > 0:
                missing = sorted(tensor_names_from_index.difference(tensor_names_from_parts))
                extra = sorted(tensor_names_from_parts.difference(tensor_names_from_index))
                missing_files = sorted(set(weight_map[n] for n in missing if n in weight_map))
                if len(extra) == 0 and len(missing_files) > 0:
                    raise ValueError(f"Missing or incomplete model files: {missing_files}\n"
                                     f"Missing tensors: {missing}")
                else:
                    raise ValueError("Mismatch between weight map and model parts for tensor names:\n"
                                     f"Missing tensors: {missing}\n"
                                     f"Extra tensors: {extra}")

        return tensors

    @staticmethod
    def _scale_is_trivial(scale: Tensor) -> bool:
        return scale.numel() <= 1 and abs(float(scale.float().sum()) - 1.0) < 1e-6

    def _write_scale_tensor(self, scale_name: str, scale: Tensor):
        if not self._scale_is_trivial(scale):
            scale_f32 = scale.float().numpy().flatten()
            logger.info(f"  + {scale_name} (per-tensor scale, shape [{scale_f32.size}])")
            self.gguf_writer.add_tensor(scale_name, scale_f32)

    def _write_scales_tensor(self, scale_name: str, scales: list[float]):
        if not np.allclose(scales, 1.0, atol=1e-6):
            scale_vals = np.array(scales, dtype=np.float32)
            logger.info(f"  + {scale_name} (per-expert scale, shape [{len(scales)}])")
            self.gguf_writer.add_tensor(scale_name, scale_vals)

    def dequant_model(self):
        # If all quantized tensors were already handled (e.g. pure NVFP4), skip
        if self._is_nvfp4 and not any(k.endswith((".weight_scale", ".weight_scale_inv")) for k in self.model_tensors):
            return

        tensors_to_remove: list[str] = []
        new_tensors: dict[str, Callable[[], Tensor]] = {}

        if (quant_config := self.hparams.get("quantization_config")) and isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method")

            def dequant_bitnet(weight: Tensor, scale: Tensor) -> Tensor:
                weight = weight.view(torch.uint8)
                orig_shape = weight.shape

                shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape((4, *(1 for _ in range(len(orig_shape)))))
                data = weight.unsqueeze(0).expand((4, *orig_shape)) >> shift
                data = data & 3
                data = (data.float() - 1).reshape((orig_shape[0] * 4, *orig_shape[1:]))

                # The scale is inverted
                return data / scale.float()

            def dequant_simple(weight: Tensor, scale: Tensor, block_size: Sequence[int] | None = None) -> Tensor:
                scale = scale.float()

                if block_size is not None:
                    dim_offset = scale.ndim - len(block_size)
                    for i, size in enumerate(block_size):
                        scale = scale.repeat_interleave(size, dim_offset + i)
                    # unpad the scale (e.g. when the tensor size isn't a multiple of the block size)
                    scale = scale[tuple(slice(0, size) for size in weight.shape)]

                # align scale dims to weight for correct broadcasting (e.g. [128] -> [128, 1, 1])
                while scale.ndim < weight.ndim:
                    scale = scale.unsqueeze(-1)

                return weight.float() * scale

            # ref: https://github.com/ModelCloud/GPTQModel/blob/037c5c0f6c9e33c500d975b038d02e7ca437546d/gptqmodel/nn_modules/qlinear/__init__.py#L437-L476
            def dequant_gptq(g_idx: Tensor, qweight: Tensor, qzeros: Tensor, scales: Tensor) -> Tensor:
                bits = quant_config["bits"]
                assert bits in (2, 3, 4, 8)
                assert qweight.dtype == qzeros.dtype
                maxq = (2 ** bits) - 1
                weight = None
                zeros = None
                pack_dtype_bits = qweight.dtype.itemsize * 8

                if bits in [2, 4, 8]:
                    pack_factor = pack_dtype_bits // bits
                    wf = torch.tensor(list(range(0, pack_dtype_bits, bits)), dtype=torch.int32).unsqueeze(0)
                    if self.lazy:
                        wf = LazyTorchTensor.from_eager(wf)

                    zeros = torch.bitwise_right_shift(
                        qzeros.unsqueeze(2).expand(-1, -1, pack_factor),
                        wf.unsqueeze(0)
                    ).to(torch.int16 if bits == 8 else torch.int8)
                    zeros = torch.bitwise_and(zeros, maxq).reshape(scales.shape)

                    weight = torch.bitwise_and(
                        torch.bitwise_right_shift(
                            qweight.unsqueeze(1).expand(-1, pack_factor, -1),
                            wf.unsqueeze(-1)
                        ).to(torch.int16 if bits == 8 else torch.int8),
                        maxq
                    )
                elif bits == 3:
                    raise NotImplementedError("3-bit gptq dequantization is not yet implemented")

                assert weight is not None
                assert zeros is not None

                weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

                # gptq_v2 doesn't need to offset zeros
                if quant_config.get("checkpoint_format", "gptq") == "gptq":
                    zeros += 1

                return (scales[g_idx].float() * (weight - zeros[g_idx]).float()).T

            def dequant_packed(w: Tensor, scale: Tensor, shape_tensor: Tensor, zero_point: Tensor | None, num_bits: int, group_size: int):
                assert w.dtype == torch.int32
                shape = tuple(shape_tensor.tolist())
                assert len(shape) == 2
                mask = (1 << num_bits) - 1

                shifts = torch.arange(0, 32 - (num_bits - 1), num_bits, dtype=torch.int32)
                if self.lazy:
                    shifts = LazyTorchTensor.from_eager(shifts)

                if zero_point is None:
                    offset = 1 << (num_bits - 1)
                else:
                    assert len(zero_point.shape) == 2
                    offset = (zero_point.unsqueeze(1) >> shifts.reshape(1, -1, 1)) & mask
                    offset = offset.reshape(-1, zero_point.shape[1])
                    # trim padding, and prepare for broadcast
                    # NOTE: the zero-point is packed along dim 0
                    offset = offset[:shape[0], :].unsqueeze(-1)

                # extract values
                # NOTE: the weights are packed along dim 1
                unpacked = (w.unsqueeze(-1) >> shifts.reshape(1, 1, -1)) & mask
                unpacked = unpacked.reshape(shape[0], -1)

                # trim padding
                unpacked = unpacked[:, :shape[1]]

                # prepare for broadcast of the scale
                unpacked = unpacked.reshape(shape[0], (unpacked.shape[-1] + group_size - 1) // group_size, group_size)
                unpacked = unpacked - offset

                return (unpacked * scale.unsqueeze(-1).float()).reshape(shape)

            if quant_method == "bitnet":
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale"):
                        weight_name = name.removesuffix("_scale")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s: dequant_bitnet(w(), s())
                        tensors_to_remove.append(name)
            elif quant_method == "fp8":
                block_size = quant_config.get("weight_block_size")
                for name in self.model_tensors.keys():
                    if name.endswith("_scale_inv"):
                        weight_name = name.removesuffix("_scale_inv")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s, bs=block_size: dequant_simple(w(), s(), bs)
                        tensors_to_remove.append(name)
                    if name.endswith(".activation_scale"):  # unused
                        tensors_to_remove.append(name)
                    if name.endswith("_activation_scale"):  # Mistral-Small-4-119B-2602, unused
                        tensors_to_remove.append(name)
                    # mistral format
                    if name.endswith(".qscale_weight"):
                        weight_name = name.removesuffix("qscale_weight") + "weight"
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s, bs=block_size: dequant_simple(w(), s(), bs)
                        tensors_to_remove.append(name)
                    if name.endswith(".qscale_act"):
                        tensors_to_remove.append(name)
            elif quant_method == "gptq":
                for name in self.model_tensors.keys():
                    if name.endswith(".qweight"):
                        base_name = name.removesuffix(".qweight")
                        g_idx = self.model_tensors[base_name + ".g_idx"]
                        qweight = self.model_tensors[base_name + ".qweight"]
                        qzeros = self.model_tensors[base_name + ".qzeros"]
                        scales = self.model_tensors[base_name + ".scales"]
                        new_tensors[base_name + ".weight"] = (
                            lambda g=g_idx, z=qzeros, w=qweight, s=scales: dequant_gptq(
                                g(), w(), z(), s()
                            )
                        )
                        tensors_to_remove += [
                            base_name + n
                            for n in (
                                ".g_idx",
                                ".qzeros",
                                ".qweight",
                                ".scales",
                            )
                        ]
            elif quant_method == "compressed-tensors":
                quant_format = quant_config["format"]
                groups = quant_config["config_groups"]
                if len(groups) > 1:
                    raise NotImplementedError("Can't handle multiple config groups for compressed-tensors yet")
                weight_config = tuple(groups.values())[0]["weights"]

                if quant_format == "float-quantized" or quant_format == "int-quantized" or quant_format == "naive-quantized":
                    block_size = weight_config.get("block_structure", None)
                    strategy = weight_config.get("strategy")
                    assert strategy == "channel" or strategy == "block"
                    assert weight_config.get("group_size") is None  # didn't find a model using this yet
                    for name in self.model_tensors.keys():
                        if name.endswith(".weight_scale"):
                            weight_name = name.removesuffix("_scale")
                            w = self.model_tensors[weight_name]
                            s = self.model_tensors[name]
                            self.model_tensors[weight_name] = lambda w=w, s=s: dequant_simple(w(), s(), block_size)
                            tensors_to_remove.append(name)
                elif quant_format == "pack-quantized":
                    assert weight_config.get("strategy") == "group"
                    assert weight_config.get("type", "int") == "int"
                    num_bits = weight_config.get("num_bits")
                    group_size = weight_config.get("group_size")
                    assert isinstance(num_bits, int)
                    assert isinstance(group_size, int)
                    for name in self.model_tensors.keys():
                        if name.endswith(".weight_packed"):
                            base_name = name.removesuffix("_packed")
                            w = self.model_tensors[name]
                            scale = self.model_tensors[base_name + "_scale"]
                            shape = self.model_tensors[base_name + "_shape"]
                            zero_point = self.model_tensors.get(base_name + "_zero_point", lambda: None)
                            new_tensors[base_name] = (
                                lambda w=w, scale=scale, shape=shape, zero_point=zero_point: dequant_packed(
                                    w(), scale(), shape(), zero_point(), num_bits, group_size,
                                )
                            )
                            tensors_to_remove += [base_name + n for n in ("_packed", "_shape", "_scale")]
                            if (base_name + "_zero_point") in self.model_tensors:
                                tensors_to_remove.append(base_name + "_zero_point")
                else:
                    raise NotImplementedError(f"Quant format {quant_format!r} for method {quant_method!r} is not yet supported")
            elif quant_method == "modelopt":
                # Mixed-precision ModelOpt models: NVFP4 tensors are handled by
                # _generate_nvfp4_tensors; FP8 tensors have 1D weight_scale and
                # are dequantized here. k/v scale tensors are unused.
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale"):
                        weight_name = name.removesuffix("_scale")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s: dequant_simple(w(), s(), None)
                        tensors_to_remove.append(name)
                    if name.endswith((".input_scale", ".k_scale", ".v_scale")):
                        tensors_to_remove.append(name)
            elif quant_method is not None:
                raise NotImplementedError(f"Quant method is not yet supported: {quant_method!r}")

        for name in tensors_to_remove:
            if name in self.model_tensors:
                del self.model_tensors[name]

        for name, value in new_tensors.items():
            self.model_tensors[name] = value

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        if "language_model." in name:
            name = name.replace("language_model.", "")

        return name, gen

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, gen in self.model_tensors.items():
            yield name, gen()

    def format_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}")
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
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

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        raise NotImplementedError("set_gguf_parameters() must be implemented in subclasses")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        new_name = self.map_tensor_name(name)

        # Handle gate/up expert tensor fusion if enabled
        if self.fuse_gate_up_exps and bid is not None:
            if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_GATE_EXP, bid):
                self._gate_exp_buffer[bid] = data_torch
            elif self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_UP_EXP, bid):
                self._up_exp_buffer[bid] = data_torch

            # Check if both gate and up are buffered for this layer
            if bid in self._gate_exp_buffer and bid in self._up_exp_buffer:
                gate_data = self._gate_exp_buffer.pop(bid)
                up_data = self._up_exp_buffer.pop(bid)
                # gate/up shape: (n_expert, n_ff, n_embd), concatenate to (n_expert, n_ff*2, n_embd)
                fused_data = torch.cat([gate_data, up_data], dim=1)
                fused_name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, bid)
                logger.info(f"Fused gate_exps and up_exps for layer {bid}")
                return [(fused_name, fused_data)]

            # If we buffered a gate/up tensor, wait for the other
            if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_GATE_EXP, bid) or \
               self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_UP_EXP, bid):
                return []

        return [(new_name, data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid, n_dims  # unused

        return False

    # some models need extra generated tensors (like rope_freqs)
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    @staticmethod
    def _nvfp4_pack(weight: Tensor, scale: Tensor) -> tuple[np.ndarray, list[int]]:
        """Repack NVFP4 ModelOpt tensors into ggml super-block layout.
        Preserves original E4M3 scale bits as UE4M3 (strip sign bit).
        The per-tensor scale2 factor is stored as a separate tensor and applied at inference time via ggml_mul().
        Returns (raw_data, logical_shape)."""

        out_features = weight.shape[0]
        n_blocks = scale.shape[1]

        # Unpack ModelOpt nibble-packed weights
        w = weight.reshape(out_features, n_blocks, 8)
        vals = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(out_features, n_blocks, 16)

        # Preserve original E4M3 scale bits as UE4M3 (strip sign bit)
        d_ue = scale.view(torch.uint8).numpy().reshape(out_features, n_blocks) & 0x7F
        qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).to(torch.uint8).numpy()

        # Pack into super-blocks: [4 UE4M3 scales, 32 qs bytes] = 36 bytes per 64 elements
        n_super = n_blocks // 4
        d_grouped = d_ue.reshape(out_features, n_super, 4)
        qs_grouped = qs.reshape(out_features, n_super, 4, 8).reshape(out_features, n_super, 32)
        raw = np.concatenate([d_grouped, qs_grouped], axis=-1).reshape(out_features, n_super * 36)
        return raw, [out_features, n_super * 64]

    def _repack_nvfp4(self, name: str, weight: Tensor, scale: Tensor, scale2: Tensor, input_scale: Tensor):
        new_name = self.map_tensor_name(name)

        raw, shape = self._nvfp4_pack(weight, scale)
        logger.info(f"Repacked {new_name} with shape {shape} and quantization NVFP4")
        self.gguf_writer.add_tensor(new_name, raw, raw_dtype=gguf.GGMLQuantizationType.NVFP4)

        self._write_scale_tensor(new_name.replace(".weight", ".scale"), scale2)
        self._write_scale_tensor(new_name.replace(".weight", ".input_scale"), input_scale)

    def _generate_nvfp4_tensors(self):
        # Per-layer expert merging to avoid holding all experts in memory
        expert_blocks: dict[tuple[int, str], list[tuple[int, np.ndarray]]] = {}
        expert_scales: dict[tuple[int, str], list[tuple[int, float]]] = {}
        expert_input_scales: dict[tuple[int, str], list[tuple[int, float]]] = {}
        expert_shapes: dict[tuple[int, str], list[int]] = {}
        n_experts = self.find_hparam(["num_local_experts", "num_experts"], optional=True) or 0
        consumed: list[str] = []

        for name in self.model_tensors.keys():
            if not name.endswith(".weight"):
                continue
            scale_name = name.replace(".weight", ".weight_scale")
            scale2_name = name.replace(".weight", ".weight_scale_2")
            input_scale_name = name.replace(".weight", ".input_scale")
            if scale_name not in self.model_tensors:
                continue
            # Force eager materialization of lazy tensors
            weight = LazyTorchTensor.to_eager(self.model_tensors[name]())
            scale = LazyTorchTensor.to_eager(self.model_tensors[scale_name]())

            # Skip non-NVFP4 tensors (e.g. FP8 with per-channel 1D scales)
            if scale.ndim < 2:
                continue

            scale2 = LazyTorchTensor.to_eager(self.model_tensors.get(scale2_name, lambda: torch.tensor(1.0))())
            input_scale = LazyTorchTensor.to_eager(self.model_tensors.get(input_scale_name, lambda: torch.tensor(1.0))())

            # Mark tensors for removal from model_tensors (already written to gguf)
            consumed.extend([name, scale_name])
            if scale2_name in self.model_tensors:
                consumed.append(scale2_name)
            if input_scale_name in self.model_tensors:
                consumed.append(input_scale_name)

            # Check if this is a per-expert tensor
            m = re.search(r'\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$', name)
            if m:
                expert_id = int(m.group(1))
                proj_type = m.group(2)
                bid_m = re.search(r'\.layers\.(\d+)\.', name)
                bid = int(bid_m.group(1)) if bid_m else 0
                key = (bid, proj_type)

                raw, shape = self._nvfp4_pack(weight, scale)

                if key not in expert_blocks:
                    expert_blocks[key] = []
                    expert_scales[key] = []
                    expert_input_scales[key] = []
                    expert_shapes[key] = shape
                expert_blocks[key].append((expert_id, raw.copy()))
                # Collect per-expert scale2 (scalar per expert)
                expert_scales[key].append((expert_id, float(scale2.float().sum())))
                # Collect per-expert input_scale (scalar per expert)
                expert_input_scales[key].append((expert_id, float(input_scale.float().sum())))

                # Flush when all experts for this (layer, proj) are collected
                if n_experts > 0 and len(expert_blocks[key]) >= n_experts:
                    self._flush_nvfp4_experts(key, expert_blocks, expert_scales, expert_input_scales, expert_shapes, bid, proj_type)
            else:
                self._repack_nvfp4(name, weight, scale, scale2, input_scale)

        # Flush any remaining experts (fallback if n_experts was unknown)
        for bid, proj_type in list(expert_blocks.keys()):
            self._flush_nvfp4_experts((bid, proj_type), expert_blocks, expert_scales, expert_input_scales, expert_shapes, bid, proj_type)

        # Remove consumed tensors so get_tensors/modify_tensors won't see them
        for name in consumed:
            self.model_tensors.pop(name, None)

        # Remove any remaining unused auxiliary tensors
        for name in list(self.model_tensors.keys()):
            if name.endswith((".k_scale", ".v_scale")):
                del self.model_tensors[name]

    def _flush_nvfp4_experts(self, key, expert_blocks, expert_scales, expert_input_scales, expert_shapes, bid, proj_type):
        experts = expert_blocks.pop(key)
        scales = expert_scales.pop(key)
        input_scales = expert_input_scales.pop(key)
        shape = expert_shapes.pop(key)

        experts.sort(key=lambda x: x[0])
        merged = np.stack([e[1] for e in experts], axis=0)
        merged_name = f"model.layers.{bid}.mlp.experts.{proj_type}.weight"
        new_name = self.map_tensor_name(merged_name)
        logger.info(f"Repacked {new_name} with shape [{len(experts)}, {shape[0]}, {shape[1]}] and quantization NVFP4")
        self.gguf_writer.add_tensor(new_name, merged, raw_dtype=gguf.GGMLQuantizationType.NVFP4)

        scales.sort(key=lambda x: x[0])
        self._write_scales_tensor(new_name.replace(".weight", ".scale"), [s[1] for s in scales])

        input_scales.sort(key=lambda x: x[0])
        self._write_scales_tensor(new_name.replace(".weight", ".input_scale"), [s[1] for s in input_scales])

        del experts, merged

    def prepare_tensors(self):
        # detect NVFP4 quantization (ModelOpt format)
        quant_algo = (self.hparams.get("quantization_config") or {}).get("quant_algo")
        quant_method = (self.hparams.get("quantization_config") or {}).get("quant_method")
        quant_layers = (self.hparams.get("quantization_config") or {}).get("quantized_layers") or {}
        quant_config_file = self.dir_model / "hf_quant_config.json"

        if (not quant_algo or not quant_layers) and quant_config_file.is_file():
            with open(quant_config_file, "r", encoding="utf-8") as f:
                hf_quant_config = json.load(f)
                quant_config = hf_quant_config.get("quantization") or {}
                producer = hf_quant_config.get("producer") or {}
                producer_name = (producer.get("name") or "").lower()
                if quant_method is None:
                    self.hparams.setdefault("quantization_config", {})["quant_method"] = producer_name
                quant_algo = quant_config.get("quant_algo", quant_algo)
                quant_layers = quant_config.get("quantized_layers", quant_layers) or {}

        # Some models use per-tensor quant_algo (e.g. "MIXED_PRECISION" with
        # per-layer NVFP4/FP8) instead of a single global "NVFP4" value.
        if quant_algo != "NVFP4":
            if any(v.get("quant_algo") == "NVFP4" for v in quant_layers.values() if isinstance(v, dict)):
                quant_algo = "NVFP4"

        self._is_nvfp4 = quant_algo == "NVFP4"
        self._is_mxfp4 = quant_method == "mxfp4"

        # NVFP4 weights are repacked and written directly to gguf_writer.
        # This must run before dequant_model so NVFP4 tensors are removed
        # from model_tensors, leaving only non-NVFP4 (e.g. FP8) for dequant.
        if self._is_nvfp4:
            self._generate_nvfp4_tensors()

        self.dequant_model()

        # Handle empty tensor_map for models with block_count=0 (like MobileNetV5)
        if self.tensor_map.mapping:
            max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")
        else:
            max_name_len = len("vision_encoder.weight,")  # Default reasonable length

        for name, data_torch in chain(self.generate_extra_tensors(), self.get_tensors()):
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
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

            for new_name, data_torch in (self.modify_tensors(data_torch, name, bid)):
                # TODO: why do we squeeze here?
                # data = data_torch.squeeze().numpy()
                data = data_torch.numpy()

                n_dims = len(data.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = self.tensor_force_quant(name, new_name, bid, n_dims)

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
                            gguf.MODEL_TENSOR.FFN_GATE_INP_SHEXP,
                            gguf.MODEL_TENSOR.POS_EMBD,
                            gguf.MODEL_TENSOR.TOKEN_TYPES,
                            gguf.MODEL_TENSOR.SSM_CONV1D,
                            gguf.MODEL_TENSOR.SHORTCONV_CONV,
                            gguf.MODEL_TENSOR.TIME_MIX_FIRST,
                            gguf.MODEL_TENSOR.TIME_MIX_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_W2,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W2,
                            gguf.MODEL_TENSOR.TIME_MIX_LERP_FUSED,
                            gguf.MODEL_TENSOR.POSNET_NORM1,
                            gguf.MODEL_TENSOR.POSNET_NORM2,
                            gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                            gguf.MODEL_TENSOR.A_ENC_EMBD_POS,
                            gguf.MODEL_TENSOR.ALTUP_CORRECT_COEF,
                            gguf.MODEL_TENSOR.ALTUP_PREDICT_COEF,
                            # Kimi KDA conv weights should be F32
                            gguf.MODEL_TENSOR.SSM_CONV1D_Q,
                            gguf.MODEL_TENSOR.SSM_CONV1D_K,
                            gguf.MODEL_TENSOR.SSM_CONV1D_V,
                        )
                    )
                    or new_name[-7:] not in (".weight", ".lora_a", ".lora_b")
                ):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and any(
                    self.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.TOKEN_EMBD,
                        gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
                        gguf.MODEL_TENSOR.OUTPUT,
                        gguf.MODEL_TENSOR.ALTUP_ROUTER,
                        gguf.MODEL_TENSOR.LAUREL_L,
                        gguf.MODEL_TENSOR.LAUREL_R,
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

                shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self, vocab_only: bool):

        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()

        self.metadata = gguf.Metadata.load(self.metadata_override, self.dir_model_card, self.model_name, total_params)

        # If we are using HF model id, set the metadata name to the model id
        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = self.dir_model.name

        if self.ftype in (gguf.LlamaFileType.ALL_F32, gguf.LlamaFileType.MOSTLY_F16, gguf.LlamaFileType.MOSTLY_BF16):
            if self._is_nvfp4:
                self.ftype = gguf.LlamaFileType.MOSTLY_NVFP4
            elif self._is_mxfp4:
                self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        self.set_type()

        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)

        logger.info("Set model parameters")
        self.set_gguf_parameters()

        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def write_vocab(self):
        raise NotImplementedError("write_vocab() must be implemented in subclasses")

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    @staticmethod
    def load_hparams(dir_model: Path, is_mistral_format: bool):
        if is_mistral_format:
            with open(dir_model / "params.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            return config

        try:
            # for security reason, we don't allow loading remote code by default
            # if a model need remote code, we will fallback to config.json
            config = AutoConfig.from_pretrained(dir_model, trust_remote_code=False).to_dict()
        except Exception as e:
            logger.warning(f"Failed to load model config from {dir_model}: {e}")
            logger.warning("Trying to load config.json instead")
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        if "llm_config" in config:
            # rename for InternVL
            config["text_config"] = config["llm_config"]
        if "lm_config" in config:
            # rename for GlmASR
            config["text_config"] = config["lm_config"]
        if "thinker_config" in config:
            # rename for Qwen2.5-Omni
            config["text_config"] = config["thinker_config"]["text_config"]
        if "language_config" in config:
            # rename for DeepSeekOCR
            config["text_config"] = config["language_config"]
        if "lfm" in config:
            # rename for LFM2-Audio
            config["text_config"] = config["lfm"]
        return config

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: AnyModel) -> AnyModel:
            model_type = ModelType.MMPROJ if modelcls.model_arch == gguf.MODEL_ARCH.MMPROJ else ModelType.TEXT
            for name in names:
                cls._model_classes[model_type][name] = modelcls
            return modelcls
        return func

    @classmethod
    def print_registered_models(cls):
        for model_type, model_classes in cls._model_classes.items():
            logger.error(f"{model_type.name} models:")
            for name in sorted(model_classes.keys()):
                logger.error(f"  - {name}")

    @classmethod
    def from_model_architecture(cls, arch: str, model_type = ModelType.TEXT) -> type[ModelBase]:
        try:
            return cls._model_classes[model_type][arch]
        except KeyError:
            raise NotImplementedError(f'Architecture {arch!r} not supported!') from None


class TextModel(ModelBase):
    model_type = ModelType.TEXT
    hf_arch: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_mistral_format:
            self.hf_arch = get_model_architecture(self.hparams, self.model_type)
        else:
            self.hf_arch = ""

        if "text_config" in self.hparams:
            # move the text_config to the root level
            self.hparams = {**self.hparams, **self.hparams["text_config"]}

        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self.rope_parameters = self.hparams.get("rope_parameters", self.hparams.get("rope_scaling")) or {}

        rope_theta = self.find_hparam(["global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base"], optional=True)
        local_rope_theta = self.find_hparam(["local_rope_theta", "rope_local_theta", "rope_theta_local", "swa_rope_theta", "rope_local_base_freq"], optional=True)

        # Ensure "rope_theta" and "rope_type" is mirrored in rope_parameters
        if "full_attention" not in self.rope_parameters and "sliding_attention" not in self.rope_parameters:
            if local_rope_theta is not None:
                self.rope_parameters["sliding_attention"] = {"rope_theta": local_rope_theta}
            if "rope_theta" not in self.rope_parameters and rope_theta is not None:
                self.rope_parameters["rope_theta"] = rope_theta
            if "rope_type" not in self.rope_parameters and (rope_type := self.rope_parameters.get("type")) is not None:
                self.rope_parameters["rope_type"] = rope_type

    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip multimodal tensors
        if name.startswith(("mlp", "vit.", "vpm.", "siglip2.", "conformer.", "merger.", "resampler.", "sound_encoder.", "sound_projection.", "speech_embeddings.")) \
                or "visual." in name or "vision." in name or "audio." in name or "talker." in name \
                or "vision_" in name or "audio_" in name or "sam_model" in name \
                or "token2wav." in name or "code2wav." in name \
                or "projector." in name or "pre_mm_projector_norm" in name \
                or "image_newline" in name or "view_seperator" in name \
                or "patch_embed" in name or "patch_embedding" in name \
                or "patch_merger." in name or "model.connector." in name:
            return None

        return super().filter_tensors(item)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)

        total_params = self.gguf_writer.get_total_parameter_count()[0]
        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = self.ftype.name.partition("_")[2]

        # Filename Output
        if self.fname_out.is_dir():
            # Generate default filename based on model specification and available metadata
            if not vocab_only:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, self.metadata.size_label, output_type, model_type="LoRA" if total_params < 0 else None)
            else:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, size_label=None, output_type=None, model_type="vocab")

            # Use the default filename
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
        else:
            # Output path is a custom defined templated filename
            # Note: `not is_dir()` is used because `.is_file()` will not detect
            #       file template strings as it doesn't actually exist as a file

            # Process templated file name with the output ftype, useful with the "auto" ftype
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)

        logger.info("Set model tokenizer")
        self.set_vocab()

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        if (n_embd := self.find_hparam(["hidden_size", "n_embd", "dim"], optional=True)) is not None:
            self.gguf_writer.add_embedding_length(n_embd)
            logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner", "hidden_dim"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        if (n_head := self.find_hparam(["num_attention_heads", "n_head", "n_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count(n_head)
            logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.find_hparam(["num_key_value_heads", "n_kv_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if self.hparams.get("is_causal") is False:
            self.gguf_writer.add_causal_attention(False)
            logger.info("gguf: causal attention = False")

        # TODO: Handle "sliding_attention" similarly when models start implementing it
        rope_params = self.rope_parameters.get("full_attention", self.rope_parameters)
        if (rope_type := rope_params.get("rope_type")) is not None:
            rope_factor = rope_params.get("factor")
            rope_gguf_type = gguf.RopeScalingType.NONE
            if rope_type == "linear" and rope_factor is not None:
                rope_gguf_type = gguf.RopeScalingType.LINEAR
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
                self.gguf_writer.add_rope_scaling_factor(rope_factor)
            elif rope_type == "yarn" and rope_factor is not None:
                rope_gguf_type = gguf.RopeScalingType.YARN
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
                self.gguf_writer.add_rope_scaling_factor(rope_factor)
                self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_params["original_max_position_embeddings"])
                if (yarn_ext_factor := rope_params.get("extrapolation_factor")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_ext_factor(yarn_ext_factor)
                if (yarn_attn_factor := rope_params.get("attention_factor", rope_params.get("attn_factor"))) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_attn_factor(yarn_attn_factor)
                if (yarn_beta_fast := rope_params.get("beta_fast")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_beta_fast(yarn_beta_fast)
                if (yarn_beta_slow := rope_params.get("beta_slow")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_beta_slow(yarn_beta_slow)
                # self.gguf_writer.add_rope_scaling_yarn_log_mul(rope_params["mscale_all_dim"])
            elif rope_type == "su" or rope_type == "longrope":
                rope_gguf_type = gguf.RopeScalingType.LONGROPE
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
            elif rope_type == "dynamic":
                # HunYuan, handled in model class
                pass
            elif rope_type.lower() == "llama3":
                # Handled in generate_extra_tensors
                pass
            else:
                logger.warning(f"Unknown RoPE type: {rope_type}")
            logger.info(f"gguf: rope scaling type = {rope_gguf_type.name}")

        if "mrope_section" in self.rope_parameters:
            mrope_section = self.rope_parameters["mrope_section"]
            # Pad to 4 dimensions [time, height, width, extra]
            while len(mrope_section) < 4:
                mrope_section.append(0)
            self.gguf_writer.add_rope_dimension_sections(mrope_section[:4])
            logger.info(f"gguf: mrope sections: {mrope_section[:4]}")

        if (rope_theta := rope_params.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (local_rope_theta := self.rope_parameters.get("sliding_attention", {}).get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base_swa(local_rope_theta)
            logger.info(f"gguf: rope theta swa = {local_rope_theta}")
        if (f_rms_eps := self.find_hparam(["rms_norm_eps", "norm_eps"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.find_hparam(["num_local_experts", "num_experts"], optional=True)) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.find_hparam(["num_experts_per_tok", "num_experts_per_token", "top_k_experts"], optional=True)) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")
        if (n_expert_groups := self.hparams.get("n_group")) is not None:
            self.gguf_writer.add_expert_group_count(n_expert_groups)
            logger.info(f"gguf: expert groups count = {n_expert_groups}")
        if (n_group_used := self.hparams.get("topk_group")) is not None:
            self.gguf_writer.add_expert_group_used_count(n_group_used)
            logger.info(f"gguf: expert groups used count = {n_group_used}")

        if (score_func := self.find_hparam(["score_function", "scoring_func", "score_func", "moe_router_activation", "moe_router_activation_func"], optional=True)) is not None:
            if score_func == "sigmoid":
                self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
            elif score_func == "softmax":
                self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
            else:
                raise ValueError(f"Unsupported expert score gating function value: {score_func}")
            logger.info(f"gguf: expert score gating function = {score_func}")

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def write_vocab(self):
        if len(self.gguf_writer.tensors) != 1:
            raise ValueError('Splitting the vocabulary is not supported')

        self.prepare_metadata(vocab_only=True)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token

        # Some models mark some added tokens which ought to be control tokens as not special.
        # (e.g. command-r, command-r-plus, deepseek-coder, gemma{,-2})
        seems_special = token_text in (
            "<pad>",  # deepseek-coder
            "<mask>", "<2mass>", "[@BOS@]",  # gemma{,-2}
        )

        seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
        seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))  # deepseek-coder

        # TODO: should these be marked as UNUSED instead? (maybe not)
        seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))  # gemma{,-2}

        return seems_special

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))  # ty: ignore[unresolved-attribute]
        assert max(tokenizer.vocab.values()) < vocab_size  # ty: ignore[unresolved-attribute]

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}  # ty: ignore[unresolved-attribute]
        added_vocab = tokenizer.get_added_vocab()  # ty: ignore[unresolved-attribute]

        added_tokens_decoder = tokenizer.added_tokens_decoder  # ty: ignore[unresolved-attribute]

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    # The tokenizer in llama.cpp assumes the CONTROL and USER_DEFINED tokens are pre-normalized.
                    # To avoid unexpected issues - we make sure to normalize non-normalized tokens
                    if not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))  # ty: ignore[unresolved-attribute, invalid-assignment]
                        if previous_token != token:
                            logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using AutoTokenizer")

                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        # NOTE: this was added for Gemma.
                        # Encoding and decoding the tokens above isn't sufficient for this case.
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert_hf_to_gguf_update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggml-org/llama.cpp/pull/6920
    # Marker: Start get_vocab_base_pre
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2":
            # ref: https://huggingface.co/THUDM/glm-4-9b-hf
            res = "glm4"
        if chkhsh == "9ca2dd618e8afaf09731a7cf6e2105b373ba6a1821559f258b272fe83e6eb902":
            # ref: https://huggingface.co/zai-org/GLM-4.5-Air
            res = "glm4"
        if chkhsh == "cdf5f35325780597efd76153d4d1c16778f766173908894c04afc20108536267":
            # ref: https://huggingface.co/zai-org/GLM-4.7-Flash
            res = "glm4"
        if chkhsh == "1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35":
            # ref: https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0
            res = "minerva-7b"
        if chkhsh == "7e57df22b1fe23a7b1e1c7f3dc4e3f96d43a4eb0836d0c6bdc3436d7b2f1c664":
            # ref: https://huggingface.co/tencent/Hunyuan-A13B-Instruct
            res = "hunyuan"
        if chkhsh == "bba3b3366b646dbdded5dbc42d59598b849371afc42f7beafa914afaa5b70aa6":
            # ref: https://huggingface.co/tencent/Hunyuan-4B-Instruct
            res = "hunyuan-dense"
        if chkhsh == "a6b57017d60e6edb4d88ecc2845188e0eb333a70357e45dcc9b53964a73bbae6":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base
            res = "falcon-h1"
        if chkhsh == "60476e1243776c4fb1b993dbd7a5f15ac22f83c80afdf425fa5ae01c8d44ef86":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-1B-Base
            res = "falcon-h1"
        if chkhsh == "3eda48b4c4dc7de733d1a8b3e3b4a85243dbbf704da2ee9d42c6beced8897896":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-7B-Base
            res = "falcon-h1"
        if chkhsh == "48f8e02c0359c0bbdd82f26909171fac1c18a457bb47573ed1fe3bbb2c1cfd4b":
            # ref: https://huggingface.co/tiiuae/Falcon-H1-34B-Base
            res = "falcon-h1"
        if chkhsh == "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890":
            # ref: https://huggingface.co/moonshotai/Kimi-K2-Base
            res = "kimi-k2"
        if chkhsh == "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c":
            # ref: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
            res = "qwen2"
        if chkhsh == "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f":
            # ref: https://huggingface.co/openbmb/MiniCPM-V-4_6
            res = "qwen35"
        if chkhsh == "66b8d4e19ab16c3bfd89bce5d785fb7e0155e8648708a1f42077cb9fe002c273":
            # ref: https://huggingface.co/alvarobartt/grok-2-tokenizer
            res = "grok-2"
        if chkhsh == "b3d1dd861f1d4c5c0d2569ce36baf3f90fe8a102db3de50dd71ff860d91be3df":
            # ref: https://huggingface.co/aari1995/German_Semantic_V3
            res = "jina-v2-de"
        if chkhsh == "0fe1cf6eda062318a1af7270f3331a85c539a01778ff948e24388e949c5282f4":
            # ref: https://huggingface.co/evilfreelancer/ruGPT3XL
            res = "gpt-2"
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "9d032fcbd5501f4a38150912590928bfb36091efb5df11b8e2124b0390e3fb1e":
            # ref: https://huggingface.co/tiiuae/Falcon3-7B-Base
            res = "falcon3"
        if chkhsh == "8e62295832751ca1e8f92f2226f403dea30dc5165e448b5bfa05af5340c64ec7":
            # ref: https://huggingface.co/BAAI/bge-large-zh-v1.5
            res = "bert-bge-large"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3":
            # ref: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
            res = "stablelm2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"
        if chkhsh == "d772b220ace2baec124bed8cfafce0ead7d6c38a4b65ef11261cf9d5d62246d1":
            # ref: https://huggingface.co/CohereLabs/tiny-aya-base
            res = "tiny_aya"
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            # ref: https://huggingface.co/Qwen/Qwen1.5-7B
            res = "qwen2"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/allenai/OLMo-1.7-7B-hf
            res = "olmo"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e":
            # ref: https://huggingface.co/databricks/dbrx-base
            res = "dbrx"
        if chkhsh == "c7699093ba4255a91e702aa38a596aa81669f3525dae06c2953267dde580f448":
            # ref: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
            res = "jina-v1-en"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-en
            res = "jina-v2-en"
        if chkhsh == "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-es
            res = "jina-v2-es"
        if chkhsh == "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-de
            res = "jina-v2-de"
        if chkhsh == "a023e9fdc5a11f034d3ef515b92350e56fb2af1f66c6b6811a4444ea9bf8763d":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano
            res = "jina-v5-nano"
        if chkhsh == "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d":
            # ref: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct
            res = "smaug-bpe"
        if chkhsh == "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360":
            # ref: https://huggingface.co/LumiOpen/Poro-34B-chat
            res = "poro-chat"
        if chkhsh == "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
            res = "jina-v2-code"
        if chkhsh == "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee":
            # ref: https://huggingface.co/LumiOpen/Viking-7B
            res = "viking"
        if chkhsh == "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901":
            # ref: https://huggingface.co/core42/jais-13b
            res = "jais"
        if chkhsh == "bc5108ee1eb6a3d600cadd065f63190fbd0554dbc9e4bbd6a0d977970afc8d2a":
            # ref: https://huggingface.co/inceptionai/Jais-2-8B-Chat
            res = "jais-2"
        if chkhsh == "7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f":
            # ref: https://huggingface.co/WisdomShell/CodeShell-7B
            res = "codeshell"
        if chkhsh == "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e":
            # ref: https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
            res = "tekken"
        if chkhsh == "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249":
            # ref: https://huggingface.co/HuggingFaceTB/SmolLM-135M
            res = "smollm"
        if chkhsh == "3c30d3ad1d6b64202cd222813e7736c2db6e1bd6d67197090fc1211fbc612ae7":
            # ref: https://huggingface.co/bigscience/bloom
            res = "bloom"
        if chkhsh == "bc01ce58980e1db43859146dc51b1758b3b88729b217a74792e9f8d43e479d21":
            # ref: https://huggingface.co/TurkuNLP/gpt3-finnish-small
            res = "gpt3-finnish"
        if chkhsh == "4e2b24cc4770243d65a2c9ec19770a72f08cffc161adbb73fcbb6b7dd45a0aae":
            # ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
            res = "exaone"
        if chkhsh == "fcace8b9cac38ce847670c970cd5892031a753a1ef381abd1d9af00f713da085":
            # ref: https://huggingface.co/microsoft/phi-2
            res = "phi-2"
        if chkhsh == "60824e3c0d9401f89943cbb2fff727f0e2d4c545ba4df2d6e4f09a6db0f5b450":
            # ref: https://huggingface.co/facebook/chameleon-7b
            res = "chameleon"
        if chkhsh == "8b5a93ed704057481f240da0be7e7dca721d7f8f4755263b6807227a2cbeae65":
            # ref: https://huggingface.co/sentence-transformers/stsb-roberta-base
            res = "roberta-bpe"
        if chkhsh == "ad851be1dba641f2e3711822f816db2c265f788b37c63b4e1aeacb9ee92de8eb":
            # ref: https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct
            res = "gigachat"
        if chkhsh == "d4c8f286ea6b520b3d495c4455483cfa2302c0cfcd4be05d781b6a8a0a7cdaf1":
            # ref: https://huggingface.co/Infinigence/Megrez-3B-Instruct
            res = "megrez"
        if chkhsh == "877081d19cf6996e2c4ff0e1236341e9b7bde288f5311a56a937f0afbbb3aeb5":
            # ref: https://huggingface.co/deepseek-ai/DeepSeek-V3
            res = "deepseek-v3"
        if chkhsh == "b3f499bb4255f8ca19fccd664443283318f2fd2414d5e0b040fbdd0cc195d6c5":
            # ref: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
            res = "deepseek-r1-qwen"
        if chkhsh == "ccc2ef013c104be7bae2965776d611e1d7a8a2a9c547dd93a682c9a9fc80352e":
            # ref: https://huggingface.co/Xenova/gpt-4o
            res = "gpt-4o"
        if chkhsh == "7dec86086fcc38b66b7bc1575a160ae21cf705be7718b9d5598190d7c12db76f":
            # ref: https://huggingface.co/UW/OLMo2-8B-SuperBPE-t180k
            res = "superbpe"
        if chkhsh == "1994ffd01900cfb37395608534236ecd63f2bd5995d6cb1004dda1af50240f15":
            # ref: https://huggingface.co/trillionlabs/Trillion-7B-preview
            res = "trillion"
        if chkhsh == "96a5f08be6259352137b512d4157e333e21df7edd3fcd152990608735a65b224":
            # ref: https://huggingface.co/inclusionAI/Ling-lite
            res = "bailingmoe"
        if chkhsh == "d353350c764d8c3b39c763113960e4fb4919bea5fbf208a0e3b22e8469dc7406":
            # ref: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
            res = "llama4"
        if chkhsh == "0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3":
            # ref: https://huggingface.co/mistral-community/pixtral-12b
            res = "pixtral"
        if chkhsh == "d5f1dd6f980fec569fb218a81a7658ac45fc56b38c5a0adeb1c232fbe04ef5ec":
            # ref: https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base
            res = "seed-coder"
        if chkhsh == "b0a6b1c0bd5998ebd9df08611efde34a4ff03faed45ae09c43e6b31ebd4b94cf":
            # ref: https://huggingface.co/skt/A.X-4.0
            res = "a.x-4.0"
        if chkhsh == "f6791d196f87ce6b56a7d234be618e0d58f8cda3549416635b2bebcd22cd95c4":
            # ref: https://huggingface.co/K-intelligence/Midm-2.0-Base-Instruct
            res = "midm-2.0"
        if chkhsh == "169bf0296a13c4d9b7672313f749eb36501d931022de052aad6e36f2bf34dd51":
            # ref: https://huggingface.co/LiquidAI/LFM2-Tokenizer
            res = "lfm2"
        if chkhsh == "2085e1638f6c377a0aa4ead21b27bb4cb941bf800df86ed391011769c1758dfb":
            # ref: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B
            res = "exaone4"
        if chkhsh == "a1e163ecab2e718a4c829d1148b6e86824ec36163bb71941c3dca9cd5ac25756":
            # ref: https://huggingface.co/JetBrains/Mellum-4b-base
            res = "mellum"
        if chkhsh == "a0b64b4385f123663873756336c085744376d015ff328bb1d901598f63c44152":
            # ref: https://huggingface.co/answerdotai/ModernBERT-base
            res = "modern-bert"
        if chkhsh == "49fc0303c9e0d2c2c565c510f64b2d9b271276acdcdadff733249eda9f7d59df":
            # ref: https://huggingface.co/arcee-ai/Trinity-Tokenizer
            res = "afmoe"
        if chkhsh == "9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206":
            # ref: https://huggingface.co/inclusionAI/Ling-mini-base-2.0
            res = "bailingmoe2"
        if chkhsh == "53e325976a6e142379c19b09afcae354f2f496f147afa8f9e189a33fe4e3024e":
            # ref: https://huggingface.co/ibm-granite/granite-docling-258M
            res = "granite-docling"
        if chkhsh == "f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95":
            # ref: https://huggingface.co/MiniMaxAI/MiniMax-M2
            res = "minimax-m2"
        if chkhsh == "4a2e2abae11ca2b86d570fc5b44be4d5eb5e72cc8f22dd136a94b37da83ab665":
            # ref: https://huggingface.co/KORMo-Team/KORMo-tokenizer
            res = "kormo"
        if chkhsh == "9d70134b369a70e5735009b6de918f7581b5211f7c074d1f89f753aea8248af1":
            # ref: https://huggingface.co/tencent/Youtu-LLM-2B
            res = "youtu"
        if chkhsh == "16389f0a1f51ee53e562ffd51c371dc508639ab0e4261502071836e50e223e91":
            # ref: https://huggingface.co/upstage/Solar-Open-100B
            res = "solar-open"
        if chkhsh == "6c81ce329e0802883b22eabab0d3fa48357337ef1ecb45443828bf1f6254833f":
            # ref: https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
            res = "exaone-moe"
        if chkhsh == "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4":
            # ref: https://huggingface.co/Qwen/Qwen3.5-9B-Instruct
            res = "qwen35"
        if chkhsh == "b4b8ca1f9769494fbd956ebc4c249de6131fb277a4a3345a7a92c7dd7a55808d":
            # ref: https://huggingface.co/jdopensource/JoyAI-LLM-Flash
            res = "joyai-llm"
        if chkhsh == "e4d54df1ebc1f2b91acd986c5b51aa50837d5faf7c7398e73c1f9e9ee5d19869":
            # ref: https://huggingface.co/kakaocorp/kanana-2-30b-a3b-instruct-2601
            res = "kanana2"
        if chkhsh == "862f827721df956049dff5ca81a57f29e575280bc622e290d3bf4e35eca29015":
            # ref: https://huggingface.co/codefuse-ai/F2LLM-v2-4B
            res = "f2llmv2"
        if chkhsh == "62f6fb0a6fd5098caeabb19b07a5c1099cafc8b9c40eab6ea89ece4ec02fbc57":
            # ref: https://huggingface.co/sarvamai/sarvam-30b
            res = "sarvam-moe"

        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggml-org/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res
        # Marker: End get_vocab_base_pre

    def _set_vocab_none(self) -> None:
        self.gguf_writer.add_tokenizer_model("none")

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        from .qwen import QwenModel

        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size  # ty: ignore[unresolved-attribute]

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks  # ty: ignore[unresolved-attribute]
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **added_vocab}.items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        # only add special tokens when they were not already loaded from config.json
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self, add_to_gguf=True):
        tokens, scores, toktypes = self._create_vocab_sentencepiece()

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _create_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.find_hparam([
            "vocab_size_per_layer_input", # gemma3n
            "vocab_size",
        ], optional=True) or tokenizer.vocab_size()

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            if token_id >= vocab_size:
                logger.warning(f'ignore tokens from {token_id}: id is out of range, max={vocab_size - 1}')
                break

            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get("added_tokens_decoder", {})
                for token_id, token_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token: str = token_data["content"]
                    if token_id >= vocab_size:
                        logger.warning(f'ignore token {token_id}: id is out of range, max={vocab_size - 1}')
                        continue
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token.encode("utf-8"):
                            logger.warning(f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token!r}')
                    if token_data.get("special") or self.does_token_look_special(token):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL
                    else:
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

                    scores[token_id] = -1000.0
                    tokens[token_id] = token.encode("utf-8")

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        return tokens, scores, toktypes

    def _set_vocab_llama_hf(self):
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_rwkv_world(self):
        assert (self.dir_model / "rwkv_vocab_v20230424.txt").is_file()
        vocab_size = self.hparams.get("vocab_size", 65536)

        tokens: list[bytes] = ['<s>'.encode("utf-8")]
        toktypes: list[int] = [gguf.TokenType.CONTROL]

        with open(self.dir_model / "rwkv_vocab_v20230424.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split(' ')
                assert len(parts) >= 3
                token, token_len = ast.literal_eval(' '.join(parts[1:-1])), int(parts[-1])
                token = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token, bytes)
                assert len(token) == token_len
                token_text: str = repr(token)[2:-1]  # "b'\xff'" -> "\xff"
                tokens.append(token_text.encode("utf-8"))
                toktypes.append(gguf.TokenType.NORMAL)
        remainder = vocab_size - len(tokens)
        assert remainder >= 0
        for i in range(len(tokens), vocab_size):
            tokens.append(f"[PAD{i}]".encode("utf-8"))
            toktypes.append(gguf.TokenType.UNUSED)

        self.gguf_writer.add_tokenizer_model("rwkv")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        if special_vocab.chat_template is None:
            template_path = Path(__file__).parent.parent / "models" / "templates" / "llama-cpp-rwkv-world.jinja"
            if template_path.is_file():
                with open(template_path, "r", encoding="utf-8") as f:
                    template = f.read()
            else:
                template = "rwkv-world"
            special_vocab.chat_template = template
        # hack: Add '\n\n' as the EOT token to make it chat normally
        special_vocab._set_special_token("eot", 261)
        # hack: Override these as they have already been set (incorrectly)
        special_vocab.special_token_ids["bos"] = 0
        special_vocab.special_token_ids["eos"] = 0

        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_builtin(self, model_name: Literal["gpt-neox", "llama-spm"], vocab_size: int):
        tokenizer_path = Path(sys.path[0]) / "models" / f"ggml-vocab-{model_name}.gguf"
        logger.warning(f"Using tokenizer from '{os.path.relpath(tokenizer_path, os.getcwd())}'")
        vocab_reader = gguf.GGUFReader(tokenizer_path, "r")

        default_pre = "mpt" if model_name == "gpt-neox" else "default"

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.MODEL)
        assert field  # tokenizer model
        self.gguf_writer.add_tokenizer_model(bytes(field.parts[-1]).decode("utf-8"))

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.PRE)
        self.gguf_writer.add_tokenizer_pre(bytes(field.parts[-1]).decode("utf-8") if field else default_pre)

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.LIST)
        assert field  # token list
        self.gguf_writer.add_token_list([bytes(field.parts[i]) for i in field.data][:vocab_size])

        if model_name == "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.SCORES)
            assert field  # token scores
            self.gguf_writer.add_token_scores([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
        assert field  # token types
        self.gguf_writer.add_token_types([field.parts[i].tolist()[0] for i in field.data][:vocab_size])

        if model_name != "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.MERGES)
            assert field  # token merges
            self.gguf_writer.add_token_merges([bytes(field.parts[i]) for i in field.data])

        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.BOS_ID)) is not None:
            self.gguf_writer.add_bos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.EOS_ID)) is not None:
            self.gguf_writer.add_eos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.UNK_ID)) is not None:
            self.gguf_writer.add_unk_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.PAD_ID)) is not None:
            self.gguf_writer.add_pad_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_BOS)) is not None:
            self.gguf_writer.add_add_bos_token(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_EOS)) is not None:
            self.gguf_writer.add_add_eos_token(field.parts[-1].tolist()[0])

    def _try_set_pooling_type(self) -> None:
        # get pooling path
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"].endswith("Pooling"):
                    pooling_path = mod["path"]
                    break

        mode_mapping = {
            "mean": gguf.PoolingType.MEAN,
            "cls": gguf.PoolingType.CLS,
            "lasttoken": gguf.PoolingType.LAST,
        }

        # get pooling type
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling.get("pooling_mode_mean_tokens"):
                pooling_type = gguf.PoolingType.MEAN
            elif pooling.get("pooling_mode_cls_token"):
                pooling_type = gguf.PoolingType.CLS
            elif pooling.get("pooling_mode_lasttoken"):
                pooling_type = gguf.PoolingType.LAST
            elif (pooling_mode := pooling.get("pooling_mode")) in mode_mapping:
                pooling_type = mode_mapping[pooling_mode]
            else:
                raise NotImplementedError("Only MEAN, CLS, and LAST pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)

    def _set_vocab_glmedge(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_glm(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        # Special tokens
        # Note: Using <|endoftext|> (151329) for eot causes endless generation
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["[gMASK]"])  # ty: ignore[unresolved-attribute]  # 151331
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # ty: ignore[unresolved-attribute]  # 151336
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]  # 151329
        special_vocab._set_special_token("eom", tokenizer.get_added_vocab()["<|observation|>"])  # ty: ignore[unresolved-attribute]  # 151338
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_interns1(self):
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        vocab = getattr(tokenizer, 'vocab', tokenizer.get_vocab())  # ty: ignore[unresolved-attribute]
        vocab_size = self.hparams.get("vocab_size", len(vocab))
        assert max(vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
        added_vocab = tokenizer.get_added_vocab()  # ty: ignore[unresolved-attribute]

        added_tokens_decoder = tokenizer.added_tokens_decoder  # ty: ignore[unresolved-attribute]

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    # The tokenizer in llama.cpp assumes the CONTROL and USER_DEFINED tokens are pre-normalized.
                    # To avoid unexpected issues - we make sure to normalize non-normalized tokens
                    if not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))  # ty: ignore[unresolved-attribute, invalid-assignment]
                        if previous_token != token:
                            logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using AutoTokenizer")

                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab._set_special_token("bos", 151643)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_mistral(self):
        from .mistral import MistralModel

        if not _mistral_common_installed:
            raise ImportError(_mistral_import_error_msg)

        vocab = MistralVocab(self.dir_model)
        logger.info(
            f"Converting tokenizer {vocab.tokenizer_type} of size {vocab.vocab_size}."
        )

        self.gguf_writer.add_tokenizer_model(vocab.gguf_tokenizer_model)

        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size, (
            f"token count ({len(tokens)}) != vocab size ({vocab.vocab_size})"
        )

        if vocab.tokenizer_type == MistralTokenizerType.tekken:
            self.gguf_writer.add_tokenizer_pre("tekken")
            self.gguf_writer.add_token_merges(
                vocab.extract_vocab_merges_from_model()
            )

        logger.info(
            f"Setting bos, eos, unk and pad token IDs to {vocab.bos_id}, {vocab.eos_id}, {vocab.unk_id}, {vocab.pad_id}."
        )

        self.gguf_writer.add_bos_token_id(vocab.bos_id)
        self.gguf_writer.add_eos_token_id(vocab.eos_id)
        self.gguf_writer.add_unk_token_id(vocab.unk_id)
        self.gguf_writer.add_pad_token_id(vocab.pad_id)

        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_vocab_size(vocab.vocab_size)

        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(False)

        local_template_file_path = self.dir_model / "chat_template.jinja"

        if self.is_mistral_format and local_template_file_path.is_file():
            # Ministral-3 and other new Mistral models come with chat templates.
            # ref: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/tree/main
            logger.info("Using an existing Mistral local chat template.")

            with open(local_template_file_path, "r", encoding="utf-8") as f:
                template = f.read()
        elif not self.is_mistral_format or not self.disable_mistral_community_chat_template:
            template_dir = Path(__file__).parent.parent / "models/templates/"

            # Log only for Mistral format that the official tokenization and detokenization is via `mistral-common`.
            if self.is_mistral_format:
                logger.info(
                    "Using a Mistral community chat template. These templates can be subject to errors in early days or weeks after a release. "
                    "Mistral recommends to use `mistral-common` to perform tokenization and detokenization."
                )
            template = MistralModel.get_community_chat_template(vocab, template_dir, self.is_mistral_format)
        else:
            logger.info("Not using a Mistral local or community chat template. Ensure to perform the tokenization and detokenization via `mistral-common`.")
            template = None

        if template is not None:
            self.gguf_writer.add_chat_template(template)

    def _set_vocab_plamo(self):
        # PLaMo models use a custom tokenizer with a .jsonl file
        tokenizer_jsonl_path = self.dir_model / "tokenizer.jsonl"
        tokenizer_config_path = self.dir_model / "tokenizer_config.json"

        if not tokenizer_jsonl_path.is_file():
            raise FileNotFoundError(f"PLaMo tokenizer file not found: {tokenizer_jsonl_path}")

        # Load tokenizer config
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        # Load tokens from JSONL file (actually a list format)
        tokens = []
        scores = []
        toktypes = []

        with open(tokenizer_jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    token_data = json.loads(line)
                    # Format: [token, score, type, ?, ?, ?, ?]
                    token = token_data[0].encode("utf-8")
                    score = float(token_data[1])
                    token_type_str = token_data[2] if len(token_data) > 2 else "NORMAL"

                    tokens.append(token)
                    scores.append(score)

                    if token_type_str == "UNKNOWN":
                        toktypes.append(gguf.TokenType.UNKNOWN)
                    elif token_type_str == "CONTROL":
                        toktypes.append(gguf.TokenType.CONTROL)
                    elif token_type_str == "BYTE":
                        toktypes.append(gguf.TokenType.BYTE)
                    else:
                        token_str = token_data[0]
                        if token_str.startswith("<|plamo:") and token_str.endswith("|>"):
                            toktypes.append(gguf.TokenType.CONTROL)
                        else:
                            toktypes.append(gguf.TokenType.NORMAL)

        vocab_size = self.hparams["vocab_size"]
        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]")
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(gguf.TokenType.UNUSED)

        self.gguf_writer.add_tokenizer_model("plamo2")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        if "bos_token" in tokenizer_config and tokenizer_config["bos_token"] is not None:
            token_id = tokens.index(tokenizer_config["bos_token"].encode("utf-8"))
            self.gguf_writer.add_bos_token_id(token_id)
        if "eos_token" in tokenizer_config and tokenizer_config["eos_token"] is not None:
            token_id = tokens.index(tokenizer_config["eos_token"].encode("utf-8"))
            self.gguf_writer.add_eos_token_id(token_id)
        if "pad_token" in tokenizer_config and tokenizer_config["pad_token"] is not None:
            token_id = tokens.index(tokenizer_config["pad_token"].encode("utf-8"))
            self.gguf_writer.add_pad_token_id(token_id)
        if "sep_token" in tokenizer_config and tokenizer_config["sep_token"] is not None:
            token_id = tokens.index(tokenizer_config["sep_token"].encode("utf-8"))
            self.gguf_writer.add_sep_token_id(token_id)
        if "unk_token" in tokenizer_config and tokenizer_config["unk_token"] is not None:
            token_id = tokens.index(tokenizer_config["unk_token"].encode("utf-8"))
            self.gguf_writer.add_unk_token_id(token_id)

        # Add <|plamo:op|> as EOT to ensure appropriate end of generation
        self.gguf_writer.add_eot_token_id(4)

        self.gguf_writer.add_add_space_prefix(False)


class MmprojModel(ModelBase):
    model_type = ModelType.MMPROJ
    model_arch = gguf.MODEL_ARCH.MMPROJ
    preprocessor_config: dict[str, Any]
    global_config: dict[str, Any]

    n_block_keys = ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth", "layers", "encoder_layers", "vt_num_hidden_layers"]

    has_vision_encoder: bool = True # by default
    has_audio_encoder: bool = False

    # for models having multiple encoders, we need to separate their hparams
    hparams_vision: dict[str, Any] | None = None
    hparams_audio: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model_arch != gguf.MODEL_ARCH.MMPROJ:
            raise TypeError("MmprojModel must be subclassed with model_arch = gguf.MODEL_ARCH.MMPROJ")

        # get n_embd of the text model
        if not self.is_mistral_format:
            if "text_config" not in self.hparams:
                self.hparams["text_config"] = {}
            if "audio_config" not in self.hparams:
                self.hparams["audio_config"] = {}
            text_config = {**self.hparams, **self.hparams["text_config"]}
            self.n_embd_text = text_config.get("hidden_size", text_config.get("n_embd", 0))
        else:
            text_config = {
                k: v for k, v in self.hparams.items() if k not in ["vision_encoder", "audio_encoder"]
            }
            # mistral native params.json: "dim" is the text hidden size ("hidden_dim" is the FFN intermediate size)
            self.n_embd_text = text_config.get("dim", 0)

        assert self.n_embd_text > 0, "n_embd not found in hparams"

        # move vision config to the top level, while preserving the original hparams in global_config
        import copy
        self.global_config = copy.deepcopy(self.hparams)
        self.hparams_vision = self.get_vision_config()
        self.hparams_audio = self.get_audio_config()

        if self.hparams_vision is None and self.hparams_audio is None:
            raise ValueError("vision_config / audio_config not found in hparams")

        # for compat with vision-only models
        self.hparams = self.hparams_vision or self.hparams_audio or self.hparams

        # TODO @ngxson : this is a hack to support both vision and audio encoders
        have_multiple_encoders = self.has_audio_encoder and self.has_vision_encoder
        self.block_count = 128 if have_multiple_encoders else self.find_hparam(self.n_block_keys, True)
        self.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, self.block_count)

        # load preprocessor config
        self.preprocessor_config = {}

        # prefer preprocessor_config.json if possible
        preprocessor_config_path = self.dir_model / "preprocessor_config.json"
        if preprocessor_config_path.is_file():
            with open(preprocessor_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # move media_proc_cfg to root level for compat
                if "media_proc_cfg" in cfg:
                    cfg = {
                        **cfg,
                        **cfg["media_proc_cfg"],
                    }
                # merge configs
                self.preprocessor_config = {**self.preprocessor_config, **cfg}

        # prefer processor_config.json if possible
        processor_config_path = self.dir_model / "processor_config.json"
        if processor_config_path.is_file():
            with open(processor_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # move image_processor to root level for compat
                if "image_processor" in cfg:
                    cfg = {
                        **cfg,
                        **cfg["image_processor"],
                    }
                # merge configs
                self.preprocessor_config = {**self.preprocessor_config, **cfg}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip non-multimodal tensors
        if "language_model." in name:
            return None

        return super().filter_tensors(item)

    def get_vision_config(self) -> dict[str, Any] | None:
        config_name = "vision_config" if not self.is_mistral_format else "vision_encoder"
        return self.global_config.get(config_name)

    def get_audio_config(self) -> dict[str, Any] | None:
        mm_config_key = "whisper_config" if "whisper_config" in self.hparams else "audio_config"
        return self.global_config.get(mm_config_key)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MMPROJ)

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)

        output_type: str = self.ftype.name.partition("_")[2]

        if self.fname_out.is_dir():
            fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, size_label=None, output_type=output_type, model_type=None)
            self.fname_out = self.fname_out / f"mmproj-{fname_default}.gguf"
        else:
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        if self.has_vision_encoder:
            self.gguf_writer.add_clip_has_vision_encoder(True)
            self.gguf_writer.add_vision_projection_dim(self.n_embd_text)

            # vision config
            self.image_size = self.find_vparam(["image_size"])
            self.gguf_writer.add_vision_image_size(self.image_size)
            self.gguf_writer.add_vision_patch_size(self.find_vparam(["patch_size"]))
            self.gguf_writer.add_vision_embedding_length(self.find_vparam(["hidden_size", "width", "vt_hidden_size"]))
            self.gguf_writer.add_vision_feed_forward_length(self.find_vparam(["intermediate_size", "vt_intermediate_size"]))
            self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys))
            self.gguf_writer.add_vision_head_count(self.find_vparam(["num_attention_heads", "num_heads", "heads", "vt_num_attention_heads"]))

            # preprocessor config
            image_mean = _MISTRAL_COMMON_DATASET_MEAN if self.is_mistral_format else self.preprocessor_config["image_mean"]
            image_std = _MISTRAL_COMMON_DATASET_STD if self.is_mistral_format else self.preprocessor_config["image_std"]

            self.gguf_writer.add_vision_image_mean(image_mean)
            self.gguf_writer.add_vision_image_std(image_std)

        if self.has_audio_encoder:
            self.gguf_writer.add_clip_has_audio_encoder(True)
            self.gguf_writer.add_audio_projection_dim(self.n_embd_text)

            # audio config
            self.gguf_writer.add_audio_embedding_length(self.find_aparam(["hidden_size"]))
            self.gguf_writer.add_audio_feed_forward_length(self.find_aparam(["intermediate_size"]))
            self.gguf_writer.add_audio_block_count(self.find_aparam(self.n_block_keys))
            self.gguf_writer.add_audio_head_count(self.find_aparam(["num_attention_heads"]))

        if not self.has_vision_encoder and not self.has_audio_encoder:
            raise ValueError("MmprojModel must have either vision or audio encoder")

    def write_vocab(self):
        raise ValueError("MmprojModel does not support vocab writing")

    def find_vparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_vision is not None
        return self._find_param(self.hparams_vision, keys, optional)

    def find_aparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_audio is not None
        return self._find_param(self.hparams_audio, keys, optional)

    def _find_param(self, obj: dict[str, Any], keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in obj), None)
        if key is not None:
            return obj[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del bid, name, n_dims  # unused
        if ".patch_embd.weight" in new_name or ".patch_merger.weight" in new_name:
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return False


class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.uint8: np.uint8,
    }

    # only used when byteswapping data. Only correct size is needed
    # TODO: uncomment uint64, uint32, and uint16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_byteswap_map: dict[torch.dtype, type] = {
        torch.float64: np.float64,
        torch.float32: np.float32,
        torch.bfloat16: np.float16,
        torch.float16: np.float16,
        torch.int64: np.int64,
        # torch.uint64: np.uint64,
        torch.int32: np.int32,
        # torch.uint32: np.uint32,
        torch.int16: np.int16,
        # torch.uint16: np.uint16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.uint8,
        torch.float8_e4m3fn: np.uint8,
        torch.float8_e5m2: np.uint8,
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
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[...] if len(s.get_shape()) == 0 else s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_local_tensor(cls, t: gguf.utility.LocalTensor) -> Tensor:
        def load_tensor(tensor: gguf.utility.LocalTensor) -> Tensor:
            def byteswap_tensor(tensor: np.ndarray, dtype: type) -> np.ndarray:
                if sys.byteorder == 'big':
                    # switch data back to big endian
                    tensor = tensor.view(dtype).byteswap(inplace=False)
                return tensor
            dtype = cls._dtype_str_map[tensor.dtype]
            numpy_dtype = cls._dtype_byteswap_map[dtype]
            return torch.from_numpy(byteswap_tensor(tensor.mmap_bytes(), numpy_dtype)).view(dtype).reshape(tensor.shape)
        dtype = cls._dtype_str_map[t.dtype]
        shape = t.shape
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(t,), func=lambda r: load_tensor(r))
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_remote_tensor(cls, remote_tensor: gguf.utility.RemoteTensor):
        def byteswap_tensor(tensor: np.ndarray, dtype: type) -> np.ndarray:
            if sys.byteorder == 'big':
                # switch data back to big endian
                tensor = tensor.view(dtype).byteswap(inplace=False)
            return tensor
        dtype = cls._dtype_str_map[remote_tensor.dtype]
        numpy_dtype = cls._dtype_byteswap_map[dtype]
        shape = remote_tensor.shape
        meta = cls.meta_with_dtype_and_shape(dtype, shape)
        lazy = cls(meta=meta, args=(remote_tensor,), func=lambda r: torch.from_numpy(byteswap_tensor(np.frombuffer(r.data(), dtype=numpy_dtype), numpy_dtype)).view(dtype).reshape(shape))
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            assert len(args)
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


def get_model_architecture(hparams: dict[str, Any], model_type: ModelType) -> str:
    # TODO @ngxson : this won't work correctly if the model has both audio & vision encoders
    # maybe we should fallback to text model's arch in that case, since not many models have both
    text_config = hparams.get("text_config", {})
    vision_config = hparams.get("vision_config", {})
    arch = None
    if (arches := hparams.get("architectures")) is not None and len(arches) > 0:
        arch = arches[0]
    elif "ssm_cfg" in hparams:
        # For non-hf Mamba and Mamba2 models
        arch = hparams["ssm_cfg"].get("layer", "Mamba") + "ForCausalLM"

    # Step3-VL keeps text config under text_config but uses a custom top-level architecture.
    # For text conversion we route to a dedicated text-only class.
    # TODO: refactor this later to avoid adding exception here
    if model_type == ModelType.TEXT and arch in ("StepVLForConditionalGeneration", "Sarashina2VisionForCausalLM"):
        return arch

    # if "architectures" is found in the sub-config, use that instead
    if model_type == ModelType.TEXT and text_config.get("architectures") is not None:
        arch = text_config["architectures"][0]
    elif model_type == ModelType.MMPROJ and vision_config.get("architectures") is not None:
        arch = vision_config["architectures"][0]
    if arch is None:
        raise ValueError("Failed to detect model architecture")
    return arch
