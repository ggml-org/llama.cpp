#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import ast
import logging
import contextlib
import json
import os
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


# Import QwenModel for its static methods (used in _set_vocab_qwen)
# This is a lazy import to avoid circular dependencies
def _get_qwen_model():
    from .qwen import QwenModel
    return QwenModel


try:
    from mistral_common.tokens.tokenizers.base import TokenizerVersion # pyright: ignore[reportUnusedImport]
    from mistral_common.tokens.tokenizers.multimodal import DATASET_MEAN as _MISTRAL_COMMON_DATASET_MEAN, DATASET_STD as _MISTRAL_COMMON_DATASET_STD
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer # pyright: ignore[reportUnusedImport]
    from mistral_common.tokens.tokenizers.sentencepiece import ( 
        SentencePieceTokenizer, # pyright: ignore[reportUnusedImport]
    )
    _mistral_common_installed = True
    _mistral_import_error_msg = ""
except ImportError:
    _MISTRAL_COMMON_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _MISTRAL_COMMON_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    _mistral_common_installed = False
    TokenizerVersion = None
    Tekkenizer = None
    SentencePieceTokenizer = None
    _mistral_import_error_msg = (
        "Mistral format requires `mistral-common` to be installed. Please run "
        "`pip install mistral-common[image,audio]` to install it."
    )
logger = logging.getLogger("hf-to-gguf")


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


AnyModel = TypeVar("AnyModel", bound="type[ModelBase]")


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

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, *, is_big_endian: bool = False,
                 use_temp_file: bool = False, eager: bool = False,
                 metadata_override: Path | None = None, model_name: str | None = None,
                 split_max_tensors: int = 0, split_max_size: int = 0, dry_run: bool = False,
                 small_first_shard: bool = False, hparams: dict[str, Any] | None = None, remote_hf_model_id: str | None = None,
                 disable_mistral_community_chat_template: bool = False,
                 sentence_transformers_dense_modules: bool = False):
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
        self.hparams = ModelBase.load_hparams(self.dir_model, self.is_mistral_format) if hparams is None else hparams
        self.model_tensors = self.index_tensors(remote_hf_model_id=remote_hf_model_id)
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py
        # Apply heuristics to figure out typical tensor encoding based on first layer tensor encoding type
        if self.ftype == gguf.LlamaFileType.GUESSED:
            # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16
        self.dequant_model()
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
                tensors[name] = lambda r=remote_tensor: LazyTorchTensor.from_remote_tensor(r)
            return tensors
        prefix = "model" if not self.is_mistral_format else "consolidated"
        part_names: list[str] = ModelBase.get_model_part_names(self.dir_model, prefix, ".safetensors")
        is_safetensors: bool = len(part_names) > 0
        if not is_safetensors:
            part_names = ModelBase.get_model_part_names(self.dir_model, "pytorch_model", ".bin")
        tensor_names_from_index: set[str] = set()
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
            else:
                weight_map = {}
        else:
            weight_map = {}
        for part_name in part_names:
            logger.info(f"gguf: indexing model part '{part_name}'")
            ctx: ContextManager[Any]
            if is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))
            with ctx as model_part:
                assert model_part is not None
                for name in model_part.keys():
                    if is_safetensors:
                        if self.lazy:
                            data = model_part.get_slice(name)
                            data_gen = lambda data=data: LazyTorchTensor.from_safetensors_slice(data)  # noqa: E731
                        else:
                            data = model_part.get_tensor(name)
                            data_gen = lambda data=data: data  # noqa: E731
                    else:
                        data = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data: LazyTorchTensor.from_eager(data)  # noqa: E731
                        else:
                            data_gen = lambda data=data: data  # noqa: E731
                    tensors[name] = data_gen
        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_index) > 0:
            tensor_names_from_parts = set(tensors.keys())
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

    def dequant_model(self):
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

            def dequant_simple(weight: Tensor, scale: Tensor) -> Tensor:
                scale = scale.float()
                if (weight_block_size := quant_config.get("weight_block_size")):
                    # TODO: make sure it's a list of integers
                    for i, size in enumerate(weight_block_size):
                        scale = scale.repeat_interleave(size, i)
                # unpad the scale (e.g. when the tensor size isn't a multiple of the block size)
                scale = scale[tuple(slice(0, size) for size in weight.shape)]
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
            if quant_method == "bitnet":
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale"):
                        weight_name = name.removesuffix("_scale")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s: dequant_bitnet(w(), s())
                        tensors_to_remove.append(name)
            elif quant_method == "fp8":
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale_inv"):
                        weight_name = name.removesuffix("_scale_inv")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s: dequant_simple(w(), s())
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
            else:
                raise NotImplementedError(f"Quant method is not yet supported: {quant_method!r}")
        for name in tensors_to_remove:
            if name in self.model_tensors:
                del self.model_tensors[name]
        for name, value in new_tensors.items():
            self.model_tensors[name] = value

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
        del bid  # unused
        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid, n_dims  # unused
        return False

    # some models need extra generated tensors (like rope_freqs)
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")
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
                        )
                    )
                    or not new_name.endswith(".weight")
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
        if "thinker_config" in config:
            # rename for Qwen2.5-Omni
            config["text_config"] = config["thinker_config"]["text_config"]
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

    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

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
        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx", "n_positions", "max_length"], optional=True)) is not None:
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
        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.find_hparam(["rms_norm_eps", "norm_eps"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")
        if (n_expert_groups := self.hparams.get("n_group")) is not None:
            self.gguf_writer.add_expert_group_count(n_expert_groups)
            logger.info(f"gguf: expert groups count = {n_expert_groups}")
        if (n_group_used := self.hparams.get("topk_group")) is not None:
            self.gguf_writer.add_expert_group_used_count(n_group_used)
            logger.info(f"gguf: expert groups used count = {n_group_used}")
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
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size
        tokpre = self.get_vocab_base_pre(tokenizer)
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder
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
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
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
        if chkhsh == "66b8d4e19ab16c3bfd89bce5d785fb7e0155e8648708a1f42077cb9fe002c273":
            # ref: https://huggingface.co/alvarobartt/grok-2-tokenizer
            res = "grok-2"
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
        if chkhsh == "9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206":
            # ref: https://huggingface.co/inclusionAI/Ling-mini-base-2.0
            res = "bailingmoe2"
        if chkhsh == "53e325976a6e142379c19b09afcae354f2f496f147afa8f9e189a33fe4e3024e":
            # ref: https://huggingface.co/ibm-granite/granite-docling-258M
            res = "granite-docling"
        if chkhsh == "f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95":
            # ref: https://huggingface.co/MiniMaxAI/MiniMax-M2
            res = "minimax-m2"
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
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        assert max(tokenizer.get_vocab().values()) < vocab_size
        tokpre = self.get_vocab_base_pre(tokenizer)
        QwenModel = _get_qwen_model()
        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))
        # for this kind of tokenizer, added_vocab is not a subset of vocab, so they need to be combined
        added_vocab = tokenizer.special_tokens
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
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])
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
            template_path = Path(__file__).parent / "models" / "templates" / "llama-cpp-rwkv-world.jinja"
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
                if mod["type"] == "sentence_transformers.models.Pooling":
                    pooling_path = mod["path"]
                    break
        # get pooling type
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            elif pooling["pooling_mode_lasttoken"]:
                pooling_type = gguf.PoolingType.LAST
            else:
                raise NotImplementedError("Only MEAN, CLS, and LAST pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)

    def _set_vocab_interns1(self):
        tokens: list[str] = []
        toktypes: list[int] = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        vocab = getattr(tokenizer, 'vocab', tokenizer.get_vocab())
        vocab_size = self.hparams.get("vocab_size", len(vocab))
        assert max(vocab.values()) < vocab_size
        tokpre = self.get_vocab_base_pre(tokenizer)
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder
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
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
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


class MmprojModel(ModelBase):
    model_type = ModelType.MMPROJ
    model_arch = gguf.MODEL_ARCH.MMPROJ
    preprocessor_config: dict[str, Any]
    global_config: dict[str, Any]
    n_block_keys = ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth"]
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
            self.n_embd_text = text_config.get("hidden_dim", 0)
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
        if not self.is_mistral_format:
            with open(self.dir_model / "preprocessor_config.json", "r", encoding="utf-8") as f:
                self.preprocessor_config = json.load(f)

    def get_vision_config(self) -> dict[str, Any] | None:
        config_name = "vision_config" if not self.is_mistral_format else "vision_encoder"
        return self.global_config.get(config_name)

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("audio_config")

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
            self.gguf_writer.add_vision_embedding_length(self.find_vparam(["hidden_size"]))
            self.gguf_writer.add_vision_feed_forward_length(self.find_vparam(["intermediate_size"]))
            self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys))
            self.gguf_writer.add_vision_head_count(self.find_vparam(["num_attention_heads", "num_heads"]))
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
        if ".patch_embd.weight" in new_name:
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
    def from_remote_tensor(cls, remote_tensor: gguf.utility.RemoteTensor):
        dtype = cls._dtype_str_map[remote_tensor.dtype]
        shape = remote_tensor.shape
        meta = cls.meta_with_dtype_and_shape(dtype, shape)
        lazy = cls(meta=meta, args=(remote_tensor,), func=lambda r: torch.frombuffer(r.data(), dtype=dtype).reshape(shape))
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused
        if kwargs is None:
            kwargs = {}
        if func is torch.Tensor.numpy:
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
    # if "architectures" is found in the sub-config, use that instead
    if model_type == ModelType.TEXT and text_config.get("architectures") is not None:
        arch = text_config["architectures"][0]
    elif model_type == ModelType.MMPROJ and vision_config.get("architectures") is not None:
        arch = vision_config["architectures"][0]
    if arch is None:
        raise ValueError("Failed to detect model architecture")
    return arch


# Export these for use in the main script
__all__ = [
    'ModelBase', 'TextModel', 'MmprojModel', 'ModelType', 'SentencePieceTokenTypes',
    'get_model_architecture', 'LazyTorchTensor', 'logger',
    '_mistral_common_installed', '_mistral_import_error_msg', 'gguf', 'torch'
]
