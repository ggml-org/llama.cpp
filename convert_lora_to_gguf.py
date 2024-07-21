#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import logging
import argparse
import os
import sys
import json
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence, SupportsIndex, cast

import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

# reuse model definitions from convert_hf_to_gguf.py
from convert_hf_to_gguf import LazyTorchTensor, Model

logger = logging.getLogger("lora-to-gguf")


@dataclass
class PartialLoraTensor:
    A: Tensor | None = None
    B: Tensor | None = None


# magic to support tensor shape modifications and splitting
class LoraTorchTensor:
    _lora_A: Tensor  # (n_rank, row_size)
    _lora_B: Tensor  # (col_size, n_rank)
    _rank: int

    def __init__(self, A: Tensor, B: Tensor):
        assert len(A.shape) == len(B.shape)
        assert A.shape[-2] == B.shape[-1]
        if A.dtype != B.dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        self._lora_A = A
        self._lora_B = B
        self._rank = B.shape[-1]

    def get_lora_A_B(self) -> tuple[Tensor, Tensor]:
        return (self._lora_A, self._lora_B)

    def __getitem__(
        self,
        indices: (
            SupportsIndex
            | slice
            | tuple[SupportsIndex | slice | Tensor, ...]  # TODO: add ellipsis in the type signature
        ),
    ) -> LoraTorchTensor:
        shape = self.shape
        if isinstance(indices, SupportsIndex):
            if len(shape) > 2:
                return LoraTorchTensor(self._lora_A[indices], self._lora_B[indices])
            else:
                raise NotImplementedError  # can't return a vector
        elif isinstance(indices, slice):
            if len(shape) > 2:
                return LoraTorchTensor(self._lora_A[indices], self._lora_B[indices])
            else:
                return LoraTorchTensor(self._lora_A, self._lora_B[indices])
        elif isinstance(indices, tuple):
            assert len(indices) > 0
            if indices[-1] is Ellipsis:
                return self[indices[:-1]]
            # expand ellipsis
            indices = tuple(
                u
                for v in (
                    (
                        (slice(None, None) for _ in range(len(indices) - 1))
                        if i is Ellipsis
                        else (i,)
                    )
                    for i in indices
                )
                for u in v
            )

            if len(indices) < len(shape):
                indices = (*indices, *(slice(None, None) for _ in range(len(indices), len(shape))))

            # TODO: make sure this is correct
            indices_A = (
                *(
                    (
                        j.__index__() % self._lora_A.shape[i]
                        if isinstance(j, SupportsIndex)
                        else slice(None, None)
                    )
                    for i, j in enumerate(indices[:-2])
                ),
                slice(None, None),
                indices[-1],
            )
            indices_B = indices[:-1]
            return LoraTorchTensor(self._lora_A[indices_A], self._lora_B[indices_B])
        else:
            raise NotImplementedError  # unknown indice type

    @property
    def dtype(self) -> torch.dtype:
        assert self._lora_A.dtype == self._lora_B.dtype
        return self._lora_A.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        assert len(self._lora_A.shape) == len(self._lora_B.shape)
        return (*self._lora_B.shape[:-1], self._lora_A.shape[-1])

    def size(self, dim=None):
        assert dim is None
        return self.shape

    def reshape(self, *shape: int | tuple[int, ...]) -> LoraTorchTensor:
        if isinstance(shape[0], tuple):
            new_shape: tuple[int, ...] = shape[0]
        else:
            new_shape = cast(tuple[int, ...], shape)
        orig_shape = self.shape
        if len(new_shape) < 2:
            raise NotImplementedError  # can't become a vector

        # expand -1 in the shape
        if any(dim == -1 for dim in new_shape):
            n_elems = prod(orig_shape)
            n_new_elems = prod(dim if dim != -1 else 1 for dim in new_shape)
            assert n_elems % n_new_elems == 0
            new_shape = (*(dim if dim != -1 else n_elems // n_new_elems for dim in new_shape),)

        if new_shape[-1] != orig_shape[-1]:
            raise NotImplementedError  # can't reshape the row size trivially

        shape_A = (*(1 for _ in new_shape[:-2]), self._rank, orig_shape[-1])
        shape_B = (*new_shape[:-1], self._rank)
        return LoraTorchTensor(
            self._lora_A.reshape(shape_A),
            self._lora_B.reshape(shape_B),
        )

    def reshape_as(self, other: Tensor) -> LoraTorchTensor:
        return self.reshape(*other.shape)

    def view(self, *size: int) -> LoraTorchTensor:
        return self.reshape(*size)

    def permute(self, *dims: int) -> LoraTorchTensor:
        shape = self.shape
        dims = tuple(dim - len(shape) if dim >= 0 else dim for dim in dims)
        if dims[-1] == -1:
            # TODO: support higher dimensional A shapes bigger than 1
            assert all(dim == 1 for dim in self._lora_A.shape[:-2])
            return LoraTorchTensor(self._lora_A, self._lora_B.permute(*dims))
        if len(shape) == 2 and dims[-1] == -2 and dims[-2] == -1:
            return LoraTorchTensor(self._lora_B.permute(*dims), self._lora_A.permute(*dims))
        else:
            # TODO: compose the above two
            raise NotImplementedError

    def transpose(self, dim0: int, dim1: int) -> LoraTorchTensor:
        shape = self.shape
        dims = [i for i in range(len(shape))]
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)

    def swapaxes(self, axis0: int, axis1: int) -> LoraTorchTensor:
        return self.transpose(axis0, axis1)

    def to(self, *args, **kwargs):
        return LoraTorchTensor(self._lora_A.to(*args, **kwargs), self._lora_B.to(*args, **kwargs))

    @classmethod
    def __torch_function__(cls, func: Callable, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.permute:
            return type(args[0]).permute(*args, **kwargs)
        elif func is torch.reshape:
            return type(args[0]).reshape(*args, **kwargs)
        elif func is torch.stack:
            assert isinstance(args[0], Sequence)
            dim = kwargs.get("dim", 0)
            assert dim == 0
            return LoraTorchTensor(
                torch.stack([a._lora_A for a in args[0]], dim),
                torch.stack([b._lora_B for b in args[0]], dim),
            )
        elif func is torch.cat:
            assert isinstance(args[0], Sequence)
            dim = kwargs.get("dim", 0)
            assert dim == 0
            if len(args[0][0].shape) > 2:
                return LoraTorchTensor(
                    torch.cat([a._lora_A for a in args[0]], dim),
                    torch.cat([b._lora_B for b in args[0]], dim),
                )
            elif all(torch.equal(args[0][0]._lora_A, t._lora_A) for t in args[0][1:]):
                return LoraTorchTensor(
                    args[0][0]._lora_A,
                    torch.cat([b._lora_B for b in args[0]], dim),
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


def get_base_tensor_name(lora_tensor_name: str) -> str:
    base_name = lora_tensor_name.replace("base_model.model.", "")
    base_name = base_name.replace(".lora_A.weight", ".weight")
    base_name = base_name.replace(".lora_B.weight", ".weight")
    return base_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface PEFT LoRA adapter to a GGML compatible file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "auto"], default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out what will be done, without writing any new files",
    )
    parser.add_argument(
        "--base", type=Path, required=True,
        help="directory containing base model file",
    )
    parser.add_argument(
        "lora_path", type=Path,
        help="directory containing LoRA adapter file",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    ftype = ftype_map[args.outtype]

    dir_base_model: Path = args.base
    dir_lora: Path = args.lora_path
    lora_config = dir_lora / "adapter_config.json"
    input_model = dir_lora / "adapter_model.safetensors"

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_lora

    if os.path.exists(input_model):
        # lazy import load_file only if lora is in safetensors format.
        from safetensors.torch import load_file

        lora_model = load_file(input_model, device="cpu")
    else:
        input_model = os.path.join(dir_lora, "adapter_model.bin")
        lora_model = torch.load(input_model, map_location="cpu", weights_only=True)

    # load base model
    logger.info(f"Loading base model: {dir_base_model.name}")
    hparams = Model.load_hparams(dir_base_model)
    with torch.inference_mode():
        try:
            model_class = Model.from_model_architecture(hparams["architectures"][0])
        except NotImplementedError:
            logger.error(f"Model {hparams['architectures'][0]} is not supported")
            sys.exit(1)

        class LoraModel(model_class):
            model_arch = model_class.model_arch

            lora_alpha: float

            def __init__(self, *args, dir_lora_model: Path, lora_alpha: float, **kwargs):

                super().__init__(*args, **kwargs)

                self.dir_model_card = dir_lora_model
                self.lora_alpha = float(lora_alpha)

            def set_type(self):
                self.gguf_writer.add_type(gguf.GGUFType.ADAPTER)
                self.gguf_writer.add_string(gguf.Keys.Adapter.TYPE, "lora")

            def set_gguf_parameters(self):
                self.gguf_writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, self.lora_alpha)
                super().set_gguf_parameters()

            def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
                tensor_map: dict[str, PartialLoraTensor] = {}

                for name, tensor in lora_model.items():
                    if self.lazy:
                        tensor = LazyTorchTensor.from_eager(tensor)
                    base_name = get_base_tensor_name(name)
                    is_lora_a = ".lora_A.weight" in name
                    is_lora_b = ".lora_B.weight" in name
                    if not is_lora_a and not is_lora_b:
                        if ".base_layer.weight" in name:
                            continue
                        logger.error(f"Unexpected name '{name}': Not a lora_A or lora_B tensor")
                        sys.exit(1)

                    if base_name in tensor_map:
                        if is_lora_a:
                            tensor_map[base_name].A = tensor
                        else:
                            tensor_map[base_name].B = tensor
                    else:
                        if is_lora_a:
                            tensor_map[base_name] = PartialLoraTensor(A=tensor)
                        else:
                            tensor_map[base_name] = PartialLoraTensor(B=tensor)

                for name, tensor in tensor_map.items():
                    assert tensor.A is not None
                    assert tensor.B is not None
                    yield (name, cast(torch.Tensor, LoraTorchTensor(tensor.A, tensor.B)))

            def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
                dest = super().modify_tensors(data_torch, name, bid)
                for dest_name, dest_data in dest:
                    assert isinstance(dest_data, LoraTorchTensor)
                    lora_a, lora_b = dest_data.get_lora_A_B()

                    yield (dest_name + ".lora_a", lora_a)
                    yield (dest_name + ".lora_b", lora_b)

        with open(lora_config, "r") as f:
            lparams: dict[str, Any] = json.load(f)

        alpha: float = lparams["lora_alpha"]

        model_instance = LoraModel(
            dir_base_model,
            ftype,
            fname_out,
            is_big_endian=args.bigendian,
            use_temp_file=False,
            eager=args.no_lazy,
            dry_run=args.dry_run,
            dir_lora_model=dir_lora,
            lora_alpha=alpha,
        )

        logger.info("Exporting model...")
        model_instance.write()
        logger.info(f"Model successfully exported to {model_instance.fname_out}")
