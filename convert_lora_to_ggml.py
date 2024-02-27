#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np
import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))
import gguf

NUMPY_TYPE_TO_FTYPE: dict[str, int] = {"float32": 0, "float16": 1}


def write_file_header(fout: BinaryIO, params: dict[str, Any]) -> None:
    fout.write(b"ggla"[::-1])  # magic (ggml lora)
    fout.write(struct.pack("i", 1))  # file version
    fout.write(struct.pack("i", params["r"]))
    # https://opendelta.readthedocs.io/en/latest/modules/deltas.html says that `lora_alpha` is an int
    # but some models ship a float value instead
    # let's convert to int, but fail if lossless conversion is not possible
    assert (
        int(params["lora_alpha"]) == params["lora_alpha"]
    ), "cannot convert float to int losslessly"
    fout.write(struct.pack("i", int(params["lora_alpha"])))


def write_tensor_header(fout: BinaryIO, name: str, shape: Sequence[int], data_type: np.dtype[Any]) -> None:
    sname = name.encode("utf-8")
    fout.write(
        struct.pack(
            "iii",
            len(shape),
            len(sname),
            NUMPY_TYPE_TO_FTYPE[data_type.name],
        )
    )
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek((fout.tell() + 31) & -32)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path> [arch]")
        print(
            "Path must contain HuggingFace PEFT LoRA files 'adapter_config.json' and 'adapter_model.bin'"
        )
        print(f"Arch must be one of {list(gguf.MODEL_ARCH_NAMES.values())} (default: llama)")
        sys.exit(1)

    input_json = os.path.join(sys.argv[1], "adapter_config.json")
    input_model = os.path.join(sys.argv[1], "adapter_model.bin")
    output_path = os.path.join(sys.argv[1], "ggml-adapter-model.bin")

    if os.path.exists(input_model):
        model = torch.load(input_model, map_location="cpu")
    else:
        input_model = os.path.join(sys.argv[1], "adapter_model.safetensors")
        # lazy import load_file only if lora is in safetensors format.
        from safetensors.torch import load_file
        model = load_file(input_model, device="cpu")

    arch_name = sys.argv[2] if len(sys.argv) == 3 else "llama"

    if arch_name not in gguf.MODEL_ARCH_NAMES.values():
        print(f"Error: unsupported architecture {arch_name}")
        sys.exit(1)

    arch = list(gguf.MODEL_ARCH_NAMES.keys())[list(gguf.MODEL_ARCH_NAMES.values()).index(arch_name)]
    name_map = gguf.TensorNameMap(arch, 200) # 200 layers ought to be enough for anyone

    with open(input_json, "r") as f:
        params = json.load(f)

    if params["peft_type"] != "LORA":
        print(f"Error: unsupported adapter type {params['peft_type']}, expected LORA")
        sys.exit(1)

    if params["fan_in_fan_out"] is True:
        print("Error: param fan_in_fan_out is not supported")
        sys.exit(1)

    if params["bias"] is not None and params["bias"] != "none":
        print("Error: param bias is not supported")
        sys.exit(1)

    # TODO: these seem to be layers that have been trained but without lora.
    # doesn't seem widely used but eventually should be supported
    if params["modules_to_save"] is not None and len(params["modules_to_save"]) > 0:
        print("Error: param modules_to_save is not supported")
        sys.exit(1)

    with open(output_path, "wb") as fout:
        fout.truncate()

        write_file_header(fout, params)
        for k, v in model.items():
            orig_k = k
            if k.endswith(".default.weight"):
                k = k.replace(".default.weight", ".weight")
            if k in ["llama_proj.weight", "llama_proj.bias"]:
                continue
            if k.endswith("lora_A.weight"):
                if v.dtype != torch.float16 and v.dtype != torch.float32:
                    v = v.float()
                v = v.T
            else:
                v = v.float()

            t = v.detach().numpy()

            prefix = "base_model.model."
            if k.startswith(prefix):
                k = k[len(prefix) :]

            lora_suffixes = (".lora_A.weight", ".lora_B.weight")
            if k.endswith(lora_suffixes):
                suffix = k[-len(lora_suffixes[0]):]
                k = k[: -len(lora_suffixes[0])]
            else:
                print(f"Error: unrecognized tensor name {orig_k}")
                sys.exit(1)

            tname = name_map.get_name(k)
            if tname is None:
                print(f"Error: could not map tensor name {orig_k}")
                print(" Note: the arch parameter must be specified if the model is not llama")
                sys.exit(1)

            if suffix == ".lora_A.weight":
                tname += ".weight.loraA"
            elif suffix == ".lora_B.weight":
                tname += ".weight.loraB"
            else:
                assert False

            print(f"{k} => {tname} {t.shape} {t.dtype} {t.nbytes/1024/1024:.2f}MB")
            write_tensor_header(fout, tname, t.shape, t.dtype)
            t.tofile(fout)

    print(f"Converted {input_json} and {input_model} to {output_path}")

