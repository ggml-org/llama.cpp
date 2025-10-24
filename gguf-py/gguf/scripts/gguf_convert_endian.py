#!/usr/bin/env python3
from __future__ import annotations

import logging
import argparse
import os
import sys
from tqdm import tqdm
from pathlib import Path

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-convert-endian")


def byteswap_q4_0(tensor, block_offs):
    # Each block_q4_0 consists of an f16 delta (scaling factor) followed by 16 int8 quantizations.

    # Byte-Swap f16 sized delta field
    delta = tensor.data[block_offs:block_offs + 2].view(dtype=np.uint16)
    delta.byteswap(inplace=True)


def byteswap_q8_0(tensor, block_offs):
    # Each block_q8_0 consists of an f16 delta (scaling factor) followed by 32 int8 quantizations.

    # Byte-Swap f16 sized delta field
    delta = tensor.data[block_offs:block_offs + 2].view(dtype=np.uint16)
    delta.byteswap(inplace=True)


def byteswap_q4_k(tensor, block_offs):
    # Each block_q4_k consists of 2 f16 values followed by 140 int8 values.

    # Byte-Swap f16 sized fields
    delta = tensor.data[block_offs:block_offs + 2].view(dtype=np.uint16)
    delta.byteswap(inplace=True)

    delta = tensor.data[block_offs + 2:block_offs + 4].view(dtype=np.uint16)
    delta.byteswap(inplace=True)


def byteswap_q6_k(tensor, block_offs):
    # Each block_q6_k consists of 208 int8 values followed by 1 f16 value.

    # Byte-Swap f16 sized field
    delta = tensor.data[block_offs + 208:block_offs + 210].view(dtype=np.uint16)
    delta.byteswap(inplace=True)


byteswap_tensors = {
    gguf.GGMLQuantizationType.Q4_0: {
        "block_size": 18, # 18 bytes = <f16 delta scaling factor> + 16 * <int8 quant>
        "byteswap_func": byteswap_q4_0,
    },
    gguf.GGMLQuantizationType.Q8_0: {
        "block_size": 34, # 34 bytes = <f16 delta scaling factor> + 32 * <int8 quant>
        "byteswap_func": byteswap_q8_0,
    },
    gguf.GGMLQuantizationType.Q4_K: {
        "block_size": 144, # 144 bytes = 2 * <f16 delta scaling factor> + 140 * <int8 quant>
        "byteswap_func": byteswap_q4_k,
    },
    gguf.GGMLQuantizationType.Q6_K: {
        "block_size": 210, # 210 bytes = <f16 delta scaling factor> + 208 * <int8 quant>
        "byteswap_func": byteswap_q6_k,
    },
}


def convert_byteorder(reader: gguf.GGUFReader, args: argparse.Namespace) -> None:
    file_endian = reader.endianess.name
    if reader.byte_order == 'S':
        host_endian = 'BIG' if file_endian == 'LITTLE' else 'LITTLE'
    else:
        host_endian = file_endian
    order = host_endian if args.order == "native" else args.order.upper()
    logger.info(f"* Host is {host_endian} endian, GGUF file seems to be {file_endian} endian")
    if file_endian == order:
        logger.info(f"* File is already {order} endian. Nothing to do.")
        sys.exit(0)
    logger.info("* Checking tensors for conversion compatibility")
    for tensor in reader.tensors:
        if tensor.tensor_type not in byteswap_tensors and \
           tensor.tensor_type not in (
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
           ):
            raise ValueError(f"Cannot handle type {tensor.tensor_type.name} for tensor {repr(tensor.name)}")
    logger.info(f"* Preparing to convert from {file_endian} to {order}")
    if args.dry_run:
        return
    logger.warning("*** Warning *** Warning *** Warning **")
    logger.warning("* This conversion process may damage the file. Ensure you have a backup.")
    if order != host_endian:
        logger.warning("* Requested endian differs from host, you will not be able to load the model on this machine.")
    logger.warning("* The file will be modified immediately, so if conversion fails or is interrupted")
    logger.warning("* the file will be corrupted. Enter exactly YES if you are positive you want to proceed:")
    response = input("YES, I am sure> ")
    if response != "YES":
        logger.warning("You didn't enter YES. Okay then, see ya!")
        sys.exit(0)
    logger.info(f"* Converting fields ({len(reader.fields)})")
    for idx, field in enumerate(reader.fields.values()):
        logger.info(f"- {idx:4}: Converting field {repr(field.name)}, part count: {len(field.parts)}")
        for part in field.parts:
            part.byteswap(inplace=True)
    logger.info(f"* Converting tensors ({len(reader.tensors)})")

    for idx, tensor in enumerate(pbar := tqdm(reader.tensors, desc="Converting tensor")):
        log_message = (
            f"Converting tensor {repr(tensor.name)}, "
            f"type={tensor.tensor_type.name}, "
            f"elements={tensor.n_elements} "
        )

        # Byte-swap each part of the tensor's field
        for part in tensor.field.parts:
            part.byteswap(inplace=True)

        # Byte-swap tensor data if necessary
        if tensor.tensor_type in byteswap_tensors:
            # first flatten structure
            oldshape = tensor.data.shape
            newshape = 1
            for i in tensor.data.shape:
                newshape *= i

            tensor.data.resize(newshape)

            block_size    = byteswap_tensors[tensor.tensor_type]["block_size"]
            byteswap_func = byteswap_tensors[tensor.tensor_type]["byteswap_func"]

            n_blocks = len(tensor.data) // block_size
            for block_num in (inner_pbar := tqdm(range(n_blocks), desc="Byte-swapping Blocks", leave=False)):
                block_offs = block_num * block_size

                byteswap_func(tensor, block_offs)

                if block_num % 100000 == 0:
                    inner_pbar.set_description(f"Byte-swapping Blocks [{(n_blocks - block_num) // n_blocks}]")

            # restore old shape in case it's ever used
            tensor.data.resize(oldshape)
        elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            # Special case for BF16
            # It is 2-bytes data, but by default view loads it as 1-byte data.
            # Change to correct view before byteswapping.
            tensor.data.view(dtype=np.uint16).byteswap(inplace=True)
        else:
            # Handle other tensor types
            tensor.data.byteswap(inplace=True)

        pbar.set_description(log_message)

    logger.info("* Completion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GGUF file byte order")
    parser.add_argument(
        "model", type=str,
        help="GGUF format model filename",
    )
    parser.add_argument(
        "order", type=str, choices=['big', 'little', 'native'],
        help="Requested byte order",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't actually change anything",
    )
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info(f'* Loading: {args.model}')
    reader = gguf.GGUFReader(args.model, 'r' if args.dry_run else 'r+')
    convert_byteorder(reader, args)


if __name__ == "__main__":
    main()
