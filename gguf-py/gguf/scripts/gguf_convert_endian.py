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


def convert_byteorder(reader: gguf.GGUFReader, args: argparse.Namespace) -> None:
    if np.uint32(1) == np.uint32(1).newbyteorder("<"):
        # Host is little endian
        host_endian = "little"
        swapped_endian = "big"
    else:
        # Sorry PDP or other weird systems that don't use BE or LE.
        host_endian = "big"
        swapped_endian = "little"
    if reader.byte_order == "S":
        file_endian = swapped_endian
    else:
        file_endian = host_endian
    order = host_endian if args.order == "native" else args.order
    logger.info(f"* Host is {host_endian.upper()} endian, GGUF file seems to be {file_endian.upper()} endian")
    if file_endian == order:
        logger.info(f"* File is already {order.upper()} endian. Nothing to do.")
        sys.exit(0)
    logger.info("* Checking tensors for conversion compatibility")
    for tensor in reader.tensors:
        if tensor.tensor_type not in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.Q8_0,
        ):
            raise ValueError(f"Cannot handle type {tensor.tensor_type.name} for tensor {repr(tensor.name)}")
    logger.info(f"* Preparing to convert from {file_endian.upper()} to {order.upper()}")
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
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            # Handle Q8_0 tensor blocks (block_q8_0)
            # Specific handling of block_q8_0 is required.
            # Each block_q8_0 consists of an f16 delta (scaling factor) followed by 32 int8 quantizations.

            block_size = 34 # 34 bytes = <f16 delta scaling factor> + 32 * <int8 quant>

            n_blocks = len(tensor.data) // block_size
            for block_num in (inner_pbar := tqdm(range(n_blocks), desc="Byte-swapping Blocks", leave=False)):
                block_offs = block_num * block_size

                # Byte-Swap f16 sized delta field
                delta = tensor.data[block_offs:block_offs + 2].view(dtype=np.uint16)
                delta.byteswap(inplace=True)

                # Byte-Swap Q8 weights
                if block_num % 100000 == 0:
                    inner_pbar.set_description(f"Byte-swapping Blocks [{(n_blocks - block_num) // n_blocks}]")

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
