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


# byte offset and type of the multi-byte fields in each block, all other fields are single bytes
# use the width ggml reads the field with, e.g. Q5_0 qh is uint8_t[4] but read as uint32_t
byteswap_tensors: dict[gguf.GGMLQuantizationType, tuple[tuple[int, type[np.unsignedinteger]], ...]] = {
    gguf.GGMLQuantizationType.Q1_0:   ((0,   np.uint16),),                                 # d
    gguf.GGMLQuantizationType.Q2_0:   ((0,   np.uint16),),                                 # d
    gguf.GGMLQuantizationType.Q4_0:   ((0,   np.uint16),),                                 # d
    gguf.GGMLQuantizationType.Q4_1:   ((0,   np.uint16), (2, np.uint16)),                  # d, m
    gguf.GGMLQuantizationType.Q5_0:   ((0,   np.uint16), (2, np.uint32)),                  # d, qh
    gguf.GGMLQuantizationType.Q5_1:   ((0,   np.uint16), (2, np.uint16), (4, np.uint32)),  # d, m, qh
    gguf.GGMLQuantizationType.Q8_0:   ((0,   np.uint16),),                                 # d
    gguf.GGMLQuantizationType.Q2_K:   ((80,  np.uint16), (82, np.uint16)),                 # d, dmin
    gguf.GGMLQuantizationType.Q3_K:   ((108, np.uint16),),                                 # d
    gguf.GGMLQuantizationType.Q4_K:   ((0,   np.uint16), (2, np.uint16)),                  # d, dmin
    gguf.GGMLQuantizationType.Q5_K:   ((0,   np.uint16), (2, np.uint16)),                  # d, dmin
    gguf.GGMLQuantizationType.Q6_K:   ((208, np.uint16),),                                 # d
    gguf.GGMLQuantizationType.IQ4_NL: ((0,   np.uint16),),                                 # d
    gguf.GGMLQuantizationType.IQ4_XS: ((0,   np.uint16), (2, np.uint16)),                  # d, scales_h
    gguf.GGMLQuantizationType.TQ1_0:  ((52,  np.uint16),),                                 # d
    gguf.GGMLQuantizationType.TQ2_0:  ((64,  np.uint16),),                                 # d
    gguf.GGMLQuantizationType.MXFP4:  (),                                                  # noop
    gguf.GGMLQuantizationType.NVFP4:  (),                                                  # noop
    gguf.GGMLQuantizationType.F8_E4M3: (),                                                # noop
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
            # one row per block, then swap each field of all blocks at once
            type_size = gguf.constants.GGML_QUANT_SIZES[tensor.tensor_type][1]
            blocks = tensor.data.reshape(-1, type_size, copy=False)
            for offs, dtype in byteswap_tensors[tensor.tensor_type]:
                size = np.dtype(dtype).itemsize
                blocks[:, offs:offs + size].view(dtype=dtype)[:, 0].byteswap(inplace=True)
        elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            # Special case for BF16
            # It is 2-bytes data, but by default view loads it as 1-byte data.
            # Change to correct view before byteswapping.
            tensor.data.view(dtype=np.uint16).byteswap(inplace=True)
        else:
            # Handle other tensor types
            tensor.data.byteswap(inplace=True)

        pbar.set_description(log_message)

    reader.data.flush()
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
