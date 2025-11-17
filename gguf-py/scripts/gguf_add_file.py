#!/usr/bin/env python3
# gguf_add_file.py srcfile dstfile addfiles ...

from __future__ import annotations

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader, GGUFWriter, ReaderField, GGMLQuantizationType, GGUFEndian, GGUFValueType, Keys  # noqa: E402

logger = logging.getLogger("gguf_add_file")


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    host_endian = 'LITTLE' if np.uint32(1) == np.uint32(1).newbyteorder("<") else 'BIG'
    if reader.byte_order == 'S':
        file_endian = 'BIG' if host_endian == 'LITTLE' else 'LITTLE'
    else:
        file_endian = host_endian
    return (host_endian, file_endian)


def get_byteorder(reader: GGUFReader) -> GGUFEndian:
    if np.uint32(1) == np.uint32(1).newbyteorder("<"):
        # Host is little endian
        host_endian = GGUFEndian.LITTLE
        swapped_endian = GGUFEndian.BIG
    else:
        # Sorry PDP or other weird systems that don't use BE or LE.
        host_endian = GGUFEndian.BIG
        swapped_endian = GGUFEndian.LITTLE

    if reader.byte_order == "S":
        return swapped_endian
    else:
        return host_endian


def decode_field(field: ReaderField) -> Any:
    if field and field.types:
        main_type = field.types[0]

        if main_type == GGUFValueType.ARRAY:
            sub_type = field.types[-1]

            if sub_type == GGUFValueType.STRING:
                return [str(bytes(field.parts[idx]), encoding='utf8') for idx in field.data]
            else:
                return [pv for idx in field.data for pv in field.parts[idx].tolist()]
        if main_type == GGUFValueType.STRING:
            return str(bytes(field.parts[-1]), encoding='utf8')
        else:
            return field.parts[-1][0]

    return None


def get_field_data(reader: GGUFReader, key: str) -> Any:
    field = reader.get_field(key)

    return decode_field(field)


def copy_with_filename(reader: GGUFReader, writer: GGUFWriter, filename: str[Any]) -> None:
    logger.debug(f'copy_with_filename: {filename}') # debug
    val = filename
    for field in reader.fields.values():
        # Suppress virtual fields and fields written by GGUFWriter
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            logger.debug(f'Suppressing {field.name}')
            continue

        # Copy existed fields except 'embedded_files'
        if not field.name == Keys.EMBEDDED_FILES:
            cur_val = decode_field(field)
            writer.add_key(field.name)
            writer.add_val(cur_val, field.types[0])
            logger.debug(f'Copying {field.name}')
            continue

        # Update embedded_files
        val = decode_field(field)
        for path in filename:
            logger.debug(f'Adding {field.name}: {path}')
            val.append(path)

    # Add filenames to kv
    logger.info(f'* Modifying {Keys.EMBEDDED_FILES} to {val}')
    writer.add_array(Keys.EMBEDDED_FILES, val)

    for tensor in reader.tensors:
        # Dimensions are written in reverse order, so flip them first
        shape = np.flipud(tensor.shape)
        writer.add_tensor_info(tensor.name, shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    # Add file info as tensor_info
    for path in filename:
        logger.debug(f'Adding tensor_info {path}')
        with open(path, "rb") as f:
            data = f.read()
            data_len = len(data)
            dims = [data_len]
            raw_dtype = GGMLQuantizationType.I8
            writer.add_tensor_info(path, dims, np.float16, data_len, raw_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)

    # Write file body as tensor data
    for path in filename:
        logger.debug(f'Adding tensor data {path}')
        with open(path, "rb") as f:
            data = f.read()
            data_len = len(data)
            # write data with padding
            writer.write_data(data)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add files to GGUF file metadata")
    parser.add_argument("input",        type=str,            help="GGUF format model input filename")
    parser.add_argument("output",       type=str,            help="GGUF format model output filename")
    parser.add_argument("addfiles",     type=str, nargs='+', help="add filenames ...")
    parser.add_argument("--force",      action="store_true", help="Bypass warnings without confirmation")
    parser.add_argument("--verbose",    action="store_true", help="Increase output verbosity")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info(f'* Loading: {args.input}')
    reader = GGUFReader(args.input, 'r')
    arch = get_field_data(reader, Keys.General.ARCHITECTURE)
    endianess = get_byteorder(reader)

    if os.path.isfile(args.output) and not args.force:
        logger.warning('*** Warning *** Warning *** Warning **')
        logger.warning(f'* The "{args.output}" GGUF file already exists, it will be overwritten!')
        logger.warning('* Enter exactly YES if you are positive you want to proceed:')
        response = input('YES, I am sure> ')
        if response != 'YES':
            logger.info("You didn't enter YES. Okay then, see ya!")
            sys.exit(0)

    logger.info(f'* Writing: {args.output}')
    writer = GGUFWriter(args.output, arch=arch, endianess=endianess)

    alignment = get_field_data(reader, Keys.General.ALIGNMENT)
    if alignment is not None:
        logger.debug(f'Setting custom alignment: {alignment}')
        writer.data_alignment = alignment

    if args.addfiles is not None:
        filename = []
        for path in args.addfiles:
            filename.append(path)
            logger.info(f'* Adding: {path}')
        copy_with_filename(reader, writer, filename)


if __name__ == '__main__':
    main()
