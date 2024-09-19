#!/usr/bin/env python3
import logging
import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
from typing import Any, Mapping, Sequence

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf

logger = logging.getLogger("gguf-new-metadata")


def get_byteorder(reader: gguf.GGUFReader) -> gguf.GGUFEndian:
    if np.uint32(1) == np.uint32(1).newbyteorder("<"):
        # Host is little endian
        host_endian = gguf.GGUFEndian.LITTLE
        swapped_endian = gguf.GGUFEndian.BIG
    else:
        # Sorry PDP or other weird systems that don't use BE or LE.
        host_endian = gguf.GGUFEndian.BIG
        swapped_endian = gguf.GGUFEndian.LITTLE

    if reader.byte_order == "S":
        return swapped_endian
    else:
        return host_endian


def decode_field(field: gguf.ReaderField) -> Any:
    if field and field.types:
        main_type = field.types[0]

        if main_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[-1]

            if sub_type == gguf.GGUFValueType.STRING:
                return [str(bytes(field.parts[idx]), encoding='utf8') for idx in field.data]
            else:
                return [pv for idx in field.data for pv in field.parts[idx].tolist()]
        if main_type == gguf.GGUFValueType.STRING:
            return str(bytes(field.parts[-1]), encoding='utf8')
        else:
            return field.parts[-1][0]

    return None


def get_field_data(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)

    return decode_field(field)


def copy_with_new_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, new_metadata: Mapping[str, str], remove_metadata: Sequence[str]) -> None:
    for field in reader.fields.values():
        # Suppress virtual fields and fields written by GGUFWriter
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            logger.debug(f'Suppressing {field.name}')
            continue

        # Skip old chat templates if we have new ones
        if field.name.startswith(gguf.Keys.Tokenizer.CHAT_TEMPLATE) and gguf.Keys.Tokenizer.CHAT_TEMPLATE in new_metadata:
            logger.debug(f'Skipping {field.name}')
            continue

        if field.name in remove_metadata:
            logger.debug(f'Removing {field.name}')
            continue

        old_val = decode_field(field)
        val = new_metadata.get(field.name, old_val)

        if field.name in new_metadata:
            logger.debug(f'Modifying {field.name}: "{old_val}" -> "{val}"')
            del new_metadata[field.name]
        elif val is not None:
            logger.debug(f'Copying {field.name}')

        if val is not None:
            writer.add_key(field.name)
            writer.add_val(val, field.types[0])

    if gguf.Keys.Tokenizer.CHAT_TEMPLATE in new_metadata:
        logger.debug('Adding chat template(s)')
        writer.add_chat_template(new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE])
        del new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE]

    # TODO: Support other types than string?
    for key, val in new_metadata.items():
        logger.debug(f'Adding {key}: {val}')
        writer.add_key(key)
        writer.add_val(val, gguf.GGUFValueType.STRING)

    for tensor in reader.tensors:
        # Dimensions are written in reverse order, so flip them first
        shape = np.flipud(tensor.shape)
        writer.add_tensor_info(tensor.name, shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a copy of a GGUF file with new metadata")
    parser.add_argument("input",                                       type=Path, help="GGUF format model input filename")
    parser.add_argument("output",                                      type=Path, help="GGUF format model output filename")
    parser.add_argument("--general-name",                              type=str,  help="The models general.name")
    parser.add_argument("--general-description",                       type=str,  help="The models general.description")
    parser.add_argument("--chat-template",                             type=str,  help="Chat template string (or JSON string containing templates)")
    parser.add_argument("--chat-template-config",                      type=Path, help="Config file (tokenizer_config.json) containing chat template(s)")
    parser.add_argument("--remove-metadata",      action="append",     type=str,  help="Remove metadata (by key name) from output model")
    parser.add_argument("--force",                action="store_true",            help="Bypass warnings without confirmation")
    parser.add_argument("--verbose",              action="store_true",            help="Increase output verbosity")
    args = parser.parse_args(None if len(sys.argv) > 2 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    new_metadata = {}
    remove_metadata = args.remove_metadata or []

    if args.general_name:
        new_metadata[gguf.Keys.General.NAME] = args.general_name

    if args.general_description:
        new_metadata[gguf.Keys.General.DESCRIPTION] = args.general_description

    if args.chat_template:
        new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE] = json.loads(args.chat_template) if args.chat_template.startswith('[') else args.chat_template

    if args.chat_template_config:
        with open(args.chat_template_config, 'r') as fp:
            config = json.load(fp)
            template = config.get('chat_template')
            if template:
                new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE] = template

    if remove_metadata:
        logger.warning('*** Warning *** Warning *** Warning **')
        logger.warning('* Most metadata is required for a fully functional GGUF file,')
        logger.warning('* removing crucial metadata may result in a corrupt output file!')

        if not args.force:
            logger.warning('* Enter exactly YES if you are positive you want to proceed:')
            response = input('YES, I am sure> ')
            if response != 'YES':
                logger.info("You didn't enter YES. Okay then, see ya!")
                sys.exit(0)

    logger.info(f'* Loading: {args.input}')
    reader = gguf.GGUFReader(args.input, 'r')

    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
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
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=endianess)

    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        logger.debug(f'Setting custom alignment: {alignment}')
        writer.data_alignment = alignment

    copy_with_new_metadata(reader, writer, new_metadata, remove_metadata)


if __name__ == '__main__':
    main()
