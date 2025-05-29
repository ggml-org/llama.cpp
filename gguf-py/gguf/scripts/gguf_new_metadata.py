#!/usr/bin/env python3
from __future__ import annotations

import logging
import argparse
import os
import sys
import json
from pathlib import Path

from tqdm import tqdm
from typing import Any, Sequence, NamedTuple

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-new-metadata")


class MetadataDetails(NamedTuple):
    type: gguf.GGUFValueType
    value: Any
    description: str = ''
    sub_type: gguf.GGUFValueType | None = None


def get_field_data(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)

    return field.contents() if field else None


def find_token(token_list: Sequence[int], token: str) -> Sequence[int]:
    token_ids = [index for index, value in enumerate(token_list) if value == token]

    if len(token_ids) == 0:
        raise LookupError(f'Unable to find "{token}" in token list!')

    return token_ids


def copy_with_new_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, new_metadata: dict[str, MetadataDetails], remove_metadata: Sequence[str]) -> None:
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

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        old_val = MetadataDetails(val_type, field.contents(), sub_type=sub_type)
        val = new_metadata.get(field.name, old_val)

        if field.name in new_metadata:
            logger.debug(f'Modifying {field.name}: "{old_val.value}" -> "{val.value}" {val.description}')
            del new_metadata[field.name]
        elif val.value is not None:
            logger.debug(f'Copying {field.name}')

        if val.value is not None:
            writer.add_key_value(field.name, val.value, val.type, sub_type=sub_type if val.sub_type is None else val.sub_type)

    if gguf.Keys.Tokenizer.CHAT_TEMPLATE in new_metadata:
        logger.debug('Adding chat template(s)')
        writer.add_chat_template(new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE].value)
        del new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE]

    for key, val in new_metadata.items():
        logger.debug(f'Adding {key}: "{val.value}" {val.description}')
        writer.add_key_value(key, val.value, val.type)

    total_bytes = 0

    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    writer.close()


def main() -> None:
    tokenizer_metadata = (getattr(gguf.Keys.Tokenizer, n) for n in gguf.Keys.Tokenizer.__dict__.keys() if not n.startswith('_'))
    token_names = dict((n.split('.')[-1][:-len('_token_id')], n) for n in tokenizer_metadata if n.endswith('_token_id'))

    parser = argparse.ArgumentParser(description="Make a copy of a GGUF file with new metadata")
    parser.add_argument("input",                                       type=Path, help="GGUF format model input filename")
    parser.add_argument("output",                                      type=Path, help="GGUF format model output filename")
    parser.add_argument("--general-name",                              type=str,  help="The models general.name", metavar='"name"')
    parser.add_argument("--general-description",                       type=str,  help="The models general.description", metavar='"Description ..."')
    parser.add_argument("--chat-template",                             type=str,  help="Chat template string (or JSON string containing templates)", metavar='"{% ... %} ..."')
    parser.add_argument("--chat-template-config",                      type=Path, help="Config file containing chat template(s)", metavar='tokenizer_config.json')
    parser.add_argument("--pre-tokenizer",                             type=str,  help="The models tokenizer.ggml.pre", metavar='"pre tokenizer"')
    parser.add_argument("--remove-metadata",      action="append",     type=str,  help="Remove metadata (by key name) from output model", metavar='general.url')
    parser.add_argument("--special-token",        action="append",     type=str,  help="Special token by value", nargs=2, metavar=(' | '.join(token_names.keys()), '"<token>"'))
    parser.add_argument("--special-token-by-id",  action="append",     type=str,  help="Special token by id", nargs=2, metavar=(' | '.join(token_names.keys()), '0'))
    parser.add_argument("--force",                action="store_true",            help="Bypass warnings without confirmation")
    parser.add_argument("--verbose",              action="store_true",            help="Increase output verbosity")
    args = parser.parse_args(None if len(sys.argv) > 2 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    new_metadata = {}
    remove_metadata = args.remove_metadata or []

    if args.general_name:
        new_metadata[gguf.Keys.General.NAME] = MetadataDetails(gguf.GGUFValueType.STRING, args.general_name)

    if args.general_description:
        new_metadata[gguf.Keys.General.DESCRIPTION] = MetadataDetails(gguf.GGUFValueType.STRING, args.general_description)

    if args.chat_template:
        new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE] = MetadataDetails(gguf.GGUFValueType.STRING, json.loads(args.chat_template) if args.chat_template.startswith('[') else args.chat_template)

    if args.chat_template_config:
        with open(args.chat_template_config, 'r') as fp:
            config = json.load(fp)
            template = config.get('chat_template')
            if template:
                new_metadata[gguf.Keys.Tokenizer.CHAT_TEMPLATE] = MetadataDetails(gguf.GGUFValueType.STRING, template)

    if args.pre_tokenizer:
        new_metadata[gguf.Keys.Tokenizer.PRE] = MetadataDetails(gguf.GGUFValueType.STRING, args.pre_tokenizer)

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

    token_list = get_field_data(reader, gguf.Keys.Tokenizer.LIST) or []

    for name, token in args.special_token or []:
        if name not in token_names:
            logger.warning(f'Unknown special token "{name}", ignoring...')
        else:
            ids = find_token(token_list, token)
            new_metadata[token_names[name]] = MetadataDetails(gguf.GGUFValueType.UINT32, ids[0], f'= {token}')

            if len(ids) > 1:
                logger.warning(f'Multiple "{token}" tokens found, choosing ID {ids[0]}, use --special-token-by-id if you want another:')
                logger.warning(', '.join(str(i) for i in ids))

    for name, id_string in args.special_token_by_id or []:
        if name not in token_names:
            logger.warning(f'Unknown special token "{name}", ignoring...')
        elif not id_string.isdecimal():
            raise LookupError(f'Token ID "{id_string}" is not a valid ID!')
        else:
            id_int = int(id_string)

            if id_int >= 0 and id_int < len(token_list):
                new_metadata[token_names[name]] = MetadataDetails(gguf.GGUFValueType.UINT32, id_int, f'= {token_list[id_int]}')
            else:
                raise LookupError(f'Token ID {id_int} is not within token list!')

    if os.path.isfile(args.output) and not args.force:
        logger.warning('*** Warning *** Warning *** Warning **')
        logger.warning(f'* The "{args.output}" GGUF file already exists, it will be overwritten!')
        logger.warning('* Enter exactly YES if you are positive you want to proceed:')
        response = input('YES, I am sure> ')
        if response != 'YES':
            logger.info("You didn't enter YES. Okay then, see ya!")
            sys.exit(0)

    logger.info(f'* Writing: {args.output}')
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        logger.debug(f'Setting custom alignment: {alignment}')
        writer.data_alignment = alignment

    copy_with_new_metadata(reader, writer, new_metadata, remove_metadata)


if __name__ == '__main__':
    main()
