#!/usr/bin/env python3
import logging
import argparse
import os
import sys
from pathlib import Path

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader  # noqa: E402

logger = logging.getLogger("gguf-set-metadata")


def minimal_example(filename: str) -> None:
    reader = GGUFReader(filename, 'r+')
    field = reader.fields['tokenizer.ggml.bos_token_id']
    if field is None:
        return
    part_index = field.data[0]
    field.parts[part_index][0] = 2  # Set tokenizer.ggml.bos_token_id to 2
    #
    # So what's this field.data thing? It's helpful because field.parts contains
    # _every_ part of the GGUF field. For example, tokenizer.ggml.bos_token_id consists
    # of:
    #
    #  Part index 0: Key length (27)
    #  Part index 1: Key data ("tokenizer.ggml.bos_token_id")
    #  Part index 2: Field type (4, the id for GGUFValueType.UINT32)
    #  Part index 3: Field value
    #
    # Note also that each part is an NDArray slice, so even a part that
    # is only a single value like the key length will be a NDArray of
    # the key length type (numpy.uint32).
    #
    # The .data attribute in the Field is a list of relevant part indexes
    # and doesn't contain internal GGUF details like the key length part.
    # In this case, .data will be [3] - just the part index of the
    # field value itself.


def set_metadata(reader: GGUFReader, args: argparse.Namespace) -> None:
    field = reader.get_field(args.key)
    if field is None:
        logger.error(f'! Field {repr(args.key)} not found')
        sys.exit(1)
    # Note that field.types is a list of types. This is because the GGUF
    # format supports arrays. For example, an array of UINT32 would
    # look like [GGUFValueType.ARRAY, GGUFValueType.UINT32]
    handler = reader.gguf_scalar_to_np.get(field.types[0]) if field.types else None
    if handler is None:
        logger.error(f'! This tool only supports changing simple values, {repr(args.key)} has unsupported type {field.types}')
        sys.exit(1)
    current_value = field.parts[field.data[0]][0]
    new_value = handler(args.value)
    logger.info(f'* Preparing to change field {repr(args.key)} from {current_value} to {new_value}')
    if current_value == new_value:
        logger.info(f'- Key {repr(args.key)} already set to requested value {current_value}')
        sys.exit(0)
    if args.dry_run:
        sys.exit(0)
    if not args.force:
        logger.warning('*** Warning *** Warning *** Warning **')
        logger.warning('* Changing fields in a GGUF file can make it unusable. Proceed at your own risk.')
        logger.warning('* Enter exactly YES if you are positive you want to proceed:')
        response = input('YES, I am sure> ')
        if response != 'YES':
            logger.info("You didn't enter YES. Okay then, see ya!")
            sys.exit(0)
    field.parts[field.data[0]][0] = new_value
    logger.info('* Field changed. Successful completion.')


def main() -> None:
    parser = argparse.ArgumentParser(description="Set a simple value in GGUF file metadata")
    parser.add_argument("model",     type=str,            help="GGUF format model filename")
    parser.add_argument("key",       type=str,            help="Metadata key to set")
    parser.add_argument("value",     type=str,            help="Metadata value to set")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually change anything")
    parser.add_argument("--force",   action="store_true", help="Change the field without confirmation")
    parser.add_argument("--verbose",      action="store_true",    help="increase output verbosity")

    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info(f'* Loading: {args.model}')
    reader = GGUFReader(args.model, 'r' if args.dry_run else 'r+')
    set_metadata(reader, args)


if __name__ == '__main__':
    main()
