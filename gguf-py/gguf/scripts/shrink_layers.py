#!/usr/bin/env python3
"""
Shrink a GGUF model by keeping only selected transformer layers.

In interactive mode (the default), the script loads a GGUF file, displays the
layer structure, prompts for which layers to keep, and asks for an output name.

Batch mode is also supported by passing --layers and the output path.

Usage:
    python shrink_layers.py input.gguf                         # interactive
    python shrink_layers.py input.gguf -l 0 -o output.gguf     # keep layer 0 only
    python shrink_layers.py input.gguf -l 0,5,10 -o out.gguf   # keep layers 0, 5, 10
    python shrink_layers.py input.gguf -d 0,5 -o out.gguf      # delete layers 0, 5
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# Allow running from the gguf-py directory directly
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / 'gguf').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf

logger = logging.getLogger("shrink-layers")

BLOB_RE = re.compile(r'^blk\.(\d+)\.')


def get_field_data(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    return field.contents() if field else None


def format_bytes(n: int) -> str:
    """Return a human-readable byte count."""
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    elif n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.2f} MiB"
    elif n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def format_counts(n: int) -> str:
    """Return a human-readable parameter count."""
    if n >= 1000 ** 3:
        return f"{n / 1000 ** 3:.1f}B"
    elif n >= 1000 ** 2:
        return f"{n / 1000 ** 2:.1f}M"
    elif n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


def analyze_model(reader: gguf.GGUFReader) -> tuple[str, int, dict[int, int], dict[int, int]]:
    """Return (arch, block_count, per_layer_params, per_layer_bytes)."""
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)

    block_count_key = gguf.Keys.LLM.BLOCK_COUNT.format(arch=arch)
    block_count = get_field_data(reader, block_count_key)
    if block_count is None:
        max_blk = -1
        for tensor in reader.tensors:
            m = BLOB_RE.match(tensor.name)
            if m:
                max_blk = max(max_blk, int(m.group(1)))
        block_count = max_blk + 1

    per_layer_params = defaultdict(int)
    per_layer_bytes = defaultdict(int)
    for tensor in reader.tensors:
        m = BLOB_RE.match(tensor.name)
        if m:
            idx = int(m.group(1))
            per_layer_params[idx] += tensor.n_elements
            per_layer_bytes[idx] += tensor.n_bytes

    return arch, block_count, dict(per_layer_params), dict(per_layer_bytes)


def build_remap(keep_layers: set[int]) -> dict[int, int]:
    """Build a mapping from old layer index to new consecutive index."""
    return {old: new for new, old in enumerate(sorted(keep_layers))}


def remap_tensor_name(name: str, remap: dict[int, int]) -> str:
    """Rename a block tensor's layer index; returns unchanged if not a block tensor."""
    m = BLOB_RE.match(name)
    if m is None:
        return name
    old_idx = int(m.group(1))
    if old_idx not in remap:
        return name
    return name.replace(f'blk.{old_idx}.', f'blk.{remap[old_idx]}.', 1)


def shrink_layers(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, keep_layers: set[int]) -> None:
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    block_count_key = gguf.Keys.LLM.BLOCK_COUNT.format(arch=arch)

    new_block_count = len(keep_layers)
    remap = build_remap(keep_layers)
    logger.info(f"Keeping {new_block_count} layer(s): {sorted(keep_layers)}")
    if remap != {k: k for k in keep_layers}:
        logger.info(f"Layer renumbering: {remap}")

    # Copy metadata, adjusting block count
    kv_count = 0
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()

        if field.name == block_count_key:
            val = new_block_count

        # Cap leading_dense_block_count if present
        leading_dense_key = gguf.Keys.LLM.LEADING_DENSE_BLOCK_COUNT.format(arch=arch)
        if field.name == leading_dense_key:
            val = min(val, new_block_count)

        writer.add_key_value(field.name, val, val_type, sub_type)
        kv_count += 1

    # Filter and renumber tensors
    kept_tensors = 0
    removed_tensors = 0
    for tensor in reader.tensors:
        new_name = remap_tensor_name(tensor.name, remap)
        m = BLOB_RE.match(tensor.name)
        if m:
            if int(m.group(1)) in keep_layers:
                kept_tensors += 1
                writer.add_tensor_info(
                    new_name, tensor.data.shape, tensor.data.dtype,
                    tensor.data.nbytes, tensor.tensor_type,
                )
            else:
                removed_tensors += 1
        else:
            kept_tensors += 1
            writer.add_tensor_info(
                new_name, tensor.data.shape, tensor.data.dtype,
                tensor.data.nbytes, tensor.tensor_type,
            )

    logger.info(f"Metadata keys copied: {kv_count}")
    logger.info(f"Tensors kept: {kept_tensors}, removed: {removed_tensors}")

    write_output(reader, writer, keep_layers, remap)


def write_output(reader: gguf.GGUFReader, writer: gguf.GGUFWriter,
                 keep_layers: set[int], remap: dict[int, int]) -> None:
    is_kept = lambda name: (  # noqa: E731
        (m := BLOB_RE.match(name)) is None or int(m.group(1)) in keep_layers
    )

    total_bytes = sum(tensor.n_bytes for tensor in reader.tensors if is_kept(tensor.name))
    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Write tensor data in the same order as add_tensor_info was called.
    # write_tensor_data pops from the writer's internal dict in order, so we
    # must iterate reader.tensors in the same sequence.
    for tensor in reader.tensors:
        if not is_kept(tensor.name):
            continue
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()
    bar.close()


def parse_layers_arg(s: str) -> set[int]:
    """Parse a comma-separated list of layer indices."""
    layers = set()
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            layers.update(range(int(a), int(b) + 1))
        else:
            layers.add(int(part))
    return layers


def _parse_and_validate(arg: str, reader: gguf.GGUFReader) -> set[int]:
    """Parse a layer arg string and validate against the model's total layers."""
    try:
        layers = parse_layers_arg(arg)
    except ValueError:
        logger.error(f"Invalid layers format: {arg}")
        sys.exit(1)

    _, total_layers, *_ = analyze_model(reader)
    invalid = [i for i in layers if i < 0 or i >= total_layers]
    if invalid:
        logger.error(f"Invalid layer indices: {invalid} (valid range: 0-{total_layers - 1})")
        sys.exit(1)
    if not layers:
        logger.error("No layers specified")
        sys.exit(1)
    return layers


def prompt_layers(total_layers: int) -> set[int]:
    """Prompt the user for which layers to keep (or delete with '-' prefix)."""
    print()
    while True:
        raw = input(f"Layers to keep? (0-{total_layers - 1}, comma-sep, default=0; prefix with '-' to delete): ").strip()
        if raw == '':
            return {0}

        delete_mode = raw.startswith('-')
        if delete_mode:
            raw = raw[1:].strip()

        try:
            layers = parse_layers_arg(raw)
        except ValueError:
            print(f"Invalid input. Enter comma-separated indices (e.g., '0,1,2' or '0-3', or '-0,5' to delete).")
            continue

        invalid = [i for i in layers if i < 0 or i >= total_layers]
        if invalid:
            print(f"Invalid layer indices: {invalid} (valid range: 0-{total_layers - 1})")
            continue
        if not layers:
            print("No layers specified.")
            continue

        if delete_mode:
            keep = set(range(total_layers)) - layers
            if not keep:
                print("Cannot delete all layers.")
                continue
            return keep
        return layers


def prompt_output(input_path: Path) -> Path:
    """Prompt the user for the output file path."""
    print()
    stem = input_path.stem
    default = input_path.with_name(f"{stem}-shrunk.gguf")
    raw = input(f"Output file name [{default}]: ").strip()
    if raw == '':
        return default
    out = Path(raw)
    if out.suffix != '.gguf':
        out = out.with_suffix('.gguf')
    return out


def run_interactive(input_path: Path, reader: gguf.GGUFReader) -> None:
    arch, total_layers, per_layer_params, per_layer_bytes = analyze_model(reader)

    total_params = sum(per_layer_params.values())
    total_bytes = sum(per_layer_bytes.values())

    # Step 1: display model info
    print(f"\nModel architecture: {arch}")
    print(f"Total layers: {total_layers}  |  params/layer: ~{format_counts(total_params // max(total_layers, 1))}  |  size/layer: ~{format_bytes(total_bytes // max(total_layers, 1))}")

    # Step 2: ask which layers to keep
    keep_layers = prompt_layers(total_layers)

    # Step 3: ask for output name
    output_path = prompt_output(input_path)

    if output_path.exists():
        ans = input(f"File '{output_path}' already exists. Overwrite? [y/N]: ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            sys.exit(0)

    _do_write(input_path, reader, output_path, keep_layers)


def run_batch(input_path: Path, reader: gguf.GGUFReader, output_path: Path, layers: set[int]) -> None:
    if output_path.exists():
        logger.warning(f"File '{output_path}' already exists, will be overwritten")
    _do_write(input_path, reader, output_path, layers)


def _do_write(input_path: Path, reader: gguf.GGUFReader, output_path: Path, keep_layers: set[int]) -> None:
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)

    logger.info(f"Writing: {output_path}")
    writer = gguf.GGUFWriter(str(output_path), arch=arch, endianess=reader.endianess)

    if alignment is not None:
        writer.data_alignment = alignment

    shrink_layers(reader, writer, keep_layers)
    logger.info("Done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shrink a GGUF model by keeping only selected transformer layers",
    )
    parser.add_argument("input", type=Path, help="Input GGUF file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output GGUF file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--layers", type=str, default=None,
                       help="Layers to KEEP, comma-separated (e.g. '0' or '0,5,10' or '0-3')")
    group.add_argument("-d", "--delete", type=str, default=None,
                       help="Layers to DELETE, comma-separated (e.g. '0,5' or '0-3')")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output file if it exists")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info(f"Loading: {args.input}")
    reader = gguf.GGUFReader(args.input, 'r')

    # Batch mode: --layers/--delete and --output provided
    if args.layers is not None and args.output is not None:
        layers = _parse_and_validate(args.layers, reader)
        run_batch(args.input, reader, args.output, layers)
    elif args.delete is not None and args.output is not None:
        delete_layers = _parse_and_validate(args.delete, reader)
        _, total_layers, *_ = analyze_model(reader)
        keep_layers = set(range(total_layers)) - delete_layers
        if not keep_layers:
            logger.error("Cannot delete all layers")
            sys.exit(1)
        logger.info(f"Deleting {len(delete_layers)} layer(s): {sorted(delete_layers)}")
        run_batch(args.input, reader, args.output, keep_layers)
    elif args.layers is not None or args.delete is not None or args.output is not None:
        parser.error("-l/-d and -o must be used together for batch mode")
    else:
        # Interactive mode
        run_interactive(args.input, reader)


if __name__ == '__main__':
    main()
