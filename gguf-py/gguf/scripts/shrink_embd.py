#!/usr/bin/env python3
"""
Shrink the embedding dimension of a GGUF model by slicing along the n_embd axis.

In interactive mode (the default), the script loads a GGUF file, displays the
current embedding dimension, prompts for how to shrink, and asks for an output name.

Batch mode is also supported by passing --new-embd/--drop-first/--drop-last and -o.

Usage:
    python shrink_embd.py input.gguf                              # interactive
    python shrink_embd.py input.gguf -n 500 -o output.gguf        # keep first 500 dims
    python shrink_embd.py input.gguf -n 500 --offset 500 -o out.gguf  # keep dims 500-999
    python shrink_embd.py input.gguf --drop-first 500 -o out.gguf     # drop first 500 dims
    python shrink_embd.py input.gguf --drop-last 500 -o out.gguf      # drop last 500 dims
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running from the gguf-py directory directly
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / 'gguf').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

logger = logging.getLogger("shrink-embd")

_UNQUANTIZED_TYPES = frozenset({
    GGMLQuantizationType.F16, GGMLQuantizationType.F32, GGMLQuantizationType.F64,
    GGMLQuantizationType.I8, GGMLQuantizationType.I16, GGMLQuantizationType.I32,
    GGMLQuantizationType.I64, GGMLQuantizationType.BF16,
})


def get_field_data(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    return field.contents() if field else None


def get_n_embd(reader: gguf.GGUFReader) -> int:
    """Read the embedding dimension from metadata."""
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    n_embd = get_field_data(reader, gguf.Keys.LLM.EMBEDDING_LENGTH.format(arch=arch))
    if n_embd is None:
        logger.error(f"Cannot find {arch}.embedding_length in model metadata")
        sys.exit(1)
    return n_embd


def _is_quantized(qtype: GGMLQuantizationType) -> bool:
    return qtype not in _UNQUANTIZED_TYPES


def _embd_axes(name: str, gguf_shape: list[int], n_embd: int) -> list[int]:
    """Return the GGUF axes that represent the embedding dimension.

    Uses tensor naming heuristics to avoid misidentifying dimensions that
    coincidentally equal n_embd (e.g. n_ff == n_embd in some models).
    """
    axes = []
    ndim = len(gguf_shape)

    for axis, dim in enumerate(gguf_shape):
        if dim != n_embd:
            continue

        # For 1D tensors, the only dimension is n_embd.
        if ndim == 1:
            axes.append(axis)
            continue

        # -- FFN weight tensors -------------------------------------------
        # When n_ff == n_embd, we must distinguish them by position:
        #
        #   ffn_gate(_exps), ffn_up(_exps): [n_embd, n_ff, ...]
        #     axis 0 = n_embd,  axis 1 = n_ff (NOT n_embd)
        #   ffn_down(_exps):                [n_ff, n_embd, ...]
        #     axis 0 = n_ff (NOT n_embd),  axis 1 = n_embd
        #
        if 'ffn_gate' in name or 'ffn_up' in name:
            # Bias for gate/up: [n_ff, n_expert] – no n_embd dimension at all
            if name.endswith('.bias'):
                continue
            if axis == 1:
                continue  # axis 1 is n_ff, not n_embd

        elif 'ffn_down' in name:
            # Bias for down: [n_embd, n_expert] – axis 0 IS n_embd
            if not name.endswith('.bias'):
                if axis == 0:
                    continue  # axis 0 is n_ff, not n_embd

        axes.append(axis)

    return axes


def _slice_tensor_data(
    tensor: gguf.ReaderTensor,
    n_embd: int,
    offset: int,
    new_n_embd: int,
) -> tuple[np.ndarray, list[int], int]:
    """
    Slice a tensor along embedding dimension axes.

    Returns (sliced_data, new_gguf_shape, new_nbytes).
    """
    gguf_shape = tensor.shape.tolist()  # [d0, d1, ...] in GGUF order

    # Determine which axes carry the embedding dimension.
    embd_axes = _embd_axes(tensor.name, gguf_shape, n_embd)

    # Build numpy slice tuple. numpy axes are reversed from GGUF order:
    #   GGUF [d0, d1, d2]  <->  numpy (d2, d1, d0)
    ndim = len(gguf_shape)
    np_slices = [slice(None)] * ndim
    new_gguf_shape = list(gguf_shape)

    for gguf_axis in embd_axes:
        np_axis = ndim - 1 - gguf_axis
        np_slices[np_axis] = slice(offset, offset + new_n_embd)
        new_gguf_shape[gguf_axis] = new_n_embd

    np_slices = tuple(np_slices)

    if not _is_quantized(tensor.tensor_type):
        sliced = tensor.data[np_slices].copy()
        return sliced, new_gguf_shape, sliced.nbytes

    # Quantized: data is uint8 with byte shape.
    # Only the last numpy axis (first GGUF dim) is block-encoded.
    block_size, type_size = GGML_QUANT_SIZES[tensor.tensor_type]
    byte_slices = list(np_slices)
    num_sliced_axes = 0

    for np_axis in range(ndim):
        s = np_slices[np_axis]
        if s == slice(None):
            continue
        num_sliced_axes += 1
        if np_axis == ndim - 1:
            # Last numpy axis = first GGUF axis = quantized dimension.
            # Must be block-aligned.
            start_elems = s.start
            stop_elems = s.stop
            if start_elems % block_size != 0:
                logger.error(
                    f"Tensor '{tensor.name}': slice start {start_elems} must be "
                    f"a multiple of {tensor.tensor_type.name} block size ({block_size})"
                )
                sys.exit(1)
            if stop_elems % block_size != 0:
                logger.error(
                    f"Tensor '{tensor.name}': slice end {stop_elems} must be "
                    f"a multiple of {tensor.tensor_type.name} block size ({block_size})"
                )
                sys.exit(1)
            byte_start = (start_elems // block_size) * type_size
            byte_stop = (stop_elems // block_size) * type_size
            byte_slices[np_axis] = slice(byte_start, byte_stop)

    if num_sliced_axes == 0:
        # No change needed
        sliced = tensor.data.copy()
        nbytes = tensor.n_bytes
    else:
        sliced = tensor.data[tuple(byte_slices)].copy()
        nbytes = int(np.prod(sliced.shape))

    return sliced, new_gguf_shape, nbytes


def shrink_embd(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
    n_embd: int,
    offset: int,
    new_n_embd: int,
) -> dict[str, tuple[np.ndarray, int]]:
    """
    Copy metadata and slice tensors to reduce embedding dimension.

    Returns a dict mapping tensor name -> (sliced_data, nbytes) for writing.
    """
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    embd_key = gguf.Keys.LLM.EMBEDDING_LENGTH.format(arch=arch)
    embd_out_key = gguf.Keys.LLM.EMBEDDING_LENGTH_OUT.format(arch=arch)
    features_key = gguf.Keys.LLM.FEATURES_LENGTH.format(arch=arch)

    logger.info(f"Shrinking embedding dim from {n_embd} to {new_n_embd} (offset={offset})")

    # Copy metadata, adjusting embedding length keys
    kv_count = 0
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()

        if field.name == embd_key:
            val = new_n_embd
        elif field.name == embd_out_key:
            val = new_n_embd
        elif field.name == features_key and isinstance(val, int) and val == n_embd:
            val = new_n_embd

        writer.add_key_value(field.name, val, val_type, sub_type)
        kv_count += 1

    # Slice and register tensors
    sliced_data: dict[str, tuple[np.ndarray, int]] = {}
    kept_count = 0
    modified_count = 0

    for tensor in reader.tensors:
        data, new_gguf_shape, new_nbytes = _slice_tensor_data(
            tensor, n_embd, offset, new_n_embd,
        )

        # Check if tensor was actually modified
        if new_gguf_shape != tensor.shape.tolist():
            modified_count += 1
        kept_count += 1

        # add_tensor_info expects shape in numpy order (matching tensor.shape convention)
        writer.add_tensor_info(
            tensor.name,
            list(data.shape),
            data.dtype,
            new_nbytes,
            raw_dtype=tensor.tensor_type if _is_quantized(tensor.tensor_type) else None,
        )
        sliced_data[tensor.name] = (data, new_nbytes)

    logger.info(f"Metadata keys copied: {kv_count}")
    logger.info(f"Tensors kept: {kept_count}, modified: {modified_count}")

    return sliced_data


def write_output(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
    sliced_data: dict[str, tuple[np.ndarray, int]],
) -> None:
    """Write the GGUF file with sliced tensor data."""
    total_bytes = sum(nbytes for _, nbytes in sliced_data.values())
    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        data, _ = sliced_data[tensor.name]
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        bar.update(data.nbytes)

    writer.close()
    bar.close()


def parse_positive_int(s: str) -> int:
    """Parse a positive integer, raising ValueError on failure."""
    val = int(s)
    if val <= 0:
        raise ValueError(f"Value must be positive, got {val}")
    return val


def resolve_slice_params(
    args: argparse.Namespace,
    n_embd: int,
) -> tuple[int, int]:
    """Resolve CLI args to (offset, new_n_embd)."""
    if args.new_embd is not None:
        # --new-embd N [--offset O]
        new_n_embd = args.new_embd
        offset = args.offset
        if offset + new_n_embd > n_embd:
            logger.error(
                f"Slice [{offset}:{offset + new_n_embd}] exceeds n_embd={n_embd}"
            )
            sys.exit(1)
        return offset, new_n_embd
    elif args.drop_first is not None:
        drop = args.drop_first
        if drop >= n_embd:
            logger.error(f"Cannot drop all {n_embd} embedding dimensions")
            sys.exit(1)
        return drop, n_embd - drop
    elif args.drop_last is not None:
        drop = args.drop_last
        if drop >= n_embd:
            logger.error(f"Cannot drop all {n_embd} embedding dimensions")
            sys.exit(1)
        return 0, n_embd - drop
    else:
        # Should not reach here if called from main
        return 0, n_embd


def prompt_embd(n_embd: int) -> tuple[int, int]:
    """Prompt the user for how to shrink the embedding dimension."""
    print(f"\nCurrent embedding dimension: {n_embd}")
    print()
    while True:
        raw = input(
            f"How to shrink? (e.g.: 'n=500' keep first 500, "
            f"'drop-first=500', 'drop-last=500', "
            f"'n=500,offset=500'): "
        ).strip()
        if raw == '':
            # Default: keep first half
            new_n = n_embd // 2
            print(f"Defaulting to: keep first {new_n} dimensions")
            return 0, new_n

        # Parse key=value pairs
        kwargs: dict[str, int] = {}
        for part in raw.split(','):
            part = part.strip()
            if '=' not in part:
                print(f"Invalid format. Use key=value pairs (e.g., 'n=500')")
                break
            k, v = part.split('=', 1)
            k = k.strip().lower()
            try:
                kwargs[k] = int(v.strip())
            except ValueError:
                print(f"Invalid integer value: {v}")
                break
        else:
            # All parts parsed successfully
            if 'drop-first' in kwargs:
                drop = kwargs['drop-first']
                if drop >= n_embd:
                    print(f"Cannot drop all {n_embd} dimensions")
                    continue
                return drop, n_embd - drop
            elif 'drop-last' in kwargs:
                drop = kwargs['drop-last']
                if drop >= n_embd:
                    print(f"Cannot drop all {n_embd} dimensions")
                    continue
                return 0, n_embd - drop
            elif 'n' in kwargs:
                new_n = kwargs['n']
                offset = kwargs.get('offset', 0)
                if offset + new_n > n_embd:
                    print(f"Slice [{offset}:{offset+new_n}] exceeds n_embd={n_embd}")
                    continue
                return offset, new_n
            else:
                print("Unknown option. Use 'n', 'drop-first', or 'drop-last'.")


def prompt_output(input_path: Path) -> Path:
    """Prompt the user for the output file path."""
    print()
    stem = input_path.stem
    default = input_path.with_name(f"{stem}-shrunk-embd.gguf")
    raw = input(f"Output file name [{default}]: ").strip()
    if raw == '':
        return default
    out = Path(raw)
    if out.suffix != '.gguf':
        out = out.with_suffix('.gguf')
    return out


def run_interactive(input_path: Path, reader: gguf.GGUFReader) -> None:
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    n_embd = get_n_embd(reader)

    print(f"\nModel architecture: {arch}")
    print(f"Current embedding dimension: {n_embd}")

    offset, new_n_embd = prompt_embd(n_embd)
    output_path = prompt_output(input_path)

    if output_path.exists():
        ans = input(f"File '{output_path}' already exists. Overwrite? [y/N]: ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            sys.exit(0)

    _do_write(input_path, reader, output_path, n_embd, offset, new_n_embd)


def run_batch(
    input_path: Path,
    reader: gguf.GGUFReader,
    output_path: Path,
    n_embd: int,
    offset: int,
    new_n_embd: int,
) -> None:
    if output_path.exists():
        logger.warning(f"File '{output_path}' already exists, will be overwritten")
    _do_write(input_path, reader, output_path, n_embd, offset, new_n_embd)


def _do_write(
    input_path: Path,
    reader: gguf.GGUFReader,
    output_path: Path,
    n_embd: int,
    offset: int,
    new_n_embd: int,
) -> None:
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)

    logger.info(f"Writing: {output_path}")
    writer = gguf.GGUFWriter(str(output_path), arch=arch, endianess=reader.endianess)

    if alignment is not None:
        writer.data_alignment = alignment

    sliced_data = shrink_embd(reader, writer, n_embd, offset, new_n_embd)
    write_output(reader, writer, sliced_data)
    logger.info("Done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shrink the embedding dimension of a GGUF model",
    )
    parser.add_argument("input", type=Path, help="Input GGUF file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output GGUF file")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--new-embd", type=parse_positive_int, default=None,
                       help="New embedding dimension (keep first N dims)")
    group.add_argument("--drop-first", type=parse_positive_int, default=None,
                       help="Number of embedding dimensions to drop from the start")
    group.add_argument("--drop-last", type=parse_positive_int, default=None,
                       help="Number of embedding dimensions to drop from the end")

    parser.add_argument("--offset", type=int, default=0,
                        help="Starting dimension offset (used with -n/--new-embd)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.input.is_file():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info(f"Loading: {args.input}")
    reader = gguf.GGUFReader(args.input, 'r')
    n_embd = get_n_embd(reader)

    # Determine batch vs interactive mode
    has_embd_arg = args.new_embd is not None or args.drop_first is not None or args.drop_last is not None

    if has_embd_arg and args.output is not None:
        offset, new_n_embd = resolve_slice_params(args, n_embd)
        logger.info(f"New embedding dim: {new_n_embd} (offset={offset})")
        run_batch(args.input, reader, args.output, n_embd, offset, new_n_embd)
    elif has_embd_arg or args.output is not None:
        parser.error("-n/--new-embd/--drop-first/--drop-last and -o/--output must be used together for batch mode")
    else:
        run_interactive(args.input, reader)


if __name__ == '__main__':
    main()
