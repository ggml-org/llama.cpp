#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import logging
import argparse

from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf


logger = logging.getLogger("imatrix-to-gguf")


class IMatrixWriter(gguf.GGUFWriter):
    def add_architecture(self) -> None:
        # no arch is stored in imatrix files
        pass


@dataclass
class IMatrixEntry:
    values: np.ndarray[Any, np.dtype[np.float32]]
    counts: np.ndarray[Any, np.dtype[np.float32]]


class IMatrixReader:
    chunk_size: int = 512  # guess
    offset: int = 0
    data: np.ndarray[Any, np.dtype[np.uint8]]
    n_enties: int
    entries: dict[str, IMatrixEntry]
    chunk_count: int
    dataset: str

    def _get(self, dtype: npt.DTypeLike, count: int = 1) -> npt.NDArray[Any]:
        count = int(count)
        itemsize = int(np.empty([], dtype=dtype).itemsize)
        offset = self.offset
        self.offset = offset + itemsize * count
        return self.data[offset:self.offset].view(dtype=dtype)[:count]

    def __init__(self, imatrix: Path):
        self.offset = 0
        self.entries = {}
        self.data = np.memmap(imatrix)
        n_entries = self._get(np.int32).item()
        assert n_entries >= 0
        for _ in range(n_entries):
            len = self._get(np.int32).item()
            name = self._get(np.uint8, len).tobytes().decode("utf-8")
            ncall = self._get(np.int32).item()
            nval = self._get(np.int32).item()
            data = self._get(np.float32, nval)
            assert name not in self.entries, f"duplicated name: {name!r}"

            self.entries[name] = IMatrixEntry(data * np.float32(self.chunk_size), np.array([ncall * self.chunk_size], dtype=np.float32))

        self.chunk_count = self._get(np.int32).item()
        dataset_len = self._get(np.int32).item()
        self.dataset = self._get(np.uint8, dataset_len).tobytes().decode("utf-8")

    def to_writer(self, outfile: Path) -> IMatrixWriter:
        writer = IMatrixWriter(path=outfile, arch="")

        writer.add_type(gguf.GGUFType.IMATRIX)
        writer.add_key_value(gguf.Keys.IMatrix.CHUNK_COUNT, self.chunk_count, gguf.GGUFValueType.UINT32)
        writer.add_key_value(gguf.Keys.IMatrix.CHUNK_SIZE, self.chunk_size, gguf.GGUFValueType.UINT32)
        writer.add_key_value(gguf.Keys.IMatrix.DATASET, self.dataset, gguf.GGUFValueType.STRING)

        for name, entry in self.entries.items():
            writer.add_tensor(name + ".sums", entry.values)
            writer.add_tensor(name + ".counts", entry.counts)

        return writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an old imatrix.dat file to a GGUF compatible file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "imatrix", type=Path,
        help="path to an imatrix file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.outfile is None:
        input_file: Path = args.imatrix
        if input_file.suffix != ".gguf":
            args.outfile = input_file.with_suffix(".gguf")
            if args.outfile.exists():
                logger.error(f"default file exists, specify with --outfile to overwrite: {args.outfile}")
                exit(1)

    writer = IMatrixReader(args.imatrix).to_writer(args.outfile)

    writer.write_header_to_file(args.outfile)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
