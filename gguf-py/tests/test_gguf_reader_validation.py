import struct
import numpy as np
import pytest

from gguf.gguf_reader import GGUFReader


def _write_gguf(path, n_dims_field, dims):
    buf = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)  # version 3, 1 tensor, 0 kv
    name = b'bad_tensor'
    buf += struct.pack('<Q', len(name)) + name
    buf += struct.pack('<I', n_dims_field)
    for d in dims:
        buf += struct.pack('<Q', d)
    buf += struct.pack('<I', 0)  # dtype F32
    buf += struct.pack('<Q', 0)  # tensor offset
    buf += b'\x00' * 64
    path.write_bytes(buf)


def test_n_dims_upper_bound(tmp_path):
    # crafted file claims 1_000_000 dims; must be rejected, not read past EOF
    p = tmp_path / 'evil_ndims.gguf'
    _write_gguf(p, 1_000_000, [1] * 8)
    with pytest.raises(ValueError, match='exceeds GGML_MAX_DIMS'):
        GGUFReader(p)


def test_dims_product_no_uint64_wraparound(tmp_path):
    # dims whose true product overflows uint64; np.prod would wrap to 4 and
    # silently pass an undersized read. The reader must not accept it.
    dims = [4194305, 4194305, 211106198978564]
    assert int(np.prod(np.array(dims, dtype=np.uint64))) == 4  # the wrap bug
    p = tmp_path / 'evil_overflow.gguf'
    _write_gguf(p, len(dims), dims)
    with pytest.raises(ValueError):
        GGUFReader(p)


def _write_gguf_array_kv(path, elem_type, alen, payload=b''):
    buf = b'GGUF' + struct.pack('<IQQ', 3, 0, 1)  # version 3, 0 tensors, 1 kv
    name = b'k'
    buf += struct.pack('<Q', len(name)) + name
    buf += struct.pack('<I', 9)  # vtype ARRAY
    buf += struct.pack('<I', elem_type)
    buf += struct.pack('<Q', alen)
    buf += payload
    path.write_bytes(buf)


def test_array_len_larger_than_data(tmp_path):
    # Declared scalar array count exceeds the file's remaining bytes. Before
    # the check, each missing element read as an empty array, so offs never
    # advanced and the loop spent `alen` iterations growing parts lists.
    p = tmp_path / 'evil_alen.gguf'
    _write_gguf_array_kv(p, 4, 1_000_000_000)  # UINT32 elements, no data
    with pytest.raises(ValueError, match='does not fit in the remaining'):
        GGUFReader(p)


def test_array_len_partial_data(tmp_path):
    # Same for an array truncated mid-way: 3 declared UINT32, 1 present.
    p = tmp_path / 'evil_alen2.gguf'
    _write_gguf_array_kv(p, 4, 3, struct.pack('<I', 7))
    with pytest.raises(ValueError, match='does not fit in the remaining'):
        GGUFReader(p)
