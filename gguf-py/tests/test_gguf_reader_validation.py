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


def _write_gguf_array(path, elem_type, count, payload=b'', pad=0):
    key = b'arr'
    buf = b'GGUF' + struct.pack('<IQQ', 3, 0, 1)  # version 3, 0 tensors, 1 kv
    buf += struct.pack('<Q', len(key)) + key
    buf += struct.pack('<I', 9)  # GGUFValueType.ARRAY
    buf += struct.pack('<I', elem_type)
    buf += struct.pack('<Q', count)
    buf += payload + b'\x00' * pad
    path.write_bytes(buf)


def test_array_length_bounded_by_element_size(tmp_path):
    # A file that declares 5_000_000 FLOAT64 elements but is only ~5 MB. The
    # element count alone is smaller than the bytes remaining, so a bound that
    # compares the count against remaining bytes lets this through and the
    # per-element loop runs 5_000_000 times. Those elements need 40 MB, so the
    # declared array cannot fit and must be rejected up front.
    p = tmp_path / 'evil_array.gguf'
    _write_gguf_array(p, 12, 5_000_000, pad=5_200_008)  # 12 = FLOAT64
    with pytest.raises(ValueError, match='requires at least'):
        GGUFReader(p)


def test_array_of_minimum_size_elements_still_parses(tmp_path):
    # Guard against over-rejecting: a UINT8 array whose elements really are one
    # byte each fits exactly, and an empty array of any element type is legal.
    p = tmp_path / 'ok_uint8.gguf'
    _write_gguf_array(p, 0, 64, payload=b'\x01' * 64)  # 0 = UINT8
    assert GGUFReader(p).fields['arr'].contents() == [1] * 64

    p = tmp_path / 'ok_empty.gguf'
    _write_gguf_array(p, 12, 0)  # 12 = FLOAT64, no elements
    assert GGUFReader(p).fields['arr'].contents() == []
