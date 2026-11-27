import struct
import numpy as np
import pytest

from gguf.gguf_reader import GGUFReader


def _write_gguf(path, n_dims_field, dims, offset=0):
    buf = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)  # version 3, 1 tensor, 0 kv
    name = b'bad_tensor'
    buf += struct.pack('<Q', len(name)) + name
    buf += struct.pack('<I', n_dims_field)
    for d in dims:
        buf += struct.pack('<Q', d)
    buf += struct.pack('<I', 0)  # dtype F32
    buf += struct.pack('<Q', offset)  # tensor offset
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


def test_tensor_offset_no_uint64_wraparound(tmp_path):
    # offset 2**64 - 1 must be rejected, not wrap around
    p = tmp_path / 'evil_offset.gguf'
    _write_gguf(p, 1, [4], offset=2**64 - 1)
    with pytest.raises(ValueError, match='exceeds file size'):
        GGUFReader(p)


def _write_tensor_layout(path, tensors, alignment_key=None):
    buf = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), int(alignment_key is not None))
    if alignment_key is not None:
        buf += struct.pack('<Q', len(alignment_key)) + alignment_key
        buf += struct.pack('<II', 4, 64)  # UINT32 alignment value
    for i, (n_elements, offset) in enumerate(tensors):
        name = f'tensor_{i}'.encode('ascii')
        buf += struct.pack('<Q', len(name)) + name
        buf += struct.pack('<IQIQ', 1, n_elements, 0, offset)  # 1 dimension, F32
    alignment = 64 if alignment_key == b'general.alignment' else 32
    buf += bytes(-len(buf) % alignment)
    path.write_bytes(buf + bytes(128))


@pytest.mark.parametrize(('tensors', 'alignment_key', 'error'), [
    ([(4, 32), (4, 0)], None, 'tensor_0.*offset 32, expected 0'),
    ([(4, 0), (4, 0)], None, 'tensor_1.*offset 0, expected 32'),
    ([(0, 0), (4, 32)], None, 'tensor_1.*offset 32, expected 0'),
    ([(4, 0), (4, 32)], b'general.alignment\x00x', 'metadata key.*NUL'),
], ids=['out-of-order', 'overlap', 'gap-after-empty-tensor', 'nul-in-key'])
def test_strict_reader_rejects_invalid_layout(tmp_path, tensors, alignment_key, error):
    path = tmp_path / 'layout.gguf'
    _write_tensor_layout(path, tensors, alignment_key)

    assert len(GGUFReader(path).tensors) == len(tensors)
    with pytest.raises(ValueError, match=error):
        GGUFReader(path, strict=True)


@pytest.mark.parametrize(('tensors', 'alignment_key'), [
    ([(4, 0), (4, 32)], None),
    ([(4, 0), (4, 64)], b'general.alignment'),
    ([(0, 0), (4, 0)], None),
], ids=['default-alignment', 'custom-alignment', 'empty-tensor'])
def test_strict_reader_accepts_valid_layout(tmp_path, tensors, alignment_key):
    path = tmp_path / 'layout.gguf'
    _write_tensor_layout(path, tensors, alignment_key)

    reader = GGUFReader(path, strict=True)
    assert [tensor.data_offset - reader.data_offset for tensor in reader.tensors] == [
        offset for _, offset in tensors
    ]
