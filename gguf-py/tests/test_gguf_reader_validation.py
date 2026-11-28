import struct
import numpy as np
import pytest

from gguf.constants import GGMLQuantizationType, GGUFEndian
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter


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


def test_mixed_quantized_tensor_roundtrip(tmp_path):
    path = tmp_path / 'mixed.gguf'
    # A Q8_0 block stores an F16 scale and 32 signed bytes.
    q8_bytes = struct.pack('<e', 1.0) + bytes(range(32))
    floats = np.array([1.25, -2.5], dtype=np.float32)
    writer = GGUFWriter(path, 'test')
    try:
        writer.add_tensor('quantized', np.frombuffer(q8_bytes, dtype=np.uint8),
                          raw_dtype=GGMLQuantizationType.Q8_0, tensor_endianess=GGUFEndian.LITTLE)
        writer.add_tensor('floats', floats)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    for strict in (False, True):
        reader = GGUFReader(path, strict=strict)
        quantized, floating = reader.tensors
        assert quantized.data_offset - reader.data_offset == 0
        # The 34-byte Q8_0 block takes 64 bytes at 32-byte alignment.
        assert floating.data_offset - reader.data_offset == 64
        assert quantized.data.tobytes() == q8_bytes
        assert floating.data.tobytes() == struct.pack('<2f', 1.25, -2.5)


def test_multidim_tensor_with_quantized_types(tmp_path):
    """Test multi-dimensional tensors with different quantization types."""
    path = tmp_path / 'multidim.gguf'

    # Create tensors with varying dimensions and quantization types
    writer = GGUFWriter(path, 'multidim_test')
    try:
        # 2D tensor: F32
        f32_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        writer.add_tensor('f32_2d', f32_data)

        # 2D tensor: F16
        f16_data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
        writer.add_tensor('f16_2d', f16_data)

        # 3D tensor: F32
        f32_3d = np.ones((2, 3, 4), dtype=np.float32)
        writer.add_tensor('f32_3d', f32_3d)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    # Verify strict mode accepts correct layout
    reader_strict = GGUFReader(path, strict=True)
    assert len(reader_strict.tensors) == 3
    assert reader_strict.tensors[0].name == 'f32_2d'
    assert reader_strict.tensors[1].name == 'f16_2d'
    assert reader_strict.tensors[2].name == 'f32_3d'

    # Verify data integrity
    np.testing.assert_array_equal(reader_strict.tensors[0].data, f32_data)
    np.testing.assert_array_equal(reader_strict.tensors[1].data, f16_data)
    np.testing.assert_array_equal(reader_strict.tensors[2].data, f32_3d)


def test_q4_q8_mixed_quantization(tmp_path):
    """Test mixed Q4_0 and Q8_0 quantized tensors with correct offset calculation."""
    path = tmp_path / 'q4q8_mix.gguf'

    # Q4_0: type_size = 18 bytes per block (encodes 32 elements)
    # Create 18-byte Q4_0 block: 2 bytes scale + 16 bytes qvals
    q4_block = struct.pack('<e', 0.1) + bytes(range(16))
    q4_data = np.frombuffer(q4_block, dtype=np.uint8)

    # Q8_0: 2 bytes scale + 32 bytes data = 34 bytes
    q8_block = struct.pack('<e', 0.2) + bytes(range(32))
    q8_data = np.frombuffer(q8_block, dtype=np.uint8)

    writer = GGUFWriter(path, 'q4q8test')
    try:
        writer.add_tensor('q4', q4_data, raw_dtype=GGMLQuantizationType.Q4_0, tensor_endianess=GGUFEndian.LITTLE)
        writer.add_tensor('q8', q8_data, raw_dtype=GGMLQuantizationType.Q8_0, tensor_endianess=GGUFEndian.LITTLE)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    # Verify strict mode works with mixed quantization
    reader = GGUFReader(path, strict=True)
    q4_tensor, q8_tensor = reader.tensors

    # Q4_0 (18 bytes) starts at 0, padded to 32 bytes
    assert q4_tensor.data_offset - reader.data_offset == 0
    assert q4_tensor.n_bytes == 18

    # Q8_0 (34 bytes) starts after padding, takes 64 bytes at 32-byte alignment
    assert q8_tensor.data_offset - reader.data_offset == 32
    assert q8_tensor.n_bytes == 34


def test_alignment_boundary_exact_fit(tmp_path):
    """Test tensor that exactly fills alignment boundary."""
    path = tmp_path / 'align_exact.gguf'

    writer = GGUFWriter(path, 'align_test')
    try:
        # Create data that's exactly 64 bytes (at 32-byte alignment, 2 blocks)
        data = np.arange(16, dtype=np.float32)  # 16 * 4 = 64 bytes
        writer.add_tensor('exact', data)

        # Second tensor should start at 64
        data2 = np.array([1.0, 2.0], dtype=np.float32)
        writer.add_tensor('second', data2)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    reader = GGUFReader(path, strict=True)
    first, second = reader.tensors

    assert first.data_offset - reader.data_offset == 0
    assert first.n_bytes == 64
    assert second.data_offset - reader.data_offset == 64


def test_zero_length_tensors_same_offset(tmp_path):
    """Test that zero-length tensors can share the same offset."""
    path = tmp_path / 'zero_len.gguf'

    writer = GGUFWriter(path, 'zero_test')
    try:
        # Zero-length F32 tensors
        empty1 = np.array([], dtype=np.float32)
        empty2 = np.array([], dtype=np.float32)
        normal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        writer.add_tensor('empty1', empty1)
        writer.add_tensor('empty2', empty2)
        writer.add_tensor('normal', normal)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    reader = GGUFReader(path, strict=True)
    e1, e2, normal = reader.tensors

    # Empty tensors can have same offset as next non-empty
    assert e1.n_bytes == 0
    assert e2.n_bytes == 0
    # Normal tensor starts where the data actually begins
    assert normal.n_bytes == 16


def test_real_data_roundtrip_with_strict_validation(tmp_path):
    """Test that real tensor data is correctly read and preserved with strict validation."""
    path = tmp_path / 'real_data.gguf'

    # Create tensors with distinctive, non-zero data patterns
    f32_pattern = np.array([3.14159, -2.71828, 1.41421, 0.0], dtype=np.float32)
    f16_pattern = np.array([1.5, -2.5, 3.5], dtype=np.float16)
    i32_pattern = np.array([-100, 200, -300, 400], dtype=np.int32)

    writer = GGUFWriter(path, 'real_data_test')
    try:
        writer.add_tensor('floats32', f32_pattern)
        writer.add_tensor('floats16', f16_pattern)
        writer.add_tensor('ints32', i32_pattern)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    # Read with strict validation
    reader = GGUFReader(path, strict=True)

    # Verify tensors are in correct order
    assert [t.name for t in reader.tensors] == ['floats32', 'floats16', 'ints32']

    # Verify data integrity with real values
    np.testing.assert_array_almost_equal(reader.tensors[0].data, f32_pattern, decimal=5)
    np.testing.assert_array_equal(reader.tensors[1].data, f16_pattern)
    np.testing.assert_array_equal(reader.tensors[2].data, i32_pattern)


def test_large_tensor_with_padding(tmp_path):
    """Test large tensor spanning multiple alignment blocks."""
    path = tmp_path / 'large_padded.gguf'

    writer = GGUFWriter(path, 'large_test')
    try:
        # Create a tensor larger than alignment (32 bytes * 10 = 320 bytes)
        large = np.random.randn(80).astype(np.float32)  # 320 bytes
        writer.add_tensor('large', large)

        # Add another tensor after padding
        small = np.array([1.0], dtype=np.float32)
        writer.add_tensor('small', small)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    reader = GGUFReader(path, strict=True)
    large_t, small_t = reader.tensors

    # Verify large tensor starts at beginning
    assert large_t.data_offset - reader.data_offset == 0
    assert large_t.n_bytes == 320

    # Verify small tensor starts after padding
    # 320 is already aligned to 32, so small should start at 320
    assert small_t.data_offset - reader.data_offset == 320

    # Verify data integrity
    np.testing.assert_array_equal(large_t.data, large)
