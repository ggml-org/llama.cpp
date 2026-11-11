import struct
import numpy as np
import pytest

from gguf.gguf_reader import GGUFReader
from gguf import GGUFWriter, GGUFEndian, GGUFValueType, Keys


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


@pytest.mark.parametrize("endian", [GGUFEndian.LITTLE, GGUFEndian.BIG])
@pytest.mark.parametrize("ids", [[7], [7, 2, 7], [0, 0xFFFFFFFF]])
def test_eog_ids_round_trip(tmp_path, endian, ids):
    path = tmp_path / "eog.gguf"
    writer = GGUFWriter(path, "llama", endianess=endian)
    writer.add_eog_token_ids(ids)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()

    reader = GGUFReader(path)
    field = reader.get_field(Keys.Tokenizer.EOG_IDS)
    assert field is not None
    assert field.types == [GGUFValueType.ARRAY, GGUFValueType.UINT32]
    assert field.contents() == ids