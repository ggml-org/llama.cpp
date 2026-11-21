import numpy as np
import pytest

from gguf import GGUFReader, GGUFWriter


@pytest.mark.parametrize("use_temp_file", [False, True])
def test_tensor_filter_preserves_data_offsets(tmp_path, use_temp_file):
    path = tmp_path / "filtered.gguf"
    data = np.arange(32, dtype=np.float32).reshape(1, 32)
    writer = GGUFWriter(path, "qwen4exp", use_temp_file=use_temp_file,
                        tensor_filter=lambda name: name != "excluded")
    writer.add_tensor("first", data)
    writer.add_tensor("excluded", data + 100)
    writer.add_tensor("last", data + 200)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    tensors = GGUFReader(path).tensors
    assert [t.name for t in tensors] == ["first", "last"]
    np.testing.assert_array_equal(tensors[0].data, data)
    np.testing.assert_array_equal(tensors[1].data, data + 200)
