import numpy as np
import pytest

from gguf import GGUFReader, GGUFValueType, GGUFWriter


def test_metadata_array_scalar_types(tmp_path):
    path = tmp_path / "metadata-arrays.gguf"
    writer = GGUFWriter(path, "llama")
    writer.add_array("test.np_int32", [np.int32(1), np.int32(-2)])
    writer.add_array("test.np_float32", [np.float32(1.5), np.float32(-2.5)])
    writer.add_array("test.np_bool", [np.bool_(True), np.bool_(False)])
    writer.add_array("test.int32_max", [2**31 - 1, -(2**31)])
    writer.add_array("test.int64", [2**31, -(2**31) - 1, 2**40])
    writer.add_array("test.uint64", [2**63])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()

    fields = GGUFReader(path).fields
    assert fields["test.np_int32"].types[-1] == GGUFValueType.INT32
    assert fields["test.np_int32"].contents() == [1, -2]
    assert fields["test.np_float32"].types[-1] == GGUFValueType.FLOAT32
    assert fields["test.np_float32"].contents() == [1.5, -2.5]
    assert fields["test.np_bool"].types[-1] == GGUFValueType.BOOL
    assert fields["test.np_bool"].contents() == [True, False]
    assert fields["test.int32_max"].types[-1] == GGUFValueType.INT32
    assert fields["test.int64"].types[-1] == GGUFValueType.INT64
    assert fields["test.int64"].contents() == [2**31, -(2**31) - 1, 2**40]
    assert fields["test.uint64"].types[-1] == GGUFValueType.UINT64
    assert fields["test.uint64"].contents() == [2**63]


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
