import os
import subprocess

import numpy as np
import pytest

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, quantize, dequantize


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


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.PTQ1_0, GGMLQuantizationType.TQ1_0])
def test_ternary_tensor_payload_roundtrip(tmp_path, qtype):
    path = tmp_path / "ternary.gguf"
    data = (np.arange(512, dtype=np.int32) % 3 - 1).astype(np.float32).reshape(2, 256)
    payload = quantize(data, qtype)
    writer = GGUFWriter(path, "llama")
    writer.add_tensor("blk.0.ffn_down.weight", payload, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    tensors = GGUFReader(path).tensors
    assert len(tensors) == 1
    tensor = tensors[0]
    assert tensor.tensor_type == qtype
    np.testing.assert_array_equal(tensor.shape, [256, 2])
    np.testing.assert_array_equal(tensor.data, payload)
    np.testing.assert_array_equal(dequantize(tensor.data, qtype), data)


@pytest.mark.parametrize("width,qtype", [
    (256, GGMLQuantizationType.PTQ1_0),
    (96, GGMLQuantizationType.Q4_0),
    (127, GGMLQuantizationType.F16),
])
def test_ptq1_quantize_cli_roundtrip(tmp_path, width, qtype):
    binary = os.environ.get("LLAMA_TEST_QUANTIZE_BIN")
    if binary is None:
        pytest.skip("set LLAMA_TEST_QUANTIZE_BIN to a CPU llama-quantize executable")
    source = tmp_path / "source.gguf"
    target = tmp_path / "ptq1.gguf"
    copied = tmp_path / "copied.gguf"
    data = (np.arange(128 * width, dtype=np.int32) % 3 - 1).astype(np.float32).reshape(128, width)
    data[:, :128] *= 0.5
    writer = GGUFWriter(source, "llama")
    writer.add_context_length(32)
    writer.add_embedding_length(128)
    writer.add_feed_forward_length(width)
    writer.add_block_count(1)
    writer.add_head_count(4)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_tensor("blk.0.ffn_down.weight", data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    subprocess.run([binary, "--pure", str(source), str(target), "PTQ1_0", "1"], check=True, capture_output=True)
    subprocess.run([binary, str(target), str(copied), "COPY", "1"], check=True, capture_output=True)
    expected = quantize(data, qtype)
    for path in (target, copied):
        reader = GGUFReader(path)
        assert reader.fields["general.file_type"].contents() == 143
        assert reader.tensors[0].tensor_type == qtype
        np.testing.assert_array_equal(reader.tensors[0].data, expected)
        np.testing.assert_array_equal(dequantize(reader.tensors[0].data, qtype), dequantize(expected, qtype))
        if qtype == GGMLQuantizationType.PTQ1_0:
            np.testing.assert_array_equal(dequantize(reader.tensors[0].data, qtype), data)
