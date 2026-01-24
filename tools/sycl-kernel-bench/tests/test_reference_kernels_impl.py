import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REF_DIR = os.path.join(ROOT, "tools", "sycl-kernel-bench", "kernels", "reference")


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_memory_bandwidth_counts_read_and_write():
    text = _read(os.path.join(REF_DIR, "memory_bandwidth.cpp"))
    # Expect bytes moved to account for both read and write traffic.
    assert re.search(r"bytes_moved\s*=\s*static_cast<double>\(alloc_bytes\)\s*\*\s*2\.0", text), (
        "memory_bandwidth should count read+write bytes (2x alloc_bytes)"
    )
    assert "bytes_moved" in text and "bandwidth_gbps" in text


def test_onednn_int8_uses_activation_strides_like_fp16():
    text = _read(os.path.join(REF_DIR, "onednn_int8_gemm.cpp"))
    # Match the activation (B) stride order used by DnnlGemmWrapper (str_b2, str_b0, str_b1).
    assert re.search(r"b_strides\s*=\s*\{\s*str_b2\s*,\s*str_b0\s*,\s*str_b1\s*\}", text), (
        "onednn_int8_gemm B strides should match FP16 wrapper layout (str_b2, str_b0, str_b1)"
    )
