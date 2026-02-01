import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REGISTRY = os.path.join(ROOT, "tools", "sycl-kernel-bench", "kernel_registry.hpp")


def test_reference_kernels_registered():
    with open(REGISTRY, "r", encoding="utf-8") as f:
        text = f.read()
    for name in [
        "onednn_fp16_gemm",
        "onednn_int8_gemm",
        "onednn_woq_gemm",
        "memory_bandwidth",
        "roofline_compute",
    ]:
        assert name in text, f"{name} not registered in kernel_registry.hpp"
