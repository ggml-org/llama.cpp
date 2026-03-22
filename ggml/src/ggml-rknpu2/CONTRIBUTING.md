# Contributing to rk-llama.cpp

This document outlines the project's structure, current development priorities, and workflow guidelines.

## Project Structure

All backend-specific code is contained within the `ggml/src/ggml-rknpu2` directory. Here is a brief overview of the key files and their responsibilities:

*   **`ggml-rknpu2.cpp`**
    The core GGML backend implementation. It bridges the GGML framework with the Rockchip NPU.
    *   `ggml_backend_rknpu_buffer_type_alloc_buffer`: Handles the allocation of DMA-Heap memory blocks.
    *   `ggml_backend_rknpu_buffer_set_tensor`: Dequantizes incoming GGUF weights, requantizes them to the target NPU format, packs them into native RKNN layout, and writes them to the DMA buffer.
    *   `ggml_backend_rknpu_graph_compute`: Orchestrates the matrix multiplication, assigns workload segments to different NPU cores, and handles activation transformations.

*   **`rknpu2-configuration.cpp`**
    Defines chip-specific configurations, available hardware pipelines (data types, alignment rules, packing functions), and manages the hybrid quantization patterns.

*   **`rknpu2-quantization.cpp`**
    Contains mathematical utilities for scaling and symmetric quantization from FP32 to target NPU integer formats (INT8, INT4), as well as dequantization routines.

*   **`rknpu2-calibration.cpp`**
    Provides statistical methods for finding optimal quantization parameters (e.g., KL-Divergence, Min-MSE) and matrix transformations (like the Fast Walsh-Hadamard Transform).

*   **`rknpu2-allocation.cpp`**
    A lightweight wrapper for allocating and freeing physically contiguous, zero-copy memory via the Linux DMA-Heap subsystem.

## Future Directions

This section outlines the roadmap for future contributions and improvements.

### Ongoing Needs
Contributions in these areas are always welcome:

*   **Bug Fixes:** Any reproducible issues confirmed by the community.
*   **Optimizations:** Improvements to calculation accuracy, memory usage reduction, or inference speedups with experimental benchmark data.

### Hot Topics
Here are several "cutting-edge" ideas and active development goals:

*  **Support for Other Chipsets:** Expanding configurations in `rknpu2-configuration.cpp` to support other Rockchip SoCs, including RISC-V variants.
*  **Advanced Low-Bit Optimizations:** Currently, pure `INT4_STANDARD` produces garbage output. The Hadamard Transform solves the accuracy issue but introduces significant $O(K \log K)$ CPU overhead. Architectural optimizations, faster math routines or advanced algorithms are welcomed to solve the problem.
*  **Split Quantization:** Implementing a system where a single weight matrix is split into two: a sparse matrix containing outliers (computed in high-bit pipeline) and a dense matrix for the rest (computed in low-bit pipeline). Orchestrating this efficiently on the NPU is a major milestone.
*  **Smart Hybrid Quantization:** Currently, hybrid quantization applies a cyclical pattern across layers. Adding support for "smart" quantization-such as using Regex patterns to target specific, sensitive neural network layers with high-bit pipelines-would vastly improve the performance/accuracy ratio.

## Development Workflow

**Important:** This fork undergoes frequent `rebase` and `push --force` operations against the upstream `llama.cpp` repository to ensure compatibility with the newest architectural updates from the community. 

To prevent merge conflicts:
*   If you are working on a feature, **please open a Draft Pull Request as early as possible.** This avoids breaking your work during an upstream sync.
*   Ensure your code matches the existing C/C++ style of the project.
*   Keep your commits clean and descriptive.
