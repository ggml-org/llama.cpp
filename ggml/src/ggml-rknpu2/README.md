# rk-llama.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

The [llama.cpp](https://github.com/ggml-org/llama.cpp) fork with the Rockchip NPU integration as a GGML backend.

## Description

This project integrates Rockchip's Neural Processing Unit as a computational backend, enabling energy-efficient, high-performance inference for Large Language Models.

The primary motivation is to leverage the NPU as a powerful accelerator. This approach aims to achieve speedups over CPU-only inference while maintaining comparable accuracy with reduced power consumption.

Currently, the backend is optimized and tested for the **RK3588** SoC. Support for other Rockchip SoCs can be added by updating the configuration file.

## Quick start

1. Clone the repository

```sh
git clone https://github.com/invisiofficial/rk-llama.cpp
cd rk-llama.cpp
```

2. Build the project

```sh
mkdir build && cd build
cmake .. -DLLAMA_RKNPU2=ON
make -j8
```

3. Run inference

```sh
# For Dense models
./build/bin/llama-cli -m ~/Projects/gemma-3-1b-it-Q8_0.gguf

# For MoE models
./build/bin/llama-cli -m ~/Projects/LFM2-8B-A1B-Q4_0.gguf --cpu-moe
```

## Quantizations

### Weights

Weights are converted based on the input type. Implemented types:

`FP16`

Input **F16** weights are directly used in native **FP16** format.

`INT8`

Input **Q8_0** weights are dequantized to FP32, then re-quantized to a uniform per-tensor **INT8** format.

`INT4`

Input **Q4_0** weights are dequantized, rotated using a randomized Hadamard transform (see [2404.00456](https://arxiv.org/abs/2404.00456)), calibrated using a KL-Divergence (see [2411.02530](https://arxiv.org/abs/2411.02530)), and then re-quantized to per-tensor **INT4**.

### Activations

Activations are converted based on the operation type. Implemented types:

`FP16`

Input **F32** activations are converted to **FP16** format.

`INT8`

Input **F32** activations are quantized to **INT8** using per-channel scaling.

`INT4`

Input **F32** activations are rotated using a Hadamard transform and then quantized to **INT4** using per-channel scaling.

### Results

Results (of a matrix multiplication) are converted based on the output type. Implemented types:

`FP32`

**FP32** results from the NPU are already in **F32** and used directly.

`INT32`

**INT32** results from the NPU are dequantized to **F32** using combined weight and activation scales.

`INT16`

**INT16** results from the NPU are dequantized to **F32** with an additional normalization factor for rotated computations.

## Chipsets

### RK3588

The backend supports the following computation types:
*   **W16A16**: FP16 weights & FP16 activations
*   **W8A8**: INT8 weights & INT8 activations
*   **W4A4**: INT4 weights & INT4 activations

## Benchmarks

The following benchmarks were conducted on an RK3588, comparing the performance, accuracy, and power consumption of the NPU backend against the standard CPU (NEON) backend.

| Model | Type | Backend | Perplexity | PP (tok/s) | TG (tok/s) | Power (W) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Granite4.0 350M** | F16 | CPU | 🟢 20.73±0.74 | 🔴 154.1±0.1 | 🟢 25.8±0.1 | 🔴 6.4±0.4 |
| | | NPU | 🟢 20.74±0.74 | 🟢 432.3±0.6 | 🟡 20.2±0.4 | 🟢 3.2±0.2 |
| | Q8_0 | CPU | 🟢 20.71±0.74 | 🔴 163.4±0.2 | 🟢 40.6±0.1 | 🔴 6.4±0.4 |
| | | NPU | 🟡 22.68±0.82 | 🟢 311.8±2.1 | 🔴 25.4±0.4 | 🟢 3.6±0.2 |
| | Q4_0 | CPU | 🟢 24.46±0.88 | 🟢 340.4±0.9 | 🟢 55.2±0.1 | 🔴 6.2±0.4 |
| | | NPU | 🔴 74.09±2.87 | 🔴 163.6±0.2 | 🔴 26.7±0.5 | 🟢 4.0±0.2 |
| **Gemma3 1B** | F16 | CPU | 🟢 26.20±1.08 | 🔴 68.5±0.1 | 🟢 11.1±0.1 | 🔴 6.8±0.4 |
| | | NPU | 🟢 26.18±1.07 | 🟢 249.6±0.2 | 🟢 10.8±0.2 | 🟢 2.8±0.2 |
| | Q8_0 | CPU | 🟢 26.08±1.07 | 🔴 73.3±0.1 | 🟢 19.5±0.1 | 🔴 7.4±0.4 |
| | | NPU | 🟡 29.15±1.22 | 🟢 378.6±0.4 | 🟡 16.5±0.3 | 🟢 3.0±0.2 |
| | Q4_0 | CPU | 🟢 30.77±1.31 | 🟢 164.7±0.2 | 🟢 28.3±0.1 | 🔴 7.0±0.4 |
| | | NPU | 🔴 55.53±2.30 | 🔴 51.4±0.1 | 🔴 16.7±0.3 | 🟢 3.0±0.2 |
| **LFM2 8B A1B** | F16 | CPU | 🟢 15.79±0.58 | 🟡 31.1±2.9 | 🟢 6.8±0.2 | 🟡 7.0±0.6 |
| | | NPU | 🟢 15.82±0.58 | 🟢 38.3±3.2 | 🟢 6.3±0.4 | 🟢 5.8±0.4 |
| | Q8_0 | CPU | 🟢 15.92±0.59 | 🟡 31.7±0.1 | 🟢 12.9±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 16.76±0.62 | 🟢 40.7±0.7 | 🟢 12.5±0.3 | 🟢 5.8±0.4 |
| | Q4_0 | CPU | 🟢 18.24±0.53 | 🟢 62.2±0.1 | 🟢 22.7±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 26.09±1.06 | 🟡 47.5±0.1 | 🟢 19.0±0.1 | 🟢 5.8±0.4 |

## Contributing

Feel free to open an issue to discuss a bug or feature, or submit a pull request with your improvements.
