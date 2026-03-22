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
make -j4
```

3. Run inference

```sh
# For Dense models
./build/bin/llama-cli -m ./gemma-3-1b-it-Q8_0.gguf

# For MoE models
./build/bin/llama-cli -m ./LFM2-8B-A1B-Q4_0.gguf --cpu-moe
```

## Benchmarks

The following benchmarks were conducted on an RK3588, comparing the performance, accuracy, and power consumption of the NPU backend against the standard CPU (NEON) backend.

| Model | Type | Backend | Perplexity | PP (tok/s) | TG (tok/s) | Power (W) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Granite4.0 350M** | F16 | CPU | 🟢 20.73±0.74 | 🔴 154.1±0.1 | 🟢 25.8±0.1 | 🔴 6.4±0.4 |
| | | NPU | 🟢 20.74±0.74 | 🟢 432.3±0.6 | 🟡 20.2±0.4 | 🟢 3.2±0.2 |
| | Q8_0 | CPU | 🟢 20.71±0.74 | 🔴 163.4±0.2 | 🟢 40.6±0.1 | 🔴 6.4±0.4 |
| | | NPU | 🟡 22.68±0.82 | 🟢 311.8±2.1 | 🔴 25.4±0.4 | 🟢 3.6±0.2 |
| | Q6_K | CPU | 🟢 21.55±0.77 | 🔴 78.7±0.1 | 🟢 42.5±0.1 | 🔴 6.8±0.4 |
| | | NPU | 🔴 29.52±1.07 | 🟢 142.4±1.3 | 🔴 25.8±1.2 | 🟢 3.8±0.2 |
| | Q4_0 | CPU | 🟢 24.46±0.88 | 🟢 340.4±0.9 | 🟢 55.2±0.1 | 🔴 6.2±0.4 |
| | | NPU | 🔴 74.09±2.87 | 🔴 163.6±0.2 | 🔴 26.7±0.5 | 🟢 4.0±0.2 |
| **Gemma3 1B** | F16 | CPU | 🟢 26.20±1.08 | 🔴 68.5±0.1 | 🟢 11.1±0.1 | 🔴 6.8±0.4 |
| | | NPU | 🟢 26.18±1.07 | 🟢 249.6±0.2 | 🟢 10.8±0.2 | 🟢 2.8±0.2 |
| | Q8_0 | CPU | 🟢 26.08±1.07 | 🔴 73.3±0.1 | 🟢 19.5±0.1 | 🔴 7.4±0.4 |
| | | NPU | 🟡 29.15±1.22 | 🟢 378.6±0.4 | 🟡 16.5±0.3 | 🟢 3.0±0.2 |
| | Q6_K | CPU | 🟢 25.94±1.06 | 🔴 51.9±0.1 | 🟢 18.7±0.1 | 🔴 7.2±0.4 |
| | | NPU | 🟡 38.46±1.68 | 🟢 209.4±3.2 | 🟡 16.6±0.2 | 🟢 3.0±0.2 |
| | Q4_0 | CPU | 🟢 30.77±1.31 | 🟢 164.7±0.2 | 🟢 28.3±0.1 | 🔴 7.0±0.4 |
| | | NPU | 🔴 55.53±2.30 | 🔴 51.4±0.1 | 🔴 16.7±0.3 | 🟢 3.0±0.2 |
| **LFM2 8B A1B** | F16 | CPU | 🟢 15.79±0.58 | 🟡 31.1±2.9 | 🟢 6.8±0.2 | 🟡 7.0±0.6 |
| | | NPU | 🟢 15.82±0.58 | 🟢 38.3±3.2 | 🟢 6.3±0.4 | 🟢 5.8±0.4 |
| | Q8_0 | CPU | 🟢 15.92±0.59 | 🟡 31.7±0.1 | 🟢 12.9±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 16.76±0.62 | 🟢 40.7±0.7 | 🟢 12.5±0.3 | 🟢 5.8±0.4 |
| | Q6_K | CPU | 🟢 15.91±0.58 | 🟡 16.4±0.1 | 🟡 12.9±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 21.16±0.83 | 🟢 21.4±0.1 | 🟢 13.7±0.1 | 🟢 5.8±0.4 |
| | Q4_0 | CPU | 🟢 18.24±0.53 | 🟢 62.2±0.1 | 🟢 22.7±0.1 | 🟡 7.4±0.6 |
| | | NPU | 🟡 26.09±1.06 | 🟡 47.5±0.1 | 🟢 19.0±0.1 | 🟢 5.8±0.4 |

**Legend**: 🟢 Excellent | 🟡 Acceptable | 🔴 Poor

### Methodology

*   **Perplexity:** Evaluates accuracy. Measured over 32 chunks of 512 tokens using `wiki.test.raw`. Lower is better.

    ```sh
    taskset -c 4-7 ./build/bin/llama-perplexity -m ./model.gguf -f ./wiki.test.raw -t 4 -b 512 --chunks 32
    ```

*   **PP / TG (tok/s):** Prompt Processing and Text Generation speeds. Evaluated using standard pp512 and tg128. Higher is better.

    ```sh
    taskset -c 4-7 ./build/bin/llama-bench -m ./model.gguf -t 4
    ```

*   **Power (W):** Represents relative active power consumption. Calculated as `Power(TextGeneration) - Power(Idle)`. Lower is better.

## Hybrid Quantization

Hybrid quantization is a technique designed to strike an optimal balance between memory consumption, inference speed, and generation quality.

Instead of forcing the entire model into a single format, the backend processes the model layer by layer, cyclically applying a predefined sequence of hardware pipelines. This approach is entirely transparent and works with any input GGUF weight type. You can find the default pipeline sequences for each chip in the **Chipsets** section below.

### Custom Patterns

Building on this layer-by-layer approach, you can easily override the default behavior to experiment with your own quantization strategies.

By setting the `HYBRID_PATTERN` environment variable, you define a custom sequence of hardware pipelines. The backend will cyclically iterate through this sequence as it loads the model's layers. For example, if you specify two pipelines, Layer 1 will use the first, Layer 2 will use the second, Layer 3 will revert to the first, and so on.

```sh
# Alternates model layers between FP16, INT8 and INT4 NPU pipelines
# This pipeline results in (16 + 8 + 4) / 3 ≈ 9.3 BPW
HYBRID_PATTERN="FP16_STANDARD,INT8_STANDARD,INT4_HADAMARD" ./build/bin/llama-cli -m ./model.gguf

# When the backend encounters a wrong pipeline name, it offloads layer to the default CPU backend
# This pipeline will offload to the NPU around half of the model's layers
HYBRID_PATTERN="CPU_STANDARD,INT8_STANDARD" ./build/bin/llama-cli -m ./model.gguf
```

### Weights Requantizations

It is important to understand how weights are loaded into the NPU. The backend does not execute GGUF formats (like `Q4_0` or `Q8_0`) natively. Instead, input weights are first dequantized to `FP32` on the CPU, calibrated, and then requantized into the NPU's native hardware formats. 

Because of this double conversion process, you must carefully pair your input GGUF model with your target NPU pipelines:

1. **Ideal Case (GGUF Precision > NPU Precision)**<br>
   Using an `F16` or `Q6_K` model to run `INT8` or `INT4` NPU pipelines. The NPU quantization algorithm receives highly accurate FP32 reference data, allowing it to calculate optimal scales and minimize the final quantization error.

2. **Recommended Case (GGUF Precision ≈ NPU Precision)**<br>
   Using `Q8_0` for an `INT8` pipeline, or `Q4_0` for an `INT4` pipeline. The precision levels match closely, keeping the initial GGUF file size small on disk while minimizing further information loss during NPU requantization.

3. **Terrible Case (GGUF Precision < NPU Precision)**<br>
   Upscaling a `Q4_0` model to run on an `FP16` or `INT8` NPU pipeline. Information was irreversibly lost when the model was originally compressed to `Q4_0`. Requantizing it to a higher bit-depth cannot restore the lost accuracy, meaning you will get the poor quality of a 4-bit model while wasting the memory bandwidth and compute resources of an 8-bit or 16-bit model.

## Chipsets

The backend configures operations based on hardware pipelines-specific hardware-accelerated paths mapping mathematical operations to native NPU types. Each supported chipset defines its own set of pipelines and default quantization behaviors.

### RK3588

#### Available Pipelines

Below is a comparison of all available hardware pipelines on the RK3588. 

The **Perplexity** metrics were measured on the `Granite-4.0-350M-F16` model to isolate the accuracy impact of the NPU's internal quantization algorithms.

| Name | Operation | Perplexity | Notes |
| :--- | :--- | :--- | :--- |
| `FP16_HADAMARD` | FP16xFP16 | 20.74 ± 0.74 | Hadamard Transform\* |
| `FP16_STANDARD` | FP16xFP16 | 20.74 ± 0.74 | - |
| `INT8_HADAMARD` | INT8xINT8 | 20.85 ± 0.74 | Hadamard Transform\* |
| `INT8_STANDARD` | INT8xINT8 | 22.91 ± 0.83 | - |
| `INT4_HADAMARD` | INT4xINT4 | 109.15 ± 4.46 | Hadamard\*, KL-Div\*\* |
| `INT4_STANDARD` | INT4xINT4 | 240048.97 ± 9261.03 | KL-Divergence\*\* |

\* **Hadamard Transform:** Applies a randomized Fast Walsh-Hadamard Transform to smooth out activation outliers before quantization (see [2404.00456](https://arxiv.org/abs/2404.00456)).<br>
\*\* **KL-Divergence:** Uses entropy-based calibration to find the optimal scaling factor for 4-bit weights by minimizing information loss (see [2411.02530](https://arxiv.org/abs/2411.02530)).

> **Note:** As seen in the table, pure `INT4_STANDARD` produces completely broken outputs (extremely high perplexity). This is due to massive outliers being heavily clipped in the limited 4-bit range. The `INT4_HADAMARD` pipeline mitigates this by mathematically distributing the outliers across all channels, making 4-bit inference actually usable, though it introduces some CPU overhead.

#### Default Mappings

If no custom `HYBRID_PATTERN` is provided, the RK3588 backend will automatically map your input GGUF model types to the following default NPU pipelines to provide the best out-of-the-box balance of speed and accuracy:

| Input Weight Type | Default Hardware Pipeline | Bits Per Weight |
| :--- | :--- | :--- |
| `F16` | [`FP16_STANDARD`] | 16 |
| `Q8_0` | [`INT8_STANDARD`] | 8 |
| `Q6_K` | [`INT8_STANDARD`, `INT4_HADAMARD`] | 6 |
| `Q4_0` | [`INT4_HADAMARD`] | 4 |

## Contributing

Feel free to open an issue to discuss a bug or feature, or submit a pull request with your improvements. Please refer to CONTRIBUTING.md for a breakdown of the project structure, current development goals, and workflow guidelines.
