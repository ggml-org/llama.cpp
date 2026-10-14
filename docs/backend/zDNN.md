# llama.cpp for IBM zDNN Accelerator

> [!WARNING]
> **Note:** zDNN is **not** the same as ZenDNN.
> - **zDNN** (this page): IBM's Deep Neural Network acceleration library for IBM Z & LinuxONE Mainframes
> - **ZenDNN**: AMD's deep learning library for AMD EPYC CPUs ([see ZenDNN documentation](ZenDNN.md))

## Background

IBM zDNN (Z Deep Neural Network) is a hardware acceleration library designed specifically to leverage the IBM NNPA (Neural Network Processor Assist) accelerator located within IBM Telum I and II processors. It provides significant performance improvements for neural network inference operations.

### Llama.cpp + IBM zDNN

The llama.cpp zDNN backend is designed to enable llama.cpp on IBM z17 and later systems via the IBM zDNN hardware acceleration library.

## Software & Hardware Support

| Hardware Level       | Status        | Verified                   |
| -------------------- | ------------- | -------------------------- |
| IBM z17 / LinuxONE 5 | Supported     | RHEL 9.6, IBM z17, 40 IFLs |
| IBM z16 / LinuxONE 4 | Not Supported |                            |

## Data Types Supported

| Data Type | Status    |
| --------- | --------- |
| F32       | Supported |
| F16       | Supported |
| BF16      | Supported |
| Q8_0      | Supported |

Note: Quantized weights are requantized to int8 with one scale per group of up to 2048 values in each row (or 1024 with `GGML_ZDNN_HIGH_PREC=1`), after each input channel is divided by its RMS (the activations are multiplied by it instead), and each activation is quantized to two int8 values (about 15 bits). This requires `NNPA_PARMBLKFORMAT_1` (IBM z17 / LinuxONE 5).

## CMake Options

The IBM zDNN backend has the following CMake options that control the behaviour of the backend.

| CMake Option | Default Value | Description                         |
| ------------ | ------------- | ----------------------------------- |
| `GGML_ZDNN`  | `OFF`         | Compile llama.cpp with zDNN support |
| `ZDNN_ROOT`  | `""`          | Override zDNN library lookup        |

## Environment Variables

| Environment Variable  | Default Value | Description                                                                            |
| --------------------- | ------------- | -------------------------------------------------------------------------------------- |
| `GGML_ZDNN_HIGH_PREC` | `0`           | Set to `1` to requantize Q8_0 weights in smaller groups, for lower loss at lower speed |

## 1. Install zDNN Library

Note: A minimum of libzdnn 1.1.2 is required for the IBM zDNN backend to link correctly.

### Install using apt

```sh
sudo apt update
sudo apt install -y libzdnn-dev
```

### Install using yum

```sh
sudo yum update
sudo yum install -y libzdnn-devel
```

### Install from source

```sh
git clone --recurse-submodules https://github.com/IBM/zDNN
cd zDNN

autoreconf .
./configure --prefix=/opt/zdnn-libs

make build
sudo make install
```

## 2. Build llama.cpp

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -S . -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_ZDNN=ON \
    -DZDNN_ROOT=/opt/zdnn-libs
cmake --build build --config Release -j$(nproc)
```

## 3. Run a model

```sh
build/bin/llama-server -hf taronaeo/Granite-3.3-2B-Instruct-BE-GGUF:Q4_K_M -fa on
```

Note: FlashAttention (`-fa on`) is strongly recommended as it forces attention to run on the IBM zAIU accelerator instead of the CPU ([see PR](https://github.com/ggml-org/llama.cpp/pull/30512)). By default, FlashAttention is turned off.
