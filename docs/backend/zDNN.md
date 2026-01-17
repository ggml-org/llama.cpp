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

## CMake Options

The IBM zDNN backend has the following CMake options that control the behaviour of the backend.

| CMake Option | Default Value | Description                         |
| ------------ | ------------- | ----------------------------------- |
| `GGML_ZDNN`  | `OFF`         | Compile llama.cpp with zDNN support |
| `ZDNN_ROOT`  | `""`          | Override zDNN library lookup        |

## 1. Install zDNN Library

Note: Using the zDNN library provided via `apt` or `yum` may not work correctly as reported in [#15772](https://github.com/ggml-org/llama.cpp/issues/15772). It is preferred that you compile from source.

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
