# llama.cpp for CANN

 - [Background](#background)
 - [News](#news)
 - [OS](#os)
 - [Hardware](#hardware)
 - [Model Supports](#model-supports)
 - [DataType Supports](#datatype-supports)
 - [Docker](#docker)
 - [Linux](#linux)
 - [Environment variable setup](#environment-variable-setup)
 - [TODO](#todo)


## Background

**Ascend NPU** is a range of AI processors using Neural Processing Unit. It will efficiently handle matrix-matrix multiplication, dot-product and scalars.

**CANN** (Compute Architecture for Neural Networks) is a heterogeneous computing architecture for AI scenarios, providing support for multiple AI frameworks on the top and serving AI processors and programming at the bottom. It plays a crucial role in bridging the gap between upper and lower layers, and is a key platform for improving the computing efficiency of Ascend AI processors. Meanwhile, it offers a highly efficient and easy-to-use programming interface for diverse application scenarios, allowing users to rapidly build AI applications and services based on the Ascend platform.

**Llama.cpp + CANN**

The llama.cpp CANN backend is designed to support Ascend NPU. It utilize the ability of AscendC and ACLNN which are intergrated to CANN Toolkit and kernels to using Ascend NPU directly.

## News

- 2024.11
  - Support F16 and F32 data type model for Ascend 310P NPU.
- 2024.8
  - Support `Q4_0` and `Q8_0` data type for Ascend NPU.
- 2024.7
  - Create CANN backend for Ascend NPU.

## OS

| OS      | Status  | Verified                                       |
|:-------:|:-------:|:----------------------------------------------:|
| Linux   | Support | Ubuntu 22.04, OpenEuler22.03                   |


## Hardware

### Ascend NPU

**Verified devices**

| Ascend NPU                    | Status  |
|:-----------------------------:|:-------:|
| Atlas 300T A2                 | Support |
| Atlas 300I Duo                | Support |

*Notes:*

- If you have trouble with Ascend NPU device, please create a issue with **[CANN]** prefix/tag.
- If you run successfully with your Ascend NPU device, please help update the upper table.


## Model Supports

| Model Name                  | FP16  | Q4_0 | Q8_0 |
|:----------------------------|:-----:|:----:|:----:|
| Llama-2                     |   √   |   √  |   √  |
| Llama-3                     |   √   |   √  |   √  |
| Mistral-7B                  |   √   |   √  |   √  |
| Mistral MOE                 |   √   |   √  |   √  |
| DBRX                        |   -   |   -  |   -  |
| Falcon                      |   √   |   √  |   √  |
| Chinese LLaMA/Alpaca        |   √   |   √  |   √  |
| Vigogne(French)             |   √   |   √  |   √  |
| BERT                        |   x   |   x  |   x  |
| Koala                       |   √   |   √  |   √  |
| Baichuan                    |   √   |   √  |   √  |
| Aquila 1 & 2                |   √   |   √  |   √  |
| Starcoder models            |   √   |   √  |   √  |
| Refact                      |   √   |   √  |   √  |
| MPT                         |   √   |   √  |   √  |
| Bloom                       |   √   |   √  |   √  |
| Yi models                   |   √   |   √  |   √  |
| stablelm models             |   √   |   √  |   √  |
| DeepSeek models             |   x   |   x  |   x  |
| Qwen models                 |   √   |   √  |   √  |
| PLaMo-13B                   |   √   |   √  |   √  |
| Phi models                  |   √   |   √  |   √  |
| PhiMoE                      |   √   |   √  |   √  |
| GPT-2                       |   √   |   √  |   √  |
| Orion                       |   √   |   √  |   √  |
| InternlLM2                  |   √   |   √  |   √  |
| CodeShell                   |   √   |   √  |   √  |
| Gemma                       |   √   |   √  |   √  |
| Mamba                       |   √   |   √  |   √  |
| Xverse                      |   √   |   √  |   √  |
| command-r models            |   √   |   √  |   √  |
| Grok-1                      |   -   |   -  |   -  |
| SEA-LION                    |   √   |   √  |   √  |
| GritLM-7B                   |   √   |   √  |   √  |
| OLMo                        |   √   |   √  |   √  |
| OLMo 2                      |   √   |   √  |   √  |
| OLMoE                       |   √   |   √  |   √  |
| Granite models              |   √   |   √  |   √  |
| GPT-NeoX                    |   √   |   √  |   √  |
| Pythia                      |   √   |   √  |   √  |
| Snowflake-Arctic MoE        |   -   |   -  |   -  |
| Smaug                       |   √   |   √  |   √  |
| Poro 34B                    |   √   |   √  |   √  |
| Bitnet b1.58 models         |   √   |   x  |   x  |
| Flan-T5                     |   √   |   √  |   √  |
| Open Elm models             |   x   |   √  |   √  |
| chatGLM3-6B + ChatGLM4-9b +  GLMEdge-1.5b + GLMEdge-4b    |   √   |   √  |   √  |
| GLM-4-0414                  |   √   |   √  |   √  |
| SmolLM                      |   √   |   √  |   √  |
| EXAONE-3.0-7.8B-Instruct    |   √   |   √  |   √  |
| FalconMamba Models          |   √   |   √  |   √  |
| Jais Models                 |   -   |   x  |   x  |
| Bielik-11B-v2.3             |   √   |   √  |   √  |
| RWKV-6                      |   -   |   √  |   √  |
| QRWKV-6                     |   √   |   √  |   √  |
| GigaChat-20B-A3B            |   x   |   x  |   x  |
| Trillion-7B-preview         |   √   |   √  |   √  |
| Ling models                 |   √   |   √  |   √  |


**Multimodal**
| Model Name                  | FP16  | Q4_0 | Q8_0 |
|:----------------------------|:-----:|:----:|:----:|
| LLaVA 1.5 models, LLaVA 1.6 models      |   x   |   x  |   x  |
|  BakLLaVA                   |   √   |   √  |   √  |
|  Obsidian                   |   √   |   -  |   -  |
|  ShareGPT4V                 |   x   |   -  |   -  |
|  MobileVLM 1.7B/3B models   |   -   |   -  |   -  |
|  Yi-VL                      |   -   |   -  |   -  |
|  Mini CPM                   |   √   |   √  |   √  |
|  Moondream                  |   √   |   √  |   √  |
|  Bunny                      |   √   |   -  |   -  |
|  GLM-EDGE                   |   √   |   √  |   √  |
|  Qwen2-VL                   |   √   |   √  |   √  |



## DataType Supports

| DataType               | Status  |
|:----------------------:|:-------:|
| FP16                   | Support |
| Q8_0                   | Support |
| Q4_0                   | Support |

## Docker

### Build Images
You can get a image with llama.cpp in one command.
```sh
docker build -t llama-cpp-cann -f .devops/llama-cli-cann.Dockerfile .
```

### Run container

```sh
# Find all cards.
npu-smi info

# Select the cards that you want to use, make sure these cards are not used by someone.
# Following using cards of device0.
docker run --name llamacpp --device /dev/davinci0  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /PATH_TO_YOUR_MODELS/:/app/models -it llama-cpp-cann -m /app/models/MODEL_PATH -ngl 32 -p "Building a website can be done in 10 simple steps:"
```

*Notes:*

- You may need to install Ascend Driver and firmware on the **host** machine *(Please refer to the [Linux configuration](#linux) for details)*.

## Linux

### I. Setup Environment

1. **Install Ascend Driver and firmware**

    ```sh
    # create driver running user.
    sudo groupadd -g HwHiAiUser
    sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    sudo usermod -aG HwHiAiUser $USER

    # download driver from https://www.hiascend.com/hardware/firmware-drivers/community according to your system
    # and install driver.
    sudo sh Ascend-hdk-910b-npu-driver_x.x.x_linux-{arch}.run --full --install-for-all
    ```

    Once installed, run `npu-smi info` to check whether driver is installed successfully.
    ```sh
    +-------------------------------------------------------------------------------------------+
    | npu-smi 24.1.rc2               Version: 24.1.rc2                                          |
    +----------------------+---------------+----------------------------------------------------+
    | NPU   Name           | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
    | Chip                 | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
    +======================+===============+====================================================+
    | 2     xxx            | OK            | 64.4        51                15   / 15            |
    | 0                    | 0000:01:00.0  | 0           1873 / 15077      0    / 32768         |
    +======================+===============+====================================================+
    | 5     xxx            | OK            | 64.0        52                15   / 15            |
    | 0                    | 0000:81:00.0  | 0           1874 / 15077      0    / 32768         |
    +======================+===============+====================================================+
    | No running processes found in NPU 2                                                       |
    +======================+===============+====================================================+
    | No running processes found in NPU 5                                                       |
    +======================+===============+====================================================+
    ```

2. **Install Ascend Firmware**
    ```sh
    # download driver from https://www.hiascend.com/hardware/firmware-drivers/community according to your system
    # and install driver.
    sudo sh Ascend-hdk-910b-npu-firmware_x.x.x.x.X.run --full
    ```
    If the following messaage appers, firmware is installed successfully.
    ```sh
    Firmware package installed successfully!
    ```


3. **Install CANN toolkit and kernels**

    CANN toolkit and kernels can be obtained from the official [CANN Toolkit](https://www.hiascend.com/zh/developer/download/community/result?module=cann) page.

    Please download the corresponding version that satified your system. The minimum version required is 8.0.RC2.alpha002 and here is the install command.
    ```sh
    pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
    sh Ascend-cann-toolkit_8.0.RC2.alpha002_linux-aarch64.run --install
    sh Ascend-cann-kernels-910b_8.0.RC2.alpha002_linux.run --install
    ```

    Set Ascend Variables:
    ```sh
    echo "source ~/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
    source ~/.bashrc
    ```

Upon a successful installation, CANN is enabled for the available ascend devices.

### II. Build llama.cpp

```sh
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
cmake --build build --config release
```

### III. Run the inference

1. **Retrieve and prepare model**

    You can refer to the general [*Prepare and Quantize*](../../README.md#prepare-and-quantize) guide for model prepration.

    **Notes**:

      - CANN backend only supports FP16/Q4_0/Q8_0 models currently.

2. **Launch inference**

    There are two device selection modes:

    - Single device: Use one device target specified by the user.
    - Multiple devices: Automatically choose the devices with the same backend.

    | Device selection | Parameter                              |
    |:----------------:|:--------------------------------------:|
    | Single device    | --split-mode none --main-gpu DEVICE_ID |
    | Multiple devices | --split-mode layer (default)           |

    Examples:

    - Use device 0:

    ```sh
    ./build/bin/llama-cli -m path_to_model -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm none -mg 0
    ```

    - Use multiple devices:

    ```sh
    ./build/bin/llama-cli -m path_to_model -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm layer
    ```

### **GitHub contribution**:
Please add the **[CANN]** prefix/tag in issues/PRs titles to help the CANN-team check/address them without delay.

## Updates
### Basic Flash Attention Support
The basic FA kernel with aclnnops has been added in aclnn_ops.cpp.
Currently, the FA only supports the cases with FP16 KV tensors and NO logit softcap.
Since the aclnn interface for flash attention cannot support the logit softcap, we will only update the quantized version in the future.

Authors from Peking University: Bizhao Shi (bshi@pku.edu.cn), Yuxin Yang (yxyang@pku.edu.cn), Ruiyang Ma (ruiyang@stu.pku.edu.cn), and Guojie Luo (gluo@pku.edu.cn).

We would like to thank Tuo Dai, Shanni Li, and all of the project maintainers from Huawei Technologies Co., Ltd for their help during the code development and pull request.

## Environment variable setup

### GGML_CANN_MEM_POOL

Specifies the memory pool management strategy, Default is vmm.

- vmm: Utilizes a virtual memory manager pool. If hardware support for VMM is unavailable, falls back to the legacy (leg) memory pool.

- prio: Employs a priority queue-based memory pool management.

- leg: Uses a fixed-size buffer pool.

### GGML_CANN_DISABLE_BUF_POOL_CLEAN

Controls automatic cleanup of the memory pool. This option is only effective when using the prio or leg memory pool strategies.

### GGML_CANN_WEIGHT_NZ

Converting the matmul weight format from ND to NZ to improve performance. Enabled by default.

### GGML_CANN_ACL_GRAPH

Operators are executed using ACL graph execution, rather than in op-by-op (eager) mode. Enabled by default. This option is only effective if `USE_ACL_GRAPH` was enabled at compilation time. To enable it, recompile using:

```sh
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release -DUSE_ACL_GRAPH=ON
cmake --build build --config release
```

### GGML_CANN_GRAPH_CACHE_CAPACITY

Maximum number of compiled CANN graphs kept in the LRU cache, default is 12. When the number of cached graphs exceeds this capacity, the least recently used graph will be evicted.

### GGML_CANN_PREFILL_USE_GRAPH

Enable ACL graph execution during the prefill stage, default is false. This option is only effective when FA is enabled.
