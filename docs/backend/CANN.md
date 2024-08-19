# llama.cpp for CANN

 - [Background](#background)
 - [News](#news)
 - [OS](#os)
 - [Hardware](#hardware)
 - [Model Supports](#model-supports)
 - [DataType Supports](#datatype-supports)
 - [Docker](#docker)
 - [Linux](#linux)
 - [TODO](#todo)


## Background

**Ascend NPU** is a range of AI processors using Neural Processing Unit. It will efficiently handle matrix-matrix multiplication, dot-product and scalars.

**CANN** (Compute Architecture for Neural Networks) is a heterogeneous computing architecture for AI scenarios, providing support for multiple AI frameworks on the top and serving AI processors and programming at the bottom. It plays a crucial role in bridging the gap between upper and lower layers, and is a key platform for improving the computing efficiency of Ascend AI processors. Meanwhile, it offers a highly efficient and easy-to-use programming interface for diverse application scenarios, allowing users to rapidly build AI applications and services based on the Ascend platform.

**Llama.cpp + CANN**

The llama.cpp CANN backend is designed to support Ascend NPU. It utilize the ability of AscendC and ACLNN which are intergrated to CANN Toolkit and kernels to using Ascend NPU directly.

## News

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

*Notes:*

- If you have trouble with Ascend NPU device, please create a issue with **[CANN]** prefix/tag.
- If you run successfully with your Ascend NPU device, please help update the upper table.


## Model Supports

| Model Name                  | FP16  | Q8_0 | Q4_0 |
|:----------------------------|:-----:|:----:|:----:|
| AquilaChat2-7B              |   √   |   √  |   √  |
| Baichuan-7b                 |   √   |   √  |   √  |
| Baichuan2-7B-Chat           |   √   |   √  |   √  |
| bitnet_b1_58-large          |   √   |   √  |   √  |
| bloom-560m                  |   √   |   x  |   √  |
| bloomz-alpaca-560m          |   √   |   x  |   √  |
| c4ai-command-r-35B-v01      |   x   |   x  |   x  |
| chatglm3-6B                 |   x   |   x  |   x  |
| chinese-alpaca-2-1.3b       |   √   |   √  |   √  |
| CodeShell-7B                |   √   |   √  |   √  |
| deepseek-ai_deepseek-coder-1.3B-base | x |   x  |   x  |
| deepseek-ai_DeepSeek-V2-Lite | x   |   x  |   x   |
| deepseek-coder-6.7B-instruct | x   |   x  |   x   |
| DeepSeek-V2-Lite-64x1.5B    |   x   |   x  |   x  |
| falcon-7b-instruct          |   √   |   √  |   √  |
| flan-t5-large               |   √   |   √  |   √  |
| gemma-2-9b-it               |   √   |   √  |   √  |
| glm-4-9B                    |   x   |   x  |   x  |
| gpt2                        |   √   |   √  |   √  |
| Gpt2-163M                   |   √   |   √  |   √  |
| granite-3B-code-instruct    |   √   |   √  |   √  |
| GritLM-7B                   |   √   |   √  |   √  |
| internlm2_5-7b-chat         |   √   |   √  |   √  |
| koala-7B-HF                 |   √   |   √  |   √  |
| Llama-2-7b-chat-hf          |   √   |   √  |   √  |
| Llama-3-Smaug-8B            |   √   |   √  |   √  |
| Llama2-Chinese-7b-Chat      |   √   |   √  |   √  |
| Llama3-8B                   |   √   |   √  |   √  |
| Llama3-8b-chinese           |   √   |   √  |   √  |
| mamba-130m-hf               |   √   |   √  |   √  |
| Mistral-7B-Instruct-v0.2    |   √   |   √  |   √  |
| Mixtral-8x7B-Instruct-v0.1  |   x   |   √  |   √  |
| mpt-7B                      |   √   |   √  |   √  |
| OLMo-1B-hf                  |   √   |   √  |   √  |
| OpenELM-3B-Instruct         |   √   |   √  |   √  |
| Orion-14b-base              |   √   |   √  |   √  |
| phi1                        |   x   |   x  |   x  |
| phi2                        |   x   |   x  |   x  |
| Phi-3-mini-4k-instruct      |   √   |   √  |   √  |
| plamo-13b                   |   √   |   √  |   √  |
| pythia-70M                  |   x   |   x  |   x  |
| Qwen-7B                     |   √   |   √  |   √  |
| Qwen2-1.5B-Instruct         |   √   |   x  |   √  |
| Refact-1_6B-fim             |   √   |   √  |   √  |
| SmolLM-135M                 |   √   |   √  |   √  |
| stablelm-zephyr             |   x   |   x  |   x  |
| stablelm-2-zephyr-1_6b      |   x   |   x  |   x  |
| starcoderbase-1b            |   √   |   √  |   √  |
| starcoder2-3b               |   √   |   √  |   √  |
| vigogne-7b-chat             |   √   |   √  |   √  |
| xverse-7b-chat              |   √   |   √  |   √  |
| Yi-6b-Chat                  |   √   |   √  |   √  |



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


## TODO
- Support more models and data types.
