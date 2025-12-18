# llama.cpp/example/sycl

This example program provides the tools for llama.cpp for SYCL on Intel GPU.

## Tool

|Tool Name| Function|Status|
|-|-|-|
|llama-ls-sycl-device| List all SYCL devices with ID, compute capability, max work group size, ect.|Support|

### llama-ls-sycl-device

List all SYCL devices with ID, compute capability, max work group size, ect.

1. Build the llama.cpp for SYCL for the specified target *(using GGML_SYCL_TARGET)*.

2. Enable oneAPI running environment *(if GGML_SYCL_TARGET is set to INTEL -default-)*

```
source /opt/intel/oneapi/setvars.sh
```

3. Execute

```
./build/bin/llama-ls-sycl-device
```

Check the ID in startup log, like:

```
found 2 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Arc A770 Graphics|    1.3|    512|    1024|   32| 16225M|            1.3.29138|
| 1| [level_zero:gpu:1]|                 Intel UHD Graphics 750|    1.3|     32|     512|   32| 62631M|            1.3.29138|

```

