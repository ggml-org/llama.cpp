# Token generation performance troubleshooting

## Verifying that the model is running on the GPU
Make sure you compiled llama with a GPU backend according to [this guide](/docs/build/build.md#sycl) (SYCL, or [Vulkan](/docs/build/build.md#vulkan)), so that the `-ngl N` (or `--n-gpu-layers N`) flag can offload layers. `N` also accepts `auto` (the default) and `all`. When running llama, you may configure `N` to be very large, and llama will offload the maximum possible number of layers to the GPU, even if it's less than the number you configured. For example:
```shell
./llama-cli -m "path/to/model.gguf" -ngl 200000 -p "Please sir, may I have some "
```

When running llama, before it starts the inference work, it will output diagnostic information that shows whether layers are offloaded to the GPU. Look for lines like these (`SYCL0` is the first SYCL device; Vulkan reports `Vulkan0`):
```shell
load_tensors: offloading output layer to GPU
load_tensors: offloading <n> repeating layers to GPU
load_tensors: offloaded <n>/<total> layers to GPU
load_tensors:        SYCL0 model buffer size = <size> MiB
... rest of inference
```

If you see these lines, then the GPU is being used.

## Verifying that the CPU is not oversaturated
llama accepts a `-t N` (or `--threads N`) parameter. It's extremely important that this parameter is not too large. If your token generation is extremely slow, try setting this number to 1. If this significantly improves your token generation speed, then your CPU is being oversaturated and you need to explicitly set this parameter to the number of the physical CPU cores on your machine (even if you utilize a GPU). If in doubt, start with 1 and double the amount until you hit a performance bottleneck, then scale the number down.

# Example of runtime flags effect on inference speed benchmark
These upstream runs used a CUDA build, a backend this fork does not ship. They are kept to illustrate the effect of `-t` with `-ngl`; the numbers were not measured on this fork. They were tested on the following machine:
GPU: A6000 (48GB VRAM)
CPU: 7 physical cores
RAM: 32GB

Model: `TheBloke_Wizard-Vicuna-30B-Uncensored-GGML/Wizard-Vicuna-30B-Uncensored.q4_0.gguf` (30B parameters, 4bit quantization, GGML)

Run command: `./llama-cli -m "path/to/model.gguf" -p "An extremely detailed description of the 10 best ethnic dishes will follow, with recipes: " -n 1000 [additional benchmark flags]`

Result:

| command | tokens/second (higher is better) |
| - | - |
| -ngl 2000000 | N/A (less than 0.1) |
| -t 7 | 1.7 |
| -t 1 -ngl 2000000 | 5.5 |
| -t 7 -ngl 2000000 | 8.7 |
| -t 4 -ngl 2000000 | 9.1 |
