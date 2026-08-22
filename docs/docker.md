# Docker

## Prerequisites
* Docker must be installed and running on your system.
* Create a folder to store big models & intermediate files (ex. /llama/models)

## Images
We have three Docker images available for this project:

1. `ghcr.io/ggml-org/llama.cpp:full`: This image includes both the `llama-cli` and `llama-completion` executables and the tools to convert LLaMA models into ggml and convert into 4-bit quantization. (platforms: `linux/amd64`, `linux/arm64`, `linux/s390x`)
2. `ghcr.io/ggml-org/llama.cpp:light`: This image only includes the `llama-cli` and `llama-completion` executables. (platforms: `linux/amd64`, `linux/arm64`, `linux/s390x`)
3. `ghcr.io/ggml-org/llama.cpp:server`: This image only includes the `llama-server` executable. (platforms: `linux/amd64`, `linux/arm64`, `linux/s390x`)

Additionally, there the following images, similar to the above:

- `ghcr.io/ggml-org/llama.cpp:full-cuda`: Same as `full` but compiled with CUDA 12 support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:full-cuda13`: Same as `full` but compiled with CUDA 13 support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:light-cuda`: Same as `light` but compiled with CUDA 12 support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:light-cuda13`: Same as `light` but compiled with CUDA 13 support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:server-cuda`: Same as `server` but compiled with CUDA 12 support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:server-cuda13`: Same as `server` but compiled with CUDA 13 support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:full-rocm`: Same as `full` but compiled with ROCm support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:light-rocm`: Same as `light` but compiled with ROCm support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:server-rocm`: Same as `server` but compiled with ROCm support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:full-musa`: Same as `full` but compiled with MUSA support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:light-musa`: Same as `light` but compiled with MUSA support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:server-musa`: Same as `server` but compiled with MUSA support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:full-intel`: Same as `full` but compiled with SYCL support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:light-intel`: Same as `light` but compiled with SYCL support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:server-intel`: Same as `server` but compiled with SYCL support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:full-vulkan`: Same as `full` but compiled with Vulkan support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:light-vulkan`: Same as `light` but compiled with Vulkan support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:server-vulkan`: Same as `server` but compiled with Vulkan support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggml-org/llama.cpp:full-openvino`: Same as `full` but compiled with OpenVino support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:light-openvino`: Same as `light` but compiled with OpenVino support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:server-openvino`: Same as `server` but compiled with OpenVino support. (platforms: `linux/amd64`)
- `ghcr.io/ggml-org/llama.cpp:full-s390x`: Identical to `full`, an alias for the `s390x` platform. (platforms: `linux/s390x`)
- `ghcr.io/ggml-org/llama.cpp:light-s390x`: Identical to `light`, an alias for the `s390x` platform. (platforms: `linux/s390x`)
- `ghcr.io/ggml-org/llama.cpp:server-s390x`: Identical to `server`, an alias for the `s390x` platform. (platforms: `linux/s390x`)

The GPU enabled images are not currently tested by CI beyond being built. They are not built with any variation from the ones in the Dockerfiles defined in [.devops/](../.devops/) and the GitHub Action defined in [.github/workflows/docker.yml](../.github/workflows/docker.yml). If you need different settings (for example, a different CUDA, ROCm or MUSA library, you'll need to build the images locally for now).

## Usage

The easiest way to download the models, convert them to ggml and optimize them is with the --all-in-one command which includes the full docker image.

Replace `/path/to/models` below with the actual path where you downloaded the models.

```bash
docker run -v /path/to/models:/models ghcr.io/ggml-org/llama.cpp:full --all-in-one "/models/" 7B
```

On completion, you are ready to play!

```bash
docker run -v /path/to/models:/models ghcr.io/ggml-org/llama.cpp:full --run -m /models/7B/ggml-model-q4_0.gguf
docker run -v /path/to/models:/models ghcr.io/ggml-org/llama.cpp:full --run-legacy -m /models/32B/ggml-model-q8_0.gguf -no-cnv -p "Building a mobile app can be done in 15 steps:" -n 512
```

or with a light image:

```bash
docker run -v /path/to/models:/models --entrypoint /app/llama-cli ghcr.io/ggml-org/llama.cpp:light -m /models/7B/ggml-model-q4_0.gguf
docker run -v /path/to/models:/models --entrypoint /app/llama-completion ghcr.io/ggml-org/llama.cpp:light -m /models/32B/ggml-model-q8_0.gguf -no-cnv -p "Building a mobile app can be done in 15 steps:" -n 512
```

or with a server image:

```bash
docker run -v /path/to/models:/models -p 8080:8080 ghcr.io/ggml-org/llama.cpp:server -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512
```

In the above examples, `--entrypoint /app/llama-cli` is specified for clarity, but you can safely omit it since it's the default entrypoint in the container.

## Docker With CUDA

Assuming one has the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) properly installed on Linux, or is using a GPU enabled cloud, `cuBLAS` should be accessible inside the container.

## Building Docker locally

```bash
docker build -t local/llama.cpp:full-cuda --target full -f .devops/cuda.Dockerfile .
docker build -t local/llama.cpp:light-cuda --target light -f .devops/cuda.Dockerfile .
docker build -t local/llama.cpp:server-cuda --target server -f .devops/cuda.Dockerfile .
```

You may want to pass in some different `ARGS`, depending on the CUDA environment supported by your container host, as well as the GPU architecture.

The defaults are:

- `CUDA_VERSION` set to `12.8.1`
- `CUDA_DOCKER_ARCH` set to the cmake build default, which includes all the supported architectures

The resulting images, are essentially the same as the non-CUDA images:

1. `local/llama.cpp:full-cuda`: This image includes both the `llama-cli` and `llama-completion` executables and the tools to convert LLaMA models into ggml and convert into 4-bit quantization.
2. `local/llama.cpp:light-cuda`: This image only includes the `llama-cli` and `llama-completion` executables.
3. `local/llama.cpp:server-cuda`: This image only includes the `llama-server` executable.

## Usage

After building locally, Usage is similar to the non-CUDA examples, but you'll need to add the `--gpus` flag. You will also want to use the `--n-gpu-layers` flag.

```bash
docker run --gpus all -v /path/to/models:/models local/llama.cpp:full-cuda --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/llama.cpp:light-cuda -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/llama.cpp:server-cuda -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512 --n-gpu-layers 1
```

## Docker With ROCm

The ROCm images are built from `rocm/dev-ubuntu-26.04:7.14.0-full` and contain the ROCm 7.14.0 user-space runtime.

The `amdgpu` kernel module and the GPU firmware are **not** part of the image — they always come from the host, and they have to be recent enough for the ROCm version running inside the container. Either:

- install AMD's packages on the host (`amdgpu-install --usecase=dkms`, or the `amdgpu-dkms` package), matching the same ROCm release, or
- if you would rather not use AMD's repositories, run an upstream kernel new enough to support your GPU together with a current [`linux-firmware`](https://gitlab.com/kernel-firmware/linux-firmware).

A host driver or firmware that is too old typically shows up as the GPU not being listed at all, rather than as an explicit error. Check the host side before involving llama.cpp:

```bash
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add $(getent group video | cut -d: -f3) \
    --group-add $(getent group render | cut -d: -f3) \
    --entrypoint rocminfo ghcr.io/ggml-org/llama.cpp:server-rocm
```

Note that the group IDs have to be passed numerically: the images do not define `video` and `render` groups, so `--group-add render` fails to resolve.

### Supported GPU architectures

ROCm 7.14.0 ships an optimized library package for every architecture below, but the prebuilt images do not cover all of them: building a fat binary for the full set exceeds the CI time budget, so the images are limited to the discrete GPUs plus the fastest APUs, Strix Point (`gfx1150`) and Strix Halo (`gfx1151`). The smaller APUs are several times slower, and remain supported targets that you build locally — see [Building Docker locally](#building-docker-locally-1).

The "Prebuilt" column says whether a target is included in the published images. Parentheses group SKU variants that share a target, so `RX 6700(XT)` covers both the RX 6700 and the RX 6700 XT.

| Family | Target | Prebuilt | Products |
| --- | --- | --- | --- |
| CDNA (server & data center accelerators) | `gfx908` | yes | Instinct MI100 |
| | `gfx90a` | yes | Instinct MI210, MI250(X) |
| | `gfx942` | yes | Instinct MI300A, MI300X(-HF), MI308X, MI325X |
| | `gfx950` | yes | Instinct MI350(X/P), MI355X |
| RDNA 1 (Navi 1X — consumer & workstation) | `gfx1010` | yes | Radeon RX 5700(XT/M), RX 5600(XT/OEM/M), Pro W5700(X), Pro 5700(XT) |
| | `gfx1011` | yes | Radeon Pro 5600M, Pro V520, Pro V540 |
| | `gfx1012` | yes | Radeon RX 5500(XT/OEM/M), RX 5300(XT OEM/M), Pro W5500(X/M), Pro 5500(XT/M), Pro 5300(M) |
| RDNA 2 (Navi 2X — consumer, workstation & APUs) | `gfx1030` | yes | Radeon RX 6950 XT, RX 6900 XT, RX 6800(XT), Pro W6800(X), Pro W6800X Duo, Pro W6900X, Pro V620 |
| | `gfx1031` | yes | Radeon RX 6750(XT/GRE), RX 6700(XT/M), RX 6850M XT, RX 6800M, Pro W6700 |
| | `gfx1032` | yes | Radeon RX 6650(XT/M/M XT), RX 6600(XT/M/S), RX 6800S, RX 6700S, Pro W6600(X/M) |
| | `gfx1033` | no | Steam Deck LCD/OLED APU — "Van Gogh" ("Aerith", "Sephiroth") |
| | `gfx1034` | yes | Radeon RX 6500(XT/M), RX 6400, RX 6300(M), RX 6550M, RX 6450M, Pro W6400, Pro W6300(M), Pro W6500M |
| | `gfx1035` | no | Radeon 680M, 660M, 610M — Ryzen 6000 "Rembrandt"/"Rembrandt-R" mobile APUs |
| | `gfx1036` | no | Radeon Graphics 2CU — Ryzen 7000/9000 desktop, Threadripper 7000/9000, EPYC 4004/4005; Radeon 610M — "Dragon Range" mobile |
| RDNA 3 (Navi 3X — consumer, workstation & APUs) | `gfx1100` | yes | Radeon RX 7900(XT/XTX/GRE/M), Pro W7900, Pro W7800 |
| | `gfx1101` | yes | Radeon RX 7800 XT, RX 7700 XT, Pro W7700, Pro V710 |
| | `gfx1102` | yes | Radeon RX 7600(XT/S/M/M XT), RX 7650 GRE, RX 7700S, RX 7550M, Pro W7600, Pro W7500 |
| | `gfx1103` | no | Radeon 780M, 760M, 740M, 610M — Ryzen 7040/8040 "Phoenix/Phoenix 2/Hawk Point" mobile APUs; Ryzen Z1(Extreme), Ryzen Z2 Go |
| RDNA 3.5 (Ryzen AI 300 / Max — APUs) | `gfx1150` | yes | Radeon 890M, 880M — Ryzen AI 300 "Strix Point" APUs; Ryzen Z2 Extreme |
| | `gfx1151` | yes | Radeon 8060S, 8050S — Ryzen AI Max "Strix Halo" APUs |
| | `gfx1152` | no | Radeon 860M, 820M — Ryzen AI "Krackan Point" APUs |
| | `gfx1153` | no | Radeon 840M — Ryzen AI "Krackan Point 2" APUs |
| RDNA 4 (Navi 4X — consumer & pro) | `gfx1200` | yes | Radeon RX 9060(XT/M/S), RX 9050 |
| | `gfx1201` | yes | Radeon RX 9070(XT/GRE/S/M XT), RX 9080M, Radeon AI PRO R9700 |

ROCm 7.14.0 has optimized libraries for every target above, so `HSA_OVERRIDE_GFX_VERSION` is only needed when the image itself has no code for your GPU:

- prebuilt image, `yes` row — leave the override unset;
- prebuilt image, `no` row — set the override, or better, build locally for your target;
- local build — leave it unset and pass your target to `ROCM_DOCKER_ARCH`.

Spoofing another architecture costs performance and can give incorrect results, so use it only as a fallback.

This is a wider set than AMD's official support matrix, which is worth keeping in mind:

- RDNA 1 (`gfx1010`, `gfx1011`, `gfx1012`) is not officially supported at all. ROCm still ships packages for it and it does work in practice — an RX 5500M (`gfx1012`) handles Qwen 3.5 4B fine, for instance — but expect it to be validated rather than tuned.
- For RDNA 2, only `gfx1030` is officially supported. The notable change in 7.14.0 is that the rest of the generation (`gfx1031` through `gfx1036`) now has its own packages too, so those cards no longer need `HSA_OVERRIDE_GFX_VERSION=10.3.0` to masquerade as `gfx1030` — provided llama.cpp was built for the target. `gfx1031`, `gfx1032` and `gfx1034` are in the prebuilt images and work without the override; `gfx1035` and `gfx1036` are not, so on a prebuilt image they still need it, and a local build for the real target is the better option.

Treat everything outside the official matrix as working but less optimized than RDNA 3 and newer, where AMD puts most of the tuning effort.

## Building Docker locally

```bash
docker build -t local/llama.cpp:full-rocm --target full -f .devops/rocm.Dockerfile .
docker build -t local/llama.cpp:light-rocm --target light -f .devops/rocm.Dockerfile .
docker build -t local/llama.cpp:server-rocm --target server -f .devops/rocm.Dockerfile .
```

`ROCM_DOCKER_ARCH` selects the architectures to compile for. It defaults to the same set as the prebuilt images, so pass it explicitly in either of these cases:

- your GPU is one of the `no` rows in the table above — the APU targets `gfx1033`, `gfx1035`, `gfx1036`, `gfx1103`, `gfx1152` and `gfx1153`;
- you only care about your own GPU and want a much shorter build and a smaller image.

```bash
docker build -t local/llama.cpp:server-rocm --target server \
    --build-arg ROCM_DOCKER_ARCH=gfx1103 -f .devops/rocm.Dockerfile .
```

Multiple targets are separated by semicolons, for example `--build-arg ROCM_DOCKER_ARCH='gfx1100;gfx1103'`. Build time and image size scale with the number of architectures, which is why the published images stop at the discrete GPUs and Strix Halo.

Refer to [.devops/rocm.Dockerfile](../.devops/rocm.Dockerfile) for the remaining `ARGS` and their defaults.

## Usage

After building locally, usage is similar to the non-ROCm examples, but you'll need to expose the GPU devices to the container and use the `--n-gpu-layers` flag.

```bash
docker run --device /dev/kfd --device /dev/dri \
    --group-add $(getent group video | cut -d: -f3) \
    --group-add $(getent group render | cut -d: -f3) \
    -v /path/to/models:/models local/llama.cpp:full-rocm --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 99
docker run --device /dev/kfd --device /dev/dri \
    --group-add $(getent group video | cut -d: -f3) \
    --group-add $(getent group render | cut -d: -f3) \
    -v /path/to/models:/models local/llama.cpp:light-rocm -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 99
docker run --device /dev/kfd --device /dev/dri \
    --group-add $(getent group video | cut -d: -f3) \
    --group-add $(getent group render | cut -d: -f3) \
    -p 8080:8080 -v /path/to/models:/models local/llama.cpp:server-rocm -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512 --n-gpu-layers 99
```

*Notes:*
- `--device /dev/dri` exposes every render node on the host. To restrict the container to one GPU, pass the individual nodes instead, for example `--device /dev/dri/renderD128 --device /dev/dri/card0`.
- If llama.cpp reports `no usable GPU found` while `rocminfo` inside the same container does list the GPU, the HIP backend failed to load rather than the GPU being unavailable. `--list-devices` shows the loader error.

## Docker With MUSA

Assuming one has the [mt-container-toolkit](https://developer.mthreads.com/musa/native) properly installed on Linux, `muBLAS` should be accessible inside the container.

## Building Docker locally

```bash
docker build -t local/llama.cpp:full-musa --target full -f .devops/musa.Dockerfile .
docker build -t local/llama.cpp:light-musa --target light -f .devops/musa.Dockerfile .
docker build -t local/llama.cpp:server-musa --target server -f .devops/musa.Dockerfile .
```

You may want to pass in some different `ARGS`, depending on the MUSA environment supported by your container host, as well as the GPU architecture.

The defaults are:

- `MUSA_VERSION` set to `rc4.3.0`

The resulting images, are essentially the same as the non-MUSA images:

1. `local/llama.cpp:full-musa`: This image includes both the `llama-cli` and `llama-completion` executables and the tools to convert LLaMA models into ggml and convert into 4-bit quantization.
2. `local/llama.cpp:light-musa`: This image only includes the `llama-cli` and `llama-completion` executables.
3. `local/llama.cpp:server-musa`: This image only includes the `llama-server` executable.

## Usage

After building locally, Usage is similar to the non-MUSA examples, but you'll need to set `mthreads` as default Docker runtime. This can be done by executing `(cd /usr/bin/musa && sudo ./docker setup $PWD)` and verifying the changes by executing `docker info | grep mthreads` on the host machine. You will also want to use the `--n-gpu-layers` flag.

```bash
docker run -v /path/to/models:/models local/llama.cpp:full-musa --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run -v /path/to/models:/models local/llama.cpp:light-musa -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run -v /path/to/models:/models local/llama.cpp:server-musa -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512 --n-gpu-layers 1
```

## Docker With SYCL

## Building Docker locally

```bash
docker build -t local/llama.cpp:full-intel --target full -f .devops/intel.Dockerfile .
docker build -t local/llama.cpp:light-intel --target light -f .devops/intel.Dockerfile .
docker build -t local/llama.cpp:server-intel --target server -f .devops/intel.Dockerfile .
```

You may want to pass in some different `ARGS`, depending on the SYCL environment supported by your container host, as well as the GPU architecture.
Refer to [.devops/intel.Dockerfile](../.devops/intel.Dockerfile) for the available `ARGS` and their defaults.

The resulting images, are essentially the same as the non-SYCL images:

1. `local/llama.cpp:full-intel`: This image includes both the `llama-cli` and `llama-completion` executables and the tools to convert LLaMA models into ggml and convert into 4-bit quantization.
2. `local/llama.cpp:light-intel`: This image only includes the `llama-cli` and `llama-completion` executables.
3. `local/llama.cpp:server-intel`: This image only includes the `llama-server` executable.

## Usage

After building locally, usage is similar to the non-SYCL examples, but you'll need to add the `--device` flag.

```bash
# First, find all the DRI cards
ls -la /dev/dri
# Then, pick the card that you want to use (here for e.g. /dev/dri/card0).
docker run --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 -v /path/to/models:/models local/llama.cpp:full-intel -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 99
docker run --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 -v /path/to/models:/models local/llama.cpp:light-intel -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 99
docker run --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 -v /path/to/models:/models local/llama.cpp:server-intel -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512 --n-gpu-layers 99
```

*Notes:*
- Docker has been tested successfully on native Linux. WSL support has not been verified yet.
- You may need to install Intel GPU driver on the **host** machine *(Please refer to the [Linux configuration](./backend/SYCL.md#linux) for details)*.
