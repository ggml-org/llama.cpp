# Docker

> [!NOTE]
> This fork ships **two Dockerfiles** in `.devops/`: `cpu.Dockerfile` and `vulkan.Dockerfile`. The OpenVINO backend is built from source (see [docs/build.md](build.md)) but is not packaged as a Docker image in this fork. CUDA, HIP/ROCm, Metal, OpenCL, CANN, MUSA, and other upstream Dockerfiles are not part of this fork; pull them from upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) if you need them. No images from this fork are published to a registry; all images must be built locally.

## Prerequisites

* Docker must be installed and running on your system
* Create a folder to store big models and intermediate files (e.g. `/llama/models`)

## Build

```bash
# CPU image (multi-target: full, light, server)
docker build -t local/llama.cpp:full    --target full    -f .devops/cpu.Dockerfile .
docker build -t local/llama.cpp:light   --target light   -f .devops/cpu.Dockerfile .
docker build -t local/llama.cpp:server  --target server  -f .devops/cpu.Dockerfile .

# Vulkan image (multi-target: full, light, server)
docker build -t local/llama.cpp:full-vulkan    --target full    -f .devops/vulkan.Dockerfile .
docker build -t local/llama.cpp:light-vulkan   --target light   -f .devops/vulkan.Dockerfile .
docker build -t local/llama.cpp:server-vulkan  --target server  -f .devops/vulkan.Dockerfile .
```

## Usage

Replace `/path/to/models` with your host-side model directory in the examples below.

### CPU image

```bash
docker run -v /path/to/models:/models local/llama.cpp:full -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 steps:" -n 512
```

The CPU image's `full` target also bundles `llama-quantize` and the conversion tooling, so you can run the model acquisition + conversion pipeline in one container:

```bash
docker run -v /path/to/models:/models local/llama.cpp:full --all-in-one "/models/" 7B
```

The light and server targets are slimmer:

```bash
docker run -v /path/to/models:/models local/llama.cpp:light  -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 steps:" -n 512

docker run -v /path/to/models:/models -p 8080:8080 local/llama.cpp:server -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512
```

### Vulkan image

Assumes the Vulkan driver, SDK, and the relevant Intel / AMD / NVIDIA GPU drivers are installed on the host.

```bash
docker run -v /path/to/models:/models --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 local/llama.cpp:full-vulkan -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 99
```

## OpenVINO

The OpenVINO backend source is in the tree (`ggml/src/ggml-openvino/`) and is built when `-DGGML_OPENVINO=ON` is passed to CMake. There is no `.devops/openvino.Dockerfile` in this fork; if you need an OpenVINO image, write a Dockerfile that derives from `.devops/cpu.Dockerfile` and flips that flag on. See [docs/build.md](build.md#openvino) for the in-tree build flags.

## Notes

* Docker has been tested on native Linux. WSL support has not been verified in this fork.
* The upstream `ghcr.io/ggml-org/llama.cpp:*` tags do **not** include the TurboQuant+ codec stack; pull them only if you want upstream mainline with no fork-specific features.
