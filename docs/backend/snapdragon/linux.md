# Snapdragon-based Linux devices

The cross-compilation is performed using the Snapdragon Linux Docker toolchain image (see
[github.com/snapdragon-toolchain](https://github.com/snapdragon-toolchain)):

* **Linux toolchain**: `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`

The unified build utility (`scripts/snapdragon/build.py`) automatically pulls
and orchestrates this container to perform target compilation. You only need to
ensure that Docker is running on your host machine.


## How to Build

### Using build.py script (Recommended)

The easiest way to build llama.cpp is by using the `scripts/snapdragon/build.py` script. It automatically copies the CMake presets,
launches the correct compilation Docker container, builds the libraries and tools,
installs them, and optionally pushes them to your target device.

Build and deploy for a Linux target (using SSH deployment alias `lnx` or `linux`):
```
$ ./scripts/snapdragon/build.py --target lnx:user@host --push
```

### Manual CMake Build

Alternatively, you can build llama.cpp manually by entering the cross-compilation Docker container and running the CMake commands:

```bash
# Start the cross-compilation container manually:
~/src/llama.cpp$ docker run -it --rm -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-linux:v0.7

# Inside the container, build the project using presets:
[d]/workspace> cp docs/backend/snapdragon/CMakeUserPresets.json .

[d]/workspace> cmake --preset arm64-linux-snapdragon-release -B build-snapdragon

[d]/workspace> cmake --build build-snapdragon -j $(nproc)
```

To generate an installable "package" simply use cmake --install, then zip it:

```
[d]/workspace> cmake --install build-snapdragon --prefix pkg-linux
[d]/workspace> zip -r pkg-linux.zip pkg-linux
```

## How to Install

For this step, you will deploy the built binaries and libraries to the target
Linux device. Transfer `pkg-linux.zip` to the target device, then unzip it
and set up the environment variables:

```
$ unzip pkg-linux.zip
$ cd pkg-linux
$ export LD_LIBRARY_PATH=./lib
$ export ADSP_LIBRARY_PATH=./lib
```

At this point, you should also download some models onto the device:

```
$ wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf
```

## How to Run
You can run locally on the Snapdragon Linux device:
```
$ ./scripts/snapdragon/run.py --devices HTP0 -- llama-cli -m Llama-3.2-3B-Instruct-Q4_0.gguf -ngl 99 -p "what is the most popular cookie in the world?"
```

Or run remotely from your host development machine using the SSH target option:
```
$ ./scripts/snapdragon/run.py --target lnx:user@host --devices HTP0 -- llama-cli -m Llama-3.2-3B-Instruct-Q4_0.gguf -ngl 99 -p "what is the most popular cookie in the world?"
```

For multi-NPU systems, you can run a tensor split completion command targeting a remote Linux system:
```
$ ./scripts/snapdragon/run.py --target ubuntu:maxk@192.168.1.87 --device HTP0:0,HTP1:0 -- llama-completion -m models/gemma-2b-it-Q4_0.gguf -f prompts/sample_prompt_1024.txt --jinja -st --split-mode tensor --ctx-size 8192
```

This translates to the following command being executed remotely via SSH:
```
+ ssh maxk@192.168.1.87 "cd ~/llama.cpp && ulimit -c unlimited && LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0:0,HTP1:0 GGML_HEXAGON_OPPOLL=1 ./bin/llama-completion -m models/gemma-2b-it-Q4_0.gguf -f prompts/sample_prompt_1024.txt --jinja -st --split-mode tensor --ctx-size 8192 -v -n 16 --device HTP0:0,HTP1:0 -ngl 99 --ubatch-size 1024 -fa on -t 6"
```

Alternatively, you can run the binary directly on the device:
```
$ ./bin/llama-cli -m Llama-3.2-3B-Instruct-Q4_0.gguf --device HTP0 -ngl 99 -p "what is the most popular cookie in the world?"
```

