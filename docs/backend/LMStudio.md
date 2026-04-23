# LM Studio With TurboQuant `TQ3_*`

This document explains how to use an LM Studio install with a TurboQuant-enabled `llama.cpp` fork.

The goal is simple:

- keep LM Studio as the frontend
- replace the bundled `llama.cpp` runtime with a build from this fork
- load `TQ3_*` GGUF files through LM Studio instead of stock GGUF-only runtimes

## Requirements

- LM Studio desktop app or `lms` CLI installed
- a TurboQuant-enabled build of this repository
- a matching CUDA runtime family between LM Studio and the custom build

## Choose The Right LM Studio Runtime

LM Studio ships multiple `llama.cpp` backend packages. List what is available on your machine:

```bash
lms runtime ls
```

For NVIDIA/CUDA setups, the common choices are:

- `llama.cpp-linux-x86_64-nvidia-cuda12-avx2`
- `llama.cpp-linux-x86_64-nvidia-cuda-avx2`

Use the backend family that matches the custom build you plan to drop in.

## Build This Fork

Build a release tree that produces the shared libraries LM Studio expects:

```bash
cmake -S . -B build-lmstudio \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-lmstudio \
  --target llama-server llama-cli llama-bench llama-quantize \
  -j"$(nproc)"
```

If you want to target a specific CUDA major version, point CMake at the matching toolkit before building.

Check the linked CUDA runtime:

```bash
ldd build-lmstudio/bin/libllama.so | grep cudart
```

## Replace The LM Studio Backend

Find the runtime directory from `lms runtime ls`. It will look similar to:

```text
~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.13.0
```

Back up the existing backend first:

```bash
BACKEND_DIR="$HOME/.lmstudio/extensions/backends/<lmstudio-backend-dir>"
BACKUP_DIR="$BACKEND_DIR/orig-backup-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$BACKUP_DIR"
cp "$BACKEND_DIR"/*.so "$BACKUP_DIR"/
```

Then copy the custom runtime libraries into the LM Studio backend directory:

```bash
CUSTOM_DIR="<path-to-this-repo>/build-lmstudio/bin"

cp -f "$CUSTOM_DIR/libllama.so" "$BACKEND_DIR/libllama.so"
cp -f "$CUSTOM_DIR/libggml.so" "$BACKEND_DIR/libggml.so"
cp -f "$CUSTOM_DIR/libggml-base.so" "$BACKEND_DIR/libggml-base.so"
cp -f "$CUSTOM_DIR/libggml-cpu.so" "$BACKEND_DIR/libggml-cpu.so"
cp -f "$CUSTOM_DIR/libggml-cuda.so" "$BACKEND_DIR/libggml-cuda.so"
cp -f "$CUSTOM_DIR/libmtmd.so" "$BACKEND_DIR/libmtmd.so"
```

Create the compatibility symlinks LM Studio expects:

```bash
cd "$BACKEND_DIR"

ln -sfn libggml.so libggml_llamacpp.so
ln -sfn libggml.so libggml.so.0
ln -sfn libggml-base.so libggml-base.so.0
ln -sfn libggml-cpu.so libggml-cpu.so.0
ln -sfn libggml-cuda.so libggml-cuda.so.0
ln -sfn libllama.so libllama.so.0
ln -sfn libmtmd.so libmtmd.so.0
```

## Sanity Check

Before opening LM Studio, confirm the backend resolves cleanly:

```bash
ldd "$BACKEND_DIR/libllama.so"
ldd "$BACKEND_DIR/libmtmd.so"
```

You want:

- no `not found` entries
- the CUDA libraries you expect
- `libllama.so` and `libmtmd.so` resolving against the replaced backend bundle

## Import A TQ3 Model

Import a local GGUF in LM Studio:

```bash
lms import /path/to/your/model-TQ3_4S.gguf
```

Useful formats in this fork include:

- `TQ3_1S`
- `TQ3_4S`

If the model is multimodal, also provide the matching `mmproj` file.

## Rollback

If LM Studio fails to load after the swap, restore the backup bundle:

```bash
cp -f "$BACKUP_DIR/libllama.so" "$BACKEND_DIR/libllama.so"
cp -f "$BACKUP_DIR/libggml_llamacpp.so" "$BACKEND_DIR/libggml_llamacpp.so"
cp -f "$BACKUP_DIR/libggml-base.so" "$BACKEND_DIR/libggml-base.so"
cp -f "$BACKUP_DIR/libggml-cpu.so" "$BACKEND_DIR/libggml-cpu.so"
cp -f "$BACKUP_DIR/libggml-cuda.so" "$BACKEND_DIR/libggml-cuda.so"
cp -f "$BACKUP_DIR/libmtmd.so" "$BACKEND_DIR/libmtmd.so"

rm -f \
  "$BACKEND_DIR/libggml.so" \
  "$BACKEND_DIR/libggml.so.0" \
  "$BACKEND_DIR/libggml-base.so.0" \
  "$BACKEND_DIR/libggml-cpu.so.0" \
  "$BACKEND_DIR/libggml-cuda.so.0" \
  "$BACKEND_DIR/libllama.so.0" \
  "$BACKEND_DIR/libmtmd.so.0"
```

## Notes

- The custom fork is not stock LM Studio runtime support.
- If the runtime package and the custom build disagree on CUDA major version, rebuild the custom fork against the same CUDA family first.
- Keep the first test model simple and single-file before trying split or multimodal variants.
