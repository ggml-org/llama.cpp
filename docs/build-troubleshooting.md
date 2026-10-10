# Build troubleshooting notes

Case studies of build/runtime issues that weren't obvious from the error message alone.

## CUDA error: invalid argument in ggml_cuda_mul_mat_q on newer GPUs (mixed CUDA toolkit installs)

**Symptom:** Server loads a MoE model fine, then crashes on the first inference request with:

```
CUDA error: invalid argument
  current device: 0, in function ggml_cuda_mul_mat_q at ggml/src/ggml-cuda/mmq.cu:202
  cudaGetLastError()
```

This reproduces on a single GPU with no tensor-split/multi-GPU flags involved, so despite
appearances it is not a multi-GPU or `--tensor-split` issue. It is not a "toolkit too old
for this GPU architecture" issue either — `nvcc --version` can report a toolkit newer than
required and `CMAKE_CUDA_ARCHITECTURES` can already be set correctly, and the crash still
happens.

**Root cause:** the system has more than one CUDA toolkit installed (e.g. Ubuntu's
`nvidia-cuda-toolkit` apt package alongside a manually installed newer toolkit under
`/usr/local/cuda-<version>`). `CMAKE_CUDA_COMPILER` picks up the newer `nvcc` correctly, but
CMake's CUDA toolkit *library* detection independently resolves `libcudart`/`libcublas` to
whichever copy is registered in the default system linker path first — which can be the
older apt-installed one. Check with:

```sh
readelf -d build/bin/libggml-cuda.so | grep NEEDED
```

If this shows a `libcudart.so.<major>` older than the toolkit `nvcc --version` reports, that's
the mismatch.

The `cudaDeviceProp` struct's layout differs between CUDA runtime major versions. Code
compiled against the newer toolkit's headers but linked against the older runtime's
`cudaGetDeviceProperties()` gets a partially-unfilled struct — in the case that motivated
this note, `sharedMemPerBlockOptin` came back as `4294967297` (`0x100000001`) instead of the
real ~99KB value. That garbage `size_t` is then passed into `cudaFuncSetAttribute(...,
cudaFuncAttributeMaxDynamicSharedMemorySize, int value)` — whose real parameter type is
`int`, not `size_t` — so it silently truncates to `1`, capping a kernel's dynamic shared
memory budget at 1 byte. The MoE expert-routing kernel (`mm_ids_helper` in
`ggml/src/ggml-cuda/mmid.cu`) needs a few dozen bytes even for a tiny prompt, so its very next
launch is rejected by the driver with `invalid argument`. Single-expert/dense models can dodge
this because they don't hit that particular kernel path, which is why the symptom can look
MoE-specific.

**Fix:** force CMake to resolve CUDA libraries from the same toolkit as the compiler:

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch> \
  -DCUDAToolkit_ROOT=/usr/local/cuda-<version> \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<version>/bin/nvcc
cmake --build build -j"$(nproc)"
```

A stale `CMakeCache.txt` can keep the old library paths cached even after adding
`CUDAToolkit_ROOT`, so if reconfiguring in place doesn't change the `readelf -d` output, wipe
the build directory and reconfigure from scratch.

As a longer-term fix, remove the older toolkit's runtime packages (e.g. via
`apt purge nvidia-cuda-toolkit libcudart<N> libcublas<N> libcublaslt<N>`) so future builds
that omit `CUDAToolkit_ROOT` can't silently re-pick the wrong one.
