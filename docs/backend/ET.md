# llama.cpp for ET

- [Background](#background)
- [Limitations](#limitations)
- [Build](#build)
- [Develop](#develop)
- [Roadmap](#roadmap)


## Background

**ET** is a llama.cpp backend targeting the fully open source manycore
RISC-V accelerator platform [ET-SOC](TODO).


## Limitations

The ET backend is at the proof-of-concept stage and comes with a bouquet of
limitations:

- Only limited set of operations is supported (check [../ops.md](../ops.md)
  and [../ops/ET.csv](../ops/ET.csv)).
- Only `q8_0` (and partially `fp16`) quantization is supported.
- Only one llama.cpp instance can use device at the same time (current firmware
  limitation).
- It is SLOW (compute kernels are very naive, sometimes not even parallel).
- Backend code has overly exhaustive logging (will be removed eventually).
- Incorrect reporting of free memory (show all of it as free).
- Numerous bugs.

As a result of the above, only select models can run fully on ET-SOC
(you can actually run any model llama.cpp supports, but some/most operations
will likely fallback to CPU backend).

Fully supported models (`q8_0` quantization):
- Qwen3 models (without MoE), e.g.
  [ggml-org/Qwen3-0.6B-GGUF:q8_0](https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q8_0.gguf) or
  [ggml-org/Qwen3-14B-GGUF:q8_0](https://huggingface.co/ggml-org/Qwen3-14B-GGUF/blob/main/Qwen3-14B-Q8_0.gguf).
- Llama3.2 (1B/3B), e.g.
  [lmstudio-community/Llama-3.2-1B-Instruct-GGUF:q8_0](https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q8_0.gguf).
- SmolLM2, e.g.
  [unsloth/SmolLM2-135M-Instruct-GGUF:q8_0](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/blob/main/SmolLM2-135M-Instruct-Q8_0.gguf)
- Llama3.1 model family.

So far we are at the milestone "one can easily write llama.cpp backend for
ET-SOC".


## Build

### I. Prerequisites

1. **Install custom RISC-V toolchain** - Follow instructions at:
   [https://github.com/aifoundry-org/riscv-gnu-toolchain/tree/et/aifoundry](https://github.com/aifoundry-org/riscv-gnu-toolchain/tree/et/aifoundry)

2. **Install ET platform** - Follow instructions at:
   [https://github.com/aifoundry-org/et-platform](https://github.com/aifoundry-org/et-platform)

Both should be installed to `/opt/et` (or set `ET_TOOLCHAIN` and `ET_PLATFORM`
environment variables accordingly).

```sh
# Set toolchain and ET platform path (/opt/et is default)
export ET_TOOLCHAIN=/opt/et
export ET_PLATFORM=/opt/et
```

### II. Build llama.cpp

Check out llama.cpp with ET backend (this should checkout `et` branch):

```sh
git clone https://github.com/aifoundry-org/llama.cpp
cd llama.cpp
```

Build:

```sh
cmake -B build -DGGML_ET=ON
cmake --build build --config Release
# Optionally:
# cmake --install build
```

### III. Run

Run llama.cpp binaries as usual. (Of course, please make sure you have the
ET-SOC device installed and kernel driver loaded).

```sh
llama-cli -m mymodel.gguf
# or
llama-server -hf ggml-org/Qwen3-8B-GGUF:q8_0
```

If you want to run llama.cpp binaries (e.g. `llama-cli`) inside docker
container, you should let it access device files:

```sh
docker run \
    --device=/dev/et0_mgmt:/dev/et0_mgmt \
    --device=/dev/et0_ops:/dev/et0_ops \
    ...
```

> [!NOTE]
> You may want to disable flash attention with:
> ```sh
> export LLAMA_ARG_FLASH_ATTN=off # or cli argument --flash-attn off
> ```
> Llama.cpp sets flash attention to "auto" by default and tries to detect if it's
> supported by the backend. When autodetection fails to find support (it's not
> supported in ET), KV tensors are left in transposed configuration which results
> in considerable slowdown.


## Develop

Compute kernels are developed within `ggml/src/ggml-et/et-kernels` folder.
Build is performed using custom RISC-V GNU toolchain and is managed by cmake.
At the moment kernels are build as baremetal elf files, without
standard lib or any other dependencies. All the yummy parts are written
in inline assembler.

Most kernels are very naive with lots of low hanging fruits left:
- some are single threaded (on a manycore chip!)
- some could use SIMD extensions
- memory access is not really cache aware (or using scratchpad)
- chip has tensor extensions that aren't used yet
- no operation uses temporary buffers
- there are no attempts at tiling (MUL_MAT family of operations)

Basically, there are at least 2-3 orders of magnitude performance
laying on the table.

Individual compute kernels can be used (tested) without rebuilding
ggml or llama.cpp - while by default compiled kernels are embedded
into ggml source code at build time, they can also be loaded at runtime
if `GGML_ET_KERNELS_PATH` is set. You just build the kernel (using provided cmake config)
and put resulting elf under the aforementioned path.

> [!IMPORTANT]
> Several assembly instructions emmited by the compiler are not implemented
> in hardware and software emulation in firmware is not ready yet.
> Eventually firmware will transparently trap unimplemented instructions
> and will emulate them inside exception handler. Until then, kernel
> build process includes step that checks compiled kernels and fails if any unimplemented
> instructions are found. Problematic ones follow:
> `FDIV.PI`, `FDIVU.PI`, `FREMU.PI`, `FREM.PI`, `FDIV.S`, `FDIV.PS`, `FSQRT.S`, `FSQRT.PS`, `FRSQ.PS`, `FSIN.PS`
>  and (long cast) `FCVT.S.L`, `FCVT.S.LU`, `FCVT.L.S`, `FCVT.LU.S`
> What this means, is that for now you should avoid doing any division involving floats,
> any trigonometry or casting longs into floats.
> Some workarounds are implemented in `math_fp.h` (`et_fdiv`, `et_powf` etc) and
> long casting (presuming longs are small enough to fit into 32bits) can be
> done via `int` like `a = (float)(int)(b)`.

> [!TIP]
> There are some slightly higher level helpers (abstracting more
> complex instructions like tensor extension or synchronization primitives)
> inside `et_platform`, directory `et-common-libs/include/etsoc/isa/`. It was
> originally developed for firmware needs and is not included into compute
> kernel build process. Feel free to take ideas/code from there or try linking
> it in.

Before commiting any changes to operations and/or kernels, don't forget
to update supported ops reports (instructions at `docs/ops.md`).

When logging is enabled (e.g. by setting `--log-file` cli param),
each compute kernel run outputs a line with
pipe-delimited key-value pairs containing kernel level performance infomation.
Line is prefixed with `ET_PERF`:

```
ET_PERF|op=MUL_MAT|kernel=mul_mat_f32_Q8_0xf32|duration_us=3112|tensor=Qcur-0|shape=[4096,2,1,1]|start_us=48437862009|end_us=48437865121|flops=67100672
ET_PERF|op=ROPE|kernel=rope_f32|duration_us=9266|tensor=Qcur-0|shape=[128,32,2,1]|start_us=48437865128|end_us=48437874394|mode=0x0|n_dims=128|freq_base=500000.00|freq_scale=1.00
```
Keys depend on the operation, but some are always present.
`flops` in this case counts effective floating point operations and not floating
point operations per second.

You can enable ET-SOC runtime level ET-SOC profiling by setting environment
variable `GGML_ET_PROFILE` to a path. Profiling/tracing results will be written
to `GGML_ET_PROFILE/et_runtime_trace.json` on exit.


## Roadmap

While advancing the ET backend from proof-of-concept to production-ready status,
development efforts will focus on the following areas:

- Optimize compute kernels
- More operations
- More quants
- Async buffer ops
- Async compute ops
- Graph optimization
- Improve debugging
