> [!IMPORTANT]
> This build documentation is specific only to RISC-V SpacemiT SOCs.

## Build llama.cpp locally (for riscv64)

1. Prepare Toolchain For RISCV
~~~
wget https://github.com/spacemit-com/toolchain/releases/download/v1.2.4/spacemit-toolchain-linux-glibc-x86_64-v1.2.4.tar.xz
~~~

2. Build
Below is the build script: it requires utilizing RISC-V vector instructions for acceleration. Ensure the `GGML_CPU_RISCV64_SPACEMIT` compilation option is enabled. The currently supported optimization version is `RISCV64_SPACEMIT_IME1` and `RISCV64_SPACEMIT_IME2`, corresponding to the `RISCV64_SPACEMIT_IME_SPEC` compilation option. Compiler configurations are defined in the `riscv64-spacemit-linux-gnu-gcc.cmake` file. Please ensure you have installed the RISC-V compiler and set the environment variable via `export RISCV_ROOT_PATH={your_compiler_path}`.
```bash

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_RISCV64_SPACEMIT=ON \
    -DGGML_CPU_REPACK=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DGGML_RVV=ON \
    -DGGML_RV_ZVFH=ON \
    -DGGML_RV_ZFH=ON \
    -DGGML_RV_ZICBOP=ON \
    -DGGML_RV_ZIHINTPAUSE=ON \
    -DGGML_RV_ZBA=ON \
    -DCMAKE_TOOLCHAIN_FILE=${PWD}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake \
    -DCMAKE_INSTALL_PREFIX=build/installed

cmake --build build --parallel $(nproc) --config Release

pushd build
make install
popd
```

## Simulation
You can use QEMU to perform emulation on non-RISC-V architectures.

1. Download QEMU
~~~
wget https://archive.spacemit.com/spacemit-ai/qemu/jdsk-qemu-v0.0.14.tar.gz
~~~

2. Run Simulation
After build your llama.cpp, you can run the executable file via QEMU for simulation, for example:
~~~
export QEMU_ROOT_PATH={your QEMU file path}
export RISCV_ROOT_PATH_IME1={your RISC-V compiler path}

${QEMU_ROOT_PATH}/bin/qemu-riscv64 -L ${RISCV_ROOT_PATH_IME1}/sysroot -cpu max,vlen=256,elen=64,vext_spec=v1.0 ${PWD}/build/bin/llama-cli -m ${PWD}/models/Qwen2.5-0.5B-Instruct-Q4_0.gguf -t 1
~~~

## Run On Device
On a SpacemiT board the backend reads `marchid` from `/proc/cpuinfo` and selects cores with the variables below. `SPACEMIT_CORE_ARCH` is read only when `/proc/cpuinfo` has no SpacemiT core, which is the QEMU case above. A `0x` prefix on the arch id is optional.

| Variable | Default | Value |
| --- | --- | --- |
| `SPACEMIT_PERFER_CORE_ARCH` | cores with arch id high bits `0xA` (`arch >> 12 == 0xA`) | `503c` X60, `5064` X100, `50c8` X200, `a03c` A60, `a064` A100, `a0c8` A200 |
| `SPACEMIT_PERFER_CORE_ID` | every core selected above | comma-separated `processor` ids from `/proc/cpuinfo` |
| `SPACEMIT_CORE_ARCH` | parse `/proc/cpuinfo` | same hex arch id as `SPACEMIT_PERFER_CORE_ARCH` |
| `SPACEMIT_MEM_BACKEND` | `HPAGE` | `NONE`, `POSIX`, `HPAGE`, `HPAGE1GB` |
| `SPACEMIT_DISABLE_TCM` | use TCM when the device provides it | any value other than `0` disables TCM |

X100 (`5064`) and A60 (`a03c`) enable IME1. A100 (`a064`) enables IME2. X60, X200 and A200 do not enable IME1 or IME2.

To make any child processes get sent to the A100 cores, you need to write the PID of the current process to `/proc/set_ai_thread`. In many shell programs, such as bash or zsh, this can be done with `echo $$ > /proc/set_ai_thread`.
~~~
echo $$ > /proc/set_ai_thread
~~~

Set them in the environment of the process you start. For example, to run on A100 cores 8-15:
~~~
echo $$ > /proc/set_ai_thread
export SPACEMIT_PERFER_CORE_ARCH=a064
export SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15

${PWD}/build/bin/llama-cli -m ${PWD}/models/Qwen2.5-0.5B-Instruct-Q4_0.gguf -t 8
~~~

The same variables can be written in front of the command: `SPACEMIT_PERFER_CORE_ARCH=a064 SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15 ${PWD}/build/bin/llama-cli ...`. They are read when the process starts.

Each id in `SPACEMIT_PERFER_CORE_ID` must be in range and must have the arch given by `SPACEMIT_PERFER_CORE_ARCH`, otherwise the process aborts. An unknown `SPACEMIT_MEM_BACKEND` aborts with the four names above.

## Quantization Support For Matrix

| Quantization Type | X60 | A100 |
| ---: | ---: | ---: |
| Q2_K |  | :heavy_check_mark: |
| Q3_K |  | :heavy_check_mark: |
| Q4_0 | :heavy_check_mark: | :heavy_check_mark: |
| Q4_1 | :heavy_check_mark: | :heavy_check_mark: |
| Q4_K | :heavy_check_mark: | :heavy_check_mark: |
| Q5_0 |  | :heavy_check_mark: |
| Q5_1 |  | :heavy_check_mark: |
| Q5_K |  | :heavy_check_mark: |
| Q6_K |  | :heavy_check_mark: |
| Q8_0 | :heavy_check_mark: | :heavy_check_mark: |


## Performance
* Spacemit(R) X60
~~~
model name      : Spacemit(R) X60
isa             : rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt
mmu             : sv39
uarch           : spacemit,x60
mvendorid       : 0x710
marchid         : 0x8000000058000001
~~~

| model                          |       size |     params | backend    | threads | n_ubatch | fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -: | ---: | --------------: | -------------------: |
| qwen35 2B Q4_1                 |   1.19 GiB |     1.88 B | CPU        |       4 |      128 |  1 |    0 |           pp128 |         10.32 ± 0.02 |
| qwen35 2B Q4_1                 |   1.19 GiB |     1.88 B | CPU        |       4 |      128 |  1 |    0 |           tg128 |          3.07 ± 0.01 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       4 |      128 |  1 |    0 |           pp128 |         49.15 ± 0.25 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       4 |      128 |  1 |    0 |           tg128 |         11.73 ± 0.02 |


* Spacemit(R) A100
~~~
model name      : Spacemit(R) A100
isa             : rv64imafdcvh_zicbom_zicbop_zicboz_zicntr_zicond_zicsr_zifencei_zihintntl_zihintpause_zihpm_zimop_zaamo_zalrsc_zawrs_zfa_zfh_zfhmin_zca_zcb_zcd_zcmop_zba_zbb_zbc_zbs_zkt_zvbb_zvbc_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkb_zvkg_zvkned_zvknha_zvknhb_zvksed_zvksh_zvkt_smaia_smstateen_ssaia_sscofpmf_sstc_svinval_svnapot_svpbmt_sdtrig
mmu             : sv39
mvendorid       : 0x710
marchid         : 0x8000000041000002
mimpid          : 0x10000000d5686200
hart isa        : rv64imafdcv_zicbom_zicbop_zicboz_zicntr_zicond_zicsr_zifencei_zihintntl_zihintpause_zihpm_zimop_zaamo_zalrsc_zawrs_zfa_zfh_zfhmin_zca_zcb_zcd_zcmop_zba_zbb_zbc_zbs_zkt_zvbb_zvbc_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkb_zvkg_zvkned_zvknha_zvknhb_zvksed_zvksh_zvkt_smaia_smstateen_ssaia_sscofpmf_sstc_svinval_svnapot_svpbmt_sdtrig
~~~

| model                          |       size |     params | backend    | threads | n_ubatch | fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -: | ---: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       8 |      128 |  1 |    0 |           pp128 |        565.83 ± 0.31 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       8 |      128 |  1 |    0 |           tg128 |         55.77 ± 0.02 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | CPU        |       8 |      128 |  1 |    0 |           pp128 |         79.74 ± 0.04 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | CPU        |       8 |      128 |  1 |    0 |           tg128 |         11.29 ± 0.00 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B | CPU        |       8 |      128 |  1 |    0 |           pp128 |         57.88 ± 0.31 |
| qwen3moe 30B.A3B Q4_0          |  16.18 GiB |    30.53 B | CPU        |       8 |      128 |  1 |    0 |           tg128 |         12.79 ± 0.00 |
| qwen35 2B Q4_1                 |   1.19 GiB |     1.88 B | CPU        |       8 |      128 |  1 |    0 |           pp128 |        115.23 ± 0.04 |
| qwen35 2B Q4_1                 |   1.19 GiB |     1.88 B | CPU        |       8 |      128 |  1 |    0 |           tg128 |         16.49 ± 0.01 |
| gemma4 E4B Q4_K - Medium       |   4.76 GiB |     7.52 B | CPU        |       8 |      128 |  1 |    0 |           pp128 |         21.13 ± 0.01 |
| gemma4 E4B Q4_K - Medium       |   4.76 GiB |     7.52 B | CPU        |       8 |      128 |  1 |    0 |           tg128 |          5.66 ± 0.00 |
