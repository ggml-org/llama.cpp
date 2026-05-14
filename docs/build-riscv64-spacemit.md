> [!IMPORTANT]
> This build documentation is specific only to RISC-V SpacemiT SOCs.

## Build llama.cpp locally (for riscv64)

1. Prepare Toolchain For RISCV
~~~
wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz
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
| Q8_0 |  | :heavy_check_mark: |


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
