# ggml-xdna kernels

Precompiled NPU block-GEMM kernels loaded by the XDNA backend at runtime. Each kernel is a pair

```
mm_bf16_f32_M<M>_K<K>_N<N>.xclbin      # AIE array configuration + compiled tile programs
mm_bf16_f32_M<M>_K<K>_N<N>.insts.bin   # runtime-sequence (DMA) instruction stream
```

computing, for one fixed block shape,

```
A : [M x K] bf16 row-major   (weight rows       = ggml src0, ne01 x ne00)
B : [N x K] bf16 row-major   (activation rows   = ggml src1, ne11 x ne10)
C : [N x M] f32  row-major   C[t][r] = sum_k A[r][k] * B[t][k]   (= ggml dst layout)
```

They are the mlir-aie `programming_examples/basic/matrix_multiplication/whole_array` design built
with `--dtype_in bf16 --dtype_out f32 --b-col-maj 1 --c-col-maj 1`, so ggml's native layouts go
straight into the NPU with no transposes. The backend tiles larger matmuls over (M, K, N) blocks,
picks the kernel with the least padding, and accumulates partial K products on the host.

## Building

Device code is compiled with the open-source IRON toolchain (mlir-aie + Peano); no NPU is needed
at compile time, so this works from WSL:

```bash
# once: IRON environment (Python 3.12), see https://github.com/Xilinx/mlir-aie
git clone --depth 1 --branch v1.4.3 https://github.com/Xilinx/mlir-aie ~/mlir-aie
cd ~/mlir-aie && source utils/env_install.sh

# every time
source ~/mlir-aie/ironenv/bin/activate && source ~/mlir-aie/utils/env_setup.sh
bash build_kernels.sh            # writes *.xclbin / *.insts.bin next to this README
```

`build_kernels.sh` targets `--dev npu` (XDNA1: Phoenix / Hawk Point, e.g. Ryzen 8700G). Set
`XDNA_DEV=npu2 XDNA_COLS=8` for XDNA2 parts. Shapes are constrained by the design:
`M % (m*4) == 0`, `K % k == 0`, `N % (n*4) == 0`, `(M/m/4) % 2 == 0`.

The final packing step needs XRT's `xclbinutil`, which mlir-aie does not ship. If you have no XRT
install (e.g. building from WSL), `xrt-utils.Dockerfile` + `xclbinutil-docker.sh` provide it from
the Ubuntu 26.04 `libxrt-utils` package: build the image, then copy the shim onto the venv's PATH
as `xclbinutil` (see the comments in the script).

Shipped kernels (XDNA1, 4 columns, bf16 in / f32 out):

| kernel | block (M x K x N) | tiles (m,k,n) | intended use |
|---|---|---|---|
| `mm_bf16_f32_M512_K512_N128`  | 512 x 512 x 128  | 64,64,32 | general |
| `mm_bf16_f32_M512_K1024_N128` | 512 x 1024 x 128 | 64,64,32 | K >= 1024 layers, fewer launches |
| `mm_bf16_f32_M512_K512_N256`  | 512 x 512 x 256  | 64,64,32 | 256+ token batches |
| `mm_bf16_f32_M256_K512_N64`   | 256 x 512 x 64   | 32,32,16 | small batches / narrow layers |

Measured on a Ryzen 7 8700G (via `pyxrt`, bf16 random data): the 512 x 512 x 128 block runs in
~360 us including dispatch (~185 GFLOPS) with NMSE ~3e-14 against the bf16 reference.

At runtime the backend searches `$GGML_XDNA_KERNEL_DIR`, falling back to this directory's
build-time path. `GGML_XDNA_MIN_BATCH` (default 32) sets the smallest token count a MUL_MAT
needs before it is sent to the NPU; decode-sized products always stay on the CPU.
