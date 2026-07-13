# ppu_so — external kernel `.so` build (FlashAttention + DeepGEMM)

Builds the heavy kernels **outside** the llama.cpp build into two standalone shared objects:

| `.so` | source submodule | C ABI header (in ggml-cuda) | wraps |
|---|---|---|---|
| `libppu_fa.so`  | `thirdparty/flash-attention` | `ggml/src/ggml-cuda/ppu-fa-so.h`  | FA2 `run_mha_fwd_<T, headdim, Is_causal>` (sm80 fwd) |
| `libppu_moe.so` | `thirdparty/DeepGEMM`        | `ggml/src/ggml-cuda/ppu-moe-so.h` | DeepGEMM `sm90_m_grouped_bf16_gemm_contiguous` |

**Decoupling contract.** llama.cpp compiles with **zero** cutlass / flash-attention / DeepGEMM / torch headers or
link deps. It only knows the two thin `extern "C"` headers and `dlopen`s the `.so` at runtime (`ppu-so.cu`). If a
`.so` is missing or has no kernel for a shape, the caller falls back to the inline ggml path. Build llama.cpp with
`-DGGML_PPU_SO=ON`; the default OFF is fully inert.

**Scope.** fp16/bf16 only, as intended:
* FA — fp16 **and** bf16, head_dim ∈ {64, 96, 128, 192, 256}, GQA, softcap, `is_causal`.
* MoE GEMM — **bf16 only**. Public DeepGEMM has no fp16 kernel at all (`cutlass::half_t` appears nowhere in
  `deep_gemm/include/deep_gemm/impls/`); its dtypes are bf16, fp8/fp4 and tf32. fp16 MoE weights fall back to ggml.

## 1. Submodules

Already registered in `.gitmodules` under `thirdparty/`:

```sh
git submodule update --init --recursive thirdparty/flash-attention thirdparty/DeepGEMM
```

## 2. Build the two `.so`

```sh
cd ppu_so
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DTorch_DIR=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share/cmake/Torch"))')
cmake --build build -j            # -> build/libppu_fa.so, build/libppu_moe.so, build/test_fa, build/test_moe
```

Torch is a **compile-time dependency only** (header include dirs). Neither `.so` links libtorch — verify:

```sh
ldd build/libppu_fa.so build/libppu_moe.so | grep -i torch          # must be empty
nm -D -u build/libppu_moe.so | c++filt | grep -iE 'c10::|at::|torch' # must be empty
```

`libppu_fa.so` needs only `libcudart`; `libppu_moe.so` also needs `libnvrtc` (DeepGEMM JITs its kernel).

On the PPU box (ppu001) pass `-DCMAKE_CUDA_COMPILER=<ppu-nvcc> -DCMAKE_CUDA_ARCHITECTURES=OFF` instead: a forced
`-arch=sm_XX` routes onto ppu0015 and rejects ppu001-only atoms (see the `ppu-ptx` skill). Note that public DeepGEMM
is **SM90/SM100 only** (wgmma / TMA / tcgen05); on the PPU (CC 8.0) `libppu_moe.so` returns non-zero and llama.cpp
falls back — the PPU needs the internal `ppu_open_source/DeepGemm` fork, whose `BF16GemmCutlass3Runtime` retargets
the codegen onto the PPU AIU atoms. The FA side has no such problem: its sm80 forward is native.

## 3. Standalone tests (no ggml, no model)

```sh
export CUDA_HOME=/usr/local/cuda
export DG_LIBRARY_ROOT=$PWD/../thirdparty/DeepGEMM/deep_gemm
export DG_JIT_CACHE_DIR=$HOME/.deep_gemm_cache

./build/test_fa 0 128       # non-causal, head_dim 128   (arg1 = is_causal, arg2 = head_dim)
./build/test_fa 1 128       # causal
./build/test_moe 4 128 256 128   # G rows_per_expert N K
```

Both compare against an fp32 CPU reference.

## 4. Point llama.cpp at them

```sh
cmake -B build -DGGML_CUDA=ON -DGGML_PPU_SO=ON -DCMAKE_CUDA_ARCHITECTURES=90
export GGML_PPU_FA_SO=/abs/path/ppu_so/build/libppu_fa.so     # env wins over the bare soname on the loader path
export GGML_PPU_MOE_SO=/abs/path/ppu_so/build/libppu_moe.so
export CUDA_HOME=/usr/local/cuda                              # MoE only: DeepGEMM's JIT needs these
export DG_LIBRARY_ROOT=/abs/path/thirdparty/DeepGEMM/deep_gemm
export DG_JIT_CACHE_DIR=/abs/path/deep_gemm_cache             # ship this warm cache to skip runtime nvcc
```

## Two non-obvious things this build had to solve

**1. `libppu_moe.so` would not be torch-free out of the box.** DeepGEMM's `DeviceRuntime` *constructor* allocates a
32 MB cuBLASLt workspace with `torch::empty()` and holds it in a `torch::Tensor` member. That is a hard link
dependency (`at::_ops::empty_memory_format`, `c10::UndefinedTensorImpl::_singleton`, the `AutogradMeta` vtable, …)
which `--gc-sections` cannot strip, because the ctor genuinely references it — and `dlopen(RTLD_NOW)` would fail on
the unresolved data relocations. Our bf16 grouped-GEMM path never touches cuBLASLt, so `patches/0001-deepgemm-no-
torch-cublaslt.patch` guards that member and its ctor/dtor/accessors behind `DG_NO_TORCH`, which CMake applies
(idempotently) and defines. One inline `TORCH_CHECK` still bottoms out in `c10::detail::torchCheckFail`, which
`moe/deepgemm_c.cpp` defines itself. Symmetrically, FA's launch templates call `C10_CUDA_CHECK`, whose only symbol
`c10::cuda::c10_cuda_check_implementation` is defined in `fa/flash_api_c.cpp`. Both stubs throw, and both callers
catch → non-zero rc → ggml inline fallback.

**2. The MoE compact layout must be padded per expert.** DeepGEMM's grouped-contiguous kernel pins `BLOCK_M` to
`get_mk_alignment_for_contiguous_layout()` (=128, `csrc/jit_kernels/heuristics/sm90.hpp:33`) and reads **one expert
id per `BLOCK_M` row block**, from that block's first row. If an expert's row segment is not a multiple of 128, a
block straddles two experts and is **silently** multiplied by the wrong weights — no assert, just wrong numbers
(`test_moe 4 100 256 128` used to report `rel_rms=0.15`). So the `.so` exports `ppu_moe_row_alignment()` and
`ggml_cuda_mul_mat_id_ppu_so` pads every expert's segment up to it; pad rows gather src1 row 0, carry their own
expert's id, and their output rows are simply never scattered back.

## Integration status (llama.cpp side)

- **FA** (`fattn.cu: ggml_cuda_flash_attn_ext_ppu_so`) — converts Q F32→F16 and O F16→F32 (pool scratch), passes F16
  K/V with their strides (ggml's `ne={d,seqlen,head,batch}` is physical `[b,h,s,d]`, i.e. head/seqlen transposed vs a
  packed FA tensor — the ABI carries element strides so there is zero repacking). Engages for full attention
  (`mask == null`) or pure-causal. Causal is signalled by the host via `ggml_flash_attn_ext_set_causal` (op_params[4],
  set in `llama-graph.cpp build_attn_mha` when `causal_attn && no-ALiBi && !SWA && !kv_unified`); the hook engages
  causal only when the hint is set **and** `mask->ne[3] == 1` (single stream). The sm80 forward has no additive-mask
  input, so arbitrary masks (SWA / padding / cross-seq / ALiBi) and attention sinks fall through to the inline path.
- **MoE** (`ggml-cuda.cu: ggml_cuda_mul_mat_id_ppu_so`) — reuses ggml's own ragged→compact machinery: host-sorts
  (token,slot) pairs by expert, pads each expert's segment (see above), `get_rows_cuda` gathers src1 F32→bf16 into
  compact `A[padded_rows, K]`, builds `m_indices`, runs ONE grouped GEMM, `get_rows_cuda` scatters the real bf16 rows
  back to dst as F32. Gates: bf16 expert weights (src0), F32 src1/dst, all contiguous. Only for bf16-weight MoE (e.g.
  Qwen3-MoE) — **not** mxfp4 gpt-oss, which needs a different fp8/fp4 DeepGEMM entry.
