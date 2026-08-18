# Native RDNA2 RCCL and P2P coordination

A HIP/RCCL build packages a guarded RCCL tuner beside `libggml-hip`. The backend discovers it before communicator initialization. No launcher, tuner path, or P2P environment bundle is required.

Configure with HIP and RCCL; HIP graphs are already enabled by default:

```bash
cmake -S . -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Launch `llama-server` normally. Native gfx1030 kernel flags remain independent.

## Automatic policy

Policy is selected from collective and hardware properties rather than model identity:

- RCCL operation, byte count, and datatype;
- actual communicator rank count;
- GPU name and architecture;
- all-pairs peer access plus PCIe link type/hop matrix;
- installed RCCL tuner ABI;
- absence of user collective overrides.

The initial certified policy is TP4 on four V620/gfx1030 devices with the measured all-pairs PCIe-hop2 topology. It selects `NCCL_P2P_LEVEL=PXB` before communicator creation and applies Ring/LL/3 only to 20,480-byte AllReduce. Other payload sizes remain RCCL Auto. TP1 and TP2/3/5/6/7/8 also remain RCCL Auto until a policy for that rank/topology tuple is measured. This prevents one machine's PXB result from being forced on PHB or other layouts.

The tuner is model-independent: another model using the same certified collective tuple can benefit. Unknown tuples are not modified.

## Custom host-snapshot reductions

The host-snapshot path uses allocation-local coherent mapped memory; users do not need `HSA_FORCE_FINE_GRAIN_PCIE`. Startup compares both supported custom schedules (`5,120` and `25,600` F32 elements) against the installed RCCL with four adversarial patterns and sixteen chained reductions. A schedule activates only after byte-for-byte agreement. Therefore newer ABI-compatible RCCL releases may use the tuner, while custom reductions independently fall back if chunk/rank order changed.

Runtime tensor gates remain strict:

- four canonical distinct RDNA2 ranks and the certified peer/link topology;
- contiguous F32 `[5120,5,1,1]` tensors named `linear_attn_out-*`, `ffn_out-*`, or `attn_output-*`;
- the validated `[5120,1,1,1]` `linear_attn_out-*` case.

All misses use RCCL.

## Model-specific automatic paths

Only graph transformations remain model-specific:

- Deferred catch-up defaults on for `general.architecture=qwen35`, width 5120, one sequence/layer, non-shared/non-chained MTP, `n_max=4`, width five, and logits on every row.
- TP output sharding defaults on for validated Qwen35 27B heads after the existing split/head checks.

## Overrides

User settings are preserved. The following disable automatic behavior or select an explicit policy:

```text
GGML_HIP_RCCL_TUNE=off|auto|force
GGML_HIP_GFX1030_P2P_ALLREDUCE=off|auto|host|host-fused|host-mtp
GGML_MTP_DEFER_CATCHUP=0|auto|1
GGML_TP_SHARDED_OUTPUT=0|auto|1
```

A user-provided `NCCL_TUNER_PLUGIN`, `NCCL_P2P_LEVEL`, `NCCL_ALGO`, `NCCL_PROTO`, channel count, or `NCCL_NTHREADS` is never overwritten. `GGML_CUDA_ALLREDUCE` also retains its existing explicit override; Linux already defaults to NCCL/RCCL. NCCL P2P defaults enabled, and NCCL builds already establish VMM peer mappings, so `GGML_CUDA_P2P=1` and `NCCL_P2P_DISABLE=0` are not required.
