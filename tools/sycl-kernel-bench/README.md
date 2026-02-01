# sycl-kernel-bench

Standalone micro-benchmark harness for SYCL MMVQ/MMQ kernels.

## Build

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build sycl-kernel-bench
```

## Run

```bash
./build/bin/sycl-kernel-bench \
  --kernel=mmvq_aos \
  --quant=Q4_0 \
  --batch=1,4,8,16,32,64 \
  --dim=4096 \
  --iterations=100 \
  --warmup=10 \
  --output=csv \
  --validate
```

Add `--include-percentiles` to include p50/p90/p99 latency columns in CSV/JSON output.
Reference kernels (e.g. `onednn_fp16_gemm`, `onednn_int8_gemm`, `memory_bandwidth`, `roofline_compute`) emit
additional metrics by default. Use `--include-ref-metrics` to force those columns for any kernel.

## Performance gates

Fail fast when throughput falls below expected thresholds:
```bash
./build/bin/sycl-kernel-bench \
  --kernel=dpas_baseline --dim_m=4096 --dim_n=256 --dim_k=4096 \
  --dpas-type-a=int8 --dpas-type-b=int8 --dpas-acc=int32 \
  --dpas-memory=reg_prefetch --dpas-grf=128 --dpas-repeat=8 \
  --xmx-peak-tops=75 --expect-tops=25 --expect-xmx-util=30 \
  --expect-bandwidth=50 --output=jsonl
```

## Model-derived matmul shapes

Compare kernels using actual GGUF weight shapes and emit a summary JSON:
```bash
./build/bin/sycl-kernel-bench \
  --kernel=onednn_woq_gemm,onednn_fp16_gemm,unified_matmul \
  --model=/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  --batch=1,16,64 --sample-strategy=both \
  --limit-shapes=10 \
  --emit-json=/tmp/onednn_unified_bench.json \
  --output=jsonl
```

## Profiling helpers

Dump zebin metadata and inspect kernel execution properties:
```bash
SYCL_DUMP_IMAGES=1 SYCL_DUMP_DIR=debug_output/dump \
  ./build/bin/sycl-kernel-bench --kernel=dpas_baseline --dim_m=2048 --dim_n=256 --dim_k=4096

python3 tools/sycl-kernel-bench/parse_ze_info.py \
  --file debug_output/dump/.ze_info \
  --kernel _ZTSN10sycl_bench15dpas_kernel_tagIaaiLi8ELNS_17DpasMemoryPatternE2ELNS_11DpasGrfModeE0EEE
```

Estimate instruction mix and execution env (DPAS/send counts + ze_info):
```bash
python3 tools/sycl-kernel-bench/extract_kernel_metrics.py \
  --dump-dir debug_output/dump \
  --kernel _ZTSN10sycl_bench15dpas_kernel_tagIaaiLi8ELNS_17DpasMemoryPatternE2ELNS_11DpasGrfModeE0EEE
```

## Test matrix

The canonical test matrix lives in `test_matrix.json` with a human-readable summary in
`test_matrix.md`.

Validate the matrix:
```bash
./tools/sycl-kernel-bench/validate_matrix.sh
```

## DPAS exploration (ESIMD)

Configuration matrix for DPAS exploration lives in `dpas_configs.json`.

Example single run:
```bash
./build/bin/sycl-kernel-bench \
  --kernel=dpas_sweep \
  --quant=Q4_0 \
  --batch=1 \
  --dim_m=512 --dim_n=128 --dim_k=128 \
  --dpas-config=baseline_int8_r8 \
  --dpas-type-a=int8 --dpas-type-b=int8 --dpas-acc=int32 \
  --dpas-memory=direct_global --dpas-grf=128 --dpas-repeat=8 --dpas-ntiles=1 \
  --output=json
```

Run the full sweep (matrix + priority configs):
```bash
python3 tools/sycl-kernel-bench/run_dpas_sweep.py
```

Targeted sweep for dpas_memory_patterns across dims/repeats:
```bash
python3 tools/sycl-kernel-bench/run_dpas_memory_patterns_sweep.py \
  --dims 4096,8192,16384 --repeats 2,4,8 --patterns lsc_streaming,lsc_prefetch,lsc_prefetch2
```
On multi-GPU hosts, set `ONEAPI_DEVICE_SELECTOR=level_zero:1` to target the Arc device and avoid XeLPG DPAS2 compiler errors.

DPAS exploration outputs include `dim_*`, `ntiles`, `prefetch_dist`, `grf_mode`, and `type_acc` fields in CSV/JSON/JSONL for easy sweep analysis.

Summarize a sweep (best-per-pattern):
```bash
python3 tools/sycl-kernel-bench/summarize_dpas_sweep.py benchmark_results/.../dpas_sweep.jsonl
python3 tools/sycl-kernel-bench/summarize_dpas_sweep.py benchmark_results/.../dpas_sweep.jsonl --group-by-repeat
```

Emit CLI commands for DPAS device-opt tests:
```bash
python3 tools/sycl-kernel-bench/emit_dpas_tests.py
```

Device-aware tuning for `dpas_memory_patterns` (autotune covers `lsc_streaming` and `lsc_prefetch*`):
```bash
./build/bin/sycl-kernel-bench \
  --kernel=dpas_memory_patterns --dim_m=2048 --dim_n=256 --dim_k=4096 \
  --dpas-autotune --dpas-autotune-cache=benchmark_results/dpas_tuning_cache.jsonl \
  --output=jsonl
```
Use `--dpas-device-opt` for a quick heuristic (no sweep), `--dpas-autotune-force` to ignore cache, and
`--dpas-autotune-metric=throughput|bandwidth` to pick the scoring metric. Use
`--dpas-autotune-override-ntiles/--dpas-autotune-override-prefetch` to clamp results.
Heuristic defaults today:
- repeat-aware ntiles: repeat=1 -> 8, repeat=2 -> 4, repeat=4 -> 8, repeat=8 -> 4 (clamped to dim_n/16 and SLM capacity when relevant). Other repeats fall back to EU-based defaults.
- repeat-aware memory pattern: repeat>=4 -> lsc_streaming, repeat<=2 -> lsc_prefetch2. Other repeats fall back to EU-based prefetch distance.
- EU-based fallback: ntiles=4 for >=128 EUs, ntiles=2 for >=96 EUs, else 1; prefetch distance=3 for >=128 EUs, 2 for >=96 EUs, else 1.
- Dim-aware lookup: for specific (repeat, dim_n/16) pairs collected from sweeps, device-opt uses the table to select ntiles/memory/grf/acc (unless explicitly overridden; acc only applies to int8 inputs).

Best-known defaults from sweeps (int8 inputs; dim_n_tiles = dim_n/16):
- Default table (non-B580): repeat=2: dim_n_tiles=256 -> streaming, ntiles=4, grf=128, acc=float; dim_n_tiles=512 -> prefetch2, ntiles=8, grf=256, acc=float.
- Default table (non-B580): repeat=4: dim_n_tiles=128 -> streaming, ntiles=8, grf=128, acc=float; dim_n_tiles=256 -> streaming, ntiles=8, grf=256, acc=float.
- Default table (non-B580): repeat=4: dim_n_tiles=512 -> prefetch2, ntiles=4, grf=128, acc=int32; dim_n_tiles=1024 -> streaming, ntiles=8, grf=128, acc=int32.
- Default table (non-B580): repeat=8: dim_n_tiles=256 -> streaming, ntiles=4, grf=256, acc=float; dim_n_tiles=512 -> streaming, ntiles=4, grf=128, acc=float; dim_n_tiles=1024 -> streaming, ntiles=4, grf=256, acc=int32.
- B580 table: repeat=4: dim_n_tiles=128 -> streaming, ntiles=8, grf=128, acc=float; dim_n_tiles=256 -> streaming, ntiles=8, grf=128, acc=float.
- B580 table: repeat=8: dim_n_tiles=128 -> streaming, ntiles=4, grf=256, acc=float; dim_n_tiles=256 -> streaming, ntiles=4, grf=128, acc=float.

## Notes

- Coalesced layout requires `(K / block_size) % 32 == 0`.
- Q6_K coalesced is restricted to K multiples of 8192.

## Reference kernel hints

- `memory_bandwidth` uses `--bytes=<size>` or defaults to `batch GiB` when `--bytes` is omitted.
- `roofline_compute` uses `--elements=<count>` and `--ops=<count>` (defaults to `dim_m * dim_k` and 1000 ops).
