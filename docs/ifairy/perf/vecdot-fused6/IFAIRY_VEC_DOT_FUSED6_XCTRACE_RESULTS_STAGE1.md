# Stage 1 Results: `ifairy-vecdot-microbench` + `xctrace` chain (Time Profiler)

This records the “Stage 1” work from `IFAIRY_VEC_DOT_FUSED6_XCTRACE_PLAN.md`: add a dedicated vec_dot microbench and verify the `xctrace record → export → leaf summary` pipeline works (no vec_dot kernel changes yet).

## Environment

- Date (UTC): `2025-12-29T06:33:21Z`
- Repo commit: `1094369c`
- OS:
  - `macOS 26.2 (25C56)`
  - `Darwin Kernel Version 25.2.0 ... arm64`

## Build

Built target:

```bash
cmake -B build-rel -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-rel --target ifairy-vecdot-microbench -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

## Microbench sanity (no xctrace)

### Tensor-scale activations (all blocks share `d_real/d_imag`)

Command:

```bash
./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 500000 --warmup 5000 --seed 1 --x-scale tensor
```

Output:

```text
verify: ref=(48.500000, 12.000000) opt=(48.500000, 12.000000) max_abs_diff=0
ifairy-vecdot-microbench: k=1536 nb=6 iters=500000 warmup=5000 seed=1 x_scale=tensor
ns/iter=59.09 out=(48.500000, 12.000000) checksum=1.620500e+03
```

### Per-block activations (each block has its own `d_real/d_imag`)

Command:

```bash
./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 500000 --warmup 5000 --seed 1 --x-scale block
```

Output:

```text
verify: ref=(268.000000, -1440.000000) opt=(268.000000, -1440.000000) max_abs_diff=0
ifairy-vecdot-microbench: k=1536 nb=6 iters=500000 warmup=5000 seed=1 x_scale=block
ns/iter=60.29 out=(268.000000, -1440.000000) checksum=-1.883720e+05
```

## xctrace: record

Create output directory:

```bash
mkdir -p tmp/xctrace
```

### Time Profiler (tensor-scale)

Command:

```bash
xcrun xctrace record --template 'Time Profiler' \
  --output tmp/xctrace/ifairy-vecdot_tensor_nowindow.trace \
  --time-limit 8s --no-prompt \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench \
    --k 1536 --iters 1000000000 --warmup 5000 --seed 1 --x-scale tensor --no-verify \
  > /dev/null
```

Notes:
- `xctrace record` returned exit code `54` at time limit; trace file was still written successfully (expected behavior).

### Time Profiler (per-block scale)

Command:

```bash
xcrun xctrace record --template 'Time Profiler' \
  --output tmp/xctrace/ifairy-vecdot_block_nowindow.trace \
  --time-limit 8s --no-prompt \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench \
    --k 1536 --iters 1000000000 --warmup 5000 --seed 1 --x-scale block --no-verify \
  > /dev/null
```

Notes:
- `xctrace record` returned exit code `54` at time limit; trace file was still written successfully (expected behavior).

## xctrace: export + leaf summary

Leaf summarizer:
- `scripts/ifairy_xctrace_leaf.py` (parses `time-profile` schema)

### Tensor-scale

Export + summarize:

```bash
xcrun xctrace export \
  --input tmp/xctrace/ifairy-vecdot_tensor_nowindow.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py --top 20 \
  > tmp/xctrace/ifairy-vecdot_tensor_nowindow.leaf.txt
```

Leaf output (top):

```text
total_ms: 4958.000
 98.93%   4905.000 ms  ggml_vec_dot_ifairy_q16_K
  0.63%     31.000 ms  DYLD-STUB$$ggml_vec_dot_ifairy_q16_K
  0.32%     16.000 ms  main
```

### Per-block scale

Export + summarize:

```bash
xcrun xctrace export \
  --input tmp/xctrace/ifairy-vecdot_block_nowindow.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py --top 20 \
  > tmp/xctrace/ifairy-vecdot_block_nowindow.leaf.txt
```

Leaf output (top):

```text
total_ms: 7981.000
 99.06%   7906.000 ms  ggml_vec_dot_ifairy_q16_K
  0.59%     47.000 ms  DYLD-STUB$$ggml_vec_dot_ifairy_q16_K
  0.30%     24.000 ms  main
```

## Artifacts

- Traces:
  - `tmp/xctrace/ifairy-vecdot_tensor_nowindow.trace`
  - `tmp/xctrace/ifairy-vecdot_block_nowindow.trace`
- Leaf summaries:
  - `tmp/xctrace/ifairy-vecdot_tensor_nowindow.leaf.txt`
  - `tmp/xctrace/ifairy-vecdot_block_nowindow.leaf.txt`

## Conclusion (Stage 1)

- The microbench runs and `generic` vs `optimized` outputs match for the tested seed (`max_abs_diff=0` in both modes).
- The `xctrace(Time Profiler) → export(time-profile) → leaf summary` pipeline works, and the dominant leaf hotspot is `ggml_vec_dot_ifairy_q16_K` (~99%).

