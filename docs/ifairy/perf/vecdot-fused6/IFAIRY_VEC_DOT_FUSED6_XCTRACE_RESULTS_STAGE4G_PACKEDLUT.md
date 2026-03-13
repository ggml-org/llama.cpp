# iFairy vec_dot（非 LUT）Stage 4G：packed-LUT（2×`tbl` + shift split）实验记录

目标：尝试把每个 nibble 的 `tbl` 从 4 次降到 2 次，通过 packed table（wr/wi 4-bit）+ `shl/sshr` 拆分得到 `wr/wi`，以降低 `tbl` 压力。

## 0) 改动点（实验版）

- 位置：`ggml/src/ggml-cpu/arch/arm/quants.c::ggml_vec_dot_ifairy_q16_K`
- 将原先 4 张 LUT（`wr0/wi0/wr1/wi1`）改为 2 张 packed LUT：
  - idx0: `lut_wr_wi_packed_idx0_data[16]`
  - idx1: `lut_wr_wi_packed_idx1_data[16]`
- 每个 nibble：
  - `tbl` 得到 packed bytes
  - `sshr` 取高 nibble（wi），`shl+sshr` 取低 nibble（wr）

## 1) 正确性

均通过（max_abs_diff=0）：
- `tmp/ifairy-stage4g-packedlut/test-ifairy.txt`
- `tmp/ifairy-stage4g-packedlut/test-ifairy_strict.txt`
- `tmp/ifairy-stage4g-packedlut/vecdot_verify_k1536.txt`
- `tmp/ifairy-stage4g-packedlut/vecdot_verify_k4096.txt`

## 2) 性能（结论：明显回退）

microbench（k=1536, iters=50M, no-verify）：
- `tmp/ifairy-stage4g-packedlut/vecdot_microbench_k1536.txt`: `ns/iter=56.85`
- `tmp/ifairy-stage4g-packedlut/vecdot_microbench_k1536_run2.txt`: `ns/iter=57.39`

参考（上一轮 banking，k=1536）：
- `tmp/ifairy-stage3e-opt/vecdot_microbench_k1536.txt`: `ns/iter=46.75`

xctrace CPU Counters（k=1536）：
- `tmp/ifairy-stage4g-packedlut/vecdot_cpu_counters_k1536.metric.summary.txt`
  - cycles: `11971426555`
- core type（ns）：`tmp/ifairy-stage4g-packedlut/core_type_sums.txt`

## 3) 结论与处理

- packed-LUT 方案在 M4（NEON + DOTPROD）上 **显著变慢**（~20% 量级），推测原因是额外 `shl/sshr` 指令与依赖链开销大于节省的 `tbl` 数量。
- 本方案不建议继续深挖，后续应回滚该实验改动，转向更“流水化”的 `tbl` 隐藏（或更深层的数据布局/软件流水）方向。

