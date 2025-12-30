# iFairy vec_dot（非 LUT）Stage 4H：4-bank accumulator（使用 v8..v15）实验记录

目标：进一步打断 `sdot` RAW 依赖链，把低/高 nibble 与 v3/v4 两维都分离到 4 个 accumulator bank（代价：使用 callee-saved `v8..v15` + 更多合并 add）。

## 0) 实验实现摘要

- 位置：`ggml/src/ggml-cpu/arch/arm/quants.c::ggml_vec_dot_ifairy_q16_K`
- 新增 bank2/bank3：
  - bank0/bank1：低 nibble（idx0/idx1），使用 caller-saved（原有）
  - bank2/bank3：高 nibble（idx2/idx3），绑定到 `v8..v15`（callee-saved），并在 block 末尾做 4-bank 合并

## 1) 正确性

- `tmp/ifairy-stage4h-4bank/test-ifairy.txt`
- `tmp/ifairy-stage4h-4bank/test-ifairy_strict.txt`
- microbench verify（max_abs_diff=0）：
  - `tmp/ifairy-stage4h-4bank/vecdot_verify_k1536.txt`
  - `tmp/ifairy-stage4h-4bank/vecdot_verify_k4096.txt`

## 2) 性能（结论：回退）

microbench（k=1536, iters=50M, no-verify）：
- `tmp/ifairy-stage4h-4bank/vecdot_microbench_k1536.txt`: `ns/iter=48.53`
- `tmp/ifairy-stage4h-4bank/vecdot_microbench_k1536_run2.txt`: `ns/iter=48.55`

参考（上一轮 banking，k=1536）：
- `tmp/ifairy-stage3e-opt/vecdot_microbench_k1536.txt`: `ns/iter=46.75`

xctrace CPU Counters（k=1536）：
- `tmp/ifairy-stage4h-4bank/vecdot_cpu_counters_k1536.metric.summary.txt`
  - cycles: `10051647049`

参考（上一轮 banking，k=1536）：
- `tmp/ifairy-stage3e-opt/vecdot_cpu_counters_k1536.metric.summary.txt`
  - cycles: `9524211864`

## 3) 结论与处理

- 4-bank 在当前实现下 **明显回退**（~3–4%），推测主因：
  - 引入 callee-saved `v8..v15` 的保存/恢复开销
  - 额外的 4-bank 合并 `vaddq_s32` 开销
- 本实验不保留：后续应回滚到 2-bank（banking）版本，继续从 `tbl`/软件流水方向找收益。

