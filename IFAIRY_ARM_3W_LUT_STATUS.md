# iFairy ARM 3‑Weight LUT · 现状与后续工作（NEON 标量混合版）

本文记录当前 `GGML_IFAIRY_ARM_LUT`（CPU-only）下 iFairy 3-weight LUT 的代码现状（含 NEON 加速实现）、可复现的 tok/s 记录、以及下一步工作列表。接口/路由约定见 `IFAIRY_ARM_3W_LUT_API_PLAN.md`，算法与数据结构见 `IFAIRY_ARM_3W_LUT_DESIGN.md`；80 tok/s 的分阶段路线图与实现方案统一收敛在 `IFAIRY_ARM_3W_LUT_API_PLAN.md` 的 `6.8`。

## 0. 快速使用（建议默认）

- 前提：构建时启用了 `GGML_IFAIRY_ARM_LUT`（`ggml/CMakeLists.txt` 会在 configure 阶段强制关闭 Metal/CUDA/HIP/MUSA/Vulkan/OpenCL/SYCL/WebGPU/zDNN 等加速后端，以保证 CPU-only）。
- 平台约束：当前 LUT 路由要求 `__aarch64__ + __ARM_NEON`；不满足时会回退（ARM32/无 NEON 不走 LUT）。
- 推荐性能基准：`./build-rel/bin/llama-bench`（tok/s 记录统一走 bench；`llama-cli` 仅用于 sanity-check）
- 推荐扫参脚本：`bash scripts/ifairy_lut_sweep.sh`（llama-bench 版；见脚本内 `TEST_MODE/N_PROMPT/N_GEN`）
- LUT 表布局默认走 `legacy`（更稳；`compact` 在部分设备/形状上更快），如需测试紧凑表：`GGML_IFAIRY_LUT_LAYOUT=compact`（见 1.1 / 0.1 记录）
- `BK/BM/FULLACC` 调参在不同形状/版本上波动较大：以 sweep 输出为准，不建议凭经验固定写死
- 每次修改 LUT 相关代码后，先做一次 `llama-cli` sanity check（固定 seed/prompt）确认不输出 gibberish（见 3.1）
- 一键复现（含 `test-ifairy`/strict/`llama-cli`/`llama-bench`）：`scripts/ifairy_lut_repro.sh`

**常用环境变量（LUT 路径）**

- `GGML_IFAIRY_LUT=0/1`：禁用/启用 LUT（默认启用）
- `GGML_IFAIRY_LUT_LAYOUT=legacy|compact|auto`：LUT 表布局选择（默认 `legacy`；`auto` 走默认策略）
- `GGML_IFAIRY_LUT_BK_BLOCKS=<int>`：K 维按 `QK_K=256` 的 block 做 tiling（0=禁用）
- `GGML_IFAIRY_LUT_BM=<int>`：M 维行块大小（仅 tiling 时生效）
- `GGML_IFAIRY_LUT_FULLACC=0/1`：tiled 下启用共享大累加器，减少重复 `preprocess + barrier`
- `GGML_IFAIRY_LUT_VALIDATE_STRICT=0/1`：严格对照 reference（用于验证，不用于性能跑分）
- `GGML_IFAIRY_LUT_DEBUG=0/1`：路由/形状诊断（默认关闭；跑分时不要开）
- `GGML_IFAIRY_LUT_PREFETCH=0/1`：控制 LUT 路径内的 prefetch（默认启用；设为 `0` 用于 profile/sweep 对照；覆盖 legacy/compact 的 `qgemm_ex/accum4_ex`）
- `GGML_IFAIRY_LUT_PREFETCH_DIST=<int>`：prefetch 距离（默认 `2`；设为 `0` 关闭距离预取；结合 profile 调参）
- `GGML_IFAIRY_LUT_N1_FASTPATH=0/1`：控制 `compact` 的 `N==1` fast-path（默认启用；设为 `0` 强制走通用路径，用于回归/调优 A/B）
- `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL=2|4`：控制 `compact` 的 `N==1` fast-path 里 group-loop 的 4-way unroll（默认 `4`；设为 `2` 用于 A/B，对照“2-way 是否反而更快”）
- `GGML_IFAIRY_LUT_KERNEL=auto|sdot|tbl|merged64`：强制选择 kernel 路径（默认 `auto`）；当前仅 `sdot` 在 `N==1` 快路上可用，`tbl/merged64` 为预留。

计划新增（未实现，先做文档约定）：

- `GGML_IFAIRY_LUT_LAYOUT=tbl64|merged64`：新增 LUT 布局（配合 TBL / merged64 方案），默认仍由 `auto` 策略决定。

## 0.0 当前共识（按优先级）

> 更新：`0ec52a5a` 之后出现大幅 tok/s 回归（见 `IFAIRY_LUT_PERF_REGRESSION_ANALYSIS.md`）。因此当前优先级以“先恢复到已验证过的高 tok/s 档位”为最高主线；地基工作并行推进，但不再驱动新的热路径改动扩面。

- P0（主线）：回归恢复 —— 先把 tok/s 拉回 `0ec52a5a` 档位（按 `## 4` 的 recovery steps 逐个回退/对照）。
- P0（并行）：可复现性与回归门槛 —— 固定命令/固定 seed/固定 build 目录；`test-ifairy + strict` 必跑，tok/s 必记录。
- P1：错误处理/路由健壮性 —— 只做“不影响热路径”的硬化与可观测性（现阶段已补齐一部分，见 `## 2` 摘要）。
- P2：可维护性重构/线程安全/生命周期 —— 在回归恢复完成后再推进（避免重构掩盖性能回归点）。

## 0.1 tok/s 记录（更新本文档时必填）

> 约定：每次改动 LUT 性能相关代码后，都在这里追加 **可复现** 的 tok/s 记录，避免“写了很多优化但没有数字/难复现”。
>
> 记录口径（强制）：
>
> - `git` 列优先写 **真实 commit hash**（`git rev-parse --short HEAD`），避免使用 `HEAD`（否则历史不可追溯）。
> - A/B 调优先用短测 `llama-bench --n-prompt 8 --n-gen 8` 做 `ABABAB` 交替跑，减少热漂移偏置；若短测明确更快，再用长测 `--n-prompt 128 --n-gen 256` 做 3-run 记录。
> - tok/s 会受温度/后台负载/系统调度影响，短时间内出现 `±20~30%` 甚至更大波动并不罕见；若 3 次连续跑出现明显“单调下降”，先冷却/关后台后重测，否则 A/B 结论无效。
> - **长测 3-run 必须给足冷却**：`--n-prompt 128 --n-gen 256` 往往会快速触发热降频；若看到 `min/max` 差距过大（经验上 > 20%）或出现明显 outlier，先延长间隔（例如 `sleep 60` 以上）再重测，否则“快/慢”结论很容易被热漂移污染。
> - A/B 原始日志建议用 `-o jsonl` 输出到 `/tmp/` 并在本文引用路径（避免只剩结论没证据）。
> - 性能结论以 `llama-bench` 的 `avg_ts`（tokens/s）为准。
>
> 回归/恢复判断优先对照 `0ec52a5a` / `0aeaa6c9` 节点（见 `IFAIRY_LUT_PERF_REGRESSION_ANALYSIS.md`）。

### 0.1.0 llama-bench 记录（当前口径）

**基准命令（CPU-only，固定 threads / n_prompt / n_gen）**

> 注：`llama-bench` 的 `n_ctx = n_prompt + n_gen + n_depth`；为避免口径漂移，固定 `--n-prompt/--n-gen`。

`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 128 --n-gen 256 -ngl 0 --device none --repetitions 1 --no-warmup -o jsonl`

**短测（A/B 调优，降低热漂移影响）**

`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 8 --n-gen 8 -ngl 0 --device none --repetitions 1 --no-warmup -o jsonl`

注：本阶段新增 `--n-gen 32` 记录，表中以 `tg32` 标注。

| time (UTC) | git | machine | threads | test | env | avg tok/s | log |
|---|---|---|---:|---|---|---:|---|
| 2025-12-22T16:05:53Z | `2337850b` | Apple M4 | 4 | pp128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0` | 1.41 | `/tmp/ifairy_bench_20251223T000553.jsonl` |
| 2025-12-22T16:07:24Z | `2337850b` | Apple M4 | 4 | tg256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0` | 6.76 | `/tmp/ifairy_bench_20251223T000553.jsonl` |
| 2025-12-20T08:00:08Z | `b9f0a57f+dirty` | Apple M4 | 4 | pp128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=legacy` | 3.00 | `/tmp/ifairy_bench_legacy_1766217492.jsonl` |
| 2025-12-20T08:00:08Z | `b9f0a57f+dirty` | Apple M4 | 4 | tg256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=legacy` | 14.72 | `/tmp/ifairy_bench_legacy_1766217492.jsonl` |
| 2025-12-20T08:00:08Z | `b9f0a57f+dirty` | Apple M4 | 4 | pp128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact` | 15.15 | `/tmp/ifairy_bench_compact_1766217565.jsonl` |
| 2025-12-20T08:00:08Z | `b9f0a57f+dirty` | Apple M4 | 4 | tg256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact` | 12.38 | `/tmp/ifairy_bench_compact_1766217565.jsonl` |
| 2025-12-22T02:48:46Z | `fe740e0a` | Apple M4 | 4 | pp128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=auto` | 19.20 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T104846.jsonl` |
| 2025-12-22T02:48:53Z | `fe740e0a` | Apple M4 | 4 | tg256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=auto` | 20.85 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T104846.jsonl` |
| 2025-12-22T02:49:14Z | `fe740e0a` | Apple M4 | 4 | pp128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=sdot` | 18.43 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T104914.jsonl` |
| 2025-12-22T02:49:21Z | `fe740e0a` | Apple M4 | 4 | tg256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=sdot` | 15.08 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T104914.jsonl` |
| 2025-12-22T03:34:07Z | `9de72065` | Apple M4 | 4 | pp128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=auto` | 17.11 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T113407.jsonl` |
| 2025-12-22T03:34:15Z | `9de72065` | Apple M4 | 4 | tg32 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=auto` | 12.77 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T113407.jsonl` |
| 2025-12-22T03:34:29Z | `9de72065` | Apple M4 | 4 | pp128 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=sdot` | 17.08 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T113429.jsonl` |
| 2025-12-22T03:34:36Z | `9de72065` | Apple M4 | 4 | tg32 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=compact GGML_IFAIRY_LUT_KERNEL=sdot` | 10.61 | `/var/folders/mf/jqbwxvls37d2lhmhhvht2_pm0000gn/T//ifairy_bench.20251222T113429.jsonl` |

### 0.1.1 legacy（llama-cli，已停更）

> 说明：以下 `llama-cli` 口径的记录/ABABAB 仅作历史参考；当前 tok/s 以 `0.1.0 llama-bench` 为准。

| time (UTC) | git | machine | threads | tokens | env | eval tok/s |
|---|---|---|---:|---:|---|---:|
| 2025-12-17T13:17:09Z | `0ec52a5a` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 15.39 |
| 2025-12-17T13:17:09Z | `0ec52a5a` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 16.99 |
| 2025-12-17T14:20:20Z | `0aeaa6c9` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 2.71 |
| 2025-12-17T14:20:20Z | `0aeaa6c9` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 4.75 |
| 2025-12-18T04:04:30Z | `34d8df05` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 5.01 |
| 2025-12-18T04:04:30Z | `34d8df05` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 4.98 |
| 2025-12-18T05:31:30Z | `79c915e5` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 18.33 |
| 2025-12-18T05:31:30Z | `79c915e5` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 19.47 |
| 2025-12-20T07:06:38Z | `d75031f1+dirty` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy` | 18.61 |
| 2025-12-20T07:06:38Z | `d75031f1+dirty` | Apple M4 | 4 | 256 | `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact` | 20.32 |
示例（长测 3-run，热漂移影响可见，非“最终结论”）：在 `1e1177a9` 下连续 3 次长测结果为 `legacy mean=14.58 min=14.23 max=14.99`；`compact mean=13.19 min=12.14 max=14.10`。

更新（长测 3-run，热漂移/outlier 示范，**不作为结论**）：在 `12c83d14` 下用 `legacy/compact` 交替跑且只 `sleep 8`，得到 `legacy mean=15.55 min=14.06 max=16.76`；`compact mean=13.23 min=9.32 max=16.69`（`compact` 出现明显 outlier）。这类结果应先冷却（加大间隔）后重测再谈 A/B。

更新（长测 3-run，冷却后可复现，作为当前 baseline 参考）：在 `12c83d14` 下用 `legacy/compact` 交替跑，`initial sleep 120 + 每次 sleep 75`，得到 `legacy mean=19.47 min=18.39 max=20.43`；`compact mean=17.02 min=16.12 max=18.24`。结论：当前 `compact` 仍未稳定胜出，下一步应按 `API_PLAN.md:6.1` 继续压 `qgemm_ex(compact)`。

更新（长测 3-run，冷却后可复现，作为当前 baseline 参考）：在 `8b02f3b4+dirty`（本地 `GGML_IFAIRY_LUT_PREFETCH` env cache 变更）下用 `legacy/compact` 交替跑，`initial sleep 180 + 每次 sleep 90`，得到 `legacy mean=19.45 min=18.99 max=20.23`；`compact mean=16.80 min=15.35 max=18.47`。说明：`compact` 波动仍偏大且均值仍落后，继续按 `6.1` 聚焦压 `qgemm_ex(compact)`。

（说明）`N==1` fast-path / unroll / prefetch 等 A/B 调优建议使用上面的短测命令做 `ABABAB` 交替跑；短测结果仅用于判断方向，最终结论仍以长测 3-run 为准（并在表中记录）。

**短测 ABABAB 汇总（`de40a2fb`，`compact`，`-n 64`）**

- `GGML_IFAIRY_LUT_N1_FASTPATH=1`：mean `19.62`，min `18.49`，max `20.34`
- `GGML_IFAIRY_LUT_N1_FASTPATH=0`：mean `20.36`，min `17.89`，max `22.42`
- `GGML_IFAIRY_LUT_PREFETCH=1`：mean `21.28`，min `20.54`，max `22.15`
- `GGML_IFAIRY_LUT_PREFETCH=0`：mean `20.94`，min `19.99`，max `21.43`

解读：上述两组 A/B 都存在较强重叠（尤其 `N1_FASTPATH` 的方差较大），暂不据此修改默认策略；若要据此做决策，需要在冷却后重跑短测，并用长测 3-run 做最终确认。

**短测 ABABAB（历史样本：unroll knob sanity，`8b5452b1+dirty`，`compact`，`-n 64`）**

- `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL=4`：mean `17.45`，min `15.37`，max `18.63`
- `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL=2`：mean `15.19`，min `14.45`，max `15.66`
- 结论：在该次复跑下 `unroll=2` 未显示优势；默认仍保持 `4`。后续如要据此做默认策略调整，必须在冷却后复跑并用长测 3-run 确认（避免热/负载噪声）。

**短测 ABABAB（减少冗余 mask：`c2 = pat >> 4`，`67e7c7e8+dirty`，`compact`，`-n 64`）**

- A（before）：`build-rel-a/bin/llama-cli` mean `23.08`，min `21.57`，max `24.71`
- B（after）：`build-rel/bin/llama-cli` mean `23.79`，min `21.33`，max `24.69`
- 备注：该次复跑中出现明显 “频率/调度 ramp”（前两次 ~21 tok/s，后两次 ~24 tok/s），因此目前只认为 **无明显回归**；若要据此做方向判断，需要在更稳定的热/负载状态下复跑并做长测 3-run 确认。原始日志：`/tmp/ifairy_lut_abab_compact_drop_and3_20251218T101156Z.tsv`。

**短测尝试（失败案例：避免重复踩坑）**

- 尝试：在 `compact` 的 `N==1` fast-path 中删除 4-way unroll，仅保留 2-way unroll（意图降低寄存器压力）。
- 结果：短测（`-n 64`，`GGML_IFAIRY_LUT_LAYOUT=compact`，`N1_FASTPATH=1`，`PREFETCH=1`）6 次复跑 `mean 20.12 -> 14.20`（约 `-29%`），且方差变大；已回退该方向，后续不要轻易将 4-way unroll 收敛为 2-way（至少在 Apple M4 上不成立）。
- 备注：已补充 `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL=2|4` 作为 perf-safe A/B 开关，后续复跑该方向不需要改代码。
- 尝试：在 `compact` 的 `N==1` fast-path 中，把每个 group 的 3 个 position 表（每个 16B）按 `int32_t[4]` 视角做“基址 + stride”访问（减少指针加法/地址计算）。
  - 对照：旧/新二进制交替跑两轮（`A=../llama_cpp_perfbase_406715f2/build-rel/bin/llama-cli`，`B=./build-rel/bin/llama-cli`），均为 `-n 64`，对比指标取 `llama_perf_context_print: eval time ... tokens per second`。
  - 结果：`ABABAB` 显示微弱正向（A mean `5.6467`，B mean `5.6667`；原始 TSV：`/tmp/ifairy_lut_abab_compact_addrchain_20251218T150914Z.tsv`），但 `BABABA` 反向后结论翻转（A mean `5.7667`，B mean `5.7367`；原始 TSV：`/tmp/ifairy_lut_bababa_compact_addrchain_20251218T151445Z.tsv`）。
  - 结论：该点在当前机器/条件下 **无稳定提升**（更像噪声/热漂移），已回退；后续不要在没有更强证据（profile 指令计数/IPC 或更大样本）前继续堆叠这类“微弱地址计算”改动。
- 尝试：`GGML_IFAIRY_LUT_LAYOUT=compact2`（实验性 2-lookups）：把 `pos0+pos1` 预合并成 16-way `int16` 表（`idx01 = pat & 0x0f`），`pos2` 仍为 16B `int8` 表（`c2 = pat >> 4`），希望把每 group 的查表从 3 次降到 2 次。
  - 结果：短测（`-n 64`）出现大幅回退：`compact mean=5.477` vs `compact2 mean=3.473`（原始 TSV：`/tmp/ifairy_lut_abab_compact_vs_compact2_20251218T153800Z.tsv`）；对 preprocess 做 NEON 向量化后仍显著落后：`compact mean=5.953` vs `compact2 mean=4.623`（原始 TSV：`/tmp/ifairy_lut_abab_compact_vs_compact2_v2_20251218T154608Z.tsv`）。
  - 结论：该方向在当前实现/机器上属于 **明确负收益**；已停止推进（不合入默认路径），后续只在“preprocess 不变或更便宜 + profile 明确显示 qgemm 指令瓶颈且缓存足够”时再考虑重启。

## 0.2 Xcode Profile（以 decode 场景为准）

> 目的：明确“该优化应该打在哪儿”，避免继续做低收益的微调。

**配置（你提供的条件）**

- 环境：`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=compact`
- 命令：`./build-rel/bin/llama-cli -m /Users/liweitao/Downloads/Codefield/cpp/llama.cpp/models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 128 -no-cnv`

**热点占比（Xcode 采样结果）**

- `ggml_ifairy_lut_qgemm_ex`：63%
- `ggml_graph_compute_thread`：24%
- `ggml_compute_forward_mul_mat`：6%
- 其他：< 2.5%

**近期复采样（波动示例）**

- sample 1：`ggml_ifairy_lut_qgemm_ex` 52%，`ggml_graph_compute_thread` 30%
- sample 2：`ggml_ifairy_lut_qgemm_ex` 69%，`ggml_graph_compute_thread` 12%
- 备注：profile 占比会随温度/调度/输入分布波动；用它做“定位主矛盾”，不要把单次占比当作精确 KPI。

**解读**

- 主要瓶颈已非常明确：继续提升 tok/s，优先级应集中在 `ggml_ifairy_lut_qgemm_ex`（降低每次 matmul 的单位成本）
- `ggml_graph_compute_thread` 的占比说明“线程调度/同步/图执行框架开销”也不可忽略；需要减少 barrier/减少 kernel 次数/减少不必要的工作区搬运

## 1. 当前现状（可工作的 LUT 路径：NEON 优先，标量回退）

- 路由位置：`ggml/src/ggml-cpu/ggml-cpu.c` 的 `ggml_compute_forward_mul_mat()` 内，当
  - `src0->type == GGML_TYPE_IFAIRY`
  - `src1->type == GGML_TYPE_F32` 或 `GGML_TYPE_IFAIRY_Q16`
  - `dst->type == GGML_TYPE_F32`
  - `K % QK_K == 0`
  - 且未被 `GGML_IFAIRY_LUT=0` 禁用
  时，走 LUT 路径（否则回退到原有 mul_mat）。
- 索引：`ggml/src/ggml-quants.c` 生成 **3-weight 直接 6-bit pattern**（1 byte 存 0..63）：
  - `pat = c0 | (c1<<2) | (c2<<4)`
  - 分组按 `QK_K=256` block 内部进行：`85` 个 triplet + `1` 个尾组（`{255, pad, pad}`），不跨 block、不丢维度。
- LUT 构表：`ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_preprocess()`
  - 支持两种表布局（通过 `GGML_IFAIRY_LUT_LAYOUT=legacy|compact` 选择，默认 `legacy`）：
    - `legacy`：每组构造完整的 `4 × 64`（`int16`）pattern 表（`512B/group`）
    - `compact`：每组构造紧凑表：`3 positions × 4 codes × 4 channels = 48B/group`（`int8`）
  - `compact` 的每个 position 是一个 16B 表：`tbl_pos[code*4 + 0..3] = {ac,ad,bc,bd}`（`int8`），其中 `code∈{0,1,2,3}` 对应 `(-1,0)/(1,0)/(0,-1)/(0,1)`
  - scale：每个 **block** 2 个 `float`（`d_real/d_imag`，被该 block 的全部 `86` 个 group 共享）
- GEMM：`ggml/src/ggml-ifairy-lut.cpp::ggml_ifairy_lut_qgemm()`
  - `legacy`：每 group 直接读取 `{ac,ad,bc,bd}`（`int16`）并 widen+accum 到 `int32`
  - `compact`：对每个 group 做 3 次 position 查表并相加得到 `{ac,ad,bc,bd}`，再 widen+accum；NEON 下使用 3 次 32-bit load（4B）+ `vaddw_s16` 走整数累加
  - 输出默认以 **bf16-pair packed in F32** 的方式写回（与现有 ifairy vec_dot 约定一致）。
  - 在 `__aarch64__ + __ARM_NEON` 下使用 NEON（否则走标量）。

### 1.1 选择 `legacy` 还是 `compact`？

- `legacy/compact` 在同一机器上多次运行会有一定波动：以 0.1 的 tok/s 记录为准；当前默认策略仍为 `legacy`（更稳）
- `compact` 的主要价值是显著降低 per-group LUT 带宽/工作集（`512B -> 48B`），后续要想稳定胜出，需要继续压低 per-group 的额外指令开销

### 1.2 为什么 LUT 是“四通道”而不是直接存实部/虚部？

当前 correctness-first 采用 `sum_ac/sum_ad/sum_bc/sum_bd` 四通道（本质是把复数乘法拆成 4 个可独立累加的基底和），原因：

- **严格复现 baseline 语义**：`ggml_vec_dot_ifairy_q16_K_generic` 在 `w * conj(x)` 下天然需要 `Σ(xr*wr) / Σ(xi*wr) / Σ(xr*wi) / Σ(xi*wi)` 四项，最后再组合成 `(out_r,out_i)`。
- **scale/系数无法在 LUT 阶段完全合并**：激活块有 `d_real/d_imag` 两套 scale，权重行还有 `d_real/d_imag` 两个系数；其中权重系数是 **per-row** 的，LUT 预处理是 **per-column** 的，不能把权重系数 bake 进 LUT，否则会退化成“每行一份 LUT”，内存/构表成本不可接受。
- **累加阶段不可消除**：点积跨 `K` 的求和必须在 qgemm 里做，因为每个 group 查哪个 `pat` 是由权重索引决定的；能做的优化是把乘法移出 inner-loop（当前四通道设计已经把权重系数的浮点乘法移到每个输出一次）。

## 2. 近期地基工作（摘要：不影响热路径的健壮性/一致性）

> 说明：本节只保留“对可复现与稳定性有直接帮助”的摘要；历史细节/大段公式/临时脚本不再堆在本文档中。

- 工作区 size/overflow 断言：为 `ggml_ifairy_lut_get_wsize()` 与 `ggml-cpu.c` 的 LUT 工作区切分补齐 overflow 断言，避免 size_t wrap 导致 silent 越界（`2a39f249`）。
- prefetch 可控：新增并打通 `GGML_IFAIRY_LUT_PREFETCH=0/1`，确保 legacy/compact 的 `qgemm_ex/accum4_ex` 所有 prefetch 点位都可完全关闭以便 profile 对照（`627dea55` / `46dcb0cb`）。
- env 解析收敛：将 LUT 相关 env 解析 helper 集中到 `ggml/src/ggml-ifairy-lut.h` 并复用，减少重复与语义漂移（`62a4ad8f`）。
- 错误可观测性：`transform_tensor` 在 debug 下对 shape/alloc/encode 失败输出原因，减少 silent fallback（`0f1af549`）。
- 路由边界更明确：LUT 路由要求 `__aarch64__ + __ARM_NEON`；并对无效 `GGML_IFAIRY_LUT_LAYOUT`、非法 `BK_BLOCKS/BM` 在 debug 下 warn/clamp（`34d8df05` / `10c98502`）。

## 3. 推荐验证方式（本地）

1) 重新编译（Release）  
`cmake --build build-rel --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`

2) 单测  
`./build-rel/bin/test-ifairy`

3) 运行 LUT 并开启严格对照（慢，但用于确认一致性）  
`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy`

4) CLI 快速 sanity（tok/s 与输出可读性）  
`GGML_IFAIRY_LUT=1 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`

5) 可选：BK/BM tile（用于探索 cache/带宽优化）  
`GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=2 GGML_IFAIRY_LUT_BM=64 ./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 -p "I believe life is" -n 16 -no-cnv`  
备注：`GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 时会自动禁用 tiling（strict 目前假设 full-K 单次计算）。

6) 回归（decode/布局/tiling 一致性）  
`./build-rel/bin/test-ifairy` 内置 `Test 5: iFairy LUT backend tiling regression`（会在测试内部设置 `GGML_IFAIRY_LUT=1`：先对比 `BK/BM` tiling 与非 tiling 的输出 bitwise 一致性；再对比 `N==1`（decode-like）下 `GGML_IFAIRY_LUT_LAYOUT=legacy` vs `compact` 的输出 bitwise 一致性；若 `GGML_IFAIRY_ARM_LUT` 未启用则自动跳过）。

7) 性能跑分/扫参（推荐）  
优先使用仓库脚本：`bash scripts/ifairy_lut_sweep.sh`（固定 seed/prompt，输出按 tok/s 排序，兼容 `tokens per second`/`tok/s` 两种日志格式）。  
如只复现单点（与 `0.1` 对齐）：直接使用 `0.1 tok/s 记录` 的基准命令即可。

### 3.1 常见问题：`llama-cli` 输出 gibberish（例如 `I believe life isDocuments CeUNTares cred`）

这类输出在 iFairy 上通常说明“算子输出已不可信”，常见原因：

1) **跑了旧的二进制**（二进制与源码/模型不匹配）：模型仍能加载，但类型表/算子实现落后，从而生成乱码  
2) **LUT 路径回归**（改了 LUT/调度代码但未做 sanity check）：算子返回了错误结果，采样就会变成乱码

**快速判断（看启动日志）**

- 若出现 `llama_model_loader: unknown type ifairy` 或 `print_info: file type   = unknown, may not work`，基本可以确定你在跑旧的 `llama-cli`。
- 正常情况下应看到 `print_info: file type   = IFairy`，且不应出现 `unknown type ifairy`。

**修复方式**

1) 重新编译你实际在用的那套 build 目录（不要只改源码不 rebuild）：

- `build-rel`（推荐）：
  - `cmake --build build-rel --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`
- `build`（如果你习惯用 `./build/bin/llama-cli`）：
  - 建议重新配置为 Release：`cmake -B build -DCMAKE_BUILD_TYPE=Release`
  - 然后编译：`cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`

2) 用固定 seed 做一次 sanity check（输出应可读）：

`./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf --gpu-layers 0 -t 4 -b 1 --seed 1 -p "I believe life is" -n 16 -no-cnv`

如果仍然是乱码：

- 先确认不是旧二进制（例如 `which llama-cli` / `ls -la build*/bin/llama-cli`）
- 再用 `GGML_IFAIRY_LUT=0` 复测：如果关闭 LUT 后输出恢复正常，则优先排查 LUT 路径（预处理/查表/写回/并行分工等）

## 4. 后续工作（按优先级）

> 更新：`0ec52a5a` 之后出现大幅 tok/s 回归，当前先以“回归恢复”为主线（详见 `IFAIRY_LUT_PERF_REGRESSION_ANALYSIS.md`）。本节先列 recovery steps，再列“恢复后继续优化”的原计划。

### 4.0 回归恢复（最高优先级：先回到 `0ec52a5a` 的 tok/s 档位）

每一步都必须：Release rebuild + `./build-rel/bin/test-ifairy` + `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy` + 在 `0.1 tok/s 记录` 追加一条记录。

1) **R0：恢复 `ggml_ifairy_lut_preprocess_ex` 的构表热路径（优先 `compact`）**  
   - 按 `0ec52a5a` 的 direct store 形态回退/重写构表实现，避免 `pack`/临时变量/`vcreate+vcombine` 的额外开销。
   - 验收：以 `0.1 tok/s 记录` 的“长测 3-run（min/max/mean）”为准；单次高 tok/s 只能视为 best-case，不作为稳定结论。

2) **R1：回退/对照 `qgemm_ex` 的 unroll 与 prefetch 策略**  
   - 先做 A/B：2-way vs 4-way unroll；prefetch unconditional vs conditional；以 tok/s 与 profile 为准。

3) **R2：暂时关闭 `N==1` fast-path（直到明确稳定增益）**  
   - decode 的快路要么“显著更快”，要么就先关掉，避免复杂度上升但吞吐下降。

4) **R3：回退/简化 decode 场景下的激活量化并行切分**  
   - decode (`N≈1`) 更怕调度/分片开销；先用更简单策略恢复基线，再讨论更细的并行化。

### 4.1（恢复后）继续优化（原计划）

> 目标：优先提升 Apple Silicon（ARM64 + NEON）的 tok/s，且不破坏 `w * conj(x)` 语义与现有输出一致性。

（优先级依据：见 0.2 的 Xcode Profile，`ggml_ifairy_lut_qgemm_ex` 占比 63%）

1) **把 63% 的热点继续压下去：优化 `ggml_ifairy_lut_qgemm_ex`（compact 优先）**  
   - 目标：减少每 group 的 load/widen/add 指令数与依赖链，减少 L1 miss（尤其是 decode：`N≈1`、每 token 都要跑一遍）
   - 任务拆解（不需要形状专用内核；目标是把 inner-loop 变“更像纯带宽”）：
     - **unroll + 多累加器**：对 group 循环做 2/4-way unroll，采用 `isum0/isum1` 交错累加，减少 load-use 依赖链
     - **减少地址计算**：把每个 position 的 16B 表当作 `4×int32`，用 `t0[c0]` 方式索引（减少 `*4`/LEA）
     - **prefetch 策略**：prefetch `grp + k_ifairy_lut_group_bytes` 与 `idx_g + k`；对比“prefetch 太早/太晚/无效”的差异（Xcode 可直接看到 L1 miss）
     - **N==1 快路（仍属于 LUT，不是形状模板）**：为 decode 常见 `N==1` 在 `qgemm_ex` 内加一个 runtime 分支，消掉 col 循环与部分指针运算
     - **减少 call/拷贝开销（仍属于 LUT）**：非 tiling 情况下避免“每 row 调一次 qgemm + memcpy”，改为每线程处理连续 row-block 并直接写回 `dst`
     - **SDOT 快路优化（实验）**：当前 `sdot` 慢于 `auto`，优先做“去分支 + 专用循环”与“布局对齐”验证：
       - 把 `sdot` 分支移出 inner-loop（拆成两套 loop，避免每 group 分支）。
       - 评估 `compact` 的 LUT layout：改为“4 groups × 16B”连续布局，以匹配 `vdotq_s32` 的 16-byte 向量宽度（减少 `vdupq_n_s32`/mask 开销）。
       - 仅保留 `sdot` 在 `N==1` 场景，并与 `GGML_IFAIRY_LUT_COMPACT_N1_UNROLL` 组合做 A/B。
   - 交付/验收：
     - `./build-rel/bin/test-ifairy` 全通过
     - 在 0.2 的 decode 配置下，`ggml_ifairy_lut_qgemm_ex` 占比下降（目标 < 55%），同时 eval tok/s 上升（目标 +10%）

### 4.2 近期进展（历史/与原计划相关）

- 修复 strict 验证误报：legacy `qgemm_ex` 的 strict reference 之前用错了 `ifairy` 权重 bit-pack 解码（导致 `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 直接断言失败）；已修正为与 `ggml-quants.c`/单测一致的 `chunk/lane/part` 解码。
- compact decode 优化尝试：为 `GGML_IFAIRY_LUT_LAYOUT=compact` 增加 `N==1` fast-path，并把 group 循环改为 4-way unroll（交错 `isum0/isum1`）。当前 sanity 输出正常、单测与 strict 全通过；但 tok/s 仍未回升到预期（见 0.1 最新两条记录）。
- 工程健壮性：把 `compact` 的 group bytes 统一为头文件常量 `GGML_IFAIRY_LUT_COMPACT_GROUP_BYTES`，并在 `ggml-cpu.c` 加入 `GGML_ASSERT(need == ggml_ifairy_lut_get_wsize(...))`，避免工作区切分公式漂移导致的 silent memory corruption。
- 小幅 hot-path 清理：`ggml_ifairy_lut_qgemm_ex` 的非 strict 路径不再计算 `act_blocks` 指针（strict 才需要 reference 对照）。
- decode 场景并行化：当 `src1=F32` 且 `N < nth`（常见 `N==1`）时，把激活量化从“按列分片”改为“按 K-block range 分片”，减少线程空转（`a3296bec`）。
- 可控 prefetch：新增 `GGML_IFAIRY_LUT_PREFETCH=0/1`（默认启用）用于 profile/sweep 对照，避免“盲目 prefetch”只能靠改代码来试（`627dea55`）。
- prefetch 工程修复：确保 `GGML_IFAIRY_LUT_PREFETCH=0` 能覆盖 legacy/compact 的 `qgemm_ex/accum4_ex` 全部 prefetch 点位，避免 fast-path 里出现“关不掉的 prefetch”（`46dcb0cb`）。
- overflow 断言：为 `ggml_ifairy_lut_get_wsize` 与 `ggml-cpu.c` 的 LUT 工作区切分补齐 size_t overflow 断言，避免 size wrap 导致的 silent 越界（`2a39f249`）。
- P1 小步：将 LUT 相关 env 解析 helper 集中到 `ggml/src/ggml-ifairy-lut.h`，并在 `ggml-cpu.c`/`ggml-ifairy-lut.cpp` 复用，减少重复与语义漂移（`HEAD`）。
- 错误可观测性：`transform_tensor` 在 debug 下对 shape/alloc/encode 失败给出明确日志，避免 silent fallback；并在路由时明确要求 `__aarch64__ + __ARM_NEON`（否则回退）（`HEAD`）。
- 配置健壮性：`GGML_IFAIRY_LUT_LAYOUT` 无效值在 debug 下只 warn 一次并回退默认；`BK_BLOCKS/BM` 的非法值在 debug 下提示并 clamp（`HEAD`）。
- SDOT 快路：新增 `GGML_IFAIRY_LUT_KERNEL=sdot`；已把 `N==1` 的 `sdot` 分支外提到 group-loop 外减少分支，但在 M4 + `compact` 上 `llama-bench` 仍低于 `auto`（见 0.1 表，`tg32` 10.61 vs 12.77），保持实验态。

2) **降低 `ggml_graph_compute_thread` 的框架开销（24%）**  
   - 目标：减少同步与小 kernel 调度开销，让更多时间落在“有效算术”上
   - 可做：
     - 继续减少 LUT 路径里的 barrier 次数（尤其 tiled/BK 版本），能用 `FULLACC` 解决的重复构表/重复同步尽量消掉
     - 检查是否存在 “很小但很频繁” 的算子（例如某些额外拷贝/转换）可以在 LUT 路径内合并或延后
     - 对 decode 场景（`N≈1`）评估线程数与切分策略：避免线程空转/争用（Xcode 能看到大量线程在等待就说明要改切分）

3) **减少重复 preprocess 与同步开销（先让 BK/BM “不再变慢”）**  
   - 当前已落地：NEON 构表（`pat` 维度向量化）+ NEON 累加（标量回退）。  
   - 已有实验性 BK/BM tiling，但在部分 workload 上会因 `preprocess + barrier` 频繁而变慢。  
   - 已实现一条“full accumulator” 的 tiled 路径（默认对小 `N` 自动启用，可用 `GGML_IFAIRY_LUT_FULLACC=0/1` 控制）：为整个 `M×N` 维护共享 `{ac,ad,bc,bd}` 累加器，使每个 K-tile 的 `preprocess` 只做一次（不再按 BM 行块重复），显著减少 barrier 次数。
   - 已实现（保持 LUT 路线）：
     - `preprocess` 多线程协作构表：`N>=nth` 按 col 切分；`N<nth` 按 group stride 切分以避免 false sharing（减少“线程 0 构表，其余线程等待”）
     - activation 量化（`src1=F32 -> ifairy_q16`）按 col 并行，减少 thread 0 独占量化带来的空转
   - 下一步优先项：
     - 在 tiled/BK 路径引入 **双缓冲 LUT + pipeline**（一部分线程构下一 tile，另一部分线程消费上一 tile），减少每 tile 的同步次数并尽量重叠构表与累加

4) **降低 LUT 工作区与带宽（提高上限）**  
   - 已完成一项低风险带宽优化：把 activation `d_real/d_imag` 的 `scales` 从“每 group 一份”改为“每 block 一份”（1 个 block 内 86 个 group 共享），显著减少 `scales` 读写与工作区占用。
   - 已完成一项结构性带宽优化：把原本的 `4×64 int16` per-group pattern LUT 压缩为 **`3×4×4 int8`（48B/group）**（`GGML_IFAIRY_LUT_LAYOUT=compact`），并在 NEON 下用 32-bit load（4B）做 position 查表 + 整数累加（`vqtbl` 版本曾尝试但在 M4 上不占优，暂不作为默认实现）。
   - 已完成一项低风险算术/带宽优化：在 `qgemm/accum4` 内先按 block 累加 `int32` 的 `{ac,ad,bc,bd}`，再做一次 `float` 转换与缩放（减少 per-group 的 `vcvt + mul`）。
   - 下一步（更激进）：在不破坏 `w * conj(x)` 语义的前提下，进一步减少 per-group 的指令数（例如把 3 次 position 查表的开销通过 unroll/流水化隐藏，或探索更“共享 LUT”的 canonical 方案）。

5) **索引生命周期/缓存策略升级（工程化与复用）**  
   - 已完成：`ggml_ifairy_lut_transform_tensor()` 把索引缓冲改为使用 `ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ...)` 分配，并缓存到全局 map（key: `{data ptr, nbytes, k, rows}`），同一份权重 data（含 view/复用场景）只生成一次索引。
   - 已完成：`ifairy_lut_extra` 增加 `index_buffer` 字段，区分 backend-buffer 分配与 legacy `ggml_aligned_malloc()`；`ggml_ifairy_lut_free()` 统一释放缓存 buffer + extras，避免 double-free。
   - 后续：如果要进一步“按 ctx/图生命周期”做精确释放，需要引入 refcount/弱引用或把 indexes 变成真正的 tensor（目前先以工程可用、复用明确为主）。

6) **再做 BM/BK 调参（把调参放在结构优化之后）**  
   - 在 1/2 完成前，单纯调 `GGML_IFAIRY_LUT_BK_BLOCKS / GGML_IFAIRY_LUT_BM` 往往波动大且不可复现；结构性开销下降后再调参更稳定。
   - 推荐方法：用脚本扫参，固定 threads/n_prompt/n_gen，直接输出按 tok/s 排序的结果：  
     `bash scripts/ifairy_lut_sweep.sh`  
     可通过环境变量覆盖：`THREADS=4 N_PROMPT=128 N_GEN=256 TEST_MODE=pg BK_LIST="0 1 2 4" BM_LIST="32 64 128" FULLACC_LIST="0 1" bash scripts/ifairy_lut_sweep.sh`
   - 备注：扫参会多次启动 `llama-bench`（每次都会加载模型），所以默认只扫少量组合；确认方向后再扩大 `BK_LIST/BM_LIST` 范围。

（贯穿）**测试与性能记录**  
   - 已补充 `tests/test-ifairy.cpp` 的 **CPU backend tiling 回归**：固定小形状（`K=512` 强制多 tile），对比 tiling 与非 tiling 输出 **bitwise 一致**。  
   - 继续补充 “LUT vs reference” 单测形状覆盖，并固定 `llama-bench` 命令/seed 记录 LUT=0 vs LUT=1 tok/s；必要时用 `llama-perplexity` 做对照。

## 5. 进一步性能提升路线图（参考 `BitNet/docs/lut-arm.md`）

> 注：本节属于“回归恢复完成后的长期路线图”。在 tok/s 未恢复前，先不要以此为导向继续扩展热路径复杂度。

`BitNet/docs/lut-arm.md` 的 ARM LUT 实现有几个很“工程化”的性能关键点：**int8 QLUT + `vqtbl` 查表**、**int32 累加**、**更少的 scale/元数据**、以及（在它的约束下）**对固定形状做专用内核**。（iFairy 的 `compact` 布局当前用 32-bit load 查表；`vqtbl` 版本曾尝试但在 M4 上更慢。）iFairy 这条 3-weight LUT 路径虽然语义/布局不同，但可以借鉴同样的方向来继续提速。

### 5.1 BitNet ARM LUT 的关键做法（可借鉴的点）

- **激活一次性量化 + 构表**：先对激活做 `max(abs(x))` 归一化得到 `lut_scales[0]`，再把激活量化并重排为 `QLUT`（查表友好、按 nibble 直接索引）。
- **`vqtbl` 代替“随机查表”**：把 LUT 排成 16-entry（或多表拼接）的 byte 表，靠 `vqtbl1q_s8` 做“向量化查表”，避免 NEON 不擅长的通用 gather-load。
- **累加与反量化分离**：K 维循环里只做整数查表与 `int32` 累加，最后统一用 `Scales/LUT_Scales` 做一次反量化到 float。
- **分块与工作区布局清晰**：`QLUT`、`lut_scales`、（可选）fp16 缓冲按固定顺序布置，线程 0 构表后 barrier，再并行 qgemm。
- **形状专用内核（取舍明确）**：BitNet 的 ARM 路径只覆盖少量固定 `(m,k)`，用编译期常量把 unroll/布局都做死，以换取吞吐。

### 5.2 映射到 iFairy：优先级最高的提升方向

1) **已完成：把“64-pattern × 4ch × int16” LUT 压缩到 `compact int8` 的形态（`48B/group`）**  
   - 通过“3 个 position 的可加性分解”（每个 code 只对 `{ac,ad}` 或 `{bc,bd}` 贡献 ±x），把 per-group LUT 从 `512B` 降到 `48B`。
   - `compact` 的 NEON 热路径当前用 32-bit load（4B）做 position 查表 + `int32` 累加；`vqtbl` 版本曾尝试但在 M4 上不占优。

2) **进一步降低 `preprocess` 的开销与同步频率（让 tiling 更稳定地赢）**  
   - 继续沿着当前 `FULLACC` 的方向，把 “一份 LUT / 多次消费” 做到更彻底：减少 barrier 次数、减少重复构表、把构表做成对 `K-tile` 的 pipeline（例如线程 0 预取/构表下一 tile，其余线程消费上一 tile）。
   - 参考 BitNet 的做法，尽量把 scale/元数据压到最少（例如：在误差可接受的前提下，把 per-block scale 进一步合并到 per-tile/per-col）。

3) **把热路径“常见形状”做成更激进的专用内核（可选，但上限高）**  
   - BitNet 选择“只做 matvec + 固定 (m,k)”来换取最强内核；iFairy 目前是通用 `mul_mat`（N 可变、K 可变），上限会被通用性拖累。  
   - 实际落地方式：先用 profile/日志统计 iFairy 推理中最热的 `(M,K,N)` 组合（例如 N 常为 1/2，K 常为 256 的倍数），然后为 1~2 组最热形状提供专用 fast-path（模板化 K-tile、固定 unroll、固定布局）。

4) **继续做 cache/预取与布局细化（低风险“小刀”）**  
   - 参考 BitNet 对工作区布局/对齐的强调：保证 LUT、indexes、scales 都是 64B 对齐；对 `indexes` 做更有针对性的预取；对 LUT 做 “按访问顺序” 的线性布局，降低 cache miss。  
   - 对当前 `accum4` 的 unroll/预取策略继续微调（配合 `scripts/ifairy_lut_sweep.sh` 扫参，记录稳定的 tok/s）。

### 5.3 建议的推进顺序（避免做无用功）

1) 先把 **`compact int8` LUT 压缩方案**做出一个 correctness 版本（严格对照 + 单测覆盖），明确误差与内存/速度收益。  
2) 再把它接入现有 `BK/BM/FULLACC` 框架，跑 sweep 找稳定配置，并用 `llama-bench` 固定 prompt/seed 记录 tok/s。  
3) 最后再决定是否要走 BitNet 那种“形状专用内核”的路线（收益高，但维护成本也高）。
