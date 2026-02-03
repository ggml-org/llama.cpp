# iFairy 文档索引

本目录汇总了 repo 内 iFairy 相关的设计/契约/性能记录文档，避免散落在仓库根目录。

## 从哪里开始

- **当前（V2）主线**：先看 `v2/IFAIRY_ARM_3W_LUT_V2_STATUS.md`（最新状态、复现命令、后续增量记录）
- **实现与约束（代码侧）**：`ggml/src/ggml-cpu/AGENTS.md`（语义不变量、路由/开关、验证门槛）
- **性能/trace 记录**：`perf/vecdot-fused6/IFAIRY_VEC_DOT_FUSED6_XCTRACE_PLAN.md`（计划与阶段性结果）
- **测试说明**：`tests/README-IFAIRY-TESTS.md`

## 目录结构

- `v2/`：V2 方案与“仍在维护”的文档（需要更新时只更新这里）
  - `v2/IFAIRY_ARM_3W_LUT_V2_REFACTOR_PLAN.md`
  - `v2/IFAIRY_ARM_3W_LUT_V2_LUT_C_INTEGRATION_PLAN.md`
  - `v2/IFAIRY_ARM_3W_LUT_V2_STATUS.md`
  - `v2/IFAIRY_ARM_3W_LUT_V2_XCTRACE_CPU_COUNTERS.md`
- `legacy/`：历史参考（内容不再修改；仅用于追溯思路/背景）
  - `legacy/IFAIRY_ARM_3W_LUT_DESIGN.md`
  - `legacy/IFAIRY_ARM_3W_LUT_API_PLAN.md`
  - `legacy/IFAIRY_ARM_3W_LUT_STATUS.md`
  - `legacy/IFAIRY_ARM_3W_LUT_ROADMAP.md`
- `perf/`：实验/跑分/trace 记录（按主题分组）
  - `perf/vecdot-fused6/`：vec_dot fused6 的 xctrace 计划与各阶段结果

