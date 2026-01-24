# iFairy ARM 3W LUT (V2) — 重构方案（仅保留最快路径）

Status: Draft (2026-01-24)

本文件是对当前 `ggml` 内 iFairy 3-weight complex LUT 路径的**重构方案**（不直接改现有 `IFAIRY_ARM_3W_LUT_*.md` 系列文档）。

目标：在保持现有最佳 tok/s 的前提下，显著降低 LUT 架构复杂度（env 变量/分支/布局过多），并为 `lut_c/` 目录中的 LUT 算法接入预留清晰的接口位点。

---

## 1. 背景：当前 LUT 数据流（现状梳理）

当前 iFairy LUT 路径大致分三段：

1) **权重侧 transform**（一次性 / 可缓存）
- 入口：`ggml_ifairy_lut_transform_tensor()`
- 现状：把 `GGML_TYPE_IFAIRY` 权重按 3-weight 分组编码为 per-row 的 `indexes`（每组 1 byte，内容为 6-bit pattern：`pat=c0|(c1<<2)|(c2<<4)`），缓存到 `tensor->extra`（`ifairy_lut_extra`）。
- 关键文件：`ggml/src/ggml-ifairy-lut-transform.cpp`, `ggml/src/ggml-quants.c::ggml_ifairy_3w_encode()`

2) **激活侧 preprocess**（每次 mul_mat / 每列 / 依赖 act）
- 入口：`ggml_ifairy_lut_preprocess_ex_*()`
- 现状：支持多个 LUT layout（legacy/compact/tbl64/merged64/sym16），生成 `lut` 与 `lut_scales`（每 block 2 个 scale：real/imag）。
- 关键文件：`ggml/src/ggml-ifairy-lut-preprocess.cpp`

3) **qgemm / accum**（mul_mat 核心）
- 入口：`ggml_ifairy_lut_qgemm_ex_*()`、`ggml_ifairy_lut_accum4_ex_*()`
- 现状：多 layout 多内核 + 多 env 微调（prefetch/unroll/fullacc 等），通过 `ggml/src/ggml-cpu/ggml-cpu.c` 的 dispatch 选择路径。
- 关键文件：`ggml/src/ggml-ifairy-lut-qgemm.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`

---

## 2. 痛点：为何需要 V2

### 2.1 运行时配置面过大
当前存在大量 env 变量：
- 路由/布局：`GGML_IFAIRY_LUT_LAYOUT`, `GGML_IFAIRY_LUT_KERNEL`
- 分块/全累加：`GGML_IFAIRY_LUT_BK_BLOCKS`, `GGML_IFAIRY_LUT_BM`, `GGML_IFAIRY_LUT_FULLACC`
- 微调与 A/B：prefetch / unroll / fastpath / acc 模式等若干

问题：
- 维护成本高（分支组合爆炸、文档难以同步）
- 很多 knob 只用于历史实验，长期默认值才是真正稳定的“最快路径”
- 为接入新算法（`lut_c/`）增加额外复杂度

### 2.2 代码路径分叉明显
- `ggml-cpu.c` 里存在大量 layout/qgemm/preprocess/accum dispatch
- `ggml-ifairy-lut-*.cpp` 内部也做 env 解析与 fallback

结果：
- “最快路径”被大量防御性分支包裹
- 阅读与调试负担重

---

## 3. V2 目标与非目标

### 3.1 目标（必须满足）
- **性能不回退**：以当前默认 auto 策略的最快路径为基线（Apple Silicon / `__aarch64__` + `__ARM_NEON__`），tok/s 目标 **±1% 以内**（以固定命令/固定 seed 复现）。
- **语义不变**：严格保持 `w * conj(x)`。
- **接口稳定**：不破坏 ggml 对外 API；尽量保持 `ggml_ifairy_lut_*.h` 对调用方不变。
- **配置收敛**：运行时 env 变量只保留极少数“开关级”选项（enable/debug/impl 选择），其余变为内部固定策略（或仅在测试中启用验证）。
- **便于接入 lut_c**：V2 必须引入清晰的“backend/driver”接口，把“权重 transform + preprocess + kernel”作为一个可替换单元。

### 3.2 非目标（明确不做）
- 不改动 `IFAIRY_ARM_3W_LUT_DESIGN.md` / `IFAIRY_ARM_3W_LUT_API_PLAN.md` 等旧文档内容（仅参考）。
- 不在本阶段引入新第三方依赖。
- 不在本阶段做跨平台（非 ARM）LUT 支持。

---

## 4. 设计：V2 的最小化架构

### 4.1 只保留“最快路径”的定义（V2 baseline）
以当前代码中 **默认 auto** 选择的主线路径为 baseline：
- Layout：`merged64`（group 内 64 pattern，每 pattern 4ch int8 打包）
- Kernel：`merged64`（包含 `n==1` decode fastpath）
- 必要 micro-opt：维持当前默认“开”的优化（prefetch、acc16、unroll 等）但**不再暴露为 env knob**，改为内部固定或受 CPU feature gating。

> 注：严格验证（historical strict/legacy）不作为 V2 的常驻生产路径；放到测试/调试机制中实现（见 §6）。

### 4.2 引入内部 backend 接口（为 lut_c 做接口位点）
建议新增内部接口（示意）：

- `struct ggml_ifairy_lut_backend`（仅在 `ggml/src` 内可见）
  - `name`
  - `can_mul_mat(...)`（含 feature/shape gate）
  - `transform_weights(...)`：为权重生成/缓存 backend 需要的额外数据（indexes / pack / transpose）
  - `get_wsize(...)`：返回 preprocess + kernel 需要的 workspace
  - `preprocess_ex(...)`：act → `lut + scales`
  - `qgemm_ex(...)`：核心 matmul
  - （可选）`accum4_ex(...)`：如果 baseline 需要 fullacc/分块方案

并把当前对外函数变为 thin wrapper：
- `ggml_ifairy_lut_transform_tensor()` → `backend->transform_weights()`
- `ggml_ifairy_lut_preprocess_ex()` → `backend->preprocess_ex()`
- `ggml_ifairy_lut_qgemm_ex()` → `backend->qgemm_ex()`

### 4.3 统一配置入口：threadpool config 只保留三类
V2 建议保留（或新增）最小 env 集合：
- `GGML_IFAIRY_LUT=0/1`：总开关（现有）
- `GGML_IFAIRY_LUT_IMPL=auto|merged64|lut_c`：选择 backend（V2 新增；默认 auto）
- `GGML_IFAIRY_LUT_DEBUG=0/1`：日志（现有）

测试/开发用途的验证建议：
- `GGML_IFAIRY_LUT_VALIDATE_STRICT=0/1`：保留名称但仅用于**测试/诊断**（见 §6），生产默认不走 legacy。

其余 env（layout/kernel/bk_blocks/bm/prefetch/unroll/...）进入 **Deprecated**，V2 不再读取或仅打印一次 warning（便于发现“旧脚本还在设置”）。

---

## 5. 迁移与落地步骤（分阶段，降低风险）

### Phase 0：建立 V2 骨架（无行为改变）
- 新增 backend 抽象与注册表（仅 1 个 backend：`merged64`，内部仍可调用现有实现）。
- `ggml-cpu.c` 的 dispatch 保持原样，但 V2 入口先旁路（或只在 debug build 开启）以便 A/B。

### Phase 1：默认切到 V2 merged64（功能等价）
- 在 `GGML_IFAIRY_ARM_LUT` 开启时，默认走 `GGML_IFAIRY_LUT_IMPL=auto` → `merged64` backend。
- 把“layout/kernel auto policy”与“qgemm 内部 env gating”集中到一个地方（backend init/config），避免重复解析 env。

### Phase 2：删掉非 baseline 的 layout/kernel（最小化）
- 删除或隔离以下路径（如确需保留，仅保留到 `test-ifairy` 的 reference 计算中）：
  - `legacy/compact/tbl64/sym16` 的 preprocess + qgemm 生产路由
  - `ggml_ifairy_lut_*_dispatch()` switch（变为直接调用 backend）
- `ifairy_lut_extra` 从“只存 indexes”演进为 “backend 私有数据指针 + tag”。

### Phase 3：配置收敛与文档切换
- 把旧 env knob 标为 deprecated（设置则 warn），文档只维护 V2。
- 将性能/trace 记录统一到 `IFAIRY_ARM_3W_LUT_V2_STATUS.md`。

---

## 6. 正确性与验证策略（V2 推荐）

### 6.1 生产路径：只跑 fastest backend
- 生产 runtime 不再依赖 legacy layout 做 strict gate（避免额外分支污染 hot path）。

### 6.2 测试路径：提供强一致性校验
建议在 `test-ifairy` 中提供两类校验：

1) **reference 校验**：和现有 `ggml_vec_dot_ifairy_q16_K_generic` 或等价 baseline 对比（逐元素 / 相对误差阈值）
2) **backend A/B**：当 `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 时：
   - 强制启用 reference（或旧 legacy 实现）并对每个用例 compare
   - 可选：同时跑 `merged64` vs `lut_c`（当 lut_c backend 接入后）

---

## 7. 性能基线与验收（建议写入 V2_STATUS）

建议固定至少两条命令作为“性能守门员”（示例，按你本地模型路径调整）：
- decode（N==1）：`./build-rel/bin/llama-bench ...`
- prefill（N>1）：`./build-rel/bin/llama-bench ...`（或选一个能代表 prefill 的 workload）

验收标准：
- `tok/s` 不低于 baseline 的 99%（同机器、同线程、同 seed、同参数）
- 若引入新 backend（lut_c），必须同时提供 merged64 与 lut_c 的对比日志

---

## 8. 与 lut_c 集成的接口契约（V2 预留点）

为保证后续“只换 backend，不改 ggml-cpu 调用侧”，V2 需要明确以下契约：

- 权重 transform 的产物由 backend 自己定义，但必须能缓存到 `tensor->extra`（并支持并发 transform）。
- preprocess 只依赖 act（以及 act_stride），输出 `lut + scales` 的 layout 由 backend 定义。
- qgemm 必须支持：
  - `pack_bf16` 输出（与现有一致）
  - `add` 语义（与现有一致）
  - 支持 `n==1` decode fastpath（关键性能点）

接入方案详见：`IFAIRY_ARM_3W_LUT_V2_LUT_C_INTEGRATION_PLAN.md`。

