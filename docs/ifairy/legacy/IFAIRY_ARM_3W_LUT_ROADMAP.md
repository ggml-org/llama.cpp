# iFairy ARM 3‑Weight LUT · 路线图（tbl64 优先，decode first）

本文档是 iFairy 3W LUT 的“候选布局/内核”路线图与落地计划，优先推进 `tbl64`（decode / `N≈1`）方向。

## 0. 文档定位（避免冲突）

- **现状/契约（以实现为准）**：`IFAIRY_ARM_3W_LUT_API_PLAN.md`（接口、数据约定、路由与线程模型；只描述已实现/已生效的行为）
- **性能记录与复盘证据**：`IFAIRY_ARM_3W_LUT_STATUS.md`（tok/s 表、raw 日志路径、失败案例与结论）
- **算法/数据结构设计**：`IFAIRY_ARM_3W_LUT_DESIGN.md`
- **候选方案与分阶段落地计划**：本文（`IFAIRY_ARM_3W_LUT_ROADMAP.md`）

规则：

- 本文允许描述“未实现/实验性”的内容，但必须明确 **env gating**、回退行为与验收口径。
- 一旦某项方案进入默认路径或改变 API/语义，必须同步更新 `IFAIRY_ARM_3W_LUT_API_PLAN.md`（从“路线图”升级为“契约”）。
- 所有跑分与 A/B 结论只记在 `IFAIRY_ARM_3W_LUT_STATUS.md`（避免本文变成结论堆积、难追溯）。

## 1. 不变量（任何候选方案都必须满足）

- 语义：严格匹配 baseline（`w * conj(x)`），并通过 `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 的对照测试。
- 回退：所有新布局/新内核必须可通过 env 一键回退到 `legacy/compact` + `auto`。
- 默认策略：默认不启用新的 layout/kernel；需要显式 env 或 `auto` 策略阈值明确且可复现。

## 2. 共同的 env gating 约定（现有 + 计划）

已存在（当前实现）：

- `GGML_IFAIRY_LUT_LAYOUT=auto|legacy|compact|tbl64|merged64`
- `GGML_IFAIRY_LUT_KERNEL=auto|sdot|tbl|merged64`
  - `sdot`：`compact` 的 `N==1` dotprod 实验内核
  - `tbl/merged64`：decode-first（当前主要用于 `N==1`）；`GGML_IFAIRY_LUT_LAYOUT=auto` 时可由 kernel 触发切到 `tbl64/merged64`（严格模式强制回退到 `legacy`）

约定：

- `strict` 模式下：强制回退到 `legacy`（避免引入未验证的优化路径）。
- `tbl64/merged64` 当前强制禁用 BK tiling（先保证 correctness + decode-first）。

## 3. tbl64（decode first）目标与范围

### 3.1 目标（为什么先做 tbl64）

`compact` 的 decode 路径当前每 group 需要 3 次 position 查表 + widen + add；tbl64 的目标是把每 group 的“查表次数”压到 **每个 channel 一次 64‑entry 表查表**，并用 NEON `vqtbl4q_*` 做批量查表，从而降低依赖链与指令数。

### 3.2 首期覆盖范围

- 场景：decode 优先（`N==1` 或 `N` 很小）
- 平台：`__aarch64__ + __ARM_NEON`（必要时可进一步要求 `__ARM_FEATURE_DOTPROD`，但 tbl64 本身不强依赖 dotprod）
- 形状：维持现有 `K % QK_K == 0` 约束与索引格式（pat 0..63）
- 模式：`strict` 必须正确；性能目标只在非 strict 下讨论

非目标（首期不做）：

- 不改索引编码（仍是 6-bit pattern）
- 不引入形状模板爆炸（除 decode‑first 的条件分支外）
- 不改变 tbl64 的默认策略：tbl64 仍要求显式 env 验证（当前 auto 默认偏向 merged64）

## 4. tbl64：数据布局草案（preprocess 输出）

### 4.1 直觉与工作集

- `legacy`：`4ch × 64pat × int16 = 512B / group / col`
- `tbl64`（拟）：`4ch × 64pat × int8 = 256B / group / col`
- `compact`：`3pos × 4codes × 4ch × int8 = 48B / group / col`

tbl64 的核心 tradeoff：用更大的 LUT（比 compact 大）换取更少的 inner-loop 指令与更好的 NEON 查表形态。

### 4.2 建议布局（便于 vqtbl4q）

对每个 `(col, group)` 存 4 个 channel 的 64‑entry `int8` 表：

- channels：`ac, ad, bc, bd`
- 每个 channel：64 bytes，按 4×16B 向量组织，便于直接 load 到 `int8x16x4_t` 后用 `vqtbl4q_s8`

建议内存布局（连续存放，便于预取/顺序访问）：

```
tbl64[group][ch][vec][lane]
  group: 0..groups-1
  ch   : 0..3  (ac, ad, bc, bd)
  vec  : 0..3  (16B chunks; together form 64-entry table)
  lane : 0..15
```

其中 `pat`（0..63）直接作为 `vqtbl4q` 的 index（一个字节即可），无需额外拆分/mask。

## 5. tbl64：落地步骤（最小可验证切片）

### 5.1 P0：最小 correctness（先跑通，已落地）

1) **新增 layout 枚举与路由**
   - 在 layout parsing 中加入 `tbl64`（仅当显式 env 设置时生效）
   - `ggml_ifairy_lut_get_wsize()` / 工作区切分支持 tbl64 的 `lut_bytes`

2) **preprocess：生成 tbl64**
   - 复用现有 `compact` 的 position 语义：对每个 `pat`（0..63），把 `c0,c1,c2` 分解后累加出 `{ac,ad,bc,bd}` 的 `int8` 值
   - 先用标量实现确保正确；再决定是否需要 NEON 向量化构表

3) **qgemm：decode‑first 内核（N==1）**
   - 先实现“标量 tbl64”作为对照（便于定位 bug）
   - 再实现 NEON：批量读取 `pat`（例如 16 个），用 `vqtbl4q_s8` 为每个 channel 做 64‑entry 查表，然后 widen/accum 到 `int32`

4) **strict gate**
   - `GGML_IFAIRY_LUT_VALIDATE_STRICT=1` 下必须 bitwise 一致（或至少与当前 strict 的定义一致）

验收（P0）：

- `./build-rel/bin/test-ifairy` 通过
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy` 通过
- `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_LAYOUT=tbl64 GGML_IFAIRY_LUT_KERNEL=tbl` 能跑通（必要时仅 decode / `N==1` 生效，其他形状回退）

### 5.2 P1：性能与稳定性（decode 优先）

目标：tbl64 在 decode（`N==1`）场景下稳定胜过 `compact/legacy`，且不引入大幅 preprocess 开销回退。

建议实验矩阵：

- layout：`legacy`, `compact`, `tbl64`
- kernel：`auto` vs `tbl`（仅 tbl64 下）
- knobs：`GGML_IFAIRY_LUT_PREFETCH`, `GGML_IFAIRY_LUT_PREFETCH_DIST`

记录口径：

- tok/s 只写 `IFAIRY_ARM_3W_LUT_STATUS.md` 的 `0.1` 表（并附 `/tmp/` raw 日志路径）

### 5.3 P2：auto 策略与回退体验

在明确“哪些形状/哪些机器” tbl64 稳定赢后，再考虑：

- `GGML_IFAIRY_LUT_LAYOUT=auto` 下的启用门槛（例如仅 decode + 某些 cache/工作集条件）
- debug 输出：当用户指定 `tbl` kernel 但 layout 非 tbl64 时的提示（减少误用）

## 6. 风险与检查点（tbl64 特有）

- **preprocess 成本**：tbl64 构表 64‑entry，若 preprocess 变成主矛盾，整体可能变慢；需要 profile 证明收益来自 qgemm 而不是“把热点搬家”。
- **工作集变大**：256B/group 可能导致 L1 压力；必要时需要配合 BK tiling / FULLACC 或更激进的 pipeline。
- **实现复杂度**：NEON tbl 查表 + widen/accum 易出错；必须优先把 strict 对照做成“第一道门槛”。

## 7. 后续候选（简要）

- `merged64`：已落地并验证可用（pattern → `{ac,ad,bc,bd}`），当前 `N==1` 场景表现更好；后续主要工作是完善 auto 策略与 tiling/工作集门槛。
- 批量索引解码：对 `pat` 做更激进的向量化读取/预取，减少前端开销。
- 框架/同步：结合 `IFAIRY_ARM_3W_LUT_API_PLAN.md` 的 `6.1/6.4`，避免“算子更快但被 barrier 吞掉”。
