# Vocoder（codec）NPU 加速可行性评估

2026-11-14。基于 SM6650（Snapdragon 6 Gen 4）实测数据的评估，回答三个问题：
vocoder 能否上 NPU、结构如何改才适合 NPU、需要哪些 NPU 算子。

## 1. 硬件与软件栈事实

| 项目 | 实况 |
| --- | --- |
| DSP | Hexagon v73，2×HVX 128B，**无 HMX**（无矩阵引擎），2 MB VTCM |
| 精度 | HVX 为整数 SIMD；fp32 只能模拟（约 2 倍代价），无 fp16 矩阵路径 |
| QNN 运行时 | 设备侧齐全：`/odm/lib64/libQnnHtp.so`、`libQnnHtpV73Stub.so`、`rfsa/libQnnHtpV73Skel.so`、`libQnnGpu.so`、SNPE 全套；FastRPC `libcdsprpc.so` |
| ORT | 1.23.2 定制版，providers = [CPU, NNAPI, XNNPACK]，`addQnn()` 抛异常——**QNN EP 未编入** |
| 本地工具链 | Docker Hexagon SDK 6.7.0.0（HVX/skel 构建可用，ggml-hexagon 就是用它）；**不含 QNN 头文件**（QNN SDK 是独立下载包） |
| NNAPI | 单算子实测 0.96 ms vs CPU 0.93 ms，未触达 DSP，排除 |

## 2. Vocoder 工作负载（设备 ORT profile，每 chunk 实测）

4 帧/chunk，CPU 4 线程共 200 ms/chunk（29 chunk ≈ 7.1 s/句）：

| 家族 | ms/chunk | 占比 | NPU 亲和性 |
| --- | ---: | ---: | --- |
| Conv（int8 QLinearConv + 4 个 fp32 尾层） | ~50 | 25% | 好（int8 conv 是 QNN 强项） |
| ConvTranspose（fp32，6 个） | ~50 | 25% | 中（无整数内核；MatMul 分解后变 GEMM） |
| Sin/Cos（Snake 激活 + RoPE） | 28 | 14% | **差**（超越函数，HVX 无原生） |
| 逐元素 Mul/Add/Pow | 37 | 18% | 中（带宽瓶颈，需全 int8） |
| 布局 Transpose/Concat/Slice | ~20 | 10% | 中（需重排） |
| Q/DQ + MatMul int8 | 16 | 8% | 好 |

要点：**这不是一个 GEMM 为主的负载**。GEMM 等价物（Conv+CT+MatMul）只占 ~53%，
其余 47% 是逐元素/超越函数/布局——HVX 上跑 fp32 逐元素是灾难，必须先整模型
int8 化并把 Sin 换成 LUT/多项式。

## 3. 实测：CPU vs HTP（本次新测，4 线程 CPU 对 2 HVX）

| 探针 | CPU | HTP0 | 结论 |
| --- | ---: | ---: | --- |
| codec 真实 conv 形状（384ch k=7，64 行，fp32） | 6.13 ms | 3.78 ms | HTP 快 **1.6×** |
| 692 个小 GEMV（depth 形状）f32 | 108.5 ms | 142.5 ms | HTP **慢** 1.3× |
| 同上 q8_0 | 43.3 ms | 97.2 ms | HTP **慢 2.2×** |
| 640 行 q8_0 GEMM（历史） | — | 15.8 ms | HTP 有效 10.6 GMAC/s |

小 GEMV/小算子上 HTP 反而输给 CPU——FastRPC 提交与调度开销吃掉算力。
HTP 只在大而批的 int8 GEMM 上有优势，幅度约 1.6–2×，**没有数量级差距**（v73 无 HMX 的直接后果）。

## 4. 架构约束：DSP 已经是 depth 的资源

流水线现状：depth stage 在 DSP 上跑（每句 ~5.4 s DSP 工作量），codec 在 CPU 上
与之**并行**。墙钟 = max(depth 6.92 s, codec 7.08 s) + backbone 0.92 s ≈ 8.0 s。

若把 codec 整体搬到 DSP，两个阶段在**同一颗 DSP** 上串行排队，工作量相加：

- 即便 codec 在 DSP 上快 2×（3.5 s）：墙钟 = 6.9 + 3.5 + 0.9 = **11.3 s，比现在慢 40%**
- 要打平，codec 的 DSP 加速比须 > 5.4×；实测上限是 1.6×（conv）和 0.45×（GEMV）

还有一条隐性账：CPU codec 与 DSP depth 并发时 DSP 侧 core_ms +20%（DDR 争抢）。
把 codec 挪到 DSP 不会减少总 DDR 流量，只是换个人发读请求，争抢甚至更糟。

**结论 1：在这颗芯片上，vocoder 整体上 NPU 是负收益。**

## 5. 更关键的：瓶颈即将转移

MatMul int8 实测（本轮）：codec 7.59→7.08 s，depth 7.08→6.92 s（DDR 减压红利），
墙钟 RTF 0.905→0.882。注意 **codec（7.08）只比 depth（6.92）多 0.16 s**——
把 CT→MatMul int8 做完（预计 codec 再降 ~0.5 s），瓶颈立刻变成 **depth 的
15 次串行 decode**。届时继续压 vocoder 对墙钟**毫无收益**。

**结论 2：vocoder 优化的墙钟上限只剩 ~0.2 s；下一步该压 depth 或 backbone。**

## 6. 结构改造清单（无论是否上 NPU 都值得做）

1. **ConvTranspose → 单 MatMul + 交错/pad**（已验证恒等，SNR 30 dB；恒等脚本
   `decompose_convtranspose.py`）：-25% kernel 时间、-80 MB/chunk 流量。
2. **Snake(x)=x+sin²(αx)/α → 多项式或 256 项 LUT**：Sin 占 kernel 时间 14%。
   用 sin²(a)=a²−a⁴/3+2a⁶/45（|a|<π，误差 1e-4）或分段 LUT。
   这是 CPU 提速项，**也是任何 NPU 移植的前置条件**（HVX/QNN 无 Sin 算子）。
3. **RoPE 的 sin/cos 查表**：位置有界，可预烘焙成常量 Gather。
4. **逐元素链 int8 化**（Mul/Add/Pow 的 per-tensor scale），把 47% 的非 GEMM
   部分也压到 int8 带宽。
5. **8 层注意力的 q/k/v 权重堆叠**成一次 MatMul（减少 op 数与提交）。
6. 21 个 conv cache 状态保持显式 state 输入（QNN/ggml 都能表达）。

## 7. NPU 路线与自定义算子（供将来 HMX 机型复用）

三条路线的可行性（按现有资产排序）：

| 路线 | 需要什么 | 评价 |
| --- | --- | --- |
| **ggml-hexagon 扩展** | 无需新 SDK；给 skel 加 conv1d/Snake/查表算子 | 唯一现在就能走的路；与 depth 共用 DSP，仍受串行约束 |
| ORT + QNN EP | 需下载 QNN SDK（开发者账号）重建 ORT | 设备 QNN 运行时齐全，缺构建侧 SDK |
| QNN C API 直调 | 同上 QNN SDK + 自写 context 管理 | 全图一次 FastRPC，调度最优，工程量最大 |

NPU 算子清单（任一路线都需要）：

- **Snake**：HVX 多项式（5 次乘加链）或 256 项 int8 LUT + 线性插值
- **sin/cos（RoPE）**：查表（位置≤2048，常量表 8 KB）
- **Conv1d 直卷积**（im2col-free）：省掉 im2col 的 DDR 流量，HVX 上 2× 以上收益
- **QLinearConvTranspose**：若不做 MatMul 分解则需要；分解后并入 GEMM
- **Q/DQ 折叠**：图级把 Quantize/Dequantize 融进相邻算子，省 4.8% 开销

## 8. 建议

1. **不上 NPU**（本机型）。把 CT→MatMul int8 做完（已在半路），然后把 Snake
   换成多项式/LUT——CPU 路径预计 RTF 0.88 → ~0.80。
2. 瓶颈转到 depth 后，下一个杠杆是 **depth 的 15 次串行 decode 合并**
   （模型图改造或投机解码）与 **htp-fusion 扩大融合**（吃掉每 decode ~2 ms 的
   FastRPC 提交开销）。`npu-gemm-batch-probe` 的数据（HTP GEMV 比 CPU 慢 2.2×）
   说明 depth 的小 GEMV 也不是 DSP 算力问题，而是提交开销问题。
3. 唯一值得考虑的 NPU 混合方案：只把 conv_6/conv_13 这类大权重带宽型层放 DSP
   （f≈11% 时与 CPU 侧负载平衡，理论墙钟 ~7.2 s）。但调度脆弱、收益 <0.8 s，
   不建议现在做。
4. 把第 6 节的结构改造做完后，模型即成为"QNN-ready"的全 int8 + LUT 激活图；
   将来换 HMX 机型（v75+，带矩阵引擎）可直接移植 QNN，预期收益 3–5×。

## 附：关键数据出处

- CPU↔HTP 探针：`codec-conv-probe`（384ch k=7 conv）、`npu-gemm-batch-probe`
  （692 GEMV）、`npu-roofline-probe`，2026-11-14 于 SM6650 实测
- vocoder op 分布：`M5_CODEC_PROFILE` ORT profile，mm-safe 模型，7 chunk 1397 ms
- 干扰测量：depth core_ms 35.9（无 codec）vs 42.9 ms/帧（codec 并发）
- QNN 栈盘点：`/odm/lib64/libQnnHtp*.so`、`/odm/lib64/rfsa/libQnnHtpV73Skel.so`

## 附：2026-11-14 晚更新（结构改造已落地）

评估中的第 6 节清单已执行完毕（详见 m5-tts-port-notes.md 的"Codec 深度量化"节）：

- ConvTranspose→MatMul：6/6 完成（`decompose_convtranspose.py`）
- 全 int8：68 个权重 MatMul + 27 个 Conv int8；fp32 仅剩 codebook 表（33.6 MB）
  与 17 个注意力激活 MatMul（质量敏感，必须保留 fp32）
- Snake 融合算子：`m5_snake_op.h` 单遍 NEON（范围归约 + 9 阶多项式），
  这就是 NPU 迁移所需的 Snake 自定义算子的**参考实现**——HVX 版本可直接
  复用同一数学（NEON intrinsics 换 HVX intrinsic，sin 无需 libm）
- 模型从 362 MB → 141 MB（每 chunk 权重流量 -61%），RTF 0.898 → 0.837

**迁移清单（QNN-ready 状态）**：模型已是全 int8 + 融合激活图。剩余迁移工作：
1. Snake 算子移植为 QNN Custom Op（把 m5_snake_op.h 的多项式转 HVX）
2. Conv1d/ConvTranspose 的 NHWC 布局检查（QNN HTP 偏好）
3. 21 个 conv cache 状态改写为 QNN state tensors
4. 需要 QNN SDK（开发者站下载）+ 带 QNN EP 的 ORT（或 QNN C API 直调）
5. 注意：v73 无 HMX，迁移收益有限（见正文第 3/4 节）；**换 HMX 机型（v75+）
   后预期 3–5×**，届时优先迁移 depth（15 次串行 decode 是新瓶颈）
