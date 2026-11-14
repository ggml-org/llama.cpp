# M5-TTS 优化尝试总账（2026-11-14）

> 本文是 M5-TTS（Snapdragon HTP/llama.cpp 移植）性能与工程优化的**结构化台账**：
> 每一项尝试的方法、实测数字、结论与状态。负结果（排除的路线）单独成章——
> 它们定义了问题的边界，和成功项同等重要。
> 时间线细节与操作步骤见 `m5-tts-port-notes.md`；本文只回答"试过什么、结果如何、现在到哪了"。

**测试约定**（贯穿全文）：
- 性能对比一律**同轮交替 A/B**（设备热漂移可让同代码 RTF 0.75→0.91，跨轮绝对值不可比）；
- 生成正确性门禁 = codes 确定性 + ASR 转写 + 音频统计族（RMS/过零率/峰值）；
- codes 与 Java 参考的逐位一致性在 f16 heads 上线后**明确放弃**（f16/int8 数值扰动
  会让自回归采样分叉——同分布的另一条轨迹，不影响可懂度）。

---

## 1. 总账

### ✅ 成功项（已上线为默认）

| # | 尝试 | 关键数字 | 状态 |
|---|------|----------|------|
| 1 | **HTP 环境旋钮固化** `GGML_HEXAGON_OPPOLL=1 / NHMX=0` | 同句 RTF 1.62→0.75（**2.2×**）；丢失会静默慢一倍 | deploy.sh 写死 |
| 2 | **codec 三步量化 + Snake 融合**（CT→MatMul、权重 int8、融合 Snake 算子） | RTF 0.898→**0.837**；codec 7.49→**4.68 s**（115 帧句）；体积 362→141 MB | 默认部署 |
| 3 | **f16 heads（FMLAL 内核）**——含宏名事故修复 | 每帧 -16%（57.3→48.3 ms）；RTF -11~13%（5/5 句） | 默认部署 |
| 4 | **heads int8 量化**（per-row 对称，NEON） | heads_s 0.94→**0.38 s（2.5×）**；RTF 0.86→**0.75（-13%）** | 默认部署（`M5_HEADS_INT8=0` 回退） |
| 5 | **codec ORT 线程调优** | 2/3/4 线程 = 3.56/2.84/3.20 s → **3 最优** | 即默认值 |
| 6 | **按请求 chunk 策略**（wav 4/16、pcm 2/4） | wav 路径 codec ONNX 调用 18→5 次；pcm 保首包 0.33-0.59 s | 默认部署 |
| 7 | **WavWriter 三模式**（文件/内存/丢弃） | wav 响应零磁盘 I/O；pcm 流式不再白写临时 wav | 默认部署 |
| 8 | **deploy.sh 一键部署** | 停服→推二进制/web→备份模型→固化 env→启动→forward；设备重启后一条命令恢复 | 已固化 |
| 9 | **--depth-chain 协议**（16-book 链进子进程，1 次往返/帧） | 与 legacy **逐位一致**（宿主 200 组 0 mismatch）；性能打平（-1.5%）→ 保留为批量/量化落点，默认仍 --depth-pipe | 代码保留，opt-in |
| 10 | **X-M5 分项统计头** + 埋点修复 | Backbone-S/Depth-S/Codec-S 回传；`g_heads_s` 双计 bug 修复 | 默认部署 |
| 11 | 工程小项：voices.json mtime 缓存、CORS 头去重、GET / no-store、pcm 错误预校验、Web 统计显示修复（`Access-Control-Expose-Headers`） | — | 默认部署 |

### ❌ 负结果（实测排除，详见 §3）

| # | 尝试 | 结论 |
|---|------|------|
| 12 | **depth 15 步串行 decode 的六条精确压缩路径**（批量合并 / 投机草稿 / Jacobi 迭代 / 跨帧批量 / 后端融合 / 哑输入） | 全部实测排除，见 §3.1 |
| 13 | **HTP 后端旋钮 sweep**（OPBATCH/OPQUEUE/OPFUSION/NHMX/NHVX/POWER…） | OPPOLL 之外无增益，当前配置已最优（§3.2） |
| 14 | **m5_ring 共享内存环** | 比匿名管道慢（futex 等待拖累 HTP 提交路径），维持 opt-in |
| 15 | **噪声预生成向量化** | JavaRandom 位精确约束不可向量化；实测占比 ≤2%，收益不足以破约束 |
| 16 | codec 尾部 CT 保留 fp32 / 激活×激活 MatMul 量化 | 前者无收益（12.78 vs 12.75 dB）；后者掉 1.3 dB（Softmax 敏感） |
| 17 | 跨请求 KV 复用 / voice embedding 预计算 | 文本不同无法复用；预计算收益 <2 ms |

### 🔬 待研究（复活条件已写明）

| 方向 | 解锁的收益 | 条件 |
|------|-----------|------|
| **学习型草稿**（code[b]←code[b-1] 小映射器） | depth 核心 2.1×（RTF ~0.55） | 需训练一个小 draft 模型 |
| 改 depth 模型图（一次前向出多 book） | 同上 | 需重训 |
| codec 权重 int8 化保 QLinearConv 快速路径 | codec 流量减半 | 动态量化路已试不通（§3.3） |

---

## 2. 性能演进与当前账目

### 演进链（同句口径注明，绝对值随热状态漂移）

| 时点 | 配置 | RTF | 备注 |
|------|------|-----|------|
| 11-14 早 | 基线（qdq-ex4 codec，f32 heads，无 OPPOLL） | 0.898（CLI 长句） | JNI 0.964 |
| 11-14 午 | + codec snake 融合 | 0.837 | codec 7.49→4.68 s |
| 11-14 傍晚 | + f16 heads 修复 + HTP env 固化 | **0.717** | 11.0 s 长句，首包 0.526 s |
| 11-14 夜 | + heads int8 | **0.75**（24 字句 A/B）/ 0.798（server） | f16 同口径 0.86 → -13% |

### 当前每帧账目（24 字句 / 70 帧 / HTP0，int8 heads）

| 环节 | 每帧 | 占比 | 说明 |
|------|-----:|-----:|------|
| depth 15 次串行 decode | ~41 ms | ~62% | 成本模型：1.25 ms/次提交 + 0.91 ms/行（depth-bench 拟合） |
| backbone（prefill+decode 摊销） | ~9 ms | ~13% | 33 MB Q8_0 |
| heads（int8 后） | ~5.4 ms | ~8% | 原 13.4 ms；权重流量 34 MB f16→17 MB int8 |
| codec（与 depth 重叠） | ~42 ms | 0%（隐藏） | depth 再提速才会浮出 |

**结论：depth 的 15 次串行提交仍是唯一的大头**，而它的精确压缩路径已被 §3.1
六条证据排除——不改模型/不训练草稿就到头了。

---

## 3. 负结果详录

### 3.1 depth 15 步串行 decode：六条压缩路径全部排除

成本模型（`depth_batch_bench.cpp`，HTP0，28 行四种分解）：

| 形状 | 总耗时 | 含义 |
|------|-----:|------|
| 1×28 | 26.7 ms | 一次提交 |
| 14×2 | 42.6 ms | 现状近似 |
| 28×1 | 60.4 ms | 逐行提交 |
| 1×2 | 3.07 ms | 单调用基线 |

拟合 **F ≈ 1.25 ms/次提交 + R ≈ 0.91 ms/行**。现状（15 调用 16 行）≈33 ms/帧，
理想批量（1 调用 16 行）≈16 ms/帧 → **理论 2.1×**（RTF ~0.55）。但：

1. **批量合并（哑输入）**：`M5_ZERO_TRANSFORM` 消融把 15 步的 code 条件 transform
   行全部置零——codes 翻转 **99.6%**（首帧 15/15 全翻），音频 70→500 帧发散不终止。
   transform 行完全承重（含 EOS 决策），串行链是模型语义的一部分。
2. **投机草稿（上一帧 codes）**：帧间 codes 匹配率仅 **0~1.5%**（近随机，lag 1-5
   一致）——codec codes 无时间局部性，草稿无预测力。
3. **Jacobi 迭代**：沿 15 步链传播，收敛需 O(15) 轮，比串行更差（分析排除）。
4. **跨帧批量**：`next_hidden`（codes 表查找和）反馈给下一帧 backbone 输入，
   帧间被自回归锁死。
5. **后端融合**：OPFUSION 默认已全开；0 vs 1 只差 5%（44.6 vs 42.5 ms），
   深度图太小，减 op 数量见底。
6. **--depth-chain 减往返**：16 次 pipe 往返→1 次，逐位一致验证通过（宿主
   200 组随机输入 0 mismatch；设备 chain==legacy codes/wav 双哈希相同），
   但实测性能打平（depth 4.15 vs 4.08 s）——瓶颈是权重带宽/提交开销而非
   往返次数。协议保留为后续批量化的落点。

> 复活条件：学习型草稿模型（code[b]←code[b-1] 映射器，verify 批量接受），
> 或重训 depth 头支持多 book 并行解码。在此之前，任何"把 15 次合成 1 次"
> 的尝试都应先对照本节证据。

### 3.2 HTP 旋钮 sweep：当前配置已最优

`ggml-hexagon.cpp` 全部环境旋钮，depth-bench 同轮对照（28x1 / 14x2 / 1x28）：

| 配置 | 28x1 | 14x2 | 1x28 | 结论 |
|------|-----:|-----:|-----:|------|
| **baseline（OPPOLL=1, NHMX=0）** | **60.6** | **42.6** | **26.8** | 基准 |
| OPPOLL=0 | 111.3 | 69.7 | 31.4 | **-1.8×，OPPOLL=1 必须保留** |
| NHMX=1 | 61.4 | 42.8 | 26.8 | 无差 |
| OPBATCH=16 / 64 | 76.3 / 64.6 | 52.6 / 44.7 | 27.1 / 26.7 | 更慢 |
| OPQUEUE=1 / 2 | 60.6 / 61.0 | 42.7 / 42.9 | 26.7 / 26.7 | 无差 |
| OPFUSION=0 | 64.6 | 44.7 | 27.1 | 略慢（默认 1 正确） |

（更早的后端参数全扫 OPQUEUE 1-32、HOSTBUF、NHVX 4、POWER_MODE 亦在 ±1% 噪声内。）

### 3.3 codec 量化的两个死胡同

- **尾部 CT 保留 fp32**：12.78 vs 12.75 dB，无收益 → 全量化即可。
- **激活×激活 MatMul（QK^T/AV）量化**：掉 1.3 dB（Softmax 敏感）→ 17 个必须排除。
- **动态量化保 QLinearConv 快速路径**：路不通（权重流量 int8 化的前置条件不成立）。

### 3.4 其他

- **m5_ring**：共享内存环在本 SoC 上比管道慢（futex 等待拖累 HTP 提交路径）。
- **噪声生成向量化**：JavaRandom 顺序敏感（位精确约束），且实测占比 ≤2%（≤1.4 ms/帧）。
- **voice embedding 跨请求预计算**：<2 ms，不值得。

---

## 4. 成功项要点（方法与门禁）

### 4.1 f16 heads：从死代码到真收益

- **事故**：f16 内核守卫用了 GCC 风格宏 `__ARM_FEATURE_FP16[_FML]`，NDK clang 19
  实际定义 `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` / `__ARM_FEATURE_FP16_FML`——
  宏从未匹配，f16 是死代码；当时的 objdump 检查误报、"RTF 持平"假象。
- **修复**：守卫改宏名 + `-march=armv8.2-a+fp16+fp16fml`（CMakeLists 全局），
  objdump 128 条 fmlal 确认。交替 A/B（同种子）：每帧 -16%，RTF 5/5 句改善。
- **教训**：编译期优化必须以**反汇编/符号**验证，不能只看构建日志；宏名要按
  实际编译器的特性宏写。

### 4.2 heads int8（本轮新增）

- **动机**：heads_s 13.4 ms/帧 ≈ 纯权重带宽（34 MB f16 @ ~2.5 GB/s）。
- **实现**：`m5_depth_heads.h` 的 `build_int8`（per-row 对称量化，max/127）+
  `project8`（NEON，int8 权重→f32 累加；clang 19 无 `vcvtq_f32_s16`，
  用 `vmovl_s16 + vcvtq_f32_s32` 中转）；legacy 与 chain 子进程两路都接。
- **A/B**（同轮交替 ×2）：heads_s 0.94→0.38 s（2.5×），RTF 0.86→0.75（-13%）。
- **质量门禁**：
  - 首帧 codes 0/16 翻转 → 量化数值正确、无系统性误差；
  - 帧 2 起经 `next` 反馈链混沌分叉至 ~90% 翻转（帧数 70→75）——与换随机
    种子同级的轨迹分叉，**非误差累积**（同分布另一条采样轨迹）；
  - ASR 两句全对（24 字长句 + 疑问句），RMS/ZCR/峰值同族，模式内确定性 0% 翻。
- **开关**：默认 ON，`M5_HEADS_INT8=0` 回退 f16。

### 4.3 测量工具链（可复用）

| 工具 | 用途 |
|------|------|
| `depth_batch_bench.cpp` | llama_decode 固定提交成本 vs 每行成本分解 |
| `M5_HEAD_STATS=1` | heads_s / stage_call_s 分项 |
| `M5_ZERO_TRANSFORM=1` | transform 行消融（串行依赖敏感度） |
| `--dump-codes` | codes 逐帧 dump（确定性/翻转率分析） |
| `X-M5-*` 响应头 + `/v1` 统计 | 端到端 RTF/首包/帧数/分项 |
| AudioTranscribe | ASR 门禁 |
| `deploy.sh` | 一键部署 + env 固化 + 模型备份 |

---

## 5. 方法论教训（踩坑清单）

1. **热漂移**：连续测试下同代码 RTF 0.75→0.87→0.91。A/B 必须同轮交替跑，
   绝对值跨轮不可比。
2. **macOS BSD grep 不支持 BRE `\|` 交替**：三次假阴性误判（GGML_HEXAGON 旋钮、
   f16 宏、源码状态）——一律 `grep -E`。
3. **objdump/构建日志会误报**：f16"已生效"实为死代码；以反汇编指令计数为准。
4. **`pkill -f` 会自杀**（匹配到自身 shell cmdline）→ 用 `killall` 按进程名。
5. **adb shell heredoc 循环**会被 adb 偷 stdin → 循环内 `</dev/null`。
6. **`/tmp` 会被系统清理**（模型/ORT/脚本丢过一次）→ 重要产物放 `~/m5-assets/`。
7. **clang 19 arm_neon.h 无 `vcvtq_f32_s16`** → `vmovl_s16 + vcvtq_f32_s32` 中转。
8. **ORT 自定义算子**：注册回调置空会段错误；`KernelContext_GetInput` 是 3 参数。
9. **图级多项式展开会打爆流量**：Snake 链 5 遍读写展开成 13+ 算子更慢，
   必须融合成单算子才有效（反直觉但已实测）。
10. **变调克隆幅度 >1.2× 伤可懂度**，1.15-1.22× 是甜点区。

---

## 6. 当前状态快照

- **部署**：`/data/local/tmp/llama-m5-htp-v73`，m5-server :8080（forward 18080），
  HTP env 固化，int8 heads 默认开。
- **服务回归**（2026-11-14 21:47）：wav RTF 0.798 / 首包 0.58 s / 75 帧 / 7 项统计头齐；
  pcm 流式正常（首包 0.33 s）。
- **Web 控制台**：http://127.0.0.1:18080 （临时音色 localStorage、录音克隆、统计卡片）。
- **Android App**：NativeEngine + CPU 后端（HTP 被 OS 对 app uid 封禁），JNI RTF ~0.79-0.90。

## 7. 下一步（按性价比）

1. **学习型草稿模型** → 解锁 depth 批量 2.1×（唯一剩余大杠杆，需训练）。
2. **codec 权重流量**（depth 提速后 codec 会浮出）：保快速路径的权重 int8 重试
   或分块权重驻留。
3. 文档化之外的持续项：`M5_HEADS_INT4`（4-bit 权重带宽再减半）可作 int8 的
   后续 A/B，代价是量化误差再升一档，门禁同 §4.2。

## 8. 二次 review（2026-11-14 23:35，int8 上线后）

### 8.1 新鲜账目（三条长度句子，CLI 分项）

| 句子 | 帧 | 音频 | 墙钟 | backbone | depth | codec | RTF |
|------|----:|-----:|-----:|---------:|------:|------:|----:|
| 7 字 | 23 | 1.74 s | 1.60 | 0.293 | 1.236 | 1.182 | 0.918 |
| 19 字 | 45 | 3.50 s | 2.93 | 0.428 | 2.389 | 1.956 | 0.837 |
| 24 字 | 75 | 5.90 s | 4.38 | 0.633 | 3.651 | 3.015 | 0.742 |

线性拟合：**prefill ≈152 ms（固定）+ backbone 6.1 ms/帧 + depth 48.6 ms/帧 +
codec 33 ms/帧（被 overlap 隐藏）**。24 字句里 depth 占墙钟 **83%**。

### 8.2 本轮复验关闭的门

- **codec prime 已有跨请求缓存**（`Codec::restore_prime` 按 voice codes 命中即跳过）——
  同音色第二次请求起不付 8 帧 prime；CLI 单次进程的 0.42 s 截距只影响首次。
- **--depth-chain 在 int8 后仍略慢**（2 轮交替：RTF 0.7605/0.7575 vs legacy
  0.7493/0.7417）——pipe 往返不是瓶颈，维持 legacy。
- **GGML_HEXAGON_NHMX=1 全链路无差**（RTF 0.7510/0.7496，backbone 0.63 不变）。
- **depth 低位量化不可行**（既有证据：Q8_0 21 MB 对 Q4_0 10.7 MB 只慢 9%，
  若带宽受限应翻倍 → 每次调用是提交开销+计算主导，非权重带宽）。

### 8.3 剩余方向（按 ROI 排序）

| 优先级 | 方向 | 预期收益 | 成本/风险 | 说明 |
|--------|------|----------|-----------|------|
| A | **学习型草稿**（蒸馏 code[b]←(hidden, code[b-1])，批量 verify） | depth 2.1× → RTF ~0.55 | 高（需生成训练数据+训练+verify 回退逻辑） | 唯一大杠杆；六条精确路径已排除，只剩这一条 |
| B | **backbone Q4_0** | 每帧 -2~3 ms，总 -3~5% | 低-中（需 host 构建 llama-quantize，ASR 门禁） | 33 MB→18 MB，per-token decode 或受益 · **已实测，听感不过 → 保持 Q8（§9）** |
| C | **heads int4** | 每帧 -2~3 ms，总 -2~3% | 低（project8 加 q4 解包，门禁同 int8） | 17 MB→8.5 MB 流量；量化误差再升一档 |
| D | **prefill 前缀 KV 缓存**（caption+reference 固定段） | 省 60~100 ms/请求（短句 4-6%） | 中（stage 协议加 state save/restore） | **已上线（§9），实测省 ~45-50 ms/请求** |
| E | **server 启动预热**（启动后跑一句短合成） | 首请求 -150~200 ms 首包 | 极低（~10 行） | **已上线（§9）** |
| F | **pcm 首块 1 帧**（first_frames 2→1） | 首包 -40 ms | 极低 | **已实测：1 帧反而更慢（0.51 vs 0.285 s）→ 维持 2 帧（§9）** |
| G | 多段文本 KV 复用 | 长文每段省 prefill | 中（D 的延伸） | 段落之间共享同一音色前缀 |
| H | /metrics 汇总 X-M5 统计 | 可观测性 | 低 | **已上线（§10）：JSON + Prometheus，pcm 路径唯一可见处** |

**不建议做**：
- codec 再压缩——被 overlap 隐藏（codec 3.0 s < depth 3.65 s），除非 depth 提速；
- codec 卸载到 CPU——比 HTP 慢一倍以上，会击穿 overlap；
- 跨帧 KV 保留——改变每帧 reset 的语义，模型行为风险大，且深度图小 KV 非瓶颈；
- depth 换更低量化——见 8.2 证据（非带宽瓶颈）；
- 并发/render_mutex 改造——单用户场景无收益。

**另一个观察**：depth 在 codec 并发时慢 ~20%（port notes：35.9→42.9 ms/帧），
这是 DSP/内存竞争；overlap 的净收益仍远大于损失（codec 全暴露要 +2.5 s）。

## 9. E/F/B/D 落地（2026-11-14 深夜）

### E：server 启动预热（已上线）

- 设计：`engine.open()` 后立即用 "你好，世界。" 跑一次丢弃模式合成
  （`render_text(text, "", nullptr, ...)`），把 backbone prefill、15 步 depth、
  codec 图、ORT kernel 全部热身；同时把共享前缀（见 D）预先载入 stage 的 KV。
- 收益：消除部署后首请求惩罚（此前 First-Pcm 0.58 s vs 稳态 0.33-0.45 s）；
  启动多花 ~1.5 s，日志打印 `warmup 1.50s (rtf=0.949 frames=21)`。

### F：pcm 首块调优（结论：维持 2 帧）

- 假设：首块 1 帧（80 ms 音频）能再省一个 depth step。
- 实测：1 帧首包 **0.51 s**，2 帧 **0.285 s**——codec 单帧调用的固定开销更大。
  维持 `set_chunking(2, 4)`；E+F 后短句首包 0.285 s、7 字句总耗时 1.44 s。

### B：backbone Q4_0（试听后决定：**保持 Q8**）

- 工具链：host 构建 `llama-quantize`（`~/m5-assets/host-build`），
  `--allow-requantize backbone-Q8_0.gguf backbone-Q4_0.gguf Q4_0` → 32.4→17.1 MB。
- A/B（同轮交替 ×2，24 字句）：backbone_s **0.632/0.619 → 0.495/0.498**（-21%），
  depth 3.65→3.10 s（-15%，DDR 竞争减少），**音频 5.90→4.94 s（-16%，语速变快）**。
- 质量：两版 ASR 全对；时长变化来自韵律漂移，需人工试听定夺（样本见
  `~/m5-assets/quant/samples/backbone-Q8.wav` / `backbone-Q4.wav`）。
- 状态：Q4 文件已上设备（`backbone-Q4_0.gguf`），未改默认命令。
- **决定（人工试听）**：保持 **Q8_0**。Q4 虽然 backbone 快 21%、总耗时 -0.7 s，
  但语速/韵律漂移（音频短 16%）在听感上不可接受——TTS 的音质与自然度优先于
  这部分 RTF 收益，且 backbone 仅占墙钟 14%（大头 depth 不受影响）。
  `m5_engine.h` 默认命令维持 `backbone-Q8_0.gguf`；Q4 仅在显式 `--backbone` 时可用。
  教训：**量化收益必须用耳朵验收**，ASR 全对 + 指标同族不等于听感可接受。

### D：prefill 前缀 KV 缓存（已上线）

- 结构：每段 prefix 行 = [条件标记+参考文本 token][本段文本 token][参考音频 codes]。
  前两段共享（同音色同 caption），但 BPE 会在参考尾字与文本首字间合并，
  因此**共享边界固定为「共享 token 数 - 2」**（只依赖音色，不依赖文本，hash 稳定）。
- 协议：新增 `count == -1` 预填充消息 `[shared_rows][hash][rest_rows][rows...]`；
  子进程在 hash 命中时 `llama_memory_seq_rm(seq, shared, -1)` 只保留共享段 KV，
  否则 clear 后重放；引擎对 ring 传输自动回退到全量预填充。
- 收益（同音色不同文本，server）：Backbone-S **0.603 冷 → 0.555-0.572 命中**，
  短句 0.237-0.329；即每请求省 ~45-50 ms 预填充。预热后首个请求即命中。
- 质量：多段长文本 ASR 全对；token 序列与全量预填充一致（仅批形状不同）。
- 坑：**缓存命中时子进程必须读掉共享段字节**，否则管道错位、下一条消息头读出
  垃圾 count（表现为 "embedding message exceeds remaining context" 后服务崩溃）。

### 环境坑：8080 上出现过两个监听进程

- 早前会话遗留的 `m5-server-cockpit` 与新 `m5-server` 同时监听 0.0.0.0:8080，
  连接被随机分发 → 表现为「wav 一半 500、pcm 正常」的假回归。
  `deploy.sh` 的 pidof 停止逻辑只杀 `m5-server`，不动其他名字。
  排查此类问题时先 `netstat -tlnp | grep 8080` 确认监听者唯一。

## 10. H：GET /metrics 汇总统计（2026-11-15 00:2x）

动机：`X-M5-*` 头只在 wav 全量响应里存在——chunked pcm 在首个分块前就定稿了响应头。
于是流式路径的性能数据此前只落在设备日志里，回归对比必须上设备 `tail`。

- 端点：`GET /metrics`（JSON，控制台与脚本用）；`?format=prometheus`
  返回 `text/plain; version=0.0.4`，可直接刮。
- 内容：累计 `requests / failed / render_s / audio_s`、`avg_rtf`、
  最近 64 次窗口的 `p50_rtf` 与 `p50_first_pcm_s`、`warmup`、`last`
  （取**最近一次成功**，避免被拒绝请求把 gauge 打成 0）、以及 `recent[]`
  逐条明细（voice / format / text_bytes / ok / rtf / elapsed / audio /
  first_pcm / frames / segments / backbone / depth / codec）。
- 埋点位置：wav 路径（成功与失败）、pcm worker（成功与失败）、启动预热
  （单独存档，不进窗口，避免污染分位数）。失败一律计入 `failed` 并保留 ok=false 记录。
- 控制台：合成完成后再拉一次 `/metrics`，在统计卡片下追加一行
  「服务端累计 N 次（失败 M）· 均值 RTF · 首包中位 · 启动预热」——
  pcm 模式下也有一致的服务端视角（卡片本身仍走客户端实测兜底）。
- 验证：三笔请求（wav 长句 / pcm 短句 / 404 坏音色）后
  `requests=3 failed=1 avg_rtf=0.800 p50_first=0.428 m5_last_rtf=0.818`，
  `last_*` 正确跳过末尾那次 404。
- 回归用法：`curl -s .../metrics?format=prometheus | grep m5_p50`，
  不再需要 `adb shell tail` 抓日志。
