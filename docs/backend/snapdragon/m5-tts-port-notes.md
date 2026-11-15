# M5-TTS 在 SM6650 上的移植笔记

> 结构化的优化尝试总账（含全部负结果与证据链）见 `m5-tts-optimization-log.md`；
> 本文是按时间线的实验日志。

## 目标与验收

在 MTN-AN80（SM6650，Android 15，arm64-v8a）上，用 GGML Hexagon 后端执行
M5-TTS temporal backbone，保留官方 ONNX Runtime 的帧内解码和 codec。
用同机 CPU 基线验证 hidden state、KV cache、音频输出、热启动首段延迟和 RTF。
记录实际算子分配和各阶段耗时，依据数据选择后续 Hexagon-MLIR 实验。

## 2026-11-13 环境和初步结果

- llama.cpp 基线为 `31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`。
- Android NDK r28b 构建的 GGML CPU 测试在真机通过 58/58 个 FP32 ADD 用例。
- 这个结果覆盖交叉编译、ADB 部署和 Android 原生执行。HTP 需要单独验证。
- Docker Desktop 已运行。macOS 上将其 Resources/bin 加入 PATH 后，Docker
  可以找到自带的凭据助手。工具链镜像拉取中有 layer 重试。
- DSP 架构和 FastRPC 会话权限待实测。厂商 RPC 库存在只能说明用户态库可见。

## 验证顺序

1. 使用 Snapdragon Android v0.8 镜像构建 `test-backend-ops` 和 HTP 库。
   部署到设备临时目录，选运行时报告的 HTP 设备跑 `MUL_MAT`。
   记录通过用例数、架构、会话日志和算子耗时。空过滤器结果 0/0 不能证明执行。
2. 在官方 Android runner 上测量原版 M5-TTS。分别记录 backbone、depth frame
   和 cached codec 的耗时。固定输入文本、参考音频、随机噪声和 chunk 策略。
3. 使用 `backbone.onnx` 的 `hidden`、`positions`、`past` 与
   `last_hidden`、`present` 建立对照。先做单帧 decode，再验证 prefill
   和连续帧 cache 更新。先验证 CPU GGML，再切 HTP，最后接回流式 runner。
4. 权重和 KV cache 在会话内复用。记录每帧在 CPU 与 HTP 之间的交接次数，
   同时测量子图加速和端到端收益。只有 profile 定位到具体瓶颈后才选 kernel。

## 可以展开成博客的实验

- 通过算子测试和 profiling 证明图确实在 HTP 执行。
- 用 ONNX 子图做参考实现，定位权重映射、RoPE、mask 或 KV layout 的误差。
- 比较单算子耗时、首段音频延迟和 RTF，说明设备提交与同步如何影响整体收益。

参考：[M5-TTS 导出器](https://github.com/ASLP-lab/M5-TTS/blob/main/onnx/export.py)、
[Snapdragon 后端开发文档](developer.md)。

## 真机确认 CDSP 架构

用 NDK 编译一个 ARM64 主机端 probe，在 ADB shell 下用
`LD_LIBRARY_PATH=/vendor/lib64` 加载 `libcdsprpc.so`。
调用公开的 `remote_handle_control`，使用 `DSPRPC_GET_DSP_INFO`（2）查询
`CDSP_DOMAIN_ID`（3）的 `ARCH_VER`（6），返回 `result=0x0 capability=0x4d73`。
按 llama.cpp 的 `htpdrv_get_arch` 解码，低字节 `0x73` 对应 **Hexagon v73**。
首个测试只需构建 v73 skel。该查询不验证 DSP 库加载或 kernel 执行。

Docker VM 日志记录镜像下载重试原因为 `unexpected EOF`。镜像可用后，
优先构建 `test-backend-ops` 和 `htp-v73`，并部署对应的 ARM64 和 DSP 库。

参考：[FastRPC 公开头文件](https://github.com/qualcomm/fastrpc/blob/development/inc/remote.h)。

## v73 真机矩阵乘通过

工具链镜像 `ghcr.io/snapdragon-toolchain/arm64-android:v0.8` 已拉取，digest
`sha256:e39e5e9c0957550a92c0de756b7556d81196c7c7f7328ed80de252bfb4878adf`。
用 Android Snapdragon release preset，设置 `GGML_OPENCL=OFF` 和
`LLAMA_BUILD_TESTS=ON`，分别构建 `test-backend-ops` 和 `htp-v73`。
后者产生 `libggml-htp-v73.so`。将它和测试程序、Android 侧 GGML 库推到
`/data/local/tmp/llama-m5-htp-v73` 后执行：

```sh
cd /data/local/tmp/llama-m5-htp-v73
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH="$PWD" GGML_HEXAGON_PROFILE=1 \
  GGML_HEXAGON_VERBOSE=1 ./test-backend-ops test -b HTP0 -o '^MUL_MAT$' -p 'm=16,n=1,k=256'
```

- HTP0 打开 CDSP 会话并加载 v73 skel。动态域查询返回 0x6c，静态 CDSP 域仍能工作。
- 测试汇总为 106/106。profile 记录了实际的 HTP0 MUL_MAT，示例为
  FP32 × FP32、256×16 权重乘单列输入，走 HVX tiled，单次记录 34 µs。
  该数字来自正确性测试，只能作为执行证据。
- 设备名是 `HTP0`。第一次用 `HTP0:0`，测试程序跳过了全部设备但仍退出成功。
  验收要同时检查会话、匹配的测试与 profile。

## M5-TTS 源码对照

本地检出官方 M5-TTS `4e7cc91545ad5ecd368d7726129cefb41fc6e1d7`。
`onnx/graphs.py` 里的 Backbone 使用半维旋转 RoPE、GQA 和按绝对位置比较的
因果 mask。输入 cache 布局为 [layer, K/V, batch, kv_head, past_length, head_dim]。
对照时应分别验证 RoPE 后写入的 K 和原样写入的 V，并保留最后位置的归一化 hidden。
官方 Android APK 构建需要 JDK 17 及与项目校验值匹配的 codec_encoder.onnx。

## Android CPU 基线的计时边界

本地 M5-TTS 的 Android `M5Engine` 会先生成整段的音频 codes，再用
cached codec 分块写 WAV。cached codec 的内部状态可以跨 chunk 复用，
但原生示例的调度目前是逐段合成。Python ONNX runner 另有流式调度。
在 Android engine 的诊断 JSON 中加入了 backbone、depth 和 codec 的
ORT Run 累计时间，以及首次写入非空 PCM 的时间。阶段累计不含 Java
张量准备与结果读取，首次 PCM 则从合成入口开始计时。这组字段可以作为
原生 CPU 对照，逐帧流式端到端对照还需要将生成和 codec 消费交错调度。

## 复用 llama.cpp 的 Llama 图

官方 temporal backbone 是 8 层、宽度 512、8 个 query heads、2 个 KV heads 的
Llama 变体。第一版通过 `examples/m5-tts/convert_backbone.py` 导出独立的
Llama GGUF。调用方负责提供 M5 原始混合 embedding，并消费最终归一化 hidden。
这样可以复用 llama.cpp 的 KV cache、backend scheduler 和 HTP offload。
转换会将 Q/K 权重从 Hugging Face 的半维旋转重排到 GGML 的相邻配对顺序，
并把 Llama 3 波长缩放写到 `rope_freqs.weight`。GGUF 里的占位 token embedding
供通用加载器使用，文本和音频 embedding 继续由 M5 runner 管理。

用与官方结构一致的合成 BF16 checkpoint 验证了 75 个 FP32 GGUF tensor 的
序列化和读取。真实模型数值对齐需要同一 checkpoint 生成的 ONNX 对照，
尤其要检查不同位置的 RoPE 和连续帧 cache。

`examples/m5-tts/backbone.cpp` 接收 FP32 embedding 文件，批量 prefill 后用同一
`llama_context` 逐帧 decode，并输出每次调用最后一个位置的 hidden。原型将 K/V
cache 保持为 FP32，关闭 embedding pooling，便于后续与 ONNX 对照。
Mac CPU 上用 4 行 prefill 和 3 行连续 decode 验证了 4 个输出与 NumPy RMSNorm
参考的最大绝对误差为 2.38e-7。这个 fixture 的 attention 和 FFN 权重为零，
因此可用于验证 embedding 路由、位置和输出边界，无法验证注意力或 cache 的数值。

同一 fixture 已在 SM6650 的 HTP0 执行。使用 Snapdragon 工具链构建
`llama` 目标，使用 NDK 链接 `backbone.cpp`，部署时同时带上
`libllama.so`、GGML 库、v73 skel 和 NDK 的 `libc++_shared.so`。
原型调用为 `./backbone-android m5.gguf HTP0 input.f32 4 output.f32`，
沿用前面的 HTP 环境变量。4 个 hidden 输出相对 NumPy 的最大绝对误差
为 1.19e-6。profile 确认 `blk.0.attn_q.weight` 的 MUL_MAT 在 HVX 执行。
这是合成权重下整图路径的 smoke test，仍需用官方权重和非零 attention
验证 RoPE、GQA、跨帧 cache 和输出波形。

## 官方非零权重对齐和 HMX 路径排查

通过镜像取得官方 ONNX bundle 的 backbone。VoiceClone safetensors 的 SHA256
与 ONNX manifest 所列不同，首层 Q 投影也存在非零权重差。转换器增加了直接
读取该 ONNX 图 initializer 的路径，让 GGUF 和参考运行共享同一份 FP32 权重、
逆频率和模型结构。测试固定随机种子 12，输入为 32 行 prefill 加 7 次逐帧 decode。
参考脚本校验 ONNX 每步输出形状和历史 cache 前缀，并逐行比较最终 hidden。

| 路径 | 最大绝对误差 |
| --- | ---: |
| Mac GGML CPU，关闭 Flash Attention | 2.10e-5 |
| 手机 GGML CPU，关闭 Flash Attention | 2.57e-5 |
| 手机 HTP，默认 HMX 路径 | 19.78 |
| 手机 HTP，`GGML_HEXAGON_NHMX=0` | 0.0111 |

Mac CPU 的默认 Flash Attention 路径在该输入上误差约 0.00358，显式关闭后
达到上述 CPU 对照精度。HTP 单独禁用 RoPE、SwiGLU、softmax offload 或关闭
后端融合未消除大偏差，关闭 HMX 后误差明显下降，且 profile 确认仍有 HVX
矩阵乘。当前只证明 HMX 开关与偏差相关，尚未定位具体 kernel 或数据布局问题。
HVX 结果通过本实验的 atol=0.012、rtol=0.002 检查，仍需在真实 embedding、
帧内采样和音频上验证可接受性。

## 接入逐帧生成循环

`backbone.cpp --pipe` 在一个进程内保留 llama context，接收 uint32 小端行数
和对应的 FP32 embedding，再返回一行 512 维 hidden。上板时用 `adb shell -T`
避免 PTY 改写二进制；诊断走 stderr。在 v73 的 HVX 路径上，用官方 ONNX
转换的权重做了 32 行 prefill 和 7 次 decode，pipe 与文件模式的 8 行输出逐位相同。
`run_stream.py` 将这个接口适配到官方 M5Onnx 的 backbone session 位置，
复用官方采样、depth 和 codec 逻辑。经 ADB 的方案用于快速验证接口，
同机性能实验需要把 CPU 端 ORT 和 backbone 都放在手机上。

## SM6650 同机 backbone 初测

使用官方权重 GGUF、32 行 prefill 和 7 帧 decode，CPU / HTP0 交替各执行
3 次，HTP 禁用 HMX，双方关闭 Flash Attention。计时从调用 llama_decode
开始到读出 hidden 为止，包含同步和输出读取，排除模型加载、ADB 和写盘。
每次创建新进程，线程数采用当前 runner 的 llama.cpp 默认值。

| 后端 | Prefill 中位数 | 7 次 decode 中位数 | Decode 平均每帧 |
| --- | ---: | ---: | ---: |
| CPU | 70.192 ms | 281.571 ms | 40.22 ms |
| HTP0 / HVX | 73.911 ms | 87.691 ms | 12.53 ms |

CPU 的 7 帧总时间在 157.870–285.101 ms 间波动，需进一步控制线程、
热状态和采样长度。这组数据只覆盖 backbone。端到端需让两个版本在同一
设备上使用相同的帧内解码、codec 和 chunk 策略。流式入口已记录每个
session 的累计调用耗时和次数，用来区分各阶段与端到端耗时。

## 同机端到端流式对照

M5 Android runner 现在每生成 4 帧就调用官方 CPU cached codec，在 EOS 后
处理剩余帧，并统一保留最后 2400 个样本做尾部裁剪。CPU 与 HTP 路径共享
tokenizer、参考音色、采样噪声、depth 和 codec。混合版通过手机本地 pipe
交换 embedding 与 hidden，KV cache 留在 llama context。ADB 只启动命令和取回 WAV。

设备使用 4 个 ORT 线程、种子 1234、输入「你好，今天天气很好。」和官方参考音频。
CPU 和 HTP 先后运行，各 4 次，以下是剔除首轮后的 3 次中位数。

| 路径 | 音频 | 合成耗时 | 首次 PCM | RTF | Backbone | Depth | Codec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方 ONNX CPU | 2.22 s | 6.486 s | 1.380 s | 2.922 | 0.567 s | 3.374 s | 2.490 s |
| GGML HTP/HVX + ONNX CPU | 2.46 s | 8.254 s | 2.108 s | 3.355 | 0.884 s | 4.161 s | 3.119 s |

两条路径都完成合成并产生非静音的 24 kHz 单声道 PCM16 WAV。串行实验
第 3 次的两份 WAV 前 19200 个样本完全一致，之后分岔，生成长度也不同。
同一随机种子并不能消除浮点差异对自回归采样和 EOS 的影响。这一轮没有
做听感或 ASR 准确率评测。

混合版每个文本段启动一个 backbone 进程，首个请求承担模型加载与图预留，
而 CPU ONNX session 在计时前已加载。表中体现原型的实际用户延迟，不能把
差异全归因于硬件。CPU 基线中 depth + codec 的中位耗时合计约占 90%，
解释了为什么单测中 backbone decode 更快也没有带来整条链路加速。
复现步骤见 [Android benchmark](../../../examples/m5-tts/README.md)。

## 实时优化实验

实时验收以同机热运行 RTF ≤ 1 为准，同时记录首段 PCM 延迟与生成差异。
CPU baseline 为 4 个 ORT 线程。相同输入和种子下，线程扫描的热运行 RTF
中位数为 2 线程 3.636、6 线程 3.537、8 线程 3.506。历史 4 线程对照为
2.922。继续以 4 线程做分块和图优化实验。

帧内图包含 692 个 MatMul，其中 542 个右侧参数可以追溯到 initializer，
其余涉及激活之间的计算等。直接量化只识别出 52 个静态右操作数。实验先
解析权重的 Identity 别名，再做动态 INT8 量化，转换 542 个 MatMul。
新图约 249 MB，原图约 280 MB。实验还需要比较实际帧耗时与音频差异。

4 线程、首块固定为 4 帧、同一文本和种子，剔除首轮后：

| 模型与后续块大小 | RTF 中位数 | 首次 PCM | Depth | Codec | 生成音频 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32，4 帧 | 2.982 | 1.407 s | 3.390 s | 2.575 s | 2.22 s |
| FP32，8 帧 | 2.892 | 1.376 s | 3.375 s | 2.435 s | 2.22 s |
| FP32，16 帧 | 2.885 | 1.379 s | 3.354 s | 2.361 s | 2.22 s |
| 动态 INT8 depth，8 帧 | 2.699 | 1.301 s | 1.583 s | 2.786 s | 1.82 s |

INT8 提前结束，按生成帧数归一化后，depth 约从 116 ms/帧降到 66 ms/帧
（不含最终 EOS 调用的单独拆分）。生成时长变化说明不能以耗时下降单独验收，
需要对比语音内容和音质。下一步用 ORT profiling 确认 codec kernel 分布，
并以相同设置评估 XNNPACK。实验开关为 `M5_FIRST_CHUNK`、`M5_CHUNK`、
`M5_DEPTH_MODEL`、`M5_PROFILE_DIR`、`M5_XNNPACK_STAGE` 和 `M5_RUNS`。

### Codec 卷积的 HTP 可行性

二维卷积改写在 Mac ORT CPU 上用参考语音的码本跑了 4、8、8 帧，
PCM 和 22 项跨块状态逐位一致。SM6650 上固定首块 4 帧、后续 8 帧，
默认 CPU 热运行 RTF 中位数为 2.822，XNNPACK 为 4.847。

`examples/m5-tts/codec_conv_probe.cpp` 取 codec 热点 `/conv_14/Conv` 的
384 输入通道、384 输出通道和 7 taps，将 64 个预展开位置表示成
`[2688,384] × [2688,64]` 的 GGML MUL_MAT。固定合成数据，预先上传，
一次预热、八次计算加读回取中位数。4 线程 GGML CPU 为 6.770 ms，
HTP0 禁用 HMX 为 3.751 ms，最大绝对误差 5.08e-5。设备 profiling
确认走 HVX tiled。这个探针只测矩阵乘，后续需要把真实权重、输入展开、
卷积偏置、跨块缓存和相邻算子的交接成本一起计入。

用 `capture_codec_conv.py` 从官方 cached codec 捕获实际权重、偏置和
三块连续参考码本运行产生的激活。首块 4 帧，其后两块各 8 帧。
`codec-conv-probe conv-real OFFSET` 每 64 个位置展开输入，上传、执行、
读回并加偏置。权重只上传一次，报告预热后八次中位数。

| 音频帧偏移 | 卷积位置数 | GGML CPU / 4 线程 | HTP0 / HVX | HTP 对 ORT 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 640 | 143.225 ms | 68.084 ms | 4.14e-4 |
| 4 | 1280 | 285.796 ms | 127.508 ms | 3.74e-4 |
| 12 | 1280 | 294.922 ms | 136.121 ms | 4.82e-4 |

约 2.1–2.2 倍是同一展开算法的 CPU/HTP 比较。ORT profiling 中
原生 Conv 在整句所有调用累计约 160 ms，说明逐 tile 上传的移植
还需与原生 Conv 同形状公平对照，并优先减少提交与展开开销。

### 同形状 ORT 对照和 HMX 复现

以音频帧偏移 4、1280 个输出位置为固定输入，扩大 tile 将 HTP/HVX 的
热运行中位数从 64 位置时 133.116 ms 降到 1280 位置时 79.708 ms。
同形状 ORT 原生 Conv 由 `build_conv_baseline.py` 提取，用
`AndroidConvBenchmark` 测得 39.916 ms，含输出复制，和完整图捕获的
输出逐位相同。该路径有输入展开开销的 HVX 原型仍慢于 ORT。

开启 HMX 后同一个卷积的矩阵乘设备侧约 2.75 ms，主机全路径 12.516 ms，
但最大绝对误差为 1.349。用 `M5_CONV_F16=1` 将权重转为 FP16 后仍然
复现同样的 HMX 最大误差，而 CPU FP16 最大误差为 2.31e-4。这将此前
整图偏差缩小为可单独复现的一个算子。HMX 结果不能用于音频推理，
下一步要对照 HMX 分块和数据布局定位问题。

### HMX 输出幅值和直接卷积

对偏移 4 的块测试 32、64、128、256、640 和 1280 个位置的 tile。
HMX 的最大绝对误差均为 1.3485672。设置 `GGML_HEXAGON_RT_REPACK=0`
切换到 profile 标记的 hmx-flat 后仍复现。新增的幅值探针确认整块
矩阵乘在加偏置前全为零，CPU 对照最大绝对值约为 1.349。
`M5_CONV_DIRECT=1` 改用现有 `GGML_OP_CONV_2D`，跳过主机 im2col，
仍在 HMX 上得到全零，含传输中位数 7.140 ms。关闭 HMX 时，该形状的
直接卷积未通过 HTP 支持检查。现有证据将排查范围缩到共享的 HMX
执行路径或设备资源条件，尚未确认根因。

另测 CPU codec 和帧生成重叠（`M5_OVERLAP_CODEC=1`，4 线程、
首块 4 帧、后续 8 帧）。本轮热运行 RTF 中位数 3.327 → 2.801，
首段 PCM 中位数约 1.395 → 1.729 s。两个模式的历史基线存在波动，
仍需交替重复以分离调度收益和热状态影响。

### HMX v73 锁与指令路径诊断

在独立 worktree 中构建两份诊断版 v73 skel，通过
`GGML_HEXAGON_LIBHTP_DIR` 从单独目录加载。第一份在队列线程调用
`HAP_compute_res_hmx_lock` 后遇到非零返回就终止。真机没有终止，
但 HMX 乘积仍全零。第二份保留该检查，把 FP16 dot chunk 的
`:deep` 批量加载换成每 32×32 tile 一次普通加载。该版本同样
返回全零，CPU 的未加偏置输出峰值仍为 1.348567。两次均用
`./codec-conv-probe conv-real 4 1280`。证据不支持把问题单独归因于
锁 API 的返回错误或 deep 加载模式，尚需区分输入 tile 内容与 accumulator
读回。诊断改动保存在 `examples/m5-tts/hmx-v73-diagnostics.patch`，
用于复现故障。

### CDSP 报告的 HMX 能力

`cdsp_capabilities.c` 在 ADB shell 中通过 FastRPC 读取 CDSP 能力。三个
查询都返回成功。ARCH_VER 为 0x4d73，HMX_SUPPORT_DEPTH 和
HMX_SUPPORT_SPATIAL 均为 0。llama.cpp 的 DSP 端 `htp_iface_hwinfo`
直接填入 `*n_hmx = 1`，因此 session 打印的 hmx 1 不是独立硬件证据。
另一个诊断 skel 在 worker 里检查首个 activation、weight 和输出 tile，
当任一输入全零或输出非零时终止。真实卷积探针正常完成且乘积仍全零，
说明在这个诊断路径里首个输入 tile 有非零值，但首个写回 tile 全零。
这些结果支持在此设备继续使用 HVX，并避免把 HMX 的全零输出计为加速；
它们不区分芯片配置与当前固件暴露的能力。诊断补丁不用于日常推理。

## NPU 加速上限探测

目标：测出 M5-TTS 热点算子在 SM6650 cDSP 上可达到的最优加速比，并判断
RTF < 1 能否经 NPU 路线达到。所有数字均为真机热运行中位数，带 CPU 参考、
精度校验和 `GGML_HEXAGON_PROFILE` 的执行路径确认。新增探针为
`examples/m5-tts/npu_roofline_probe.cpp`（权重类型与形状扫描）和
`npu_gemm_batch_probe.cpp`（多算子单图）。

### 瓶颈定位

拉取设备上的 ORT kernel profile 并按 `op_name` 聚合，剔除首轮与建模开销后
的相对分布：

| 阶段 | 主导算子 | 占比 |
| --- | --- | ---: |
| depth | MatMul | 75.1% |
| depth | 其余（Concat/Reshape/Mul/Add 等） | 24.9% |
| codec | Conv | 58.1% |
| codec | ConvTranspose | 15.6% |
| codec | Sin | 8.0% |
| codec | MatMul | 5.4% |

`/conv_14/Conv` 累计只有 159.87 ms，不是 codec 的主要矛盾；codec 有
约 20 个 100–160 ms 的 Conv/ConvTranspose。设备 `/vendor/lib64` 只有
`libcdsprpc.so`/`libadsprpc.so`，没有 `libQnnHtp*.so`，ORT QNN EP 路线
在这台机器上不可用。

### HMX 能力确证与默认路径修复

把能力查询扩展到 `remote.h` 的全部 12 项，交叉验证接口可信：
`DOMAIN_SUPPORT=1`、`HVX_SUPPORT_128B=2`、`VTCM_PAGE=2 MB`、
`ARCH_VER=0x4d73`，与 hwinfo 一致；`HMX_SUPPORT_DEPTH=0`、
`HMX_SUPPORT_SPATIAL=0`。这颗芯片的 CDSP 没有 FP16 HMX，整图偏差和
卷积全零是硬件能力缺失，不是 tile 布局或 `:deep` 加载模式问题。

`htp/main.c` 的 `htp_iface_hwinfo` 无条件写 `*n_hmx = 1`，
`ggml-hexagon.cpp` 的 hwinfo 失败兜底也强制 1，且 `htp_iface_start`
收到的是全局 `opt_nhmx` 而非探测结果。新增 `htpdrv_get_hmx_support()`
（`htp-drv.cpp`/`.h`），在 `ggml_hexagon_init()` 查询 CDSP 能力，
depth 与 spatial 均为 0 时把 `opt_nhmx` 置 0。用工具链镜像重建并部署
`libggml-hexagon.so` 后（原库备份为 `libggml-hexagon.so.orig`）：

| 检查（默认环境，无 `GGML_HEXAGON_NHMX`） | 修复前 | 修复后 |
| --- | ---: | ---: |
| hwinfo 报告的 hmx | 1 | 0，并打印 `CDSP reports no HMX support` |
| `conv-real 4 1280` HTP 最大绝对误差 | 1.348567（全零） | 3.74e-4 |
| 32 行 prefill + 7 帧 decode hidden 最大绝对误差 | 19.78 | 0.0111 |
| `test-backend-ops` MUL_MAT | 106/106 | 106/106 |

修复后默认走 HVX，无需再手工设置 `GGML_HEXAGON_NHMX=0`。

### 算子级最优配置

`conv-real 4 1280`（`[2688,384] x [2688,1280]`，FP32 权重 4032 KiB），
权重常驻设备、只计 compute + 读回的时间：

| 权重 / 激活 | GGML CPU / 4 线程 | HTP0 / HVX | 权重大小 | HTP 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| f32 / f32 | 71.4 ms | 68.3 ms | 4032 KiB | 3.7e-4 |
| f16 / f32 | 94.3 ms | 67.4 ms | 2016 KiB | 3.7e-4 |
| f16 / f16 | 95.3 ms | 67.6 ms | 2016 KiB | 3.7e-4 |
| bf16 / f32 | 166.7 ms | 68.8 ms | 2016 KiB | 2.1e-3 |
| q8_0 / f32 | 30.0 ms | 31.8 ms | 1071 KiB | 1.45e-2 |
| q4_0 / f32 | 35.5 ms | 38.3 ms | 567 KiB | 1.48e-1 |

`q8_0` 是最佳量化点：相对 FP32 有约 2.2 倍收益，但 HTP 仍略慢于 CPU。
`tq*`/`iq3*`/`iq4_xs` 等类型不被 HTP 支持；`f16` 激活搭配量化权重在
HTP 上输出全零（误差约 1.36），不能用于推理。`M5_CONV_DIRECT=1` 的
`GGML_OP_CONV_2D` 在禁用 HMX 时不受支持，开启 HMX 时虽只要 6.633 ms
却全零，因为它只走 HMX kernel。

按输出位置数扫描 `q8_0 / f32`，HTP 的优势只出现在小 tile：

| 输出位置 | CPU | HTP | HTP/CPU |
| ---: | ---: | ---: | ---: |
| 64 | 5.06 ms | 2.09 ms | 2.42x |
| 160 | 8.78 ms | 4.54 ms | 1.93x |
| 320 | 18.37 ms | 8.58 ms | 2.14x |
| 640 | 22.85 ms | 15.52 ms | 1.47x |
| 1280 | 29.90 ms | 31.43 ms | 0.95x |

但把 1280 拆成 20 个 64 行 tile 需要 20 x 2.09 = 41.8 ms，比整块
31.4 ms 更慢，因此这个优势无法在真实卷积上兑现。同形状 ORT 原生 Conv
为 39.9 ms；最好的 NPU 路径 31.4 ms 只有 1.27x，而 CPU q8_0 的
30.0 ms 反而更快。

### 多算子单图（depth 形状）

按 llama-20M depth decoder（hidden 512、FFN 2048、5 层）构造
512/2048 维的 7 种投影，循环 N 次组成单张 GGML 图：

| 算子数 | 最优 CPU | 最优 HTP | HTP/CPU |
| ---: | ---: | ---: | ---: |
| 64 | q8_0 6.97 ms | q8_0 7.82 ms | 0.89x |
| 256 | q8_0 17.49 ms | q8_0 26.18 ms | 0.67x |
| 692 | q8_0 32.29 ms | q8_0 94.55 ms | 0.34x |

`GGML_HEXAGON_OPBATCH=64` 把 692 算子的 HTP 从 96.2 ms 降到 76.0 ms，
`HOSTBUF=0`、`OPQUEUE=4` 和 power mode 均无改善，仍比 CPU 慢 2.4 倍。
HTP 的 op batching 没能抵消每算子约 137 us 的提交与同步成本。

### 端到端

4 个 ORT 线程、种子 1234、4 次运行取后 3 次，用 `conv.dex` 的开关：

| 配置 | RTF 中位数 | 首次 PCM | Depth | Codec |
| --- | ---: | ---: | ---: | ---: |
| FP32 depth，4 帧块，无重叠 | 3.77 | 1.87 s | 4.16 s | 3.60 s |
| INT8 depth，8 帧块 | 2.77 | 1.56 s | 1.58 s | 2.88 s |
| INT8 depth，16 帧块，重叠 | 2.30 | 1.42 s | 1.75 s | 2.86 s |
| INT8 depth，首块 2 帧，16 帧块，重叠 | 2.13 | 1.03 s | 1.75 s | 2.51 s |
| 上述最优配置 + HTP backbone（HVX） | 2.43 | 1.65 s | 1.94 s | 2.87 s |

混合版每文本段新起一个 backbone 进程，backbone 累计从 0.53 s 涨到
0.86 s，首次 PCM 从 1.03 s 涨到 1.65 s。FP32 基线这轮为 3.77 而非
历史记录中的 2.92，说明热状态与 dex 版本带来的波动，跨轮比较只看
同一次扫描内的相对值。INT8 版本生成 1.82 s 音频而 FP32 生成 2.22 s，
RTF 仍不能跨量化配置直接比较。

### 结论

- SM6650 的 CDSP 是 v73、2 个 HVX 128B 单元、2 MB VTCM、无 HMX。
  HMX 路线在这颗芯片上不可用，之前的"精度问题"是硬件能力缺失。
- 单算子最优 NPU 加速是 2.4x（64 行小 tile 的 q8_0 GEMM），但在模型
  真实形状上 NPU 为 0.95x–3x 减速；端到端 NPU 混合路径比纯 CPU 更慢。
  因此 RTF < 1 无法靠把现有算子搬到这块 NPU 达到。
- 当前可达的最好端到端结果是 RTF 2.13 / 首次 PCM 1.03 s，收益来自
  CPU 侧：depth 动态 INT8、块大小、codec 重叠和首块 2 帧。
- 继续压低 RTF 应优先做 codec 的 Conv/ConvTranspose 量化或算子替换
  （当前约 2.5 s，占 60%），以及把 backbone 改成进程内 JNI 调用。
  若换用具备 FP16 HMX 的 SoC，`GGML_OP_CONV_2D` 的 6.633 ms 路径
  理论上对同形状 ORT Conv 有约 6x 空间，但需先验证精度。

## 硬件规格复核与量化优化

### 硬件身份（更正上一节的推测）

`ro.soc.model=SM6650`、`ro.board.platform=volcano`、CPU part 0xd80/0xd81、
大核最高 2.304 GHz。这是 Snapdragon 6 Gen 4（2025-02，中端），不是
8 Gen 4 / 8 Elite（SM8750），也不是 8s Gen 4（SM8735）。高通官方产品
简介列出的 Hexagon NPU 能力为 Fused AI Accelerator、Tensor Accelerator、
HVX、Scalar Accelerator，精度 INT4/INT8/INT16，没有 HMX。

设备侧 FastRPC 全属性扫描（只有 domain=3 返回详细能力，其它域仅有
attr=0 的支持位）：

| 属性 | 值 | 含义 |
| --- | --- | --- |
| HVX_SUPPORT_64B (2) | 0 | 无 64B HVX |
| HVX_SUPPORT_128B (3) | 2 | 2 个 128B HVX 单元 |
| VTCM_PAGE (4) | 0x200000 | 2 MB VTCM |
| VTCM_COUNT (5) | 1 | 1 组 VTCM 页 |
| ARCH_VER (6) | 0x4d73 | Hexagon v73 |
| HMX_SUPPORT_DEPTH (7) | 0 | 无 HMX |
| HMX_SUPPORT_SPATIAL (8) | 0 | 无 HMX |

`libggml-hexagon.so` 的 hwinfo 打印 `threads 4, hvx 2, hmx 0, vtcm 2 MB`。
NHVX 吞吐扫描（640 行 q8_0 GEMM）：1 -> 28.6 ms、2 -> 15.8 ms、
4 -> 16.0 ms，加到 4 个 HVX 线程没有额外收益，确认物理上限是 2 个。
设备上其实有 `/odm/lib64/libQnnHtp.so`、`libQnnHtpV73Stub.so` 和
`/odm/lib64/rfsa/libQnnHtpV73Skel.so`；上一节说"设备侧没有 QNN 库"
是只查了 `/vendor/lib64` 导致的误判。

### ORT EP 可用性

设备 ORT 是 1.23.2 定制版（`OrtGetApiBase@@VERS_1.23.2`）。
`OrtEnvironment.getAvailableProviders()` 返回 `[CPU, NNAPI, XNNPACK]`；
`addQnn()` 抛 `QNN execution provider is not supported in this build`。
用 codec 真实权重构造的 QDQ 卷积模型（1x384x134，384->384，k=7）
测单算子：CPU 0.93 ms、NNAPI 0.96 ms、XNNPACK 1.65 ms，NNAPI 没有
触达 DSP 加速。ORT QNN EP 需要自行编译带 QNN EP 的 ORT 并配 QNN SDK，
当前产物不可用。

### codec 动态 INT8 为什么不可用

`quantize_dynamic` 对 Conv 生成 `ConvInteger(10)`，本机 ORT 1.18 和
设备 ORT 1.23.2 的 CPU EP 都没有这个 kernel：
`Could not find an implementation for ConvInteger(10)`。整图、仅 Conv、
仅最耗时 12 个 Conv 三种变体都在 `createSession` 阶段抛
ORT_NOT_IMPLEMENTED（表现为 app_process 被杀）。depth 的动态量化只
涉及 MatMulInteger，CPU EP 支持，所以能用。

### 静态 QDQ 量化（有效路线）

ORT CPU 会把 QDQ Conv 融合成 INT8 QLinearConv 快速路径：同一真实卷积
FP32 为 5.6-5.8 ms，QDQ per-channel INT8 为 0.93 ms，QOperator 为
0.99 ms；重复两轮并打乱顺序后结果稳定。数值上 QDQ 的平均绝对误差
是信号 std 的 2.4%，QOperator 是 8.8%，故采用 QDQ 格式。

`quantize_codec_static.py` 用 `codec_cached.json` + `voices.json` 构造
6 组校准输入，对 Conv/ConvTranspose 做 per-channel QDQ 量化。ORT 1.18
的 `adjust_tensor_ranges` 会给校准没收集到的 Softmax 输出写范围并抛
RuntimeError，脚本对 `TensorsData.__setitem__` 做了最小放宽；Percentile
校准在同一版本上因动态 shape 直方图 bug 失败
（`shape (1024,16) into (1,1024)`），因此使用 MinMax。

端到端 A/B（4 ORT 线程、种子 1234、INT8 depth、首块 2 帧、16 帧块、
codec 重叠、各 4 次运行）：

| codec 模型 | RTF 中位数 | 首次 PCM | codec 累计 | WAV 对照 |
| --- | ---: | ---: | ---: | --- |
| FP32 基线 | 2.16 | 1.03 s | 2.49 s | 基准 |
| QDQ Conv + ConvTranspose（282 MB） | 1.80 | 0.81 s | 1.89 s | corr 0.976 / SNR 13.3 dB |
| QDQ 仅 Conv（362 MB） | 1.76 | 0.80 s | 1.82 s | corr 0.977 / SNR 13.4 dB |

同配置基线各轮 WAV 完全相同（corr 1.0），量化版长度相同但有可听差异，
需主观听测；两种量化范围的音质几乎一致，说明误差主要来自 Conv 激活的
MinMax 校准而非 ConvTranspose。仅 Conv 版 RTF 略好但文件更大，可按
存储/速度权衡选择。改进 SNR 的下一步是用真实非零 states 的激活做校准，
或升级量化工具后启用 Percentile/Entropy。

### 其它调参（同次扫描内的相对比较）

| 配置 | RTF 中位数 | 说明 |
| --- | ---: | --- |
| 4 线程 / 16 帧块 | 2.31 | 该轮基线 |
| XNNPACK 接管 codec | 3.74 | codec 2.8 -> 5.6 s |
| 24 / 32 帧块 | 2.56 / 2.53 | 更慢 |
| 6 / 8 线程 | 2.35 / 2.37 | depth 1.75 -> 2.13 s |

即 4 线程 + 16 帧块仍最优，XNNPACK 显著劣化。注意各轮基线在
2.07-2.42 之间波动，跨轮只比较同次扫描内的相对值。

### 结论更新

- 硬件确认：v73、2 个 HVX、无 HMX、2 MB VTCM，只支持整数精度；
  "8 Gen 4" 是误认，实际为 Snapdragon 6 Gen 4 (SM6650)。
- QNN 库齐全但 ORT build 没启用 QNN EP，NNAPI 也没有实测加速；
  NPU 通道在这套软件栈上仍不可用。
- 量化收益来自 CPU 侧的 INT8 快速路径而非 NPU：codec 静态 QDQ 把
  RTF 从 2.16 降到 1.76，首次 PCM 从 1.0 s 降到 0.8 s。
- RTF < 1 仍未达到；剩余 depth 约 1.8 s、codec 约 1.8 s、backbone
  约 0.53 s。下一步优先 depth 的 QDQ 量化、真实激活校准提升 SNR、
  以及把 backbone 改成进程内 JNI 调用。


### 帧内解码 gate/up 投影的 HVX 初测

使用官方 depth_frame.onnx 第一个 MLP 的 gate 和 up 权重，各为
[512, 2048] ONNX MatMul 右参数，转置成 GGML 行布局后顺序写入，
再写入 seed 1234 的 512 维 FP32 随机输入。`depth_projection_probe.cpp`
比较同一 GGML 图内的两个投影与沿输出维拼接的一个投影。Android 上
权重预上传，3 次预热后取 32 次中位数，包含输入上传和输出读回。

| 路径 | 两个投影 | 拼接投影 | 拼接相对 CPU FP32 最大绝对误差 |
| --- | ---: | ---: | ---: |
| CPU 4 线程 FP32 | 1.575 ms | 1.211 ms | 0 |
| HTP/HVX FP32 | 0.887 ms | 0.580 ms | 0.00138 |
| HTP/HVX Q8_0 | 0.459 ms | 0.421 ms | 0.02968 |

这是单层两个投影和随机激活的初测。整帧有逐码本采样依赖，实际收益取决于
层间交接、常驻权重、其他算子及同机 codec 调度。Q8_0 仍需真实语音验证。

### 五层 depth core 的首个码本步骤

`capture_depth_reference.py` 使用官方参考码本、固定中文文本和种子 1234，
构造 79 行 temporal prefix。它保存官方 ONNX 的 16 路采样分数、采样结果、
next_hidden 以及第一次本地 Transformer 的两个输入位置和输出。
`convert_backbone.py --depth` 从同一个 depth ONNX 提取五层 Llama core，
解析共享权重的 Identity 别名。用这两个真实位置运行 GGUF，在 Mac GGML CPU
上相对 ONNX 的末位置 hidden 最大绝对误差为 1.81e-5，在 SM6650 HTP/HVX
为 0.00469，RMS 为 0.00127。HTP profile 确认 Q 投影在 HVX 执行。
本次范围为 codebook 0 采样之后、codebook 1 采样之前的五层计算，
尚未覆盖后续逐码本 cache 更新、音频头或整段语音 A/B。

### 15 次 depth core 调用与逐码本重放

`capture_depth_reference.py` 导出官方 codebook 0 采样后的两个位置以及
后续 14 个位置，合计 16 行 FP32 embedding。将
`local_sequence` 和 `decoded_steps` 以小端 FP32 写出，用
`backbone-android depth-core.gguf HTP0 depth-all-input.f32 2 depth-all-HTP0.f32`
在禁用 HMX 的同一个 context 中跑 2 行 prefill 和 14 次单行 decode。
CPU 对照通过 atol/rtol=1e-4。HTP hidden 最大误差 0.02814，
atol=0.02、rtol=0.002 时有 1/7680 个元素超差。

`check_depth_codes.py` 对每个 hidden 使用官方对应音频头和固定噪声，
按官方温度、top-k、softmax 和 argmax 重放采样。官方 hidden 与
HTP hidden 都匹配 codebook 1–15 的 15 个决策。该实验以官方 token
对应的 embedding 喂入下一步，尚未验证自由生成和 EOS。

单次 HTP 记录为 prefill 7.48 ms 加 decode 107.78 ms，合计 115.26 ms，
还未包含音频头、采样和 codec，已经超过每帧 80 ms 的实时预算。
后续必须降低整步开销，不能把单层投影的加速比例当作整帧加速。

### Depth 权重量化初测

使用 `convert_backbone.py --depth --quant Q8_0` 或 `Q4_0` 导出矩阵权重，
RMSNorm 和 RoPE 仍用 FP32。沿用官方参考 embedding、固定噪声和禁用 HMX
的 SM6650 实验。以下为各方案单次 2 行 prefill 加 14 次 decode，排除
模型与 context 初始化，包含读回，尚未计入采样和 codec。

| 权重 | GGUF | Core 耗时 | 码本 1–15 与官方一致 |
| --- | ---: | ---: | ---: |
| FP32 | 76.05 MB | 115.27 ms | 15/15 |
| Q8_0 | 20.22 MB | 93.68 ms | 15/15 |
| Q4_0 | 10.72 MB | 89.61 ms | 13/15 |

Q4_0 的码本 5 和 15 分岔，且比 Q8_0 只少约 4 ms。本样本优先保留 Q8_0，
需要在常驻会话和多条真实输入上补跑自由生成、EOS、首段延迟与 RTF。

### 常驻 Q8_0 depth 和 FastRPC 轮询

`backbone-android MODEL HTP0 --depth-pipe` 使用 256 容量 context、
最多 2 行的提交和最多 16 个帧内位置。协议沿用 uint32 小端行数加 FP32
embedding，0 行消息清空 KV 元数据并重置位置，保留权重、设备会话以及分配。
同一个进程重放 12 次真实帧输入，15 个 hidden 逐位一致，最后一帧
用原音频头和噪声重放仍匹配 15/15 个码本。

固定 `GGML_HEXAGON_NHMX=0` 和 `GGML_HEXAGON_POWER_MODE=sustained_high_performance`，
同机热运行中位数（各 12 次，剔除首轮）随 `GGML_HEXAGON_OPPOLL` 从
0 到 1 由 104.104 ms/帧降到 39.705 ms/帧。计时包含 core 输出读回，
排除码本头和采样、codec、进程初始化。轮询两组的二进制 hidden 完全相同。
需在 CPU codec 同时运行时重测，确认轮询的 CPU 开销与端到端收益。

### 自由逐码本生成初测

`pipe_depth.py` 从官方 graph 导出 16 组音频头、输入 projection 和两种
audio embedding。主机按温度 0.9、top-k=10、softmax/noise、argmax
采样，仅在码本 0 应用 EOS mask。每次实际选中的码本决定送给常驻
Q8_0 HTP core 的下一行输入，帧末用官方 embedding 求和。
在同一驻留进程里用官方真实首帧输入重复三次，均匹配 16/16 个码本，
next_hidden 最大误差为 0。测试使用 Mac 采样和 ADB pipe，尚未验证
整句 EOS 或同机吞吐。流式 Python runner 增加了 depth adapter 入口。

### 手机同机的常驻 depth 与 codec 流水

Android 补丁增加常驻 `NativeDepth`，复用套接在本地进程 pipe 上的 HTP
context。CPU 侧保留 ONNX backbone、音频头、官方采样规则和 cached codec。
`pipe_depth.py --android depth_frame.onnx depth-weights.f32` 导出头部与
embedding。用 NDK 构建 `depth_heads_jni.cpp` 为 `libm5_depth_heads.so`，
本次使用 -O3 -ffast-math。设 `M5_NATIVE_DEPTH=./depth-resident`、
`M5_DEPTH_MODEL=depth-Q8_0.gguf`、`M5_OVERLAP_CODEC=1`、
`M5_CHUNK=8`、`GGML_HEXAGON_OPPOLL=1`、`GGML_HEXAGON_NHMX=0` 和
`GGML_HEXAGON_POWER_MODE=sustained_high_performance`。

同一短句、参考音频和种子 1234，纯 Java 音频头的流水 RTF 为 4.22–4.30
（两次热运行）；JNI 后为 1.64–1.74，首段 PCM 为 1.36–1.38 秒。
JNI 版生成 2.46 秒语音，CPU 官方路径生成 2.22 秒；仍需确认差异的听感与内容。
JNI 两次热运行 depth 约 2.09–2.14 秒，codec 约 3.52–3.74 秒，
二者与端到端时间因重叠而不能相加。

### 同机分块与 codec INT8 组合

保持常驻 Q8_0 HTP depth、JNI 音频头、CPU backbone 和 codec 流水，首块 4 帧，
种子 1234，同一条中文短句。FP32 codec 后续块 8、16、24 的热运行 RTF
中位数分别为 1.648、1.786、1.930。8 与 16 帧的 WAV 长度相同，
PCM16 最大差 1。大块减少 codec 累计执行时间，但最后一块更晚入队。

静态 per-channel QDQ 仅量化 codec Conv，在每块 4 帧时三次热运行
RTF 中位数为 1.370，首次 PCM 中位数为 1.036 秒，音频仍为 2.46 秒。
相对 FP32 codec 8 帧块的对齐 WAV，SNR 为 11.90 dB，需听测确认是否
符合可接受的音质损失。两组均未达到 RTF ≤ 1。

### 常驻双 HTP 会话

Android engine 在会话创建时启动 backbone 进程，每段用 0 行包清除
KV 元数据。Depth 保持独立常驻 HTP context，每帧清 KV。codec 使用仅
Conv 的 QDQ CPU 版并与生成重叠。禁用 HMX，启用 FastRPC 轮询和 sustained
high performance。固定种子 1234，4 ORT 线程，短句首块及后续均 4 帧。

| 短句 backbone | 热运行 RTF 中位数 | 首段 PCM | 合成耗时 | 音频长度 |
| --- | ---: | ---: | ---: | ---: |
| HTP FP32，每段加载 | 1.606 | 1.720 s | 3.564 s | 2.22 s |
| HTP FP32，常驻 | 1.348 | 1.144 s | 2.993 s | 2.22 s |
| HTP Q8_0，常驻 | 1.318 | 1.053 s | 2.610 s | 1.98 s |

改用较长中文句并将首块改为 2 帧，常驻 Q8_0 双 HTP 合成 6.14 秒音频，
两次热运行 RTF 1.091、1.068，首段 PCM 0.988、0.888 秒。
较长句未与相同分块的 CPU baseline 对照，输出长度差异也尚未听测或做 ASR。

### 长句同机基线与 codec 预热重叠

固定长句「今天的天气很好，我们一起去公园散步，看看路边的花，听听树上的鸟叫。」、
参考音色、种子 1234、4 ORT 线程、首块 2 帧、后续 4 帧和流水模式。
串行运行两种配置，剔除首轮后取三次中位数。

| 路径 | RTF | 首段 PCM | 总耗时 | 音频长度 |
| --- | ---: | ---: | ---: | ---: |
| 全官方 ONNX CPU | 2.457 | 1.320 s | 15.479 s | 6.30 s |
| 常驻双 HTP Q8_0 + CPU codec Conv QDQ | 1.084 | 0.945 s | 6.654 s | 6.14 s |

两边均成功写出 24 kHz 单声道 PCM。输出时长不同，仍需转写或听感核验。
增加 `M5_OVERLAP_PRIME=1` 实验选项，在生成开始前把参考音色的 codec
预热入队，让它与首批帧生成并行。首个生成音频块等待预热结束后再提交，
沿用同一 codec state。需要通过真机 A/B 验证 PCM 与效果。

长句 Conv QDQ 后续 4、6、8 帧的热运行 RTF 中位数分别为
1.08365、1.08355、1.09000。对应 codec 累计从 5.887 降到 5.318 秒，
整句耗时却近乎持平。增大分块已经无法单独消除剩余差距。
Conv + ConvTranspose QDQ 在后续 4 帧时为 RTF 1.146，
慢于只量化 Conv。增加独立的 `M5_CODEC_THREADS`，用于后续在 HTP
执行时单独调整 CPU codec 线程数。

长句完整分块扫描中，Conv + ConvTranspose QDQ 的 4/6/8 帧 RTF 为
1.146/1.121/1.128，均慢于 Conv-only QDQ 的 1.084/1.084/1.090。
用本地 Whisper base、语言 zh、temperature=0、FP32，对 CPU 参考 WAV
和双 HTP Conv QDQ WAV 做转写。两份都覆盖了句子至末尾，无明显整段遗漏。
CPU 转写末词为「鳥架」，混合版为「鸟角」，目标末词是「鸟叫」。
这项检查覆盖单句内容和结尾，不能据此判断音色自然度或其他文本的正确率。

`M5_OVERLAP_PRIME` 的同机长句 A/B 每组 4 次、剔除首轮：
关闭时 RTF 1.0622、首段 PCM 0.832 s，开启时 RTF 1.0364、首段 PCM
0.625 s。抽查两组第 3 次输出，147360 个 PCM16 样本全部相同。
当前长句距实时预算还差 0.223 秒。

Codec 线程独立扫描，保持预热重叠、首块 2 帧、后续 4 帧，其余不变。
剔除首轮的三次热运行表明 4 线程最快：

| Codec CPU 线程 | RTF | 首段 PCM | 总耗时 |
| --- | ---: | ---: | ---: |
| 2 | 1.0497 | 0.684 s | 6.445 s |
| 4 | 1.0345 | 0.620 s | 6.352 s |
| 6 | 1.0412 | 0.646 s | 6.393 s |

后续分块实验继续用 4 线程。另排队测试 Q4_0 depth 的整句表现，
对照 Q8_0 保持同设置。由于 Q4_0 在固定帧测试出现不同码本选择，
需将听感和转写与速度一起报告。

再扫首块/后续块 2/6、3/6、2/5，三次热运行中位 RTF 分别为 1.0361、
1.0296、1.0468。3/6 的首段 PCM 为 0.714 秒，2/6 为 0.630 秒。
这三组的 9 次热运行均未低于 RTF 1。最小中位数尚差 0.181 秒，
吞吐与首包折中仍优先保留 2/4 配置用于下一轮 A/B。

### Q4_0 depth 首次在长句达到实时

同一 seed 1234 长句、本地双 HTP、Q8_0 temporal backbone、CPU codec Conv QDQ、
4 codec 线程、预热重叠、首块 2 帧后续 4 帧。仅替换 depth GGUF。各 4 次，
剔除首轮后：

| Depth | RTF 中位数 | 热运行范围 | 首段 PCM | 总耗时 | 音频长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 1.031 | 1.030–1.037 | 0.624 s | 6.330 s | 6.14 s |
| Q4_0 | 0.943 | 0.941–0.948 | 0.619 s | 5.866 s | 6.22 s |

Whisper base 的同设置转写均覆盖整句，Q8_0 将末词「鸟叫」识为「鸟角」，
Q4_0 此外将「散步」和「听听」识为「站步」和「停停」。ASR 差异
提示 Q4_0 音质/发音风险，单个测试不能将识别错字直接归因于发音。
进一步检查 Q4_0 的短句、第二条长句、EOS 和跨样本表现。

### Codec 音质对照与当前速度边界

固定 temporal Q8_0、depth Q4_0、参考音色、文本和种子，用 FP32 cached
codec 替换 Conv QDQ。FP32 和此前 INT8 样本均输出 149280 个样本。
对齐波形 SNR 12.64 dB，峰值差 8973 PCM16 单位。用户报告 INT8 样本
有沙沙声，codec 量化是优先排查项。此对比未单独记录 codec 输入码本的 hash，
仍需逐帧固定码本重放与盲听来确认来源。短句 Q4_0 热运行中位 RTF 1.144，
第二条较长句为 0.995（三次中两次达标），转写分别覆盖全文和句末，
后者「顺路」被识成「深路」。

以第一条长句的 Q4_0 热运行数据估算，在保留当前生成速度且假设 codec
完全免费时，backbone 与 depth 串行合计约 5.39 秒，对应 RTF 约 0.87。
这是当前实现的理想重叠估算，忽略其他等待，并非硬件极限。

### Codec 量化复盘：C++ 原生推理、XNNPACK 与尾层 FP32

此前 codec 走 Java + ORT 的 harness（`M5Engine`/`AndroidBenchmark`）。
把同一套递推解码改写成 C++（`examples/m5-tts/codec_bench.cpp`，直接调
ORT C API）后，同一模型、同样 4 线程、同样 20 个 chunk 的隔离耗时从
引擎侧的 12.3 s / 9.82 s 音频降到 5.0 s / 6.3 s 音频，约 1.5×。
差距来自 Java 侧每 chunk 的状态张量搬运，不在算子本身。

设备上 `libonnxruntime.so` 编进了 XNNPACK。单层 `/conv_14/Conv`
FP32 由 5.92 ms 降到 5.05 ms（1.17×），数值误差 9.8e-6；对 FP32 整网
codec 为 4.96 s（chunk 8，5 轮取最小）对 CPU 4.97 s，收益有限。
QNN EP 仍不可用，NNAPI 也不占优。

逐算子 profile（4 chunk，ex4 配置 259.5 ms）显示量化 Conv 之后的热点：
ConvTranspose 26.8%、Conv 17%、Snake `Sin` 11.9%、MatMul 8.6%。
ConvTranspose 全部未量化，是下一步主要速度空间。

量化精度上限的定位：

| 配置 | codec 耗时 | codec RTF | 波形 SNR | 6–9 kHz SNR |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 4963 ms | 0.788 | 参考 | — |
| ex4（尾 4 层 FP32） | 3432 ms | 0.545 | 12.5 dB | +5.8 dB |
| 全量化 Conv | 3501 ms | 0.556 | 9.8 dB | −9.5 dB |
| 权重-only（无输出量化） | 7064 ms | 1.12 | 19.8 dB | +1.6 dB |

- 尾层 FP32 同时改善速度和质量：`/conv_31..34` 的相对误差 0.23–1.21，
  远高于其他层，且去掉输出 Q/DQ 后少了一次往返，反而更快。
- 继续放大到 ex6/ex8/ex10 只到 12.74/12.74/12.75 dB，平台期；
  中段层（conv_8/15–19、conv_10–14）逐个剔出 FP32 也只到 12.92 dB。
  残差是 31 层 8-bit 激活误差的分布式累积，不是某一层的病态。
- 分位裁剪（p99.9/p99.99）反而恶化到 3.18 dB：波形层的峰值就是信号，
  截掉尾部等于削波。u8 非对称激活也差于 s8（10.54 vs 12.65 dB）得
  注意该 12.65 来自旧 harness 流，本 runner 同一模型为 9.81 dB。
- 按通道激活量化不可行：ORT `QLinearConv:68/93` 要求输出 scale 与
  zero point 均为标量，per-axis 直接抛错。
- 权重-only 精度好但慢：无输出 Q/DQ 时 ORT 无法融合成 QLinearConv，
  退化成 FP32 Conv + 逐层反量化，反而比 FP32 慢。

线程数：4 线程最优，6/8 线程因大核只有 2 个而退化（FP32 0.79 → 1.01）。
chunk 8 比 chunk 4 快约 5–8%，代价是首段延迟翻倍，稳态可与
`M5_FIRST_CHUNK=2` 组合使用。

结论：codec 若要 20 dB 级音质只能不量化激活；C++ 化后 FP32 codec
的 RTF 为 0.79，配合 depth Q4_0（0.943）重叠仍有机会压在 1 以内。
默认档保留 ex4（0.545 / 12.5 dB），高音质档直接用 FP32 model。

### 当前最佳配置（已上线为默认）

用户盲听确认 ex4 样本可接受，因此把默认 codec 换成 ex4：
`model/codec_decoder_cached.onnx` 已用 `codec-qdq-ex4.onnx` 覆盖，
原 FP32 备份为 `model/codec-fp32.onnx`，量化档另存 `model/codec-qdq-ex4.onnx`。
引擎不再需要设置 `M5_CODEC_MODEL`，缺省即走 ex4（实测 codec_s
5.11 s → 3.87 s 于 3.9 s 短句）。

配合 `M5_CODEC_THREADS=3`（原为 4）。同一长句、种子 1234、三次取稳态：

| 线程 | RTF | codec_s | depth_s | 首段 PCM |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.979 | 9.50 | 7.14 | 0.688 s |
| **3** | **0.890** | **8.37** | **7.27** | **0.583 s** |
| 4 | 0.930 | 9.00 | 7.38 | 0.625 s |
| 6 | 0.949 | 9.21 | 7.45 | 0.642 s |

只有 2 个大核，codec 线程再多会和 depth 的 host 线程互抢。chunk 6/8 能压低
codec_s（7.86 s）但整句 RTF 反而升到 0.90，因为尾部 chunk 变大后
最后一段 PCM 出现更晚，故 chunk 保持 4、`M5_FIRST_CHUNK=2`。

两条否决结论：

- **量化 ConvTranspose**：SNR 只掉 0.28 dB（12.65→12.37）且模型从
  362 MB 降到 282 MB，但 codec 变慢（3.44→3.80 s），ORT 的
  QLinearConvTranspose 内核不如 FP32 路径。
- **关闭忙轮询**（`GGML_HEXAGON_OPPOLL=0`）：codec 仅快 8%，
  depth 从 7.3 s 劣化到 14.5 s，整句 RTF 1.78。

### codec 与 depth 的并发账

同一句、`M5_CODEC_THREADS=3`，只切 `M5_OVERLAP_CODEC`：

| 重叠 | 整句 RTF | codec_s | depth_s |
| --- | ---: | ---: | ---: |
| 开 | 0.890 | 8.38 | 7.26 |
| 关 | 1.379 | 6.86 | 5.67 |

拆开看，codec 的 8.38 s 里约 1.5 s 是并发代价；而串行的 6.86 s 本身
仍比 C++ 隔离测量（3.44 s / 6.3 s 音频，折算到 9.82 s 约 5.3 s）慢约
1.3 倍。也就是说，Java 侧每 chunk 创建并搬运 24 个状态张量的开销，
以及它与 depth host 线程的带宽争抢，各占一部分。把 codec 换成独立
C++ 进程是下一步最大的单点收益，但需要引擎源码（`M5Engine.java`
在官方仓库，本地仅有 `android-stream.patch`）。

### 引擎原生化：m5-engine（C++ 取代 M5Engine 的推理部分）

`examples/m5-tts/m5_engine.cpp` 复刻了 Java 引擎的推理链路：backbone 与
depth 两条管道直接驱动，depth 头（投影、逐 book 变换、top-10 过滤、Gumbel
采样、hidden 累加）在本地计算，codec 递推走 ORT C API，分段、prime 丢弃、
2400 样本 tail、重叠线程与 WAV 写入全部一致。分词仍留在 Java
（`TokenDump.java` 导出 ids），因为它依赖 Unicode 属性类，而推理侧不需要。

构建与运行（设备端）：

```
aarch64-linux-android31-clang++ -std=c++17 -O3 -ffp-contract=off \
  -o m5-engine m5_engine.cpp -I<ort>/headers -L<ort> -lonnxruntime -llog -landroid -pthread
./m5-engine model --tokens ids.bin --out out.wav --cwd $PWD --threads 3
```

对齐过程中确认的几个语义细节，任何一条错了都会静默跑偏：

- depth 的第 1 个 book 送 2 行（投影后的 hidden + code0 的 embedding），
  第 2 个 book 起只送 1 行（上一个 code 的变换）。Java 端的滚动 buffer
  在 `flip()` 之后 remaining 会从 1024 掉到 512，容易读成「一直是 2 行」。
- 投影的行数不是常量：book 头是 2051 行，`projectedHidden` 只有 512 行。
  最初按 2051 行写 512 长度的缓冲区，越界踩栈。
- prime 段（参考音色的最后 8 帧）必须推进 codec 状态但丢弃输出。
- 重叠模式下 prime 和正常批次要带 emit 标记一起排队，否则 prime 的音频
  会混进结果（时长会多 0.64 s）。
- 生产者等待 `!has_pending`，消费者取走批次后必须 notify，否则收尾死锁。

实测（同一长句、种子 1234、3 线程、ex4 codec）：

| 实现 | 总耗时 | RTF | backbone | depth | codec | 首段 PCM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Java M5Engine | 8.94 s | 0.910 | 1.03 s | 7.42 s | 8.47 s | 0.648 s |
| m5-engine (C++) | 8.31 s | 0.913 | 0.94 s | 7.16 s | 7.52 s | 0.635 s |

两者音频长度不同（采样对浮点极度敏感，码本在前 3 帧后分叉），因此按帧
归一后速度基本持平，逐段看 depth 与 codec 各快 4–11%。Whisper base 对两段
音频的转写完全一致，RMS 0.057 对 0.055、峰值 0.416 对 0.414，可判定等价。

下一步：把 depth 每帧 15 次管道往返改成共享内存环，以及把 tokenizer 用
ICU 或预编译 trie 落到 C++，才能真正脱离 JVM。

### 分词器原生实现（m5_tokenizer）

`Tokenizer.java` 之前是唯一还需要 JVM 的环节。现在 `m5_tokenizer.cpp` 复刻了同一套
流程，`m5-engine --encode` 用来和 `TokenDump`/`IdDump` 逐 id 对齐：

- `M5TK1` 是 Java `DataInputStream` 写的**大端**格式：magic、bos/eos、切分正则、
  词表、256 项字节→字符/ id 表、merge 表（rank + merged id）、特殊 token 表。
- 切分正则不引 ICU，直接按 GPT-4 那条式子的分支顺序手写：
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)`、`[^\r\n\p{L}\p{N}]?\p{L}+`、`\p{N}{1,3}`、
  ` ?[^\s\p{L}\p{N}]+[\r\n]*`、`\s*[\r\n]+`、`\s+(?!\S)`、`\s+`。
  两条空白分支的**回溯语义**必须照抄：`\s+(?!\S)` 在遇到非空白时会吐出最后一个
  空白字符（"  hello" → `" "` + `" hello"`），`\s*[\r\n]+` 只吃到最后一个换行。
- `\p{L}/\p{N}/\p{Z}` 用 `gen_unicode_tables.py` 从 Python 的 unicodedata 生成区间表
  （letters 622 段、numbers 133 段、separators 8 段，共约 7 KB），编译期烤进
  `unicode_tables.h`，省掉 ICU 依赖。
- BPE 仍按 `ignore_merges=true`：整片先查词表，再按 rank 最小优先合并。

验证：设备上用 `IdDump`（Java）与 `tok-test`（C++）对同一批字符串比对，**200 条
随机串（含中文、英文缩写、emoji、全角标点、Tab、全角空格）与 3 条含 `\r` 的边界
用例输出逐字节相同**。加载 5.4 MB 词表约 0.2 s（128k 词表 + 100k merge），一次性。

### 子进程传输：共享内存环（m5_ring.h）

depth 阶段每帧 16 次往返（1 次 reset + 15 次 1~2 行推理），每次走两条管道要付
两次调度和两次拷贝。改成 `memfd_create` + `MAP_SHARED` 的固定块，fd 通过
`--ring-fd N` 传给子进程，握手全是原子序号，交换本身零系统调用。

`ring-bench`（`ring_stub` 假装 stage，`--work-us` 模拟计算）实测每次往返：

| 模拟计算 | 管道 | 环 | 每次节省 |
| --- | ---: | ---: | ---: |
| 0 µs（纯传输） | 122.8 µs | 1.4 µs | 121 µs（85×） |
| 300 µs | 391.5 µs | 287.1 µs | 104 µs（26.7%） |
| 3700 µs（接近真实 depth core） | 3583 µs | 3471 µs | 112 µs（3.1%） |

结论要按真实量级读：真实 depth core 一次 1 行 decode 约 3.5–3.7 ms，传输只占
3%，整句 depth 段（约 125 帧 × 15 次）能省 0.2 s 上下。收益有限，但它同时取消了
backbone 首段 2048 行（4 MB）的拷贝，并且让"谁在占用 host 线程"变得可控。

引擎先按环启动，再发一次 reset 做探针：老二进制不认识 `--ring-fd` 会立刻退出，
探针失败就自动重启成管道模式（实测老 `depth-resident` 会报 `invalid prefix length`
并退回）。**要让环真正生效，需要用本仓库的 `backbone.cpp` 重新编译
`backbone-resident`/`depth-resident`**；本机没有 Hexagon SDK 与 Android 版
llama.cpp 构建树，这一步留到有工具链的环境执行。

### 引擎库化与 JNI

`m5_engine.cpp` 拆成 API（`m5_engine.h`）+ 实现，CLI 在 `m5_engine_main.cpp`，
JNI 在 `m5_jni.cpp`，逻辑上没有第二份实现：

| 文件 | 职责 |
| --- | --- |
| `m5_tokenizer.{h,cpp}` | M5TK1 解析、切分、BPE、特殊 token |
| `m5_ring.h` | 环协议（父子两端共用） |
| `m5_engine.{h,cpp}` | 加载模型包、条件化、分段、两个 stage、depth 头、codec、WAV |
| `m5_engine_main.cpp` | CLI：`--text` / `--tokens` / `--encode` |
| `m5_jni.cpp` | `NativeEngine` 的 5 个 native 方法 |
| `backbone.cpp` | stage 进程，新增 `--ring-fd` |

JNI 侧只做三件事：字符串转换、`Engine*` 句柄、把 PCM 分块回调给
`PcmListener.onPcm(float[], int)`。回调发生在 codec 工作线程，因此每次都要
Attach/Detach —— 只 Attach 不 Detach 会让 VM 永远不退出（实测进程卡住直到超时）。

设备实测（同一句、种子 1234、3 线程）：

| 路径 | 音频 | 耗时 | RTF | 首段 PCM | 回调 |
| --- | ---: | ---: | ---: | ---: | ---: |
| JNI（`NativeEngine.synthesize`） | 4.14 s | 4.00 s | 0.966 | 0.666 s | 14 块 / 99360 样本 |
| CLI（`m5-engine --text`） | 4.14 s | 4.00 s | 0.966 | 0.666 s | — |

两条路径输出的 WAV **逐字节相同**，同一句重复运行也完全相同（确定性）。
多段文本（含句号切分）验证过：`segments=2`、两段各自 reset、WAV 顺序拼接。

与 Java 引擎同句对比（`AndroidBenchmark`，同样 3 线程与 ex4 codec）：Java 3.86 s /
RTF 0.989，原生 3.99 s / RTF 0.966（音频长度不同，因采样对浮点极其敏感，前 6 帧
逐样本一致后分叉）。**按每帧耗时两者持平（约 72 ms/帧），速度不是这次改造的收益**；
收益是卸载 JVM（分词、条件化、分段、状态搬运全在 C++）、去掉 `TokenDump` 往返、
以及给环传输留好接口。

下一步只剩两件：用本仓库源码重建两个 stage 二进制以启用环；把
`depth` 的 15 次逐 book 往返做成常驻 batch（受自回归依赖限制，需改 depth 模型图）。

### Docker 工具链：stage 二进制现在可以重建

之前两次卡在"没有 Hexagon SDK，两个常驻进程无法重建"。实际上本机有
`ghcr.io/snapdragon-toolchain/arm64-android:v0.8`（7.45 GB，含 Hexagon SDK
6.7.0.0 / Tools 21.0.04 / NDK r29），用 Docker 就能完整构建：

```
examples/m5-tts/docker_build_stage.sh /tmp/out
```

镜像只有 amd64 版，在 Apple Silicon 上是模拟执行；全量首次构建约 4 分钟
（llama + ggml + 5 个 HTP skel 里只编 v73），增量只要十几秒。两个坑：

- 镜像里没有 cmake/ninja，脚本先 `apt-get install`。
- 顶层 configure 必须显式传 `-DPREBUILT_LIB_DIR=toolv21_v73`：SDK 的
  `hexagon_fun.cmake` 在被 include 时就执行
  `string(FIND ${PREBUILT_LIB_DIR} ...)`，变量为空在 CMake 3.31 下是致命错误。

产物（`libllama.so`、`libggml*.so`、`libggml-hexagon.so`、`libggml-htp-v73.so`、
`backbone-resident`）与设备上原有那套逐字节等价：同一长句、3 线程，新旧库
RTF 都是 0.898–0.903，depth 7.02–7.06 s、codec 7.50–7.56 s。也就是说设备上
原来那套就是这么构建的，这次只是把过程变成可复现的。

### 环传输实测：在这颗 SoC 上是负收益（结论修订）

上一节按 stub 基准估过"整句能省 0.2 s"。真实流水线跑下来是反的，必须修订：

| 配置 | depth 段 | 子进程自报 core_ms/帧 | 整句 RTF |
| --- | ---: | ---: | ---: |
| 管道（pipe） | 7.06 s | 36.7 ms | 0.903 |
| 环 + 忙等（sched_yield） | 7.67 s | — | 0.976 |
| 环 + futex 睡眠 | 12.4 s | 62.4 ms | 1.54 |

关键不是握手变慢，而是**子进程自己的 `llama_decode` 变慢了**：62.4 ms 对
36.7 ms。握手只占其中一小部分。继续二分后得到一个更意外的结论——把环路径
**编进二进制**（哪怕 `use_ring == false`、代码从不执行）就会让管道路径慢 40%：

| 变体 | 是否含环代码 | core_ms/帧 | JNI RTF |
| --- | --- | ---: | ---: |
| 只含 `m5_ring.h`（不调用） | 头文件 | 40.8 | 0.965 |
| 只构造 `ring::Server`（不调用） | 对象 | 43.2 | 0.986 |
| `if constexpr` 两个实例都在 | 是 | 63.5 | 1.558 |
| 只实例化管道路径 | 否 | 41.8 | 0.960 |

同一份源码在 CLI 下新旧都是 3.34 s，只有在 JVM 父进程下差异明显；换成
NDK 28 / clang 19 重编、改成动态链接 libc++、去掉计时插桩都复现同样的差异，
所以不是工具链也不是插桩，而是代码生成/布局层面的副作用。具体机制没有追到
底——继续追下去性价比太低，而结论已经够用：**ring 在 SM6650 上不要用**。

因此 `backbone.cpp` 用 `M5_STAGE_RING`（默认 0）把环做成编译期开关：

- 默认构建不含环，管道路径保持最快；
- `-DM5_STAGE_RING=1` 才有 `--ring-fd`，引擎用 `--ring` 启用，否则拒绝 fd 并
  自动回退管道（实测回退正常，`this build has no shared-ring transport` 之后
  打印 `pipe transport` 继续跑完）。

另外修掉一个关闭竞态：父进程退出时把 `seq_in` 加一，子进程会把这次"告别"
当成请求去处理，越过 `rows` 边界后抛
`embedding message exceeds remaining context`。现在 `Server::next` 在醒来后
先查 `quit` 再读 `rows`。

`m5-engine` 的默认传输相应改回管道，`--ring` 变成显式开关，`--pipe` 保留为
兼容别名。设备当前部署的是不含环的构建。

### 串行路径的完整账目与剩余瓶颈（2026-11-14 下午）

对长句（115 帧、9.10 s 音频、3 线程、ex4 codec）逐项拆解，串行路径每帧
约 70 ms：

| 环节 | 每帧 | 全句 | 备注 |
| --- | ---: | ---: | --- |
| backbone decode | 8.2 ms | 0.94 s | 33 MB Q8_0，1 token |
| depth stage 往返（含握手 4 ms） | 47.3 ms | 5.44 s | 15 次 1-2 行 decode |
| 其中 DSP 侧 `llama_decode`（core_ms） | 42.9 ms | 4.94 s | 19M 参数 5 层 |
| 引擎侧 depth heads | ~14.5 ms | ~1.7 s | 16 路 2051×512 matvec |
| codec（与 depth 重叠） | — | 7.54 s | 隐藏在串行路径下 |

关键测量与结论：

- **depth decode 不是带宽地板，是每次提交的开销地板**。19M 参数 1 token
  的计算量在 2 HVX 上不足 1 ms，实测 2.4–2.9 ms；Q8_0（21 MB）对 Q4_0
  （10.7 MB）只慢 9%，若带宽受限应翻倍。后端参数全扫（OPQUEUE 1–32、
  HOSTBUF、NHVX 4、OPFUSION 2）都在 ±1% 噪声内，无可用杠杆。
- backbone 与 depth 的耗时与各自权重字节数成正比（32 MB→8.2 ms、
  10.7 MB→2.9 ms，等效 ~3.8 GB/s），但这更像每 op 提交路径的固定成本
  而非 DDR 带宽，因为字节翻倍只慢 9%。
- **depth heads 改 f16 存储**（`m5_engine.cpp` 的 `project_f16`，
  `-march=armv8.2-a+fp16`，FMLAL 内核，权重 68 MB→34 MB/帧）：
  音频完全等价（时长/RMS/ASR 三项一致），RTF 持平（0.901 对 0.899）——
  单独看不挣速度，但它把每帧 ~350 MB 的总 DDR 流量砍掉 10%，属于
  系统性减压。代码保留，编译期守卫，无 fp16 时退回 f32 内核。
- **DSP 侧在 codec 并发时会变慢**：同一路径无重叠时 core 35.9 ms/帧，
  有重叠 42.9 ms/帧（+20%）。整帧的 DDR 流量约 350 MB（depth 160 + codec
  90 + heads 68 + backbone 32），四个客户端在共享内存系统里互相拖累。

下一步真正的杠杆（按性价比排序，都需要较大投入）：

1. **depth 的 15 次 decode 合并**：受逐 book 自回归依赖限制，唯一出路是
   改 depth 模型图（一次前向直接出多 book，或用 draft/verify 投机）。
2. **后端融合**：每 decode ~2 ms 的提交开销在 `htp-fusion` 里继续扩大融合
   图案，把更多节点并进一次 FastRPC 调用。
3. **codec 权重流量**：每 chunk 全量流 362 MB 权重；若权重能 int8 化且保住
   QLinearConv 快速路径（当前动态量化路不通），帧级流量减半，DSP 侧也能受益。

### Codec 深度量化：ConvTranspose→MatMul + 全 int8 + 融合 Snake（2026-11-14 下午）

三步递进，长句（115 帧、9.10 s 音频、3 线程）实测：

| 模型 | 体积 | RTF | codec | depth | 质量（vs fp32） |
| --- | ---: | ---: | ---: | ---: | --- |
| qdq-ex4（原部署，Conv int8） | 362 MB | 0.898 | 7.49 s | 7.01 s | corr 0.982 / 14.4 dB |
| mm-safe（+MatMul int8） | 221 MB | 0.884 | 7.08 s | 6.95 s | corr 0.976 / 13.3 dB |
| ctmm-q（+CT→MatMul int8） | 141 MB | 0.873 | 6.08 s | 6.87 s | corr 0.973 / 12.8 dB |
| **snake（+融合 Snake 算子）** | 141 MB | **0.837** | **4.68 s** | **6.60 s** | 同种子 corr 0.9998 |

构建流水线（全部可复现，`examples/m5-tts/`）：

1. `decompose_convtranspose.py`：6 个 ConvTranspose → 单 MatMul + 交错/pad。
   数学恒等（pad=0/dil=1/group=1，K=S 或 K=2S）：out[b,c,i*S+r] += x[b,:,i]@W[:,c,r]
   折叠成 [B,L,Cin]×[Cin,K*Cout] 一个 GEMM，K=2S 时两组 overlap-add。
   验证 corr 0.9995 / 30 dB（残差为 Pow→Mul 的浮点重排）。顺手把 Pow(x,2)→Mul(x,x)。
   坑：W_flat 必须按 (c,r) 顺序展平（直接 reshape W，先 transpose 会错位）；
   ORT 图优化器给无名节点起的融合名会撞车，所有生成节点必须显式命名。
2. `quantize_codec_static.py --ops MatMul`：68 个权重 MatMul int8（含 6 个 CT 分解的），
   17 个激活×激活 MatMul（QK^T/AV）必须排除——量化它们掉 1.3 dB（Softmax 敏感）。
   尾部 CT（conv_20/27）保留 fp32 无收益（12.78 vs 12.75 dB），全量化即可。
3. `snake_fuse.py` + `m5_snake_op.h`：29 处 Snake 链
   （Mul→Sin→Pow→Mul→Add，5 遍读写）融合为一个 `m5::Snake` 自定义 ORT 算子：
   范围归约（u = t − π·round(t/π)）+ 9 阶 sin 多项式，单遍 NEON。
   实测 codec 6.08→4.68 s（-23%），同种子 WAV corr 0.9998 / 34 dB。
   算子注册坑：ORT 建 session 时会调 `GetInputCharacteristic` 等回调，
   置空会段错误（exit 139）；`KernelContext_GetInput` 是 3 参数。

Snake 输入范围实测（16 帧块）：p99 大多 <3，但极值可达 ±158（decoder.5），
所以范围归约不可省。图级多项式展开（13+ 算子）会把内存流量打爆（每链 5 遍
→ 13 遍），必须融合成单算子才有效——这是本轮最重要的反直觉结论。

**瓶颈已转移**：snake 版后 codec 4.68 s < depth 6.60 s，vocoder 不再是墙钟约束。
下一个杠杆是 depth 的 15 次串行 decode（模型图改造/投机解码）与 htp-fusion
的提交开销（每 decode ~2 ms FastRPC）。

### Depth decode 成本模型与 f16 宏事故（2026-11-14 傍晚）

**llama_decode 成本模型**（新工具 `examples/m5-tts/depth_batch_bench.cpp`，设备实测
depth-Q4_0 / HTP0，28 行总工作量的四种分解）：

| 形状 | 总耗时 | 含义 |
| --- | ---: | --- |
| 1×28 | 26.8 ms | 一次提交 |
| 14×2 | 42.6 ms | 现状（15 次串行提交） |
| 28×1 | 61.1 ms | 逐行提交 |
| 1×2 | 3.05 ms | 单调用基线 |

拟合：**F ≈ 1.2 ms/次提交 + R ≈ 0.9 ms/行**。F 是 DSP 侧每 op 的 kernel 启动
开销（图一次 flush_sync，~50 op × 25µs），R 是逐行计算。15 次调用合并为 1 次
理论上省 37%（约 2 s/句），但被自回归采样链堵死：

- 帧间 code 重合率 = **0%**（16 book 全查过，lag 1-5 都是 0）——draft/投机解码无草稿可用；
- Jacobi 迭代沿 15 步链传播，收敛需 O(15) 轮，比串行更差；
- `next_hidden`（codes 的表查找和）反馈给下一帧 backbone 输入，帧间批量化也被 AR 锁死。

**htp-fusion 余量已尽**：`opt_opfusion` 默认已是 1（全开）。基准隔离实测
OPFUSION=0 vs 1 只差 5%（44.6 vs 42.5 ms）——深度图太小，减 op 数量这条路收益见底。

**f16 heads 一直是死代码（宏名事故）**：当初加 f16 时守卫写的是
`__ARM_FEATURE_FP16` / `__ARM_FEATURE_FP16FML`（GCC 风格），而 NDK r28 的
clang 19 定义的是 `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` / `__ARM_FEATURE_FP16_FML`
（带下划线）。两个宏都没匹配上，project_f16 从未编译进去——当时的 objdump
"fp16 已生效"检查是误报，"RTF 持平"也不足为奇。修复守卫（m5_engine.cpp 顶部
与 FMLAL 内核两处）+ `-march=armv8.2-a+fp16+fp16fml` 后 objdump 128 条 fmlal。

**修复后的真实收益**（交替 A/B，同种子）：

| | 帧数 | depth/帧 | RTF（5 句） |
| --- | ---: | ---: | --- |
| f32 heads | 115 | 57.3 ms | 0.83–1.06 |
| **f16 FMLAL** | 139 | **48.3 ms** | **0.72–0.89** |

每帧 -16%，RTF 5/5 句改善（-11~-13%），JNI 0.90→0.80。ASR 内容两版一致
（f16 版还把"的花"听对了）。注意：f16 数值扰动会让 AR 采样分叉——同句帧数
轻微上偏（4 句合计 +8%，115→139 是单句的韵律分叉），墙钟 4/5 句更快。
与 Java 参考的逐位一致性从此明确放弃（此前"完全等价"是死代码假象）。

结论：**depth 的 15 次串行 decode 在本机型没有精确压缩路径**（成本模型已量化、
草稿/迭代/跨帧批量化/融合四条路全部实测排除）。剩余大杠杆只有改模型图
（一次前向出多 book，需训练）或投机解码（需训练草稿模型）。

**transform 行消融（最后一个假设，已排除）**：`M5_ZERO_TRANSFORM=1` 把 15 步的
code 条件 transform 行全部置零（`m5_engine.cpp` 留有该环境开关），同种子对比：

- serial 69 帧正常 EOS；零 transform **500 帧跑满 kMaxFrames 都不终止**
- 前 69 帧 codes 重合率 **0.27%**（≈随机水平），逐 book 全部 0–1.4%

即 transform 行完全承重（包括 EOS 决策），串行链是模型语义的一部分。
至此 15 次串行 decode 的全部精确压缩路径（批量合并/投机草稿/Jacobi/跨帧/
融合/哑输入）**六条全部实测排除**，结论：不改模型（需训练）就到头了。

当前基线（f16 FMLAL 版，已部署）：CLI 长句 RTF **0.717**（11.0 s 音频，
首包 0.526 s），JNI RTF **0.791**。相对今天早上的 0.898/0.964 累计提升 20%/18%。

### 声音克隆：离线一次，缓存复用（2026-11-14 晚）

M5 的克隆语义（`infer_clone.py` / `infer_onnx.py`）：参考音频 + 逐字转写 →
`codec_encoder.onnx` 编成 16-book codes → 与 `prompt_text` 一起作为条件
（`condition()` = 控制 token + 参考文本 + 目标文本；codes 逐帧求和 audio_emb
进 backbone 前缀）。设备端没有编码器，所以克隆在主机离线做一次，结果缓存
到 `voices.json`（codes 即克隆结果，推理直接消费，不再跑编码器）。

新增工具 `examples/m5-tts/clone_voice.py`：

```
# 克隆/更新一个音色（名字后续用 --voice 选）
./clone_voice.py ref.wav "参考音频的逐字转写" --name alice \
    --model /tmp/m5-official/onnx --voices model/voices.json

# 查看已缓存的音色
./clone_voice.py --list --voices model/voices.json
```

- 预处理与 M5 `read_reference` 严格对齐（混单声道、24k、峰值归一、静音裁剪：
  0.15s 间隙切除 + 50ms 保留），编码前校验 encoder SHA256；
- **一致性已验证**：用官方样例音频重新克隆，43×16 codes 与原 voices.json
  **逐位相同**；
- voices.json 按 name 增改（同名覆盖、追加新声），encoder 哈希不一致时拒绝混编；
- 引擎侧 `--voice N|NAME`（纯数字=索引，否则=名字；同名同种子输出逐字节一致），
  `--voices FILE` 指定别的缓存文件。JNI 仍走索引（Java Config.voice 是 jint）。

多音色 A/B（变调参考音频造第二音色）：两个 voice 的 ASR 内容一致，
谱质心 1878→2072Hz、时长/RMS 均不同——音色载体确实是 codes 前缀。

Pipeline 复查（对照 M5 源码）：

- text_normalization.py（WeTextProcessing）不在 M5 默认推理路径里
  （infer_onnx.py 也未用），引擎不实现它是对齐行为，非差距；
- `--first N`（首 chunk 帧数）实测 1/2/3 档首包 0.45–0.59s，在噪声内，
  默认 2 已最优——首包成本在固定启动（stage 进程拉起 + prefill）；
- 克隆链路本身即优化后的 pipeline：编码器只在克隆时跑一次，推理路径
  只读 voices.json（几百字节 JSON，加载可忽略）。

### 交付：m5-server（HTTP）与安卓 App（2026-11-14 夜）

**m5-server**（`examples/m5-tts/m5_server.cpp`，vendor/cpp-httplib，CMake 目标 m5-server）：

- `GET /health`、`GET /v1/models`、`GET /v1/voices`（列出 voices.json 缓存）；
- `POST /v1/audio/speech`（OpenAI 兼容）：`{"input","voice"?,"response_format"?"wav"|"pcm"}`；
  wav 返回完整音频并在 `X-M5-RTF`/`X-M5-First-Pcm-S`/`X-M5-Frames` 头带统计；
  pcm 为 s16le 24k 分块流式（生成线程 + chunked provider，断连可取消）；
- `POST /v1/voices/clone`：导入离线提取的 codes（clone_voice.py 或 App 产物），
  按 name upsert 进 voices.json（保留原条目原文、encoder_sha256 校验、30s/16book 校验）。
- 引擎新增 `Engine::set_voice(selector)`：运行时重读 voices.json 切换音色（每请求可选声）。
- 设备实测：wav 合成 RTF 0.81（首包 0.40s）；pcm 流 1.66s 音频正常；克隆导入后
  立即可选可用；400/404 错误路径齐全。运行：设备上
  `./m5-server model --port 8080 --cwd $D`，主机 `adb forward tcp:18080 tcp:8080`。

**安卓 App**（M5-TTS/android，gradle 8.9 + AGP 8.7.3，JDK 17）：

- 构建：`ANDROID_HOME=$HOME/Library/Android/sdk JAVA_HOME=<jdk17> \
  M5_VOICE_ENCODER=/tmp/m5-official/onnx/codec_encoder.onnx ./gradlew :app:assembleDebug`；
  codec_encoder（190MB，SHA256 校验）烘进 assets，APK 197MB。
- 模型包 `.m5model`：M5MODEL-ZIP-V1 头 + ZIP（bundle.json + 9 件：backbone/depth_frame/
  codec_decoder_cached onnx、codec_cached.json、双 embeddings.npy、tokenizer.bin、
  voices.json、manifest.json）。可由 package_model.py 产（需 npz 音色）或手工组 ZIP
  （bundle.json 仅需 size+sha256）。App 解包校验后装入 files/m5-model。
  **bundle.json 是生成按钮的使能开关**（`generateButton.setEnabled(...bundle.json.isFile())`），
  run-as 手动预置模型时必须带上它，否则按钮永久灰显。
- 真机全流程实测（Honor，Android 15）：
  1. 模型识别："模型已导入 · 2 个内置音色 · 完全离线"；
  2. 内置音色生成：4.06s 音频，ASR 与输入一致；
  3. **新增音色（设备端克隆）**：SAF 选录音 → 对话框填名称/原文 → "离线创建" →
     VoiceEncoder（assets 的 codec_encoder.onnx，ORT）离线编码 → .m5voice 存入
     files/m5-user-voices → 自动选中；随后生成 3.90s 音频（与内置音色产物不同），
     ASR 内容保留；
  4. **导入音色包**：SAF 选 .m5voice → "自定义 · 导入测试"入列表并选中。
- UI 自动化注记（Android 15）：`input text` 对 CJK 抛 NPE（KeyEvent.getEvents null），
  ASCII 可用——对话框的中文原文输入需人工或剪贴板；`uiautomator dump` + 坐标 tap
  在应用无僵死时可靠；锁屏/SLEEP-WAKE 后需重新拉起 app 恢复注入。
- 性能定性：Java/ORT 引擎（fp32 ONNX，CPU）明显慢于原生 m5-engine（HTP），
  单句分钟级；App 走的是 M5-TTS 原设计的可移植路径。NativeEngine（JNI+HTP，
  RTF 0.72）已在库侧就绪，接线进 MainActivity 是后续可选优化。

### App 接入 NativeEngine：CPU 后端提速 1.6×，HTP 被系统策略封禁（2026-11-14 深夜）

**结论先说**：普通 App 在本机（Honor/Android 15）**无法使用 HTP**——fastrpc 设备
`/dev/adsprpc-smd` 权限为 0664 system:system，app uid 打开即 EACCES（DAC 拦截，
且 untrusted_app 的 linker namespace 也不可见 /vendor/lib64 的 libcdsprpc.so 依赖闭包）。
HTP 路径只属于 shell 域（adb CLI / m5-server）。App 走 **native CPU 后端**：
量化 gguf（backbone Q8_0 32MB / depth Q4_0 10.7MB）+ 原生 f16 depth heads + 原生 codec，
比纯 Java/ORT fp32 路径快约 1.6×：

| App 合成路径 | 4.0s 音频耗时 | RTF |
| --- | ---: | ---: |
| Java/ORT（fp32 onnx，兼容回退） | 12.6 s | 3.1 |
| **Native CPU（量化 gguf + 原生 heads）** | **7.9 s** | **1.98** |
| Native HTP（仅 shell 域可用） | ~3 s | 0.72 |

**架构**（`useLegacyPackaging=true` 是关键）：

- Android 10+ 禁止 exec 应用数据目录的文件，stage 二进制以 **libbackbone_resident.so /
  libdepth_resident.so** 之名进 jniLibs，安装时提取到 nativeLibraryDir（可 exec）。
  同目录再放 libllama/libggml{,-base,-cpu,-hexagon,-htp-v73}.so；JNI 的
  nativeCreate 里 setenv `LD_LIBRARY_PATH`+`ADSP_LIBRARY_PATH` 指向该目录
  （app 进程环境不可信，子进程继承 setenv 后的环境）。
- 模型包 bundle.json 新增可选 **native_files**（backbone-Q8_0.gguf / depth-Q4_0.gguf /
  depth-weights.f32，共 245MB），ModelInstaller 按同款哈希校验解入 files/m5-model；
  不带 native_files 的旧包照常工作（App 自动回退 Java 引擎）。
- JNI 扩展：nativeCreate 增 voices_path/voice_name/adsp_dir；新增
  `nativeSetVoice`（复用 Engine::set_voice，切音色不重启 stage）。
  App 把 内置+自定义 音色合并写 files/m5-voices.json 缓存，按索引切声
  （实测两音色谱质心 2415 vs 1994Hz，确为不同音色）。
- 后端自动选择：`htpAvailable()` 探测 /dev/adsprpc-smd 可写性 → HTP0，否则 CPU；
  native 任何异常自动回退 Java（诊断 JSON 的 engine 字段记
  native-htp/native-cpu/java-ort）。

**排查链**（备忘）：backbone stage failed → run-as 复刻 spawn 环境 →
先 "library libllama.so not found"（LD_LIBRARY_PATH 缺失，JNI setenv 修复）→
再 "libcdsprpc.so not found"（app namespace 不含 /vendor/lib64）→
dd /dev/adsprpc-smd Permission denied（DAC 终判）。验证 exec 技巧时注意
`install -r` 会换 nativeLibraryDir 的随机路径，脚本要动态取。

### Server + Web 组合：浏览器拿到 HTP 加速（2026-11-14 傍晚补）

**结论：server + web 是 App 之外拿到 HTP 的正确形态。** HTP 的封禁只针对
app uid 域（/dev/adsprpc-smd EACCES）；m5-server 跑在 adb shell 域，fastrpc
全通。浏览器只是瘦客户端，DSP 加速发生在手机侧——本机浏览器实测
`X-M5-RTF: 0.805`（wav 合成 4.54s 音频 / 3.66s 墙钟）、首包 0.447s。

- `m5_server.cpp` 新增 `GET /`：托管 `<workdir>/web.html`（无则返回内嵌提示页）；
  CORS 此前已是 `Access-Control-Allow-Origin: *` + OPTIONS 预检，
  因此同一页面从 server 托管（同源）或本机 file:// 打开（跨源）都能调 API。
- Web 控制台 `examples/m5-tts/web/index.html`（推到设备为 `web.html`）：
  文本→语音（wav 回放 / **流式 pcm**：fetch ReadableStream + Web Audio
  逐块调度播放，边生成边出声）、音色下拉（/v1/voices 实时刷新）、
  X-M5-* 统计（RTF/首包/帧数）、**克隆导入**（选 .m5voice / codes JSON
  → POST /v1/voices/clone，实测"网页导入"入列即可选）、API 基址可改。
- 测试路径（本机电脑）：
  1. 设备 shell 域启动：`./m5-server model --port 8080 --cwd $D`（HTP）；
  2. `adb forward tcp:18080 tcp:8080` → 浏览器开 `http://127.0.0.1:18080`；
  3. 同 Wi-Fi 也可直连 `http://<device-ip>:8080`（server bind 0.0.0.0），
     任意 LAN 设备的浏览器都能用，手机即"TTS 一体机"。
- 注：在线发布（公网链接）无法回连手机 API——web 与 server 要在同一可达
  网络内；LAN 直连是产品形态，adb forward 是调试形态。

### Web 录音克隆：麦克风 → server 端编码（2026-11-14 晚）

克隆闭环补全：浏览器录音（或选音频文件）→ `POST /v1/voices/encode` →
server 自己跑 codec_encoder.onnx 一次 → codes 入 voices.json → 立即可选。
推理路径不变（仍只读缓存），编码器只在克隆时跑一次。

**server 侧**（`m5_server.cpp`）：

- `POST /v1/voices/encode {"name","reference_text","wav_base64"}`：
  base64 → WAV 解析（PCM 8/16/32 + float32、多声道混单）→ 线性重采样到 24k →
  read_reference（与 clone_voice.py 逐行对齐：峰值归一 + 0.15s 间隙裁剪 + 50ms padding）→
  ORT 跑 encoder（input "audio"[1,T]，output "codes"[1,16,F]，F=ceil(T/1920) 切片，
  值域 0..2047 校验）→ upsert voices.json（与 /clone 共用 upsert_voice_entry）。
- `--voice-encoder PATH`（默认 model_dir/codec_encoder.onnx）；SHA256 校验
  codec_identity.json（不匹配拒绝启动编码），voices 缓存的 encoder_sha256 不一致回 409。
- **codes 一致性已验证**：同一参考 wav 走 encode 端点 vs clone_voice.py，
  102×16 codes **逐位相同**（match rate 1.0），SHA 戳一致。48k 输入重采样路径正常；
  静音 wav → 400 "no speech above the silence threshold"。

**web 侧**（`web/index.html` 新增"录音克隆"区）：

- 录音/停止（30s 自停）/试听；音色名 + **逐字转写**（必填）；或选音频文件走同管线。
- 管线：MediaRecorder → decodeAudioData → OfflineAudioContext 重采样 24k 单声道
  （浏览器原生重采样器，质量好于服务端线性插值）→ 16-bit WAV → base64 → encode 端点。
- 录音约束关闭 echoCancellation/noiseSuppression/AGC（保音色特征）。

**端到端验证**（CDP 驱动 Chrome）：文件路径全流程实测通过（48k tone.wav →
解码→24k→WAV→base64→encode→"克隆成功 38 帧"→下拉出现新音色）。
麦克风路径在 macOS 自动化下无法验证（Chrome 假音频栈 getUserMedia 挂起，
headless/有头/授权/喂 wav 文件均如此；真实浏览器人工点击不受影响，
页面对拒绝有"麦克风不可用"提示）。Mac/Linux 真机浏览器打开即测。

### Web 音色临时化 + 统计修复 + 情感评估（2026-11-14 夜）

**音色策略调整**：服务端 voices.json 只留内置音色（官方样例）；Web 克隆/导入的
音色只存浏览器 localStorage（`m5-voices-v2`），合成时随请求内联 codes+reference_text，
服务端零落盘。

- `Engine::set_voice_codes(reference_text, codes)`：直接应用临时音色（校验 [T][16]、
  值域 0..2047），不碰任何缓存文件。
- `POST /v1/audio/speech` 接受内联 `codes`+`reference_text`（优先于 voice 名）；
  `POST /v1/voices/encode` 改为**纯计算**——返回 codes/encoder_sha256，不再 upsert。
- Web：下拉合并服务端+本机音色（本机带"本机"标记，双击删除）；.m5voice 导入也在
  浏览器本地完成；每请求发送内联 codes。实测 localStorage 克隆→合成链路通，
  服务端音色列表保持只有官方样例。

**首包延迟/帧数不显示的根因与修复**：CORS 下浏览器只暴露简单响应头，自定义
`X-M5-*` 头被过滤（RTF 因有客户端兜底才显示）。修复：`svr.set_default_headers`
加 `Access-Control-Expose-Headers: X-M5-RTF, X-M5-Elapsed-S, X-M5-First-Pcm-S,
X-M5-Frames`。浏览器实测五项统计全部正确显示（RTF/音频/首包/帧数/墙钟）。

**情感表达难度评估**（对照实验：同句同种子 7 档 caption + 4 种子基线）：

| 路径 | 实测证据 | 难度 |
| --- | --- | --- |
| caption 风格槽（`<|caption_begin|>`） | 非空 caption 使时长 +0~42%（超种子噪声），但 F0/能量变化在种子噪声（±8%）内——弱训练槽位，无可靠语义情感控制 | 接线低（已通，标 beta）；效果不可靠 |
| 情感参考音频克隆 | codes 强烈改变输出（F0/能量大幅迁移，同二次元/豆包风的音色迁移）；"悲伤→低沉"方向性在快速实验中不保真 | **低**：录音克隆链路已建成，录对应语气的参考即得；局限=情感绑定音色 |
| DSP 韵律后处理 | 可预测但机械（音高/语速/能量预设） | 低 |
| 训练侧（情感标注 CPT/finetune） | 模型卡明言 delimiter 改动需 CPT；caption 弱训练即因训练侧稀疏 | **高**（数据+算力+训练管线） |

推荐：短期用情感参考克隆（零新工程，可各情绪录一个音色）+ 可选 DSP 预设；
要每句可控情感则必须训练投入。种子对照结论：F0 采样噪声 ±8%，任何 <10% 的
"情感差异"都不可信，验证情感效果必须多种子平均。

**环境注记**：/tmp 被系统清理（模型/ORT AAR/实验产物全失）。恢复路径：
ORT AAR 从 gradle 缓存 `~/.gradle/caches/.../onnxruntime-android-1.23.2`（已拷到
~/m5-assets/ort-aar）；模型可从设备回拉；构建产物落 ~/m5-assets/build。

## 优化批次（2026-11-14 晚）

- **deploy.sh 一键部署**：停服务→推二进制/web→备份模型→固化 env→启动→forward。
  GGML_HEXAGON_OPPOLL=1/NHMX=0 实测 2.2×（ggml-hexagon.cpp），丢失会静默慢一倍，已写死在脚本。
- **--depth-chain 协议**（m5_depth_heads.h + backbone.cpp）：16-book 链整体进 stage 子进程，
  每帧 1 次 pipe 往返（原 16 次）。**与 legacy 逐位一致**（宿主 200 组随机输入 0 mismatch；
  设备 chain==legacy 位一致），但实测性能打平（depth 4.15 vs 4.08s，-1.5%）——
  heads 是权重带宽瓶颈而非往返瓶颈。默认保持 --depth-pipe，chain 作为
  M5_ZERO_TRANSFORM/批量化的落点保留（旧子进程自动探测回退）。
- WavWriter 三模式（文件/内存/丢弃）：wav 响应零磁盘 I/O，pcm 流式不再写临时 wav。
- 按请求 chunk 策略：wav 4/16（codec 调用 18→5 次），pcm 2/4（首包）。
- X-M5 增加 Backbone-S/Depth-S/Codec-S 分项；g_heads_s 双计 bug 已修；
  voices.json mtime 缓存；CORS 头去重；GET / no-store。
- codec ORT 线程 A/B：2/3/4 = 3.56/2.84/3.20s → **3 线程最优**（现默认值）。
- codes 对 18:44 旧二进制有 2-book/帧 级采样差异：已证重构逐位等价、chain==legacy、
  contract=off 不变——判定为该二进制与其源码状态不一致（同分钟编辑+链接竞争）。
- 观测：设备热状态影响显著（同代码 RTF 0.75→0.87→0.91 随测试时长漂移），对比务必同轮 A/B。

## 批量化可行性结论 + heads int8（2026-11-14 夜）

### 15 步批量化：证据链否定（暂搁置）

- 收益模型（depth_batch_bench，HTP0 实测）：decode 成本 = 1.25ms 固定提交 + 0.91ms/行。
  现状 15 调用 16 行 ≈33ms/帧；理想 1 调用 16 行 ≈16ms/帧 → 理论 2.1×（RTF 0.85→~0.55）。
- ❌ 朴素批量（M5_ZERO_TRANSFORM 消融）：零掉 transform 行后 codes 翻转 99.6%
  （首帧 15/15 全翻）、音频 70→500 帧发散。串行依赖是强训练的，dummy rows 不可行。
- ❌ 推测式批量（上一帧 codes 做草稿）：帧间 codes 匹配率仅 1.5%（近随机，
  lag 2/3 同样）——codec codes 无时间局部性，草稿无预测力。
- 复活条件：学习型草稿（小模型预测 code[b]←code[b-1]）或训练侧改动；记录待查。
- HTP 旋钮 sweep（depth-bench）：OPPOLL=1 已是 1.8× 大头（已固化）；
  OPBATCH 16/64 更慢、OPQUEUE 无差、OPFUSION=0 更慢、NHMX=1 无差 → 当前配置已最优。

### heads int8：默认启用（RTF -13%）

- 动机：heads_s 13.4ms/帧 ≈ 纯权重带宽（34MB f16 / 2.5GB/s）。int8 per-row 对称量化
  （m5_depth_heads.h：build_int8/project8，NEON f32 累加）带宽减半。
- A/B（同轮交替 ×2）：heads_s 0.94→0.38s（2.5×）；RTF 0.86→0.75（-13%）。
- 质量门禁：首帧 codes 0/16 翻转（数值正确）；帧 2 起经 next 反馈链混沌分叉
  至 90% 翻转（帧数 70→75）——与换种子同级的轨迹分叉，非误差累积。
  ASR 两句全对（24 字长句 + 疑问句），RMS/ZCR/峰值同族，mode 内确定性 0% 翻。
- 默认 ON；M5_HEADS_INT8=0 回退 f16。legacy（--depth-pipe）与 chain 子进程两路都接。
- 坑：clang 19 arm_neon.h 无 vcvtq_f32_s16（需 s32 中转：vmovl_s16+vcvtq_f32_s32）。

## 无 HMX 的 depth 回读与 FP32 codec 复测

SM6650 / Android shell，HTP0，OPPOLL=1，NHMX=0。实验部署在
`/data/local/tmp/llama-m5-htp-v73/quality-ab`，已有服务继续使用原来的二进制。
宿主原始日志、模型和听感样本位于 `~/m5-assets/htp-quality`。

### Depth 的同步发生在哪里

- `backbone.cpp` 用 `-DM5_STAGE_TIMING` 累计 `llama_decode` 与 embedding
  getter 耗时。长句的前 1900 次 depth 调用在 INT8 codec 并行时分别约 5.08 s
  与 5.3 ms。这是 API 边界耗时，`llama_decode` 包含等待。
- 用单独的 `libggml-hexagon.so`，打开 `M5_HTP_READBACK_TIMING=1`
  与 `GGML_HEXAGON_PROFILE=1`。累计 3900 次 HTP 输出读取，flush
  5.894 s、读取和拷贝 4.15 ms。平均每次 decode 有约两次读取，首个输出
  读取前会提交并等待 DSP，第二个读取多半命中已完成的结果。此计时含 profile
  开销。1820 个 80-op batch 的 DSP 计时中位数为 2.166 ms，不能把整个
  flush 耗时计作 RPC 固定开销。
- `depth_projection_probe --head model/depth-weights.f32 fixed-hidden.f32`
  用真实 book-0 权重、一个冻结 hidden 和 128 组固定噪声测试。Q8_0 HVX
  projection 228.4 us，加 logits 回读 1.2 us 和 CPU 采样 8.4 us。
  GGML 图中接上 top-k、按 log-noise 打分和 argmax 后，只回传 4 字节，
  连同 log-noise 上传共 274.0 us，排除主机生成 log-noise 的时间。
  DSP 采样与同一 Q8_0 logits 在 CPU 采样一致 128/128；现有 per-row INT8
  CPU head 则一致 124/128，包含不同量化格式的影响。此原型未接到
  depth transformer 的图末尾。独立提交不值得采用，下一步可在同一 DSP
  图内验证能否用一个融合 kernel 完成归约并省下一次同步。原型用现有
  GGML/HVX 算子执行，未使用 Hexagon-MLIR 编译。

### Codec 固定 codes 和端到端结果

`codec_bench` 使用设备原有 `codec-codes.bin` 前 68 帧（17×4），CPU 3 线程，
每版 3 次取中位数。音频 5.34 s，用 `compare_pcm.py` 对齐 PCM16 输出。

| Codec | ORT Run 累计 | 相对 FP32 SNR | PCM16 最大绝对差 |
| --- | ---: | ---: | ---: |
| 原始 FP32 | 5.281 s | 参考 | 0 |
| FP32 + Snake | 4.755 s | 94.46 dB | 1 |
| FP32 + Snake + ConvTranspose 分解 | 4.578 s | 94.06 dB | 1 |
| 当前 INT8 + Snake | 2.014 s | 12.05 dB | 13717 |

重写用 `snake_fuse.py` 后接 `decompose_convtranspose.py`。删除改写后
不再引用的 ConvTranspose 原始权重，模型由 563 MB 降至约 457 MB。
固定 codes 测试包含一次设备并行负载重叠，个别样本有显著抖动；
端到端 sweep 在单任务下串行执行。

固定音色 0、种子 1234，文本为「今天的天气很好，我们去公园散步吧。
公园里有很多树，风吹过的时候，还能听见远处的鸟叫。等太阳落山，
我们再一起回家。」共 3 段、127 帧、9.86 s。首块 2 帧，其余 4 帧，
3 个 codec 线程。两轮交替测得：

| Codec | RTF |
| --- | ---: |
| 原始 FP32 | 1.286、1.820 |
| FP32 + Snake | 1.149、1.174 |
| FP32 + Snake + ConvTranspose 分解 | 1.124、1.153 |
| 当前 INT8 + Snake | 0.785、0.788 |

FP32 分解版改为首块 2 / 后续 8 帧，3 线程最快单次 RTF 1.086，
首 PCM 0.816 s。另一个 A/B 用相同的瘦身模型和分块配置，复用同一组
参考 codes 得到的 codec 预热状态，将 RTF 中位数从 1.093 降至 1.070
（最快 1.063），三轮 WAV SHA256 全部一致。独立复测关闭 ORT worker
忙等时没有收益。当前 FP32 这条链路距离实时还需要降低约 0.69 s
墙钟时间（以 RTF 1.070 中位数计）。这组负载主要受 codec 制约。

最终二进制重新构建后复测该配置，RTF 1.061，首 PCM 0.783 s，
输出 SHA256 与缓存关闭的对照相同。

### 试听样本重复复查

上节旧 `codec-codes.bin` 共 88 帧，每隔 43 帧逐帧重复；截取前 68 帧
包含第一遍 43 帧和第二遍头 25 帧。因此上节 WAV 可以比较同一输入上
的波形误差，但不应作为端到端朗读样本。正式合成复核用新文本
「请把蓝色的雨伞放在门口，明天出门时别忘了带上。」和独立的新转储文件。
三种 codec 输出同样的 63 帧和 4.94 秒音频，原始生成 codes 的 SHA256
完全一致，且 63 帧互不相同。本地 Whisper base 三份均未转写出重复短语，
三份也都把「雨伞」写作近音词，单句 ASR 不能区分实际发音误差和识别误差。
RTF 依次为原始 FP32 1.162、优化 FP32 1.052、INT8 0.768。
优化 FP32 相对原始 FP32 的 PCM16 最大绝对差仍为 1，INT8 SNR 为 13.15 dB。

`codec_bench` 现拒绝短输入而不补零；多轮测试按输入名识别 recurrent
state 并在每轮恢复初始形状。新试听源文件在
`~/m5-assets/htp-quality/recheck-*.wav`。线上服务没有切换。

### INT8 移除 depth 占位 logits 的 A/B

SM6650 shell，INT8 Snake codec，首块 2 帧、后续 4 帧，固定同一条 9.86 s
长句和种子。用同一实验 libllama，仅给 depth stage 设置
`M5_SKIP_DUMMY_LOGITS=1` 来跳过单词表输出投影。保留原自回归图、KV cache
和 embedding 回读。基线 4 次、实验 3 次，codes 与 WAV 哈希全部一致。

RTF 中位数 0.776 → 0.769（约 0.8%），整句约省 62 ms，样本区间重叠。
depth stage 调用累计中位数 5.580 → 5.514 s。实验仍需等待 embedding
就绪，省掉的是占位投影及其读取，无法消除 step 间等待。实验补丁、脚本
和日志保存在 `~/m5-assets/htp-quality/dummy-logits*`，默认路径未启用。

### Depth core 与图尾投影同批提交

用现有 llama.cpp backend sampler，在 depth GGUF 加入真实 book 1 的 Q8_0
输出权重。保持 Q4_0 core、FP32 KV cache 和 HTP0。单帧形状为一次
2-row decode 加 14 次 1-row decode，使用冻结合成输入。实验只使用
greedy argmax 来确认 4 字节 token 的设备回传路径，未替换生产采样逻辑。

三轮交替，每轮每配置 30 帧取中位数。原图分别为 34.20/34.44/34.27
ms/frame，图尾采样版为 35.22/35.23/35.20 ms/frame。独立 profile 的
352 个包含 ARGMAX 的 batch 均同时包含第一层与输出投影。Q8_0 输出投影
和 argmax 中位 71 us、6 us。图尾合入同批可行，15 个自回归提交仍在。
约 0.95 ms/frame 的 DSP 侧额外耗时需要与生产 CPU heads 成本对比，
同时计入真实采样、动态切换 codebook 与并行 codec 的干扰。尚无端到端 RTF
或音质结论。原始日志位于 `~/m5-assets/htp-quality/depth-graph*.log`。

### Depth 图尾 top-10 加噪采样

`M5_BENCH_SAMPLED=1 M5_BENCH_NOISY=1` 启用图尾温度 0.9、top-10、
减 log-noise 和 argmax。计时包含主机生成指数噪声、取 log 和上传，
固定 book 1 投影及输入轨迹。两轮每配置 20 帧，基线 34.15/34.22、
加噪采样 37.39/37.42 ms/frame，同轮 greedy 35.10 ms/frame。
Profile 中 118 个 TOP_K_MASK 批次全部包含 core、投影和 ARGMAX。
这是每帧约 3.22 ms 的额外成本门槛；未计入移除 CPU heads 的收益，
也未接入生产 RNG、动态码本头和 codec 并行路径。
日志：`~/m5-assets/htp-quality/depth-noisy-ab.log`。

## backbone Q4_0 结论（2026-11-15 凌晨）

Q4_0 量化（32.4→17.1 MB）实测 backbone 快 21%、总耗时 -0.7 s，但音频短 16%
（语速/韵律漂移），人工试听判定不可接受。**决定保持 Q8_0**（默认命令不变），
Q4 模型仅作为 `--backbone` 显式指定时的实验选项。质量验收口径补充：
ASR 全对 + 指标同族不足以代表听感，TTS 的韵律自然度必须人耳确认。
