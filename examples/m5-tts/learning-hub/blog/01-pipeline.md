# M5-TTS 端侧推理实战（一）：把模型拆给 HTP 和 CPU

M5-TTS 的 backbone 每生成一帧就交出一个 hidden state。depth 先产生首个码本索引，再根据前一步的索引继续算剩余码本。一帧的 16 个索引交给 codec，codec 根据连续帧还原波形。24 kHz 下，每帧对应 1920 个样本，也就是 80 ms 音频。

## GGUF 与 ONNX 各负责什么

| 部件 | 运行方式 | 数据交接 |
| --- | --- | --- |
| Backbone、depth core | llama.cpp 读取 GGUF，主要计算交给 Hexagon HTP | 主机侧组织输入并提交，depth 各步依赖上一索引 |
| 音频 heads | CPU 上执行投影和采样 | hidden 转成码本索引 |
| Codec decoder | ONNX Runtime CPU | 码本索引分块转成 PCM |

移植时先在短输入上对照上游实现，验证输入位置、KV 缓存和逐步码本输出，再测全句。直接做整句对照容易把第一步的输入错误和后续自回归差异混在一起。

## 为什么 codec 还在 CPU

这台设备跑通的是无 HMX 条件下的 HTP 路径。一个 codec 卷积探针在 ORT CPU 上约 11 ms，在两种 HTP 实现上约 46 ms 和 80 ms。这个结果让该子图留在 CPU。迁移其他子图时，还需要测真实形状、边界搬运和对主流水线的影响。

[学习中心第 1 章](../index.html#ch1) · [移植笔记](../../../../docs/backend/snapdragon/m5-tts-port-notes.md)
