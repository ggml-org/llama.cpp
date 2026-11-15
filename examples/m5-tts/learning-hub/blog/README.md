# M5-TTS 端侧推理优化系列

这组文章以一台无可用 HMX 路径的 Snapdragon 设备为例，记录从模型移植到实时语音播报的工程决策。每篇可独立阅读，实验表格和代码入口集中在[学习中心](../index.html)与[优化复盘](../../../../docs/backend/snapdragon/m5-tts-optimization-log.md)。

| 篇目 | 要回答的问题 | 学习中心 |
| --- | --- | --- |
| [01 · 把 M5-TTS 拆给 HTP 和 CPU](01-pipeline.md) | 生成一帧音频要跑哪些计算？ | [第 1–3 章](../index.html#ch1) |
| [02 · 给一次语音请求做性能剖析](02-profiling.md) | RTF 变低时，实际响应真的更快了吗？ | [第 4–5 章](../index.html#ch4)、[第 10 章](../index.html#ch10) |
| [03 · KV32、codec 分块和听感回归](03-optimization.md) | 哪些改动能在既定音质约束下上线？ | [第 6–9 章](../index.html#ch6)、[第 11 章](../index.html#ch11) |
| [04 · 用 Qwen3-TTS 的 Serena 做座舱播报](04-serena.md) | 怎样迁移音色，怎样验证中文导航播报？ | [第 13 章](../index.html#ch13) |

发布时将仓库相对链接替换为公开链接。学习中心使用内联 SVG 和交互脚本，引用图表时可从浏览器导出静态图，并配上图注与实验口径。
