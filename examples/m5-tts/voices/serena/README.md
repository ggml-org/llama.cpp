# Serena 座舱音色

| 场景 | 音色 ID | 示例 |
| --- | --- | --- |
| 车控 | `serena` | [已为你打开座椅加热，风量调到二档。](vehicle.wav) |
| 导航 | `serena` | [前方三百米右转，随后靠左行驶。](navigation.wav) |
| 闲聊 | `serena-calm` | [辛苦啦，路上不用着急。等到了，我们再慢慢聊。](chat.wav) |

车控和导航用中性版，闲聊可选较慢的安抚版。示例由现有 INT8 codec
在 SM6650 HTP 上整句生成。三次请求 RTF 分别为 0.850、0.839 和
0.893。这些是单次示例数据。

`voices.json` 存储使用 M5 codec 编码过的参考音色，可作为 CLI 的
`--voices` 输入。Web 服务要合并到现有音色列表时，将每条音色及同文件
的 `encoder_sha256` 一起发送到 `POST /v1/voices/clone`。
