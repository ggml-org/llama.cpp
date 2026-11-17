# llama.cpp

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>基于 C/C++ 的大语言模型（LLM）推理引擎</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [算子文档 (ops)](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [维护者 PR 列表](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [开发统计 (dev stats)](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## 项目简介

`llama.cpp` 的核心目标是以极简的配置，在各类硬件设备（无论是本地端侧还是云端）上实现大语言模型（LLM）与多模态大模型（MLLM）的高性能推理。

`llama.cpp` 项目构建于 [ggml](https://github.com/ggml-org/ggml) 张量计算库之上。

## 快速上手

在本地设备上安装 `llama.cpp` 的常用方式：

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- 访问 https://llama.app 并遵循页面指引
- 使用 Docker 运行 - 详见 [Docker 文档](docs/docker.md)
- 从 [Releases 发布页面](https://github.com/ggml-org/llama.cpp/releases)下载预编译二进制文件
- 克隆本仓库并通过源码构建 - 详见[源码构建指南](docs/build.md)

安装完成后：

```sh
# 直接从 Hugging Face 下载并运行模型
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# 启动兼容 OpenAI 接口规范的 API 服务端
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="使用 `llama cli` 进行多模态视觉模型（VLM）会话" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>使用 <b>llama cli</b> 进行多模态视觉模型（VLM）会话</i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="内置网页应用 Llama Chat 连接运行 Qwen 3.6 的 `llama serve`" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>内置网页应用 Llama Chat 连接 <b>llama serve</b> 交互</i>
        </td>
    </tr>
<table>

## 文档导航

#### 工具套件

- [cli 命令行工具](tools/cli/README.md)
- [completion 文本补全工具](tools/completion/README.md)
- [server 推理服务端](tools/server/README.md)
- [文本转语音 (TTS) 模型](docs/multimodal/tts.md)
- [GBNF 约束文法](grammars/README.md)

#### 开发与进阶

- [源码构建指南](docs/build.md)
- [在 Docker 中运行](docs/docker.md)
- [在 Android 上构建](docs/android.md)
- [多 GPU 并行使用指南](docs/multi-gpu.md)
- [推理性能调优技巧](docs/development/token_generation_performance_tips.md)
- [GGML 技巧与实用指南](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework 框架构建](docs/xcframework.md)
- [终端补全配置](docs/completions.md)
- [支持的模型架构](docs/models.md)
- [版本发布流程](docs/release.md)

## 参与贡献

- 欢迎贡献者提交 PR
- 项目协作者（Collaborators）将根据日常贡献情况受邀加入
- 维护者（Maintainers）拥有向 `llama.cpp` 仓库分支推送并合并 PR 至 `master` 分支的权限
- 非常感谢社区在管理 Issue、PR 及推进项目方面提供的任何帮助！
- 更多信息请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)

## 致谢

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - `llama-server` 使用的单头文件 HTTP 服务端库 - MIT 许可证
- [nothings/stb](https://github.com/nothings/stb) - 多模态子系统使用的单头文件图像格式解码器 - 公共领域 (Public domain)
- [nlohmann/json](https://github.com/nlohmann/json) - 各类工具与示例使用的单头文件 JSON 库 - MIT 许可证
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - 多模态子系统使用的单头文件音频格式解码器 - 公共领域 (Public domain)
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - C 与 C++ 单头文件进程启动库 - 公共领域 (Public domain)

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区开发者维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年11月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
