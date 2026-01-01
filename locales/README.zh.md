<!--START_SECTION:navbar-->
<div align="center">
  <a href="../README.md">🇺🇸 English</a> | <a href="README.de.md">🇩🇪 Deutsch</a> | <a href="README.fr.md">🇫🇷 Français</a> | <a href="README.hi.md">🇮🇳 हिंदी</a> | <a href="README.ja.md">🇯🇵 日本語</a> | <a href="README.ko.md">🇰🇷 한국어</a> | <a href="README.pt.md">🇵🇹 Português</a> | <a href="README.ru.md">🇷🇺 Русский</a> | <a href="README.zh.md">🇨🇳 中文</a>
</div>
<!--END_SECTION:navbar-->

# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## 热门话题

- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- 支持 `gpt-oss` 模型的原生 MXFP4 格式已添加 | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [与 NVIDIA 合作](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [评论](https://github.com/ggml-org/llama.cpp/discussions/15095)
- `llama-server` 现在支持多模态 | [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [文档](.././docs/multimodal.md)
- VS Code 扩展用于 FIM 补全：https://github.com/ggml-org/llama.vscode
- Vim/Neovim 插件用于 FIM 补全：https://github.com/ggml-org/llama.vim
- Hugging Face 推理端点现在原生支持 GGUF！https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF 编辑器：[讨论](https://github.com/ggml-org/llama.cpp/discussions/9268) | [工具](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## 快速入门

使用 llama.cpp 上手非常简单。以下是几种在您的机器上安装它的方法：

- 使用 [brew, nix 或 winget](../docs/install.md) 安装 `llama.cpp`
- 使用 Docker 运行 - 请查看我们的 [Docker 文档](../docs/docker.md)
- 从 [发布页面](https://github.com/ggml-org/llama.cpp/releases) 下载预构建的二进制文件
- 通过克隆此仓库从源代码构建 - 请查看 [我们的构建指南](../docs/build.md)

安装完成后，您需要一个模型来进行操作。前往 [获取和量化模型](#obtaining-and-quantizing-models) 部分了解更多信息。

示例命令：

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## 描述

`llama.cpp` 的主要目标是在各种硬件上实现 LLM 推理，本地和云端均可，且设置简单，性能先进。

- 采用纯 C/C++ 实现，无需任何依赖
- Apple 芯片是头等公民 - 通过 ARM NEON、Accelerate 和 Metal 框架进行优化
- 支持 x86 架构的 AVX、AVX2、AVX512 和 AMX
- 支持 RISC-V 架构的 RVV、ZVFH、ZFH、ZICBOP 和 ZIHINTPAUSE
- 支持 1.5 位、2 位、3 位、4 位、5 位、6 位和 8 位整数量化，以加快推理速度并减少内存使用
- 自定义 CUDA 内核，用于在 NVIDIA GPU 上运行 LLM（通过 HIP 支持 AMD GPU，通过 MUSA 支持 Moore Threads GPU）
- 支持 Vulkan 和 SYCL 后端
- CPU+GPU 混合推理，以部分加速超出总 VRAM 容量的模型

`llama.cpp` 项目是为 [ggml](https://github.com/ggml-org/ggml) 库开发新功能的主要实验场。

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](../docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>

## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](../docs/build.md#metal-build) | Apple Silicon |
| [BLAS](../docs/build.md#blas-build) | All |
| [BLIS](../docs/backend/BLIS.md) | All |
| [SYCL](../docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](../docs/build.md#musa) | Moore Threads GPU |
| [CUDA](../docs/build.md#cuda) | Nvidia GPU |
| [HIP](../docs/build.md#hip) | AMD GPU |
| [ZenDNN](../docs/build.md#zendnn) | AMD CPU |
| [Vulkan](../docs/build.md#vulkan) | GPU |
| [CANN](../docs/build.md#cann) | Ascend NPU |
| [OpenCL](../docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](../docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](../docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](../docs/backend/hexagon/README.md) | Snapdragon |

## 获取和量化模型

[Hugging Face](https://huggingface.co) 平台托管了多个与 `llama.cpp` 兼容的 [LLMs](https://huggingface.co/models?library=gguf&sort=trending)：

- [热门](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

您可以手动下载 GGUF 文件，或者通过使用此 CLI 参数直接使用来自 [Hugging Face](https://huggingface.co/) 或其他模型托管站点（如 [ModelScope](https://modelscope.cn/)）的任何与 `llama.cpp` 兼容的模型：`-hf <user>/<model>[:quant]`。例如：

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

默认情况下，CLI 会从 Hugging Face 下载模型，你可以通过环境变量 `MODEL_ENDPOINT` 切换到其他选项。例如，你可以通过设置环境变量从 ModelScope 或其他模型共享社区下载模型检查点，例如 `MODEL_ENDPOINT=https://www.modelscope.cn/`。

下载模型后，使用 CLI 工具在本地运行它 - 请参见下文。

`llama.cpp` 要求模型以 [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 文件格式存储。其他数据格式的模型可以使用此仓库中的 `convert_*.py` Python 脚本转换为 GGUF。

Hugging Face 平台提供了多种在线工具，用于使用 `llama.cpp` 转换、量化和托管模型：

- 使用 [GGUF-my-repo 空间](https://huggingface.co/spaces/ggml-org/gguf-my-repo) 将模型转换为 GGUF 格式并量化模型权重以减小大小
- 使用 [GGUF-my-LoRA 空间](https://huggingface.co/spaces/ggml-org/gguf-my-lora) 将 LoRA 适配器转换为 GGUF 格式（更多信息：https://github.com/ggml-org/llama.cpp/discussions/10123）
- 使用 [GGUF-editor 空间](https://huggingface.co/spaces/CISCai/gguf-editor) 在浏览器中编辑 GGUF 元数据（更多信息：https://github.com/ggml-org/llama.cpp/discussions/9268）
- 使用 [推理端点](https://ui.endpoints.huggingface.co/) 直接在云端托管 `llama.cpp`（更多信息：https://github.com/ggml-org/llama.cpp/discussions/9669）

要了解更多关于模型量化的信息，请 [阅读此文档](../tools/quantize/README.md)

## [`llama-cli`](../tools/cli)

#### 一个用于访问和试验 `llama.cpp` 大部分功能的 CLI 工具。


<details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>


<details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>


<details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](../grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](../grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>

## [`llama-server`](../tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.


<details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>


<details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>


<details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>


<details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>


<details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>


<details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>

## [`llama-perplexity`](../tools/perplexity)

#### 用于衡量模型在给定文本上的 [perplexity](../tools/perplexity/README.md) [^1]（及其他质量指标）的工具。


<details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>


<details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](../tools/llama-bench)

#### 对各种参数的推理性能进行基准测试。


<details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-run`](../tools/run)

#### 运行 `llama.cpp` 模型的全面示例。适用于推理。与 RamaLama [^3] 一起使用。


<details>
    <summary>Run a model with a specific prompt (by default it's pulled from Ollama registry)</summary>

    ```bash
    llama-run granite-code
    ```

    </details>

[^3]: [RamaLama](https://github.com/containers/ramalama)

## [`llama-simple`](../examples/simple)

#### 一个用于使用 `llama.cpp` 实现应用的最小示例。对开发者有用。


<details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>

## 贡献

- 贡献者可以提交 PR
- 协作者将根据贡献情况被邀请
- 维护者可以向 `llama.cpp` 仓库的分支推送代码并合并 PR 到 `master` 分支
- 对于管理问题、PR 和项目方面的任何帮助都非常感激！
- 查看 [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 以获取适合首次贡献的任务
- 阅读 [CONTRIBUTING.md](../CONTRIBUTING.md) 以获取更多信息
- 确保阅读此内容：[Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- 对于感兴趣的人，这里有一些背景故事：[Changelog podcast](https://changelog.com/podcast/532)

## 其他文档

- [cli](../tools/cli/README.md)
- [completion](../tools/completion/README.md)
- [server](../tools/server/README.md)
- [GBNF 语法](../grammars/README.md)

#### 开发文档

- [如何构建](../docs/build.md)
- [在 Docker 上运行](../docs/docker.md)
- [在 Android 上构建](../docs/android.md)
- [性能故障排除](../docs/development/token_generation_performance_tips.md)
- [GGML 技巧与窍门](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### 基础论文和模型背景

如果你的问题与模型生成质量有关，请至少浏览以下链接和论文，以了解LLaMA模型的局限性。这在选择合适的模型大小以及理解LLaMA模型与ChatGPT之间显著和细微的差异时尤为重要：
- LLaMA:
    - [介绍LLaMA：一个基础的、650亿参数的大语言模型](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA：开放且高效的基座语言模型](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [语言模型是少样本学习者](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [对齐语言模型以遵循指令](https://openai.com/research/instruction-following)
    - [通过人类反馈训练语言模型以遵循指令](https://arxiv.org/abs/2203.02155)

## XCFramework

XCFramework 是库的预编译版本，适用于 iOS、visionOS、tvOS 和 macOS。无需从源代码编译库即可在 Swift 项目中使用。例如：

```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
```

上述示例使用了库的中间构建版本 `b5046`。可以通过更改 URL 和校验和来使用不同版本。

## 补全

部分环境支持命令行补全。

#### Bash Completion

```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```

可选地，可以将其添加到您的 `.bashrc` 或 `.bash_profile` 中，以便自动加载。例如：

```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## 依赖项

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - 单文件 HTTP 服务器，由 `llama-server` 使用 - MIT 许可证
- [stb-image](https://github.com/nothings/stb) - 单文件图像格式解码器，由多模态子系统使用 - 公共领域
- [nlohmann/json](https://github.com/nlohmann/json) - 单文件 JSON 库，由各种工具/示例使用 - MIT 许可证
- [minja](https://github.com/google/minja) - C++ 中的最小 Jinja 解析器，由各种工具/示例使用 - MIT 许可证
- [linenoise.cpp](.././tools/run/linenoise.cpp/linenoise.cpp) - 提供类似 readline 的行编辑功能的 C++ 库，由 `llama-run` 使用 - BSD 2-Clause 许可证
- [curl](https://curl.se/) - 客户端 URL 转移库，由各种工具/示例使用 - [CURL 许可证](https://curl.se/docs/copyright.html)
- [miniaudio.h](https://github.com/mackron/miniaudio) - 单文件音频格式解码器，由多模态子系统使用 - 公共领域
- [subprocess.h](https://github.com/sheredom/subprocess.h) - C 和 C++ 的单文件进程启动解决方案 - 公共领域

