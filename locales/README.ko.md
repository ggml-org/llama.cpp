<!--START_SECTION:navbar-->
<div align="center">
  <a href="../README.md">🇺🇸 English</a> | <a href="README.de.md">🇩🇪 Deutsch</a> | <a href="README.es.md">🇪🇸 Español</a> | <a href="README.fr.md">🇫🇷 Français</a> | <a href="README.hi.md">🇮🇳 हिंदी</a> | <a href="README.ja.md">🇯🇵 日本語</a> | <a href="README.ko.md">🇰🇷 한국어</a> | <a href="README.pt.md">🇵🇹 Português</a> | <a href="README.ru.md">🇷🇺 Русский</a> | <a href="README.zh.md">🇨🇳 中文</a>
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

## 핫 토픽

- **[가이드 : llama.cpp의 새 WebUI 사용법](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [가이드 : gpt-oss를 llama.cpp로 실행하기](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[피드백] llama.cpp의 더 나은 패키징으로 downstream consumers 지원 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- `gpt-oss` 모델에 네이티브 MXFP4 형식 지원이 추가됨 | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [NVIDIA와의 협업](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [댓글](https://github.com/ggml-org/llama.cpp/discussions/15095)
- `llama-server`에 멀티모달 지원 도입됨: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [문서](.././docs/multimodal.md)
- FIM 완성용 VS Code 확장: https://github.com/ggml-org/llama.vscode
- FIM 완성용 Vim/Neovim 플러그인: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints가 GGUF를 기본 지원합니다! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF 편집기: [토론](https://github.com/ggml-org/llama.cpp/discussions/9268) | [도구](https://huggingface.co/spaces/CISCai/gguf-editor)

## 빠른 시작

llama.cpp를 사용하는 것은 간단합니다. 다음과 같은 방법으로 컴퓨터에 설치할 수 있습니다:

- [brew, nix 또는 winget](../docs/install.md)을 사용하여 `llama.cpp`를 설치합니다.
- Docker로 실행 - [Docker 문서](../docs/docker.md)를 참조하세요.
- [릴리스 페이지](https://github.com/ggml-org/llama.cpp/releases)에서 사전 빌드된 바이너리를 다운로드합니다.
- 이 저장소를 클로닝하여 소스에서 빌드 - [빌드 가이드](../docs/build.md)를 확인하세요.

설치가 완료되면 사용할 모델이 필요합니다. 자세한 내용은 [모델을 얻고 양자화하는 방법](#obtaining-and-quantizing-models) 섹션을 참조하세요.

예제 명령어:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## 설명

`llama.cpp`의 주요 목표는 넓은 범위의 하드웨어에서 최소한의 설정과 최첨단 성능으로 LLM 추론을 가능하게 하는 것입니다 - 로컬에서 클라우드까지.

- 의존성 없이 순수 C/C++ 구현
- Apple silicon은 일등 시민 - ARM NEON, Accelerate 및 Metal 프레임워크를 통해 최적화됨
- x86 아키텍처를 위한 AVX, AVX2, AVX512 및 AMX 지원
- RISC-V 아키텍처를 위한 RVV, ZVFH, ZFH, ZICBOP 및 ZIHINTPAUSE 지원
- 더 빠른 추론과 메모리 사용 감소를 위한 1.5비트, 2비트, 3비트, 4비트, 5비트, 6비트 및 8비트 정수 양자화
- NVIDIA GPU에서 LLM을 실행하기 위한 커스텀 CUDA 커널 (AMD GPU는 HIP을 통해, Moore Threads GPU는 MUSA를 통해 지원)
- Vulkan 및 SYCL 백엔드 지원
- 전체 VRAM 용량보다 큰 모델을 부분적으로 가속화하기 위한 CPU+GPU 하이브리드 추론

`llama.cpp` 프로젝트는 [ggml](https://github.com/ggml-org/ggml) 라이브러리에 대한 새로운 기능 개발의 주요 실험장입니다.

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

## 지원 가능한 백엔드

| 백엔드 | 대상 장치 |
| --- | --- |
| [Metal](../docs/build.md#metal-build) | Apple Silicon |
| [BLAS](../docs/build.md#blas-build) | All |
| [BLIS](../docs/backend/BLIS.md) | All |
| [SYCL](../docs/backend/SYCL.md) | Intel 및 Nvidia GPU |
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

## 모델을 얻고 정량화하기

[Hugging Face](https://huggingface.co) 플랫폼은 `llama.cpp`와 호환되는 [여러 LLMs](https://huggingface.co/models?library=gguf&sort=trending)을 호스팅합니다:

- [인기](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

GGUF 파일을 수동으로 다운로드하거나, 이 CLI 인수를 사용하여 [Hugging Face](https://huggingface.co/) 또는 [ModelScope](https://modelscope.cn/)와 같은 다른 모델 호스팅 사이트에서 `llama.cpp`와 호환되는 모델을 직접 사용할 수 있습니다: `-hf <user>/<model>[:quant]`. 예를 들어:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

기본적으로 CLI는 Hugging Face에서 모델을 다운로드하지만, 환경 변수 `MODEL_ENDPOINT`를 사용하여 다른 옵션으로 전환할 수 있습니다. 예를 들어, `MODEL_ENDPOINT=https://www.modelscope.cn/`와 같이 설정하여 ModelScope 또는 기타 모델 공유 커뮤니티에서 모델 체크포인트를 다운로드하도록 선택할 수 있습니다.

모델을 다운로드한 후 CLI 도구를 사용하여 로컬에서 실행할 수 있습니다 - 아래를 참조하십시오.

`llama.cpp`은 모델이 [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 파일 형식에 저장되어 있어야 합니다. 다른 데이터 형식의 모델은 이 저장소의 `convert_*.py` 파이썬 스크립트를 사용하여 GGUF로 변환할 수 있습니다.

Hugging Face 플랫폼은 `llama.cpp`와 함께 모델을 변환, 양자화 및 호스팅하는 온라인 도구를 다양하게 제공합니다:

- [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo)를 사용하여 GGUF 형식으로 변환하고 모델 가중치를 더 작은 크기로 양자화합니다.
- [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora)를 사용하여 LoRA 어댑터를 GGUF 형식으로 변환합니다 (더 많은 정보: https://github.com/ggml-org/llama.cpp/discussions/10123)
- [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor)를 사용하여 브라우저에서 GGUF 메타데이터를 편집합니다 (더 많은 정보: https://github.com/ggml-org/llama.cpp/discussions/9268)
- [Inference Endpoints](https://ui.endpoints.huggingface.co/)를 사용하여 `llama.cpp`를 클라우드에서 직접 호스팅합니다 (더 많은 정보: https://github.com/ggml-org/llama.cpp/discussions/9669)

모델 양자화에 대해 더 알아보려면 [이 문서](../tools/quantize/README.md)를 참조하십시오.

## [`llama-cli`](../tools/cli)

#### `llama.cpp`의 대부분의 기능에 접근하고 실험할 수 있는 CLI 도구입니다.


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

#### 가볍고, [OpenAI API](https://github.com/openai/openai-openapi) 호환 가능한 HTTP 서버로, LLM을 제공합니다.


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

#### 주어진 텍스트에 대한 모델의 [perplexity](../tools/perplexity/README.md) [^1] (및 기타 품질 지표)를 측정하는 도구.


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

#### 다양한 파라미터의 추론 성능을 벤치마킹합니다.


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

#### `llama.cpp` 모델을 실행하는 데 사용되는 포괄적인 예제입니다. 추론에 유용합니다. RamaLama [^3]와 함께 사용됩니다.


<details>
    <summary>Run a model with a specific prompt (by default it's pulled from Ollama registry)</summary>

    ```bash
    llama-run granite-code
    ```

    </details>

[^3]: [RamaLama](https://github.com/containers/ramalama)

## [`llama-simple`](../examples/simple)

#### `llama.cpp`를 사용하여 앱을 구현하는 최소한의 예제입니다. 개발자에게 유용합니다.


<details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>

## 기여

- 기여자는 PR을 열 수 있습니다
- 기여에 따라 협업자로 초대됩니다
- 유지 관리자는 `llama.cpp` 저장소의 브랜치에 푸시하고 `master` 브랜치에 PR을 병합할 수 있습니다
- 이슈, PR 및 프로젝트 관리에 도움을 주시면 매우 감사하겠습니다!
- 첫 기여에 적합한 작업은 [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)에서 확인할 수 있습니다
- 더 많은 정보는 [CONTRIBUTING.md](../CONTRIBUTING.md)를 참조하세요
- 다음을 반드시 읽어보세요: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- 관심 있는 분들을 위한 배경 이야기: [Changelog podcast](https://changelog.com/podcast/532)

## 기타 문서

- [cli](../tools/cli/README.md)
- [completion](../tools/completion/README.md)
- [server](../tools/server/README.md)
- [GBNF 문법](../grammars/README.md)

#### 개발 문서

- [빌드 방법](../docs/build.md)
- [Docker에서 실행](../docs/docker.md)
- [Android에서 빌드](../docs/android.md)
- [성능 문제 해결](../docs/development/token_generation_performance_tips.md)
- [GGML 팁 및 기술](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### 기초 논문 및 모델에 대한 배경 정보

모델 생성 품질과 관련된 문제가 있다면, LLaMA 모델의 한계를 이해하기 위해 다음 링크와 논문을 최소한 스캔해 주세요. 이는 적절한 모델 크기를 선택하고, LLaMA 모델과 ChatGPT 사이의 중요한 차이점과 미묘한 차이점을 인식하는 데 특히 중요합니다:
- LLaMA:
    - [LLaMA 소개: 650억 파라미터를 갖는 기초 대형 언어 모델](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: 개방적이고 효율적인 기초 언어 모델](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [언어 모델은 샘플 학습을 수행한다](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [언어 모델을 지시사항을 따르도록 정렬하기](https://openai.com/research/instruction-following)
    - [인간 피드백을 사용하여 지시사항을 따르도록 언어 모델을 훈련시키기](https://arxiv.org/abs/2203.02155)

## XCFramework

XCFramework은 iOS, visionOS, tvOS 및 macOS를 위한 라이브러리의 사전 컴파일된 버전입니다. 소스 코드에서 라이브러리를 컴파일할 필요 없이 Swift 프로젝트에서 사용할 수 있습니다. 예를 들어:

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

위 예제는 라이브러리의 중간 빌드 `b5046`을 사용하고 있습니다. URL과 체크섬을 변경하여 다른 버전을 사용하도록 수정할 수 있습니다.

## 완료

일부 환경에서는 명령줄 완성이 사용 가능합니다.

#### Bash Completion

```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```

선택적으로 이 명령은 `.bashrc` 또는 `.bash_profile`에 추가하여 자동으로 로드할 수 있습니다. 예를 들어:

```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## 의존성

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - 단일 헤더 HTTP 서버, `llama-server`에서 사용 - MIT 라이선스
- [stb-image](https://github.com/nothings/stb) - 단일 헤더 이미지 형식 디코더, 다중 모달 서브시스템에서 사용 - 공공 도메인
- [nlohmann/json](https://github.com/nlohmann/json) - 단일 헤더 JSON 라이브러리, 다양한 도구/예제에서 사용 - MIT 라이선스
- [minja](https://github.com/google/minja) - C++에서 사용하는 최소한의 Jinja 파서, 다양한 도구/예제에서 사용 - MIT 라이선스
- [linenoise.cpp](.././tools/run/linenoise.cpp/linenoise.cpp) - C++ 라이브러리로 readline과 유사한 라인 편집 기능 제공, `llama-run`에서 사용 - BSD 2-Clause 라이선스
- [curl](https://curl.se/) - 클라이언트 측 URL 전송 라이브러리, 다양한 도구/예제에서 사용 - [CURL 라이선스](https://curl.se/docs/copyright.html)
- [miniaudio.h](https://github.com/mackron/miniaudio) - 단일 헤더 오디오 형식 디코더, 다중 모달 서브시스템에서 사용 - 공공 도메인
- [subprocess.h](https://github.com/sheredom/subprocess.h) - C 및 C++에서 사용하는 단일 헤더 프로세스 실행 솔루션 - 공공 도메인

