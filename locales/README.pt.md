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

## Alterações recentes na API

- [Changelog para a API do `libllama`](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog para a API REST do `llama-server`](https://github.com/ggml-org/llama.cpp/issues/9291)

## Assuntos quentes

- **[guia : usando a nova WebUI do llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guia : executando gpt-oss com llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Melhor empacotamento para llama.cpp para suportar consumidores downstream 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Suporte para o modelo `gpt-oss` com formato MXFP4 nativo foi adicionado | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Colaboração com NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comentário](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Suporte multimodal chegou ao `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentação](.././docs/multimodal.md)
- Extensão do VS Code para completions FIM: https://github.com/ggml-org/llama.vscode
- Plugin para Vim/Neovim para completions FIM: https://github.com/ggml-org/llama.vim
- Pontos de inferência da Hugging Face agora suportam GGUF de fábrica! https://github.com/ggml-org/llama.cpp/discussions/9669
- Editor GGUF da Hugging Face: [discussão](https://github.com/ggml-org/llama.cpp/discussions/9268) | [ferramenta](https://huggingface.co/spaces/CISCai/gguf-editor)

## Início rápido

Começar com o llama.cpp é direto. Aqui estão várias formas de instalá-lo em sua máquina:

- Instale o `llama.cpp` usando [brew, nix ou winget](../docs/install.md)
- Execute com Docker - veja nossa [documentação do Docker](../docs/docker.md)
- Baixe binários pré-construídos da [página de releases](https://github.com/ggml-org/llama.cpp/releases)
- Construa a partir da fonte clonando este repositório - confira [nosso guia de compilação](../docs/build.md)

Após a instalação, você precisará de um modelo para trabalhar. Vá para a seção [Obtendo e quantizando modelos](#obtaining-and-quantizing-models) para aprender mais.

Comando de exemplo:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Descrição

O objetivo principal do `llama.cpp` é permitir a inferência de LLM com mínima configuração e desempenho de ponta em uma ampla gama de hardware - localmente e no cloud.

- Implementação em C/C++ pura sem quaisquer dependências
- O Apple silicon é um cidadão de primeira classe - otimizado via ARM NEON, Accelerate e Metal frameworks
- Suporte a AVX, AVX2, AVX512 e AMX para arquiteturas x86
- Suporte a RVV, ZVFH, ZFH, ZICBOP e ZIHINTPAUSE para arquiteturas RISC-V
- Quantização em 1,5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit e 8-bit de inteiros para inferência mais rápida e uso reduzido de memória
- Núcleos personalizados CUDA para executar LLMs em GPUs NVIDIA (suporte a GPUs AMD via HIP e GPUs Moore Threads via MUSA)
- Suporte a backends Vulkan e SYCL
- Inferência híbrida CPU+GPU para acelerar parcialmente modelos maiores que a capacidade total de VRAM

O projeto `llama.cpp` é o principal playground para desenvolver novas funcionalidades para a biblioteca [ggml](https://github.com/ggml-org/ggml).

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

## Backends suportados

| Backend | Dispositivos de destino |
| --- | --- |
| [Metal](../docs/build.md#metal-build) | Apple Silicon |
| [BLAS](../docs/build.md#blas-build) | Todos |
| [BLIS](../docs/backend/BLIS.md) | Todos |
| [SYCL](../docs/backend/SYCL.md) | Intel e Nvidia GPU |
| [MUSA](../docs/build.md#musa) | GPU da Moore Threads |
| [CUDA](../docs/build.md#cuda) | GPU da Nvidia |
| [HIP](../docs/build.md#hip) | GPU da AMD |
| [ZenDNN](../docs/build.md#zendnn) | CPU da AMD |
| [Vulkan](../docs/build.md#vulkan) | GPU |
| [CANN](../docs/build.md#cann) | Ascend NPU |
| [OpenCL](../docs/backend/OPENCL.md) | GPU Adreno |
| [IBM zDNN](../docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](../docs/build.md#webgpu) | Todos |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | Todos |
| [Hexagon [In Progress]](../docs/backend/hexagon/README.md) | Snapdragon |

## Obter e quantizar modelos

A plataforma [Hugging Face](https://huggingface.co) hospeda um [número de LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatíveis com `llama.cpp`:

- [Em alta](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

Você pode baixar manualmente o arquivo GGUF ou usar diretamente qualquer modelo compatível com `llama.cpp` de [Hugging Face](https://huggingface.co/) ou outros sites de hospedagem de modelos, como [ModelScope](https://modelscope.cn/), usando esse argumento da CLI: `-hf <user>/<model>[:quant]`. Por exemplo:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

Por padrão, o CLI baixaria do Hugging Face, você pode alternar para outras opções com a variável de ambiente `MODEL_ENDPOINT`. Por exemplo, você pode optar por baixar os checkpoints do modelo de ModelScope ou outras comunidades de compartilhamento de modelos definindo a variável de ambiente, por exemplo, `MODEL_ENDPOINT=https://www.modelscope.cn/`.

Após baixar um modelo, use as ferramentas do CLI para executá-lo localmente - veja abaixo.

O `llama.cpp` requer que o modelo esteja armazenado no formato de arquivo [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Modelos em outros formatos de dados podem ser convertidos para GGUF usando os scripts Python `convert_*.py` neste repositório.

A plataforma Hugging Face fornece uma variedade de ferramentas online para converter, quantizar e hospedar modelos com `llama.cpp`:

- Use o [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) para converter para o formato GGUF e quantizar os pesos do modelo para tamanhos menores
- Use o [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) para converter adaptadores LoRA para o formato GGUF (mais informações: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use o [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) para editar os metadados GGUF no navegador (mais informações: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use os [Endpoints de Inferência](https://ui.endpoints.huggingface.co/) para hospedar diretamente o `llama.cpp` na nuvem (mais informações: https://github.com/ggml-org/llama.cpp/discussions/9669)

Para aprender mais sobre a quantização de modelos, [leia esta documentação](../tools/quantize/README.md)

## [`llama-cli`](../tools/cli)

#### Uma ferramenta CLI para acessar e experimentar com a maioria das funcionalidades do `llama.cpp`.


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

#### Um servidor leve, compatível com a [OpenAI API](https://github.com/openai/openai-openapi), HTTP para servir LLMs.


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

#### Uma ferramenta para medir a [perplexidade](../tools/perplexity/README.md) [^1] (e outras métricas de qualidade) de um modelo sobre um dado texto.


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

#### Benchmark the performance of the inference for various parameters.


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

#### Um exemplo abrangente para executar modelos `llama.cpp`. Útil para inferência. Usado com RamaLama [^3].


<details>
    <summary>Run a model with a specific prompt (by default it's pulled from Ollama registry)</summary>

    ```bash
    llama-run granite-code
    ```

    </details>

[^3]: [RamaLama](https://github.com/containers/ramalama)

## [`llama-simple`](../examples/simple)

#### Um exemplo mínimo para implementar aplicativos com `llama.cpp`. Útil para desenvolvedores.


<details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>

## Contribuindo

- Contribuidores podem abrir PRs
- Colaboradores serão convidados com base nas contribuições
- Mantenedores podem enviar commits para branches no repositório `llama.cpp` e mesclar PRs na branch `master`
- Qualquer ajuda com o gerenciamento de issues, PRs e projetos é muito apreciada!
- Veja [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) para tarefas adequadas para primeiras contribuições
- Leia o [CONTRIBUTING.md](../CONTRIBUTING.md) para mais informações
- Certifique-se de ler isso: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- Um pouco de história para aqueles que estiverem interessados: [Changelog podcast](https://changelog.com/podcast/532)

## Outras documentações

- [cli](../tools/cli/README.md)
- [completação](../tools/completion/README.md)
- [servidor](../tools/server/README.md)
- [gramáticas GBNF](../grammars/README.md)

#### Documentação de desenvolvimento

- [Como compilar](../docs/build.md)
- [Executando no Docker](../docs/docker.md)
- [Compilação no Android](../docs/android.md)
- [Dicas de solução de problemas de desempenho](../docs/development/token_generation_performance_tips.md)
- [Dicas e truques do GGML](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Papers fundamentais e contexto sobre os modelos

Se seu problema está relacionado à qualidade de geração do modelo, então por favor, pelo menos, examine os seguintes links e papers para entender as limitações dos modelos LLaMA. Isso é especialmente importante ao escolher um tamanho de modelo apropriado e apreciar tanto as diferenças significativas quanto as sutis entre os modelos LLaMA e ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework

O XCFramework é uma versão pré-compilada da biblioteca para iOS, visionOS, tvOS
e macOS. Ele pode ser usado em projetos Swift sem a necessidade de compilar a
biblioteca a partir da fonte. Por exemplo:

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

O exemplo acima está usando uma versão intermediária de build `b5046` da biblioteca. Isso pode ser modificado para usar uma versão diferente alterando a URL e o checksum.

## Completions

Complementação de linha de comando está disponível para alguns ambientes.

#### Complementação do Bash

```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```

Opcionalmente, isso pode ser adicionado ao seu `.bashrc` ou `.bash_profile` para carregá-lo automaticamente. Por exemplo:

```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependências

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Servidor HTTP de cabeçalho único, usado por `llama-server` - Licença MIT
- [stb-image](https://github.com/nothings/stb) - Decodificador de formato de imagem de cabeçalho único, usado pelo subsistema multimodal - Domínio público
- [nlohmann/json](https://github.com/nlohmann/json) - Biblioteca JSON de cabeçalho único, usada por várias ferramentas/exemplos - Licença MIT
- [minja](https://github.com/google/minja) - Parser minimalista de Jinja em C++, usado por várias ferramentas/exemplos - Licença MIT
- [linenoise.cpp](.././tools/run/linenoise.cpp/linenoise.cpp) - Biblioteca C++ que fornece capacidades de edição de linha semelhantes ao readline, usada por `llama-run` - Licença BSD 2-Clause
- [curl](https://curl.se/) - Biblioteca cliente para transferência de URL, usada por várias ferramentas/exemplos - [CURL License](https://curl.se/docs/copyright.html)
- [miniaudio.h](https://github.com/mackron/miniaudio) - Decodificador de formato de áudio de cabeçalho único, usado pelo subsistema multimodal - Domínio público
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Solução de cabeçalho único para lançamento de processos em C e C++ - Domínio público

