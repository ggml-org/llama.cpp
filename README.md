# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Server](https://github.com/ggerganov/llama.cpp/actions/workflows/server.yml/badge.svg?branch=master&event=schedule)](https://github.com/ggerganov/llama.cpp/actions/workflows/server.yml)
[![Conan Center](https://shields.io/conan/v/llama-cpp)](https://conan.io/center/llama-cpp)

[Roadmap](https://github.com/users/ggerganov/projects/7) / [Project status](https://github.com/ggerganov/llama.cpp/discussions/3471) / [Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205) / [ggml](https://github.com/ggerganov/ggml)

Inference of Meta's [LLaMA](https://arxiv.org/abs/2302.13971) model (and others) in pure C/C++

> [!IMPORTANT]
[2024 Jun 12] Binaries have been renamed w/ a `llama-` prefix. `main` is now `llama-cli`, `server` is `llama-server`, etc (https://github.com/ggerganov/llama.cpp/pull/7809)

### Recent API changes

- [2024 Jun 26] The source code and CMake build scripts have been restructured https://github.com/ggerganov/llama.cpp/pull/8006
- [2024 Apr 21] `llama_token_to_piece` can now optionally render special tokens https://github.com/ggerganov/llama.cpp/pull/6807
- [2024 Apr 4] State and session file functions reorganized under `llama_state_*` https://github.com/ggerganov/llama.cpp/pull/6341
- [2024 Mar 26] Logits and embeddings API updated for compactness https://github.com/ggerganov/llama.cpp/pull/6122
- [2024 Mar 13] Add `llama_synchronize()` + `llama_context_params.n_ubatch` https://github.com/ggerganov/llama.cpp/pull/6017
- [2024 Mar 8] `llama_kv_cache_seq_rm()` returns a `bool` instead of `void`, and new `llama_n_seq_max()` returns the upper limit of acceptable `seq_id` in batches (relevant when dealing with multiple sequences) https://github.com/ggerganov/llama.cpp/pull/5328
- [2024 Mar 4] Embeddings API updated https://github.com/ggerganov/llama.cpp/pull/5796
- [2024 Mar 3] `struct llama_context_params` https://github.com/ggerganov/llama.cpp/pull/5849

### Hot topics

- **`convert.py` has been deprecated and moved to `examples/convert-legacy-llama.py`, please use `convert-hf-to-gguf.py`** https://github.com/ggerganov/llama.cpp/pull/7430
- Initial Flash-Attention support: https://github.com/ggerganov/llama.cpp/pull/5021
- BPE pre-tokenization support has been added: https://github.com/ggerganov/llama.cpp/pull/6920
- MoE memory layout has been updated - reconvert models for `mmap` support and regenerate `imatrix` https://github.com/ggerganov/llama.cpp/pull/6387
- Model sharding instructions using `gguf-split` https://github.com/ggerganov/llama.cpp/discussions/6404
- Fix major bug in Metal batched inference https://github.com/ggerganov/llama.cpp/pull/6225
- Multi-GPU pipeline parallelism support https://github.com/ggerganov/llama.cpp/pull/6017
- Looking for contributions to add Deepseek support: https://github.com/ggerganov/llama.cpp/issues/5981
- Quantization blind testing: https://github.com/ggerganov/llama.cpp/discussions/5962
- Initial Mamba support has been added: https://github.com/ggerganov/llama.cpp/pull/5328

----

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#get-the-code">Get the Code</a></li>
        <li><a href="#build">Build</a></li>
        <li><a href="#blas-build">BLAS Build</a></li>
        <li><a href="#prepare-and-quantize">Prepare and Quantize</a></li>
        <li><a href="#run-the-quantized-model">Run the quantized model</a></li>
        <li><a href="#memorydisk-requirements">Memory/Disk Requirements</a></li>
        <li><a href="#quantization">Quantization</a></li>
        <li><a href="#interactive-mode">Interactive mode</a></li>
        <li><a href="#constrained-output-with-grammars">Constrained output with grammars</a></li>
        <li><a href="#obtaining-and-using-the-facebook-llama-2-model">Obtaining and using the Facebook LLaMA 2 model</a></li>
        <li><a href="#seminal-papers-and-background-on-the-models">Seminal papers and background on the models</a></li>
        <li><a href="#perplexity-measuring-model-quality">Perplexity (measuring model quality)</a></li>
        <li><a href="#android">Android</a></li>
        <li><a href="#docker">Docker</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#coding-guidelines">Coding guidelines</a></li>
    <li><a href="#docs">Docs</a></li>
  </ol>
</details>

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
variety of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2 and AVX512 support for x86 architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

Since its [inception](https://github.com/ggerganov/llama.cpp/issues/33#issuecomment-1465108022), the project has
improved significantly thanks to many contributions. It is the main playground for developing new features for the
[ggml](https://github.com/ggerganov/ggml) library.

**Supported platforms:**

- [X] Mac OS
- [X] Linux
- [X] Windows (via CMake)
- [X] Docker
- [X] FreeBSD

**Supported models:**

Typically finetunes of the base models below are supported as well.

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggerganov/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggerganov/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggerganov/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggerganov/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggerganov/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggerganov/llama.cpp/pull/5118)
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
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)

(instructions for supporting more models: [HOWTO-add-model.md](./docs/HOWTO-add-model.md))

**Multimodal models:**

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)

**HTTP server**

[llama.cpp web server](./examples/server) is a lightweight [OpenAI API](https://github.com/openai/openai-openapi) compatible HTTP server that can be used to serve local models and easily connect them to existing clients.

[simplechat](./examples/server/public_simplechat) is a simple chat client, which can be used to chat with the model exposed using above web server (use --path to point to simplechat), from a local web browser.

**Bindings:**

- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggerganov/llama.cpp/pull/6326)

**UI:**

Unless otherwise noted these projects are open-source with permissive licensing:

- [iohub/collama](https://github.com/iohub/coLLaMA)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [nat/openplayground](https://github.com/nat/openplayground)
- [Faraday](https://faraday.dev/) (proprietary)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [Layla](https://play.google.com/store/apps/details?id=com.laylalite) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all)
- [ollama/ollama](https://github.com/ollama/ollama)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [RAGNA Desktop](https://ragna.app/) (proprietary)
- [RecurseChat](https://recurse.chat/) (proprietary)
- [semperai/amica](https://github.com/semperai/amica)
- [withcatai/catai](https://github.com/withcatai/catai)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Msty](https://msty.app) (proprietary)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file)(Apachev2.0 or later)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [MindMac](https://mindmac.app) (proprietary)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [AIKit](https://github.com/sozercan/aikit) (MIT)
- [LARS - The LLM & Advanced Referencing Solution](https://github.com/abgulati/LARS) (AGPL)

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

**Tools:**

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption

**Infrastructure:**

- [Paddler](https://github.com/distantmagic/paddler) - Stateful load balancer custom-tailored for llama.cpp

---

Here is a typical run using LLaMA v2 13B on M2 Ultra:

```
$ make -j && ./llama-cli -m models/llama-13b-v2/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e
I llama.cpp build info:
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.            -O3 -std=c11   -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -pthread -DGGML_USE_K_QUANTS -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./common -O3 -std=c++11 -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -pthread -DGGML_USE_K_QUANTS
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.3 (clang-1403.0.22.14.1)
I CXX:      Apple clang version 14.0.3 (clang-1403.0.22.14.1)

make: Nothing to be done for `default'.
main: build = 1041 (cf658ad)
main: seed  = 1692823051
llama_model_loader: loaded meta data with 16 key-value pairs and 363 tensors from models/llama-13b-v2/ggml-model-q4_0.gguf (version GGUF V1 (latest))
llama_model_loader: - type  f32:   81 tensors
llama_model_loader: - type q4_0:  281 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_print_meta: format         = GGUF V1 (latest)
llm_load_print_meta: arch           = llama
llm_load_print_meta: vocab type     = SPM
llm_load_print_meta: n_vocab        = 32000
llm_load_print_meta: n_merges       = 0
llm_load_print_meta: n_ctx_train    = 4096
llm_load_print_meta: n_ctx          = 512
llm_load_print_meta: n_embd         = 5120
llm_load_print_meta: n_head         = 40
llm_load_print_meta: n_head_kv      = 40
llm_load_print_meta: n_layer        = 40
llm_load_print_meta: n_rot          = 128
llm_load_print_meta: n_gqa          = 1
llm_load_print_meta: f_norm_eps     = 1.0e-05
llm_load_print_meta: f_norm_rms_eps = 1.0e-05
llm_load_print_meta: n_ff           = 13824
llm_load_print_meta: freq_base      = 10000.0
llm_load_print_meta: freq_scale     = 1
llm_load_print_meta: model type     = 13B
llm_load_print_meta: model ftype    = mostly Q4_0
llm_load_print_meta: model size     = 13.02 B
llm_load_print_meta: general.name   = LLaMA v2
llm_load_print_meta: BOS token = 1 '<s>'
llm_load_print_meta: EOS token = 2 '</s>'
llm_load_print_meta: UNK token = 0 '<unk>'
llm_load_print_meta: LF token  = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.11 MB
llm_load_tensors: mem required  = 7024.01 MB (+  400.00 MB per state)
...................................................................................................
llama_new_context_with_model: kv self size  =  400.00 MB
llama_new_context_with_model: compute buffer total size =   75.41 MB

system_info: n_threads = 16 / 24 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | VSX = 0 |
sampling: repeat_last_n = 64, repeat_penalty = 1.100000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
generate: n_ctx = 512, n_batch = 512, n_predict = 400, n_keep = 0


 Building a website can be done in 10 simple steps:
Step 1: Find the right website platform.
Step 2: Choose your domain name and hosting plan.
Step 3: Design your website layout.
Step 4: Write your website content and add images.
Step 5: Install security features to protect your site from hackers or spammers
Step 6: Test your website on multiple browsers, mobile devices, operating systems etc…
Step 7: Test it again with people who are not related to you personally – friends or family members will work just fine!
Step 8: Start marketing and promoting the website via social media channels or paid ads
Step 9: Analyze how many visitors have come to your site so far, what type of people visit more often than others (e.g., men vs women) etc…
Step 10: Continue to improve upon all aspects mentioned above by following trends in web design and staying up-to-date on new technologies that can enhance user experience even further!
How does a Website Work?
A website works by having pages, which are made of HTML code. This code tells your computer how to display the content on each page you visit – whether it’s an image or text file (like PDFs). In order for someone else’s browser not only be able but also want those same results when accessing any given URL; some additional steps need taken by way of programming scripts that will add functionality such as making links clickable!
The most common type is called static HTML pages because they remain unchanged over time unless modified manually (either through editing files directly or using an interface such as WordPress). They are usually served up via HTTP protocols – this means anyone can access them without having any special privileges like being part of a group who is allowed into restricted areas online; however, there may still exist some limitations depending upon where one lives geographically speaking.
How to
llama_print_timings:        load time =   576.45 ms
llama_print_timings:      sample time =   283.10 ms /   400 runs   (    0.71 ms per token,  1412.91 tokens per second)
llama_print_timings: prompt eval time =   599.83 ms /    19 tokens (   31.57 ms per token,    31.68 tokens per second)
llama_print_timings:        eval time = 24513.59 ms /   399 runs   (   61.44 ms per token,    16.28 tokens per second)
llama_print_timings:       total time = 25431.49 ms
```

And here is another demo of running both LLaMA-7B and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) on a single M1 Pro MacBook:

https://user-images.githubusercontent.com/1991296/224442907-7693d4be-acaa-4e01-8b4f-add84093ffff.mp4

## Usage

Here are the end-to-end binary build and model conversion steps for most supported models.

### Get the Code

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Build

In order to build llama.cpp you have four different options.

- Using `make`:
  - On Linux or MacOS:

      ```bash
      make
      ```

  - On Windows:

    1. Download the latest fortran version of [w64devkit](https://github.com/skeeto/w64devkit/releases).
    2. Extract `w64devkit` on your pc.
    3. Run `w64devkit.exe`.
    4. Use the `cd` command to reach the `llama.cpp` folder.
    5. From here you can run:
        ```bash
        make
        ```

  - Notes:
    - For faster compilation, add the `-j` argument to run multiple jobs in parallel. For example, `make -j 8` will run 8 jobs in parallel.
    - For faster repeated compilation, install [ccache](https://ccache.dev/).
    - For debug builds, run `make LLAMA_DEBUG=1`

- Using `CMake`:

  ```bash
  cmake -B build
  cmake --build build --config Release
  ```

  **Notes**:

    - For faster compilation, add the `-j` argument to run multiple jobs in parallel. For example, `cmake --build build --config Release -j 8` will run 8 jobs in parallel.
    - For faster repeated compilation, install [ccache](https://ccache.dev/).
    - For debug builds, there are two cases:

      1. Single-config generators (e.g. default = `Unix Makefiles`; note that they just ignore the `--config` flag):

      ```bash
      cmake -B build -DCMAKE_BUILD_TYPE=Debug
      cmake --build build
      ```

      2. Multi-config generators (`-G` param set to Visual Studio, XCode...):

      ```bash
      cmake -B build -G "Xcode"
      cmake --build build --config Debug
      ```

-   Using `gmake` (FreeBSD):

    1. Install and activate [DRM in FreeBSD](https://wiki.freebsd.org/Graphics)
    2. Add your user to **video** group
    3. Install compilation dependencies.

        ```bash
        sudo pkg install gmake automake autoconf pkgconf llvm15 openblas

        gmake CC=/usr/local/bin/clang15 CXX=/usr/local/bin/clang++15 -j4
        ```

### Homebrew

On Mac and Linux, the homebrew package manager can be used via
```
brew install llama.cpp
```
The formula is automatically updated with new `llama.cpp` releases. More info: https://github.com/ggerganov/llama.cpp/discussions/7668

### Nix

On Mac and Linux, the Nix package manager can be used via
```
nix profile install nixpkgs#llama-cpp
```
For flake enabled installs.

Or
```
nix-env --file '<nixpkgs>' --install --attr llama-cpp
```
For non-flake enabled installs.

This expression is automatically updated within the [nixpkgs repo](https://github.com/NixOS/nixpkgs/blob/nixos-24.05/pkgs/by-name/ll/llama-cpp/package.nix#L164).

#### Flox

On Mac and Linux, Flox can be used to install llama.cpp within a Flox environment via
```
flox install llama-cpp
```
Flox follows the nixpkgs build of llama.cpp.

### Metal Build

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU.
To disable the Metal build at compile time use the `GGML_NO_METAL=1` flag or the `GGML_METAL=OFF` cmake option.

When built with Metal support, you can explicitly disable GPU inference with the `--n-gpu-layers|-ngl 0` command-line
argument.

### BLAS Build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Support with CPU-only BLAS implementations doesn't affect the normal generation performance. We may see generation performance improvements with GPU-involved BLAS implementations, e.g. cuBLAS, hipBLAS. There are currently several different BLAS implementations available for build and use:

- #### Accelerate Framework:

  This is only available on Mac PCs and it's enabled by default. You can just build using the normal instructions.

- #### OpenBLAS:

  This provides BLAS acceleration using only the CPU. Make sure to have OpenBLAS installed on your machine.

  - Using `make`:
    - On Linux:
      ```bash
      make GGML_OPENBLAS=1
      ```

    - On Windows:

      1. Download the latest fortran version of [w64devkit](https://github.com/skeeto/w64devkit/releases).
      2. Download the latest version of [OpenBLAS for Windows](https://github.com/xianyi/OpenBLAS/releases).
      3. Extract `w64devkit` on your pc.
      4. From the OpenBLAS zip that you just downloaded copy `libopenblas.a`, located inside the `lib` folder, inside `w64devkit\x86_64-w64-mingw32\lib`.
      5. From the same OpenBLAS zip copy the content of the `include` folder inside `w64devkit\x86_64-w64-mingw32\include`.
      6. Run `w64devkit.exe`.
      7. Use the `cd` command to reach the `llama.cpp` folder.
      8. From here you can run:

          ```bash
          make GGML_OPENBLAS=1
          ```

  - Using `CMake` on Linux:

      ```bash
      cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
      cmake --build build --config Release
      ```

- #### BLIS

  Check [BLIS.md](docs/BLIS.md) for more information.

- #### SYCL
  SYCL is a higher-level programming model to improve programming productivity on various hardware accelerators.

  llama.cpp based on SYCL is used to **support Intel GPU** (Data Center Max series, Flex series, Arc series, Built-in GPU and iGPU).

  For detailed info, please refer to [llama.cpp for SYCL](README-sycl.md).

- #### Intel oneMKL
  Building through oneAPI compilers will make avx_vnni instruction set available for intel processors that do not support avx512 and avx512_vnni. Please note that this build config **does not support Intel GPU**. For Intel GPU support, please refer to [llama.cpp for SYCL](./README-sycl.md).

  - Using manual oneAPI installation:
    By default, `GGML_BLAS_VENDOR` is set to `Generic`, so if you already sourced intel environment script and assign `-DGGML_BLAS=ON` in cmake, the mkl version of Blas will automatically been selected. Otherwise please install oneAPI and follow the below steps:
      ```bash
      source /opt/intel/oneapi/setvars.sh # You can skip this step if  in oneapi-basekit docker image, only required for manual installation
      cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
      cmake --build build --config Release
      ```

  - Using oneAPI docker image:
    If you do not want to source the environment vars and install oneAPI manually, you can also build the code using intel docker container: [oneAPI-basekit](https://hub.docker.com/r/intel/oneapi-basekit). Then, you can use the commands given above.

  Check [Optimizing and Running LLaMA2 on Intel® CPU](https://www.intel.com/content/www/us/en/content-details/791610/optimizing-and-running-llama2-on-intel-cpu.html) for more information.

- #### CUDA

  This provides GPU acceleration using the CUDA cores of your Nvidia GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

  For Jetson user, if you have Jetson Orin, you can try this: [Offical Support](https://www.jetson-ai-lab.com/tutorial_text-generation.html). If you are using an old model(nano/TX2), need some additional operations before compiling.

  - Using `make`:
    ```bash
    make GGML_CUDA=1
    ```
  - Using `CMake`:

    ```bash
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release
    ```

  The environment variable [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) can be used to specify which GPU(s) will be used. The following compilation options are also available to tweak performance:

  | Option                        | Legal values           | Default | Description                                                                                                                                                                                                                                                                             |
  |-------------------------------|------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | GGML_CUDA_FORCE_DMMV          | Boolean                | false   | Force the use of dequantization + matrix vector multiplication kernels instead of using kernels that do matrix vector multiplication on quantized data. By default the decision is made based on compute capability (MMVQ for 6.1/Pascal/GTX 1000 or higher). Does not affect k-quants. |
  | GGML_CUDA_DMMV_X              | Positive integer >= 32 | 32      | Number of values in x direction processed by the CUDA dequantization + matrix vector multiplication kernel per iteration. Increasing this value can improve performance on fast GPUs. Power of 2 heavily recommended. Does not affect k-quants.                                         |
  | GGML_CUDA_MMV_Y               | Positive integer       | 1       | Block size in y direction for the CUDA mul mat vec kernels. Increasing this value can improve performance on fast GPUs. Power of 2 recommended.                                                                                                                                         |
  | GGML_CUDA_FORCE_MMQ           | Boolean                | false   | Force the use of custom matrix multiplication kernels for quantized models instead of FP16 cuBLAS even if there is no int8 tensor core implementation available (affects V100, RDNA3). MMQ kernels are enabled by default on GPUs with int8 tensor core support. With MMQ force enabled, speed for large batch sizes will be worse but VRAM consumption will be lower.                       |
  | GGML_CUDA_FORCE_CUBLAS        | Boolean                | false   | Force the use of FP16 cuBLAS instead of custom matrix multiplication kernels for quantized models                                                                                                                                                                                       |
  | GGML_CUDA_F16                 | Boolean                | false   | If enabled, use half-precision floating point arithmetic for the CUDA dequantization + mul mat vec kernels and for the q4_1 and q5_1 matrix matrix multiplication kernels. Can improve performance on relatively recent GPUs.                                                           |
  | GGML_CUDA_KQUANTS_ITER        | 1 or 2                 | 2       | Number of values processed per iteration and per CUDA thread for Q2_K and Q6_K quantization formats. Setting this value to 1 can improve performance for slow GPUs.                                                                                                                     |
  | GGML_CUDA_PEER_MAX_BATCH_SIZE | Positive integer       | 128     | Maximum batch size for which to enable peer access between multiple GPUs. Peer access requires either Linux or NVLink. When using NVLink enabling peer access for larger batch sizes is potentially beneficial.                                                                         |
  | GGML_CUDA_FA_ALL_QUANTS       | Boolean                | false   | Compile support for all KV cache quantization type (combinations) for the FlashAttention CUDA kernels. More fine-grained control over KV cache size but compilation takes much longer.                                                                                                  |

- #### hipBLAS

  This provides BLAS acceleration on HIP-supported AMD GPUs.
  Make sure to have ROCm installed.
  You can download it from your Linux distro's package manager or from here: [ROCm Quick Start (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick).

  - Using `make`:
    ```bash
    make GGML_HIPBLAS=1
    ```
  - Using `CMake` for Linux (assuming a gfx1030-compatible AMD GPU):
    ```bash
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
        cmake -S . -B build -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build --config Release -- -j 16
    ```
    On Linux it is also possible to use unified memory architecture (UMA) to share main memory between the CPU and integrated GPU by setting `-DGGML_HIP_UMA=ON`.
    However, this hurts performance for non-integrated GPUs (but enables working with integrated GPUs).

    Note that if you get the following error:
    ```
    clang: error: cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library
    ```
    Try searching for a directory under `HIP_PATH` that contains the file
    `oclc_abi_version_400.bc`. Then, add the following to the start of the
    command: `HIP_DEVICE_LIB_PATH=<directory-you-just-found>`, so something
    like:
    ```bash
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -p)" \
    HIP_DEVICE_LIB_PATH=<directory-you-just-found> \
        cmake -S . -B build -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build -- -j 16
    ```

  - Using `make` (example for target gfx1030, build with 16 CPU threads):
    ```bash
    make -j16 GGML_HIPBLAS=1 GGML_HIP_UMA=1 AMDGPU_TARGETS=gfx1030
    ```

  - Using `CMake` for Windows (using x64 Native Tools Command Prompt for VS, and assuming a gfx1100-compatible AMD GPU):
    ```bash
    set PATH=%HIP_PATH%\bin;%PATH%
    cmake -S . -B build -G Ninja -DAMDGPU_TARGETS=gfx1100 -DGGML_HIPBLAS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ```
    Make sure that `AMDGPU_TARGETS` is set to the GPU arch you want to compile for. The above example uses `gfx1100` that corresponds to Radeon RX 7900XTX/XT/GRE. You can find a list of targets [here](https://llvm.org/docs/AMDGPUUsage.html#processors)
    Find your gpu version string by matching the most significant version information from `rocminfo | grep gfx | head -1 | awk '{print $2}'` with the list of processors, e.g. `gfx1035` maps to `gfx1030`.


  The environment variable [`HIP_VISIBLE_DEVICES`](https://rocm.docs.amd.com/en/latest/understand/gpu_isolation.html#hip-visible-devices) can be used to specify which GPU(s) will be used.
  If your GPU is not officially supported you can use the environment variable [`HSA_OVERRIDE_GFX_VERSION`] set to a similar GPU, for example 10.3.0 on RDNA2 (e.g. gfx1030, gfx1031, or gfx1035) or 11.0.0 on RDNA3.
  The following compilation options are also available to tweak performance (yes, they refer to CUDA, not HIP, because it uses the same code as the cuBLAS version above):

  | Option                 | Legal values           | Default | Description                                                                                                                                                                                                                                    |
  |------------------------|------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | GGML_CUDA_DMMV_X       | Positive integer >= 32 | 32      | Number of values in x direction processed by the HIP dequantization + matrix vector multiplication kernel per iteration. Increasing this value can improve performance on fast GPUs. Power of 2 heavily recommended. Does not affect k-quants. |
  | GGML_CUDA_MMV_Y        | Positive integer       | 1       | Block size in y direction for the HIP mul mat vec kernels. Increasing this value can improve performance on fast GPUs. Power of 2 recommended. Does not affect k-quants.                                                                       |
  | GGML_CUDA_KQUANTS_ITER | 1 or 2                 | 2       | Number of values processed per iteration and per HIP thread for Q2_K and Q6_K quantization formats. Setting this value to 1 can improve performance for slow GPUs.                                                                             |

- #### Vulkan

  **With docker**:

  You don't need to install Vulkan SDK. It will be installed inside the container.

  ```sh
  # Build the image
  docker build -t llama-cpp-vulkan -f .devops/llama-cli-vulkan.Dockerfile .

  # Then, use it:
  docker run -it --rm -v "$(pwd):/app:Z" --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card1:/dev/dri/card1 llama-cpp-vulkan -m "/app/models/YOUR_MODEL_FILE" -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
  ```

  **Without docker**:

  Firstly, you need to make sure you have installed [Vulkan SDK](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html)

  For example, on Ubuntu 22.04 (jammy), use the command below:

  ```bash
  wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
  wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
  apt update -y
  apt-get install -y vulkan-sdk
  # To verify the installation, use the command below:
  vulkaninfo
  ```

  Alternatively your package manager might be able to provide the appropriate libraries.
  For example for Ubuntu 22.04 you can install `libvulkan-dev` instead.
  For Fedora 40, you can install `vulkan-devel`, `glslc` and `glslang` packages.

  Then, build llama.cpp using the cmake command below:

  ```bash
  cmake -B build -DGGML_VULKAN=1
  cmake --build build --config Release
  # Test the output binary (with "-ngl 33" to offload all layers to GPU)
  ./bin/llama-cli -m "PATH_TO_MODEL" -p "Hi you how are you" -n 50 -e -ngl 33 -t 4

  # You should see in the output, ggml_vulkan detected your GPU. For example:
  # ggml_vulkan: Using Intel(R) Graphics (ADL GT2) | uma: 1 | fp16: 1 | warp size: 32
  ```

### Prepare and Quantize

> [!NOTE]
> You can use the [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space on Hugging Face to quantise your model weights without any setup too. It is synced from `llama.cpp` main every 6 hours.

To obtain the official LLaMA 2 weights please see the <a href="#obtaining-and-using-the-facebook-llama-2-model">Obtaining and using the Facebook LLaMA 2 model</a> section. There is also a large selection of pre-quantized `gguf` models available on Hugging Face.

Note: `convert.py` has been moved to `examples/convert-legacy-llama.py` and shouldn't be used for anything other than `Llama/Llama2/Mistral` models and their derivatives.
It does not support LLaMA 3, you can use `convert-hf-to-gguf.py` with LLaMA 3 downloaded from Hugging Face.

```bash
# obtain the official LLaMA model weights and place them in ./models
ls ./models
llama-2-7b tokenizer_checklist.chk tokenizer.model
# [Optional] for models using BPE tokenizers
ls ./models
<folder containing weights and tokenizer json> vocab.json
# [Optional] for PyTorch .bin models like Mistral-7B
ls ./models
<folder containing weights and tokenizer json>

# install Python dependencies
python3 -m pip install -r requirements.txt

# convert the model to ggml FP16 format
python3 convert-hf-to-gguf.py models/mymodel/

# quantize the model to 4-bits (using Q4_K_M method)
./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

# update the gguf filetype to current version if older version is now unsupported
./llama-quantize ./models/mymodel/ggml-model-Q4_K_M.gguf ./models/mymodel/ggml-model-Q4_K_M-v2.gguf COPY
```

### Run the quantized model

```bash
# start inference on a gguf model
./llama-cli -m ./models/mymodel/ggml-model-Q4_K_M.gguf -n 128
```

When running the larger models, make sure you have enough disk space to store all the intermediate files.

### Running on Windows with prebuilt binaries

You will find prebuilt Windows binaries on the release page.

Simply download and extract the latest zip package of choice: (e.g. `llama-b1380-bin-win-avx2-x64.zip`)

From the unzipped folder, open a terminal/cmd window here and place a pre-converted `.gguf` model file. Test out the main example like so:

```
.\main -m llama-2-7b.Q4_0.gguf -n 128
```

### Memory/Disk Requirements

As the models are currently fully loaded into memory, you will need adequate disk space to save them and sufficient RAM to load them. At the moment, memory and disk requirements are the same.

| Model | Original size | Quantized size (Q4_0) |
|------:|--------------:|----------------------:|
|    7B |         13 GB |                3.9 GB |
|   13B |         24 GB |                7.8 GB |
|   30B |         60 GB |               19.5 GB |
|   65B |        120 GB |               38.5 GB |

### Quantization

Several quantization methods are supported. They differ in the resulting model disk size and inference speed.

*(outdated)*

| Model | Measure      |    F16 |   Q4_0 |   Q4_1 |   Q5_0 |   Q5_1 |   Q8_0 |
|------:|--------------|-------:|-------:|-------:|-------:|-------:|-------:|
|    7B | perplexity   | 5.9066 | 6.1565 | 6.0912 | 5.9862 | 5.9481 | 5.9070 |
|    7B | file size    |  13.0G |   3.5G |   3.9G |   4.3G |   4.7G |   6.7G |
|    7B | ms/tok @ 4th |    127 |     55 |     54 |     76 |     83 |     72 |
|    7B | ms/tok @ 8th |    122 |     43 |     45 |     52 |     56 |     67 |
|    7B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |
|   13B | perplexity   | 5.2543 | 5.3860 | 5.3608 | 5.2856 | 5.2706 | 5.2548 |
|   13B | file size    |  25.0G |   6.8G |   7.6G |   8.3G |   9.1G |    13G |
|   13B | ms/tok @ 4th |      - |    103 |    105 |    148 |    160 |    131 |
|   13B | ms/tok @ 8th |      - |     73 |     82 |     98 |    105 |    128 |
|   13B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |

- [k-quants](https://github.com/ggerganov/llama.cpp/pull/1684)
- recent k-quants improvements and new i-quants
  - [#2707](https://github.com/ggerganov/llama.cpp/pull/2707)
  - [#2807](https://github.com/ggerganov/llama.cpp/pull/2807)
  - [#4773 - 2-bit i-quants (inference)](https://github.com/ggerganov/llama.cpp/pull/4773)
  - [#4856 - 2-bit i-quants (inference)](https://github.com/ggerganov/llama.cpp/pull/4856)
  - [#4861 - importance matrix](https://github.com/ggerganov/llama.cpp/pull/4861)
  - [#4872 - MoE models](https://github.com/ggerganov/llama.cpp/pull/4872)
  - [#4897 - 2-bit quantization](https://github.com/ggerganov/llama.cpp/pull/4897)
  - [#4930 - imatrix for all k-quants](https://github.com/ggerganov/llama.cpp/pull/4930)
  - [#4951 - imatrix on the GPU](https://github.com/ggerganov/llama.cpp/pull/4957)
  - [#4969 - imatrix for legacy quants](https://github.com/ggerganov/llama.cpp/pull/4969)
  - [#4996 - k-qunats tuning](https://github.com/ggerganov/llama.cpp/pull/4996)
  - [#5060 - Q3_K_XS](https://github.com/ggerganov/llama.cpp/pull/5060)
  - [#5196 - 3-bit i-quants](https://github.com/ggerganov/llama.cpp/pull/5196)
  - [quantization tuning](https://github.com/ggerganov/llama.cpp/pull/5320), [another one](https://github.com/ggerganov/llama.cpp/pull/5334), and [another one](https://github.com/ggerganov/llama.cpp/pull/5361)

### Perplexity (measuring model quality)

You can use the `perplexity` example to measure perplexity over a given prompt (lower perplexity is better).
For more information, see [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity).

The perplexity measurements in table above are done against the `wikitext2` test dataset (https://paperswithcode.com/dataset/wikitext-2), with context length of 512.
The time per token is measured on a MacBook M1 Pro 32GB RAM using 4 and 8 threads.

#### How to run

1. Download/extract: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
2. Run `./llama-perplexity -m models/7B/ggml-model-q4_0.gguf -f wiki.test.raw`
3. Output:
```
perplexity : calculating perplexity over 655 chunks
24.43 seconds per pass - ETA 4.45 hours
[1]4.5970,[2]5.1807,[3]6.0382,...
```
And after 4.45 hours, you will have the final perplexity.

### Interactive mode

If you want a more ChatGPT-like experience, you can run in interactive mode by passing `-i` as a parameter.
In this mode, you can always interrupt generation by pressing Ctrl+C and entering one or more lines of text, which will be converted into tokens and appended to the current context. You can also specify a *reverse prompt* with the parameter `-r "reverse prompt string"`. This will result in user input being prompted whenever the exact tokens of the reverse prompt string are encountered in the generation. A typical use is to use a prompt that makes LLaMA emulate a chat between multiple users, say Alice and Bob, and pass `-r "Alice:"`.

Here is an example of a few-shot interaction, invoked with the command

```bash
# default arguments using a 7B model
./examples/chat.sh

# advanced chat with a 13B model
./examples/chat-13B.sh

# custom arguments using a 13B model
./llama-cli -m ./models/13B/ggml-model-q4_0.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

Note the use of `--color` to distinguish between user input and generated text. Other parameters are explained in more detail in the [README](examples/main/README.md) for the `llama-cli` example program.

![image](https://user-images.githubusercontent.com/1991296/224575029-2af3c7dc-5a65-4f64-a6bb-517a532aea38.png)

### Persistent Interaction

The prompt, user inputs, and model generations can be saved and resumed across calls to `./llama-cli` by leveraging `--prompt-cache` and `--prompt-cache-all`. The `./examples/chat-persistent.sh` script demonstrates this with support for long-running, resumable chat sessions. To use this example, you must provide a file to cache the initial chat prompt and a directory to save the chat session, and may optionally provide the same variables as `chat-13B.sh`. The same prompt cache can be reused for new chat sessions. Note that both prompt cache and chat directory are tied to the initial prompt (`PROMPT_TEMPLATE`) and the model file.

```bash
# Start a new chat
PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/default ./examples/chat-persistent.sh

# Resume that chat
PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/default ./examples/chat-persistent.sh

# Start a different chat with the same prompt/model
PROMPT_CACHE_FILE=chat.prompt.bin CHAT_SAVE_DIR=./chat/another ./examples/chat-persistent.sh

# Different prompt cache for different prompt/model
PROMPT_TEMPLATE=./prompts/chat-with-bob.txt PROMPT_CACHE_FILE=bob.prompt.bin \
    CHAT_SAVE_DIR=./chat/bob ./examples/chat-persistent.sh
```

### Constrained output with grammars

`llama.cpp` supports grammars to constrain model output. For example, you can force the model to output JSON only:

```bash
./llama-cli -m ./models/13B/ggml-model-q4_0.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'
```

The `grammars/` folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](./grammars/README.md).

For authoring more complex JSON grammars, you can also check out https://grammar.intrinsiclabs.ai/, a browser app that lets you write TypeScript interfaces which it compiles to GBNF grammars that you can save for local use. Note that the app is built and maintained by members of the community, please file any issues or FRs on [its repo](http://github.com/intrinsiclabsai/gbnfgen) and not this one.

### Obtaining and using the Facebook LLaMA 2 model

- Refer to [Facebook's LLaMA download page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) if you want to access the model data.
- Alternatively, if you want to save time and space, you can download already converted and quantized models from [TheBloke](https://huggingface.co/TheBloke), including:
  - [LLaMA 2 7B base](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
  - [LLaMA 2 13B base](https://huggingface.co/TheBloke/Llama-2-13B-GGUF)
  - [LLaMA 2 70B base](https://huggingface.co/TheBloke/Llama-2-70B-GGUF)
  - [LLaMA 2 7B chat](https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF)
  - [LLaMA 2 13B chat](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF)
  - [LLaMA 2 70B chat](https://huggingface.co/TheBloke/Llama-2-70B-chat-GGUF)

### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

### Android

#### Build on Android using Termux
[Termux](https://github.com/termux/termux-app#installation) is a method to execute `llama.cpp` on an Android device (no root required).
```
apt update && apt upgrade -y
apt install git make cmake
```

It's recommended to move your model inside the `~/` directory for best performance:
```
cd storage/downloads
mv model.gguf ~/
```

[Get the code](https://github.com/ggerganov/llama.cpp#get-the-code) & [follow the Linux build instructions](https://github.com/ggerganov/llama.cpp#build) to build `llama.cpp`.

#### Building the Project using Android NDK
Obtain the [Android NDK](https://developer.android.com/ndk) and then build with CMake.

Execute the following commands on your computer to avoid downloading the NDK to your mobile. Alternatively, you can also do this in Termux:
```
$ mkdir build-android
$ cd build-android
$ export NDK=<your_ndk_directory>
$ cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
$ make
```

Install [termux](https://github.com/termux/termux-app#installation) on your device and run `termux-setup-storage` to get access to your SD card (if Android 11+ then run the command twice).

Finally, copy these built `llama` binaries and the model file to your device storage. Because the file permissions in the Android sdcard cannot be changed, you can copy the executable files to the `/data/data/com.termux/files/home/bin` path, and then execute the following commands in Termux to add executable permission:

(Assumed that you have pushed the built executable files to the /sdcard/llama.cpp/bin path using `adb push`)
```
$cp -r /sdcard/llama.cpp/bin /data/data/com.termux/files/home/
$cd /data/data/com.termux/files/home/bin
$chmod +x ./*
```

Download model [llama-2-7b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf), and push it to `/sdcard/llama.cpp/`, then move it to `/data/data/com.termux/files/home/model/`

```
$mv /sdcard/llama.cpp/llama-2-7b-chat.Q4_K_M.gguf /data/data/com.termux/files/home/model/
```

Now, you can start chatting:
```
$cd /data/data/com.termux/files/home/bin
$./llama-cli -m ../model/llama-2-7b-chat.Q4_K_M.gguf -n 128 -cml
```

Here's a demo of an interactive session running on Pixel 5 phone:

https://user-images.githubusercontent.com/271616/225014776-1d567049-ad71-4ef2-b050-55b0b3b9274c.mp4

### Docker

#### Prerequisites
* Docker must be installed and running on your system.
* Create a folder to store big models & intermediate files (ex. /llama/models)

#### Images
We have three Docker images available for this project:

1. `ghcr.io/ggerganov/llama.cpp:full`: This image includes both the main executable file and the tools to convert LLaMA models into ggml and convert into 4-bit quantization. (platforms: `linux/amd64`, `linux/arm64`)
2. `ghcr.io/ggerganov/llama.cpp:light`: This image only includes the main executable file. (platforms: `linux/amd64`, `linux/arm64`)
3. `ghcr.io/ggerganov/llama.cpp:server`: This image only includes the server executable file. (platforms: `linux/amd64`, `linux/arm64`)

Additionally, there the following images, similar to the above:

- `ghcr.io/ggerganov/llama.cpp:full-cuda`: Same as `full` but compiled with CUDA support. (platforms: `linux/amd64`)
- `ghcr.io/ggerganov/llama.cpp:light-cuda`: Same as `light` but compiled with CUDA support. (platforms: `linux/amd64`)
- `ghcr.io/ggerganov/llama.cpp:server-cuda`: Same as `server` but compiled with CUDA support. (platforms: `linux/amd64`)
- `ghcr.io/ggerganov/llama.cpp:full-rocm`: Same as `full` but compiled with ROCm support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggerganov/llama.cpp:light-rocm`: Same as `light` but compiled with ROCm support. (platforms: `linux/amd64`, `linux/arm64`)
- `ghcr.io/ggerganov/llama.cpp:server-rocm`: Same as `server` but compiled with ROCm support. (platforms: `linux/amd64`, `linux/arm64`)

The GPU enabled images are not currently tested by CI beyond being built. They are not built with any variation from the ones in the Dockerfiles defined in [.devops/](.devops/) and the GitHub Action defined in [.github/workflows/docker.yml](.github/workflows/docker.yml). If you need different settings (for example, a different CUDA or ROCm library, you'll need to build the images locally for now).

#### Usage

The easiest way to download the models, convert them to ggml and optimize them is with the --all-in-one command which includes the full docker image.

Replace `/path/to/models` below with the actual path where you downloaded the models.

```bash
docker run -v /path/to/models:/models ghcr.io/ggerganov/llama.cpp:full --all-in-one "/models/" 7B
```

On completion, you are ready to play!

```bash
docker run -v /path/to/models:/models ghcr.io/ggerganov/llama.cpp:full --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512
```

or with a light image:

```bash
docker run -v /path/to/models:/models ghcr.io/ggerganov/llama.cpp:light -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512
```

or with a server image:

```bash
docker run -v /path/to/models:/models -p 8000:8000 ghcr.io/ggerganov/llama.cpp:server -m /models/7B/ggml-model-q4_0.gguf --port 8000 --host 0.0.0.0 -n 512
```

### Docker With CUDA

Assuming one has the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) properly installed on Linux, or is using a GPU enabled cloud, `cuBLAS` should be accessible inside the container.

#### Building Locally

```bash
docker build -t local/llama.cpp:full-cuda -f .devops/full-cuda.Dockerfile .
docker build -t local/llama.cpp:light-cuda -f .devops/llama-cli-cuda.Dockerfile .
docker build -t local/llama.cpp:server-cuda -f .devops/llama-server-cuda.Dockerfile .
```

You may want to pass in some different `ARGS`, depending on the CUDA environment supported by your container host, as well as the GPU architecture.

The defaults are:

- `CUDA_VERSION` set to `11.7.1`
- `CUDA_DOCKER_ARCH` set to `all`

The resulting images, are essentially the same as the non-CUDA images:

1. `local/llama.cpp:full-cuda`: This image includes both the main executable file and the tools to convert LLaMA models into ggml and convert into 4-bit quantization.
2. `local/llama.cpp:light-cuda`: This image only includes the main executable file.
3. `local/llama.cpp:server-cuda`: This image only includes the server executable file.

#### Usage

After building locally, Usage is similar to the non-CUDA examples, but you'll need to add the `--gpus` flag. You will also want to use the `--n-gpu-layers` flag.

```bash
docker run --gpus all -v /path/to/models:/models local/llama.cpp:full-cuda --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/llama.cpp:light-cuda -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/llama.cpp:server-cuda -m /models/7B/ggml-model-q4_0.gguf --port 8000 --host 0.0.0.0 -n 512 --n-gpu-layers 1
```

### Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Collaborators will be invited based on contributions
- Any help with managing issues and PRs is very appreciated!
- Make sure to read this: [Inference at the edge](https://github.com/ggerganov/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

### Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- There are no strict rules for the code style, but try to follow the patterns in the code (indentation, spaces, etc.). Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- See [good first issues](https://github.com/ggerganov/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices
- Matrix multiplication is unconventional: [`C = ggml_mul_mat(ctx, A, B)`](https://github.com/ggerganov/llama.cpp/blob/880e352277fc017df4d5794f0c21c44e1eae2b84/ggml.h#L1058-L1064) means $C^T = A B^T \Leftrightarrow C = B A^T.$

![matmul](media/matmul.png)

### Docs

- [main (cli)](./examples/main/README.md)
- [server](./examples/server/README.md)
- [jeopardy](./examples/jeopardy/README.md)
- [BLIS](./docs/BLIS.md)
- [Performance troubleshooting](./docs/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggerganov/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [GBNF grammars](./grammars/README.md)
