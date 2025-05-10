# Function Calling

[chat.h](../common/chat.h) (https://github.com/ggml-org/llama.cpp/pull/9639) adds support for [OpenAI-style function calling](https://platform.openai.com/docs/guides/function-calling) and is used in:
- `llama-server` when started w/ `--jinja` flag
- `llama-cli` (WIP: https://github.com/ggml-org/llama.cpp/pull/11556)

## Universal support w/ Native & Generic handlers

Function calling is supported for all models (see https://github.com/ggml-org/llama.cpp/pull/9639):

- Native tool call formats supported:
  - Llama 3.1 / 3.3 (including builtin tools support - tool names for `wolfram_alpha`, `web_search` / `brave_search`, `code_interpreter`), Llama 3.2
  - Functionary v3.1 / v3.2
  - Hermes 2/3, Qwen 2.5
  - Qwen 2.5 Coder (WIP: https://github.com/ggml-org/llama.cpp/pull/12034)
  - Mistral Nemo
  - Firefunction v2
  - Command R7B
  - DeepSeek R1 (WIP / seems reluctant to call any tools?)

- Generic tool call is supported when the template isn't recognized by native format handlers (you'll see `Chat format: Generic` in the logs).
  - Use `--chat-template-file` to override the template when appropriate (see examples below)
  - Generic support may consume more tokens and be less efficient than a model's native format.

<details>
<summary>Show some common templates and which format handler they use</summary>

| Template | Format |
|----------|--------|
| Almawave-Velvet-14B.jinja | Hermes 2 Pro |
| AtlaAI-Selene-1-Mini-Llama-3.1-8B.jinja | Llama 3.x |
| CohereForAI-aya-expanse-8b.jinja | Generic |
| CohereForAI-c4ai-command-r-plus-default.jinja | Generic |
| CohereForAI-c4ai-command-r-plus-rag.jinja | Generic |
| CohereForAI-c4ai-command-r-plus-tool_use.jinja | Generic |
| CohereForAI-c4ai-command-r7b-12-2024-default.jinja | Command R7B (extract reasoning) |
| CohereForAI-c4ai-command-r7b-12-2024-rag.jinja | Command R7B (extract reasoning) |
| CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja | Command R7B (extract reasoning) |
| CohereForAI-c4ai-command-r7b-12-2024.jinja | Generic |
| DavieLion-Llama-3.2-1B-SPIN-iter3.jinja | Generic |
| Delta-Vector-Rei-12B.jinja | Mistral Nemo |
| EpistemeAI-Mistral-Nemo-Instruct-12B-Philosophy-Math.jinja | Mistral Nemo |
| FlofloB-83k_continued_pretraining_Qwen2.5-0.5B-Instruct_Unsloth_merged_16bit.jinja | Hermes 2 Pro |
| FlofloB-test_continued_pretraining_Phi-3-mini-4k-instruct_Unsloth_merged_16bit.jinja | Generic |
| HelpingAI-HAI-SER.jinja | Generic |
| HuggingFaceTB-SmolLM2-1.7B-Instruct.jinja | Generic |
| HuggingFaceTB-SmolLM2-135M-Instruct.jinja | Generic |
| HuggingFaceTB-SmolLM2-360M-Instruct.jinja | Generic |
| INSAIT-Institute-BgGPT-Gemma-2-27B-IT-v1.0.jinja | Generic |
| Ihor-Text2Graph-R1-Qwen2.5-0.5b.jinja | Hermes 2 Pro |
| Infinigence-Megrez-3B-Instruct.jinja | Generic |
| Josephgflowers-TinyLlama_v1.1_math_code-world-test-1.jinja | Generic |
| LGAI-EXAONE-EXAONE-3.5-2.4B-Instruct.jinja | Generic |
| LGAI-EXAONE-EXAONE-3.5-7.8B-Instruct.jinja | Generic |
| LatitudeGames-Wayfarer-12B.jinja | Generic |
| Magpie-Align-Llama-3-8B-Magpie-Align-v0.1.jinja | Generic |
| Magpie-Align-Llama-3.1-8B-Magpie-Align-v0.1.jinja | Generic |
| MaziyarPanahi-calme-3.2-instruct-78b.jinja | Generic |
| MiniMaxAI-MiniMax-Text-01.jinja | Generic |
| MiniMaxAI-MiniMax-VL-01.jinja | Generic |
| NaniDAO-deepseek-r1-qwen-2.5-32B-ablated.jinja | DeepSeek R1 (extract reasoning) |
| NexaAIDev-Octopus-v2.jinja | Generic |
| NousResearch-Hermes-2-Pro-Llama-3-8B-default.jinja | Generic |
| NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja | Hermes 2 Pro |
| NousResearch-Hermes-2-Pro-Mistral-7B-default.jinja | Generic |
| NousResearch-Hermes-2-Pro-Mistral-7B-tool_use.jinja | Hermes 2 Pro |
| NousResearch-Hermes-3-Llama-3.1-70B-default.jinja | Generic |
| NousResearch-Hermes-3-Llama-3.1-70B-tool_use.jinja | Hermes 2 Pro |
| NovaSky-AI-Sky-T1-32B-Flash.jinja | Hermes 2 Pro |
| NovaSky-AI-Sky-T1-32B-Preview.jinja | Hermes 2 Pro |
| OnlyCheeini-greesychat-turbo.jinja | Generic |
| Orenguteng-Llama-3.1-8B-Lexi-Uncensored-V2.jinja | Llama 3.x |
| OrionStarAI-Orion-14B-Chat.jinja | Generic |
| PowerInfer-SmallThinker-3B-Preview.jinja | Generic |
| PrimeIntellect-INTELLECT-1-Instruct.jinja | Generic |
| Qwen-QVQ-72B-Preview.jinja | Generic |
| Qwen-QwQ-32B-Preview.jinja | Hermes 2 Pro |
| Qwen-Qwen1.5-7B-Chat.jinja | Generic |
| Qwen-Qwen2-7B-Instruct.jinja | Generic |
| Qwen-Qwen2-VL-72B-Instruct.jinja | Generic |
| Qwen-Qwen2-VL-7B-Instruct.jinja | Generic |
| Qwen-Qwen2.5-0.5B.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-1.5B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-14B-Instruct-1M.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-14B.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-32B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-32B.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-3B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-72B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-7B-Instruct-1M.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-7B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-7B.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-Coder-32B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-Coder-7B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-Math-1.5B.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-Math-7B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-VL-3B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-VL-72B-Instruct.jinja | Hermes 2 Pro |
| Qwen-Qwen2.5-VL-7B-Instruct.jinja | Hermes 2 Pro |
| RWKV-Red-Team-ARWKV-7B-Preview-0.1.jinja | Hermes 2 Pro |
| SakanaAI-TinySwallow-1.5B-Instruct.jinja | Hermes 2 Pro |
| SakanaAI-TinySwallow-1.5B.jinja | Hermes 2 Pro |
| Sao10K-70B-L3.3-Cirrus-x1.jinja | Llama 3.x |
| SentientAGI-Dobby-Mini-Leashed-Llama-3.1-8B.jinja | Llama 3.x |
| SentientAGI-Dobby-Mini-Unhinged-Llama-3.1-8B.jinja | Llama 3.x |
| Steelskull-L3.3-Damascus-R1.jinja | Llama 3.x |
| Steelskull-L3.3-MS-Nevoria-70b.jinja | Llama 3.x |
| Steelskull-L3.3-Nevoria-R1-70b.jinja | Llama 3.x |
| THUDM-glm-4-9b-chat.jinja | Generic |
| THUDM-glm-edge-1.5b-chat.jinja | Generic |
| Tarek07-Progenitor-V1.1-LLaMa-70B.jinja | Llama 3.x |
| TheBloke-FusionNet_34Bx2_MoE-AWQ.jinja | Generic |
| TinyLlama-TinyLlama-1.1B-Chat-v1.0.jinja | Generic |
| UCLA-AGI-Mistral7B-PairRM-SPPO-Iter3.jinja | Generic |
| ValiantLabs-Llama3.1-8B-Enigma.jinja | Llama 3.x |
| abacusai-Fewshot-Metamath-OrcaVicuna-Mistral.jinja | Generic |
| ai21labs-AI21-Jamba-1.5-Large.jinja | Generic |
| allenai-Llama-3.1-Tulu-3-405B-SFT.jinja | Generic |
| allenai-Llama-3.1-Tulu-3-405B.jinja | Generic |
| allenai-Llama-3.1-Tulu-3-8B.jinja | Generic |
| arcee-ai-Virtuoso-Lite.jinja | Hermes 2 Pro |
| arcee-ai-Virtuoso-Medium-v2.jinja | Hermes 2 Pro |
| arcee-ai-Virtuoso-Small-v2.jinja | Hermes 2 Pro |
| avemio-GRAG-NEMO-12B-ORPO-HESSIAN-AI.jinja | Generic |
| bespokelabs-Bespoke-Stratos-7B.jinja | Hermes 2 Pro |
| bfuzzy1-acheron-m1a-llama.jinja | Generic |
| bofenghuang-vigogne-2-70b-chat.jinja | Generic |
| bytedance-research-UI-TARS-72B-DPO.jinja | Generic |
| bytedance-research-UI-TARS-7B-DPO.jinja | Generic |
| bytedance-research-UI-TARS-7B-SFT.jinja | Generic |
| carsenk-phi3.5_mini_exp_825_uncensored.jinja | Generic |
| cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese.jinja | DeepSeek R1 (extract reasoning) |
| cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese.jinja | DeepSeek R1 (extract reasoning) |
| databricks-dbrx-instruct.jinja | Generic |
| deepseek-ai-DeepSeek-Coder-V2-Instruct.jinja | Generic |
| deepseek-ai-DeepSeek-Coder-V2-Lite-Base.jinja | Generic |
| deepseek-ai-DeepSeek-Coder-V2-Lite-Instruct.jinja | Generic |
| deepseek-ai-DeepSeek-R1-Distill-Llama-70B.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-R1-Distill-Qwen-14B.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-R1-Distill-Qwen-7B.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-R1-Zero.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-R1.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-V2-Lite.jinja | Generic |
| deepseek-ai-DeepSeek-V2.5.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-DeepSeek-V3.jinja | DeepSeek R1 (extract reasoning) |
| deepseek-ai-deepseek-coder-33b-instruct.jinja | Generic |
| deepseek-ai-deepseek-coder-6.7b-instruct.jinja | Generic |
| deepseek-ai-deepseek-coder-7b-instruct-v1.5.jinja | Generic |
| deepseek-ai-deepseek-llm-67b-chat.jinja | Generic |
| deepseek-ai-deepseek-llm-7b-chat.jinja | Generic |
| dicta-il-dictalm2.0-instruct.jinja | Generic |
| ehristoforu-Falcon3-8B-Franken-Basestruct.jinja | Hermes 2 Pro |
| fireworks-ai-llama-3-firefunction-v2.jinja | FireFunction v2 |
| godlikehhd-alpaca_data_sampled_ifd_new_5200.jinja | Hermes 2 Pro |
| godlikehhd-alpaca_data_score_max_0.7_2600.jinja | Hermes 2 Pro |
| google-gemma-2-27b-it.jinja | Generic |
| google-gemma-2-2b-it.jinja | Generic |
| google-gemma-2-2b-jpn-it.jinja | Generic |
| google-gemma-7b-it.jinja | Generic |
| huihui-ai-DeepSeek-R1-Distill-Llama-70B-abliterated.jinja | DeepSeek R1 (extract reasoning) |
| huihui-ai-DeepSeek-R1-Distill-Llama-8B-abliterated.jinja | DeepSeek R1 (extract reasoning) |
| huihui-ai-DeepSeek-R1-Distill-Qwen-14B-abliterated-v2.jinja | DeepSeek R1 (extract reasoning) |
| huihui-ai-DeepSeek-R1-Distill-Qwen-32B-abliterated.jinja | DeepSeek R1 (extract reasoning) |
| huihui-ai-DeepSeek-R1-Distill-Qwen-7B-abliterated-v2.jinja | DeepSeek R1 (extract reasoning) |
| huihui-ai-Qwen2.5-14B-Instruct-1M-abliterated.jinja | Hermes 2 Pro |
| ibm-granite-granite-3.1-8b-instruct.jinja | Generic |
| indischepartij-MiniCPM-3B-OpenHermes-2.5-v2.jinja | Generic |
| inflatebot-MN-12B-Mag-Mell-R1.jinja | Generic |
| jinaai-ReaderLM-v2.jinja | Generic |
| kms7530-chemeng_qwen-math-7b_24_1_100_1_nonmath.jinja | Hermes 2 Pro |
| knifeayumu-Cydonia-v1.3-Magnum-v4-22B.jinja | Mistral Nemo |
| langgptai-qwen1.5-7b-chat-sa-v0.1.jinja | Generic |
| lightblue-DeepSeek-R1-Distill-Qwen-7B-Japanese.jinja | DeepSeek R1 (extract reasoning) |
| mattshumer-Reflection-Llama-3.1-70B.jinja | Generic |
| meetkai-functionary-medium-v3.1.jinja | Functionary v3.1 Llama 3.1 |
| meetkai-functionary-medium-v3.2.jinja | Functionary v3.2 |
| meta-llama-Llama-2-7b-chat-hf.jinja | Generic |
| meta-llama-Llama-3.1-8B-Instruct.jinja | Llama 3.x |
| meta-llama-Llama-3.2-11B-Vision-Instruct.jinja | Llama 3.x |
| meta-llama-Llama-3.2-1B-Instruct.jinja | Llama 3.x |
| meta-llama-Llama-3.2-3B-Instruct.jinja | Llama 3.x |
| meta-llama-Llama-3.3-70B-Instruct.jinja | Llama 3.x |
| meta-llama-Meta-Llama-3-8B-Instruct.jinja | Generic |
| meta-llama-Meta-Llama-3.1-8B-Instruct.jinja | Llama 3.x |
| microsoft-Phi-3-medium-4k-instruct.jinja | Generic |
| microsoft-Phi-3-mini-4k-instruct.jinja | Generic |
| microsoft-Phi-3-small-8k-instruct.jinja | Generic |
| microsoft-Phi-3.5-mini-instruct.jinja | Generic |
| microsoft-Phi-3.5-vision-instruct.jinja | Generic |
| microsoft-phi-4.jinja | Generic |
| migtissera-Tess-3-Mistral-Nemo-12B.jinja | Generic |
| ministral-Ministral-3b-instruct.jinja | Generic |
| mistralai-Codestral-22B-v0.1.jinja | Generic |
| mistralai-Mistral-7B-Instruct-v0.1.jinja | Generic |
| mistralai-Mistral-7B-Instruct-v0.2.jinja | Generic |
| mistralai-Mistral-7B-Instruct-v0.3.jinja | Mistral Nemo |
| mistralai-Mistral-Large-Instruct-2407.jinja | Mistral Nemo |
| mistralai-Mistral-Large-Instruct-2411.jinja | Generic |
| mistralai-Mistral-Nemo-Instruct-2407.jinja | Mistral Nemo |
| mistralai-Mistral-Small-24B-Instruct-2501.jinja | Generic |
| mistralai-Mixtral-8x7B-Instruct-v0.1.jinja | Generic |
| mkurman-Qwen2.5-14B-DeepSeek-R1-1M.jinja | Hermes 2 Pro |
| mlabonne-AlphaMonarch-7B.jinja | Generic |
| mlx-community-Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1-float32.jinja | Hermes 2 Pro |
| mlx-community-Qwen2.5-VL-7B-Instruct-8bit.jinja | Hermes 2 Pro |
| mobiuslabsgmbh-DeepSeek-R1-ReDistill-Qwen-1.5B-v1.1.jinja | DeepSeek R1 (extract reasoning) |
| netcat420-MFANNv0.20.jinja | Generic |
| netcat420-MFANNv0.24.jinja | Generic |
| netease-youdao-Confucius-o1-14B.jinja | Hermes 2 Pro |
| nvidia-AceMath-7B-RM.jinja | Hermes 2 Pro |
| nvidia-Eagle2-1B.jinja | Hermes 2 Pro |
| nvidia-Eagle2-9B.jinja | Hermes 2 Pro |
| nvidia-Llama-3.1-Nemotron-70B-Instruct-HF.jinja | Llama 3.x |
| onnx-community-DeepSeek-R1-Distill-Qwen-1.5B-ONNX.jinja | DeepSeek R1 (extract reasoning) |
| open-thoughts-OpenThinker-7B.jinja | Hermes 2 Pro |
| openchat-openchat-3.5-0106.jinja | Generic |
| pankajmathur-orca_mini_v6_8b.jinja | Generic |
| princeton-nlp-Mistral-7B-Base-SFT-RDPO.jinja | Generic |
| princeton-nlp-Mistral-7B-Instruct-DPO.jinja | Generic |
| princeton-nlp-Mistral-7B-Instruct-RDPO.jinja | Generic |
| prithivMLmods-Bellatrix-Tiny-1.5B-R1.jinja | Hermes 2 Pro |
| prithivMLmods-Bellatrix-Tiny-1B-R1.jinja | Llama 3.x |
| prithivMLmods-Bellatrix-Tiny-1B-v3.jinja | Generic |
| prithivMLmods-Bellatrix-Tiny-3B-R1.jinja | Llama 3.x |
| prithivMLmods-Blaze-14B-xElite.jinja | Generic |
| prithivMLmods-Calcium-Opus-14B-Elite2-R1.jinja | Hermes 2 Pro |
| prithivMLmods-Calme-Ties-78B.jinja | Generic |
| prithivMLmods-Calme-Ties2-78B.jinja | Generic |
| prithivMLmods-Calme-Ties3-78B.jinja | Generic |
| prithivMLmods-ChemQwen2-vL.jinja | Generic |
| prithivMLmods-GWQ2b.jinja | Generic |
| prithivMLmods-LatexMind-2B-Codec.jinja | Generic |
| prithivMLmods-Llama-3.2-6B-AlgoCode.jinja | Llama 3.x |
| prithivMLmods-Megatron-Opus-14B-Exp.jinja | Hermes 2 Pro |
| prithivMLmods-Megatron-Opus-14B-Stock.jinja | Hermes 2 Pro |
| prithivMLmods-Megatron-Opus-7B-Exp.jinja | Hermes 2 Pro |
| prithivMLmods-Omni-Reasoner-Merged.jinja | Hermes 2 Pro |
| prithivMLmods-Omni-Reasoner4-Merged.jinja | Hermes 2 Pro |
| prithivMLmods-Primal-Opus-14B-Optimus-v1.jinja | Hermes 2 Pro |
| prithivMLmods-QwQ-Math-IO-500M.jinja | Hermes 2 Pro |
| prithivMLmods-Qwen-7B-Distill-Reasoner.jinja | DeepSeek R1 (extract reasoning) |
| prithivMLmods-Qwen2.5-1.5B-DeepSeek-R1-Instruct.jinja | Hermes 2 Pro |
| prithivMLmods-Qwen2.5-14B-DeepSeek-R1-1M.jinja | Hermes 2 Pro |
| prithivMLmods-Qwen2.5-32B-DeepSeek-R1-Instruct.jinja | Hermes 2 Pro |
| prithivMLmods-Qwen2.5-7B-DeepSeek-R1-1M.jinja | Hermes 2 Pro |
| prithivMLmods-Triangulum-v2-10B.jinja | Hermes 2 Pro |
| qingy2024-Falcon3-2x10B-MoE-Instruct.jinja | Hermes 2 Pro |
| rubenroy-Zurich-14B-GCv2-5m.jinja | Hermes 2 Pro |
| rubenroy-Zurich-7B-GCv2-5m.jinja | Hermes 2 Pro |
| silma-ai-SILMA-Kashif-2B-Instruct-v1.0.jinja | Generic |
| simplescaling-s1-32B.jinja | Hermes 2 Pro |
| sometimesanotion-Lamarck-14B-v0.7.jinja | Hermes 2 Pro |
| sonthenguyen-zephyr-sft-bnb-4bit-DPO-mtbr-180steps.jinja | Generic |
| sthenno-tempesthenno-icy-0130.jinja | Generic |
| sumink-qwft.jinja | Hermes 2 Pro |
| teknium-OpenHermes-2.5-Mistral-7B.jinja | Generic |
| thirdeyeai-elevate360m.jinja | Generic |
| tiiuae-Falcon3-10B-Instruct.jinja | Hermes 2 Pro |
| unsloth-DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit.jinja | DeepSeek R1 (extract reasoning) |
| unsloth-DeepSeek-R1-Distill-Llama-8B.jinja | DeepSeek R1 (extract reasoning) |
| unsloth-DeepSeek-R1.jinja | DeepSeek R1 (extract reasoning) |
| unsloth-Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit.jinja | Generic |
| upstage-solar-pro-preview-instruct.jinja | Generic |
| whyhow-ai-PatientSeek.jinja | Generic |
| xwen-team-Xwen-72B-Chat.jinja | Hermes 2 Pro |
| xwen-team-Xwen-7B-Chat.jinja | Hermes 2 Pro |

This table can be generated with:

```bash
./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
```

</details>

# Usage - need tool-aware Jinja template

First, start a server with any model, but make sure it has a tools-enabled template: you can verify this by inspecting the `chat_template` or `chat_template_tool_use` properties in `http://localhost:8080/props`).

Here are some models known to work (w/ chat template override when needed):

```shell
# Native support:

llama-server --jinja -fa -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
llama-server --jinja -fa -hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q6_K_L
llama-server --jinja -fa -hf bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M

# Native support for DeepSeek R1 works best w/ our template override (official template is buggy, although we do work around it)

llama-server --jinja -fa -hf bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q6_K_L \
    --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja

llama-server --jinja -fa -hf bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M \
    --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja

# Native support requires the right template for these GGUFs:

llama-server --jinja -fa -hf bartowski/functionary-small-v3.2-GGUF:Q4_K_M
    --chat-template-file models/templates/meetkai-functionary-medium-v3.2.jinja

llama-server --jinja -fa -hf bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M \
    --chat-template-file models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja

llama-server --jinja -fa -hf bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M \
    --chat-template-file models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja

llama-server --jinja -fa -hf bartowski/firefunction-v2-GGUF -hff firefunction-v2-IQ1_M.gguf \
    --chat-template-file models/templates/fireworks-ai-llama-3-firefunction-v2.jinja

llama-server --jinja -fa -hf bartowski/c4ai-command-r7b-12-2024-GGUF:Q6_K_L \
    --chat-template-file models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja

# Generic format support
llama-server --jinja -fa -hf bartowski/phi-4-GGUF:Q4_0
llama-server --jinja -fa -hf bartowski/gemma-2-2b-it-GGUF:Q8_0
llama-server --jinja -fa -hf bartowski/c4ai-command-r-v01-GGUF:Q2_K
```

To get the official template from original HuggingFace repos, you can use [scripts/get_chat_template.py](../scripts/get_chat_template.py) (see examples invocations in [models/templates/README.md](../models/templates/README.md))

> [!TIP]
> If there is no official `tool_use` Jinja template, you may want to set `--chat-template chatml` to use a default that works with many models (YMMV!), or write your own (e.g. we provide a custom [llama-cpp-deepseek-r1.jinja](../models/templates/llama-cpp-deepseek-r1.jinja) for DeepSeek R1 distills)

Test in CLI (or with any library / software that can use OpenAI-compatible API backends):

```bash
curl http://localhost:8080/v1/chat/completions -d '{
"model": "gpt-3.5-turbo",
"tools": [
    {
    "type":"function",
    "function":{
        "name":"python",
        "description":"Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.",
        "parameters":{
        "type":"object",
        "properties":{
            "code":{
            "type":"string",
            "description":"The code to run in the ipython interpreter."
            }
        },
        "required":["code"]
        }
    }
    }
],
"messages": [
    {
    "role": "user",
    "content": "Print a hello world message with python."
    }
]
}'
```

<details>
<summary>Show output</summary>

```json
{
"choices": [
    {
    "finish_reason": "tool",
    "index": 0,
    "message": {
        "content": null,
        "tool_calls": [
        {
            "name": "python",
            "arguments": "{\"code\":\" \\nprint(\\\"Hello, World!\\\")\"}"
        }
        ],
        "role": "assistant"
    }
    }
],
"created": 1727287211,
"model": "gpt-3.5-turbo",
"object": "chat.completion",
"usage": {
    "completion_tokens": 16,
    "prompt_tokens": 44,
    "total_tokens": 60
},
"id": "chatcmpl-Htbgh9feMmGM0LEH2hmQvwsCxq3c6Ni8"
}
```

</details>
