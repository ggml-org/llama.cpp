# Function Calling

[chat.h](../../common/chat.h) (https://github.com/ggml-org/llama.cpp/pull/9639) adds support for [OpenAI-style function calling](https://platform.openai.com/docs/guides/function-calling) and is used in:
- `llama-server` (Jinja templates are on by default; `--no-jinja` disables them and tool calling)

## Auto-parser & specialized handlers

Tool calls are parsed from the model's own format, derived from its chat template:

- The [auto-parser](../development/autoparser.md) diffs template renders to find the tool call markers and builds a PEG parser plus a lazy grammar for them.
- Formats it cannot derive have specialized handlers in [common/parsers/](../../common/parsers/) (e.g. GPT-OSS, Functionary v3.2, Ministral 3, Kimi K2/K3, Gemma 4, DeepSeek V3.2/V4, Qwen3-Coder).
- There is no generic fallback: a template that does not render tool calls gets no tool calling.
  - Use `--chat-template-file` to override the template when appropriate (see examples below)

- Multiple/parallel tool calls are enabled by default when the template supports them; set `"parallel_tool_calls"` in the completion endpoint payload to override.

To see how a given template is handled (detected markers, generated parser and grammar), build with `-DLLAMA_BUILD_TESTS=ON` and run:

```bash
./build/bin/test-chat-auto-parser models/templates/<template>.jinja
```

The per-template handler table that used to be here listed format handlers (`Generic`, `Hermes 2 Pro`, `Llama 3.x`, ...) that no longer exist; [autoparser.md](../development/autoparser.md#tested-templates) lists the templates covered by `tests/test-chat.cpp`.

# Usage - need tool-aware Jinja template

First, start a server with any model, but make sure it has a tools-enabled template: you can verify this by inspecting the `chat_template` or `chat_template_tool_use` properties in `http://localhost:8080/props`).

Here are some models known to work (w/ chat template override when needed):

```shell
# Native support:

llama-server --jinja -fa on -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
llama-server --jinja -fa on -hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q6_K_L
llama-server --jinja -fa on -hf bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M
llama-server --jinja -fa on -hf ibm-granite/granite-4.1-3b-GGUF:Q4_K_M

# Native support for DeepSeek R1 works best w/ our template override (official template is buggy, although we do work around it)

llama-server --jinja -fa on -hf bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q6_K_L \
    --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja

llama-server --jinja -fa on -hf bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M \
    --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja

# Native support requires the right template for these GGUFs:

llama-server --jinja -fa on -hf bartowski/functionary-small-v3.2-GGUF:Q4_K_M \
    --chat-template-file models/templates/meetkai-functionary-medium-v3.2.jinja

llama-server --jinja -fa on -hf bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M \
    --chat-template-file models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja

llama-server --jinja -fa on -hf bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M \
    --chat-template-file models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja

llama-server --jinja -fa on -hf bartowski/firefunction-v2-GGUF -hff firefunction-v2-IQ1_M.gguf \
    --chat-template-file models/templates/fireworks-ai-llama-3-firefunction-v2.jinja

llama-server --jinja -fa on -hf bartowski/c4ai-command-r7b-12-2024-GGUF:Q6_K_L \
    --chat-template-file models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja
```

To get the official template from original HuggingFace repos, you can use [scripts/get_chat_template.py](../../scripts/get_chat_template.py) (see examples invocations in [models/templates/README.md](../../models/templates/README.md))

> [!TIP]
> If there is no official `tool_use` Jinja template, write your own (e.g. we provide a custom [llama-cpp-deepseek-r1.jinja](../../models/templates/llama-cpp-deepseek-r1.jinja) for DeepSeek R1 distills). The built-in `--chat-template chatml` fallback does not render tools, so it does not enable tool calling.

> [!CAUTION]
> Beware of extreme KV quantizations (e.g. `-ctk q4_0`), they can substantially degrade the model's tool calling performance.

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


curl http://localhost:8080/v1/chat/completions -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a chatbot that uses tools/functions. Dont overthink things."},
        {"role": "user", "content": "What is the weather in Istanbul?"}
    ],
    "tools": [{
        "type":"function",
        "function":{
            "name":"get_current_weather",
            "description":"Get the current weather in a given location",
            "parameters":{
                "type":"object",
                "properties":{
                    "location":{
                        "type":"string",
                        "description":"The city and country/state, e.g. `San Francisco, CA`, or `Paris, France`"
                    }
                },
                "required":["location"]
            }
        }
    }]
}'
```

<details>
<summary>Show output</summary>

```json
{
"choices": [
    {
    "finish_reason": "tool_calls",
    "index": 0,
    "message": {
        "content": "",
        "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "python",
                "arguments": "{\"code\":\" \\nprint(\\\"Hello, World!\\\")\"}"
            }
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
