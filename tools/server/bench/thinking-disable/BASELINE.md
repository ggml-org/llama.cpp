# Thinking disable: Jinja vs --no-jinja baseline

Evidence for discussion / gate A. Same messages: one user turn, `add_generation_prompt=true`.

## Expected suffixes (`--reasoning off`, Jinja) — source of truth

From `tools/server/tests/unit/test_template.py`:

| Template file | Expected prompt suffix |
|---|---|
| Qwen-Qwen3-0.6B.jinja | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja | `<think>\n</think>` |
| Qwen-QwQ-32B.jinja | `<\|im_start\|>assistant\n<think>\n</think>` |
| CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja | `<\|START_OF_TURN_TOKEN\|><\|CHATBOT_TOKEN\|><\|START_THINKING\|><\|END_THINKING\|>` |

## Before fix (`--no-jinja --reasoning off`)

`llama_chat_apply_template` ignores `enable_thinking`. Typical suffixes:

| Template | Typical legacy suffix (fail gate A) |
|---|---|
| Qwen3 / QwQ (detected as chatml) | `<\|im_start\|>assistant\n` (no think close) |
| R1 Distill (detected as deepseek3) | `...Assistant...` without `<think>\n</think>` |
| Command-R | `...<\|CHATBOT_TOKEN\|>` without thinking tokens |

## After fix

Legacy post-process matches Jinja expected suffixes above (gate A). Verified via:

```bash
# C++ (from repo root)
./build/bin/Release/test-chat-template   # or build/bin/test-chat-template

# Server pytest
pytest tools/server/tests/unit/test_template.py -k "reasoning" -v
```

| template | jinja off suffix | no-jinja off (before) | after | A pass? |
|---|---|---|---|---|
| Qwen3-0.6B | empty think pair | bare `assistant\n` | same as jinja | yes |
| QwQ-32B | `<think>\n</think>` after assistant | bare `assistant\n` | same as jinja | yes |
| R1-Distill-Qwen | `<think>\n</think>` | bare Assistant | same as jinja | yes |
| Command-R tool_use | START/END_THINKING | bare CHATBOT_TOKEN | same as jinja | yes |

Reproduce:

```bash
pytest tools/server/tests/unit/test_template.py -k "reasoning and no_jinja" -v
```
