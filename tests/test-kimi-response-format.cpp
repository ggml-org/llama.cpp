#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"

// Regression test:
// - llama-server /chat/completions parses `response_format` into a JSON schema and passes it into
//   common_chat_templates_apply() as inputs.json_schema.
// - For templates detected as "Kimi K2", llama.cpp selected a Kimi-specific handler that did not
//   apply json_schema-to-grammar conversion, so schema enforcement was silently dropped.
//
// This test asserts that for the Kimi K2 chat template, providing a json_schema results in a
// non-empty grammar being returned by common_chat_templates_apply() (hard enforcement expected).

static const char * KIMI_K2_TEMPLATE = R"JINJA({%- if tools -%}
  <|im_system|>tool_declare<|im_middle|>
  # Tools
  {{ tools | tojson }}<|im_end|>
{%- endif -%}
{%- for message in messages -%}
  {%- if loop.first and messages[0]['role'] != 'system' -%}
    <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>
  {%- endif -%}

  {%- set role_name =  message.get('name') or  message['role'] -%}
  {%- if message['role'] == 'user' -%}
    <|im_user|>{{role_name}}<|im_middle|>
  {%- elif message['role'] == 'assistant' -%}
    <|im_assistant|>{{role_name}}<|im_middle|>
  {%- else -%}
    <|im_system|>{{role_name}}<|im_middle|>
  {% endif %}

  {%- if message['role'] == 'assistant' and message.get('tool_calls') -%}
    {%- if message['content'] -%}{{ message['content'] }}{%- endif -%}
    <|tool_calls_section_begin|>
    {%- for tool_call in message['tool_calls'] -%}
      {%- set formatted_id = tool_call['id'] -%}
      <|tool_call_begin|>{{ formatted_id }}<|tool_call_argument_begin|>{% if tool_call['function']['arguments'] is string %}{{ tool_call['function']['arguments'] }}{% else %}{{ tool_call['function']['arguments'] | tojson }}{% endif %}<|tool_call_end|>
    {%- endfor -%}
    <|tool_calls_section_end|>
  {%- elif message['role'] == 'tool' -%}
    ## Return of {{ message.tool_call_id }}
    {{ message['content'] }}
  {%- elif message['content'] is string -%}
    {{ message['content'] }}
  {%- elif message['content'] is not none -%}
    {% for content in message['content'] -%}
      {% if content['type'] == 'image' or 'image' in content or 'image_url' in content -%}
        <|media_start|>image<|media_content|><|media_pad|><|media_end|>
      {% else -%}
        {{ content['text'] }}
      {%- endif -%}
    {%- endfor -%}
  {%- endif -%}
  <|im_end|>
{%- endfor -%}
{%- if add_generation_prompt -%}
  <|im_assistant|>assistant<|im_middle|>
{%- endif -%})JINJA";

int main() {
    auto tmpls = common_chat_templates_init(/* model= */ nullptr, KIMI_K2_TEMPLATE);

    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;

    // No tools
    inputs.tools = {};
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;

    inputs.json_schema = R"JSON({
      "type": "object",
      "properties": { "ok": { "type": "boolean" } },
      "required": ["ok"],
      "additionalProperties": false
    })JSON";

    inputs.messages = {
        common_chat_msg{"system", "Return ONLY JSON with key ok.", {}, {}, "", "", ""},
        common_chat_msg{"user", "ok", {}, {}, "", "", ""},
    };

    const auto out = common_chat_templates_apply(tmpls.get(), inputs);
    
    // Confirm the Kimi K2 handler was actually selected (not a generic fallback).
    assert(out.format == COMMON_CHAT_FORMAT_KIMI_K2);
    assert(!out.grammar.empty());

    // tools + json_schema is explicitly unsupported for Kimi K2 (ambiguous composition).
    // Ensure we fail loudly rather than silently dropping schema enforcement.
    inputs.tools = {
        common_chat_tool{
            /* .name = */ "noop",
            /* .description = */ "No-op tool",
            /* .parameters = */ R"JSON({
              "type": "object",
              "properties": { "x": { "type": "string" } },
              "required": ["x"],
              "additionalProperties": false
            })JSON",
        },
    };
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;

    bool threw = false;
    try {
        (void) common_chat_templates_apply(tmpls.get(), inputs);
    } catch (const std::exception &) {
        threw = true;
    }
    // Avoid relying on assert() in Release builds (may be compiled out).
    if (!threw) {
        return 2;
    }
    return 0;
}

