# Auto-Parser Architecture

The auto-parser automatically analyzes chat templates to determine how to parse model outputs, including content, reasoning, and tool calls.

## Overview

The unified auto-parser uses a pure differential, compositional approach (inspired by the `git diff` algorithm) to analyze chat templates:

**Core Philosophy**:

- **Minimize Hardcoded Patterns**: All markers extracted through template comparison (the only heuristic is JSON detection to distinguish `JSON_NATIVE` from tag-based formats)
- **Compositional Architecture**: Separate analyzer structs for reasoning, content, and tools — each responsible for its own analysis and parser construction

**Analysis + Parser Building in Two Steps**:

1. `autoparser::autoparser tmpl_analysis; tmpl_analysis.analyze_template(tmpl)` - runs all differential comparisons and populates the analysis structs
2. `autoparser::peg_generator::generate_parser(tmpl, generation_params, tmpl_analysis)` - uses the analysis to build a PEG parser and optional GBNF grammar

## Data Structures

All structs are defined in [common/chat-auto-parser.h](../../common/chat-auto-parser.h).

### Top-Level: `autoparser` (main analyzer and generator)

[common/chat-auto-parser.h:381-409](../../common/chat-auto-parser.h#L381-L409) - top-level analysis result aggregating `jinja_caps`, `reasoning`, `content`, and `tools` sub-analyses, plus `user_start`/`assistant_start` message markers, `preserved_tokens` (union of all non-empty markers) and `additional_stops` (literal stop strings).

### `analyze_reasoning`

[common/chat-auto-parser.h:256-277](../../common/chat-auto-parser.h#L256-L277) - reasoning analysis result: `mode` enum, `start` marker (e.g. `<think>`), and `end` marker (e.g. `</think>`).

### `analyze_content`

[common/chat-auto-parser.h:283-298](../../common/chat-auto-parser.h#L283-L298) - content analysis result: `mode` enum, `start`/`end` markers, and `requires_nonnull_content` flag.

### `analyze_tools` and its sub-structs

- [common/chat-auto-parser.h:174-192](../../common/chat-auto-parser.h#L174-L192) - `tool_format_analysis`: `mode` enum, `section_start/end`, `per_call_start/end`, JSON field names (`function_field`, `name_field`, `args_field`, `id_field`, `gen_id_field`), `parameter_order`, and format flags (`fun_name_is_key`, `tools_array_wrapped`, `openai_wrapper_trigger`)
- [common/chat-auto-parser.h:194-199](../../common/chat-auto-parser.h#L194-L199) - `tool_function_analysis`: `name_prefix`, `name_suffix`, `args_separator`, `close` markers around function names
- [common/chat-auto-parser.h:201-210](../../common/chat-auto-parser.h#L201-L210) - `tool_arguments_analysis`: `start/end` container markers, `name_prefix/suffix`, `value_prefix/suffix`, `separator`, `tolerate_intertag_whitespace`
- [common/chat-auto-parser.h:212-217](../../common/chat-auto-parser.h#L212-L217) - `tool_id_analysis`: `pos` enum, `prefix`/`suffix` markers around call ID values
- [common/chat-auto-parser.h:304-375](../../common/chat-auto-parser.h#L304-L375) - `analyze_tools`: aggregates the four sub-structs above

### Enums

**`reasoning_mode`**: How the template handles reasoning/thinking blocks.

| Value           | Description                                                                       |
|-----------------|-----------------------------------------------------------------------------------|
| `NONE`          | No reasoning markers detected                                                     |
| `TAG_BASED`     | Tag-based: `<think>...</think>` (start can be empty for delimiter-style formats)  |
| `TOOLS_ONLY`    | Reasoning only appears in tool call responses, not plain content                  |

**Generation Prompt & Reasoning Prefill**: Computed by `common_chat_template_generation_prompt()` (specialized handlers call its `_impl` variant) in `common/chat.cpp`, which renders the template twice - once with `add_generation_prompt=false` and once with `add_generation_prompt=true` - and returns the diff suffix. The auto-parser and the specialized handlers store it in `common_chat_params::generation_prompt`, which is propagated into `common_chat_parser_params::generation_prompt` and `common_params_sampling::generation_prompt`.

The generation prompt is prepended to model output before PEG parsing (`common_chat_peg_parse()`). `autoparser::build_parser()` starts the parser with `p.prefix(generation_prompt, reasoning_start)`, a literal for the portion *before* the reasoning start marker (if any), so any boilerplate added by the template is consumed. The full string is also fed to the grammar sampler via `llama_sampler_accept`, advancing the grammar past tokens already in the prompt, and to the reasoning budget sampler, which therefore starts COUNTING when the generation prompt opens reasoning without closing it.

**Grammar prefill** (`common_sampler_init()`): the tokenized generation prompt is accepted by the grammar sampler only for output-format and tool-call grammars (`common_grammar_needs_prefill()`) and only when the grammar is not lazy; user-supplied grammars are never prefilled.

How the reasoning parser sees a prefilled generation prompt:

1. **Start+end in generation prompt** (e.g. `<think></think>\n`): the parser sees reasoning as opened and immediately closed; whitespace-only reasoning content is discarded by the mapper.
2. **Only start in generation prompt** (e.g. `<think>\n`): the prepended start marker opens reasoning, so the model output is parsed as reasoning until the end marker.

**`content_mode`**: How the template wraps assistant content.

| Value                    | Description                                                    |
|--------------------------|----------------------------------------------------------------|
| `PLAIN`                  | No content markers                                             |
| `ALWAYS_WRAPPED`         | Content always wrapped: `<response>...</response>`             |
| `WRAPPED_WITH_REASONING` | Content wrapped only when reasoning is present                 |

**`tool_format`**: Classification of tool call structure.

| Value            | Description                                                      |
|------------------|------------------------------------------------------------------|
| `NONE`           | No tool support detected                                         |
| `JSON_NATIVE`    | Pure JSON: `{"name": "X", "arguments": {...}}`                   |
| `TAG_WITH_JSON`  | Tag-based with JSON args: `<function=X>{...}</function>`         |
| `TAG_WITH_TAGGED`| Tag-based with tagged args: `<param=key>value</param>`           |

**`call_id_position`**: Where call IDs appear in tag-based formats.

| Value                    | Description                                  |
|--------------------------|----------------------------------------------|
| `NONE`                   | No call ID support detected                  |
| `PRE_FUNC_NAME`          | Before function name                         |
| `BETWEEN_FUNC_AND_ARGS`  | Between function name and arguments          |
| `POST_ARGS`              | After arguments                              |

## Tool Calling Formats

### JSON_NATIVE

**Structure**: The entire tool call (function name, arguments, values) is in JSON format. Optional enclosing tags around the section.

**Detection**: Function name appears inside a JSON structure (quotes preceded by `{` or `:`).

**Examples**:

Standard OpenAI-style:

```json
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call>
```

Mistral Nemo with array wrapper:

```json
[TOOL_CALLS]
[{"name": "calculate", "arguments": {"expr": "2+2"}}]
```

Function name as JSON key (Apertus style):

```json
{"get_weather": {"location": "Paris"}}
```

---

### TAG_WITH_JSON

**Structure**: Function name is outside JSON, in tag attributes or XML-style tags. Arguments are a JSON object.

**Detection**: Function name not in JSON, but argument names appear in JSON context.

**Examples**:

Functionary v3.1:

```xml
<function=get_weather>{"location": "Paris", "unit": "celsius"}</function>
```

MiniMax:

```xml
<minimax:tool_call>
<tool_name>calculate</tool_name>
<arguments>{"expr": "2+2"}</arguments>
</minimax:tool_call>
```

---

### TAG_WITH_TAGGED

**Structure**: Both function name and argument names are in XML-style tags. String values are unquoted; non-string values are JSON-formatted.

**Detection**: Neither function name nor argument names appear in a JSON context.

**Examples**:

Qwen/Hermes XML format:

```xml
<function=get_weather>
<param=location>Paris</param>
<param=unit>celsius</param>
</function>
```

Mixed types:

```xml
<function=calculate>
<param=expr>2+2</param>
<param=precision>2</param>
<param=options>{"round": true}</param>
</function>
```

String values (`Paris`, `celsius`, `2+2`) are unquoted; `options` (object type) is JSON-formatted.

---

## Analysis Flow

```text
autoparser::analyze_template(tmpl)
    |
    |-- Phase 1: analyze_reasoning(tmpl, jinja_caps.supports_tool_calls)
    |     |-- R1: compare_reasoning_presence()   — with/without reasoning_content field
    |     |-- R2: compare_thinking_enabled()     — enable_thinking=false vs true
    |     '-- R3: compare_reasoning_scope()      — reasoning+content vs reasoning+tools
    |           (only if supports_tool_calls)
    |
    |-- Phase 2: analyze_content(tmpl, reasoning)
    |     '-- C1: compares content-only vs tools output and content-only vs reasoning output
    |
    |-- Phase 3: analyze_tools(tmpl, jinja_caps, reasoning)
    |     (skipped entirely if !jinja_caps.supports_tool_calls)
    |     |
    |     |-- T1: analyze_tool_calls()           — no tools vs with tools; classifies format
    |     |         |-- JSON path → analyze_tool_call_format_json_native()
    |     |         '-- tag path → analyze_tool_call_format_non_json()
    |     |
    |     (if format != NONE and format != JSON_NATIVE:)
    |     |
    |     |-- T2: check_per_call_markers()       — 1 call vs 2 calls; moves section→per-call if needed
    |     |         (only if supports_parallel_tool_calls)
    |     |
    |     |-- T3: extract_function_markers()     — func_alpha vs func_beta; extracts name prefix/suffix/close
    |     |
    |     |-- T4: analyze_arguments()            — (TAG_WITH_TAGGED only)
    |     |         |-- A1: extract_argument_name_markers()   — arg_name_A vs arg_name_B
    |     |         '-- A2: extract_argument_value_markers()  — value "XXXX" vs "YYYY"
    |     |
    |     |-- T5: extract_argument_separator()   — 1 arg vs 2 args; finds separator between args
    |     |
    |     |-- T6: extract_args_markers()         — 0 args vs 1 arg; finds args container markers
    |     |
    |     '-- T7: extract_call_id_markers()      — call_id "call00001" vs "call99999"
    |
    |-- detect_assistant_start_marker() / detect_user_start_marker()
    |
    '-- collect_preserved_tokens()               — union of all non-empty markers
    |
    '-- apply workarounds()                      — post-hoc patches for edge-case templates
    |
    v
autoparser (analysis result)
    |
    v
autoparser::peg_generator::generate_parser(tmpl, inputs, analysis)
    |-- analysis.build_parser(inputs, generation_prompt) - builds PEG parser arena
    |     |-- reasoning.build_parser(ctx)        — reasoning parser (mode-dependent)
    |     |-- content.build_parser(ctx)          — content parser (mode-dependent)
    |     '-- tools.build_parser(ctx)            — tool parser (dispatches by tool_format)
    |           |-- build_tool_parser_json_native()
    |           |-- build_tool_parser_tag_json()
    |           '-- build_tool_parser_tag_tagged()
    |
    |-- Build GBNF grammar (json_schema set, or tools with a trigger marker or tool_choice=required)
    '-- Set grammar_triggers from section_start or per_call_start (lazy grammar only)
    |
    v
common_chat_params (prompt, parser, grammar, triggers, preserved_tokens)
```

## Entry Point

The auto-parser is invoked in [common/chat.cpp:1332-1366](../../common/chat.cpp#L1332-L1366) in `common_chat_templates_apply_jinja`. Templates that need a dedicated handler are matched first by source substrings in `common_chat_try_specialized_template()` ([common/chat.cpp:1091-1224](../../common/chat.cpp#L1091-L1224)); the handlers live in [common/parsers/](../../common/parsers/) (Ministral/Magistral Large 3, GPT-OSS, Muse Glimmer, Functionary v3.2, Kimi K2 Thinking, Kimi K3, Ling 3.0, Cohere2 MoE, LFM2/LFM2.5, GigaChatV3, MiniMax-M3, DeepSeek V3.2/V4, Gemma 4, MiniCPM5, Qwen3-Coder). The auto-parser handles everything else via `autoparser::analyze_template` + `peg_generator::generate_parser`.

## Algorithm Details

### Core Mechanism: Differential Comparison

All analysis phases use the same factorized comparison function declared in [common/chat-auto-parser-helpers.h:69](../../common/chat-auto-parser-helpers.h#L69):

```cpp
compare_variants(tmpl, params_A, params_modifier)
```

This creates variant B by applying a modifier lambda to a copy of `params_A`, renders both through the template, and computes a `diff_split` ([common/chat-auto-parser.h:30-39](../../common/chat-auto-parser.h#L30-L39)):

- `prefix` — common prefix between A and B
- `suffix` — common suffix between A and B
- `left` — unique to variant A
- `right` — unique to variant B

The diff is computed via `calculate_diff_split()`, which finds the longest-common-prefix and longest-common-suffix, then iteratively moves incomplete `<...>` or `[...]` markers from the prefix/suffix into left/right until stable (tag boundary fixing).

Text is segmentized into markers and non-marker fragments using `segmentize_markers()`, which splits on `<...>` and `[...]` boundaries.

### Phase 1: Reasoning Analysis

**R1 — `compare_reasoning_presence()`**: Compares assistant message with vs without a `reasoning_content` field.

- Searches `diff.right` (output with reasoning) for the reasoning content needle
- Uses PEG parsers to find surrounding markers:
  - If both pre/post markers found in `diff.right` → `TAG_BASED`
  - If both found but post marker only in the full output B → `TAG_BASED` (template forces markers; handled via prefill)
  - If only post marker found → `TAG_BASED` (delimiter-style, empty start)
- Sets `reasoning.start` and `reasoning.end`

**R2 — `compare_thinking_enabled()`**: Compares `enable_thinking=false` vs `true` with a generation prompt.

- Detects template-added reasoning markers: `enable_thinking=true` appends a non-empty marker → sets `reasoning.start`, mode = `TAG_BASED`
- Handles the reverse case (`enable_thinking=false` appends the marker instead): extracts both start (from the preceding segment) and end markers; mode = `TAG_BASED`
- The reasoning prefill (markers added by the template) is later extracted in `common_chat_templates_apply_jinja` and prepended to model output before parsing

**R3 — `compare_reasoning_scope()`**: Compares assistant message with reasoning+text-content vs reasoning+tool-calls.

- Only runs if `jinja_caps.supports_tool_calls`
- Detects `TOOLS_ONLY`: reasoning content present in B (with tools) but not in A (with text content)
- Extracts reasoning markers from the tool call output using PEG parsers

### Phase 2: Content Analysis

**C1**: Two comparisons in the `analyze_content` constructor:

- Comparison 1: content-only output vs tool-call output → `diff_tools`
- Comparison 2: content-only output vs reasoning+empty-content output → `diff_reasoning`

Classification logic:

- `PLAIN`: `diff_tools.left` equals the response string (content is the entire diff, no wrapper)
- `ALWAYS_WRAPPED`: markers found surrounding the content text in `pure_content` → extracts `start`/`end`

### Phase 3: Tool Call Analysis

**T1 — `analyze_tool_calls()`**: Compares no-tools vs with-tools output.

- Extracts the tool call section as `diff.right`
- Calls `analyze_tool_call_format()` which first strips reasoning markers from the haystack, then:
  - Calls `in_json_haystack()` for both function name and argument name needles
  - `in_json_haystack()` uses a PEG parser to check whether the needle appears in a JSON context (preceded by `{` or `:` with surrounding quotes)
  - If function name is in JSON → `JSON_NATIVE` → `analyze_tool_call_format_json_native()`
  - If function name not in JSON, arg name is in JSON → `TAG_WITH_JSON`
  - If neither in JSON → `TAG_WITH_TAGGED`
  - `analyze_tool_call_format_json_native()`: parses the JSON object, matches field values to needles to populate `name_field`, `args_field`, `id_field`, `gen_id_field`; detects `tools_array_wrapped`; extracts `section_start`/`section_end`
  - `analyze_tool_call_format_non_json()`: uses PEG parsers on the haystack to find up to two opening markers (section + per-call) then up to two closing markers

**T2 — `check_per_call_markers()`**: Compares 1 call vs 2 calls.

- Computes a secondary diff of the second call portion vs the common suffix
- If the second call content starts with `section_start` → the section marker is actually per-call → moves `section_start/end` to `per_call_start/end` and clears the section markers

**T3 — `extract_function_markers()`**: Compares function name `FUN_FIRST` vs `FUN_SECOND` (two different named functions).

- Finds where the function name appears in `diff.left`
- Extracts `function.name_prefix` from the common prefix up to the function marker, and `function.name_suffix` from after the name up to the next marker
- Extends `name_suffix` into `diff.suffix` (to the first marker for TAG_WITH_TAGGED; to the first `{` or `[` for TAG_WITH_JSON)
- Extracts `function.close` from after the last argument value up to the per-call/section end marker

**T4 — `analyze_arguments()`** (TAG_WITH_TAGGED only):

- **A1 `extract_argument_name_markers()`**: Compares `arg_name_A` vs `arg_name_B` (two different argument names).
  - Finds shared surrounding structure → `arguments.name_prefix`, `arguments.name_suffix`
- **A2 `extract_argument_value_markers()`**: Compares argument value `"XXXX"` vs `"YYYY"` (same arg, different value).
  - Finds markers surrounding the value → `arguments.value_prefix`, `arguments.value_suffix`

**T5 — `extract_argument_separator()`**: Compares 1 argument vs 2 arguments (same function).

- Uses `until_common_prefix(diff.right, ARG_FIRST, ARG_SECOND)` to find what separates the two argument blocks

**T6 — `extract_args_markers()`**: Compares 0 arguments vs 1 argument.

- Uses `until_common_prefix()` and `after_common_suffix()` with the empty and single-arg JSON strings as anchors to find container markers (`arguments.start`, `arguments.end`)

**T7 — `extract_call_id_markers()`**: Compares call IDs `"call00001"` vs `"call99999"`.

- Determines whether function name appears in `diff.prefix` or `diff.suffix` to classify position:
  - Function name in prefix only → `BETWEEN_FUNC_AND_ARGS` or `POST_ARGS` (further distinguished by where `{` appears)
  - Function name in suffix only → `PRE_FUNC_NAME`
- Extracts `call_id.prefix` and `call_id.suffix` markers around the call ID value
- Clears `per_call_end` if it incorrectly incorporated the call ID suffix

### Workarounds

A workaround array in `common/chat-diff-analyzer.cpp` applies post-hoc patches after analysis. Each workaround is a lambda that inspects the template source and overrides analysis results. Current workarounds:

1. **Old Qwen/DeepSeek thinking templates** - source contains `content.split('</think>')` but neither `reasoning_content` nor `<SPECIAL_12>`: sets `reasoning.mode = TAG_BASED` with `<think>`/`</think>` markers if no reasoning was detected
2. **Granite 3.3** — source contains specific "Write your thoughts" text: forces `TAG_BASED` reasoning with `<think>`/`</think>` and `WRAPPED_WITH_REASONING` content with `<response>`/`</response>`
3. **Cohere Command R+** — source contains `<|CHATBOT_TOKEN|>`: sets `ALWAYS_WRAPPED` content mode if no content start is already set
4. **Functionary 3.1** — source contains `set has_code_interpreter`: forces `PLAIN` content, specific `per_call_start/end`, clears preserved tokens to only keep Functionary-specific markers
5. **DeepSeek-R1-Distill-Qwen** — source contains `tool▁calls▁begin` markers: overrides tool section/per-call markers with the correct Unicode block characters
6. **Nemotron Nano v2** - source contains `<SPECIAL_10>`/`<SPECIAL_11>`/`<SPECIAL_12>` and `<TOOL_RESPONSE>`: forces `JSON_NATIVE` tools in an array wrapped by `<TOOLCALL>`/`</TOOLCALL>`, `TAG_BASED` reasoning, and the message start markers
7. **Fireworks Firefunction v2** - matches its system prompt line: sets the Llama 3 header user/assistant start markers
8. **Solar Open** - source contains `<|begin|>assistant<|think|><|end|>`: sets the assistant start marker
9. **Apriel 1.6** - matches its `[BEGIN FINAL RESPONSE]` check: sets the `<|begin_user|>`/`<|begin_assistant|>` start markers
10. **JSON name/parameters tool instruction** - source asks for `{"name": function name, ...}` output: sets `openai_wrapper_trigger` so the lazy grammar also triggers on the OpenAI function wrapper
11. **Laguna** - source contains `laguna_glm_thinking`: trims whitespace from reasoning and argument markers, tolerates whitespace between argument tags, and adds `</assistant>` as a literal stop
12. **Bailing V3** - source contains `Bailing V3 chat template`: trims the argument value suffix and tolerates whitespace between argument tags

### Parser Building

Each analyzer struct (`analyze_reasoning`, `analyze_content`, `analyze_tools`) implements `build_parser(parser_build_context&)`. They share a `parser_build_context` that carries the PEG builder, inference inputs, the pre-built reasoning parser, the `extracting_reasoning` flag, and pointers to the reasoning and content analyzers.

#### Reasoning Parser (`analyze_reasoning::build_parser`)

| Mode                                          | Parser                                                                    |
|-----------------------------------------------|---------------------------------------------------------------------------|
| Not extracting reasoning                      | `eps()`                                                                   |
| `TAG_BASED` or `TOOLS_ONLY` (non-empty start) | `optional(space() + optspace(start) + reasoning(until(end)) + optspace(end))` |
| `TAG_BASED` or `TOOLS_ONLY` (empty start)     | `optional(reasoning(until(end)) + optspace(end))` - delimiter-style       |

Note: The start marker is empty when the analyzer detected delimiter-style reasoning. `optspace()` makes leading/trailing whitespace inside a marker optional. Whitespace-only reasoning content (e.g. from a `<think></think>` prefill) is discarded by the mapper.

#### Content Parser (`analyze_content::build_parser`)

| Condition                              | Parser                                                                          |
|----------------------------------------|---------------------------------------------------------------------------------|
| `json_schema` present (selected in `autoparser::build_parser`) | `reasoning + space() + content(schema(json(), "response-format-schema", ...)) + end()`, optionally in a json code fence |
| Tools present (selected in `autoparser::build_parser`) | `analyze_tools::build_parser()`                                   |
| `ALWAYS_WRAPPED` with reasoning        | `reasoning + start + content(until(end)) + end + end()`                         |
| `ALWAYS_WRAPPED` without reasoning     | `content(until(start)) + start + content(until(end)) + end + end()`             |
| Default (PLAIN)                        | `reasoning + content(rest()) + end()`                                           |

#### Tool Parsers (`analyze_tools::build_parser`)

Dispatches by `format.mode`:

**`build_tool_parser_json_native()`**: Calls `p.standard_json_tools()` which internally dispatches to:

- `build_json_tools_function_is_key()` — function name is the JSON key: `{"get_weather": {...}}`
- `build_json_tools_nested_keys()` — nested: `{"function": {"name": "X", "arguments": {...}}}`
- `build_json_tools_flat_keys()` — flat: `{"name": "X", "arguments": {...}}`

Handles content wrappers, array wrapping (`tools_array_wrapped`), parallel calls, and `parameter_order`.

**`build_tool_parser_tag_json()`**: For each tool function:

```text
tool_open(name_prefix + tool_name(literal(name)) + name_suffix) +
    call_id_section +
    tool_args(schema(json(), tool_schema))
  [+ function.close if non-empty]
```

Wrapped in per-call markers (with optional parallel call repetition) then optionally in section markers.

**`build_tool_parser_tag_tagged()`**: For each tool function, builds one parser per argument:

- String types: `tool_arg_string_value(schema(until(value_suffix), ...))`
- JSON types: `tool_arg_json_value(schema(json(), ...))`
- Required args are plain; optional args wrapped in `optional()`
- Arguments joined with `space()` between consecutive parsers

For closing: uses `function.close` if present; otherwise uses `peek(per_call_end)` to avoid premature close during partial streaming; falls back to `tool_close(space())` to trigger mapper callbacks.

All three tool parsers return:

```text
reasoning + optional(content(until(trigger_marker))) + tool_calls + end()
```

`autoparser::build_parser()` prepends `p.prefix(generation_prompt, reasoning_start)` to the returned parser: a literal for any boilerplate prefix of the generation prompt (the portion before the reasoning start marker).

## Mapper

`common_chat_peg_mapper` maps PEG parse results (AST nodes) into `common_chat_msg` structures. Key design:

- **Buffered arguments**: Before `tool_name` is known, argument text goes to `args_buffer`; once the name is set, the buffer is flushed to `current_tool->arguments`
- **`args_target()`**: Returns a reference to whichever destination is currently active (buffer or tool args), eliminating branching
- **`closing_quote_pending`**: Tracks whether a closing `"` needs to be appended when a string argument value is finalized (for schema-declared string types in tagged format)
- **Whitespace-only reasoning**: Reasoning content that consists entirely of whitespace (e.g. from a `<think></think>` prefill) is cleared so the message shows no reasoning
- **Brace auto-closing**: At tool close, unclosed `{` braces are closed automatically

## Files

| File                                      | Purpose                                                                         |
|-------------------------------------------|---------------------------------------------------------------------------------|
| `common/chat-auto-parser.h`               | All analysis structs, enums, `autoparser`, `peg_generator`, `generation_params` |
| `common/chat-auto-parser-generator.cpp`   | Parser generator: `generate_parser()` and `build_parser()` methods              |
| `common/chat-diff-analyzer.cpp`           | Differential analysis implementation and workarounds                            |
| `common/chat-auto-parser-helpers.h/cpp`   | `calculate_diff_split()`, `segmentize_markers()`, `compare_variants()`,         |
|                                           | `until_common_prefix()`, `after_common_suffix()`, string helpers                |
| `common/chat-peg-parser.h/cpp`            | `common_chat_peg_builder`, `common_chat_peg_mapper`, and helpers                |
| `common/chat.cpp`                         | Entry point: `common_chat_templates_apply_jinja()`                              |
| `common/parsers/`                         | Specialized handlers, dispatched by `common_chat_try_specialized_template()`    |
| `tests/test-chat-auto-parser.cpp`         | Auto-parser unit tests; also a debug tool when given a template path            |
| `tests/test-chat-analysis.cpp`            | Template differential analysis debug tool                                       |

## Testing & Debugging

### Debug Tools

**Template Debugger**: `tests/test-chat-auto-parser.cpp`

- Usage: `./bin/test-chat-auto-parser path/to/template.jinja` (without a path, it runs the automated tests)
- Shows detected format, markers, generated parser, and GBNF grammar

**Template Analysis**: `tests/test-chat-analysis.cpp`

- Usage: `./bin/test-chat-analysis --template-file path/to/template.jinja` (without arguments, it runs on all templates from the test suite)

**Debug Logging**: Enable with `LLAMA_ARG_LOG_VERBOSITY=5` (or `-lv 5`); the debug tools above already log at maximum verbosity

- Shows detailed analysis steps, pattern extraction results, and generated parser structure

**PEG Test Builder**: Fluent API for creating test cases - see [tests/test-chat.cpp:1465-1589](../../tests/test-chat.cpp#L1465-L1589). Example usage:

```cpp
auto tst = peg_tester("models/templates/Template.jinja");
tst.test("input text")
   .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
   .tools({tool_json})
   .parallel_tool_calls(true)
   .enable_thinking(true)
   .expect(expected_message)
   .run();
```

### Tested Templates

The following templates have active tests in `tests/test-chat.cpp` (partial list; "Specialized" means the template is routed to a handler in `common/parsers/`):

| Template | Format | Notes |
| -------- | ------ | ----- |
| Ministral-3-14B-Reasoning | Reasoning | `[THINK]...[/THINK]` tags (specialized handler) |
| NVIDIA-Nemotron-3-Nano-30B | Specialized | Reasoning + tools (Qwen3-Coder handler) |
| CohereForAI Command-R7B | JSON_NATIVE | `<\|START_THINKING\|>`/`<\|START_RESPONSE\|>` markers |
| Google Gemma 2 2B | Content only | No tool support |
| Qwen-QwQ-32B | Reasoning | Forced-open thinking |
| NousResearch Hermes 2 Pro | JSON_NATIVE | `<tool_call>` wrapper |
| IBM Granite 3.3 | JSON_NATIVE | `<think></think>` + `<response></response>` |
| IBM Granite 4.0 / 4.1 | JSON_NATIVE | `<tool_call>` wrapper (separate templates, both tested) |
| ByteDance Seed-OSS | TAG_WITH_TAGGED | Custom `<seed:think>` and `<seed:tool_call>` tags |
| Qwen3-Coder | Specialized | XML-style tool format (dedicated handler) |
| DeepSeek V3.1 | JSON_NATIVE | Forced thinking mode |
| GLM-4.6 | TAG_WITH_TAGGED | `<tool_call>name\n<arg_key>...<arg_value>...` format |
| GLM-4.7-Flash | TAG_WITH_TAGGED | Updated GLM format |
| Kimi-K2-Thinking | Specialized | Reasoning + JSON tools (Kimi K2 handler) |
| Apertus-8B-Instruct | JSON_NATIVE | Function name as JSON key |
| MiniMax-M2 | TAG_WITH_JSON | XML invoke with JSON args |
| NVIDIA-Nemotron-Nano-v2 | JSON_NATIVE | `<TOOLCALL>` wrapper (nested) |
| CohereForAI Command-R Plus | JSON_NATIVE | Markdown code block format |
| Mistral-Nemo-Instruct-2407 | JSON_NATIVE | `[TOOL_CALLS]` wrapper with ID field |
| Functionary v3.1 | TAG_WITH_JSON | `<function=X>` format |
| Functionary v3.2 | Specialized | `>>>` recipient delimiter (dedicated handler) |
| Fireworks Firefunction v2 | TAG_WITH_JSON | Fireworks tool format |
| DeepSeek R1 Distill (Llama/Qwen) | Reasoning | Forced-open thinking |
| llama-cpp-deepseek-r1 | Reasoning | Forced-open thinking |
| Kimi-K2 / Kimi-K2-Instruct | Specialized | JSON tools with special markers (Kimi K2 handler) |
| Llama 3.1/3.2/3.3 | JSON_NATIVE | Standard Llama tool format |
| OpenAI GPT-OSS | Specialized | Channel-based (dedicated handler) |
| LLM-jp-4.1 | Specialized | GPT-OSS dialect (dedicated handler) |
| Apriel 1.5 | JSON_NATIVE | `<tool_calls>` wrapper with JSON array |
| Apriel 1.6 Thinker | Reasoning | Implicit reasoning start |
| Mistral Small 3.2 | JSON_NATIVE | `[TOOL_CALLS]func[ARGS]{...}` with call ID |
| Devstral | Specialized | `[TOOL_CALLS]func[ARGS]{...}` without call ID (Ministral 3 handler) |
| StepFun 3.5 Flash | Specialized | `<function=X><parameter=Y>` format (Qwen3-Coder handler) |
| Spark2.5 | TAG_WITH_TAGGED | `<tool_call>name<arg_key>...<arg_value>...` format |

## Adding Support for New Templates

To support a new template format:

1. **If it follows standard patterns** — The auto-parser should detect it automatically. Run `test-chat-auto-parser <template_path>` to verify markers are correctly extracted.
2. **If differential analysis extracts incorrect markers** — Add a workaround lambda to the `workarounds` vector in `common/chat-diff-analyzer.cpp`. Inspect the template source for a unique identifying substring.
3. **If it needs fundamentally different handling** - Add a dedicated handler in `common/parsers/` (listed in `common/parsers/sources.cmake`) and match it in `common_chat_try_specialized_template()` in `chat.cpp` (as done for GPT-OSS, Functionary v3.2, Ministral 3 and others).

## Edge Cases and Quirks

1. **Generation Prompt & Reasoning Prefill**: The generation prompt is extracted by diffing `add_generation_prompt=false` vs `true` in `common_chat_template_generation_prompt()`, so it contains exactly what the template appends - avoiding false positives from prior conversation turns.
2. **Per-Call vs Per-Section Markers**: Some templates wrap each tool call individually (`per_call_start/end`); others wrap the entire section (`section_start/end`). T2 (`check_per_call_markers()`) disambiguates by checking if the second call in a two-call output starts with the section marker.
3. **Tag Boundary Fixing**: `calculate_diff_split()` iteratively adjusts prefix/suffix boundaries to avoid splitting `<tag>` or `[marker]` tokens, ensuring clean extraction.
4. **Call ID Side Effects**: When a call ID is detected, `per_call_end` may have been incorrectly set to include the call ID suffix. T7 clears `per_call_end` in this case.
5. **Tool Analysis Gating**: `analyze_tools` is only constructed (and all tool analysis phases run) when `jinja_caps.supports_tool_calls` is true. Within tool analysis, `check_per_call_markers()` (T2) only runs if `jinja_caps.supports_parallel_tool_calls`.
6. **`analyze_arguments()` Gating**: Within tool analysis, A1 and A2 (argument name/value marker extraction) only run for `TAG_WITH_TAGGED` format. `extract_argument_separator()` and `extract_args_markers()` run for all non-`JSON_NATIVE` formats.
7. **Undetected Tool Format**: If `analyze_tools` concludes tool calling is supported but cannot determine the format, `build_parser()` logs an error and returns `eps()` (graceful degradation) rather than aborting.
