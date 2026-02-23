# Unified Auto-Parser Architecture

The auto-parser automatically analyzes chat templates to determine how to parse model outputs, including content, reasoning, and tool calls.

## Overview

The unified auto-parser uses a pure differential (inspired by `git diff` algorithm), compositional approach to analyze chat templates:

**Core Philosophy**:

- **Zero Hardcoded Patterns**: All markers extracted through template comparison (the **only heuristic** is JSON detection to distinguish `JSON_NATIVE` from tag-based formats)
- **Compositional Architecture**: Separate analyzer structs for reasoning, content, and tools — each responsible for its own analysis and parser construction

**Analysis + Parser Building in Two Steps**:

1. `autoparser::analyze_template tmpl_analysis(tmpl)` — runs all differential comparisons and populates the analysis structs
2. `autoparser::universal_peg_generator::generate_parser(tmpl, params, tmpl_analysis)` — uses the analysis to build a PEG parser and optional GBNF grammar

## Data Structures

### Top-Level: `analyze_template`

```cpp
struct analyze_template {
    jinja::caps        jinja_caps;      // Template capability flags (supports_tools, supports_parallel_tool_calls, etc.)
    analyze_reasoning  reasoning;       // Phase 1 results
    analyze_content    content;         // Phase 2 results
    analyze_tools      tools;           // Phase 3 results (only populated if jinja_caps.supports_tool_calls)

    std::vector<std::string> preserved_tokens;  // Union of all non-empty markers
};
```

### `analyze_reasoning`

```cpp
struct analyze_reasoning : analyze_base {
    reasoning_mode mode = reasoning_mode::NONE;
    std::string start;  // e.g., "<think>", ""
    std::string end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]"
};
```

### `analyze_content`

```cpp
struct analyze_content : analyze_base {
    content_mode mode = content_mode::PLAIN;
    std::string start;  // e.g., "<response>", ""
    std::string end;    // e.g., "</response>", ""
    bool requires_nonnull_content = false;
};
```

### `analyze_tools` and its sub-structs

```cpp
struct tool_format_analysis {
    tool_format mode = tool_format::NONE;
    std::string section_start;   // e.g., "<tool_call>", "[TOOL_CALLS]"
    std::string section_end;     // e.g., "</tool_call>", ""
    std::string per_call_start;  // e.g., "<|tool_call_begin|>" (for multi-call templates)
    std::string per_call_end;    // e.g., "<|tool_call_end|>"

    bool fun_name_is_key    = false;  // JSON: function name is the object key
    bool tools_array_wrapped = false; // Tool calls wrapped in a JSON array [...]
    bool uses_python_dicts  = false;  // Single-quoted strings in JSON structures

    std::string              function_field = "function";
    std::string              name_field     = "name";
    std::string              args_field     = "arguments";
    std::string              id_field;       // String call ID (e.g., "id")
    std::string              gen_id_field;   // Generated integer call ID (e.g., "tool_call_id")
    std::vector<std::string> parameter_order;
};

struct tool_function_analysis {
    std::string name_prefix;  // e.g., "<function=", ""
    std::string name_suffix;  // e.g., ">", ""
    std::string close;        // e.g., "</function>", "```"
};

struct tool_arguments_analysis {
    std::string start;         // e.g., "<|tool_call_argument_begin|>"
    std::string end;           // e.g., "<|tool_call_argument_end|>"
    std::string name_prefix;   // e.g., "<param=", "<arg_key>"
    std::string name_suffix;   // e.g., ">", "</arg_key>"
    std::string value_prefix;  // e.g., "", "<arg_value>"
    std::string value_suffix;  // e.g., "</param>", "</arg_value>"
    std::string separator;     // e.g., "", "\n"
};

struct tool_id_analysis {
    call_id_position pos = call_id_position::NONE;
    std::string prefix;  // Marker before call ID value
    std::string suffix;  // Marker after call ID value
};

struct analyze_tools : analyze_base {
    tool_format_analysis    format;
    tool_function_analysis  function;
    tool_arguments_analysis arguments;
    tool_id_analysis        call_id;
};
```

### Enums

**`reasoning_mode`**: How the template handles reasoning/thinking blocks.

| Value           | Description                                                                       |
|-----------------|-----------------------------------------------------------------------------------|
| `NONE`          | No reasoning markers detected                                                     |
| `TAG_BASED`     | Standard tag-based: `<think>...</think>`                                          |
| `DELIMITER`     | Delimiter-based: reasoning ends at a delimiter (e.g., `[BEGIN FINAL RESPONSE]`)   |
| `FORCED_OPEN`   | Template ends with open reasoning tag when `enable_thinking=true`                 |
| `FORCED_CLOSED` | `enable_thinking=false` emits both tags; `enable_thinking=true` emits only start  |
| `TOOLS_ONLY`    | Reasoning only appears in tool call responses, not plain content                  |

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
    '-- collect_preserved_tokens()               — union of all non-empty markers
    |
    '-- apply workarounds()                      — post-hoc patches for edge-case templates
    |
    v
analyze_template (analysis result)
    |
    v
autoparser::universal_peg_generator::generate_parser(tmpl, inputs, analysis)
    |-- analysis.build_parser(inputs)            — builds PEG parser arena
    |     |-- reasoning.build_parser(ctx)        — reasoning parser (mode-dependent)
    |     |-- content.build_parser(ctx)          — content parser (mode-dependent)
    |     '-- tools.build_parser(ctx)            — tool parser (dispatches by tool_format)
    |           |-- build_tool_parser_json_native()
    |           |-- build_tool_parser_tag_json()
    |           '-- build_tool_parser_tag_tagged()
    |
    |-- Build GBNF grammar (if tools present and trigger_marker non-empty)
    '-- Set grammar_triggers from section_start or per_call_start
    |
    v
common_chat_params (prompt, parser, grammar, triggers, preserved_tokens)
```

## Entry Point

The auto-parser is invoked in `common/chat.cpp` in `common_chat_templates_apply_jinja`. A few specialized templates are handled first, then the auto-parser handles everything else:

```cpp
// Ministral/Magistral Large 3 — uses special reasoning structure, can't use autoparser
if (src.find("[SYSTEM_PROMPT]") != std::string::npos &&
    src.find("[TOOL_CALLS]") != std::string::npos &&
    src.find("[ARGS]") != std::string::npos &&
    src.find("[CALL_ID]") == std::string::npos) {
    return common_chat_params_init_ministral_3(tmpl, params);
}

// GPT-OSS — channel-based format
if (src.find("###{role}") != std::string::npos) { ... }

// Functionary v3.2 — recipient-based format (>>>recipient\n{content})
if (src.find(">>>{recipient}") != std::string::npos) { ... }

// All other templates:
try {
    auto analysis = autoparser::analyze_template(tmpl);
    auto auto_params = autoparser::universal_peg_generator::generate_parser(tmpl, params, analysis);
    auto_params.supports_thinking = analysis.reasoning.mode != autoparser::reasoning_mode::NONE;
    return auto_params;
} catch (const std::exception & e) {
    LOG_WRN("Automatic parser generation failed: %s\n", e.what());
}
```

## Algorithm Details

### Core Mechanism: Differential Comparison

All analysis phases use the same factorized comparison function:

```cpp
compare_variants(tmpl, params_A, params_modifier)
```

This creates variant B by applying a modifier lambda to a copy of `params_A`, renders both through the template, and computes a `diff_split`:

```cpp
struct diff_split {
    std::string prefix;   // Common prefix between A and B
    std::string suffix;   // Common suffix between A and B
    std::string left;     // Unique to variant A
    std::string right;    // Unique to variant B
};
```

The diff is computed via `calculate_diff_split()`, which finds the longest-common-prefix and longest-common-suffix, then iteratively moves incomplete `<...>` or `[...]` markers from the prefix/suffix into left/right until stable (tag boundary fixing).

Text is segmentized into markers and non-marker fragments using `segmentize_markers()`, which splits on `<...>` and `[...]` boundaries.

### Phase 1: Reasoning Analysis

**R1 — `compare_reasoning_presence()`**: Compares assistant message with vs without a `reasoning_content` field.

- Searches `diff.right` (output with reasoning) for the reasoning content needle
- Uses PEG parsers to find surrounding markers:
  - If both pre/post markers found in `diff.right` → `TAG_BASED` (both tags visible in diff = no forced close)
  - If both found but post marker only in the full output B → `FORCED_CLOSED`
  - If only post marker found → `DELIMITER`
- Sets `reasoning.start` and `reasoning.end`

**R2 — `compare_thinking_enabled()`**: Compares `enable_thinking=false` vs `true` with a generation prompt.

- Detects `FORCED_OPEN`: `enable_thinking=true` adds a non-empty marker at the end of the prompt (where model will start generating) — sets `reasoning.start`, mode = `FORCED_OPEN`
- Detects `FORCED_CLOSED`: `enable_thinking=false` produces both start+end markers; `enable_thinking=true` produces only start marker
- Handles the reverse case: if both start and end are still empty, looks for a single-segment diff on each side to extract both markers

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

A workaround array in `chat-diff-analyzer.cpp` applies post-hoc patches after analysis. Each workaround is a lambda that inspects the template source and overrides analysis results. Current workarounds:

1. **Old Qwen/DeepSeek thinking templates** — source contains `content.split('</think>')`: sets `reasoning.mode = FORCED_OPEN` with `<think>`/`</think>` markers if no reasoning was detected
2. **Granite 3.3** — source contains specific "Write your thoughts" text: forces `TAG_BASED` reasoning with `<think>`/`</think>` and `WRAPPED_WITH_REASONING` content with `<response>`/`</response>`
3. **Cohere Command R+** — source contains `<|CHATBOT_TOKEN|>`: sets `ALWAYS_WRAPPED` content mode if no content start is already set
4. **Functionary 3.1** — source contains `set has_code_interpreter`: forces `PLAIN` content, specific `per_call_start/end`, clears preserved tokens to only keep Functionary-specific markers
5. **DeepSeek-R1-Distill-Qwen** — source contains `tool▁calls▁begin` markers: overrides tool section/per-call markers with the correct Unicode block characters

### Parser Building

Each analyzer struct (`analyze_reasoning`, `analyze_content`, `analyze_tools`) implements `build_parser(parser_build_context&)`. They share a `parser_build_context` that carries the PEG builder, inference inputs, the pre-built reasoning parser, and a pointer to the content analyzer.

#### Reasoning Parser (`analyze_reasoning::build_parser`)

| Mode                              | Parser                                                              |
|-----------------------------------|---------------------------------------------------------------------|
| Not extracting reasoning          | `eps()`                                                             |
| `FORCED_OPEN` or `FORCED_CLOSED`  | `reasoning(until(end)) + end` — opening tag was in the prompt       |
| `TAG_BASED` or `TOOLS_ONLY`       | `optional(start + reasoning(until(end)) + end)`                     |
| `DELIMITER`                       | `optional(reasoning(until(end)) + end)` — no start marker           |

#### Content Parser (`analyze_content::build_parser`)

| Condition                              | Parser                                                                          |
|----------------------------------------|---------------------------------------------------------------------------------|
| `json_schema` present                  | `reasoning + space() + content(schema(json(), "response-format", ...)) + end()` |
| Tools present                          | Dispatches to `analyze_tools::build_parser()`                                   |
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

### Python Dict Format

When `format.uses_python_dicts` is true (detected when single-quoted strings appear in JSON argument context), `build_parser()` pre-registers a `json-string` rule that accepts both single-quoted and double-quoted strings. This is done before any `p.json()` call so all JSON parsing inherits the flexible rule.

## Mapper

`common_chat_peg_mapper` maps PEG parse results (AST nodes) into `common_chat_msg` structures. Key design:

- **Buffered arguments**: Before `tool_name` is known, argument text goes to `args_buffer`; once the name is set, the buffer is flushed to `current_tool->arguments`
- **`args_target()`**: Returns a reference to whichever destination is currently active (buffer or tool args), eliminating branching
- **`closing_quote_pending`**: Tracks whether a closing `"` needs to be appended when a string argument value is finalized (for schema-declared string types in tagged format)
- **Quote normalization**: Python-style quotes (`'key': 'value'`) are converted to JSON (`"key": "value"`)
- **Brace auto-closing**: At tool close, unclosed `{` braces are closed automatically

## Files

| File                                      | Purpose                                                              |
|-------------------------------------------|----------------------------------------------------------------------|
| `common/chat-auto-parser.h`               | `universal_peg_generator` class and `templates_params` struct        |
| `common/chat-auto-parser-generator.cpp`   | Parser generator: `generate_parser()` and `build_parser()` methods   |
| `common/chat-diff-analyzer.h`             | All analysis structs, enums, and sub-structs                         |
| `common/chat-diff-analyzer.cpp`           | Differential analysis implementation and workarounds                 |
| `common/chat-auto-parser-helpers.h/cpp`   | `calculate_diff_split()`, `segmentize_markers()`,                    |
|                                           | `compare_variants()`, string helpers                                 |
| `common/chat-peg-parser.h/cpp`            | `common_chat_peg_builder`, `common_chat_peg_mapper`, and helpers     |
| `common/chat.cpp`                         | Entry point: `common_chat_templates_apply_jinja()`                   |
| `tools/parser/debug-template-parser.cpp`  | Debug tool for template analysis                                     |
| `tools/parser/template-analysis.cpp`      | Template analysis tool                                               |

## Testing & Debugging

### Debug Tools

**Template Debugger**: `tools/parser/debug-template-parser.cpp`

- Usage: `./bin/llama-debug-template-parser path/to/template.jinja`
- Shows detected format, markers, generated parser, and GBNF grammar

**Template Analysis**: `tools/parser/template-analysis.cpp`

- Usage: `./bin/llama-template-analysis path/to/template.jinja`

**Debug Logging**: Enable with `LLAMA_LOG_VERBOSITY=2`

- Shows detailed analysis steps, pattern extraction results, and generated parser structure

**PEG Test Builder**: Fluent API for creating test cases in `tests/test-chat.cpp`:

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

The following templates have active tests in `tests/test-chat.cpp`:

| Template | Format | Notes |
| -------- | ------ | ----- |
| Ministral-3-14B-Reasoning | Reasoning | `[THINK]...[/THINK]` tags (specialized handler) |
| NVIDIA-Nemotron-3-Nano-30B | TAG_WITH_TAGGED | Reasoning + tools |
| CohereForAI Command-R7B | JSON_NATIVE | `<\|START_THINKING\|>`/`<\|START_RESPONSE\|>` markers |
| Google Gemma 2 2B | Content only | No tool support |
| Qwen-QwQ-32B | Reasoning | Forced-open thinking |
| NousResearch Hermes 2 Pro | JSON_NATIVE | `<tool_call>` wrapper |
| IBM Granite 3.3 | JSON_NATIVE | `<think></think>` + `<response></response>` |
| ByteDance Seed-OSS | TAG_WITH_TAGGED | Custom `<seed:think>` and `<seed:tool_call>` tags |
| Qwen3-Coder | TAG_WITH_TAGGED | XML-style tool format |
| DeepSeek V3.1 | JSON_NATIVE | Forced thinking mode |
| GLM-4.6 | TAG_WITH_TAGGED | `<tool_call>name\n<arg_key>...<arg_value>...` format |
| GLM-4.7-Flash | TAG_WITH_TAGGED | Updated GLM format |
| Kimi-K2-Thinking | JSON_NATIVE | Reasoning + JSON tools |
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
| Kimi-K2 / Kimi-K2-Instruct | JSON_NATIVE | JSON tools with special markers |
| Llama 3.1/3.2/3.3 | JSON_NATIVE | Standard Llama tool format |
| OpenAI GPT-OSS | Specialized | Channel-based (dedicated handler) |
| Apriel 1.5 | JSON_NATIVE | `<tool_calls>` wrapper with JSON array |
| Apriel 1.6 Thinker | Reasoning | Implicit reasoning start |
| Mistral Small 3.2 | JSON_NATIVE | `[TOOL_CALLS]func[ARGS]{...}` with call ID |
| Devstral | JSON_NATIVE | `[TOOL_CALLS]func[ARGS]{...}` without call ID |
| StepFun 3.5 Flash | TAG_WITH_TAGGED | `<function=X><parameter=Y>` format |

## Adding Support for New Templates

To support a new template format:

1. **If it follows standard patterns** — The auto-parser should detect it automatically. Run `llama-debug-template-parser` to verify markers are correctly extracted.
2. **If differential analysis extracts incorrect markers** — Add a workaround lambda to the `workarounds` vector in `chat-diff-analyzer.cpp`. Inspect the template source for a unique identifying substring.
3. **If it needs fundamentally different handling** — Add a dedicated handler function in `chat.cpp` before the auto-parser block (as done for GPT-OSS, Functionary v3.2, and Ministral).

## Edge Cases and Quirks

1. **Forced Thinking**: When `enable_thinking=true` and the model prompt ends with an open reasoning tag (e.g., `<think>`), the parser enters forced thinking mode and immediately expects reasoning content without waiting for a start marker.
2. **Per-Call vs Per-Section Markers**: Some templates wrap each tool call individually (`per_call_start/end`); others wrap the entire section (`section_start/end`). T2 (`check_per_call_markers()`) disambiguates by checking if the second call in a two-call output starts with the section marker.
3. **Python Dict Format**: The Seed template family uses single-quoted JSON (`'key': 'value'`). The `uses_python_dicts` flag causes the PEG builder to register a flexible `json-string` rule accepting both quote styles before any JSON rules are built.
4. **Tag Boundary Fixing**: `calculate_diff_split()` iteratively adjusts prefix/suffix boundaries to avoid splitting `<tag>` or `[marker]` tokens, ensuring clean extraction.
5. **Call ID Side Effects**: When a call ID is detected, `per_call_end` may have been incorrectly set to include the call ID suffix. T7 clears `per_call_end` in this case.
6. **Tool Analysis Gating**: `analyze_tools` is only constructed (and all tool analysis phases run) when `jinja_caps.supports_tool_calls` is true. Within tool analysis, `check_per_call_markers()` (T2) only runs if `jinja_caps.supports_parallel_tool_calls`.
7. **`analyze_arguments()` Gating**: Within tool analysis, A1 and A2 (argument name/value marker extraction) only run for `TAG_WITH_TAGGED` format. `extract_argument_separator()` and `extract_args_markers()` run for all non-`JSON_NATIVE` formats.
