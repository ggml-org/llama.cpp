# Unified Auto-Parser Architecture

The auto-parser automatically analyzes chat templates to determine how to parse model outputs, including content, reasoning, and tool calls.

## Overview

The unified auto-parser uses a **pure differential, compositional approach** to analyze chat templates:

**Core Philosophy**:

- **Zero Hardcoded Patterns**: All markers extracted through template comparison (the **only heuristic** is JSON detection)
- **Compositional Architecture**: Separate parsers for reasoning, content, and tools that compose cleanly

**Four-Phase Analysis**:

1. **Phase 1: Reasoning Analysis** (R1-R3) - Detects reasoning markers and mode
2. **Phase 2: Content Analysis** (C1) - Detects content wrapping markers
3. **Phase 3: Tool Call Analysis** (T1-T7) - Extracts tool section, function, and call ID markers
4. **Phase 4: Argument Analysis** (A1-A2) - Extracts argument name/value markers (TAG_WITH_TAGGED only)

## Data Structures

### diff_analysis_result

The result of differential analysis contains all extracted markers and format classifications:

```cpp
struct diff_analysis_result {
    // Classification results
    reasoning_mode  reasoning = reasoning_mode::NONE;
    content_mode    content   = content_mode::PLAIN;
    tool_format     tools     = tool_format::NONE;

    // All extracted markers (see marker_registry below)
    marker_registry markers;

    // JSON field names (for JSON_NATIVE format)
    bool        fun_name_is_key = false;   // Function name is the JSON key: {"func_name": {...}}
    std::string function_field  = "function";  // Outer object key (e.g., "function" in "function.name")
    std::string name_field      = "name";
    std::string args_field      = "arguments";
    std::string id_field;                  // String call ID field (e.g., "id")
    std::string gen_id_field;              // Generated integer call ID field (e.g., "tool_call_id")
    std::vector<std::string> parameter_order;  // Order of JSON fields for parsing

    // Call ID position (for non-JSON formats)
    call_id_position call_id_pos = call_id_position::NONE;

    // Flags
    bool supports_tools           = false;
    bool supports_parallel_calls  = false;
    bool requires_nonnull_content = false;
    bool tools_array_wrapped      = false;  // Tool calls wrapped in JSON array [...]

    // Preserved tokens for tokenizer (union of all non-empty markers)
    std::vector<std::string> preserved_tokens;
};
```

### Enums

**`reasoning_mode`**: How the template handles reasoning/thinking blocks.

| Value                | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| `NONE`               | No reasoning markers detected                                                 |
| `TAG_BASED`          | Standard tag-based: `<think>...</think>`                                      |
| `DELIMITER`          | Delimiter-based: reasoning ends at delimiter (e.g., `[BEGIN FINAL RESPONSE]`) |
| `FORCED_OPEN`        | Template ends with open reasoning tag (empty start, non-empty end)            |
| `FORCED_CLOSED`      | Both tags when disabled; only start tag when enabled                          |
| `TOOLS_ONLY`         | Reasoning only appears when tool calls are present                            |

**`content_mode`**: How the template wraps content.

| Value                      | Description                                          |
|----------------------------|------------------------------------------------------|
| `PLAIN`                    | No content markers                                   |
| `ALWAYS_WRAPPED`           | Content always wrapped: `<response>...</response>`   |
| `WRAPPED_WITH_REASONING`   | Content wrapped only when reasoning is present       |

**`tool_format`**: Classification of tool call structure.

| Value              | Description                                                      |
|--------------------|------------------------------------------------------------------|
| `NONE`             | No tool support detected                                         |
| `JSON_NATIVE`      | Pure JSON: `{"name": "X", "arguments": {...}}`                   |
| `TAG_WITH_JSON`    | Tag-based with JSON args: `<function=X>{...}</function>`         |
| `TAG_WITH_TAGGED`  | Tag-based with tagged args: `<param=key>value</param>`           |

**`call_id_position`**: Where call IDs appear relative to function name and arguments (for non-JSON formats).

| Value                      | Description                              |
|----------------------------|------------------------------------------|
| `NONE`                     | No call ID support detected              |
| `PRE_FUNC_NAME`            | Before function name                     |
| `BETWEEN_FUNC_AND_ARGS`    | Between function name and arguments      |
| `POST_ARGS`                | After arguments                          |

### marker_registry

All markers are extracted via differential analysis without hardcoded patterns:

```cpp
struct marker_registry {
    // === Reasoning markers (from R1-R3) ===
    std::string reasoning_start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
    std::string reasoning_end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"

    // === Content markers (from C1) ===
    std::string content_start;  // e.g., "<response>", ""
    std::string content_end;    // e.g., "</response>", ""

    // === Tool section markers (from T1-T2) ===
    std::string tool_section_start;  // e.g., "<tool_call>", "[TOOL_CALLS]"
    std::string tool_section_end;    // e.g., "</tool_call>", ""
    std::string per_call_start;      // e.g., "<|tool_call_begin|>" (for multi-call templates)
    std::string per_call_end;        // e.g., "<|tool_call_end|>"
    std::string call_separator;      // e.g., ",", "\n"

    // === Function markers (from T3-T6) ===
    std::string func_name_prefix;  // e.g., "<function=", "functions."
    std::string func_name_suffix;  // e.g., ">", ":0"
    std::string func_close;        // e.g., "</function>"
    std::string args_start;        // e.g., "{"
    std::string args_end;          // e.g., "}"

    // === Argument markers (from A1-A2, for TAG_WITH_TAGGED) ===
    std::string arg_name_prefix;   // e.g., "<param=", "<arg_key>"
    std::string arg_name_suffix;   // e.g., ">", "</arg_key>"
    std::string arg_value_prefix;  // e.g., "", "<arg_value>"
    std::string arg_value_suffix;  // e.g., "</param>", "</arg_value>"
    std::string arg_separator;     // e.g., "", "\n"

    // === Call ID markers (from T7) ===
    std::string call_id_prefix;       // e.g., "[CALL_ID]"
    std::string call_id_suffix;       // e.g., "[ARGS]"

    // === Special markers ===
    std::string code_block_marker;    // e.g., "Action:" (for markdown code block format)
    std::string code_block_language;  // e.g., "json"
    std::string function_namespace;   // e.g., "functions." (for prefixed-indexed format)
};
```

## Tool Calling Formats

The auto-parser recognizes three tool calling formats.

### JSON_NATIVE

**Structure**: The entire tool call (function name, arguments, and values) is in JSON format. There may be enclosing tags around the tool calling section.

**Characteristics**:

- Function name is a JSON field: `"name": "function_name"`
- Arguments are a JSON object: `"arguments": {"key": "value"}`
- May be wrapped in section markers like `<tool_call>...</tool_call>` or `[TOOL_CALLS]...]`

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

**Detection**: Function name found inside a JSON structure (determined by JSON parse attempt).

---

### TAG_WITH_JSON

**Structure**: The function name is outside the JSON structure, typically within quasi-XML markers. Arguments are still provided as a JSON object.

**Characteristics**:

- Function name appears in tag attributes: `<function=function_name>` or `<tool_name>function_name</tool_name>`
- Arguments are a JSON object following the tag
- Has closing tags: `</function>` or `</tool_call>`
- Arguments remain valid JSON

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

**Detection**: Function name not in JSON, but arguments are JSON (args_start is `{`).

---

### TAG_WITH_TAGGED

**Structure**: Both the function name AND argument names are in XML-style tags. Argument values may be JSON or unquoted primitives depending on schema type.

**Characteristics**:

- Function name in tag: `<function=name>` or `<invoke=name>`
- Each argument has its own tag: `<param=key>value</param>`
- String values are **unquoted** (raw text content of the tag)
- Non-string values (objects, arrays, numbers, booleans) are still JSON-formatted
- Supports streaming: partial arguments can be parsed incrementally

**Examples**:

Qwen/Hermes XML format:

```xml
<function=get_weather>
<param=location>Paris</param>
<param=unit>celsius</param>
</function>
```

Note how string values (`Paris`, `celsius`) are unquoted inside the tags.

Mixed types example:

```xml
<function=calculate>
<param=expr>2+2</param>
<param=precision>2</param>
<param=options>{"round": true}</param>
</function>
```

Here:

- `expr` and `precision` are strings (unquoted)
- `options` is an object (JSON-formatted inside the tag)

**Detection**: `arg_name_prefix` is non-empty, arguments use tagged format rather than JSON object.

---

## Analysis Flow

```text
Template
    |
    v
differential_analyzer::analyze(tmpl)
    |
    |-- Phase 1: analyze_reasoning(tmpl, result)
    |     |-- R1: compare_reasoning_presence() — with/without reasoning_content field
    |     |-- R2: compare_thinking_enabled() — enable_thinking=false vs true
    |     '-- R3: compare_reasoning_scope() — reasoning with content vs with tools
    |
    |-- Phase 2: analyze_content(tmpl, result)
    |     '-- C1: compare_content_values() — content vs tools vs reasoning
    |
    |-- Phase 3: analyze_tools(tmpl, result)
    |     |-- T1: analyze_tool_calls() — no tools vs with tools + format classification
    |     |-- T2: check_per_call_markers() — per-section vs per-call markers
    |     |-- T3: extract_call_separator() — separator between multiple calls
    |     |-- T4: extract_function_markers() — func_alpha vs func_beta
    |     |-- T5: extract_argument_separator() — 1 arg vs 2 args
    |     |-- T6: extract_args_markers() — no args vs with args
    |     '-- T7: extract_call_id_markers() — call_id "call00001" vs "call99999"
    |
    |-- Phase 4: analyze_arguments(tmpl, result)  [TAG_WITH_TAGGED only]
    |     |-- A1: extract_argument_name_markers() — "first" arg vs "second" arg
    |     '-- A2: extract_argument_value_markers() — value "XXXX" vs "YYYY"
    |
    '-- collect_preserved_tokens(result)
    |
    v
diff_analysis_result
    |
    v
universal_peg_generator::generate_parser(tmpl, inputs, analysis)
    |-- build_parser(analysis, inputs, ...)  — builds PEG parser arena
    |     |-- Reasoning parser (based on reasoning_mode)
    |     |-- Content parser (based on content_mode)
    |     '-- Tool parser (dispatches by tool_format):
    |           |-- build_tool_parser_json_native()
    |           |-- build_tool_parser_tag_json()
    |           '-- build_tool_parser_tag_tagged()
    |
    |-- Build GBNF grammar (if tools present)
    '-- Set grammar triggers from tool markers
    |
    v
common_chat_params (prompt, parser, grammar, triggers, preserved_tokens)
```

## Entry Point

The auto-parser is invoked in `common/chat.cpp` in `common_chat_templates_apply_jinja`. A few specialized templates are handled first (Ministral/Magistral Large 3, GPT-OSS, Functionary v3.2), then the auto-parser handles everything else:

```cpp
try {
    LOG_DBG("Using differential autoparser\n");
    auto auto_params = universal_peg_generator::generate_parser(tmpl, params);
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

This creates variant B by applying a modifier lambda to a copy of params_A, renders both through the template, and computes a `diff_split`:

```cpp
struct diff_split {
    std::string prefix;   // Common prefix between A and B
    std::string suffix;   // Common suffix between A and B
    std::string left;     // Unique to variant A
    std::string right;    // Unique to variant B
};
```

The diff is computed via `calculate_diff_split()`, which uses longest-common-prefix/suffix with iterative tag boundary fixing — it moves incomplete `<...>` or `[...]` markers from prefix/suffix into the left/right parts until stable.

Text is segmentized into markers and non-marker fragments using `segmentize_markers()`, which splits on `<...>` and `[...]` boundaries.

### Phase 1: Reasoning Analysis

Three comparisons extract reasoning markers and classify the reasoning mode:

**R1 — `compare_reasoning_presence()`**: Compares assistant message with vs without a `reasoning_content` field.

- Segmentizes `diff.right` to find markers around the reasoning content
- 3+ segments → `TAG_BASED` (start marker, content, end marker)
- 2 segments → `DELIMITER` (content followed by delimiter)
- Special case: markers found in prefix/suffix → `FORCED_CLOSED`

**R2 — `compare_thinking_enabled()`**: Compares `enable_thinking=false` vs `true`.

- Detects `FORCED_OPEN`: template adds opening tag when thinking enabled
- Detects `FORCED_CLOSED`: disable mode has both markers, enable mode has only start
- Handles reverse patterns (e.g., GLM-4.6 where disabled adds empty block)

**R3 — `compare_reasoning_scope()`**: Compares reasoning with content vs with tool calls.

- Detects `TOOLS_ONLY`: reasoning appears only when tool calls are present
- Extracts reasoning markers from tool call output by segmentizing

### Phase 2: Content Analysis

**C1 — `compare_content_values()`**: Compares content-only output vs tools output vs reasoning output.

- Creates two comparisons: content→tools and content→reasoning
- Finds content text position in diff to extract surrounding markers
- Classifies:
  - `ALWAYS_WRAPPED`: content has start/end markers in both comparisons
  - `WRAPPED_WITH_REASONING`: markers only when reasoning is present
  - `PLAIN`: no wrapping markers detected

### Phase 3: Tool Call Analysis

**T1 — `analyze_tool_calls()`**: Compares no-tools vs with-tools output.

- Calls `analyze_tool_call_format()` to classify the format using the **only heuristic**: a JSON parse attempt
  - `in_json_haystack()` checks whether the function name appears inside a JSON structure
  - If function name is in JSON → `JSON_NATIVE` → `analyze_tool_call_format_json_native()`:
    - Parses JSON structure, matches needle values to extract field names
    - Detects `fun_name_is_key`, `function_field`, `name_field`, `args_field`, `id_field`, `gen_id_field`
    - Detects `tools_array_wrapped` by checking for `[` before JSON
    - Builds `parameter_order` by sorting fields by position
    - Extracts `tool_section_start`/`tool_section_end`
  - If function name is not in JSON → `analyze_tool_call_format_non_json()`:
    - Segmentizes the haystack into markers and text
    - Uses symmetry: counts opening markers, matches with closing markers
    - Extracts `tool_section_start`, `tool_section_end`, `per_call_start`, `per_call_end`

**T2 — `check_per_call_markers()`**: Compares 1 call vs 2 calls.

- If the second call starts with `tool_section_start`, markers are per-call not per-section
- Moves tool_section markers to per_call markers, clears section markers

**T3 — `extract_call_separator()`**: Compares 1 call vs 2 calls.

- Finds separator between calls using `until_common_prefix(diff.right, ...)` with the two function names as anchors

**T4 — `extract_function_markers()`**: Compares function name "foofoo" vs "barbar".

- Finds function name in diff, segmentizes to extract prefix/suffix markers
- Extracts `func_name_prefix`, `func_name_suffix`
- Searches for closing marker after args to extract `func_close`

**T5 — `extract_argument_separator()`**: Compares 1 argument vs 2 arguments.

- Uses `until_common_prefix()` with argument names as anchors to find the separator

**T6 — `extract_args_markers()`**: Compares 0 arguments vs 1 argument.

- Uses `until_common_prefix()` and `after_common_suffix()` to find container markers
- Extracts `args_start`, `args_end`

**T7 — `extract_call_id_markers()`**: Compares call IDs "call00001" vs "call99999".

- Determines position relative to function name and arguments
- Classifies as `PRE_FUNC_NAME`, `BETWEEN_FUNC_AND_ARGS`, or `POST_ARGS`
- Extracts `call_id_prefix`, `call_id_suffix`

### Phase 4: Argument Analysis (TAG_WITH_TAGGED only)

Only runs when Phase 3 detected TAG_WITH_TAGGED or TAG_WITH_JSON format with non-JSON argument structures.

**A1 — `extract_argument_name_markers()`**: Compares argument name "first" vs "second".

- Finds common prefix of diff.left/right to extract marker structure
- Extracts `arg_name_prefix`, `arg_name_suffix`

**A2 — `extract_argument_value_markers()`**: Compares value "XXXX" vs "YYYY".

- Segmentizes prefix/suffix around value to find markers
- Extracts `arg_value_prefix`, `arg_value_suffix`

### Parser Building

The parser generator (`universal_peg_generator`) takes the analysis result and builds a PEG parser arena. The entry point is `generate_parser(tmpl, inputs)`, which:

1. Runs `differential_analyzer::analyze(tmpl)` to get the analysis result
2. Calls `build_parser(analysis, inputs, ...)` to construct the PEG parser
3. Builds a GBNF grammar if tools are present (for constrained decoding)
4. Sets grammar triggers from `tool_section_start` or `per_call_start`

#### Reasoning Parser Construction

Built inline in `build_parser()` based on `reasoning_mode`:

| Mode                              | Parser                                                                                      |
|-----------------------------------|---------------------------------------------------------------------------------------------|
| `FORCED_OPEN` / `FORCED_CLOSED`   | `reasoning(until(end)) + end` — expects reasoning immediately (opening tag was in template) |
| `TAG_BASED` / `TOOLS_ONLY`        | `optional(start + reasoning(until(end)) + end)`                                             |
| `DELIMITER`                       | `optional(reasoning(until(end)) + end)` — no start marker, reasoning ends at delimiter      |

#### Content Parser Construction

| Condition                          | Parser                                                                    |
|------------------------------------|---------------------------------------------------------------------------|
| `json_schema` present              | `reasoning + space() + content(schema(json(), ...)) + end()`              |
| Tools present                      | Dispatches to tool parser builder                                         |
| `ALWAYS_WRAPPED` with reasoning    | `reasoning + start + content(until(end)) + end + end()`                   |
| `ALWAYS_WRAPPED` without reasoning | `content(until(start)) + start + content(until(end)) + end + end()`       |
| Default                            | `reasoning + content(rest()) + end()`                                     |

#### Tool Parser Construction

`build_tool_parser()` dispatches by `tool_format`:

**`build_tool_parser_json_native()`**: Uses the `standard_json_tools()` builder helper which has three internal modes:

- `build_json_tools_function_is_key()` — function name is the JSON key: `{"get_weather": {"location": "Paris"}}`
- `build_json_tools_nested_keys()` — nested object: `{"function": {"name": "X", "arguments": {...}}}`
- `build_json_tools_flat_keys()` — flat object: `{"name": "X", "arguments": {...}}`

Handles content wrappers, array wrapping, parallel calls, and section markers.

**`build_tool_parser_tag_json()`**: For each tool, builds:

```text
tool_open(prefix + tool_name(literal(name)) + suffix) +
  call_id_section +
  tool_args(schema(json(), tool_schema))
```

Wraps in per-call or section markers. Handles parallel calls.

**`build_tool_parser_tag_tagged()`**: For each tool, builds per-argument parsers:

- String types: `tool_arg_string_value(schema(until(suffix), ...))`
- JSON types: `tool_arg_json_value(schema(json(), ...))`
- Required vs optional arguments
- Arguments joined with `space()` between them

Handles `func_close`, `peek()` for partial parsing safety, and call_id sections.

All three return: `reasoning + optional(content(until(trigger))) + tool_calls + end()`

### Mapper

The `common_chat_peg_unified_mapper` maps PEG parse results (AST nodes) into `common_chat_msg` structures. Key design:

- **Buffered arguments**: Before `tool_name` is known, argument text goes to `args_buffer`; once name is set, the buffer is flushed to `current_tool->arguments`
- **`args_target()`**: Returns a reference to whichever destination is active, eliminating branching
- **`closing_quote_pending`**: Tracks whether a closing `"` needs to be appended when a string argument value is finalized
- **Quote normalization**: Python-style quotes (`'key': 'value'`) are converted to JSON (`"key": "value"`)
- **Brace auto-closing**: At tool close, unclosed `{` braces are closed automatically (tracked via `json_brace_depth()`)

## Files

| File                                      | Purpose                                                           |
|-------------------------------------------|-------------------------------------------------------------------|
| `common/chat-auto-parser.h`               | `universal_peg_generator` class and `templates_params` struct     |
| `common/chat-auto-parser-generator.cpp`   | Parser generator implementation                                   |
| `common/chat-diff-analyzer.h`             | Analysis result types, enums, and `differential_analyzer` class   |
| `common/chat-diff-analyzer.cpp`           | Differential analysis implementation                              |
| `common/chat-auto-parser-helpers.h/cpp`   | `calculate_diff_split()`, `segmentize_markers()`, string helpers  |
| `common/chat-peg-parser.h/cpp`            | PEG builder and mapper classes                                    |
| `common/chat.cpp`                         | Entry point: `common_chat_templates_apply_jinja()`                |
| `tools/parser/debug-template-parser.cpp`  | Debug tool for template analysis                                  |
| `tools/parser/template-analysis.cpp`      | Template analysis tool                                            |

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
| Ministral-3-14B-Reasoning | Reasoning | `[THINK]...[/THINK]` tags |
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

1. **If it follows standard patterns** - The auto-parser should detect it automatically using the three formats (JSON_NATIVE, TAG_WITH_JSON, TAG_WITH_TAGGED)
2. **If differential analysis doesn't extract markers correctly** - Add a workaround in the workarounds array in `chat-diff-analyzer.cpp`
3. **If it needs fundamentally different handling** - Add a dedicated handler in `chat.cpp` before the auto-parser block (as done for GPT-OSS, Functionary v3.2, and Ministral)

## Edge Cases and Quirks

1. **Forced Thinking**: If `enable_thinking` is true but the model has already started a thought block (e.g., ended the prompt with `<think>`), the parser enters "forced thinking" mode where it immediately expects reasoning content.
2. **Per-Call vs Per-Section Markers**: Some templates wrap each tool call individually (`per_call_start`/`per_call_end`), others wrap the entire tool section (`tool_section_start`/`tool_section_end`). T2 disambiguates by checking if the second call in a two-call output starts with the section marker.
3. **Double Wrapping**: Some templates (e.g., Functionary) use the same string for both the tool section start and the function prefix (e.g., `<function=`). The analyzer detects this overlap and prevents double-wrapping in the generated parser.
4. **Null Content Rendering**: Some templates render `null` content as Python "None" string. The analyzer detects this and patches content to empty string.
5. **Tag Boundary Fixing**: The `calculate_diff_split()` function iteratively adjusts the prefix/suffix boundary to avoid splitting `<tag>` or `[marker]` tokens, ensuring clean extraction.
6. **Workarounds**: A workaround array in `chat-diff-analyzer.cpp` applies post-analysis patches for templates whose differential analysis produces incomplete or incorrect results (e.g., old Qwen thinking, Granite 3.3, Cohere Command-R+, Functionary, DeepSeek-R1-Distill-Qwen).
