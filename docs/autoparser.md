# Unified Auto-Parser Architecture

The auto-parser automatically analyzes chat templates to determine how to parse model outputs, including content, reasoning, and tool calls.

## Overview

The unified auto-parser uses a **pure differential, compositional approach** to analyze chat templates:

**Core Philosophy**:

- **Zero Hardcoded Patterns**: All markers extracted through template comparison (the **only heuristic** is JSON detection)
- **Compositional Architecture**: Separate parsers for reasoning, content, and tools that compose cleanly
- **Variant Types**: Structural descriptions (strings) instead of forced enum classification

**Two-Phase Analysis**:

1. **Phase 1: Content & Reasoning Analysis** - Analyzes how the template handles basic content and reasoning, without considering tools
2. **Phase 2: Tool Call Analysis** - Analyzes tool calling patterns, layered on top of Phase 1

## Data Structures

### content_structure (Phase 1 Result)

Describes how the template handles content and reasoning:

```cpp
struct content_structure {
    enum reasoning_mode_type {
        REASONING_NONE,         // No reasoning markers detected
        REASONING_OPTIONAL,     // <think>...</think> may appear before content
        REASONING_FORCED_OPEN,  // Template ends with open reasoning tag OR starts implicitly (empty start, present end)
    };

    reasoning_mode_type reasoning_mode = REASONING_NONE;
    std::string         reasoning_start;  // e.g., "<think>", "<|START_THINKING|>"
    std::string         reasoning_end;    // e.g., "</think>", "<|END_THINKING|>"

    // Content wrapping mode
    enum content_mode_type {
        CONTENT_PLAIN,                   // No content markers
        CONTENT_ALWAYS_WRAPPED,          // <response>...</response> always present
        CONTENT_WRAPPED_WITH_REASONING,  // Content wrapped only when reasoning present
    };

    content_mode_type content_mode = CONTENT_PLAIN;
    std::string       content_start;  // e.g., "<response>", "<|START_RESPONSE|>"
    std::string       content_end;    // e.g., "</response>", "<|END_RESPONSE|>"
};
```

### diff_analysis_result (Analysis Result)

The result of differential analysis contains all extracted markers and format classifications:

```cpp
struct diff_analysis_result {
    // Classification results
    reasoning_mode  reasoning = reasoning_mode::NONE;
    content_mode    content   = content_mode::PLAIN;
    tool_format     tools     = tool_format::NONE;
    argument_format args      = argument_format::JSON;

    // All extracted markers (see marker_registry below)
    marker_registry markers;

    // JSON field names (for JSON-based formats)
    std::string name_field = "name";
    std::string args_field = "arguments";
    std::string id_field;

    // Flags
    bool supports_tools           = false;
    bool supports_parallel_calls  = false;
    bool requires_nonnull_content = false;

    // Preserved tokens for tokenizer
    std::vector<std::string> preserved_tokens;
};
```

### marker_registry (Extracted Markers)

All markers are extracted via differential analysis without hardcoded patterns:

```cpp
struct marker_registry {
    // === Reasoning markers ===
    std::string reasoning_start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>"
    std::string reasoning_end;    // e.g., "</think>", "[/THINK]", "<|END_THINKING|>"

    // === Content markers ===
    std::string content_start;  // e.g., "<response>", ">>>all\n"
    std::string content_end;    // e.g., "</response>"

    // === Tool section markers ===
    std::string tool_section_start;  // e.g., "<tool_call>", "[TOOL_CALLS]"
    std::string tool_section_end;    // e.g., "</tool_call>", "]"
    std::string per_call_start;      // e.g., "\u2985" (for multi-call templates)
    std::string per_call_end;        // e.g., " \u2985"
    std::string call_separator;      // e.g., ",", "\n"

    // === Function markers ===
    std::string func_name_prefix;  // e.g., "<function=", "\"name\": \""
    std::string func_name_suffix;  // e.g., ">", "\""
    std::string func_close;        // e.g., "</function>"
    std::string args_start;        // e.g., "{", " \u300b"
    std::string args_end;          // e.g., "}", ""

    // === Argument markers (for tagged args format) ===
    std::string arg_name_prefix;   // e.g., "<param=", "<arg_key>"
    std::string arg_name_suffix;   // e.g., ">", "</arg_key>"
    std::string arg_value_prefix;  // e.g., "", "<arg_value>"
    std::string arg_value_suffix;  // e.g., "</param>", "</arg_value>"
    std::string arg_separator;

    // === Special markers ===
    std::string code_block_marker;    // e.g., "Action:" (markdown code block format)
    std::string id_marker;            // e.g., "[CALL_ID]" (bracket-tag format)
    std::string function_namespace;   // e.g., "functions." (prefixed-indexed format)
};
```

## Tool Calling Formats

The auto-parser recognizes three primary tool calling formats. Other formats may be deprecated in future versions.

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

Hermes-style with tool_calls wrapper:
```json
<tool_calls>
{"name": "search", "arguments": {"query": "llama.cpp"}}
</tool_calls>
```

**Detection**: `args_start == "{"`, `args_end == "}"`, no function name prefix markers

---

### TAG_WITH_JSON

**Structure**: The function name is outside the JSON structure, typically within quasi-XML markers. Arguments are still provided as a JSON object.

**Characteristics**:
- Function name appears in tag attributes: `<function=function_name>` or `<tool_call name="function_name">`
- Arguments are a JSON object following the tag
- Has closing tags: `</function>` or `</tool_call>`
- Arguments remain valid JSON

**Examples**:

Nemotron-style:
```xml
<TOOLCALL>get_weather{"location": "Paris"}</TOOLCALL>
```

Functionary v3.1:
```xml
<function=get_weather>{"location": "Paris", "unit": "celsius"}</function>
```

ByteDance Seed-OSS:
```xml
<seed:tool_call>
<tool_name>get_weather</tool_name>
<parameters>{"location": "Paris"}</parameters>
</seed:tool_call>
```

MiniMax:
```xml
<minimax:tool_call>
<tool_name>calculate</tool_name>
<arguments>{"expr": "2+2"}</arguments>
</minimax:tool_call>
```

**Detection**: `func_name_prefix` starts with `<`, `args_start == "{"`, arguments are JSON

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

**Detection**: `arg_name_prefix` is non-empty, arguments use tagged format rather than JSON object

---

### Other Formats (To Be Deprecated)

The following formats are currently supported but will likely be deprecated:

| Format | Description | Example |
|--------|-------------|---------|
| `BRACKET_TAG` | Bracket-based markers | `[TOOL_CALLS]func[ARGS]{...}` |
| `PREFIXED_INDEXED` | Namespace prefix with index | `functions.name:0{...}` |
| `RECIPIENT_BASED` | Recipient routing | `>>>recipient\n{content}` |
| `MARKDOWN_BLOCK` | Markdown code blocks | `Action:\n\`\`\`json\n[...]` |

## Analysis Flow

```console
Template
    |
    v
Phase 1: analyze_content_structure()
    |-- detect_reasoning_markers() - compare outputs with reasoning_content vs without
    |-- detect_content_markers() - render with content and detect wrapping
    |-- detect_reasoning_mode() - check if prompt ends with open tag
    |
    v
content_structure
    |
    v
Phase 2: analyze_tool_structure()
    |-- Check minja.supports_tool_calls
    |-- Differential analysis for tool patterns
    |-- Classify function format (JSON vs tagged)
    |-- Classify argument format (JSON vs tagged)
    |
    v
diff_analysis_result
    |
    v
generate_parser(diff_analysis_result)
    |-- build_reasoning_block(diff_analysis_result)
    |-- build_content_block(diff_analysis_result)
    |-- build_tool_section(diff_analysis_result, tools)
    |-- Compose into final parser
    |
    v
common_chat_params (parser, grammar, triggers, preserved_tokens)
```

## Entry Point

The mechanism starts in `common/chat.cpp`, in `common_chat_templates_apply_jinja`:

```cpp
// 1. Analyze the template (two-phase)
auto analysis = differential_analyzer::analyze(tmpl);

// 2. Generate the parser and grammar
auto auto_params = universal_peg_generator::generate_parser(tmpl, params);

// 3. Use if it provides more than basic content handling
if (auto_params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY ||
    !auto_params.parser.empty()) {
    return auto_params;
}
```

## Builder Methods

The unified builder (`common_chat_peg_unified_builder`) provides high-level methods:

- `build_reasoning_block(analysis, reasoning_format, thinking_forced_open)` - Build reasoning parser
- `build_content_block(analysis, reasoning_format)` - Build content parser
- `build_tool_section(analysis, tools, parallel_tool_calls, force_tool_calls)` - Build tool section
- `build_function(analysis, name, schema)` - Build single function parser
- `build_arguments(analysis, schema)` - Build arguments parser

## Key Templates Supported

- **Granite** - `<think></think>` + `<response></response>` with tool calls
- **Nemotron** - JSON tools with `<TOOLCALL>` wrapper
- **Qwen/Hermes** - XML-style `<function=X><param=key>` format (TAG_WITH_TAGGED)
- **Command-R7B** - `<|START_THINKING|>`/`<|START_RESPONSE|>` + `<|START_ACTION|>` tools
- **DeepSeek R1** - Forced thinking + complex tools
- **Mistral Nemo** - `[TOOL_CALLS]` wrapper (JSON_NATIVE)
- **MiniMax** - `<minimax:tool_call>` wrapper with JSON args (TAG_WITH_JSON)
- **GLM-4.6** - `<minimax:tool_call>` + `<tool_call>name\n<arg_key>...<arg_value>...` format
- **Kimi-K2** - `PREFIXED_INDEXED` format with namespace and indices
- **Mistral Small 3.2** - `BRACKET_TAG` format with `[TOOL_CALLS]` markers
- **Functionary v3.2** - `RECIPIENT_BASED` format with `>>>` routing

## Files

| File | Purpose |
|------|---------|
| `common/chat-auto-parser.h` | Data structures and API declarations |
| `common/chat-diff-analyzer.h/cpp` | Differential analysis implementation |
| `common/chat-auto-parser-generator.cpp` | PEG parser generator |
| `common/chat-auto-parser-helpers.h/cpp` | Shared helper functions |
| `common/chat-peg-parser.h/cpp` | Unified builder and mapper classes |
| `common/chat.cpp` | Main entry point and wire-up |

## Algorithm Details

### Phase 1: Content & Reasoning Analysis

#### Reasoning Detection (4 Methods)

**Method 1: Differential Reasoning Content Analysis**

- Render template with `reasoning_content` field present vs absent
- Compare outputs to find markers between reasoning and content
- If only closing tag found, derive opening tag using patterns:
  - XML: `</tag>` → `<tag>`
  - Special tokens: `<|END_X|>` → `<|START_X|>`, `<|/X|>` → `<|X|>`
- Handles various tag formats including XML and special token formats

**Method 2: Enable-Thinking Toggle Analysis**

- Toggle `enable_thinking` context variable between true/false
- Detects differences in generated prompts
- Handles two scenarios:
  - **Normal case**: enable_thinking=true adds reasoning markers
  - **Reverse case**: enable_thinking=false adds empty thinking block (GLM-4.6 style)
- Uses string difference analysis to extract markers
- Validates extracted tags against blacklist of role markers

**Method 3: Prompt Ending Analysis**

- Checks if prompt ends with unclosed reasoning tag
- Looks for trailing tags in prompt with `enable_thinking=true`
- Differentiates between open tags (`<think>`) and close tags (`</think>`)
- Handles blacklisted tags (role markers, system tokens)
- Validates reasoning-like patterns (contains "think", "reason", "thought")

**Method 4: Adjacent Tag Pair Detection**

- Looks for patterns like `<minimax:tool_call></think>`, `<|START_THINKING|><|END_THINKING|>`, `[think][/think]`
- Searches for predefined tag patterns in prompt
- Validates tags are adjacent with only whitespace between
- Supports both simple and complex token formats

#### Content Detection Algorithm

1. **Dual-Mode Rendering**: Render template with content marker in both thinking-enabled and thinking-disabled modes
2. **Pattern Matching**: Search for known content wrapper patterns:
   - `<|START_RESPONSE|>` / `<|END_RESPONSE|>`
   - `<response>` / `</response>`
   - `<output>` / `</output>`
   - `<answer>` / `</answer>`
   - `<|CHATBOT_TOKEN|>` / `<|END_OF_TURN_TOKEN|>`
3. **Mode Classification**:
   - `CONTENT_ALWAYS_WRAPPED`: Found in both thinking modes
   - `CONTENT_WRAPPED_WITH_REASONING`: Found only with thinking enabled
   - `CONTENT_PLAIN`: No wrapping detected

#### Reasoning Mode Detection

- **REASONING_FORCED_OPEN**:
  - **Explicit**: Prompt ends with reasoning start marker (e.g., `<think>`).
  - **Implicit**: reasoning end marker is present but start marker is empty (e.g., `[BEGIN FINAL RESPONSE]`).
- **REASONING_OPTIONAL**: Markers present but not forced.
- **REASONING_NONE**: No markers detected.

### Phase 2: Tool Call Structure Analysis

#### Pure Differential Analysis Algorithm

**Key Principle**: All patterns are extracted through template comparison. The **only heuristic** is detecting JSON vs marker-based structures (via JSON parse attempt). No hardcoded pattern lists.

**Comparison Matrix**:

| Comparison | Purpose | What's Extracted |
|------------|---------|------------------|
| **T1**: No tools vs tools | Tool section markers | `tool_section_start`, `tool_section_end` |
| **T2**: 1 call vs 2 calls | Call separators | `per_call_start`, `call_separator` |
| **T3**: func_alpha vs func_beta | Function boundaries | `func_name_prefix`, `func_name_suffix` |
| **T4**: 1 arg vs 2 args | Argument separator | `arg_separator` |
| **T5**: No args vs args | Args container | `args_start`, `args_end` |
| **A1**: key1 vs key2 | Arg name boundaries | `arg_name_prefix`, `arg_name_suffix` |
| **A2**: value A vs B | Arg value boundaries | `arg_value_prefix`, `arg_value_suffix` |
| **A3**: number vs string | Quoting behavior | Value type handling |

**Structural Extraction Helpers**:

```cpp
// Extract last structural marker from string (finds last <, [, {, or ")
std::string extract_structural_suffix(const std::string & str);

// Extract first structural marker from string (finds first >, ], }, or ")
std::string extract_structural_prefix(const std::string & str);

// The only heuristic: detect if content is valid JSON
bool is_json_based(const std::string & content);
```

**Pattern Extraction Process** (Example - T1: Tool Section Markers):

1. Render template with/without tool calls
2. Compute diff: `calculate_diff_split(output_no_tools, output_with_tools)`
3. Use controlled function name (`func_alpha`) as anchor in `diff.right`
4. Extract structural prefix before function name → `tool_section_start`
5. Extract structural suffix after tool content → `tool_section_end`

**No Pattern Lists**: Unlike the old approach, there are no hardcoded lists like `["<tool_call>", "[TOOL_CALLS]", ...]`. All markers are discovered through differential comparison.

#### Variant Detection Logic

Instead of forcing patterns into enum types, the analyzer detects **variant types** as strings that describe the structural characteristics:

**Variant Types**:

- `"json-native"`: Pure JSON tool calls (Llama, Mistral Nemo)
- `"tagged-json"`: Function name in markers, args in JSON (Functionary v3.1, Nemotron)
- `"tagged-args"`: Full XML-style with tagged arguments (Qwen, Hermes, MiniMax)
- `"bracket-tag"`: Bracket markers (Mistral Small 3.2: `[TOOL_CALLS]func[ARGS]{...}`)
- `"recipient-based"`: Recipient routing (Functionary v3.2: `>>>func_name`)
- `"markdown-block"`: Markdown code blocks (Cohere Command-R Plus)
- `"prefixed-indexed"`: Namespace prefix with indices (Kimi-K2: `functions.name:0`)

**Detection Strategy** (from most to least distinctive):

```cpp
void detect_tool_variant(diff_analysis_result & result) {
    // 1. Check for unique markers (most distinctive)
    if (!result.markers.id_marker.empty())
        → "bracket-tag"

    if (markers contain ">>>")
        → "recipient-based"

    if (code_block_marker present)
        → "markdown-block"

    if (function_namespace or suffix contains ':')
        → "prefixed-indexed"

    // 2. Check argument structure (JSON variants)
    if (arg_name_prefix starts with '<')
        → "tagged-args"

    if (func_name_prefix starts with '<')
        → "tagged-json"

    // 3. Default
    → "json-native"
}
```

#### Compositional Parser Building

The analyzer builds separate, composable parsers for each component:

**Reasoning Parser**:

- Built from `reasoning_start` and `reasoning_end` markers
- Supports tag-based, delimiter, and forced-open modes

**Content Parser**:

- Built from `content_start` and `content_end` markers
- Supports plain, always-wrapped, and conditionally-wrapped modes

**Tool Parser** (variant-specific):

- Built based on `variant_type` detection
- Each variant has its own builder that uses the extracted markers
- No enum forcing - structure preserved as discovered

**Final Composition**:

```cpp
sequence({
    reasoning_parser,
    space(),
    content_parser,
    space(),
    tool_parser,
    end()
})
```

### Generator Algorithms

#### Unified Parser Building

**Composition Strategy**:

```cpp
// Standard format
sequence({ reasoning, space(), content, space(), tools, space(), content, end() })

// With section markers
sequence({ reasoning, space(), content_until(section_start), space(), tools, space(), content, end() })

// Forced thinking handling
optional(reasoning) when thinking_forced_open && tools present
```

**Trigger Word Detection**:

- Uses `tool_section_start` as primary trigger
- Falls back to `function_prefix` or `per_call_start`
- Raw JSON uses regex pattern trigger

**Lazy Grammar Optimization**:

- Enabled by default for performance
- Disabled when thinking forced open
- Disabled when no clear trigger word exists

## Testing & Debugging

### Comprehensive Test Coverage

The test suite covers:

**Reasoning Models**:

- Qwen-QwQ-32B (forced-open thinking)
- DeepSeek R1 variants (reasoning only)
- IBM Granite (reasoning + tools)
- ByteDance Seed-OSS (custom reasoning tags)
- Ministral-3-14B-Reasoning
- llama-cpp-deepseek-r1

**Tool Call Formats**:

- JSON_NATIVE: Llama 3.x, Mistral Nemo, Hermes, MiMo-VL
- TAG_WITH_JSON: Nemotron, Qwen3-Coder, MiniMax
- TAG_WITH_TAGGED: Qwen, Hermes (XML), ByteDance Seed-OSS
- BRACKET_TAG: Mistral Small 3.2, Devstral
- PREFIXED_INDEXED: Kimi-K2 variants
- RECIPIENT_BASED: Functionary v3.2
- MARKDOWN_BLOCK: Cohere Command-R Plus

**Edge Cases**:

- Streaming/partial parsing
- Empty content with tools
- Parallel tool calls
- Forced thinking mode
- Multi-byte Unicode markers
- Null content handling
- Multi-line code in tool arguments
- Custom reasoning tags (ByteDance Seed-OSS)

### Debug Tools

**Template Debugger**: `tests/debug-template-parser.cpp`

- Usage: `./bin/debug-template-parser path/to/template.jinja`
- Shows detected format, markers, generated parser, and GBNF grammar

**Debug Logging**: Enable with `LLAMA_LOG_VERBOSITY=2`

- Shows detailed analysis steps
- Displays pattern extraction results
- Lists generated parser structure

**PEG Test Builder**: Fluent API for creating test cases

```cpp
auto tst = peg_tester("template.jinja");
tst.test("input")
   .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
   .tools({tool})
   .expect(expected_message)
   .run();
```

## Adding Support for New Templates

To support a new template format:

1. **If it follows standard patterns** - The auto-parser should detect it automatically using the three main formats (JSON_NATIVE, TAG_WITH_JSON, TAG_WITH_TAGGED)
2. **If it has unique markers** - Add differential analysis patterns in:
   - `compare_reasoning_presence()` for reasoning tags
   - `compare_content_values()` for content wrappers
   - `extract_tool_section()` for tool call patterns
3. **If it needs special handling** - Add a dedicated handler in `chat.cpp` before the auto-parser block

## Edge Cases and Quirks

1. **Forced Thinking**: If `enable_thinking` is true but the model has already started a thought block (e.g., ended the prompt with `<think>`), the parser enters "forced thinking" mode where it immediately expects reasoning content.
2. **Ambiguous Content**: Templates that mix content and tool calls without clear delimiters can be tricky. The analyzer tries to find "common" start/end patterns across multiple examples to be robust.
3. **Double Wrapping**: Some templates (e.g., Functionary) use the same string for both the tool section start and the function prefix (e.g., `<function=`). The analyzer detects this overlap and prevents double-wrapping in the generated parser.
4. **Null Content Rendering**: Some templates render `null` content as Python "None" string. The analyzer detects this and patches content to empty string.
5. **Multi-byte Unicode Markers**: Some templates use special Unicode characters in markers that require careful handling in GBNF generation.

## State of the Autoparser (Jan 2026)

As of January 2026, the unified auto-parser successfully handles major template families including DeepSeek V3/R1, Llama 3.x (native JSON), GLM-4/4.6, and standard XML/JSON formats. It also supports Functionary v3.1/v3.2, Mistral variants, and specialized formats like Kimi-K2's prefixed-indexed structure.

### Tested Templates

The following templates have active tests in `tests/test-chat.cpp`:

| Template | Format | Notes |
|----------|--------|-------|
| DeepSeek V3.1 | `JSON_NATIVE` | Forced thinking mode |
| DeepSeek R1 Distill (Llama/Qwen) | Reasoning only | Forced-open thinking |
| llama-cpp-deepseek-r1 | Reasoning only | Forced-open thinking |
| GLM-4.6 | `TAGGED` | `<tool_call>name\n<arg_key>...<arg_value>...` format |
| Kimi-K2 / Kimi-K2-Instruct / Kimi-K2-Thinking | `PREFIXED_INDEXED` | `functions.name:0` with special markers |
| Apertus-8B-Instruct | `NAME_AS_KEY` | `{"function_name": {...}}` format |
| MiniMax-M2 | `TAG_WITH_JSON` | XML invoke with parameter tags |
| NVIDIA-Nemotron-Nano-v2 | `JSON_NATIVE` | `<TOOLCALL>` wrapper (nested) |
| Mistral-Nemo-Instruct-2407 | `JSON_NATIVE` | `[TOOL_CALLS]` wrapper with id field |
| Functionary v3.1 | `TAG_WITH_JSON` | `<function=X>` non-nested format |
| Functionary v3.2 | `RECIPIENT_BASED` | `>>>` recipient delimiter format |
| MiMo-VL / Hermes 3 / Qwen 2.5 | `JSON_NATIVE` | `<tool_call>` wrapper |
| Apriel 1.5 | `JSON_NATIVE` | `<tool_calls>` wrapper with JSON array |
| Apriel 1.6 Thinker | Reasoning only | Implicit reasoning start |
| Cohere Command-R7B | `JSON_NATIVE` | START_RESPONSE/ACTION/THINKING markers |
| Mistral Small 3.2 | `BRACKET_TAG` | `[TOOL_CALLS]func[ARGS]{...}` with ID |
| Devstral | `BRACKET_TAG` | `[TOOL_CALLS]func[ARGS]{...}` without ID |
| Ministral-3-14B-Reasoning | Custom reasoning | `[THINK]...[/THINK]` tags |
| IBM Granite | `JSON_NATIVE` | `<think></think>` + `<response></response>` |
| ByteDance Seed-OSS | `TAG_WITH_TAGGED` | Custom `<seed:think>` and `<seed:tool_call>` tags |
| Qwen3-Coder | `TAG_WITH_TAGGED` | XML-style tool format |
| Cohere Command-R Plus | `MARKDOWN_BLOCK` | `Action:\n`\`\`\`json\n[...]\n`\`\`` format |

### Currently Unsupported Templates

| Template Family | Model / Variant | Issue Description |
|-----------------|-----------------|-------------------|
| **OpenAI** | `GPT-OSS` | Complex channel markers need new format |

### Templates Without Tool Support

Some templates genuinely don't support tool calls (this is not a detection bug):

- **Phi 3.5 Mini** - The official template has no tool handling. Use Phi-4-mini-instruct for function calling, or community fine-tuned versions.
- **Google Gemma 2 2B** - Pure instruction-following model without tool capabilities.

### TODO / Roadmap

- [ ] **Fix OpenAI GPT-OSS**: Add handling for channel marker structure.
- [x] **~~Fix Cohere Command-R Plus~~**: Added `MARKDOWN_BLOCK` format for `Action:\n`\`\`\`json` structure.

### Recent Additions (Dec 2025 - Jan 2026)

- **RECIPIENT_BASED**: Support for Functionary v3.2's `>>>` recipient delimiter format
- **BRACKET_TAG**: Support for Mistral Small 3.2 and Devstral's `[TOOL_CALLS]...` format
- **Enhanced Content Detection**: Better handling of custom reasoning tags and content wrappers
- **Improved Streaming Support**: Better handling of partial parsing for all supported formats
- **Custom Tag Support**: Support for non-standard reasoning tags like `<seed:think>` (ByteDance)
- **Multi-line Tool Arguments**: Better parsing of complex tool arguments with code blocks
- **MARKDOWN_BLOCK**: Support for Cohere Command-R Plus markdown code block format
- **Implicit Reasoning Support**: Support for templates where reasoning starts implicitly without a start marker.
- **Pure Differential Refactoring (Jan 2026)**: Complete refactoring to eliminate hardcoded patterns:
  - Removed all hardcoded pattern lists (previously had `["<tool_call>", "[TOOL_CALLS]", ...]`)
  - Added structural extraction helpers (`extract_structural_suffix`, `extract_structural_prefix`)
  - Replaced enum-based classification with string-based variant types
  - Only remaining heuristic: JSON detection via parse attempt
  - All markers now discovered through differential template comparison
- **Three Primary Tool Formats**: Consolidated tool calling formats to JSON_NATIVE, TAG_WITH_JSON, and TAG_WITH_TAGGED for clarity and maintainability

The auto-parser now successfully handles 25+ different template formats across reasoning-only, tool-calling, and hybrid models, with comprehensive test coverage ensuring robust parsing across streaming and non-streaming scenarios.
