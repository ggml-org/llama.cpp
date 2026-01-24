# Unified Auto-Parser Architecture

The auto-parser automatically analyzes chat templates to determine how to parse model outputs, including content, reasoning, and tool calls.

## Overview

The unified auto-parser uses a two-phase incremental analysis approach:

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

### tool_call_structure (Phase 2 Result)

Describes how the template formats tool calls:

```cpp
struct tool_call_structure {
    bool supports_tools = false;

    // Container markers (what wraps all tool calls)
    std::string tool_section_start;  // e.g., "<tool_call>", "[TOOL_CALLS]", "<TOOLCALL>", ""
    std::string tool_section_end;    // e.g., "</tool_call>", "]", "</TOOLCALL>", ""

    // Function format (how individual functions are structured)
    enum function_format {
        FUNC_JSON_OBJECT,       // {"name": "X", "arguments": {...}}
        FUNC_TAG_WITH_NAME,     // <function=X>{...}</function>
        FUNC_TAG_NAME_ONLY,     // <X>...</X> where X is function name (rare)
        FUNC_PREFIXED_INDEXED,  // <|tool_call_begin|>functions.X:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
        FUNC_NAME_AS_KEY,       // [{"function_name": {...arguments...}}] (Apertus-style)
        FUNC_BRACKET_TAG,       // [TOOL_CALLS]X[CALL_ID]id[ARGS]{...} (Mistral Small 3.2 style)
        FUNC_RECIPIENT_BASED,   // >>>recipient\n{content} where recipient is "all" (content) or function name (tools)
        FUNC_MARKDOWN_CODE_BLOCK,  // Action:\n```json\n[{"tool_name": "X", ...}]\n``` (Cohere Command-R Plus)
    };
    function_format function_format = FUNC_JSON_OBJECT;

    // For FUNC_JSON_OBJECT format - field names (may vary between templates)
    std::string name_field = "name";       // Could be "tool_name", "function"
    std::string args_field = "arguments";  // Could be "parameters", "params", "input"
    std::string id_field;                  // Optional: "id", "tool_call_id", ""

    // For FUNC_TAG_WITH_NAME format
    std::string function_prefix;  // e.g., "<function="
    std::string function_suffix;  // e.g., ">"
    std::string function_close;   // e.g., "</function>"

    // For FUNC_PREFIXED_INDEXED format (e.g., Kimi-K2)
    std::string per_call_start;      // e.g., "<|tool_call_begin|>"
    std::string function_namespace;  // e.g., "functions." (prefix before function name)
    std::string args_marker;         // e.g., "<|tool_call_argument_begin|>"
    std::string per_call_end;        // e.g., "<|tool_call_end|>"

    // For FUNC_BRACKET_TAG format (e.g., Mistral Small 3.2)
    std::string id_marker;  // e.g., "[CALL_ID]" - marker before tool call ID

    // For FUNC_MARKDOWN_CODE_BLOCK format (Cohere Command-R Plus)
    std::string code_block_marker;    // e.g., "Action:" - text marker before code block
    std::string code_block_language;  // e.g., "json" - language identifier in code fence

    // Argument format (how arguments are structured within a function)
    enum argument_format {
        ARGS_JSON,            // Standard JSON object: {"key": "value", ...}
        ARGS_TAGGED,          // XML-style: <param=key>value</param>
        ARGS_KEY_VALUE_TAGS,  // <arg_key>key</arg_key><arg_value>value</arg_value> (GLM-4.6)
    };
    argument_format argument_format = ARGS_JSON;

    // For ARGS_TAGGED format
    std::string arg_prefix;     // e.g., "<param=", "<parameter="
    std::string arg_suffix;     // e.g., ">"
    std::string arg_close;      // e.g., "</param>", "</parameter>"
    std::string arg_separator;  // e.g., "", "\n"

    // Flag: template renders null content as "None" string, requires empty string instead
    bool requires_nonnull_content = false;
};
```

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
tool_call_structure
    |
    v
generate_parser(content_structure, tool_call_structure)
    |-- build_reasoning_block(content_structure)
    |-- build_content_block(content_structure)
    |-- build_tool_section(tool_call_structure, tools)
    |-- Compose into final parser
    |
    v
common_chat_params (parser, grammar, triggers, preserved_tokens)
```

## Entry Point

The mechanism starts in `common/chat.cpp`, in `common_chat_templates_apply_jinja`:

```cpp
// 1. Analyze the template (two-phase)
template_analysis_result analysis = template_analyzer::analyze_template(tmpl);

// 2. Generate the parser and grammar
auto auto_params = universal_peg_generator::generate_parser(analysis, tmpl, params);

// 3. Use if it provides more than basic content handling
if (auto_params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY ||
    auto_params.thinking_forced_open ||
    !auto_params.parser.empty()) {
    return auto_params;
}
```

## Builder Methods

The unified builder (`common_chat_peg_unified_builder`) provides high-level methods:

- `build_reasoning_block(cs, reasoning_format, thinking_forced_open)` - Build reasoning parser
- `build_content_block(cs, reasoning_format)` - Build content parser
- `build_tool_section(ts, tools, parallel_tool_calls, force_tool_calls)` - Build tool section
- `build_function(ts, name, schema)` - Build single function parser
- `build_arguments(ts, schema)` - Build arguments parser

## Key Templates Supported

- **Granite** - `<think></think>` + `<response></response>` with tool calls
- **Nemotron** - JSON tools with `<TOOLCALL>` wrapper
- **Qwen/Hermes** - XML-style `<function=X><param=key>` format
- **Command-R7B** - `<|START_THINKING|>`/`<|START_RESPONSE|>` + `<|START_ACTION|>` tools
- **DeepSeek R1** - Forced thinking + complex tools
- **Mistral Nemo** - `[TOOL_CALLS]` wrapper
- **MiniMax** - `<minimax:tool_call>` wrapper with XML tools
- **GLM-4.6** - `<minimax:tool_call>` + `<tool_call>name\n<arg_key>...<arg_value>...` format
- **Kimi-K2** - `FUNC_PREFIXED_INDEXED` format with namespace and indices
- **Mistral Small 3.2** - `FUNC_BRACKET_TAG` format with `[TOOL_CALLS]` markers
- **Functionary v3.2** - `FUNC_RECIPIENT_BASED` format with `>>>` routing

## Files

| File | Purpose |
|------|---------|
| `common/chat-auto-parser.h` | Data structures and API declarations |
| `common/chat-auto-parser-analyzer.cpp` | Phase 1 and Phase 2 analysis implementation |
| `common/chat-auto-parser-generator.cpp` | PEG parser generator |
| `common/chat-auto-parser-helpers.h/cpp` | Shared helper functions |
| `common/chat-peg-parser.h/cpp` | Unified builder and mapper classes |
| `common/chat.cpp` | Main entry point and wire-up |

## Algorithm Details

### Phase 1: Content & Reasoning Analysis

#### Reasoning Detection (4 Methods)

**Method 1: Differential Reasoning Content Analysis**

- Render template with `reasoning_content` field present vs absent
- Compare outputs to find markers between `THOUGHT_MARKER` and `CONTENT_MARKER`
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

#### Differential Analysis Algorithm

**Test Payload Strategy**:

1. **Base**: User + Assistant with content only (no tools)
2. **Tool 1**: User + Assistant with tool_calls (empty args)
3. **Tool 2**: User + Assistant with tool_calls (with args)
4. **Tool 3**: User + Assistant with multiple tool calls

**Pattern Extraction Process**:

1. Compute string differences between base and tool outputs
2. Use `test_function_name` as reliable search anchor (using `rfind` for last occurrence)
3. Extract structural elements:
   - `tool_call_opener`: Common prefix before function name
   - `tool_call_closer`: Common suffix after function calls
   - `function_opener`: Tag immediately before function name
   - `function_closer`: Tag after function content
   - `parameter_key_prefix/suffix`: Argument wrapping patterns

#### Format Classification Logic

**FORMAT_JSON_NATIVE**:

- Detected by `{"name":` pattern in `tool_call_opener`
- Or XML markers with JSON structure

**FORMAT_XML_CONSTRUCTED**:

- `function_opener` starts with `<`
- No substantial parameter markers

**FORMAT_RECIPIENT_BASED**:

- `tool_call_start_marker == function_opener`
- No parameter markers
- Opener doesn't start with structural chars

**FORMAT_BRACKET_TAG**:

- `function_name_suffix` contains bracket tags like `[CALL_ID]...[ARGS]`
- `tool_call_start_marker` matches `[TOOL_CALLS]` pattern

**FORMAT_PREFIXED_INDEXED**:

- `function_opener` ends with `.` (namespace separator)
- `function_name_suffix` starts with `:` followed by digit
- Example: `functions.name:0<|tool_call_argument_begin|>`

#### Specialized Format Handling

**FUNC_PREFIXED_INDEXED (Kimi-K2)**:

- Splits `function_opener` at last `>` to get `per_call_start` + `function_namespace`
- Extracts `args_marker` from `function_name_suffix`
- Derives `per_call_end` by matching structural patterns in `tool_call_closer`

**FUNC_TAG_WITH_NAME (Functionary/Nemotron)**:

- Detects nested vs non-nested formats
- Uses overlap detection between `tool_section_start` and `function_prefix`
- Handles double-wrapping prevention

**ARGS_KEY_VALUE_TAGS (GLM-4.6)**:

- Detects `<arg_key>key</arg_key><arg_value>value</arg_value>` pattern
- Cleans up suffix to extract just the key closer

**FUNC_RECIPIENT_BASED (Functionary v3.2)**:

- Detects `>>>` recipient delimiter format
- Routes to "all" for content, function name for tools
- Uses same delimiter for both content and tool routing

**FUNC_BRACKET_TAG (Mistral Small 3.2/Devstral)**:

- Detects `[TOOL_CALLS]function_name[ARGS]{...}` pattern
- Optional `[CALL_ID]id` marker for tool call identification
- No section wrapper - each call starts independently

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

- JSON: Llama 3.x, Mistral Nemo, Hermes, MiMo-VL
- XML: Nemotron, Qwen3-Coder, MiniMax
- Tagged: GLM-4.6 (key-value tags)
- Bracket-tag: Mistral Small 3.2, Devstral
- Prefixed-indexed: Kimi-K2 variants
- Name-as-key: Apertus-8B
- Recipient-based: Functionary v3.2

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

1. **If it follows standard patterns** - The auto-parser should detect it automatically
2. **If it has unique markers** - Add the markers to the detection patterns in:
   - `detect_reasoning_markers()` for reasoning tags
   - `detect_content_markers()` for content wrappers
   - `extract_patterns_from_differences()` for tool call patterns
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
| DeepSeek V3.1 | `FUNC_JSON_OBJECT` | Forced thinking mode |
| DeepSeek R1 Distill (Llama/Qwen) | Reasoning only | Forced-open thinking |
| llama-cpp-deepseek-r1 | Reasoning only | Forced-open thinking |
| GLM-4.6 | `ARGS_KEY_VALUE_TAGS` | `<tool_call>name\n<arg_key>...<arg_value>...` format |
| Kimi-K2 / Kimi-K2-Instruct / Kimi-K2-Thinking | `FUNC_PREFIXED_INDEXED` | `functions.name:0` with special markers |
| Apertus-8B-Instruct | `FUNC_NAME_AS_KEY` | `{"function_name": {...}}` format |
| MiniMax-M2 | `FUNC_TAG_WITH_NAME` | XML invoke with parameter tags |
| NVIDIA-Nemotron-Nano-v2 | `FUNC_JSON_OBJECT` | `<TOOLCALL>` wrapper (nested) |
| Mistral-Nemo-Instruct-2407 | `FUNC_JSON_OBJECT` | `[TOOL_CALLS]` wrapper with id field |
| Functionary v3.1 | `FUNC_TAG_WITH_NAME` | `<function=X>` non-nested format |
| Functionary v3.2 | `FUNC_RECIPIENT_BASED` | `>>>` recipient delimiter format |
| MiMo-VL / Hermes 3 / Qwen 2.5 | `FUNC_JSON_OBJECT` | `<tool_call>` wrapper |
| Apriel 1.5 | `FUNC_JSON_OBJECT` | `<tool_calls>` wrapper with JSON array |
| Apriel 1.6 Thinker | Reasoning only | Implicit reasoning start |
| Cohere Command-R7B | `FUNC_JSON_OBJECT` | `START_RESPONSE/ACTION/THINKING` markers |
| Mistral Small 3.2 | `FUNC_BRACKET_TAG` | `[TOOL_CALLS]func[ARGS]{...}` with ID |
| Devstral | `FUNC_BRACKET_TAG` | `[TOOL_CALLS]func[ARGS]{...}` without ID |
| Ministral-3-14B-Reasoning | Custom reasoning | `[THINK]...[/THINK]` tags |
| IBM Granite | `FUNC_JSON_OBJECT` | `<think></think>` + `<response></response>` |
| ByteDance Seed-OSS | `FUNC_TAG_WITH_NAME` | Custom `<seed:think>` and `<seed:tool_call>` tags |
| Qwen3-Coder | `FUNC_TAG_WITH_NAME` | XML-style tool format |
| Cohere Command-R Plus | `FUNC_MARKDOWN_CODE_BLOCK` | `Action:\n\`\`\`json\n[...]\n\`\`\`` format |

### Currently Unsupported Templates

| Template Family | Model / Variant | Issue Description |
|-----------------|-----------------|-------------------|
| **OpenAI** | `GPT-OSS` | Complex channel markers need new format |

### Templates Without Tool Support

Some templates genuinely don't support tool calls (this is not a detection bug):

- **Phi 3.5 Mini** - The official template has no tool handling. Use Phi-4-mini-instruct for function calling, or community fine-tuned versions.
- **Google Gemma 2 2B** - Pure instruction-following model without tool capabilities.

### TODO / Roadmap

- [ ] **Fix OpenAI GPT-OSS**: Add `FUNC_CHANNEL_BASED` format for channel marker structure.
- [x] **~~Fix Cohere Command-R Plus~~**: Added `FUNC_MARKDOWN_CODE_BLOCK` format for `Action:\n\`\`\`json` structure.

### Recent Additions (Dec 2025 - Jan 2026)

- **FUNC_RECIPIENT_BASED**: Support for Functionary v3.2's `>>>` recipient delimiter format
- **FUNC_BRACKET_TAG**: Support for Mistral Small 3.2 and Devstral's `[TOOL_CALLS]...` format
- **Enhanced Content Detection**: Better handling of custom reasoning tags and content wrappers
- **Improved Streaming Support**: Better handling of partial parsing for all supported formats
- **Custom Tag Support**: Support for non-standard reasoning tags like `<seed:think>` (ByteDance)
- **Multi-line Tool Arguments**: Better parsing of complex tool arguments with code blocks
- **FUNC_MARKDOWN_CODE_BLOCK**: Support for Cohere Command-R Plus markdown code block format
- **Implicit Reasoning Support**: Support for templates where reasoning starts implicitly without a start marker.

The auto-parser now successfully handles 25+ different template formats across reasoning-only, tool-calling, and hybrid models, with comprehensive test coverage ensuring robust parsing across streaming and non-streaming scenarios.
