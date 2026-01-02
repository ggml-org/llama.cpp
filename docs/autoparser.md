# Unified Auto-Parser Architecture

The auto-parser automatically analyzes chat templates to determine how to parse model outputs, including content, reasoning, and tool calls.

## Overview

The unified auto-parser uses a two-phase incremental analysis approach:

1. **Phase 1: Content & Reasoning Analysis** - Analyzes how the template handles basic content and reasoning, without considering tools
2. **Phase 2: Tool Call Analysis** - Analyzes tool calling patterns, layered on top of Phase 1

## Data Structures

### ContentStructure (Phase 1 Result)

Describes how the template handles content and reasoning:

```cpp
struct ContentStructure {
    enum ReasoningMode {
        REASONING_NONE,        // No reasoning markers detected
        REASONING_OPTIONAL,    // <think>...</think> may appear before content
        REASONING_FORCED_OPEN, // Template ends with open reasoning tag
    };

    ReasoningMode reasoning_mode;
    std::string reasoning_start;  // e.g., "<think>", "<|START_THINKING|>"
    std::string reasoning_end;    // e.g., "</think>", "<|END_THINKING|>"

    enum ContentMode {
        CONTENT_PLAIN,                  // No content markers
        CONTENT_ALWAYS_WRAPPED,         // <response>...</response> always present
        CONTENT_WRAPPED_WITH_REASONING, // Content wrapped only when reasoning present
    };

    ContentMode content_mode;
    std::string content_start;  // e.g., "<response>", "<|START_RESPONSE|>"
    std::string content_end;    // e.g., "</response>", "<|END_RESPONSE|>"
};
```

### ToolCallStructure (Phase 2 Result)

Describes how the template formats tool calls:

```cpp
struct ToolCallStructure {
    bool supports_tools;

    // Container markers (what wraps all tool calls)
    std::string tool_section_start;  // e.g., "<tool_call>", "[TOOL_CALLS]"
    std::string tool_section_end;    // e.g., "</tool_call>", "]"

    // Function format
    enum FunctionFormat {
        FUNC_JSON_OBJECT,      // {"name": "X", "arguments": {...}}
        FUNC_TAG_WITH_NAME,    // <function=X>{...}</function>
        FUNC_TAG_NAME_ONLY,    // <X>...</X>
        FUNC_PREFIXED_INDEXED, // <|tool_call_begin|>functions.X:0<|tool_call_argument_begin|>{...}
        FUNC_NAME_AS_KEY,      // [{"function_name": {...arguments...}}]
    };
    FunctionFormat function_format;

    // Field names for JSON format
    std::string name_field = "name";
    std::string args_field = "arguments";
    std::string id_field;

    // Tag patterns for tag-based formats
    std::string function_prefix;  // "<function="
    std::string function_suffix;  // ">"
    std::string function_close;   // "</function>"

    // Markers for prefixed-indexed formats (e.g. Kimi)
    std::string per_call_start;      // "<|tool_call_begin|>"
    std::string function_namespace;  // "functions."
    std::string args_marker;         // "<|tool_call_argument_begin|>"
    std::string per_call_end;        // "<|tool_call_end|>"

    // Argument format
    enum ArgumentFormat { ARGS_JSON, ARGS_TAGGED, ARGS_KEY_VALUE_TAGS };
    ArgumentFormat argument_format;

    // Tag patterns for tagged arguments
    std::string arg_prefix;     // "<param="
    std::string arg_suffix;     // ">"
    std::string arg_close;      // "</param>"
    std::string arg_separator;  // separator between args
};
```

## Analysis Flow

```
Template
    |
    v
Phase 1: analyze_content_structure()
    |-- detect_reasoning_markers() - compare outputs with reasoning_content vs without
    |-- detect_content_markers() - render with content and detect wrapping
    |-- detect_reasoning_mode() - check if prompt ends with open tag
    |
    v
ContentStructure
    |
    v
Phase 2: analyze_tool_structure()
    |-- Check minja.supports_tool_calls
    |-- Differential analysis for tool patterns
    |-- Classify function format (JSON vs tagged)
    |-- Classify argument format (JSON vs tagged)
    |
    v
ToolCallStructure
    |
    v
generate_parser(ContentStructure, ToolCallStructure)
    |-- build_reasoning_block(ContentStructure)
    |-- build_content_block(ContentStructure)
    |-- build_tool_section(ToolCallStructure, tools)
    |-- Compose into final parser
    |
    v
common_chat_params (parser, grammar, triggers, preserved_tokens)
```

## Entry Point

The mechanism starts in `common/chat.cpp`, in `common_chat_templates_apply_jinja`:

```cpp
// 1. Analyze the template (two-phase)
TemplateAnalysisResult analysis = TemplateAnalyzer::analyze_template(tmpl);

// 2. Generate the parser and grammar
auto auto_params = UniversalPEGGenerator::generate_parser(analysis, tmpl, params);

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
- **GLM-4.6** - `<think></think>` + `<tool_call>name\n<arg_key>...<arg_value>...` format

## Files

| File | Purpose |
|------|---------|
| `common/chat-auto-parser.h` | Data structures and API declarations |
| `common/chat-auto-parser-analyzer.cpp` | Phase 1 and Phase 2 analysis implementation |
| `common/chat-auto-parser-generator.cpp` | PEG parser generator |
| `common/chat-auto-parser-helpers.h/cpp` | Shared helper functions |
| `common/chat-peg-parser.h/cpp` | Unified builder and mapper classes |
| `common/chat.cpp` | Main entry point and wire-up |

## How Analysis Works

### Phase 1: Reasoning Detection

1. **Method 1**: Render template with `reasoning_content` field present vs absent, compare outputs. If only the closing tag is found, derive the opening tag (e.g., `</think>` → `<think>`).
2. **Method 2**: Toggle `enable_thinking` context variable, detect differences in prompt. Handles both directions: templates that add content when thinking is enabled (normal) AND templates that add content when thinking is disabled (e.g., GLM-4.6 adds `<think></think>`).
3. **Method 3**: Check if prompt ends with an unclosed reasoning tag
4. **Method 4**: Look for adjacent opening/closing tag pairs in the prompt (e.g., `<think></think>`)

### Phase 1: Content Detection

Render template with unique content marker, check if it's wrapped in known patterns like `<response>...</response>`.

### Phase 2: Tool Detection

Uses differential analysis:
1. Render template with base message (no tools)
2. Render with tool_calls (single, multiple args, parallel)
3. Compare differences to extract patterns

The analyzer probes with specific "dummy" function calls:
- **Function Name**: `test_function_name`
- **Arguments**: `{"param1": "value1", "param2": "value2"}`

This allows it to reliably search for these strings in the output to identify where function names and arguments are placed.

**Note:** The analyzer uses a reverse search (`rfind`) to locate the function name in the tool call diff. This ensures it identifies the *actual* tool call generation rather than any tool definitions that might appear earlier in the diff (common in Mistral templates).

### Phase 2: Format Classification

The analyzer classifies tool call formats based on patterns found in the diff:

1. **FUNC_PREFIXED_INDEXED** - Detected through structural analysis when:
   - `function_opener` ends with `.` (namespace separator)
   - `function_name_suffix` starts with `:` followed by digit (index)
   - Example: `<|tool_call_begin|>functions.name:0<|tool_call_argument_begin|>`
   - Splits opener at the last `>` before `.` to get `per_call_start` and `function_namespace`
   - Extracts `args_marker` from `function_name_suffix` (the tag after the index)
   - Derives `per_call_end` and `tool_section_end` by finding structurally matching markers in `tool_call_closer`

2. **FUNC_TAG_WITH_NAME** - Detected when `<function=` or similar tag-with-name pattern is found:
   - **Nested format** (e.g., Nemotron): `<tool_call><function=X>...</function></tool_call>`
   - **Non-nested format** (e.g., Functionary v3.1): `<function=X>...</function>` where section markers equal function markers
   - The analyzer detects overlap between `tool_section_start` and `function_prefix` to determine nesting

3. **FUNC_JSON_OBJECT** - Standard JSON format with `name` and `arguments` fields

4. **FUNC_NAME_AS_KEY** - Format where function name is the JSON key (e.g., Apertus)

5. **ARGS_KEY_VALUE_TAGS** - Specialized argument format using `<arg_key>` / `<arg_value>` tags (e.g., GLM-4.6)

## Testing & Debugging

### Testing

Tests are located in `tests/test-chat.cpp`. The `test_templates` function iterates through specific scenarios, applies the template, generates the expected delta, and verifies that the generated PEG parser correctly parses it back into a structured message.

### Debugging

A dedicated tool `tests/debug-template-parser.cpp` is available:
- **Usage**: `./bin/debug-template-parser path/to/template.jinja`
- Prints detected format, discovered markers, generated PEG parser, and GBNF grammar.

Enable debug logging for detailed analysis output:
```bash
LLAMA_LOG_VERBOSITY=2 ./build/bin/llama-server ...
```

Key log messages:
- `=== STARTING UNIFIED TEMPLATE ANALYSIS ===`
- `Phase 1 complete: reasoning_mode=X, content_mode=Y`
- `Phase 2 complete: supports_tools=true/false, function_format=X`
- `=== UNIFIED PEG PARSER GENERATION COMPLETED ===`

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
4. **Double Wrapping**: Some templates (e.g., Functionary) use the same string for both the tool section start and the function prefix (e.g., `<function=`). The analyzer detects this overlap and prevents double-wrapping in the generated parser.

## State of the Autoparser (Jan 2026)

As of January 2026, the unified auto-parser successfully handles major template families including DeepSeek V3, Llama 3 (native JSON), GLM-4, GLM-4.6, and standard XML/JSON formats. It also supports Functionary v3.1 and Mistral Nemo.

### Tested Templates

The following templates have active tests in `tests/test-chat.cpp`:

| Template | Format | Notes |
|----------|--------|-------|
| DeepSeek V3.1 | `FUNC_JSON_OBJECT` | Forced thinking mode |
| DeepSeek R1 Distill (Llama/Qwen) | Reasoning only | Forced-open thinking, tool tests pending |
| llama-cpp-deepseek-r1 | Reasoning only | Forced-open thinking |
| GLM-4.6 | `ARGS_KEY_VALUE_TAGS` | `<tool_call>name\n<arg_key>...<arg_value>...` format |
| Kimi-K2 / Kimi-K2-Instruct / Kimi-K2-Thinking | `FUNC_PREFIXED_INDEXED` | `functions.name:0` with special markers |
| Apertus-8B-Instruct | `FUNC_NAME_AS_KEY` | `{"function_name": {...}}` format |
| MiniMax-M2 | `FUNC_TAG_WITH_NAME` | XML invoke with parameter tags |
| NVIDIA-Nemotron-Nano-v2 | `FUNC_JSON_OBJECT` | `<TOOLCALL>` wrapper (nested) |
| Mistral-Nemo-Instruct-2407 | `FUNC_JSON_OBJECT` | `[TOOL_CALLS]` wrapper with id field |
| Functionary v3.1 | `FUNC_TAG_WITH_NAME` | `<function=X>` non-nested format |
| MiMo-VL / Hermes 3 / Qwen 2.5 | `FUNC_JSON_OBJECT` | `<tool_call>` wrapper |
| Apriel 1.5 | `FUNC_JSON_OBJECT` | `<tool_calls>` wrapper with JSON array |
| Cohere Command-R7B | `FUNC_JSON_OBJECT` | `START_RESPONSE/ACTION/THINKING` markers |

### Currently Unsupported Templates

| Template Family | Model / Variant | Issue Description |
|-----------------|-----------------|-------------------|
| **OpenAI** | `GPT-OSS` | Complex channel markers need new format |
| **Functionary** | `v3.2` | Uses `recipient\nfunction_name\n{args}` |
| **FireFunction** | `v2` | tool_section_end includes EOS marker |
| **Cohere** | `Command-R Plus` | Different format than R7B (uses `Action:`) |
| **Mistral Small 3.2** | `Mistral-Small-3.2-24B` | Uses `[TOOL_CALLS]func[ARGS]{...}` format |
| **Devstral** | `Devstral-Small-2507` | Uses `[TOOL_CALLS]func[ARGS]{...}` format |

### Templates Without Tool Support

Some templates genuinely don't support tool calls (this is not a detection bug):

- **Phi 3.5 Mini** - The official template has no tool handling. Use Phi-4-mini-instruct for function calling, or community fine-tuned versions.

### Completed Fixes

- [x] **Fix GLM-4.6**: Fixed reasoning detection (Method 1 derivation from closing tag, Method 2 reverse case handling). Tool format uses `ARGS_KEY_VALUE_TAGS`.
- [x] **Fix Kimi/Apertus**: Implemented `FUNC_TAG_WITH_NAME` (no-equals) detection for Kimi and `FUNC_NAME_AS_KEY` for Apertus.
- [x] **Add MiniMax Support**: Verified dynamic detection handles namespaced tags; fixed `arg_suffix` extraction for unquoted values.
- [x] **Fix Nemotron Streaming**: Nemotron Nano-v2 now works with streaming parsing.
- [x] **Refine DeepSeek R1**: Fixed GBNF generation for non-ASCII/multi-byte markers strings to support `R1` distillation templates.
- [x] **Fix Cohere Markers**: Add Cohere's structural markers to the detected content patterns to correctly determine content boundaries.
- [x] **Fix Mistral/Functionary**: Implemented robust detection logic (`rfind` for names, overlap detection for markers).
- [x] **Fix Kimi-K2-Thinking**: Added `FUNC_PREFIXED_INDEXED` detection when `functions.` appears in function_opener. Derives markers from `_begin|>` pattern.
- [x] **Fix Mistral ID Field**: Changed id_field search to use full tool diff instead of just tool_call_opener.
- [x] **Fix Nested vs Non-nested XML**: Added detection for when `tool_section_start` matches `function_prefix` to properly set `function_close`.
- [x] **Fix JSON Content Serialization**: Content is now `null` (not `""`) when tool_calls are present, per OpenAI spec.
- [x] **Fix Null Content Rendering**: Added `requires_nonnull_content` detection for templates that render `null` content as Python "None" string. Content is patched to empty string for these models.

### TODO / Roadmap

- [x] **Enable DeepSeek R1 Distill Tests**: Reasoning tests enabled; tool tests pending (multi-byte Unicode markers).
- [x] **Enable Kimi-K2/Kimi-K2-Instruct**: Now working with FUNC_PREFIXED_INDEXED format.
- [x] **Enable MiMo-VL/Hermes/Qwen**: Standard `<tool_call>` JSON format works.
- [x] **Enable Apriel 1.5**: Uses `<tool_calls>` wrapper with JSON array.
- [ ] **Add Mistral Small 3.2 / Devstral**: Needs new `FUNC_TAG_SEPARATED` format for `[TOOL_CALLS]func[ARGS]{...}`.
- [ ] **Fix Functionary v3.2**: Add specialized format detection for recipient-based routing.
- [ ] **Fix FireFunction v2**: Fix tool_section_end extraction to not include EOS markers.
- [ ] **Fix OpenAI GPT-OSS**: Add `FUNC_CHANNEL_BASED` format for channel marker structure.
- [x] **Enable Cohere Command-R7B**: Uses START_RESPONSE/THINKING/ACTION markers.
- [ ] **Fix Cohere Command-R Plus**: Different marker format (Action: [...]) needs investigation.
