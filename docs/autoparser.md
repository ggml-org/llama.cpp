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

1. **Method 1**: Render template with `reasoning_content` field present vs absent, compare outputs
2. **Method 2**: Toggle `enable_thinking` context variable, detect differences in prompt
3. **Method 3**: Check if prompt ends with an unclosed reasoning tag

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
3. **Fallback**: If analysis fails or finds no consistent pattern, it falls back to generic content-only handling or the legacy manual handlers if applicable.

## State of the Autoparser (Dec 2025)

As of December 2025, the unified auto-parser successfully handles major template families including DeepSeek V3, Llama 3 (native JSON), and standard XML/JSON formats. However, several specific template variants remain unsupported or have known regressions.

### Currently Unsupported Templates

| Template Family | Model / Variant | Issue Description |
|----------------|-----------------|-------------------|
| **GLM** | `GLM-4.6` | **Argument Extraction**: Autoparser detects `TAGGED` format, but fails to correctly extract arguments using the `ARGS_KEY_VALUE_TAGS` logic. |
| **Kimi** | `Kimi-K2`, `Instruct` | **Parsing as Content**: Tool calls are incorrectly identified as plain content, likely due to failure in detecting `<|tool_calls_section_begin|>` markers. |
| **Apertus** | `Apertus-8B-Instruct` | **Section Detection**: Similar to Kimi, the parser fails to identify the start of the tool section. |
| **MiniMax** | `MiniMax-M2` | **XML Variant**: Uses `<minimax:tool_call><invoke>` which is not currently detected by the XML analyzer hooks. |
| **Nemotron** | `Nano-v2` | **Streaming Regression**: Tests fail with "invalid diff" errors during streaming parsing, suggesting mismatch in incremental state updates. |
| **DeepSeek** | `R1 Distill` | **Marker Handling**: Specific issues with reasoning marker stripping and `enable_thinking` logic for distillation variants. |
| **Cohere** | `Command R+` | **Marker Stripping**: `<|START_RESPONSE|>` markers are leaking into the parsed content. |
| **Mistral** | `Nemo`, `Small` | **Tool Detection**: Fails to parse standard tool calls, possibly due to `[TOOL_CALLS]` wrapper handling specificities or JSON formatting. |
| **Functionary**| `v3.1`, `v3.2` | **Tag Format**: Specific `<function=name>` syntax parsing failures. |
| **OpenAI** | `GPT-OSS` | **Channel Markers**: Complex `<|channel|>` marker structure confuses the content/tool separation. |
| **Phi** | `3.5 Mini` | **Detection Failure**: Autoparser fails to detect that the template supports tools at all. |

### TODO / Roadmap

- [ ] **Fix GLM-4.6**: Debug `common_chat_peg_unified_builder::build_arguments` for `ARGS_KEY_VALUE_TAGS`.
- [ ] **Fix Kimi/Apertus**: Improve `extract_patterns_from_differences` to reliably detect section markers like `<|tool_calls_section_begin|>`.
- [ ] **Add MiniMax Support**: Add detection logic for namespaced XML tags (`minimax:tool_call`).
- [ ] **Fix Nemotron Streaming**: Debug `test_parser_with_streaming` for Nemotron's JSON array format.
- [ ] **Refine DeepSeek R1**: Audit marker configuration for R1 distillations to ensure consistency with V3.1 fix.
- [ ] **Fix Cohere Markers**: Add Cohere's structural markers to the ignored/skipped token list in the parser.
- [ ] **Investigate Mistral/Phi**: Analyze why Phase 2 detection returns false for these templates (run `debug-template-parser` with verbose logging).
