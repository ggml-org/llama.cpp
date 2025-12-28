# Autoparser Enhancement TODO

This document details the limitations discovered during the migration of template tests to the unified autoparser, and proposes solutions for each issue.

## Overview

The autoparser (`chat-auto-parser-analyzer.cpp` and `chat-auto-parser-generator.cpp`) analyzes Jinja templates to automatically detect content structure (reasoning tags, content wrappers) and tool call formats, then generates PEG parsers to handle model outputs. While it works well for many common formats, several templates use unique tool calling conventions that the autoparser doesn't currently recognize.

## Issue 1: NVIDIA-Nemotron-Nano-v2 Streaming Regression

### Problem
During incremental (streaming) parsing, the number of detected tool calls decreases at certain input positions, violating the monotonicity invariant that `common_chat_msg_diff::compute_diffs` enforces.

### Template Format
```
<TOOLCALL>[{"name": "function_name", "arguments": {...}}]</TOOLCALL>
```

### Root Cause
The template has `thinking_forced_open=true` (ends with `<think>\n`). When parsing input like:
```
I'm thinking</think><TOOLCALL>[{"name": "special_function", "arguments": {"arg1": 1}}]</TOOLCALL>
```

At certain character positions during streaming, the parser may:
1. Partially match a tool call structure
2. Then lose that match as more input arrives and changes the parse tree

### Proposed Fix
In `chat-auto-parser-generator.cpp`, ensure that once a tool call is detected (even partially), the parser maintains that detection even as the input grows. This may require:
- Adding "committed" states to the parser that don't backtrack
- Ensuring the reasoning close tag (`</think>`) is properly consumed before tool parsing begins
- Reviewing how `optional(reasoning)` interacts with tool call detection

### Files to Modify
- `common/chat-auto-parser-generator.cpp` - Parser generation logic
- `common/chat-peg-parser.cpp` - PEG execution engine

---

## Issue 2: Apertus-8B-Instruct Tool Format

### Problem
The autoparser doesn't detect the Apertus tool format.

### Template Format
```
<|inner_prefix|>reasoning content<|inner_suffix|>
<|tools_prefix|>[{"function_name": {"param": value}}]<|tools_suffix|>
```

Key differences from standard formats:
1. **Tool wrapper uses special tokens**: `<|tools_prefix|>` / `<|tools_suffix|>` instead of tag-based markers
2. **Tool call structure differs**: Instead of `{"name": "fn", "arguments": {...}}`, uses `{"function_name": arguments_object}`
3. **Reasoning uses different tags**: `<|inner_prefix|>` / `<|inner_suffix|>` instead of `<think>` / `</think>`

### Proposed Fix

1. **Enhance template analyzer** (`chat-auto-parser-analyzer.cpp`):
   - Detect `<|..._prefix|>` / `<|..._suffix|>` patterns as section markers
   - Recognize `inner_prefix/suffix` as reasoning markers
   - Recognize `tools_prefix/suffix` as tool section markers

2. **Add new tool format** to `ToolCallStructure`:
   ```cpp
   enum class FunctionFormat {
       // ... existing formats ...
       FUNC_OBJECT_KEY,  // {"function_name": {args}}
   };
   ```

3. **Generate appropriate parser** in `chat-auto-parser-generator.cpp`:
   - Parse JSON object where key is function name, value is arguments
   - Handle the array wrapper `[{...}]`

### Files to Modify
- `common/chat-auto-parser-analyzer.cpp` - Add detection for `<|..._prefix|>/<|..._suffix|>` patterns
- `common/chat-auto-parser-generator.cpp` - Add `FUNC_OBJECT_KEY` format support
- `common/chat-auto-parser.h` - Add new enum value

---

## Issue 3: MiniMax-M2 XML-Style Tool Format

### Problem
The autoparser doesn't support MiniMax's XML-style tool invocation format.

### Template Format
```xml
<minimax:tool_call>
<invoke name="function_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
</minimax:tool_call>
```

### Key Characteristics
1. **Namespaced XML tags**: `<minimax:tool_call>` uses XML namespace prefix
2. **Attribute-based function name**: `<invoke name="...">` puts function name in attribute
3. **Separate parameter tags**: Each parameter is a separate `<parameter name="...">value</parameter>` element
4. **Parameters are not JSON**: Values are raw text/numbers, not JSON-encoded

### Proposed Fix

1. **Add XML tool format detection** (`chat-auto-parser-analyzer.cpp`):
   - Look for `<namespace:tool_call>` patterns in template
   - Detect `<invoke name=...>` as function call pattern
   - Detect `<parameter name=...>` as argument pattern

2. **Add new tool format**:
   ```cpp
   enum class FunctionFormat {
       // ... existing formats ...
       FUNC_XML_INVOKE,  // <invoke name="fn"><parameter name="p">v</parameter></invoke>
   };

   enum class ArgumentFormat {
       // ... existing formats ...
       ARG_XML_PARAMETERS,  // <parameter name="key">value</parameter>
   };
   ```

3. **Generate XML-aware parser** (`chat-auto-parser-generator.cpp`):
   - Parse `<invoke name="...">` extracting function name from attribute
   - Parse `<parameter name="...">...</parameter>` elements
   - Convert XML parameters to JSON object for tool call arguments
   - Handle namespace prefix (make it configurable per template)

### Complexity Notes
- XML attribute parsing requires handling quotes and escaping
- Parameter values may contain XML that needs proper escaping
- Multiple parameters need to be collected into a JSON object

### Files to Modify
- `common/chat-auto-parser-analyzer.cpp` - Add XML pattern detection
- `common/chat-auto-parser-generator.cpp` - Add XML parser generation
- `common/chat-auto-parser.h` - Add new enum values

---

## Issue 4: GLM-4.6 Key-Value Tool Format

### Problem
The autoparser doesn't support GLM's `<arg_key>/<arg_value>` tool format.

### Template Format
```
<tool_call>function_name
<arg_key>param1</arg_key>
<arg_value>value1</arg_value>
<arg_key>param2</arg_key>
<arg_value>value2</arg_value>
</tool_call>
```

### Key Characteristics
1. **Function name is bare text** after `<tool_call>` tag (not in an attribute or JSON)
2. **Arguments use paired tags**: `<arg_key>` followed by `<arg_value>`
3. **Values are raw text**: Not JSON-encoded
4. **Newline-separated structure**: Each element on its own line

### Proposed Fix

1. **Add key-value detection** (`chat-auto-parser-analyzer.cpp`):
   - Detect `<arg_key>...<arg_value>` patterns in assistant message rendering
   - Identify `<tool_call>` without JSON content as signal for this format

2. **Add new argument format**:
   ```cpp
   enum class ArgumentFormat {
       // ... existing formats ...
       ARG_KEY_VALUE_TAGS,  // <arg_key>k</arg_key><arg_value>v</arg_value>
   };
   ```

3. **Generate key-value parser** (`chat-auto-parser-generator.cpp`):
   - Parse function name as text after `<tool_call>` until newline
   - Parse alternating `<arg_key>` and `<arg_value>` pairs
   - Convert to JSON object for arguments

### Files to Modify
- `common/chat-auto-parser-analyzer.cpp` - Add key-value pattern detection
- `common/chat-auto-parser-generator.cpp` - Add key-value parser generation
- `common/chat-auto-parser.h` - Add new enum value

---

## Issue 5: Kimi-K2-Thinking Indexed Function Format

### Problem
The autoparser doesn't support Kimi's `functions.name:index` tool format.

### Template Format
```
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{"arg1": 1}<|tool_call_end|>
<|tool_calls_section_end|>
```

### Key Characteristics
1. **Special token delimiters**: Uses `<|...|>` style tokens
2. **Prefixed function name**: `functions.` prefix before function name
3. **Indexed tool calls**: `:0`, `:1` etc. suffix for tool call ordering
4. **JSON arguments**: Arguments are standard JSON (this part is already supported)

### Proposed Fix

1. **Add indexed function detection** (`chat-auto-parser-analyzer.cpp`):
   - Detect `functions.` prefix pattern in tool call rendering
   - Detect `:N` suffix pattern for indexing
   - Recognize `<|..._begin|>/<|..._end|>` as section markers

2. **Add new function format**:
   ```cpp
   enum class FunctionFormat {
       // ... existing formats ...
       FUNC_PREFIXED_INDEXED,  // functions.name:index
   };
   ```

3. **Generate indexed parser** (`chat-auto-parser-generator.cpp`):
   - Parse `functions.` prefix (configurable namespace)
   - Extract function name until `:`
   - Parse index number after `:`
   - Continue with standard JSON arguments

### Additional Consideration
The index (`0`, `1`, etc.) could be used to:
- Validate parallel tool call ordering
- Map to tool call IDs if needed
- Currently can be ignored if not needed for output

### Files to Modify
- `common/chat-auto-parser-analyzer.cpp` - Add indexed function detection
- `common/chat-auto-parser-generator.cpp` - Add indexed parser generation
- `common/chat-auto-parser.h` - Add new enum value

---

## Issue 6: DeepSeek-V3.1 Reasoning Marker Detection

### Problem
The autoparser incorrectly detects reasoning markers for the DeepSeek V3.1 template, mistaking `</think>` (the close tag) as the reasoning start marker.

### Template Format
When thinking is disabled (default), the prompt ends with:
```
<｜Assistant｜></think>
```

When thinking is enabled, the prompt ends with:
```
<｜Assistant｜><think>
```

The model output format:
```
[reasoning content]</think>[content]<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>
```

### Root Cause
The autoparser's "Method 3" detection looks for tags in the prompt ending and identifies `</think>` as the start marker because:
1. When `enable_thinking=false`, the prompt ends with `</think>`
2. The autoparser assumes this is where reasoning starts, not ends

### Proposed Fix

1. **Improve reasoning marker detection** (`chat-auto-parser-analyzer.cpp`):
   - Check if detected "start" marker is actually a close tag (starts with `</`)
   - If so, invert the relationship: use the matching open tag as start
   - Add logic to detect the standard `<think>...</think>` pattern even when prompt ends with close tag

2. **Handle conditional thinking** in templates:
   - Detect when templates have both enabled/disabled thinking paths
   - Use the enabled path to determine correct reasoning markers

### Files to Modify
- `common/chat-auto-parser-analyzer.cpp` - Fix reasoning marker detection for close-tag prompts

---

## Issue 7: llama-cpp-lfm2 Template Application Error

### Problem
The autoparser fails during parser generation with a template application error.

### Error
```
Template application failed: vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
```

### Root Cause
The template tries to access `messages[0]` when messages array may be empty during differential analysis.

### Proposed Fix
1. **Ensure non-empty messages** during differential analysis
2. **Add bounds checking** in template rendering

### Files to Modify
- `common/chat-auto-parser-analyzer.cpp` - Ensure messages are populated during analysis

---

## Implementation Priority

Based on template popularity and implementation complexity:

| Priority | Issue | Complexity | Impact |
|----------|-------|------------|--------|
| 1 | Streaming regression (Nemotron-Nano-v2) | Medium | Affects correctness |
| 2 | Reasoning marker detection (DeepSeek-V3.1) | Low | Simple fix |
| 3 | Template application error (lfm2) | Low | Empty message handling |
| 4 | Indexed function format (Kimi-K2) | Low | JSON args already work |
| 5 | Key-value tags (GLM-4.6) | Medium | Common pattern |
| 6 | XML invoke format (MiniMax-M2) | High | Complex XML parsing |
| 7 | Object-key format (Apertus) | Medium | Unique JSON structure |

## Testing Strategy

For each enhancement:

1. **Add template to test matrix** in `test-chat.cpp`:
   ```cpp
   auto tst = peg_tester("models/templates/template-name.jinja");
   tst.test("input with tool call")
       .tools({ tool_definition })
       .expect(expected_message)
       .run();
   ```

2. **Test streaming behavior**:
   - Verify tool call count never decreases during incremental parsing
   - Test partial inputs at various cut-off points

3. **Test edge cases**:
   - Empty arguments
   - Multiple tool calls
   - Tool calls with reasoning
   - Tool calls with content before/after

## Architecture Considerations

### Modular Format Handlers
Consider refactoring to use a plugin-style architecture:

```cpp
class ToolFormatHandler {
public:
    virtual bool detect(const TemplateAnalysis& analysis) = 0;
    virtual std::string generateParser(const ToolCallStructure& ts) = 0;
    virtual common_chat_tool_call parseToolCall(const std::string& raw) = 0;
};

// Register handlers
std::vector<std::unique_ptr<ToolFormatHandler>> handlers = {
    std::make_unique<JsonNativeHandler>(),
    std::make_unique<XmlInvokeHandler>(),
    std::make_unique<KeyValueHandler>(),
    // etc.
};
```

This would make adding new formats cleaner and more maintainable.

### Shared Primitives
Several formats share common needs:
- XML attribute parsing
- Special token recognition (`<|...|>`)
- JSON argument handling
- Namespace/prefix handling

These could be extracted into reusable parser building blocks.
