# Auto-Generated Parser Issue - Content Markers with reasoning_format=NONE

## Original Problem Statement
When `reasoning_format = COMMON_REASONING_FORMAT_NONE`, the entire output including thinking markers (`<|START_THINKING|>...<|END_THINKING|>`) should be preserved in `message.content`, not parsed separately into a `reasoning_content` field.

Key insight from user: "Thinking format = NONE does NOT mean that `enable_thinking` is false. It only means that the server-side parser should NOT parse any reasoning markers into reasoning_content, instead outputting all the content (reasoning and whatnot) as message.content."

## Failing Test Case
### File: `/devel/tools/llama.cpp/tests/test-chat.cpp`
### Line: ~3302-3305
```cpp
// Test thinking + content with no reasoning_format (unparsed as r7b)
test_peg_parser(tmpls.get(), [&](auto & t) {
    t.input = "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
              "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>";
    t.expect = message_assist_thoughts_unparsed_r7b;
});
```

Expected output: `<|START_THINKING|>I'm\nthinking<|END_THINKING|>Hello, world!\nWhat's up?`
Actual output: `Hello, world!\nWhat's up?`

### Template Being Tested
CohereForAI Command-R 7B (2024-tool_use) has three types of markers:
1. **Reasoning markers**: `<|START_THINKING|>` and `<|END_THINKING|>`
2. **Content markers**: `<|START_RESPONSE|>` and `<|END_RESPONSE|>`
3. **Tool call markers**: `<|START_ACTION|>` and `<|END_ACTION|>`

## Changes Made

### 1. Compilation Error Fix
**Issue**: `p.first_of()` method doesn't exist on builder classes
**Fix**: Replaced with `p.choice()` which is the correct method
**Files Modified**:
- `/devel/tools/llama.cpp/common/chat-auto-parser.cpp` (line ~2068)

### 2. Added Content Marker Detection
**Purpose**: Detect and handle content markers separately from reasoning and tool call markers

**Struct Update** (in `common/chat-auto-parser.h`):
```cpp
// Content/Response markers (for templates that wrap content in tags like <|START_RESPONSE|>)
std::string content_start_marker;
std::string content_end_marker;
```

**Function Added** (in `common/chat-auto-parser.cpp`):
```cpp
static void analyze_content_markers(const minja::chat_template & tmpl, DiscoveredPattern & patterns) {
    // Renders a message with CONTENT_MARKER_12345 placeholder
    // Searches for known patterns like <|START_RESPONSE|>...<|END_RESPONSE|>
    // Can also infer custom markers based on tag-like patterns around placeholder
}
```

**Integration**:
- Updated `special_markers` map to include content markers
- Added call to `analyze_content_markers` after `analyze_reasoning` in `analyze_template_discovery()`
- Updated parser generators to use content markers

## Root Cause Analysis

### The Core Problem
When `reasoning_format = NONE`, the parser is configured as:
```cpp
reasoning = eps()  // reasoning not parsed separately

parser = reason + optional_space + build_content() + end()
       = eps() + optional_space + build_content() + end()
```

### Content Parser Implementation
In `build_native_parser()`, `build_content()` creates a choice of three alternatives:

```cpp
// First alternative (when content markers exist and reasoning_format=NONE):
match_content_with_markers() = p.until(content_start) + 
                           content_start + 
                           p.content(p.until(content_end)) + 
                           content_end
```

### Why It Strips Everything

Input: `<|START_THINKING|>I'm thinking<|END_THINKING|><|START_RESPONSE|>Hello<|END_RESPONSE|>`

1. `p.until('<|START_RESPONSE|>')` matches: `<|START_THINKING|>I'm thinking<|END_THINKING|><|START_RESPONSE|>`
2. But ONLY the last part (`Hello`) gets tagged as `content` by: `p.content(p.until('<|END_RESPONSE|>'))`
3. Result: `content` = `Hello` (stripped everything!)

### Expected Behavior
We need to preserve: `<|START_THINKING|>I'm thinking<|END_THINKING|>Hello`

This should be achieved by:
- Matching up to `<|END_RESPONSE|>` (includes reasoning markers + content markers)
- Tagging everything EXCEPT content markers as content
- Let custom mapper strip only the content markers

## Current Test Results

### Debug Output
```
Using optional content markers for content parsing
match_content_with_markers: reasoning_start='<|START_THINKING|>', reasoning_end='<|END_THINKING|>', content_start='<|START_RESPONSE|>', reasoning_format=0
```

**Note**: `reasoning_format=0` confirms `COMMON_REASONING_FORMAT_NONE` is being used.

### Parser Information
- Format: JSON_NATIVE (using PEG_NATIVE parser)
- Content markers detected: Yes (`<|START_RESPONSE|>`, `<|END_RESPONSE|>`)
- Reasoning markers: Present but NOT parsed separately

### Compilation Status
- Build: ✓ Success
- Runtime: ✗ Test fails (markers being stripped incorrectly)

## Recommended Solutions

### Option 1: Capture Up to Content End
**Approach**: Let first alternative match up to `<|END_RESPONSE|>` (not `<|START_RESPONSE|>`)

```cpp
auto match_content_with_markers = [&]() {
    if (!reasoning_start.empty() && !reasoning_end.empty() && 
        inputs.reasoning_format == COMMON_REASONING_FORMAT_NONE) {
        // Match everything up to content_end (includes reasoning + content markers)
        // Then content mapper will be everything matched (which includes reasoning markers)
        // We let content mapper handle stripping of just the content markers
        return p.until_one_of({content_start, content_end});
    } else {
        // Normal case: just match content_start + content + content_end
        return content_start + p.content(p.until(content_end)) + content_end;
    }
};
```

**Pros**: Simple, one-line fix
**Cons**: Requires mapper-level changes to strip only content markers

### Option 2: Use `until_one_of` with Conditional Matching
**Approach**: Match until either marker, then handle based on which one matched

```cpp
// Match up to either content_start or content_end
auto prefix = p.until_one_of({content_start, content_end}) + p.until(content_end);

// Alternative: just capture everything
auto fallback = p.content(p.rest());

return p.choice({prefix, fallback});
```

**Pros**: Better handles streaming scenarios
**Cons**: More complex logic needed in mapper

### Option 3: Add Custom Tag/Strip Logic
**New PEG Method**: `strip_markers(parser, [markers_to_strip])`

This would allow:
- Capturing entire content (reasoning + content markers + content)
- Stripping only specific markers from the captured text
- Preserving reasoning markers when they shouldn't be stripped

**Implementation effort**: High (requires PEG parser framework changes)

## Next Steps

### Immediate Actions
1. Choose optimal solution approach (recommend Option 1 or 2)
2. Implement the fix in `build_native_parser()`
3. Also fix `build_constructed_parser()` if needed
4. Test with CohereForAI template
5. Test with other templates that use content markers
6. Verify backward compatibility with existing tests

### Test Cases to Verify
1. ✗ CohereForAI with reasoning markers + content markers (CURRENT FAILING)
2. ✓ CohereForAI with just content markers  
3. ✓ Other templates with reasoning_format != NONE
4. ✓ Templates without content markers

## File Manifest

**Modified Files**:
- `/devel/tools/llama.cpp/common/chat-auto-parser.h` - Added content marker structs
- `/devel/tools/llama.cpp/common/chat-auto-parser.cpp` - Added detection and parser logic

**Test Files**:
- `/devel/tools/llama.cpp/tests/test-chat.cpp` - Line 3302 (test case)
- `/devel/tools/llama.cpp/models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja` - Template being tested
