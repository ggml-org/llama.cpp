# Parsing Model Output

The `common` library contains a PEG parser implementation suitable for parsing
model output.

Types with the prefix `common_peg_*` are intended for general use and may have
applications beyond parsing model output, such as parsing user-provided regex
patterns.

Types with the prefix `common_chat_peg_*` are specialized helpers for model
output.

The parser features:

- Partial parsing of streaming input
- Built-in JSON parsers
- AST generation with semantics via "tagged" nodes

## Example

Below is a contrived example demonstrating how to use the PEG parser to parse
output from a model that emits arguments as JSON.

```cpp
auto parser = build_chat_peg_native_parser([&](common_chat_peg_native_builder & p) {
    // Build a choice of all available tools
    auto tool_choice = p.choice();
    for (const auto & tool : tools) {
        const auto & function = tool.at("function");
        std::string name = function.at("name");
        const auto & schema = function.at("parameters");

        auto tool_name = p.json_member("name", "\"" + p.literal(name) + "\"");
        auto tool_args = p.json_member("arguments", p.schema(p.json(), "tool-" + name + "-schema", schema));

        tool_choice |= p.rule("tool-" + name, "{" << tool_name << "," << tool_args << "}");
    }

    // Define the tool call structure: <tool_call>[{tool}]</tool_call>
    auto tool_call = p.trigger_rule("tool-call",
        p.sequence({
            p.literal("<tool_call>["),
            tool_choice,
            p.literal("]</tool_call>")
        })
    );

    // Parser accepts content, optionally followed by a tool call
    return p.sequence({
        p.content(p.until("<tool_call>")),
        p.optional(tool_call),
        p.end()
    });
});
```

For a more complete example, see `test_example_native()` in
[tests/test-chat-peg-parser.cpp](/tests/test-chat-peg-parser.cpp).

## Parsers/Combinators

### Basic Matchers

- **`eps()`** - Matches nothing and always succeeds (epsilon/empty match)
- **`start()`** - Matches the start of input (anchor `^`)
- **`end()`** - Matches the end of input (anchor `$`)
- **`literal(string)`** - Matches an exact literal string
- **`any()`** - Matches any single character (`.`)

### Combinators

- **`sequence(...)`** - Matches parsers in order; all must succeed
- **`choice(...)`** - Matches the first parser that succeeds from alternatives (ordered choice)
- **`one_or_more(p)`** - Matches one or more repetitions (`+`)
- **`zero_or_more(p)`** - Matches zero or more repetitions (`*`)
- **`optional(p)`** - Matches zero or one occurrence (`?`)
- **`repeat(p, min, max)`** - Matches between min and max repetitions (use `-1` for unbounded)
- **`repeat(p, n)`** - Matches exactly n repetitions

### Lookahead

- **`peek(p)`** - Positive lookahead: succeeds if parser succeeds without consuming input (`&`)
- **`negate(p)`** - Negative lookahead: succeeds if parser fails without consuming input (`!`)

### Character Classes & Utilities

- **`chars(classes, min, max)`** - Matches repetitions of characters from a character class
- **`space()`** - Matches zero or more whitespace characters (space, tab, newline)
- **`until(delimiter)`** - Matches characters until delimiter is found (delimiter not consumed)
- **`until_one_of(delimiters)`** - Matches characters until any delimiter in the list is found
- **`rest()`** - Matches everything remaining (`.*`)

### JSON Parsers

- **`json()`** - Complete JSON parser (objects, arrays, strings, numbers, booleans, null)
- **`json_object()`** - JSON object parser
- **`json_array()`** - JSON array parser
- **`json_string()`** - JSON string parser
- **`json_number()`** - JSON number parser
- **`json_bool()`** - JSON boolean parser
- **`json_null()`** - JSON null parser
- **`json_string_content()`** - JSON string content without surrounding quotes
- **`json_member(key, p)`** - JSON object member with specific key and value parser

### Grammar Building

- **`ref(name)`** - Creates a lightweight reference to a named rule (for recursive grammars)
- **`rule(name, p, trigger)`** - Creates a named rule and returns a reference
- **`trigger_rule(name, p)`** - Creates a trigger rule (entry point for lazy grammar generation)
- **`schema(p, name, schema, raw)`** - Wraps parser with JSON schema metadata for grammar generation

### AST Control

- **`atomic(p)`** - Prevents AST node creation for partial parses
- **`tag(tag, p)`** - Creates AST nodes with semantic tags (multiple nodes can share tags)

## GBNF Grammar Generation

The PEG parser also acts as a convenient DSL for generating GBNF grammars, with
some exceptions.

```cpp
data.grammar = build_grammar([&](const common_grammar_builder & builder) {
    foreach_function(params.tools, [&](const json & fn) {
        builder.resolve_refs(fn.at("parameters"));
    });
    parser.build_grammar(builder, data.grammar_lazy);
});
```

The notable exception is the `negate(p)` lookahead parser, which cannot be
defined as a CFG grammar and therefore does not produce a rule. Its usage
should be limited and preferably hidden behind a `schema()` parser. In many
cases, `until(delimiter)` or `until_one_of(delimiters)` is a better choice.

Another limitation is that the PEG parser requires an unambiguous grammar. In
contrast, the `llama-grammar` implementation can support ambiguous grammars,
though they are difficult to parse.

### Lazy Grammars

During lazy grammar generation, only rules reachable from a `trigger_rule(p)`
are emitted in the grammar. All trigger rules are added as alternations in the
root rule. It is still necessary to define trigger patterns, as the parser has
no interaction with the grammar sampling.

### JSON Schema

The `schema(p, name, schema, raw)` parser will use the `json-schema-to-grammar`
implementation to generate the grammar instead of the underlying parser.

The `raw` option emits a grammar suitable for a raw string instead of a JSON
string. In other words, it won't be wrapped in quotes or require escaping
quotes. It should only be used when `type == "string"`.

The downside is that it can potentially lead to ambiguous grammars. For
example, if a user provides the pattern `^.*$`, the following grammar may be
generated:

```
root ::= "<arg>" .* "</arg>"
```

This creates an ambiguous grammar that cannot be parsed by the PEG parser. To
help mitigate this, if `.*` is found in the pattern, the grammar from the
underlying parser will be emitted instead.

## Common AST Shapes for Chat Parsing

Most model output can be placed in one of the following categories:

- Content only
- Tool calling with arguments emitted as a single JSON object
- Tool calling with arguments emitted as separate entities, either XML
  (Qwen3-Coder, MiniMax M2) or pseudo-function calls (LFM2)

To provide broad coverage,
[`common/chat-peg-parser.h`](/common/chat-peg-parser.h) contains builders and
mappers that help create parsers and visitors/extractors for these types. They
require parsers to tag nodes to conform to an AST "shape". This normalization
makes it easy to extract information and generalize parsing.

### Tag Enum

All tags are defined in the `common_chat_peg_tag` enum for type-safe, switch-based dispatch:

```cpp
enum class common_chat_peg_tag : int {
    NONE = 0,
    REASONING_BLOCK, REASONING, CONTENT,
    TOOL, TOOL_OPEN, TOOL_CLOSE, TOOL_ID, TOOL_NAME, TOOL_ARGS,
    TOOL_ARG, TOOL_ARG_OPEN, TOOL_ARG_CLOSE, TOOL_ARG_NAME,
    TOOL_ARG_STRING_VALUE, TOOL_ARG_JSON_VALUE,
};
```

Use `p.tag(Tag::XXX, parser)` or `p.atomic_tag(Tag::XXX, parser)` to tag nodes.

### Simple

The `common_chat_peg_builder` builds a `simple` parser that supports
content-only models with optional reasoning.

```cpp
build_chat_peg_parser([&](auto & p) {
    using Tag = common_chat_peg_tag;
    return p.sequence({
        p.optional("<think>" + p.tag(Tag::REASONING, p.until("</think>")) + "</think>"),
        p.tag(Tag::CONTENT, p.until("<tool_call>")),
        p.end()
    });
});
```

Use `apply_chat_peg_mapper` with `common_chat_peg_base_mapper()` to extract the content.

```cpp
auto result = parser.parse(ctx);

common_chat_msg msg;
apply_chat_peg_mapper(common_chat_peg_base_mapper(), ctx.ast, result, msg);
```

### Native

The `native` parser is for models that emit tool arguments as a direct JSON object.

```cpp
build_chat_peg_native_parser([&](auto & p) {
    using Tag = common_chat_peg_tag;
    auto get_weather_tool = p.tag(Tag::TOOL, p.sequence({
        p.atomic_tag(Tag::TOOL_OPEN, p.literal("{")),
        p.json_member("name", "\"" + p.atomic_tag(Tag::TOOL_NAME, p.literal("get_weather")) + "\""),
        p.literal(","),
        p.json_member("arguments", p.tag(Tag::TOOL_ARGS, p.json())),
        p.atomic_tag(Tag::TOOL_CLOSE, p.literal("}"))
    }));

    return p.sequence({
        p.tag(Tag::CONTENT, p.until("<tool_call>")),
        p.literal("<tool_call>"),
        get_weather_tool,
        p.literal("</tool_call>"),
        p.end()
    });
});
```

### Nemotron V3 (Constructed Arguments)

The Nemotron V3 parser is for models that emit tool arguments as separate entities (e.g., XML tags like `<function=name><parameter=key>value</parameter></function>`).

```cpp
build_chat_peg_nemotron_v3_parser([&](auto & p) {
    using Tag = common_chat_peg_tag;
    auto location_arg = p.tag(Tag::TOOL_ARG, p.sequence({
        p.atomic_tag(Tag::TOOL_ARG_OPEN, "<parameter name=\"" + p.atomic_tag(Tag::TOOL_ARG_NAME, p.literal("location")) + "\">"),
        p.tag(Tag::TOOL_ARG_STRING_VALUE, p.until("</parameter>")),
        p.atomic_tag(Tag::TOOL_ARG_CLOSE, p.literal("</parameter>"))
    }));

    auto get_weather_tool = p.tag(Tag::TOOL, p.sequence({
        p.atomic_tag(Tag::TOOL_OPEN, "<function name=\"" + p.atomic_tag(Tag::TOOL_NAME, p.literal("get_weather")) + "\">"),
        location_arg,
        p.atomic_tag(Tag::TOOL_CLOSE, p.literal("</function>"))
    }));

    return p.sequence({
        p.tag(Tag::CONTENT, p.until("<tool_call>")),
        p.literal("<tool_call>"),
        get_weather_tool,
        p.literal("</tool_call>"),
        p.end()
    });
});
```
