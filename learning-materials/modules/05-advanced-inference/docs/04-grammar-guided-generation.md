# Grammar-Guided Generation: Structured Outputs

**Module 5, Lesson 4**
**Estimated Time**: 4-5 hours
**Difficulty**: Advanced

## Overview

Grammar-guided generation (also called constrained decoding) ensures LLM outputs conform to specific formats like JSON, XML, or custom syntaxes. This is critical for production applications that need reliable, parseable outputs.

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand GBNF (GGML BNF) grammar format
- ✅ Write grammars for JSON, code, and custom formats
- ✅ Implement grammar-based token masking
- ✅ Optimize grammar parsing performance
- ✅ Handle edge cases and errors

## The Problem: Unreliable LLM Outputs

### Without Constraints

```python
# Prompt: "Generate user info as JSON"
response = llm.generate("Generate user info as JSON")

# Possible outputs:
# ✅ Good: {"name": "Alice", "age": 30}
# ❌ Bad:  {name: "Alice", age: 30}  # Missing quotes
# ❌ Bad:  {"name": "Alice", "age": 30  # Missing closing brace
# ❌ Bad:  Here's the JSON: {"name": "Alice"}  # Extra text
# ❌ Bad:  {"name": "Alice", "age": "30"}  # Wrong type

Success rate: ~60-80% (depending on model quality)
```

### With Grammar Constraints

```python
# Use grammar to enforce valid JSON
response = llm.generate(
    "Generate user info",
    grammar=json_grammar
)

# Guaranteed output: {"name": "Alice", "age": 30}
Success rate: 100%
```

## GBNF: GGML Backus-Naur Form

### What is GBNF?

GBNF is llama.cpp's grammar specification language, based on BNF (Backus-Naur Form). It defines **which tokens are valid** at each step of generation.

### Basic Syntax

```gbnf
# Comments start with #
root ::= expression

# Literals (exact strings)
greeting ::= "hello" | "hi" | "hey"

# Character classes
digit ::= [0-9]
letter ::= [a-zA-Z]

# Sequences
number ::= digit+
word ::= letter+

# Optional elements
optional-sign ::= [+-]?

# Repetition
one-or-more ::= element+
zero-or-more ::= element*

# Alternatives
bool ::= "true" | "false"

# Grouping
expression ::= "(" expression ")" | number
```

### JSON Grammar Example

Complete grammar for valid JSON:

```gbnf
# json.gbnf - Full JSON specification

root ::= value

value ::= object | array | string | number | boolean | null

# Objects
object ::= "{" ws (member ("," ws member)*)? "}" ws
member ::= string ":" ws value

# Arrays
array ::= "[" ws (value ("," ws value)*)? "]" ws

# Strings
string ::= "\"" character* "\""
character ::= [^"\\] | "\\" escape
escape ::= ["\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]

# Numbers
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+

# Primitives
boolean ::= "true" | "false"
null ::= "null"

# Whitespace
ws ::= [ \t\n\r]*
```

## Using Grammars in llama.cpp

### Command Line

```bash
# Generate with JSON grammar
./llama-cli \
    -m model.gguf \
    -p "Generate user profile" \
    --grammar-file grammars/json.gbnf \
    -n 256

# Generate with custom grammar
./llama-cli \
    -m model.gguf \
    -p "Write a function" \
    --grammar-file grammars/python.gbnf \
    -n 512
```

### C++ API

```cpp
#include "llama.h"
#include "llama-grammar.h"

// Load grammar from file
llama_grammar* grammar = llama_grammar_init_from_file(
    "grammars/json.gbnf",
    "root"  // Start symbol
);

// Create sampling parameters
llama_sampling_params sampling;
sampling.grammar = grammar;

// Generate with grammar
std::vector<llama_token> tokens;
while (!done) {
    llama_decode(ctx, batch);

    // Grammar constrains which tokens are valid
    llama_token token = llama_sample_token_grammar(
        ctx,
        sampling,
        grammar
    );

    tokens.push_back(token);

    // Update grammar state
    llama_grammar_accept_token(grammar, token);
}

// Cleanup
llama_grammar_free(grammar);
```

### Python Bindings

```python
from llama_cpp import Llama

model = Llama(model_path="model.gguf")

# Load grammar
with open("grammars/json.gbnf") as f:
    json_grammar = f.read()

# Generate with grammar
response = model(
    "Generate user profile with name, age, email",
    grammar=json_grammar,
    max_tokens=256
)

print(response["choices"][0]["text"])
# Guaranteed valid JSON: {"name": "...", "age": ..., "email": "..."}
```

## Advanced Grammar Patterns

### JSON Schema to GBNF

Convert JSON schemas to grammars automatically:

```python
def json_schema_to_gbnf(schema):
    """
    Convert JSON Schema to GBNF grammar
    """
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Build object grammar
        members = []
        for key, value_schema in properties.items():
            optional = "" if key in required else "?"
            member_rule = f'"{key}" ":" ws {schema_to_rule(value_schema)}'
            members.append(f"({member_rule}){optional}")

        return f'"{{" ws {" ws \",\" ws ".join(members)} ws "}}"'

    elif schema.get("type") == "array":
        items = schema.get("items", {})
        return f'"[" ws ({schema_to_rule(items)} ws ("," ws {schema_to_rule(items)} ws)*)? "]"'

    elif schema.get("type") == "string":
        if "enum" in schema:
            # Enum: one of fixed values
            options = " | ".join(f'"{v}"' for v in schema["enum"])
            return f"({options})"
        else:
            return "string"

    elif schema.get("type") == "number":
        return "number"

    elif schema.get("type") == "boolean":
        return "boolean"

    elif schema.get("type") == "null":
        return "null"
```

Example usage:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "email": {"type": "string"},
        "role": {"type": "string", "enum": ["admin", "user", "guest"]}
    },
    "required": ["name", "age"]
}

grammar = json_schema_to_gbnf(schema)

# Generated GBNF:
# root ::= "{" ws "name" ":" ws string ws "," ws "age" ":" ws number ws
#          ("," ws "email" ":" ws string ws)?
#          ("," ws "role" ":" ws ("admin" | "user" | "guest") ws)? "}"
```

### Programming Language Grammars

#### Python Function Grammar

```gbnf
# python-function.gbnf

root ::= function

function ::= "def" ws identifier "(" parameters ")" ":" ws block

identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*

parameters ::= (identifier ("," ws identifier)*)?

block ::= "\n" indent statement+ dedent

statement ::=
    assignment |
    return-stmt |
    if-stmt |
    for-stmt |
    expression

assignment ::= identifier ws "=" ws expression "\n"

return-stmt ::= "return" ws expression "\n"

if-stmt ::= "if" ws expression ":" ws block (else-block)?

else-block ::= "else" ":" ws block

for-stmt ::= "for" ws identifier ws "in" ws expression ":" ws block

expression ::=
    identifier |
    number |
    string |
    binary-op |
    function-call

binary-op ::= expression ws operator ws expression

operator ::= "+" | "-" | "*" | "/" | "==" | "!=" | "<" | ">"

function-call ::= identifier "(" arguments ")"

arguments ::= (expression ("," ws expression)*)?

# Primitives
number ::= [0-9]+
string ::= "\"" [^"]* "\""
ws ::= [ \t]*
indent ::= "    "
dedent ::= ""
```

#### SQL Query Grammar

```gbnf
# sql-select.gbnf

root ::= select-stmt

select-stmt ::= "SELECT" ws columns ws "FROM" ws table (where-clause)? (order-clause)?

columns ::= "*" | column-list

column-list ::= identifier ("," ws identifier)*

table ::= identifier

where-clause ::= "WHERE" ws condition

condition ::=
    identifier ws comparison-op ws value |
    condition ws "AND" ws condition |
    condition ws "OR" ws condition

comparison-op ::= "=" | "!=" | "<" | ">" | "<=" | ">=" | "LIKE"

order-clause ::= "ORDER BY" ws identifier (ws sort-order)?

sort-order ::= "ASC" | "DESC"

identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*

value ::= number | string

number ::= [0-9]+

string ::= "'" [^']* "'"

ws ::= [ \t\n]+
```

### Custom Domain Grammars

#### Email Address Grammar

```gbnf
# email.gbnf

root ::= email

email ::= local-part "@" domain

local-part ::= atom ("." atom)*

atom ::= [a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+

domain ::= subdomain ("." subdomain)*

subdomain ::= [a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?
```

#### URL Grammar

```gbnf
# url.gbnf

root ::= url

url ::= scheme "://" authority path? query? fragment?

scheme ::= "http" | "https" | "ftp"

authority ::= (userinfo "@")? host (":" port)?

userinfo ::= [a-zA-Z0-9._~!$&'()*+,;=:%-]+

host ::= domain | ip-address

domain ::= subdomain ("." subdomain)*

subdomain ::= [a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?

ip-address ::= [0-9]+ "." [0-9]+ "." [0-9]+ "." [0-9]+

port ::= [0-9]+

path ::= "/" segment ("/" segment)*

segment ::= [a-zA-Z0-9._~!$&'()*+,;=:@%-]*

query ::= "?" query-params

query-params ::= query-param ("&" query-param)*

query-param ::= [a-zA-Z0-9._~!$&'()*+,;=:@/?%-]+ "=" [a-zA-Z0-9._~!$&'()*+,;=:@/?%-]+

fragment ::= "#" [a-zA-Z0-9._~!$&'()*+,;=:@/?%-]+
```

## Implementation Details

### How Grammar Constraint Works

At each generation step:

1. **Parse current state**: Determine which grammar rules apply
2. **Compute valid tokens**: Find all tokens that satisfy grammar
3. **Mask logits**: Set invalid tokens to -inf
4. **Sample**: Choose from valid tokens only

```python
def grammar_constrained_sampling(logits, grammar_state):
    """
    Apply grammar constraints to token probabilities
    """
    # Get valid tokens from grammar
    valid_tokens = grammar_state.get_valid_tokens()

    # Create mask
    mask = torch.full_like(logits, float('-inf'))
    mask[valid_tokens] = 0

    # Apply mask
    constrained_logits = logits + mask

    # Sample from valid tokens
    probs = torch.softmax(constrained_logits, dim=-1)
    token = torch.multinomial(probs, 1)

    # Update grammar state
    grammar_state.advance(token)

    return token
```

### Grammar Parser Implementation

```cpp
// Simplified grammar parser (see llama-grammar.cpp for full implementation)

struct GrammarElement {
    enum Type {
        CHAR_LITERAL,
        CHAR_CLASS,
        RULE_REF,
        ALT,
        SEQUENCE,
        OPTIONAL,
        ZERO_OR_MORE,
        ONE_OR_MORE
    } type;

    std::string value;
    std::vector<GrammarElement> elements;
};

struct GrammarRule {
    std::string name;
    GrammarElement definition;
};

class Grammar {
private:
    std::vector<GrammarRule> rules;
    std::unordered_map<std::string, size_t> rule_indices;

    // Current parse state (stack of rule positions)
    std::vector<std::pair<size_t, size_t>> stack;

public:
    Grammar(const std::string& gbnf_text) {
        parse_gbnf(gbnf_text);
    }

    std::vector<llama_token> get_valid_tokens(
        llama_context* ctx,
        const std::vector<llama_token>& current_tokens
    ) {
        std::vector<llama_token> valid;

        // Try each token in vocabulary
        for (llama_token token = 0; token < llama_n_vocab(model); token++) {
            if (accepts_token(token, current_tokens)) {
                valid.push_back(token);
            }
        }

        return valid;
    }

    bool accepts_token(
        llama_token token,
        const std::vector<llama_token>& history
    ) {
        // Convert token to string
        std::string token_str = llama_token_to_str(ctx, token);

        // Simulate parsing with this token
        auto saved_stack = stack;

        bool accepted = try_parse(token_str);

        // Restore state
        stack = saved_stack;

        return accepted;
    }

private:
    bool try_parse(const std::string& token_str);
    void parse_gbnf(const std::string& text);
};
```

### Optimization: Precompute Valid Tokens

```cpp
class OptimizedGrammar {
private:
    // Precomputed valid tokens for each grammar state
    std::unordered_map<
        std::vector<size_t>,  // Stack state (hash)
        std::vector<llama_token>  // Valid tokens
    > valid_token_cache;

public:
    std::vector<llama_token> get_valid_tokens_fast(
        const std::vector<size_t>& stack_state
    ) {
        // Check cache
        auto it = valid_token_cache.find(stack_state);
        if (it != valid_token_cache.end()) {
            return it->second;
        }

        // Compute and cache
        auto valid = compute_valid_tokens(stack_state);
        valid_token_cache[stack_state] = valid;

        return valid;
    }
};
```

## Performance Considerations

### Grammar Evaluation Overhead

| Operation | Time per Token | Impact |
|-----------|---------------|---------|
| No grammar | 10ms | Baseline |
| Simple grammar (JSON) | 11ms | +10% |
| Complex grammar (Python) | 15ms | +50% |
| Highly ambiguous grammar | 30ms | +200% |

**Optimization strategies**:
1. Cache valid tokens per grammar state
2. Use simplified grammars when possible
3. Limit grammar complexity (depth, alternatives)

### Memory Usage

```python
def estimate_grammar_memory(grammar):
    """
    Estimate memory usage for grammar
    """
    # Number of unique grammar states
    num_states = count_reachable_states(grammar)

    # Tokens per state (average)
    tokens_per_state = 100

    # Memory per cached entry
    bytes_per_entry = (
        8 +  # State hash
        tokens_per_state * 4  # Token IDs (int32)
    )

    total_memory_mb = (num_states * bytes_per_entry) / (1024 * 1024)

    return total_memory_mb

# Example:
# JSON grammar: ~500 states → ~0.2 MB
# Python grammar: ~5000 states → ~2 MB
# Custom complex grammar: ~50000 states → ~20 MB
```

## Common Pitfalls and Solutions

### Pitfall 1: Ambiguous Grammars

```gbnf
# BAD: Ambiguous - can match same string multiple ways
identifier ::= [a-z]+ | [a-zA-Z]+

# GOOD: Unambiguous
identifier ::= [a-zA-Z]+
```

### Pitfall 2: Left Recursion

```gbnf
# BAD: Infinite left recursion
expression ::= expression "+" number | number

# GOOD: Right recursion or iterative
expression ::= number ("+" number)*
```

### Pitfall 3: Overly Restrictive

```gbnf
# BAD: Too strict - rejects valid inputs
email ::= [a-z]+ "@" [a-z]+ ".com"  # Only .com domains!

# GOOD: More permissive
email ::= [a-zA-Z0-9._-]+ "@" [a-zA-Z0-9.-]+ "." [a-zA-Z]{2,}
```

### Pitfall 4: Missing Edge Cases

```gbnf
# BAD: Doesn't handle negative numbers
number ::= [0-9]+

# GOOD: Handles negatives, decimals, scientific notation
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
```

## Production Use Cases

### Function Calling

```python
# Define function call grammar
function_call_grammar = """
root ::= function-call

function-call ::= identifier "(" arguments ")"

identifier ::= "get_weather" | "search_web" | "send_email"

arguments ::= "{" ws arguments-list ws "}"

arguments-list ::= argument ("," ws argument)*

argument ::= string-arg | number-arg

string-arg ::= "\"" key "\"" ":" ws "\"" value "\""

number-arg ::= "\"" key "\"" ":" ws number

key ::= [a-zA-Z_]+

value ::= [^"]*

number ::= [0-9]+

ws ::= [ \\t\\n]*
"""

# Generate function call
response = model(
    "Call function to get weather in San Francisco",
    grammar=function_call_grammar
)

# Output: get_weather({"location": "San Francisco"})
```

### Structured Logging

```python
log_grammar = """
root ::= log-entry

log-entry ::= "[" timestamp "]" ws "[" level "]" ws message

timestamp ::= date "T" time

date ::= [0-9]{4} "-" [0-9]{2} "-" [0-9]{2}

time ::= [0-9]{2} ":" [0-9]{2} ":" [0-9]{2}

level ::= "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"

message ::= [^\\n]+

ws ::= [ \\t]*
"""

# Generates: [2024-01-15T10:30:45] [INFO] User logged in successfully
```

### Code Generation

```python
# Ensure generated code is syntactically valid
code = model(
    "Write a function to calculate fibonacci",
    grammar=python_function_grammar,
    max_tokens=512
)

# Guaranteed valid Python function syntax
exec(code)  # Safe to execute (syntax-wise)
```

## Interview Questions

1. **What problem does grammar-guided generation solve?**
   - Ensures outputs conform to specific formats
   - Eliminates parsing errors and invalid outputs
   - Enables reliable structured data extraction
   - Critical for production systems requiring parseable output

2. **How does grammar constraint affect sampling?**
   - Masks logits for invalid tokens (set to -inf)
   - Samples only from grammatically valid tokens
   - May reduce output diversity
   - Increases compute by ~10-50% for validation

3. **Design a grammar for a REST API endpoint path.**
   ```gbnf
   root ::= "/api/" version "/" resource ("/" id)?
   version ::= "v" [0-9]+
   resource ::= "users" | "posts" | "comments"
   id ::= [0-9]+
   ```

4. **What are the trade-offs of using grammars?**
   - ✅ Pros: Guaranteed valid output, eliminates parsing errors
   - ❌ Cons: Added compute overhead, reduced diversity, complexity
   - When to use: Production APIs, structured data, critical correctness
   - When to skip: Creative tasks, exploratory generation

5. **How would you optimize grammar evaluation for production?**
   - Cache valid tokens per grammar state
   - Precompute state transitions offline
   - Use simpler grammars when possible
   - Batch grammar evaluations
   - Profile and optimize hot paths

## Summary

Grammar-guided generation ensures reliable, structured outputs:

✅ **GBNF syntax**: Powerful grammar specification language
✅ **100% valid outputs**: Eliminates parsing errors
✅ **Production-ready**: Function calling, APIs, structured data
✅ **Trade-offs**: +10-50% overhead, but worth it for critical apps

In the next lesson, we'll explore **advanced sampling algorithms** for better generation quality.

---

**Next**: [05-advanced-sampling.md](./05-advanced-sampling.md)
