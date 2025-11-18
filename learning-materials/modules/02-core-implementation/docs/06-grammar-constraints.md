# Grammar Constraints and Structured Output

**Learning Module**: Module 2 - Core Implementation
**Estimated Reading Time**: 24 minutes
**Prerequisites**: Understanding of sampling strategies, basic formal grammars
**Related Content**:
- [Sampling Strategies](./05-sampling-strategies.md)
- [Inference Pipeline](./04-inference-pipeline.md)

---

## Overview

Grammar-guided generation constrains model output to follow specific formats like JSON, YAML, or custom structures. This is critical for building reliable AI systems that integrate with traditional software.

### Learning Objectives

After completing this lesson, you will:
- ✅ Understand GBNF (GGML BNF) grammar format
- ✅ Write grammars for structured output
- ✅ Implement JSON mode and function calling
- ✅ Debug grammar-related issues
- ✅ Build production-ready structured output systems

---

## Why Grammar Constraints?

### The Problem: Unreliable Output Format

Without constraints:

```python
prompt = "Generate a JSON object with name and age"
output = model.generate(prompt)

# Possible outputs (all problematic):
# 1. "Sure! Here's the JSON: {name: John, age: 30}"  # Extra text
# 2. "{name: 'John', age: '30'}"                      # Wrong quotes
# 3. "{'name': 'John', 'age': 30}"                    # Wrong quotes
# 4. "{name: John\nage: 30}"                          # Not valid JSON
# 5. "name: John, age: 30"                            # Not JSON at all
```

**Impact**:
- Must write brittle parsers
- High failure rate in production
- Expensive retry logic
- Poor user experience

### The Solution: Grammar-Guided Generation

With grammar constraints:

```python
grammar = r'''
root ::= object
object ::= "{" pair ("," pair)* "}"
pair ::= string ":" value
value ::= string | number
string ::= "\"" [^"]* "\""
number ::= [0-9]+
'''

output = model.generate(prompt, grammar=grammar)
# Guaranteed: {"name": "John", "age": 30}
# Always valid JSON!
```

**Benefits**:
- ✅ Guaranteed format compliance
- ✅ No parsing failures
- ✅ Reliable integration with code
- ✅ Better user experience
- ✅ Production-ready

---

## GBNF: GGML Backus-Naur Form

### Syntax

GBNF is llama.cpp's grammar specification language, based on Extended BNF:

```
# Rule definition
rule-name ::= expression

# Terminals (literal strings)
"literal"

# Character ranges
[a-z]       # Any lowercase letter
[A-Z]       # Any uppercase letter
[0-9]       # Any digit
[abc]       # Any of 'a', 'b', or 'c'

# Quantifiers
*           # Zero or more
+           # One or more
?           # Optional (zero or one)

# Grouping
( ... )     # Group expressions

# Alternation
|           # Or

# Negation
[^abc]      # Anything except 'a', 'b', or 'c'
```

### Simple Grammar Example

```gbnf
# Simple email grammar
root ::= email

email ::= local "@" domain

local ::= [a-zA-Z0-9._+-]+

domain ::= [a-zA-Z0-9.-]+ "." tld

tld ::= "com" | "org" | "net" | "edu"
```

**Matches**:
- john.doe@example.com ✅
- user_123@domain.org ✅
- admin+tag@test.edu ✅

**Doesn't Match**:
- invalid@email (no TLD) ❌
- @domain.com (no local) ❌
- user@domain (no TLD) ❌

---

## JSON Grammar

### Complete JSON Specification

```gbnf
root ::= value

value ::= object | array | string | number | boolean | null

# Object
object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair ::= string ws ":" ws value

# Array
array ::= "[" ws (value (ws "," ws value)*)? ws "]"

# String
string ::= "\"" ([^"\\] | "\\" escape-char)* "\""
escape-char ::= ["\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]

# Number
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+

# Boolean and null
boolean ::= "true" | "false"
null ::= "null"

# Whitespace
ws ::= [ \t\n\r]*
```

### JSON Schema to GBNF

Constrain to specific schema:

```python
# Schema
{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "email": {"type": "string"}
    },
    "required": ["name", "age"]
}

# Converted to GBNF
root ::= "{" ws
    "\"name\"" ws ":" ws string ws "," ws
    "\"age\"" ws ":" ws number ws
    ("," ws "\"email\"" ws ":" ws string ws)?
"}"

string ::= "\"" [^"]* "\""
number ::= [0-9]+
ws ::= [ \t\n]*
```

**Guarantees**:
- Object with exact keys
- Correct types for each field
- Required fields present
- Optional fields optional

---

## Advanced Grammars

### Nested Structures

```gbnf
# Nested objects
root ::= person

person ::= "{" ws
    "\"name\"" ws ":" ws string ws "," ws
    "\"age\"" ws ":" ws number ws "," ws
    "\"address\"" ws ":" ws address ws
"}"

address ::= "{" ws
    "\"street\"" ws ":" ws string ws "," ws
    "\"city\"" ws ":" ws string ws "," ws
    "\"zip\"" ws ":" ws number ws
"}"

string ::= "\"" [^"]* "\""
number ::= [0-9]+
ws ::= [ \t\n]*
```

**Output**:
```json
{
    "name": "John Doe",
    "age": 30,
    "address": {
        "street": "123 Main St",
        "city": "Springfield",
        "zip": 12345
    }
}
```

### Arrays with Specific Types

```gbnf
# Array of objects
root ::= "{" ws
    "\"users\"" ws ":" ws "[" ws
    (user (ws "," ws user)*)? ws
    "]" ws
"}"

user ::= "{" ws
    "\"id\"" ws ":" ws number ws "," ws
    "\"name\"" ws ":" ws string ws
"}"

string ::= "\"" [^"]* "\""
number ::= [0-9]+
ws ::= [ \t\n]*
```

### Enums and Fixed Values

```gbnf
# Status must be one of specific values
root ::= "{" ws
    "\"status\"" ws ":" ws status ws "," ws
    "\"code\"" ws ":" ws number ws
"}"

status ::= "\"pending\"" | "\"approved\"" | "\"rejected\""

number ::= [0-9]+
ws ::= [ \t\n]*
```

**Output**:
```json
{
    "status": "approved",
    "code": 200
}
```

---

## Function Calling

### OpenAI-Style Function Calling

Define functions the model can call:

```python
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]
```

**Grammar for Function Call**:
```gbnf
root ::= function-call

function-call ::= "{" ws
    "\"name\"" ws ":" ws function-name ws "," ws
    "\"arguments\"" ws ":" ws arguments ws
"}"

function-name ::= "\"get_weather\""

arguments ::= "{" ws
    "\"location\"" ws ":" ws string ws
    ("," ws "\"unit\"" ws ":" ws unit ws)?
"}"

unit ::= "\"celsius\"" | "\"fahrenheit\""

string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
```

**Model Output**:
```json
{
    "name": "get_weather",
    "arguments": {
        "location": "San Francisco",
        "unit": "celsius"
    }
}
```

### Multiple Functions

```gbnf
root ::= function-call

function-call ::= "{" ws
    "\"name\"" ws ":" ws function-name ws "," ws
    "\"arguments\"" ws ":" ws arguments ws
"}"

function-name ::= "\"get_weather\"" | "\"set_alarm\"" | "\"send_email\""

arguments ::= weather-args | alarm-args | email-args

weather-args ::= "{" ws
    "\"location\"" ws ":" ws string ws
"}"

alarm-args ::= "{" ws
    "\"time\"" ws ":" ws string ws "," ws
    "\"message\"" ws ":" ws string ws
"}"

email-args ::= "{" ws
    "\"to\"" ws ":" ws string ws "," ws
    "\"subject\"" ws ":" ws string ws "," ws
    "\"body\"" ws ":" ws string ws
"}"

string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
```

---

## Implementation in llama.cpp

### Grammar Parsing

```cpp
// From llama-grammar.cpp
struct llama_grammar {
    const std::vector<std::vector<llama_grammar_element>> rules;

    std::vector<const llama_grammar_element *> stacks;

    // Get current valid tokens
    std::vector<llama_token> get_valid_tokens(
        const struct llama_vocab & vocab
    ) const;
};

// Grammar element types
enum llama_gretype {
    LLAMA_GRETYPE_END            = 0,  // End of rule
    LLAMA_GRETYPE_ALT            = 1,  // Alternation (|)
    LLAMA_GRETYPE_RULE_REF       = 2,  // Reference to another rule
    LLAMA_GRETYPE_CHAR           = 3,  // Literal character
    LLAMA_GRETYPE_CHAR_NOT       = 4,  // Character negation [^...]
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,  // Upper bound of range
    LLAMA_GRETYPE_CHAR_ALT       = 6,  // Character alternatives
};
```

### Token Filtering

During sampling, grammar filters invalid tokens:

```cpp
// Apply grammar constraints
void llama_sample_grammar(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    const struct llama_grammar * grammar
) {
    // Get valid tokens according to grammar
    std::vector<llama_token> valid_tokens = grammar->get_valid_tokens(ctx->vocab);

    // Filter candidates to only valid tokens
    size_t n_valid = 0;
    for (size_t i = 0; i < candidates->size; i++) {
        const llama_token token = candidates->data[i].id;

        if (std::find(valid_tokens.begin(), valid_tokens.end(), token) != valid_tokens.end()) {
            candidates->data[n_valid++] = candidates->data[i];
        }
    }

    candidates->size = n_valid;
}
```

### Generation with Grammar

```cpp
// Load grammar from file
llama_grammar * grammar = llama_grammar_init_from_file("grammar.gbnf", "root");

// Generation loop
while (true) {
    // Get logits
    float * logits = llama_get_logits(ctx);

    // Build candidate list
    llama_token_data_array candidates = /* ... */;

    // Apply grammar constraints
    llama_sample_grammar(ctx, &candidates, grammar);

    // Apply other sampling (temperature, top-p, etc.)
    // ...

    // Sample token (now guaranteed to be valid)
    llama_token token = llama_sample_token(ctx, &candidates);

    // Update grammar state
    llama_grammar_accept_token(grammar, token);

    // Check if grammar is complete
    if (llama_grammar_is_complete(grammar)) {
        break;
    }
}

// Free grammar
llama_grammar_free(grammar);
```

---

## JSON Mode Implementation

### Built-in JSON Mode

llama.cpp provides built-in JSON mode:

```cpp
// Create grammar for JSON
llama_grammar * json_grammar = llama_grammar_init_json("root");

// Use during generation
// (same as custom grammar)
```

### Python API

```python
from llama_cpp import Llama, LlamaGrammar

llm = Llama(model_path="model.gguf")

# Load grammar
grammar = LlamaGrammar.from_file("grammar.gbnf")

# Generate with grammar
output = llm(
    "Generate user info as JSON:",
    grammar=grammar,
    max_tokens=100
)

print(output['choices'][0]['text'])
# Guaranteed valid JSON!
```

### JSON Schema Support

```python
from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

# Convert schema to grammar
grammar = LlamaGrammar.from_json_schema(schema)

# Generate
output = llm(
    "Create a user profile:",
    grammar=grammar
)
```

---

## Common Use Cases

### 1. Structured Data Extraction

```python
# Extract entities from text
grammar = '''
root ::= "{" ws
    "\"person\"" ws ":" ws string ws "," ws
    "\"organization\"" ws ":" ws string ws "," ws
    "\"location\"" ws ":" ws string ws
"}"
string ::= "\"" [^"]* "\""
ws ::= [ \t\n]*
'''

prompt = "Extract entities: John works at OpenAI in San Francisco"
output = generate_with_grammar(prompt, grammar)
# {"person": "John", "organization": "OpenAI", "location": "San Francisco"}
```

### 2. SQL Query Generation

```gbnf
root ::= select-statement

select-statement ::= "SELECT" ws columns ws "FROM" ws table ws (where-clause)?

columns ::= "*" | (column-name (ws "," ws column-name)*)

column-name ::= [a-zA-Z_][a-zA-Z0-9_]*

table ::= [a-zA-Z_][a-zA-Z0-9_]*

where-clause ::= "WHERE" ws condition

condition ::= column-name ws operator ws value

operator ::= "=" | ">" | "<" | ">=" | "<=" | "!="

value ::= "'" [^']* "'" | [0-9]+

ws ::= [ \t\n]+
```

### 3. API Response Format

```gbnf
root ::= api-response

api-response ::= "{" ws
    "\"status\"" ws ":" ws status ws "," ws
    "\"data\"" ws ":" ws data ws "," ws
    "\"message\"" ws ":" ws string ws
"}"

status ::= number

data ::= "null" | object | array

object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair ::= string ws ":" ws value
value ::= string | number | boolean | "null"

array ::= "[" ws (value (ws "," ws value)*)? ws "]"

string ::= "\"" [^"]* "\""
number ::= [0-9]+
boolean ::= "true" | "false"
ws ::= [ \t\n]*
```

### 4. Configuration File Generation

```gbnf
# YAML-style config
root ::= config

config ::= (config-line)*

config-line ::= key ws ":" ws value ws "\n"

key ::= [a-zA-Z_][a-zA-Z0-9_]*

value ::= string | number | boolean

string ::= "\"" [^"]* "\""
number ::= [0-9]+
boolean ::= "true" | "false"
ws ::= [ \t]*
```

---

## Performance Considerations

### Grammar Complexity

```
Simple grammar (JSON):
  - Fast parsing
  - Minimal overhead
  - Good for production

Complex grammar (nested, many rules):
  - Slower parsing
  - Higher memory usage
  - May reduce throughput by 20-30%

Trade-off: Reliability vs Speed
```

### Token Filtering Cost

```cpp
// Each generation step:
// 1. Generate logits (expensive)
// 2. Get valid tokens from grammar (moderate)
// 3. Filter logits (cheap)
// 4. Sample (cheap)

// Grammar adds ~5-10% overhead in most cases
// Worth it for guaranteed format compliance
```

---

## Debugging Grammar Issues

### Problem: No Valid Tokens

```
Error: Grammar rejected all tokens

Causes:
  - Grammar too restrictive
  - Grammar doesn't match current state
  - Missing rule definitions

Debug:
  1. Simplify grammar
  2. Test grammar in isolation
  3. Print valid tokens at each step
```

### Problem: Invalid Output Still Generated

```
Output doesn't match expected format despite grammar

Causes:
  - Grammar has bugs (too permissive)
  - Root rule not correctly specified
  - Whitespace handling issues

Debug:
  1. Test grammar with known inputs
  2. Verify root rule
  3. Check whitespace rules
```

### Testing Grammars

```python
def test_grammar(grammar_str, test_cases):
    """Test grammar against known inputs"""
    grammar = load_grammar(grammar_str)

    for input_str, should_match in test_cases:
        matches = grammar.matches(input_str)
        assert matches == should_match, f"Failed: {input_str}"

# Example
test_cases = [
    ('{"name": "John", "age": 30}', True),
    ('{name: "John", age: 30}', False),  # Missing quotes
    ('{"name": "John"}', False),         # Missing required field
]

test_grammar(json_grammar, test_cases)
```

---

## Interview Questions

**Q1: How does grammar-guided generation work at the token level?**

**Answer**: At each generation step, the grammar parser determines which tokens are valid based on the current parse state. During sampling, only these valid tokens are considered, forcing the model to follow the grammar. The grammar state updates after each token is generated, constraining subsequent tokens. This guarantees output matches the grammar specification.

**Q2: What are the trade-offs of using grammar constraints?**

**Answer**:
- Pros: Guaranteed format, no parsing failures, reliable integration
- Cons: 5-10% performance overhead, reduces model's flexibility, requires grammar design
- Use when: Format compliance is critical (APIs, data extraction)
- Avoid when: Creative generation needed, performance critical

**Q3: How would you convert a JSON schema to a GBNF grammar?**

**Answer**:
1. Map `object` to GBNF object rules with specific keys
2. Map `array` to GBNF array with item type constraints
3. Map `string`, `number`, `boolean` to corresponding terminals
4. Handle `required` fields (always present) vs optional (with `?`)
5. Map `enum` to alternation (`|`)
6. Recursively handle nested schemas

---

## Further Reading

### Code References
- [llama-grammar.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp): Grammar implementation
- [llama-sampling.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampling.cpp): Grammar integration

### Examples
- [grammars/](https://github.com/ggml-org/llama.cpp/tree/master/grammars): Built-in grammars
- [JSON grammar](https://github.com/ggml-org/llama.cpp/blob/master/grammars/json.gbnf)
- [Chess PGN grammar](https://github.com/ggml-org/llama.cpp/blob/master/grammars/chess.gbnf)

### Tutorials
- [Lab 4: Structured Output](../labs/lab-04-custom-sampling.ipynb)
- [Tutorial: Grammar-Guided Generation](../tutorials/tutorial-03-grammar-generation.ipynb)
- [Code Example: JSON Mode](../code/json_mode_example.py)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Module**: 2 - Core Implementation
