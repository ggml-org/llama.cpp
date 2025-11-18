# Lab 3: Grammar Design and Testing

**Module 5 - Advanced Inference**
**Estimated Time**: 2-3 hours
**Difficulty**: Advanced

## Objectives

- Write GBNF grammars for common formats (JSON, XML, SQL)
- Validate outputs against grammars
- Optimize grammar performance
- Build structured output generators
- Handle complex nested structures

## Part 1: Basic Grammar Writing (30 min)

### Exercise 1: Email Validator

Write a grammar to validate email addresses:

```gbnf
root ::= email
email ::= local "@" domain
local ::= [a-zA-Z0-9._-]+
domain ::= subdomain ("." subdomain)+
subdomain ::= [a-zA-Z0-9-]+
```

**Test cases**:
- `user@example.com` → Valid
- `test@domain.co.uk` → Valid
- `invalid@` → Invalid

### Exercise 2: Phone Numbers

Write grammar for US phone numbers:
- Format: (XXX) XXX-XXXX
- Optional country code: +1

**Deliverable**: Working GBNF grammar + test cases

## Part 2: JSON Schema (45 min)

### Exercise 3: User Profile Schema

Generate valid JSON for user profiles:

```json
{
  "name": "string",
  "age": number,
  "email": "string",
  "role": "admin" | "user" | "guest"
}
```

Convert to GBNF:

```gbnf
root ::= user-object

user-object ::=
  "{" ws
    "\"name\"" ws ":" ws string ws "," ws
    "\"age\"" ws ":" ws number ws "," ws
    "\"email\"" ws ":" ws string ws "," ws
    "\"role\"" ws ":" ws role ws
  "}"

string ::= "\"" [a-zA-Z ]+ "\""
number ::= [0-9]+
role ::= "\"admin\"" | "\"user\"" | "\"guest\""
ws ::= [ \t\n]*
```

**Task**: Test with llama.cpp and verify 100% valid JSON

## Part 3: Code Generation (45 min)

### Exercise 4: Python Function Grammar

Ensure generated Python code is syntactically valid:

```gbnf
root ::= function

function ::= "def" ws identifier "(" parameters ")" ":" ws
             indent statement+ dedent

identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*
parameters ::= (identifier ("," ws identifier)*)?
statement ::= assignment | return-stmt | expression
assignment ::= identifier ws "=" ws expression "\n"
return-stmt ::= "return" ws expression "\n"
expression ::= identifier | number | string
number ::= [0-9]+
string ::= "\"" [^"]* "\""
ws ::= [ \t]*
indent ::= "    "
dedent ::= ""
```

**Test**: Generate 10 functions, verify all parse correctly

## Part 4: Performance Optimization (30 min)

### Exercise 5: Grammar Profiling

Benchmark grammar evaluation overhead:

```bash
# Without grammar
time ./llama-cli -m model.gguf -p "Generate JSON" -n 100

# With grammar
time ./llama-cli -m model.gguf -p "Generate JSON" -n 100 \
  --grammar-file json.gbnf
```

**Measure**: Overhead percentage

**Optimize**:
- Reduce grammar complexity
- Cache validation states
- Simplify alternatives

## Part 5: Production Use Cases (30 min)

### Exercise 6: API Response Generator

Build grammar for REST API responses:

```gbnf
root ::= api-response

api-response ::=
  "{" ws
    "\"status\"" ws ":" ws status ws "," ws
    "\"data\"" ws ":" ws data ws "," ws
    "\"message\"" ws ":" ws string ws
  "}"

status ::= "200" | "400" | "404" | "500"
data ::= array | "null"
array ::= "[" ws (object ("," ws object)*)? ws "]"
object ::= "{" ws "\"id\"" ws ":" ws number ws "}"
string ::= "\"" [^"]* "\""
number ::= [0-9]+
ws ::= [ \t\n]*
```

**Deliverable**: Working API response generator

## Deliverables

1. **Grammar Library**:
   - 5+ production-ready grammars
   - Documentation for each
   - Test cases with validation

2. **Performance Report**:
   - Grammar evaluation overhead
   - Optimization strategies applied
   - Before/after benchmarks

3. **Use Case Examples**:
   - JSON API responses
   - Code generation (Python, SQL)
   - Structured data extraction

## Evaluation

- **Correctness** (40%): Grammars work correctly
- **Coverage** (30%): Handle edge cases
- **Performance** (20%): Optimized grammars
- **Documentation** (10%): Clear, complete docs

## Extensions

- XML grammar
- Custom DSL
- Grammar debugging tools
- Auto-complete suggestions

See code example: `../code/grammar_parser.py`
