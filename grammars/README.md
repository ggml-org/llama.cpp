# GBNF Guide

GBNF (GGML BNF) is a format for defining [formal grammars](https://en.wikipedia.org/wiki/Formal_grammar) to constrain model outputs in `llama.cpp`. For example, you can use it to force the model to generate valid JSON, or speak only in emojis. GBNF grammars are supported in various ways in `examples/main` and `examples/server`.

## Background

[Bakus-Naur Form (BNF)](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form) is a notation for describing the syntax of formal languages like programming languages, file formats, and protocols. GBNF is an extension of BNF that primarily adds a few modern regex-like features.

## Basics

In GBNF, we define *production rules* that specify how a *non-terminal* (rule name) can be replaced with sequences of *terminals* (characters, specifically Unicode [code points](https://en.wikipedia.org/wiki/Code_point)) and other non-terminals. The basic format of a production rule is `nonterminal ::= sequence...`.

## Example

Before going deeper, let's look at some of the features demonstrated in `grammars/chess.gbnf`, a small chess notation grammar:
```
# `root` specifies the pattern for the overall output
root ::= (
    # it must start with the characters "1. " followed by a sequence
    # of characters that match the `move` rule, followed by a space, followed
    # by another move, and then a newline
    "1. " move " " move "\n"

    # it's followed by one or more subsequent moves, numbered with one or two digits
    ([1-9] [0-9]? ". " move " " move "\n")+
)

# `move` is an abstract representation, which can be a pawn, nonpawn, or castle.
# The `[+#]?` denotes the possibility of checking or mate signs after moves
move ::= (pawn | nonpawn | castle) [+#]?

pawn ::= ...
nonpawn ::= ...
castle ::= ...
```

## Non-Terminals and Terminals

Non-terminal symbols (rule names) stand for a pattern of terminals and other non-terminals. They are required to be a dashed lowercase word, like `move`, `castle`, or `check-mate`.

Terminals are actual characters ([code points](https://en.wikipedia.org/wiki/Code_point)). They can be specified as a sequence like `"1"` or `"O-O"` or as ranges like `[1-9]` or `[NBKQR]`.

## Characters and character ranges

Terminals support the full range of Unicode. Unicode characters can be specified directly in the grammar, for example `hiragana ::= [ぁ-ゟ]`, or with escapes: 8-bit (`\xXX`), 16-bit (`\uXXXX`) or 32-bit (`\UXXXXXXXX`).

Character ranges can be negated with `^`:
```
single-line ::= [^\n]+ "\n"`
```

## Sequences and Alternatives

The order of symbols in a sequence matters. For example, in `"1. " move " " move "\n"`, the `"1. "` must come before the first `move`, etc.

Alternatives, denoted by `|`, give different sequences that are acceptable. For example, in `move ::= pawn | nonpawn | castle`, `move` can be a `pawn` move, a `nonpawn` move, or a `castle`.

Parentheses `()` can be used to group sequences, which allows for embedding alternatives in a larger rule or applying repetition and optional symbols (below) to a sequence.

## Repetition and Optional Symbols

- `*` after a symbol or sequence means that it can be repeated zero or more times (equivalent to `{0,}`).
- `+` denotes that the symbol or sequence should appear one or more times (equivalent to `{1,}`).
- `?` makes the preceding symbol or sequence optional (equivalent to `{0,1}`).
- `{m}` repeats the precedent symbol or sequence exactly `m` times
- `{m,}` repeats the precedent symbol or sequence at least `m` times
- `{m,n}` repeats the precedent symbol or sequence at between `m` and `n` times (included)
- `{0,n}` repeats the precedent symbol or sequence at most `n` times (included)

## Comments and newlines

Comments can be specified with `#`:
```
# defines optional whitespace
ws ::= [ \t\n]+
```

Newlines are allowed between rules and between symbols or sequences nested inside parentheses. Additionally, a newline after an alternate marker `|` will continue the current rule, even outside of parentheses.

## The root rule

In a full grammar, the `root` rule always defines the starting point of the grammar. In other words, it specifies what the entire output must match.

```
# a grammar for lists
root ::= ("- " item)+
item ::= [^\n]+ "\n"
```

## Next steps

This guide provides a brief overview. Check out the GBNF files in this directory (`grammars/`) for examples of full grammars. You can try them out with:
```
./llama-cli -m <model> --grammar-file grammars/some-grammar.gbnf -p 'Some prompt'
```

`llama.cpp` can also convert JSON schemas to grammars either ahead of time or at each request, see below.

## Troubleshooting

Grammars currently have performance gotchas (see https://github.com/ggerganov/llama.cpp/issues/4218).

### Efficient optional repetitions

A common pattern is to allow repetitions of a pattern `x` up to N times.

While semantically correct, the syntax `x? x? x?.... x?` (with N repetitions) may result in extremely slow sampling. Instead, you can write `x{0,N}` (or `(x (x (x ... (x)?...)?)?)?` w/ N-deep nesting in earlier llama.cpp versions).

## Using GBNF grammars

You can use GBNF grammars:

- In [llama-server](../examples/server)'s completion endpoints, passed as the `grammar` body field
- In [llama-cli](../examples/main), passed as the `--grammar` & `--grammar-file` flags
- With [llama-gbnf-validator](../examples/gbnf-validator) tool, to test them against strings.

## JSON Schemas → GBNF

`llama.cpp` supports converting a subset of https://json-schema.org/ to GBNF grammars:

- In [llama-server](../examples/server):
    - For any completion endpoints, passed as the `json_schema` body field
    - For the `/chat/completions` endpoint, passed inside the `result_format` body field (e.g. `{"type", "json_object", "schema": {"items": {}}}`)
- In [llama-cli](../examples/main), passed as the `--json` / `-j` flag
- To convert to a grammar ahead of time:
    - in CLI, with [examples/json_schema_to_grammar.py](../examples/json_schema_to_grammar.py)
    - in JavaScript with [json-schema-to-grammar.mjs](../examples/server/public/json-schema-to-grammar.mjs) (this is used by the [server](../examples/server)'s Web UI)

Take a look at [tests](../tests/test-json-schema-to-grammar.cpp) to see which features are likely supported (you'll also find usage examples in https://github.com/ggerganov/llama.cpp/pull/5978, https://github.com/ggerganov/llama.cpp/pull/6659 & https://github.com/ggerganov/llama.cpp/pull/6555).

```bash
llama-cli \
  -hfr bartowski/Phi-3-medium-128k-instruct-GGUF \
  -hff Phi-3-medium-128k-instruct-Q8_0.gguf \
  -j '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    },
    "minItems": 10,
    "maxItems": 100
  }' \
  -p 'Generate a {name, age}[] JSON array with famous actors of all ages.'
```

<details>

<summary>Show grammar</summary>

You can convert any schema in command-line with:

```bash
examples/json_schema_to_grammar.py name-age-schema.json
```

```
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
item ::= "{" space item-name-kv "," space item-age-kv "}" space
item-age ::= ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-4] [0-9] | [5] "0")) space
item-age-kv ::= "\"age\"" space ":" space item-age
item-name ::= "\"" char{1,100} "\"" space
item-name-kv ::= "\"name\"" space ":" space item-name
root ::= "[" space item ("," space item){9,99} "]" space
space ::= | " " | "\n" [ \t]{0,20}
```

</details>

Here is also a list of known limitations (contributions welcome):

- Unsupported features are skipped silently. It is currently advised to use the command-line Python converter (see above) to see any warnings, and to inspect the resulting grammar / test it w/ [llama-gbnf-validator](../examples/gbnf-validator/gbnf-validator.cpp).
- Can't mix `properties` w/ `anyOf` / `oneOf` in the same type (https://github.com/ggerganov/llama.cpp/issues/7703)
- [prefixItems](https://json-schema.org/draft/2020-12/json-schema-core#name-prefixitems) is broken (but [items](https://json-schema.org/draft/2020-12/json-schema-core#name-items) works)
- `minimum`, `exclusiveMinimum`, `maximum`, `exclusiveMaximum`: only supported for `"type": "integer"` for now, not `number`
- Nested `$ref`s are broken (https://github.com/ggerganov/llama.cpp/issues/8073)
- [pattern](https://json-schema.org/draft/2020-12/json-schema-validation#name-pattern)s must start with `^` and end with `$`
- Remote `$ref`s not supported in the C++ version (Python & JavaScript versions fetch https refs)
- `string` [formats](https://json-schema.org/draft/2020-12/json-schema-validation#name-defined-formats) lack `uri`, `email`
- No [`patternProperties`](https://json-schema.org/draft/2020-12/json-schema-core#name-patternproperties)

And a non-exhaustive list of other unsupported features that are unlikely to be implemented (hard and/or too slow to support w/ stateless grammars):

- [`uniqueItems`](https://json-schema.org/draft/2020-12/json-schema-validation#name-uniqueitems)
- [`contains`](https://json-schema.org/draft/2020-12/json-schema-core#name-contains) / `minContains`
- `$anchor` (cf. [dereferencing](https://json-schema.org/draft/2020-12/json-schema-core#name-dereferencing))
- [`not`](https://json-schema.org/draft/2020-12/json-schema-core#name-not)
- [Conditionals](https://json-schema.org/draft/2020-12/json-schema-core#name-keywords-for-applying-subsche) `if` / `then` / `else` / `dependentSchemas`

### A word about additionalProperties

> [!WARNING]
> By default, `object`s accept [additional properties](https://json-schema.org/understanding-json-schema/reference/object#additionalproperties), which you might not want / not expect, and which will make sampling slower (not just because of the extra tokens, but also generates a slower grammar).
> You can set `"additionalProperties": false` on the schema of any object to ensure only properties listed in `properties` are generated (not needed for non-`object` types, e.g. `array` or `string`).

If you're using [Pydantic](https://pydantic.dev/) to generate schemas, you can disable additional properties with the `extra` config on each model class:

```python
# pip install pydantic
import json
from typing import Annotated, List
from pydantic import BaseModel, Extra, Field
class QAPair(BaseModel):
    class Config:
        extra = 'forbid'  # triggers additionalProperties: false in the JSON schema
    question: str
    concise_answer: str
    justification: str

class Summary(BaseModel):
    class Config:
        extra = 'forbid'
    key_facts: List[Annotated[str, Field(pattern='- .{5,}')]]
    question_answers: List[Annotated[List[QAPair], Field(min_items=5)]]

print(json.dumps(Summary.model_json_schema(), indent=2))
```

<details>
<summary>Show JSON schema & grammar</summary>

```json
{
  "$defs": {
    "QAPair": {
      "additionalProperties": false,
      "properties": {
        "question": {
          "title": "Question",
          "type": "string"
        },
        "concise_answer": {
          "title": "Concise Answer",
          "type": "string"
        },
        "justification": {
          "title": "Justification",
          "type": "string"
        }
      },
      "required": [
        "question",
        "concise_answer",
        "justification"
      ],
      "title": "QAPair",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "properties": {
    "key_facts": {
      "items": {
        "pattern": "^- .{5,}$",
        "type": "string"
      },
      "title": "Key Facts",
      "type": "array"
    },
    "question_answers": {
      "items": {
        "items": {
          "$ref": "#/$defs/QAPair"
        },
        "minItems": 5,
        "type": "array"
      },
      "title": "Question Answers",
      "type": "array"
    }
  },
  "required": [
    "key_facts",
    "question_answers"
  ],
  "title": "Summary",
  "type": "object"
}
```

```
QAPair ::= "{" space QAPair-question-kv "," space QAPair-concise-answer-kv "," space QAPair-justification-kv "}" space
QAPair-concise-answer-kv ::= "\"concise_answer\"" space ":" space string
QAPair-justification-kv ::= "\"justification\"" space ":" space string
QAPair-question-kv ::= "\"question\"" space ":" space string
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
dot ::= [^\x0A\x0D]
key-facts ::= "[" space (key-facts-item ("," space key-facts-item)*)? "]" space
key-facts-item ::= "\"" "- " key-facts-item-1{5,} "\"" space
key-facts-item-1 ::= dot
key-facts-kv ::= "\"key_facts\"" space ":" space key-facts
question-answers ::= "[" space (question-answers-item ("," space question-answers-item)*)? "]" space
question-answers-item ::= "[" space question-answers-item-item ("," space question-answers-item-item){4,} "]" space
question-answers-item-item ::= QAPair
question-answers-kv ::= "\"question_answers\"" space ":" space question-answers
root ::= "{" space key-facts-kv "," space question-answers-kv "}" space
space ::= | " " | "\n" [ \t]{0,20}
string ::= "\"" char* "\"" space
```

</details>

If you're using [Zod](https://zod.dev/), you can make your objects explicitly strict w/ `z.object(...).strict()` or `z.strictObject(...)`.

Note however that [zod-to-json-schema](https://github.com/StefanTerdell/zod-to-json-schema) currently always seems to set `"additionalProperties": false` anyway (even w/ zod schemas on which `nonstrict()` / `passthrough()` was called).

```js
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

const Foo = z.object({
  age: z.number().positive(),
  email: z.string().email(),
}).strict();

console.log(zodToJsonSchema(Foo));
```

<details>
<summary>Show JSON schema & grammar</summary>

```json
{
  "type": "object",
  "properties": {
    "age": {
      "type": "number",
      "exclusiveMinimum": 0
    },
    "email": {
      "type": "string",
      "format": "email"
    }
  },
  "required": [
    "age",
    "email"
  ],
  "additionalProperties": false,
  "$schema": "http://json-schema.org/draft-07/schema#"
}
```

```
age-kv ::= "\"age\"" space ":" space number
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
decimal-part ::= [0-9]{1,16}
email-kv ::= "\"email\"" space ":" space string
integral-part ::= [0] | [1-9] [0-9]{0,15}
number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
root ::= "{" space age-kv "," space email-kv "}" space
space ::= | " " | "\n" [ \t]{0,20}
string ::= "\"" char* "\"" space
```

</details>
