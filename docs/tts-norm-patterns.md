# TTS normalization patterns

Text-to-speech models often need their input normalized before tokenization: numbers, currencies, dates and symbols spelled out, punctuation mapped, spacing fixed. The exact rules are model-specific (they must match what the model was trained on), so they live in the model's `tts` task template rather than in C++.

Jinja has no regex support, and exposing `std::regex` / Python `re` would not help: the two differ in many details (`\d`, `\w`, `\b`, lookbehind, case folding, backtracking limits), so a template could give different results in llama.cpp and in Python. Instead, task templates get two functions with a small matching language whose semantics are defined here:

- `tts_norm_scan(text, rules)`: split text into matched and unmatched segments
- `tts_norm_sub(text, pattern, repl)`: replace every match

Implementations:
- C++: [common/chat-tts-norm.cpp](../common/chat-tts-norm.cpp), injected into task templates by `common_chat_template_render()` in [common/chat.cpp](../common/chat.cpp). They are not available to chat templates.
- Python reference: [scripts/chat_tts_norm.py](../scripts/chat_tts_norm.py), for use with `jinja2` (`env.globals["tts_norm_scan"] = chat_tts_norm.tts_norm_scan`)
- Both are checked against each other by `tests/test-jinja.cpp` (`test-jinja -py "tts norm patterns.*"`)

## Example

```jinja
{%- for s in tts_norm_scan(text, [
    ['money',   '%$(%d+)(?:%.(%d%d))?'],
    ['percent', '(%d+)%%'],
    ['time',    '(%d%d?):(%d%d)(?!%d)'],
    ['neg',     '(?<!%w)-(%d+)'],
    ['number',  '%d+'],
]) -%}
  {%- if   s.rule == 'money'   -%}{%- set whole, cents = s.groups -%}{{ num(whole) }} dollars{{ ' and ' ~ num(cents) ~ ' cents' if cents else '' }}
  {%- elif s.rule == 'percent' -%}{{ num(s.groups[0]) }} percent
  {%- elif s.rule == 'time'    -%}{%- set h, m = s.groups -%}{{ num(h) }} {{ num(m) }}
  {%- elif s.rule == 'neg'     -%}minus {{ num(s.groups[0]) }}
  {%- elif s.rule == 'number'  -%}{{ num(s.text) }}
  {%- else -%}{{ s.text }}
  {%- endif -%}
{%- endfor -%}
```

`It costs $3.50, 20% off, at 10:45, -5 degrees, 7 cats.` becomes `It costs three dollars and fifty cents, twenty percent off, at ten forty-five, minus five degrees, seven cats.` (with a `num` macro that spells numbers, see the "mini text normalizer" test in `tests/test-jinja.cpp`).

## Functions

### `tts_norm_scan(text, rules)`

`rules` is one of:
- a list of `[name, pattern]` pairs (recommended, the order is explicit)
- a dict `{name: pattern}`, in insertion order
- a single pattern string, the rule is named `match`

Scanning starts at the first codepoint. At each position, the rules are tried in order and the first one that produces a **non-empty** match wins; the scan continues after the match. If no rule matches, one codepoint is skipped. Matches never overlap.

The result is a list of segments covering the whole text, in order. Consecutive unmatched codepoints are merged into one segment. Each segment is a dict:

| key | matched segment | unmatched segment |
|---|---|---|
| `rule` | rule name | `none` |
| `text` | matched text | unmatched text |
| `groups` | list of capture groups, in order of their `(` | `[]` |

There are no named groups, unpack `groups` instead: `{% set whole, cents = s.groups %}`.

A group that did not participate in the match (e.g. inside a `?` that matched nothing) is an empty string, so it renders as nothing and is falsy.

### `tts_norm_sub(text, pattern, repl)`

Replaces every match of `pattern`, with the same scanning rules as `tts_norm_scan`. `repl` is a literal string with references:

| reference | value |
|---|---|
| `%0` | whole match |
| `%1` .. `%9` | group |
| `%%` | a literal `%` |

Any other `%` sequence in `repl`, or a reference to a group that does not exist, is an error.

## Pattern syntax

Patterns operate on Unicode codepoints. Invalid UTF-8 bytes (C++ only) are seen as U+FFFD, one per byte, and preserved in the output.

| syntax | meaning |
|---|---|
| `x` | the literal codepoint `x` |
| `%x` | escape: `%` followed by any non-alphanumeric codepoint is that literal codepoint (`%.`, `%$`, `%%`, `%(`, ...) |
| `.` | any codepoint, including newlines |
| `%d` `%s` `%p` `%a` `%w` `%u` `%l` `%h` `%c` | character classes, see below; the uppercase form (`%D`, ...) is the negation |
| `[...]` | set: literal codepoints, ranges `a-z`, classes `%d`; `[^...]` is the negation |
| `(...)` | capture group |
| `(?:...)` | group without capture |
| `a\|b` | ordered alternation |
| `?` `*` `+` `{m}` `{m,}` `{m,n}` | quantifiers, greedy and possessive; `m`, `n` up to 1000 |
| `^` `$` | start / end of the whole text |
| `%b` | boundary: between a `%w` codepoint and a non-`%w` codepoint (the edges of the text count as non-`%w`) |
| `(?=...)` `(?!...)` | lookahead, positive / negative |
| `(?<=c)` `(?<!c)` | lookbehind of exactly one codepoint: `c` is a literal, a set, a class or `.` |
| `(?i)` | at the very start of the pattern only: ignore ASCII case |

These characters must be escaped with `%` to be matched literally: `( ) [ ] { } . | ? * + ^ $ %`. Inside a set, only `]`, `%`, `-` (between two codepoints) and `^` (first position) are special.

### Matching semantics

The matcher is a PEG, not a backtracking regex:

- **Ordered alternation**: alternatives are tried left to right and the first one that matches is committed to. If what follows fails, the other alternatives are not retried. `(?:a|ab)c` never matches `abc`; write `(?:ab|a)c`.
- **Possessive quantifiers**: quantifiers match as many repetitions as possible and never give any back. `%d+0` never matches; `.*x` never matches. Write patterns that describe exactly what to consume.
- A repetition that matches the empty string ends the loop.
- Captures inside lookaheads are discarded. A capture inside a repeated group holds its last repetition.

This gives a simple and linear-time implementation with identical results in C++ and Python.

### Character classes

Classes are defined by explicit codepoint ranges, not by a Unicode database, so the result does not depend on the Unicode version of the platform.

| class | definition |
|---|---|
| `%d` | ASCII digits `0-9` only |
| `%s` | U+0009..U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000 |
| `%p` | punctuation and symbols: ASCII punctuation, U+00A1..U+00BF (except U+00AA, U+00B5, U+00BA), U+00D7, U+00F7, U+2010..U+2027, U+2030..U+205E, U+20A0..U+20CF (currency), U+2100..U+2BFF (letterlike, number forms, arrows, math, technical, box drawing, shapes, misc symbols, dingbats), CJK punctuation (U+3001..U+3003, U+3008..U+3020, U+3030, U+303D, U+30FB), U+FE10..U+FE19, U+FE30..U+FE6B, fullwidth punctuation (U+FF01..U+FF0F, U+FF1A..U+FF20, U+FF3B..U+FF40, U+FF5B..U+FF65), U+1F000..U+1FAFF (emoji) |
| `%c` | control and invisible format characters: U+0000..U+001F, U+007F..U+009F, U+200B..U+200F, U+2060..U+206F, U+FEFF |
| `%a` | ASCII letters, U+00AA, U+00B5, U+00BA, and every codepoint from U+00C0 that is not in `%s`, `%p` or `%c` |
| `%w` | `%a` or `%d` |
| `%u` / `%l` | ASCII uppercase / lowercase letters only |
| `%h` | Han: U+2E80..U+2FDF, U+3005, U+3007, U+3021..U+3029, U+3038..U+303B, U+3400..U+4DBF, U+4E00..U+9FFF, U+F900..U+FAFF, U+20000..U+323AF |

Classes may overlap (e.g. U+0085 is both `%s` and `%c`). `%a` intentionally includes all scripts, combining marks and CJK, so `(?<!%w)` and `%b` work as "not inside a word" for any language.

## Errors

Invalid patterns are rejected when the function is called, with a message and a position, for example: unbalanced parentheses, a quantifier with nothing to repeat, multiple quantifiers (`a+?`), repeating a zero-width assertion (`%b+`), unknown escapes (`%q`), unescaped `]` or `}`, an empty set, a lookbehind wider than one codepoint and `(?i)` not at the start.
