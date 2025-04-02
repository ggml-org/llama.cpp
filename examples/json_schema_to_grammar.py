#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from typing import Any, List, Optional, Set, Tuple, Union

def _build_repetition(item_rule, min_items, max_items, separator_rule=None):

    if min_items == 0 and max_items == 1:
        return f'{item_rule}?'

    if not separator_rule:
        if min_items == 1 and max_items is None:
            return f'{item_rule}+'
        elif min_items == 0 and max_items is None:
            return f'{item_rule}*'
        else:
            return f'{item_rule}{{{min_items},{max_items if max_items is not None else ""}}}'

    result = item_rule + ' ' + _build_repetition(f'({separator_rule} {item_rule})', min_items - 1 if min_items > 0 else 0, max_items - 1 if max_items is not None else None)
    return f'({result})?' if min_items == 0 else result

def _generate_min_max_int(min_value: Optional[int], max_value: Optional[int], out: list, decimals_left: int = 16, top_level: bool = True):
    has_min = min_value != None
    has_max = max_value != None

    def digit_range(from_char: str, to_char: str):
        out.append("[")
        if from_char == to_char:
            out.append(from_char)
        else:
            out.append(from_char)
            out.append("-")
            out.append(to_char)
        out.append("]")

    def more_digits(min_digits: int, max_digits: int):
        out.append("[0-9]")
        if min_digits == max_digits and min_digits == 1:
            return
        out.append("{")
        out.append(str(min_digits))
        if max_digits != min_digits:
            out.append(",")
            if max_digits != sys.maxsize:
                out.append(str(max_digits))
        out.append("}")

    def uniform_range(from_str: str, to_str: str):
        i = 0
        while i < len(from_str) and from_str[i] == to_str[i]:
            i += 1
        if i > 0:
            out.append("\"")
            out.append(from_str[:i])
            out.append("\"")
        if i < len(from_str):
            if i > 0:
                out.append(" ")
            sub_len = len(from_str) - i - 1
            if sub_len > 0:
                from_sub = from_str[i+1:]
                to_sub = to_str[i+1:]
                sub_zeros = "0" * sub_len
                sub_nines = "9" * sub_len

                to_reached = False
                out.append("(")
                if from_sub == sub_zeros:
                    digit_range(from_str[i], chr(ord(to_str[i]) - 1))
                    out.append(" ")
                    more_digits(sub_len, sub_len)
                else:
                    out.append("[")
                    out.append(from_str[i])
                    out.append("] ")
                    out.append("(")
                    uniform_range(from_sub, sub_nines)
                    out.append(")")
                    if ord(from_str[i]) < ord(to_str[i]) - 1:
                        out.append(" | ")
                        if to_sub == sub_nines:
                            digit_range(chr(ord(from_str[i]) + 1), to_str[i])
                            to_reached = True
                        else:
                            digit_range(chr(ord(from_str[i]) + 1), chr(ord(to_str[i]) - 1))
                        out.append(" ")
                        more_digits(sub_len, sub_len)
                if not to_reached:
                    out.append(" | ")
                    digit_range(to_str[i], to_str[i])
                    out.append(" ")
                    uniform_range(sub_zeros, to_sub)
                out.append(")")
            else:
                out.append("[")
                out.append(from_str[i])
                out.append("-")
                out.append(to_str[i])
                out.append("]")

    if has_min and has_max:
        if min_value < 0 and max_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(-max_value, -min_value, out, decimals_left, top_level=True)
            out.append(")")
            return

        if min_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(0, -min_value, out, decimals_left, top_level=True)
            out.append(") | ")
            min_value = 0

        min_s = str(min_value)
        max_s = str(max_value)
        min_digits = len(min_s)
        max_digits = len(max_s)

        for digits in range(min_digits, max_digits):
            uniform_range(min_s, "9" * digits)
            min_s = "1" + "0" * digits
            out.append(" | ")
        uniform_range(min_s, max_s)
        return

    less_decimals = max(decimals_left - 1, 1)

    if has_min:
        if min_value < 0:
            out.append("\"-\" (")
            _generate_min_max_int(None, -min_value, out, decimals_left, top_level=False)
            out.append(") | [0] | [1-9] ")
            more_digits(0, decimals_left - 1)
        elif min_value == 0:
            if top_level:
                out.append("[0] | [1-9] ")
                more_digits(0, less_decimals)
            else:
                more_digits(1, decimals_left)
        elif min_value <= 9:
            c = str(min_value)
            range_start = '1' if top_level else '0'
            if c > range_start:
                digit_range(range_start, chr(ord(c) - 1))
                out.append(" ")
                more_digits(1, less_decimals)
                out.append(" | ")
            digit_range(c, "9")
            out.append(" ")
            more_digits(0, less_decimals)
        else:
            min_s = str(min_value)
            length = len(min_s)
            c = min_s[0]

            if c > "1":
                digit_range("1" if top_level else "0", chr(ord(c) - 1))
                out.append(" ")
                more_digits(length, less_decimals)
                out.append(" | ")
            digit_range(c, c)
            out.append(" (")
            _generate_min_max_int(int(min_s[1:]), None, out, less_decimals, top_level=False)
            out.append(")")
            if c < "9":
                out.append(" | ")
                digit_range(chr(ord(c) + 1), "9")
                out.append(" ")
                more_digits(length - 1, less_decimals)
        return

    if has_max:
        if max_value >= 0:
            if top_level:
                out.append("\"-\" [1-9] ")
                more_digits(0, less_decimals)
                out.append(" | ")
            _generate_min_max_int(0, max_value, out, decimals_left, top_level=True)
        else:
            out.append("\"-\" (")
            _generate_min_max_int(-max_value, None, out, decimals_left, top_level=False)
            out.append(")")
        return

    raise RuntimeError("At least one of min_value or max_value must be set")

class BuiltinRule:
    def __init__(self, content: str, deps: list | None = None):
        self.content = content
        self.deps = deps or []

# Constraining spaces to prevent model "running away".
SPACE_RULE = '| " " | "\\n"{1,2} [ \\t]{0,20}'

PRIMITIVE_RULES = {
    'boolean'      : BuiltinRule('("true" | "false") space', []),
    'decimal-part' : BuiltinRule('[0-9]{1,16}', []),
    'integral-part': BuiltinRule('[0] | [1-9] [0-9]{0,15}', []),
    'number'       : BuiltinRule('("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space', ['integral-part', 'decimal-part']),
    'integer'      : BuiltinRule('("-"? integral-part) space', ['integral-part']),
    'value'        : BuiltinRule('object | array | string | number | boolean | null', ['object', 'array', 'string', 'number', 'boolean', 'null']),
    'object'       : BuiltinRule('"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space', ['string', 'value']),
    'array'        : BuiltinRule('"[" space ( value ("," space value)* )? "]" space', ['value']),
    'uuid'         : BuiltinRule(r'"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space', []),
    'char'         : BuiltinRule(r'[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})', []),
    'string'       : BuiltinRule(r'"\"" char* "\"" space', ['char']),
    'null'         : BuiltinRule('"null" space', []),
}

# TODO: support "uri", "email" string formats
STRING_FORMAT_RULES = {
    'date'            : BuiltinRule('[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( \"0\" [1-9] | [1-2] [0-9] | "3" [0-1] )', []),
    'time'            : BuiltinRule('([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )', []),
    'date-time'       : BuiltinRule('date "T" time', ['date', 'time']),
    'date-string'     : BuiltinRule('"\\"" date "\\"" space', ['date']),
    'time-string'     : BuiltinRule('"\\"" time "\\"" space', ['time']),
    'date-time-string': BuiltinRule('"\\"" date-time "\\"" space', ['date-time']),
}

DOTALL = '[\\U00000000-\\U0010FFFF]'
DOT = '[^\\x0A\\x0D]'

RESERVED_NAMES = set(["root", "dot", *PRIMITIVE_RULES.keys(), *STRING_FORMAT_RULES.keys()])

INVALID_RULE_CHARS_RE = re.compile(r'[^a-zA-Z0-9-]+')
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_RANGE_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"\]\-\\]')
GRAMMAR_LITERAL_ESCAPES = {'\r': '\\r', '\n': '\\n', '"': '\\"', '-': '\\-', ']': '\\]'}

NON_LITERAL_SET = set('|.()[]{}*+?')
ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = set('^$.[]()|{}*+?')


class SchemaConverter:
    def __init__(self, *, prop_order, allow_fetch, dotall, raw_pattern):
        self._prop_order = prop_order
        self._allow_fetch = allow_fetch
        self._dotall = dotall
        self._raw_pattern = raw_pattern
        self._rules = {
            'space': SPACE_RULE,
        }
        self._refs = {}
        self._refs_being_resolved = set()

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)) or m.group(0), literal
        )
        return f'"{escaped}"'

    def not_literal(self, literal: str, dotall: bool = True, maybe_escaped_underscores = False) -> str:
        '''
            not_literal('a') -> '[^a]'
            not_literal('abc') -> '([^a] | "a" ([^b] | "b" ([^c])?)?)?'
        '''
        assert len(literal) > 0, 'Empty literal not supported'
        def recurse(i: int):
            c = literal[i]
            if maybe_escaped_underscores and c == '_':
                yield f'[^{c}\\\\]'
                yield ' | '
                yield f'"\\\\"? "{c}"'
            else:
                yield f'[^{c}]'
            if i < len(literal) - 1:
                yield ' | '
                yield self._format_literal(c)
                yield ' ('
                yield from recurse(i + 1)
                yield ')?'

        return ''.join(('(', *recurse(0), ')'))

    def _not_strings(self, strings):
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end_of_string = False

            def insert(self, string):
                node = self
                for c in string:
                    node = node.children.setdefault(c, TrieNode())
                node.is_end_of_string = True

        trie = TrieNode()
        for s in strings:
            trie.insert(s)

        char_rule = self._add_primitive('char', PRIMITIVE_RULES['char'])
        out = ['["] ( ']

        def visit(node):
            rejects = []
            first = True
            for c in sorted(node.children.keys()):
                child = node.children[c]
                rejects.append(c)
                if first:
                    first = False
                else:
                    out.append(' | ')
                out.append(f'[{c}]')
                if child.children:
                    out.append(f' (')
                    visit(child)
                    out.append(')')
                elif child.is_end_of_string:
                    out.append(f' {char_rule}+')
            if node.children:
                if not first:
                    out.append(' | ')
                out.append(f'[^"{"".join(rejects)}] {char_rule}*')
        visit(trie)

        out.append(f' ){"" if trie.is_end_of_string else "?"} ["] space')
        return ''.join(out)

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub('-', name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f'{esc_name}{i}' in self._rules and self._rules[f'{esc_name}{i}'] != rule:
                i += 1
            key = f'{esc_name}{i}'
        self._rules[key] = rule
        return key

    def resolve_refs(self, schema: dict, url: str):
        '''
            Resolves all $ref fields in the given schema, fetching any remote schemas,
            replacing $ref with absolute reference URL and populating self._refs with the
            respective referenced (sub)schema dictionaries.
        '''
        def visit(n: dict):
            if isinstance(n, list):
                return [visit(x) for x in n]
            elif isinstance(n, dict):
                ref = n.get('$ref')
                if ref is not None and ref not in self._refs:
                    if ref.startswith('https://'):
                        assert self._allow_fetch, 'Fetching remote schemas is not allowed (use --allow-fetch for force)'
                        import requests

                        frag_split = ref.split('#')
                        base_url = frag_split[0]

                        target = self._refs.get(base_url)
                        if target is None:
                            target = self.resolve_refs(requests.get(ref).json(), base_url)
                            self._refs[base_url] = target

                        if len(frag_split) == 1 or frag_split[-1] == '':
                            return target
                    elif ref.startswith('#/'):
                        target = schema
                        ref = f'{url}{ref}'
                        n['$ref'] = ref
                    else:
                        raise ValueError(f'Unsupported ref {ref}')

                    for sel in ref.split('#')[-1].split('/')[1:]:
                        assert target is not None and sel in target, f'Error resolving ref {ref}: {sel} not in {target}'
                        target = target[sel]

                    self._refs[ref] = target
                else:
                    for v in n.values():
                        visit(v)

            return n
        return visit(schema)

    def _generate_union_rule(self, name, alt_schemas):
        return ' | '.join((
            self.visit(alt_schema, f'{name}{"-" if name else "alternative-"}{i}')
            for i, alt_schema in enumerate(alt_schemas)
        ))

    def _visit_pattern(self, pattern, name):
        '''
            Transforms a regular expression pattern into a GBNF rule.

            Input: https://json-schema.org/understanding-json-schema/reference/regular_expressions
            Output: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

            Unsupported features: negative/positive lookaheads, greedy/non-greedy modifiers.

            Mostly a 1:1 translation, except for {x} / {x,} / {x,y} quantifiers for which
            we define sub-rules to keep the output lean.
        '''

        assert pattern.startswith('^') and pattern.endswith('$'), 'Pattern must start with "^" and end with "$"'
        pattern = pattern[1:-1]
        sub_rule_ids = {}

        i = 0
        length = len(pattern)

        def to_rule(s: tuple[str, bool]) -> str:
            (txt, is_literal) = s
            return "\"" + txt + "\"" if is_literal else txt

        def transform() -> tuple[str, bool]:
            '''
                Parse a unit at index i (advancing it), and return its string representation + whether it's a literal.
            '''
            nonlocal i
            nonlocal pattern
            nonlocal sub_rule_ids

            start = i
            # For each component of this sequence, store its string representation and whether it's a literal.
            # We only need a flat structure here to apply repetition operators to the last item, and
            # to merge literals at the and (we're parsing grouped ( sequences ) recursively and don't treat '|' specially
            # (GBNF's syntax is luckily very close to regular expressions!)
            seq: list[tuple[str, bool]] = []

            def get_dot():
                if self._dotall:
                    rule = DOTALL
                else:
                    # Accept any character... except \n and \r line break chars (\x0A and \xOD)
                    rule = DOT
                return self._add_rule(f'dot', rule)

            def join_seq():
                nonlocal seq
                ret = []
                for is_literal, g in itertools.groupby(seq, lambda x: x[1]):
                    if is_literal:
                        ret.append((''.join(x[0] for x in g), True))
                    else:
                        ret.extend(g)
                if len(ret) == 1:
                    return ret[0]
                return (' '.join(to_rule(x) for x in seq), False)

            while i < length:
                c = pattern[i]
                if c == '.':
                    seq.append((get_dot(), False))
                    i += 1
                elif c == '(':
                    i += 1
                    if i < length:
                        assert pattern[i] != '?', f'Unsupported pattern syntax "{pattern[i]}" at index {i} of /{pattern}/'
                    seq.append((f'({to_rule(transform())})', False))
                elif c == ')':
                    i += 1
                    assert start > 0 and pattern[start-1] == '(', f'Unbalanced parentheses; start = {start}, i = {i}, pattern = {pattern}'
                    return join_seq()
                elif c == '[':
                    square_brackets = c
                    i += 1
                    while i < length and pattern[i] != ']':
                        if pattern[i] == '\\':
                            square_brackets += pattern[i:i+2]
                            i += 2
                        else:
                            square_brackets += pattern[i]
                            i += 1
                    assert i < length, f'Unbalanced square brackets; start = {start}, i = {i}, pattern = {pattern}'
                    square_brackets += ']'
                    i += 1
                    seq.append((square_brackets, False))
                elif c == '|':
                    seq.append(('|', False))
                    i += 1
                elif c in ('*', '+', '?'):
                    seq[-1] = (to_rule(seq[-1]) + c, False)
                    i += 1
                elif c == '{':
                    curly_brackets = c
                    i += 1
                    while i < length and pattern[i] != '}':
                        curly_brackets += pattern[i]
                        i += 1
                    assert i < length, f'Unbalanced curly brackets; start = {start}, i = {i}, pattern = {pattern}'
                    curly_brackets += '}'
                    i += 1
                    nums = [s.strip() for s in curly_brackets[1:-1].split(',')]
                    min_times = 0
                    max_times = None
                    try:
                        if len(nums) == 1:
                            min_times = int(nums[0])
                            max_times = min_times
                        else:
                            assert len(nums) == 2
                            min_times = int(nums[0]) if nums[0] else 0
                            max_times = int(nums[1]) if nums[1] else None
                    except ValueError:
                        raise ValueError(f'Invalid quantifier {curly_brackets} in /{pattern}/')

                    (sub, sub_is_literal) = seq[-1]

                    if not sub_is_literal:
                        id = sub_rule_ids.get(sub)
                        if id is None:
                            id = self._add_rule(f'{name}-{len(sub_rule_ids) + 1}', sub)
                            sub_rule_ids[sub] = id
                        sub = id

                    seq[-1] = (_build_repetition(f'"{sub}"' if sub_is_literal else sub, min_times, max_times), False)
                else:
                    literal = ''
                    while i < length:
                        if pattern[i] == '\\' and i < length - 1:
                            next = pattern[i + 1]
                            if next in ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS:
                                i += 1
                                literal += pattern[i]
                                i += 1
                            else:
                                literal += pattern[i:i+2]
                                i += 2
                        elif pattern[i] == '"' and not self._raw_pattern:
                            literal += '\\"'
                            i += 1
                        elif pattern[i] not in NON_LITERAL_SET and \
                                (i == length - 1 or literal == '' or pattern[i+1] == '.' or pattern[i+1] not in NON_LITERAL_SET):
                            literal += pattern[i]
                            i += 1
                        else:
                            break
                    if literal:
                        seq.append((literal, True))

            return join_seq()

        return self._add_rule(
            name,
            to_rule(transform()) if self._raw_pattern \
                else "\"\\\"\" (" + to_rule(transform()) + ") \"\\\"\" space")


    def _resolve_ref(self, ref):
        ref_name = ref.split('/')[-1]
        if ref_name not in self._rules and ref not in self._refs_being_resolved:
            self._refs_being_resolved.add(ref)
            resolved = self._refs[ref]
            ref_name = self.visit(resolved, ref_name)
            self._refs_being_resolved.remove(ref)
        return ref_name

    def _generate_constant_rule(self, value):
        return self._format_literal(json.dumps(value))

    def visit(self, schema, name):
        schema_type = schema.get('type')
        schema_format = schema.get('format')
        rule_name = name + '-' if name in RESERVED_NAMES else name or 'root'

        if (ref := schema.get('$ref')) is not None:
            return self._add_rule(rule_name, self._resolve_ref(ref))

        elif 'oneOf' in schema or 'anyOf' in schema:
            return self._add_rule(rule_name, self._generate_union_rule(name, schema.get('oneOf') or schema['anyOf']))

        elif isinstance(schema_type, list):
            return self._add_rule(rule_name, self._generate_union_rule(name, [{**schema, 'type': t} for t in schema_type]))

        elif 'const' in schema:
            return self._add_rule(rule_name, self._generate_constant_rule(schema['const']) + ' space')

        elif 'enum' in schema:
            rule = '(' + ' | '.join((self._generate_constant_rule(v) for v in schema['enum'])) + ') space'
            return self._add_rule(rule_name, rule)

        elif schema_type in (None, 'object') and \
             ('properties' in schema or \
              ('additionalProperties' in schema and schema['additionalProperties'] is not True)):
            required = set(schema.get('required', []))
            properties = list(schema.get('properties', {}).items())
            return self._add_rule(rule_name, self._build_object_rule(properties, required, name, schema.get('additionalProperties')))

        elif schema_type in (None, 'object') and 'allOf' in schema:
            required = set()
            properties = []
            hybrid_name = name
            def add_component(comp_schema, is_required):
                if (ref := comp_schema.get('$ref')) is not None:
                    comp_schema = self._refs[ref]

                if 'properties' in comp_schema:
                    for prop_name, prop_schema in comp_schema['properties'].items():
                        properties.append((prop_name, prop_schema))
                        if is_required:
                            required.add(prop_name)

            for t in schema['allOf']:
                if 'anyOf' in t:
                    for tt in t['anyOf']:
                        add_component(tt, is_required=False)
                else:
                    add_component(t, is_required=True)

            return self._add_rule(rule_name, self._build_object_rule(properties, required, hybrid_name, additional_properties=None))

        elif schema_type in (None, 'array') and ('items' in schema or 'prefixItems' in schema):
            items = schema.get('items') or schema['prefixItems']
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space ' +
                    ' "," space '.join(
                        self.visit(item, f'{name}{"-" if name else ""}tuple-{i}')
                        for i, item in enumerate(items)) +
                    ' "]" space')
            else:
                item_rule_name = self.visit(items, f'{name}{"-" if name else ""}item')
                min_items = schema.get("minItems", 0)
                max_items = schema.get("maxItems")
                return self._add_rule(rule_name, '"[" space ' + _build_repetition(item_rule_name, min_items, max_items, separator_rule='"," space') + ' "]" space')

        elif schema_type in (None, 'string') and 'pattern' in schema:
            return self._visit_pattern(schema['pattern'], rule_name)

        elif schema_type in (None, 'string') and re.match(r'^uuid[1-5]?$', schema_format or ''):
            return self._add_primitive(
                'root' if rule_name == 'root' else schema_format,
                PRIMITIVE_RULES['uuid']
            )

        elif schema_type in (None, 'string') and f'{schema_format}-string' in STRING_FORMAT_RULES:
            prim_name = f'{schema_format}-string'
            return self._add_rule(rule_name, self._add_primitive(prim_name, STRING_FORMAT_RULES[prim_name]))

        elif schema_type == 'string' and ('minLength' in schema or 'maxLength' in schema):
            char_rule = self._add_primitive('char', PRIMITIVE_RULES['char'])
            min_len = schema.get('minLength', 0)
            max_len = schema.get('maxLength')

            return self._add_rule(rule_name, r'"\"" ' + _build_repetition(char_rule, min_len, max_len) + r' "\"" space')

        elif schema_type in (None, 'integer') and \
                ('minimum' in schema or 'exclusiveMinimum' in schema or 'maximum' in schema or 'exclusiveMaximum' in schema):
            min_value = None
            max_value = None
            if 'minimum' in schema:
                min_value = schema['minimum']
            elif 'exclusiveMinimum' in schema:
                min_value = schema['exclusiveMinimum'] + 1
            if 'maximum' in schema:
                max_value = schema['maximum']
            elif 'exclusiveMaximum' in schema:
                max_value = schema['exclusiveMaximum'] - 1

            out = ["("]
            _generate_min_max_int(min_value, max_value, out)
            out.append(") space")
            return self._add_rule(rule_name, ''.join(out))

        elif (schema_type == 'object') or (len(schema) == 0):
            return self._add_rule(rule_name, self._add_primitive('object', PRIMITIVE_RULES['object']))

        else:
            assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
            # TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
            return self._add_primitive('root' if rule_name == 'root' else schema_type, PRIMITIVE_RULES[schema_type])

    def _add_primitive(self, name: str, rule: BuiltinRule):
        n = self._add_rule(name, rule.content)

        for dep in rule.deps:
            dep_rule = PRIMITIVE_RULES.get(dep) or STRING_FORMAT_RULES.get(dep)
            assert dep_rule, f'Rule {dep} not known'
            if dep not in self._rules:
                self._add_primitive(dep, dep_rule)
        return n

    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], name: str, additional_properties: Optional[Union[bool, Any]]):
        prop_order = self._prop_order
        # sort by position in prop_order (if specified) then by original order
        sorted_props = [kv[0] for _, kv in sorted(enumerate(properties), key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]))]

        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                fr'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}'
            )
        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]

        if additional_properties is not None and additional_properties != False:
            sub_name = f'{name}{"-" if name else ""}additional'
            value_rule = self.visit(additional_properties, f'{sub_name}-value') if isinstance(additional_properties, dict) else \
                self._add_primitive('value', PRIMITIVE_RULES['value'])
            key_rule = self._add_primitive('string', PRIMITIVE_RULES['string']) if not sorted_props \
                else self._add_rule(f'{sub_name}-k', self._not_strings(sorted_props))

            prop_kv_rule_names["*"] = self._add_rule(
                f'{sub_name}-kv',
                f'{key_rule} ":" space {value_rule}'
            )
            optional_props.append("*")

        rule = '"{" space '
        rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)

        if optional_props:
            rule += ' ('
            if required_props:
                rule += ' "," space ( '

            def get_recursive_refs(ks, first_is_optional):
                [k, *rest] = ks
                kv_rule_name = prop_kv_rule_names[k]
                comma_ref = f'( "," space {kv_rule_name} )'
                if first_is_optional:
                    res = comma_ref + ('*' if k == '*' else '?')
                else:
                    res = kv_rule_name + (' ' + comma_ref + "*" if k == '*' else '')
                if len(rest) > 0:
                    res += ' ' + self._add_rule(
                        f'{name}{"-" if name else ""}{k}-rest',
                        get_recursive_refs(rest, first_is_optional=True)
                    )
                return res

            rule += ' | '.join(
                get_recursive_refs(optional_props[i:], first_is_optional=False)
                for i in range(len(optional_props))
            )
            if required_props:
                rule += ' )'
            rule += ' )?'

        rule += ' "}" space'

        return rule

    def format_grammar(self):
        return '\n'.join(
            f'{name} ::= {rule}'
            for name, rule in sorted(self._rules.items(), key=lambda kv: kv[0])
        )


def main(args_in = None):
    parser = argparse.ArgumentParser(
        description='''
            Generates a grammar (suitable for use in ./llama-cli) that produces JSON conforming to a
            given JSON schema. Only a subset of JSON schema features are supported; more may be
            added in the future.
        ''',
    )
    parser.add_argument(
        '--prop-order',
        default=[],
        type=lambda s: s.split(','),
        help='''
            comma-separated property names defining the order of precedence for object properties;
            properties not specified here are given lower precedence than those that are, and
            are kept in their original order from the schema. Required properties are always
            given precedence over optional properties.
        '''
    )
    parser.add_argument(
        '--allow-fetch',
        action='store_true',
        default=False,
        help='Whether to allow fetching referenced schemas over HTTPS')
    parser.add_argument(
        '--dotall',
        action='store_true',
        default=False,
        help='Whether to treat dot (".") as matching all chars including line breaks in regular expression patterns')
    parser.add_argument(
        '--raw-pattern',
        action='store_true',
        default=False,
        help='Treats string patterns as raw patterns w/o quotes (or quote escapes)')

    parser.add_argument('schema', help='file containing JSON schema ("-" for stdin)')
    args = parser.parse_args(args_in)

    if args.schema.startswith('https://'):
        url = args.schema
        import requests
        schema = requests.get(url).json()
    elif args.schema == '-':
        url = 'stdin'
        schema = json.load(sys.stdin)
    else:
        url = f'file://{args.schema}'
        with open(args.schema) as f:
            schema = json.load(f)
    converter = SchemaConverter(
        prop_order={name: idx for idx, name in enumerate(args.prop_order)},
        allow_fetch=args.allow_fetch,
        dotall=args.dotall,
        raw_pattern=args.raw_pattern)
    schema = converter.resolve_refs(schema, url)
    converter.visit(schema, '')
    print(converter.format_grammar())


if __name__ == '__main__':
    main()
