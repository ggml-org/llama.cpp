#!/usr/bin/env python3
"""Reference implementation of TTS normalization patterns, the tts_norm_scan() and tts_norm_sub() template functions.

The specification is in docs/tts-norm-patterns.md. This file must give the same result as the C++ implementation
(common/chat-tts-norm.cpp) for every input; tests/test-jinja.cpp checks this in python mode (test-jinja -py).

Usage with jinja2:

    env.globals["tts_norm_scan"] = chat_tts_norm.tts_norm_scan
    env.globals["tts_norm_sub"] = chat_tts_norm.tts_norm_sub
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# character classes, explicit codepoint ranges (keep in sync with common/chat-tts-norm.cpp)

TTS_NORM_SPACE = [
    (0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085), (0x00A0, 0x00A0), (0x1680, 0x1680),
    (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000),
]

TTS_NORM_PUNCT = [
    (0x0021, 0x002F), (0x003A, 0x0040), (0x005B, 0x0060), (0x007B, 0x007E),
    (0x00A1, 0x00A9), (0x00AB, 0x00B4), (0x00B6, 0x00B9), (0x00BB, 0x00BF), (0x00D7, 0x00D7), (0x00F7, 0x00F7),
    (0x2010, 0x2027), (0x2030, 0x205E), (0x20A0, 0x20CF), (0x2100, 0x218F), (0x2190, 0x2BFF),
    (0x3001, 0x3003), (0x3008, 0x3020), (0x3030, 0x3030), (0x303D, 0x303D), (0x30FB, 0x30FB),
    (0xFE10, 0xFE19), (0xFE30, 0xFE4F), (0xFE50, 0xFE6B),
    (0xFF01, 0xFF0F), (0xFF1A, 0xFF20), (0xFF3B, 0xFF40), (0xFF5B, 0xFF65),
    (0x1F000, 0x1FAFF),
]

TTS_NORM_CONTROL = [
    (0x0000, 0x001F), (0x007F, 0x009F), (0x200B, 0x200F), (0x2060, 0x206F), (0xFEFF, 0xFEFF),
]

TTS_NORM_HAN = [
    (0x2E80, 0x2FDF), (0x3005, 0x3005), (0x3007, 0x3007), (0x3021, 0x3029), (0x3038, 0x303B),
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF),
]

CLASS_LETTERS = "dspawulhc"
INF = 1 << 31
MAX_REPEAT = 1000
MAX_PATTERN_LEN = 4096


def _in(ranges: list[tuple[int, int]], c: int) -> bool:
    return any(lo <= c <= hi for lo, hi in ranges)


def _is_alpha(c: int) -> bool:
    if ord("a") <= c <= ord("z") or ord("A") <= c <= ord("Z"):
        return True
    if c in (0xAA, 0xB5, 0xBA):
        return True
    return c >= 0xC0 and not _in(TTS_NORM_SPACE, c) and not _in(TTS_NORM_PUNCT, c) and not _in(TTS_NORM_CONTROL, c)


def _class_match(kind: str, c: int) -> bool:
    if kind == "d":
        return ord("0") <= c <= ord("9")
    if kind == "s":
        return _in(TTS_NORM_SPACE, c)
    if kind == "p":
        return _in(TTS_NORM_PUNCT, c)
    if kind == "a":
        return _is_alpha(c)
    if kind == "w":
        return _is_alpha(c) or ord("0") <= c <= ord("9")
    if kind == "u":
        return ord("A") <= c <= ord("Z")
    if kind == "l":
        return ord("a") <= c <= ord("z")
    if kind == "h":
        return _in(TTS_NORM_HAN, c)
    if kind == "c":
        return _in(TTS_NORM_CONTROL, c)
    return False


def _ascii_lower(c: int) -> int:
    return c + 32 if ord("A") <= c <= ord("Z") else c


def _ascii_swapcase(c: int) -> int:
    if ord("A") <= c <= ord("Z"):
        return c + 32
    if ord("a") <= c <= ord("z"):
        return c - 32
    return c


def _is_ascii_alnum(c: int) -> bool:
    return ord("0") <= c <= ord("9") or ord("a") <= c <= ord("z") or ord("A") <= c <= ord("Z")


# AST

LITERAL, SET, ANY, START, END, BOUNDARY, GROUP, LOOKAHEAD, LOOKBEHIND = range(9)
ZERO_WIDTH = (START, END, BOUNDARY, LOOKAHEAD, LOOKBEHIND)


@dataclass
class _Set:
    neg: bool = False
    ranges: list[tuple[int, int]] = field(default_factory=list)
    classes: list[tuple[str, bool]] = field(default_factory=list)


@dataclass
class _Node:
    op: int = GROUP
    cp: int = 0
    set: _Set = field(default_factory=_Set)
    alts: list[list[_Item]] = field(default_factory=list)
    capture: int = -1
    neg: bool = False


@dataclass
class _Item:
    atom: _Node
    min: int = 1
    max: int = 1


class PatternError(ValueError):
    pass


class _Parser:
    def __init__(self, s: list[int]):
        self.s = s
        self.p = 0
        self.n_caps = 0

    def fail(self, msg: str):
        raise PatternError(f"invalid TTS norm pattern: {msg} at position {self.p}")

    def eof(self) -> bool:
        return self.p >= len(self.s)

    def peek_is(self, c: str) -> bool:
        return not self.eof() and self.s[self.p] == ord(c)

    def expect(self, c: str, what: str):
        if not self.peek_is(c):
            self.fail(f"expected {what}")
        self.p += 1

    def parse_alts(self) -> list[list[_Item]]:
        alts = [self.parse_seq()]
        while self.peek_is("|"):
            self.p += 1
            alts.append(self.parse_seq())
        return alts

    def parse_seq(self) -> list[_Item]:
        seq: list[_Item] = []
        while not self.eof() and self.s[self.p] not in (ord("|"), ord(")")):
            item = _Item(self.parse_atom())
            self.parse_quantifier(item)
            seq.append(item)
        return seq

    def parse_int(self) -> int:
        if self.eof() or not (ord("0") <= self.s[self.p] <= ord("9")):
            self.fail("expected number in quantifier")
        v = 0
        while not self.eof() and ord("0") <= self.s[self.p] <= ord("9"):
            v = v * 10 + self.s[self.p] - ord("0")
            if v > MAX_REPEAT:
                self.fail(f"repetition count exceeds {MAX_REPEAT}")
            self.p += 1
        return v

    def parse_quantifier(self, item: _Item):
        if self.eof():
            return
        c = chr(self.s[self.p])
        if c not in "?*+{":
            return
        if item.atom.op in ZERO_WIDTH:
            self.fail("zero-width assertion cannot be repeated")
        self.p += 1
        if c == "?":
            item.min, item.max = 0, 1
        elif c == "*":
            item.min, item.max = 0, INF
        elif c == "+":
            item.min, item.max = 1, INF
        else:
            item.min = self.parse_int()
            item.max = item.min
            if self.peek_is(","):
                self.p += 1
                item.max = INF if self.peek_is("}") else self.parse_int()
            self.expect("}", "'}' to close quantifier")
            if item.max < item.min:
                self.fail("quantifier max is smaller than min")
        if not self.eof() and chr(self.s[self.p]) in "?*+{":
            self.fail("multiple quantifiers")

    def parse_group(self) -> _Node:
        node = _Node(op=GROUP)
        if self.peek_is("?"):
            self.p += 1
            if self.eof():
                self.fail("unterminated group")
            c = chr(self.s[self.p])
            if c == ":":
                self.p += 1
            elif c in "=!":
                self.p += 1
                node.op = LOOKAHEAD
                node.neg = c == "!"
            elif c == "<" and self.p + 1 < len(self.s) and self.s[self.p + 1] in (ord("="), ord("!")):
                node.op = LOOKBEHIND
                node.neg = self.s[self.p + 1] == ord("!")
                self.p += 2
                if self.eof() or self.s[self.p] == ord(")"):
                    self.fail("lookbehind must contain exactly one codepoint matcher")
                atom = self.parse_atom()
                if atom.op not in (LITERAL, SET, ANY):
                    self.fail("lookbehind must contain exactly one codepoint matcher")
                if not self.peek_is(")"):
                    self.fail("lookbehind must contain exactly one codepoint matcher")
                self.p += 1
                node.alts = [[_Item(atom)]]
                return node
            elif c == "i":
                self.fail("(?i) is only allowed at the start of the pattern")
            else:
                self.fail("unknown group type")
        else:
            node.capture = self.n_caps
            self.n_caps += 1
        node.alts = self.parse_alts()
        self.expect(")", "')' to close group")
        return node

    # returns ("class", (kind, neg)) | ("boundary", None) | ("literal", cp)
    def parse_escape(self, in_set: bool) -> tuple[str, Any]:
        if self.eof():
            self.fail("dangling '%'")
        e = self.s[self.p]
        self.p += 1
        if e < 0x80 and chr(e).lower() in CLASS_LETTERS:
            return "class", (chr(e).lower(), ord("A") <= e <= ord("Z"))
        if e == ord("b") and not in_set:
            return "boundary", None
        if _is_ascii_alnum(e):
            self.p -= 1
            self.fail(f"unknown escape '%{chr(e)}'")
        return "literal", e

    def parse_set(self) -> _Node:
        node = _Node(op=SET)
        if self.peek_is("^"):
            node.set.neg = True
            self.p += 1
        empty = True
        while True:
            if self.eof():
                self.fail("unterminated set")
            if self.s[self.p] == ord("]"):
                self.p += 1
                break
            empty = False
            if self.s[self.p] == ord("%"):
                self.p += 1
                kind, val = self.parse_escape(True)
                if kind == "class":
                    node.set.classes.append(val)
                    continue
                lo = val
            else:
                lo = self.s[self.p]
                self.p += 1
            hi = lo
            if self.peek_is("-") and self.p + 1 < len(self.s) and self.s[self.p + 1] != ord("]"):
                self.p += 1
                if self.s[self.p] == ord("%"):
                    self.p += 1
                    kind, val = self.parse_escape(True)
                    if kind == "class":
                        self.fail("character class cannot be used in a range")
                    hi = val
                else:
                    hi = self.s[self.p]
                    self.p += 1
                if hi < lo:
                    self.fail("invalid range in set")
            node.set.ranges.append((lo, hi))
        if empty:
            self.fail("empty set")
        return node

    def parse_atom(self) -> _Node:
        c = chr(self.s[self.p])
        self.p += 1
        if c == "(":
            return self.parse_group()
        if c == "[":
            return self.parse_set()
        if c == ".":
            return _Node(op=ANY)
        if c == "^":
            return _Node(op=START)
        if c == "$":
            return _Node(op=END)
        if c == "%":
            kind, val = self.parse_escape(False)
            if kind == "class":
                node = _Node(op=SET)
                node.set.classes.append(val)
                return node
            if kind == "boundary":
                return _Node(op=BOUNDARY)
            return _Node(op=LITERAL, cp=val)
        if c in "?*+{":
            self.p -= 1
            self.fail("nothing to repeat")
        if c in "]}":
            self.p -= 1
            self.fail(f"unescaped '{c}', use '%{c}'")
        return _Node(op=LITERAL, cp=ord(c))


class _Matcher:
    def __init__(self, t: list[int], icase: bool, n_groups: int):
        self.t = t
        self.icase = icase
        self.caps: list[tuple[int, int]] = [(-1, -1)] * n_groups

    def is_w(self, i: int) -> bool:
        return i < len(self.t) and _class_match("w", self.t[i])

    def set_match(self, st: _Set, c: int) -> bool:
        def test(x: int) -> bool:
            if any(lo <= x <= hi for lo, hi in st.ranges):
                return True
            return any(_class_match(kind, x) != neg for kind, neg in st.classes)

        m = test(c)
        if not m and self.icase:
            o = _ascii_swapcase(c)
            if o != c:
                m = test(o)
        return m != st.neg

    def match_alts(self, alts: list[list[_Item]], i: int) -> int:
        """returns the end position, or -1 if no alternative matches"""
        for seq in alts:
            saved = list(self.caps)
            end = self.match_seq(seq, i)
            if end >= 0:
                return end
            self.caps = saved
        return -1

    def match_seq(self, seq: list[_Item], i: int) -> int:
        pos = i
        for it in seq:
            count = 0
            while count < it.max:
                is_group = it.atom.op == GROUP
                saved = list(self.caps) if is_group else None
                e = self.match_once(it.atom, pos)
                if e < 0:
                    if saved is not None:
                        self.caps = saved
                    break
                if e == pos:
                    # an empty iteration would repeat forever, it satisfies any remaining minimum
                    count = max(count + 1, it.min)
                    break
                pos = e
                count += 1
            if count < it.min:
                return -1
        return pos

    def match_once(self, node: _Node, i: int) -> int:
        """returns the end position, or -1 on failure"""
        n = len(self.t)
        op = node.op
        if op == LITERAL:
            if i < n and (self.t[i] == node.cp or (self.icase and _ascii_lower(self.t[i]) == _ascii_lower(node.cp))):
                return i + 1
            return -1
        if op == SET:
            return i + 1 if i < n and self.set_match(node.set, self.t[i]) else -1
        if op == ANY:
            return i + 1 if i < n else -1
        if op == START:
            return i if i == 0 else -1
        if op == END:
            return i if i == n else -1
        if op == BOUNDARY:
            return i if (i > 0 and self.is_w(i - 1)) != self.is_w(i) else -1
        if op == GROUP:
            e = self.match_alts(node.alts, i)
            if e < 0:
                return -1
            if node.capture >= 0:
                self.caps[node.capture] = (i, e)
            return e
        if op == LOOKAHEAD:
            # captures made inside a lookahead are discarded
            saved = list(self.caps)
            ok = self.match_alts(node.alts, i) >= 0
            self.caps = saved
            return i if ok != node.neg else -1
        if op == LOOKBEHIND:
            ok = i > 0 and self.match_once(node.alts[0][0].atom, i - 1) == i
            return i if ok != node.neg else -1
        return -1


class Pattern:
    def __init__(self, src: str):
        if len(src.encode("utf-8")) > MAX_PATTERN_LEN:
            raise PatternError(f"invalid TTS norm pattern: pattern is longer than {MAX_PATTERN_LEN} bytes")
        self.icase = False
        parser = _Parser([ord(c) for c in src])
        if src.startswith("(?i)"):
            self.icase = True
            parser.p = 4
        self.root = _Node(op=GROUP, alts=parser.parse_alts())
        if not parser.eof():
            parser.fail("unbalanced ')'")
        self.n_caps = parser.n_caps

    def match_at(self, t: list[int], pos: int) -> tuple[int, list[tuple[int, int]]] | None:
        m = _Matcher(t, self.icase, self.n_caps)
        end = m.match_alts(self.root.alts, pos)
        if end < 0:
            return None
        return end, m.caps


# (rule index or -1, start, end, groups) with codepoint offsets
Segment = tuple[int, int, int, list[tuple[int, int]]]


def scan_segments(text: str, rules: list[Pattern]) -> list[Segment]:
    t = [ord(c) for c in text]
    n = len(t)
    out: list[Segment] = []
    i = 0
    pending = 0
    while i < n:
        matched = False
        for r, rule in enumerate(rules):
            res = rule.match_at(t, i)
            if res is not None and res[0] > i:
                end, groups = res
                if pending < i:
                    out.append((-1, pending, i, []))
                out.append((r, i, end, groups))
                i = end
                pending = i
                matched = True
                break
        if not matched:
            i += 1
    if pending < n:
        out.append((-1, pending, n, []))
    return out


def _rules_from_value(rules: Any) -> list[tuple[str, str]]:
    if isinstance(rules, str):
        return [("match", rules)]
    if isinstance(rules, dict):
        out = []
        for k, v in rules.items():
            if not isinstance(v, str):
                raise TypeError("tts_norm_scan: rule pattern must be a string")
            out.append((str(k), v))
        return out
    if isinstance(rules, (list, tuple)):
        out = []
        for item in rules:
            if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], str):
                raise TypeError("tts_norm_scan: each rule must be a [name, pattern] pair of strings")
            out.append((item[0], item[1]))
        return out
    raise TypeError("tts_norm_scan: rules must be a string, a dict or a list of [name, pattern] pairs")


def tts_norm_scan(text: str, rules: Any) -> list[dict[str, Any]]:
    rule_srcs = _rules_from_value(rules)
    patterns = [Pattern(src) for _, src in rule_srcs]
    out: list[dict[str, Any]] = []
    for r, start, end, groups in scan_segments(text, patterns):
        # groups that did not participate are empty strings, so they render as nothing and are falsy
        vals = ["" if gs < 0 else text[gs:ge] for gs, ge in groups]
        out.append({"rule": rule_srcs[r][0] if r >= 0 else None, "text": text[start:end], "groups": vals})
    return out


def _parse_repl(repl: str, pattern: Pattern) -> list[str | int]:
    pieces: list[str | int] = []
    lit = ""
    i = 0
    while i < len(repl):
        c = repl[i]
        if c != "%":
            lit += c
            i += 1
            continue
        if i + 1 >= len(repl):
            raise PatternError("invalid TTS norm replacement: dangling '%'")
        i += 1
        c = repl[i]
        if c == "%":
            lit += "%"
        elif "0" <= c <= "9":
            g = ord(c) - ord("0")
            if g > pattern.n_caps:
                raise PatternError(f"invalid TTS norm replacement: group %{c} does not exist")
            if lit:
                pieces.append(lit)
                lit = ""
            pieces.append(g)
        else:
            raise PatternError(f"invalid TTS norm replacement: unknown escape '%{c}'")
        i += 1
    if lit:
        pieces.append(lit)
    return pieces


def tts_norm_sub(text: str, pattern: str, repl: str) -> str:
    pat = Pattern(pattern)
    pieces = _parse_repl(repl, pat)
    out = []
    for r, start, end, groups in scan_segments(text, [pat]):
        if r < 0:
            out.append(text[start:end])
            continue
        for piece in pieces:
            if isinstance(piece, str):
                out.append(piece)
            elif piece == 0:
                out.append(text[start:end])
            else:
                gs, ge = groups[piece - 1]
                if gs >= 0:
                    out.append(text[gs:ge])
    return "".join(out)
