#!/usr/bin/env python3
"""
GBNF Grammar Parser and Validator

Implements a simple parser for GBNF (GGML Backus-Naur Form) grammars
and demonstrates constrained token generation.
"""

import re
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ElementType(Enum):
    """Grammar element types"""
    LITERAL = "literal"
    CHAR_CLASS = "char_class"
    RULE_REF = "rule_ref"
    SEQUENCE = "sequence"
    ALTERNATIVE = "alternative"
    OPTIONAL = "optional"
    ZERO_OR_MORE = "zero_or_more"
    ONE_OR_MORE = "one_or_more"


@dataclass
class GrammarElement:
    """Represents a grammar element"""
    type: ElementType
    value: Optional[str] = None
    elements: List['GrammarElement'] = None

    def __post_init__(self):
        if self.elements is None:
            self.elements = []


@dataclass
class GrammarRule:
    """Represents a grammar rule"""
    name: str
    definition: GrammarElement


class GBNFParser:
    """Parse GBNF grammar specifications"""

    def __init__(self):
        self.rules: Dict[str, GrammarRule] = {}

    def parse(self, gbnf_text: str) -> Dict[str, GrammarRule]:
        """
        Parse GBNF grammar text

        Returns:
            Dictionary mapping rule names to GrammarRule objects
        """
        # Remove comments
        lines = []
        for line in gbnf_text.split('\n'):
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:
                lines.append(line)

        text = ' '.join(lines)

        # Split into rules
        rule_pattern = r'(\w+)\s*::=\s*([^;]+?)(?=\w+\s*::=|$)'
        matches = re.finditer(rule_pattern, text)

        for match in matches:
            rule_name = match.group(1).strip()
            rule_body = match.group(2).strip()

            element = self._parse_expression(rule_body)
            rule = GrammarRule(rule_name, element)
            self.rules[rule_name] = rule

        return self.rules

    def _parse_expression(self, expr: str) -> GrammarElement:
        """Parse a grammar expression"""
        expr = expr.strip()

        # Handle alternatives (lowest precedence)
        if '|' in expr:
            parts = self._split_alternatives(expr)
            if len(parts) > 1:
                alternatives = [self._parse_expression(p) for p in parts]
                return GrammarElement(
                    type=ElementType.ALTERNATIVE,
                    elements=alternatives
                )

        # Handle sequences
        parts = self._split_sequence(expr)
        if len(parts) > 1:
            sequence = [self._parse_expression(p) for p in parts]
            return GrammarElement(
                type=ElementType.SEQUENCE,
                elements=sequence
            )

        # Handle single element
        return self._parse_element(expr)

    def _parse_element(self, expr: str) -> GrammarElement:
        """Parse a single grammar element"""
        expr = expr.strip()

        # Optional: expr?
        if expr.endswith('?'):
            inner = self._parse_element(expr[:-1])
            return GrammarElement(
                type=ElementType.OPTIONAL,
                elements=[inner]
            )

        # Zero or more: expr*
        if expr.endswith('*'):
            inner = self._parse_element(expr[:-1])
            return GrammarElement(
                type=ElementType.ZERO_OR_MORE,
                elements=[inner]
            )

        # One or more: expr+
        if expr.endswith('+'):
            inner = self._parse_element(expr[:-1])
            return GrammarElement(
                type=ElementType.ONE_OR_MORE,
                elements=[inner]
            )

        # Literal string: "text"
        if expr.startswith('"') and expr.endswith('"'):
            return GrammarElement(
                type=ElementType.LITERAL,
                value=expr[1:-1]
            )

        # Character class: [abc] or [a-z]
        if expr.startswith('[') and expr.endswith(']'):
            return GrammarElement(
                type=ElementType.CHAR_CLASS,
                value=expr[1:-1]
            )

        # Grouped expression: (expr)
        if expr.startswith('(') and expr.endswith(')'):
            return self._parse_expression(expr[1:-1])

        # Rule reference
        return GrammarElement(
            type=ElementType.RULE_REF,
            value=expr
        )

    def _split_alternatives(self, expr: str) -> List[str]:
        """Split expression by | (alternatives)"""
        parts = []
        current = ""
        depth = 0

        for char in expr:
            if char in '([':
                depth += 1
            elif char in ')]':
                depth -= 1
            elif char == '|' and depth == 0:
                parts.append(current.strip())
                current = ""
                continue

            current += char

        if current:
            parts.append(current.strip())

        return parts

    def _split_sequence(self, expr: str) -> List[str]:
        """Split expression into sequence elements"""
        parts = []
        current = ""
        depth = 0
        in_string = False
        in_char_class = False

        i = 0
        while i < len(expr):
            char = expr[i]

            if char == '"' and not in_char_class:
                in_string = not in_string
            elif char == '[' and not in_string:
                in_char_class = True
            elif char == ']' and not in_string:
                in_char_class = False
            elif char == '(' and not in_string and not in_char_class:
                depth += 1
            elif char == ')' and not in_string and not in_char_class:
                depth -= 1

            if (char in ' \t' and depth == 0 and
                not in_string and not in_char_class and current):
                parts.append(current)
                current = ""
                i += 1
                continue

            current += char
            i += 1

        if current:
            parts.append(current)

        return parts


class GrammarValidator:
    """Validate text against a grammar"""

    def __init__(self, grammar: Dict[str, GrammarRule]):
        self.grammar = grammar

    def validate(self, text: str, start_rule: str = "root") -> bool:
        """
        Check if text matches the grammar

        Args:
            text: Text to validate
            start_rule: Starting rule name

        Returns:
            True if text matches grammar
        """
        if start_rule not in self.grammar:
            raise ValueError(f"Unknown rule: {start_rule}")

        rule = self.grammar[start_rule]
        remaining = self._match_element(rule.definition, text)

        # Valid if all text consumed
        return remaining == ""

    def _match_element(self, element: GrammarElement, text: str) -> Optional[str]:
        """
        Try to match element against text

        Returns:
            Remaining text after match, or None if no match
        """
        if element.type == ElementType.LITERAL:
            if text.startswith(element.value):
                return text[len(element.value):]
            return None

        elif element.type == ElementType.CHAR_CLASS:
            if not text:
                return None
            if self._char_matches(text[0], element.value):
                return text[1:]
            return None

        elif element.type == ElementType.RULE_REF:
            if element.value not in self.grammar:
                raise ValueError(f"Unknown rule: {element.value}")
            ref_rule = self.grammar[element.value]
            return self._match_element(ref_rule.definition, text)

        elif element.type == ElementType.SEQUENCE:
            remaining = text
            for sub_elem in element.elements:
                remaining = self._match_element(sub_elem, remaining)
                if remaining is None:
                    return None
            return remaining

        elif element.type == ElementType.ALTERNATIVE:
            for sub_elem in element.elements:
                result = self._match_element(sub_elem, text)
                if result is not None:
                    return result
            return None

        elif element.type == ElementType.OPTIONAL:
            result = self._match_element(element.elements[0], text)
            return result if result is not None else text

        elif element.type == ElementType.ZERO_OR_MORE:
            remaining = text
            while True:
                result = self._match_element(element.elements[0], remaining)
                if result is None:
                    break
                remaining = result
            return remaining

        elif element.type == ElementType.ONE_OR_MORE:
            result = self._match_element(element.elements[0], text)
            if result is None:
                return None
            remaining = result
            while True:
                result = self._match_element(element.elements[0], remaining)
                if result is None:
                    break
                remaining = result
            return remaining

        return None

    def _char_matches(self, char: str, char_class: str) -> bool:
        """Check if char matches character class"""
        # Handle ranges like a-z
        parts = char_class.split()

        for part in char_class:
            if len(part) == 3 and part[1] == '-':
                # Range like a-z
                start, end = part[0], part[2]
                if start <= char <= end:
                    return True
            elif part == char:
                return True

        # Simple implementation: check if char in char_class
        return char in char_class


# Example grammars

JSON_GRAMMAR = """
root ::= value

value ::= object | array | string | number | boolean | null

object ::= "{" ws "}"
         | "{" ws member ( "," ws member )* ws "}"

member ::= string ws ":" ws value

array ::= "[" ws "]"
        | "[" ws value ( "," ws value )* ws "]"

string ::= "\\"" character* "\\""

character ::= [^"\\\\]

number ::= "-"? ( "0" | [1-9] [0-9]* )

boolean ::= "true" | "false"

null ::= "null"

ws ::= [ \\t\\n\\r]*
"""

EMAIL_GRAMMAR = """
root ::= email

email ::= local-part "@" domain

local-part ::= atom

atom ::= [a-zA-Z0-9]+

domain ::= subdomain ( "." subdomain )*

subdomain ::= [a-zA-Z0-9]+
"""

URL_GRAMMAR = """
root ::= url

url ::= scheme "://" host path?

scheme ::= "http" | "https"

host ::= [a-zA-Z0-9.-]+

path ::= "/" segment ( "/" segment )*

segment ::= [a-zA-Z0-9._-]*
"""


def test_grammar_parser():
    """Test GBNF parser and validator"""
    print("=" * 70)
    print("GBNF GRAMMAR PARSER TEST")
    print("=" * 70)

    # Test 1: Email grammar
    print("\n" + "─" * 70)
    print("EMAIL GRAMMAR")
    print("─" * 70)

    parser = GBNFParser()
    email_rules = parser.parse(EMAIL_GRAMMAR)

    print(f"\nParsed {len(email_rules)} rules:")
    for name in email_rules:
        print(f"  - {name}")

    validator = GrammarValidator(email_rules)

    test_emails = [
        ("user@example.com", True),
        ("test@domain.co.uk", True),
        ("invalid-email", False),
        ("@example.com", False),
        ("user@", False),
    ]

    print("\nValidation tests:")
    for email, expected in test_emails:
        result = validator.validate(email)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{email}': {result} (expected {expected})")

    # Test 2: URL grammar
    print("\n" + "─" * 70)
    print("URL GRAMMAR")
    print("─" * 70)

    url_rules = parser.parse(URL_GRAMMAR)
    url_validator = GrammarValidator(url_rules)

    test_urls = [
        ("https://example.com", True),
        ("http://test.org/path", True),
        ("https://site.com/path/to/page", True),
        ("ftp://invalid", False),
        ("example.com", False),
    ]

    print("\nValidation tests:")
    for url, expected in test_urls:
        result = url_validator.validate(url)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{url}': {result} (expected {expected})")

    # Test 3: JSON grammar (simplified)
    print("\n" + "─" * 70)
    print("JSON GRAMMAR (simple examples)")
    print("─" * 70)

    # Note: Full JSON validation complex, showing structure
    print("\nGrammar structure:")
    json_rules = parser.parse(JSON_GRAMMAR)
    for name, rule in json_rules.items():
        print(f"  {name}: {rule.definition.type.value}")


def demonstrate_constrained_generation():
    """
    Show how grammars constrain token generation
    """
    print("\n" + "=" * 70)
    print("CONSTRAINED GENERATION DEMO")
    print("=" * 70)

    # Simple example: Generate valid JSON keys
    json_key_grammar = """
    root ::= string

    string ::= "\\"" [a-zA-Z]+ "\\""
    """

    parser = GBNFParser()
    rules = parser.parse(json_key_grammar)
    validator = GrammarValidator(rules)

    print("\nGrammar: JSON string keys (letters only)")
    print("\nValid examples:")

    valid_keys = ['"name"', '"age"', '"email"', '"address"']
    for key in valid_keys:
        is_valid = validator.validate(key)
        print(f"  {key}: {is_valid}")

    print("\nInvalid examples:")

    invalid_keys = ['name', '"123"', '""', '"na me"']
    for key in invalid_keys:
        is_valid = validator.validate(key)
        print(f"  {key}: {is_valid}")


if __name__ == "__main__":
    # Run tests
    test_grammar_parser()

    # Demo constrained generation
    demonstrate_constrained_generation()

    print("\n✅ Grammar parser demo complete!")
    print("\nKey Takeaways:")
    print("  • GBNF defines valid token sequences")
    print("  • Parser converts GBNF to rule tree")
    print("  • Validator checks text against grammar")
    print("  • Used in llama.cpp for constrained generation")
    print("  • Ensures 100% valid structured outputs")
