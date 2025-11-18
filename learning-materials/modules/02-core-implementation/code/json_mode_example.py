#!/usr/bin/env python3
"""
JSON Mode and Grammar-Guided Generation Example

Demonstrates structured output generation using grammars.

Usage:
    python json_mode_example.py model.gguf
"""

import sys
import json
from llama_cpp import Llama, LlamaGrammar


# Sample grammars
JSON_GRAMMAR = r'''
root ::= object
value ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string  ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ \t\n] ws)?
'''

USER_PROFILE_GRAMMAR = r'''
root ::= "{" ws
  "\"name\":" ws string "," ws
  "\"age\":" ws number "," ws
  "\"email\":" ws string ws
"}" ws

string ::= "\"" [^"]* "\""
number ::= [0-9]+
ws ::= [ \t\n]*
'''

FUNCTION_CALL_GRAMMAR = r'''
root ::= "{" ws
  "\"function\":" ws function-name "," ws
  "\"arguments\":" ws arguments ws
"}" ws

function-name ::= "\"get_weather\"" | "\"set_alarm\"" | "\"send_email\""

arguments ::= "{" ws
  (pair ("," ws pair)*)?
ws "}"

pair ::= string ":" ws value

string ::= "\"" [^"]* "\""
value ::= string | number | boolean
number ::= [0-9]+
boolean ::= "true" | "false"
ws ::= [ \t\n]*
'''


def demonstrate_json_mode(llm: Llama):
    """Demonstrate basic JSON generation"""
    print("\n" + "=" * 80)
    print("📝 JSON Mode: Generate User Profile")
    print("=" * 80)

    prompt = """Generate a JSON object for a user profile with name, age, and email.
Example: {"name": "John Doe", "age": 30, "email": "john@example.com"}

User: Create a profile for Alice Smith, age 28, alice.smith@company.com
JSON:"""

    grammar = LlamaGrammar.from_string(JSON_GRAMMAR)

    print(f"Prompt: {prompt[:100]}...")
    print("\nGenerating...")

    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.7,
        grammar=grammar,
        stop=["}", "\n\n"]
    )

    result = output['choices'][0]['text']
    print(f"\nGenerated JSON:\n{result}")

    # Verify it's valid JSON
    try:
        parsed = json.loads(result)
        print("\n✅ Valid JSON!")
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")


def demonstrate_structured_schema(llm: Llama):
    """Demonstrate generation with specific schema"""
    print("\n" + "=" * 80)
    print("📋 Structured Schema: Strict User Profile Format")
    print("=" * 80)

    prompt = """Create a user profile for Bob Johnson, age 35, bob.j@email.com
JSON:"""

    grammar = LlamaGrammar.from_string(USER_PROFILE_GRAMMAR)

    print(f"Prompt: {prompt}")
    print("\nGenerating with strict schema (name, age, email only)...")

    output = llm(
        prompt,
        max_tokens=150,
        temperature=0.7,
        grammar=grammar
    )

    result = output['choices'][0]['text']
    print(f"\nGenerated:\n{result}")

    try:
        parsed = json.loads(result)
        print("\n✅ Valid JSON with correct schema!")
        print(f"Name:  {parsed.get('name', 'N/A')}")
        print(f"Age:   {parsed.get('age', 'N/A')}")
        print(f"Email: {parsed.get('email', 'N/A')}")

        # Verify required fields
        required = {'name', 'age', 'email'}
        if set(parsed.keys()) == required:
            print("\n✅ Exact schema match (only required fields)!")
        else:
            print(f"\n⚠️ Schema mismatch. Expected: {required}, Got: {set(parsed.keys())}")

    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")


def demonstrate_function_calling(llm: Llama):
    """Demonstrate function calling with grammar"""
    print("\n" + "=" * 80)
    print("📞 Function Calling: Weather Query")
    print("=" * 80)

    prompt = """Convert the following request to a function call:

User request: "What's the weather like in San Francisco?"

Function call:"""

    grammar = LlamaGrammar.from_string(FUNCTION_CALL_GRAMMAR)

    print(f"Prompt: {prompt}")
    print("\nGenerating function call...")

    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.3,  # Lower temperature for structured output
        grammar=grammar
    )

    result = output['choices'][0]['text']
    print(f"\nGenerated:\n{result}")

    try:
        parsed = json.loads(result)
        print("\n✅ Valid function call!")
        print(f"Function: {parsed.get('function', 'N/A')}")
        print(f"Arguments: {json.dumps(parsed.get('arguments', {}), indent=2)}")
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")


def demonstrate_array_generation(llm: Llama):
    """Demonstrate array generation"""
    print("\n" + "=" * 80)
    print("📚 Array Generation: List of Users")
    print("=" * 80)

    # Grammar for array of user objects
    array_grammar = r'''
root ::= "[" ws (user ("," ws user)*)? "]" ws

user ::= "{" ws
  "\"name\":" ws string "," ws
  "\"age\":" ws number ws
"}" ws

string ::= "\"" [^"]* "\""
number ::= [0-9]+
ws ::= [ \t\n]*
'''

    prompt = """Generate a JSON array of 3 users (name and age):
["""

    grammar = LlamaGrammar.from_string(array_grammar)

    print(f"Prompt: {prompt}")
    print("\nGenerating array...")

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.8,
        grammar=grammar
    )

    result = "[" + output['choices'][0]['text']  # Add back opening bracket
    print(f"\nGenerated:\n{result}")

    try:
        parsed = json.loads(result)
        print(f"\n✅ Valid JSON array with {len(parsed)} users!")
        for i, user in enumerate(parsed):
            print(f"  {i+1}. {user.get('name', 'N/A')} (age {user.get('age', 'N/A')})")
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")


def compare_with_without_grammar(llm: Llama):
    """Compare generation with and without grammar"""
    print("\n" + "=" * 80)
    print("⚖️  Comparison: With vs Without Grammar")
    print("=" * 80)

    prompt = "Generate a JSON object with fields: name, age, city\nJSON:"

    # Without grammar
    print("\n1. WITHOUT Grammar Constraints:")
    print("-" * 80)
    output_no_grammar = llm(
        prompt,
        max_tokens=80,
        temperature=0.7,
        stop=["\n\n"]
    )
    result_no_grammar = output_no_grammar['choices'][0]['text']
    print(result_no_grammar)

    try:
        json.loads(result_no_grammar)
        print("✅ Valid JSON (lucky!)")
    except json.JSONDecodeError:
        print("❌ Invalid JSON (as expected without constraints)")

    # With grammar
    print("\n2. WITH Grammar Constraints:")
    print("-" * 80)
    grammar = LlamaGrammar.from_string(JSON_GRAMMAR)
    output_with_grammar = llm(
        prompt,
        max_tokens=80,
        temperature=0.7,
        grammar=grammar
    )
    result_with_grammar = output_with_grammar['choices'][0]['text']
    print(result_with_grammar)

    try:
        json.loads(result_with_grammar)
        print("✅ Valid JSON (guaranteed by grammar!)")
    except json.JSONDecodeError:
        print("❌ Invalid JSON (should never happen with grammar)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python json_mode_example.py <model.gguf>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"📚 Loading model: {model_path}")

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=32,
            verbose=False
        )
        print("✅ Model loaded successfully")

        # Run demonstrations
        demonstrate_json_mode(llm)
        demonstrate_structured_schema(llm)
        demonstrate_function_calling(llm)
        demonstrate_array_generation(llm)
        compare_with_without_grammar(llm)

        print("\n" + "=" * 80)
        print("✅ All demonstrations complete!")
        print("\n💡 Key Benefits of Grammar-Guided Generation:")
        print("  • Guaranteed format compliance")
        print("  • No parsing failures")
        print("  • Reliable integration with traditional software")
        print("  • Production-ready structured output")
        print("  • Minimal overhead (~5-10% slower)")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
