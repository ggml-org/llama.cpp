#!/usr/bin/env python3
"""
Tokenizer Inspector

Analyze and visualize tokenization behavior.
Demonstrates:
- Tokenization patterns
- Token efficiency analysis
- Special token handling
- Detokenization

Usage:
    python tokenizer_inspector.py "Hello, world!"
"""

import sys
from llama_cpp import Llama
from typing import List, Tuple
import json


class TokenizerInspector:
    """Inspect tokenization behavior"""

    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=0,  # CPU only for inspection
            verbose=False
        )

    def tokenize(self, text: str, add_bos: bool = False) -> List[int]:
        """Tokenize text"""
        tokens = self.llm.tokenize(text.encode('utf-8'), add_bos=add_bos)
        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize tokens"""
        return self.llm.detokenize(tokens).decode('utf-8', errors='replace')

    def analyze_text(self, text: str) -> dict:
        """Comprehensive tokenization analysis"""
        tokens = self.tokenize(text)

        # Token-by-token breakdown
        token_details = []
        for i, token_id in enumerate(tokens):
            token_text = self.detokenize([token_id])
            token_details.append({
                'position': i,
                'token_id': token_id,
                'token_text': token_text,
                'token_bytes': len(token_text.encode('utf-8')),
                'is_special': token_id in [self.llm.token_bos(), self.llm.token_eos()],
            })

        # Statistics
        stats = {
            'text_length': len(text),
            'text_bytes': len(text.encode('utf-8')),
            'token_count': len(tokens),
            'chars_per_token': len(text) / len(tokens) if tokens else 0,
            'bytes_per_token': len(text.encode('utf-8')) / len(tokens) if tokens else 0,
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
        }

        return {
            'text': text,
            'tokens': token_details,
            'stats': stats,
            'special_tokens': {
                'bos': self.llm.token_bos(),
                'eos': self.llm.token_eos(),
            }
        }

    def compare_texts(self, texts: List[str]) -> None:
        """Compare tokenization of multiple texts"""
        print("\n📊 Tokenization Comparison")
        print("=" * 100)
        print(f"{'Text':<40} {'Tokens':<10} {'Chars/Tok':<12} {'Efficiency':<12}")
        print("-" * 100)

        for text in texts:
            tokens = self.tokenize(text)
            chars_per_tok = len(text) / len(tokens) if tokens else 0
            efficiency = "Good" if chars_per_tok >= 3.5 else "Medium" if chars_per_tok >= 2.5 else "Poor"

            display_text = text if len(text) <= 37 else text[:34] + "..."
            print(f"{display_text:<40} {len(tokens):<10} {chars_per_tok:<12.2f} {efficiency:<12}")

    def visualize_tokenization(self, text: str) -> None:
        """Visual breakdown of tokenization"""
        analysis = self.analyze_text(text)

        print("\n🔍 Tokenization Analysis")
        print("=" * 70)
        print(f"Input Text: \"{text}\"")
        print(f"Length: {analysis['stats']['text_length']} characters, {analysis['stats']['text_bytes']} bytes")
        print(f"Tokens: {analysis['stats']['token_count']}")
        print(f"Efficiency: {analysis['stats']['chars_per_token']:.2f} chars/token")
        print("=" * 70)

        print("\n📝 Token Breakdown:")
        print(f"{'Pos':<5} {'Token ID':<10} {'Token Text':<30} {'Bytes':<8} {'Special':<8}")
        print("-" * 70)

        for token in analysis['tokens']:
            # Escape special characters for display
            token_display = repr(token['token_text'])[1:-1]  # Remove quotes from repr
            if len(token_display) > 27:
                token_display = token_display[:24] + "..."

            special = "Yes" if token['is_special'] else "No"

            print(f"{token['position']:<5} {token['token_id']:<10} {token_display:<30} {token['token_bytes']:<8} {special:<8}")

        print("\n💡 Special Tokens:")
        print(f"  BOS (Beginning of Sequence): {analysis['special_tokens']['bos']}")
        print(f"  EOS (End of Sequence):       {analysis['special_tokens']['eos']}")

    def test_tokenization_patterns(self) -> None:
        """Test common tokenization patterns"""
        test_cases = [
            # Basic text
            ("Hello, world!", "Simple greeting"),
            ("The quick brown fox", "Common words"),

            # Numbers
            ("1234567890", "Digits"),
            ("The year is 2024", "Text with number"),

            # Punctuation
            ("Hello!!!!!!", "Repeated punctuation"),
            ("user@example.com", "Email address"),
            ("#hashtag @mention", "Social media"),

            # Casing
            ("HELLO", "All caps"),
            ("hello", "All lowercase"),
            ("HeLLo", "Mixed case"),

            # Spacing
            ("Hello    world", "Multiple spaces"),
            ("Hello world", "Normal spacing"),

            # Special characters
            ("Hello 😀 world", "Emoji"),
            ("Hello\nworld", "Newline"),
            ("Hello\tworld", "Tab"),

            # Code
            ("def hello():", "Python code"),
            ("SELECT * FROM users", "SQL"),

            # Multilingual
            ("你好世界", "Chinese"),
            ("Привет мир", "Russian"),
            ("مرحبا العالم", "Arabic"),
        ]

        print("\n🧪 Tokenization Pattern Tests")
        print("=" * 100)
        print(f"{'Test Case':<30} {'Text':<30} {'Tokens':<10} {'Chars/Tok':<12}")
        print("-" * 100)

        for text, description in test_cases:
            tokens = self.tokenize(text)
            chars_per_tok = len(text) / len(tokens) if tokens else 0

            display_text = text if len(text) <= 27 else text[:24] + "..."

            print(f"{description:<30} {display_text:<30} {len(tokens):<10} {chars_per_tok:<12.2f}")


def demonstrate_encoding_reversibility(inspector: TokenizerInspector, text: str):
    """Demonstrate perfect encoding/decoding"""
    print("\n🔄 Encoding Reversibility Test")
    print("=" * 70)

    print(f"Original:    \"{text}\"")

    tokens = inspector.tokenize(text)
    print(f"Tokens:      {tokens}")

    decoded = inspector.detokenize(tokens)
    print(f"Decoded:     \"{decoded}\"")

    if text == decoded:
        print("✅ Perfect reversibility!")
    else:
        print("❌ Mismatch detected!")
        print(f"   Original bytes: {text.encode('utf-8')}")
        print(f"   Decoded bytes:  {decoded.encode('utf-8')}")


def analyze_prompt_efficiency(inspector: TokenizerInspector):
    """Analyze prompt efficiency"""
    prompts = [
        # Inefficient
        ("Please can you tell me about the history of artificial intelligence?", "Verbose"),

        # Efficient
        ("History of AI:", "Concise"),

        # Very inefficient
        ("I would like you to please provide me with information regarding...", "Very verbose"),

        # Optimal
        ("Explain:", "Minimal"),
    ]

    print("\n💰 Prompt Efficiency Analysis")
    print("=" * 100)
    print(f"{'Prompt':<70} {'Tokens':<10} {'Style':<15}")
    print("-" * 100)

    for prompt, style in prompts:
        tokens = inspector.tokenize(prompt)
        display_prompt = prompt if len(prompt) <= 67 else prompt[:64] + "..."

        print(f"{display_prompt:<70} {len(tokens):<10} {style:<15}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tokenizer_inspector.py <model.gguf> [text]")
        print("\nExample:")
        print("  python tokenizer_inspector.py model.gguf \"Hello, world!\"")
        print("\nOr run interactive tests:")
        print("  python tokenizer_inspector.py model.gguf")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"📚 Loading model: {model_path}")

    try:
        inspector = TokenizerInspector(model_path)
        print("✅ Model loaded successfully")

        if len(sys.argv) >= 3:
            # Analyze specific text
            text = sys.argv[2]
            inspector.visualize_tokenization(text)
            demonstrate_encoding_reversibility(inspector, text)
        else:
            # Run comprehensive tests
            inspector.test_tokenization_patterns()
            analyze_prompt_efficiency(inspector)

            # Compare some examples
            comparison_texts = [
                "Hello, world!",
                "antiestablishmentarianism",
                "The quick brown fox jumps over the lazy dog",
                "SELECT * FROM users WHERE id = 1",
            ]
            inspector.compare_texts(comparison_texts)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
