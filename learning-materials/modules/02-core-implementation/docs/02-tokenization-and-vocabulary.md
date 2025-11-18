# Tokenization and Vocabulary

**Learning Module**: Module 2 - Core Implementation
**Estimated Reading Time**: 25 minutes
**Prerequisites**: Module 1 complete, basic understanding of text processing
**Related Content**:
- [Model Architecture Deep Dive](./01-model-architecture-deep-dive.md)
- [Inference Pipeline](./04-inference-pipeline.md)
- [Grammar Constraints](./06-grammar-constraints.md)

---

## Overview

Tokenization is the process of converting text into numerical tokens that LLMs can process. Understanding tokenization is crucial for debugging generation issues, optimizing prompts, and implementing advanced features like grammar-guided generation.

### Learning Objectives

After completing this lesson, you will:
- ✅ Understand tokenization algorithms (BPE, SentencePiece, WordPiece)
- ✅ Work with vocabularies in llama.cpp
- ✅ Handle special tokens correctly
- ✅ Debug tokenization issues
- ✅ Optimize prompts for token efficiency

---

## Why Tokenization Matters

### The Token Economy

Models process text as sequences of tokens, not characters:

```
Text:    "Hello, world!"
Tokens:  [15496, 29892, 3186, 29991]  # LLaMA tokenization
Count:   4 tokens

vs.

Text:    "antiestablishmentarianism"
Tokens:  [424, 29875, 342, 370, 1674, 358, 13956, 1608]  # 8 tokens!
Count:   8 tokens
```

**Why This Matters**:
- **Context Windows**: Limited by token count, not characters
- **Cost**: API pricing based on tokens
- **Performance**: More tokens = slower generation
- **Prompt Engineering**: Must account for tokenization

---

## Tokenization Algorithms

### Byte Pair Encoding (BPE)

The most common tokenization algorithm for modern LLMs.

#### Core Concept

BPE iteratively merges the most frequent byte/character pairs:

```
Step 0: Start with characters
"hello" → ['h', 'e', 'l', 'l', 'o']

Step 1: Merge most frequent pair (e.g., 'll')
['h', 'e', 'll', 'o']

Step 2: Continue merging
['he', 'll', 'o']  (if 'he' is frequent)

Final: Vocabulary contains both:
- Base characters: ['a', 'b', 'c', ...]
- Merged tokens: ['he', 'll', 'ing', 'tion', ...]
```

#### Algorithm

```python
def bpe_encode(text, merges, vocab):
    """
    BPE encoding algorithm

    Args:
        text: Input string
        merges: List of (pair, merged_token) ordered by frequency
        vocab: Token to ID mapping

    Returns:
        List of token IDs
    """
    # Start with individual bytes/characters
    tokens = list(text)

    # Apply merges in order
    for (pair, merged) in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair:
                # Merge the pair
                tokens[i:i+2] = [merged]
            else:
                i += 1

    # Convert to IDs
    return [vocab[t] for t in tokens]
```

**Example**:
```python
text = "lower"
vocab = {'l': 1, 'o': 2, 'w': 3, 'e': 4, 'r': 5, 'lo': 6, 'low': 7, 'er': 8, 'lower': 9}
merges = [
    (('l', 'o'), 'lo'),
    (('lo', 'w'), 'low'),
    (('e', 'r'), 'er'),
    (('low', 'er'), 'lower')
]

# Encoding process:
['l', 'o', 'w', 'e', 'r']  # Initial
['lo', 'w', 'e', 'r']      # Merge l+o
['low', 'e', 'r']          # Merge lo+w
['low', 'er']              # Merge e+r
['lower']                  # Merge low+er

# Final: token ID = 9
```

### SentencePiece

Used by LLaMA and most modern LLMs. Key improvements over basic BPE:

**1. Language-Independent**
- Treats text as raw byte stream
- No pre-tokenization needed
- Works with any language (including no spaces)

**2. Reversible**
- Can perfectly reconstruct original text
- Special character (▁) represents spaces

**3. Vocabulary Size Control**
- Trains to exact vocabulary size
- Balances frequency and coverage

**Example**:
```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,
    model_type='bpe'  # or 'unigram'
)

# Load and use
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')

# Encode
text = "Hello, world!"
tokens = sp.encode(text, out_type=int)
# [15496, 29892, 3186, 29991]

# Decode (perfectly reversible)
decoded = sp.decode(tokens)
# "Hello, world!"

# With pieces (see subword units)
pieces = sp.encode(text, out_type=str)
# ['▁Hello', ',', '▁world', '!']
```

### tiktoken (OpenAI)

Used by GPT-3.5, GPT-4:

**Key Features**:
- Byte-level BPE
- Special regex patterns for efficiency
- Optimized for English but multilingual

```python
import tiktoken

# GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")

text = "Hello, world!"
tokens = enc.encode(text)
# [9906, 11, 1917, 0]

# Decode
decoded = enc.decode(tokens)
# "Hello, world!"
```

---

## Vocabulary Structure

### Components

A typical LLM vocabulary contains:

```
┌─────────────────────────────────────────────────────┐
│              Vocabulary Structure                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Base Tokens (256 bytes)                         │
│     - All possible bytes: 0x00 to 0xFF              │
│     - Ensures any text can be encoded               │
│                                                      │
│  2. Merged Subwords (~31,000 for LLaMA)            │
│     - Common words: "the", "and", "is"              │
│     - Subwords: "ing", "tion", "pre"                │
│     - Merged pairs: "Hello", "world"                │
│                                                      │
│  3. Special Tokens                                   │
│     - <s>: Beginning of sequence (BOS)              │
│     - </s>: End of sequence (EOS)                   │
│     - <unk>: Unknown token (rare in BPE)            │
│     - [INST], <<SYS>>: Instruction markers          │
│                                                      │
│  Total: ~32,000 tokens (LLaMA)                      │
│         ~128,000 tokens (LLaMA-3)                   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### LLaMA Vocabulary

```python
# LLaMA vocabulary structure
{
    # Byte tokens (0-255)
    0: '<unk>',
    1: '<s>',      # BOS
    2: '</s>',     # EOS
    3: '\x00',     # Byte 0
    4: '\x01',     # Byte 1
    # ...
    259: ' ',      # Space (important!)

    # Common words and subwords (260-31999)
    260: 'the',
    261: 'of',
    # ...
    15496: 'Hello',
    29892: ',',
    3186: '▁world',  # Note: ▁ represents space
    29991: '!',
}
```

### Special Tokens

**BOS (Beginning of Sequence)**:
```python
BOS_TOKEN = 1  # <s> in LLaMA

# Used at start of generation
prompt_tokens = [BOS_TOKEN] + encode(prompt)
```

**EOS (End of Sequence)**:
```python
EOS_TOKEN = 2  # </s> in LLaMA

# Model generates this to signal completion
if generated_token == EOS_TOKEN:
    stop_generation()
```

**Instruction Templates** (LLaMA-2 Chat):
```python
# LLaMA-2 chat format
template = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""

# Special tokens
'<s>': 1,        # BOS
'[INST]': 29961,
'<<SYS>>': 3532,
'<</SYS>>': 3532,
'[/INST]': 29962,
```

---

## Tokenization in llama.cpp

### Vocabulary Loading

```cpp
// From llama-vocab.cpp
struct llama_vocab {
    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    // Token to ID mapping
    std::unordered_map<token, id> token_to_id;

    // ID to token mapping
    std::vector<token> id_to_token;

    // Token types (normal, control, byte, etc.)
    std::vector<ttype> token_type;

    // SentencePiece model
    llama_vocab_type type;  // BPE, SPM, WPM

    // Special token IDs
    id bos_token_id = -1;   // Beginning of sequence
    id eos_token_id = -1;   // End of sequence
    id pad_token_id = -1;   // Padding
    id sep_token_id = -1;   // Separator
};
```

### Tokenization Process

```cpp
// Tokenize text
std::vector<llama_token> llama_tokenize(
    const struct llama_vocab & vocab,
    const std::string & text,
    bool add_bos,
    bool special  // Parse special tokens
) {
    std::vector<llama_token> output;

    // Add BOS if requested
    if (add_bos && vocab.bos_token_id != -1) {
        output.push_back(vocab.bos_token_id);
    }

    // SentencePiece tokenization
    if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
        // Apply BPE merges
        // ...
    }

    return output;
}
```

### Detokenization

```cpp
// Convert tokens back to text
std::string llama_detokenize(
    const struct llama_vocab & vocab,
    const std::vector<llama_token> & tokens,
    bool special  // Include special tokens in output
) {
    std::string text;

    for (const auto & token : tokens) {
        // Skip special tokens if requested
        if (!special && is_special_token(vocab, token)) {
            continue;
        }

        // Get token text
        std::string piece = vocab.id_to_token[token];

        // Handle SentencePiece space marker
        if (piece.starts_with("▁")) {
            piece = " " + piece.substr(3);  // ▁ is 3 bytes in UTF-8
        }

        text += piece;
    }

    return text;
}
```

---

## Common Tokenization Patterns

### Word Boundaries

```python
text = "Hello world"
tokens = tokenize(text)

# LLaMA (SentencePiece with ▁)
"Hello world" → ['▁Hello', '▁world']
#                 ^space   ^space

# GPT (no explicit space marker)
"Hello world" → ['Hello', ' world']
#                          ^space as token

# Important: Spaces are meaningful!
"Hello world" != "Helloworld"
```

### Numbers

```python
# Numbers often tokenize into digits
"1234" → ['1', '2', '3', '4']  # 4 tokens

# Year patterns may be single tokens
"2023" → ['2023']  # If trained on dates

# Large numbers are inefficient
"9876543210" → ['98', '76', '54', '32', '10']  # 5 tokens
```

### Punctuation

```python
# Punctuation usually separate tokens
"Hello, world!" → ['Hello', ',', '▁world', '!']

# But common patterns may merge
"don't" → ["don", "'", "t"]  # or ["don't"] if in vocab

# Special symbols
"@username" → ['@', 'user', 'name']
"#hashtag" → ['#', 'hash', 'tag']
```

### Unicode and Multilingual

```python
# SentencePiece handles Unicode well
"你好" → ['▁你', '好']  # Chinese

"مرحبا" → ['▁مرحبا']  # Arabic

# Emoji often single tokens
"😀" → ['😀']

# But complex emoji may split
"👨‍👩‍👧‍👦" → ['👨', '‍', '👩', '‍', '👧', '‍', '👦']
```

---

## Token Efficiency

### Optimizing Prompts

**Before**:
```python
prompt = "Please tell me about the history of artificial intelligence."
tokens = encode(prompt)
# 14 tokens
```

**After** (more token-efficient):
```python
prompt = "History of AI:"
tokens = encode(prompt)
# 5 tokens (saves 9 tokens!)
```

### Common Patterns

**Inefficient**:
```python
# Excessive punctuation
"Hello!!!!!!" → ['Hello', '!', '!', '!', '!', '!', '!']  # 7 tokens

# Repeated spaces
"Hello    world" → ['Hello', '▁', '▁', '▁', '▁world']  # 5 tokens

# Unusual formatting
"H E L L O" → ['H', '▁E', '▁L', '▁L', '▁O']  # 5 tokens
```

**Efficient**:
```python
"Hello!" → ['Hello', '!']  # 2 tokens

"Hello world" → ['Hello', '▁world']  # 2 tokens

"HELLO" → ['HELLO']  # 1 token (if in vocab)
```

### Token Counting

```python
def count_tokens(text):
    """Count tokens in text"""
    return len(encode(text))

# Examples
count_tokens("Hello, world!")  # 4
count_tokens("The quick brown fox")  # 5
count_tokens("antiestablishmentarianism")  # 8

# Rule of thumb (English):
# ~0.75 tokens per word on average
# ~4 characters per token on average
```

---

## Debugging Tokenization Issues

### Problem 1: Generation Doesn't Stop

**Symptom**: Model keeps generating, never produces EOS token.

**Cause**: EOS token not in stop criteria, or template issue.

**Solution**:
```python
# Ensure EOS is in stop tokens
stop_tokens = [
    tokenizer.eos_token_id,  # </s>
    tokenizer.encode("\n\n"),  # Double newline
]

generate(prompt, stop=stop_tokens)
```

### Problem 2: Unexpected Output Format

**Symptom**: Model output has weird spacing or formatting.

**Cause**: Tokenization boundary issues.

**Example**:
```python
# Problem: Inconsistent spacing
prompt = "Hello"  # No trailing space
output = "world"  # Model generates without leading space
result = "Helloworld"  # ❌ No space!

# Solution: Control spacing explicitly
prompt = "Hello "  # With trailing space
output = "world"
result = "Hello world"  # ✅ Correct
```

### Problem 3: Special Tokens in Output

**Symptom**: Output contains `<s>`, `</s>`, etc.

**Cause**: Special tokens not filtered during decoding.

**Solution**:
```python
# Filter special tokens
output = decode(tokens, skip_special_tokens=True)

# Or in llama.cpp
llama_token_to_piece(model, token, special=false)
```

### Problem 4: Token Limit Exceeded

**Symptom**: "Context length exceeded" error.

**Cause**: Input + output exceeds model's context window.

**Solution**:
```python
# Calculate available space
max_context = model.n_ctx()  # e.g., 4096
input_tokens = len(encode(prompt))
max_new_tokens = max_context - input_tokens - 100  # Safety margin

generate(prompt, max_tokens=max_new_tokens)
```

---

## Advanced Topics

### Byte Fallback

All modern tokenizers have byte fallback:

```python
# Unknown text can always be encoded
weird_text = "\x00\x01\x02"  # Control characters
tokens = encode(weird_text)
# Falls back to byte tokens [3, 4, 5]

decoded = decode(tokens)
assert decoded == weird_text  # Perfect reconstruction
```

### Vocabulary Expansion

Adding domain-specific tokens:

```python
# Medical domain
new_tokens = ["COVID-19", "mRNA", "antibody"]

# Add to vocabulary
tokenizer.add_tokens(new_tokens)

# Resize model embeddings (requires fine-tuning)
model.resize_token_embeddings(len(tokenizer))
```

### Token Healing

Fixing tokenization boundary issues:

```python
# Problem: Prompt ends mid-token
prompt = "The antiest"  # Incomplete word
# Model must continue from awkward position

# Solution: Token healing (backtrack to token boundary)
prompt_tokens = encode(prompt)
last_token = decode([prompt_tokens[-1]])

if not prompt.endswith(last_token):
    # Prompt ends mid-token, remove last token
    prompt_tokens = prompt_tokens[:-1]
    prompt = decode(prompt_tokens)
    # Now prompt = "The anti"
```

---

## Python API Examples

### Using llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf")

# Tokenize
text = "Hello, world!"
tokens = llm.tokenize(text.encode('utf-8'))
print(f"Tokens: {tokens}")
# [15496, 29892, 3186, 29991]

# Detokenize
text = llm.detokenize(tokens).decode('utf-8')
print(f"Text: {text}")
# "Hello, world!"

# Special tokens
bos_token = llm.token_bos()
eos_token = llm.token_eos()
print(f"BOS: {bos_token}, EOS: {eos_token}")
# BOS: 1, EOS: 2
```

### Analyzing Tokenization

```python
def analyze_tokenization(text):
    """Analyze how text is tokenized"""
    tokens = llm.tokenize(text.encode('utf-8'))

    print(f"Text: {text}")
    print(f"Total tokens: {len(tokens)}")
    print("\nToken breakdown:")

    for i, token_id in enumerate(tokens):
        token_text = llm.detokenize([token_id]).decode('utf-8', errors='replace')
        print(f"  {i:3d}: {token_id:6d} → '{token_text}'")

# Example
analyze_tokenization("Hello, world!")
# Output:
# Text: Hello, world!
# Total tokens: 4
#
# Token breakdown:
#     0:  15496 → 'Hello'
#     1:  29892 → ','
#     2:   3186 → ' world'
#     3:  29991 → '!'
```

---

## Interview Questions

**Q1: Why do models use subword tokenization instead of word-level tokenization?**

**Answer**:
- **Vocabulary Size**: Word-level requires 100K+ tokens; subword uses 32K
- **Unknown Words**: Subword can handle any word via decomposition
- **Morphology**: Captures word relationships (walk, walking, walked)
- **Multilingual**: Better for languages without clear word boundaries
- **Efficiency**: Balances vocabulary size and sequence length

**Q2: What is byte fallback and why is it important?**

**Answer**: Byte fallback ensures any input text can be encoded by falling back to individual byte tokens. Important because:
- Guarantees lossless encoding/decoding
- Handles any Unicode text, even malformed
- No true "unknown" token needed
- Enables perfect text reconstruction

**Q3: How does tokenization affect prompt engineering?**

**Answer**:
- Token boundaries matter (spaces, punctuation)
- Some phrases may be single vs multiple tokens
- Context windows are token-based, not character-based
- Token efficiency directly impacts performance and cost
- Must test prompts with actual tokenizer, not assume word boundaries

**Q4: Explain the difference between BPE and SentencePiece.**

**Answer**:
- BPE: Character/byte-pair merging algorithm
- SentencePiece: Implementation framework that can use BPE
- SentencePiece advantages:
  - Language-agnostic (byte-level)
  - Reversible (special space character ▁)
  - Trains to exact vocabulary size
  - No pre-tokenization needed

---

## Further Reading

### Code References
- [llama-vocab.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp): Vocabulary implementation
- [llama-sampling.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampling.cpp): Token sampling

### Research Papers
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): Original BPE paper
- [SentencePiece: A simple and language independent approach](https://arxiv.org/abs/1808.06226)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2, BPE usage

### Tutorials
- [Lab 2: Tokenization Deep Dive](../labs/lab-02-tokenization-deep-dive.ipynb)
- [Tutorial: Custom Tokenizer](../tutorials/tutorial-01-transformer-layers.ipynb)
- [Code Example: Tokenizer Inspector](../code/tokenizer_inspector.py)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Module**: 2 - Core Implementation
