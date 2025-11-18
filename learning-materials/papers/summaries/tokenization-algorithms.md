# Tokenization Algorithms for Large Language Models

**Paper Collection**: BPE, WordPiece, SentencePiece, Tiktoken
**Key Papers**: Neural Machine Translation of Rare Words (2015), SentencePiece (2018)
**Relevance**: Module 2 - Understanding LLM Architecture
**Reading Time**: 60-75 minutes
**Practical Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

Tokenization is the process of converting text into discrete units (tokens) that language models can process. The choice of tokenization algorithm significantly impacts model performance, training efficiency, vocabulary size, and multilingual capabilities. This document covers the evolution from word-level tokenization to modern subword methods used in LLaMA and GPT models.

**Key Insight**: Subword tokenization balances vocabulary size and flexibility—frequent words get single tokens, rare words are split into meaningful subunits.

---

## 1. Foundations: Why Tokenization Matters

### 1.1 The Tokenization Trilemma

**Three competing goals**:
1. **Small vocabulary** → Less memory, faster softmax
2. **Rich representation** → Handle rare words, compounds
3. **Language coverage** → Support multiple languages

**Trade-offs**:
```
Word-level tokenization:
  ✅ Semantic units preserved
  ❌ Huge vocabulary (millions of words)
  ❌ Can't handle unseen words (OOV problem)
  ❌ Poor for morphologically rich languages

Character-level tokenization:
  ✅ Tiny vocabulary (~256 characters)
  ✅ No OOV problem
  ❌ Very long sequences
  ❌ Models must learn to compose characters → words

Subword tokenization (BPE, WordPiece, etc.):
  ✅ Moderate vocabulary (32K-100K)
  ✅ Handles rare/unseen words via subunits
  ✅ Shorter sequences than character-level
  ✅ Good multilingual support
```

---

### 1.2 Tokenization Impact on Model Performance

**Example: "unbelievable"**

```python
# Word-level (if in vocabulary)
tokens = ["unbelievable"]  # 1 token

# Character-level
tokens = ["u", "n", "b", "e", "l", "i", "e", "v", "a", "b", "l", "e"]  # 12 tokens

# BPE (subword)
tokens = ["un", "believ", "able"]  # 3 tokens - preserves morphology!
```

**Sequence Length Impact**:
```
Text: "The quick brown fox jumps over the lazy dog."

Word-level: ~9 tokens
Character-level: ~44 tokens (including spaces/punctuation)
BPE (GPT-2): ~11 tokens
BPE (GPT-4): ~8 tokens (better compression)

Why it matters:
- Attention is O(n²) in sequence length
- Longer sequences = more memory + slower inference
- But: larger tokens = model must learn more token embeddings
```

---

## 2. Byte Pair Encoding (BPE)

### 2.1 Original BPE Algorithm (Gage, 1994)

**Paper**: "A New Algorithm for Data Compression"
**Original use**: Data compression, not NLP

**Algorithm**:
```python
def byte_pair_encoding(corpus, num_merges):
    """
    Learn BPE vocabulary from corpus

    Args:
        corpus: List of words (preprocessed text)
        num_merges: Number of merge operations (controls vocab size)

    Returns:
        List of merge operations (vocabulary)
    """
    import re
    from collections import Counter

    # Initialize vocabulary with character-level tokens
    vocab = Counter()
    for word in corpus:
        # Add end-of-word symbol
        word_tokens = ' '.join(list(word)) + ' </w>'
        vocab[word_tokens] += 1

    merges = []

    for i in range(num_merges):
        # Find most frequent pair
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pairs[(symbols[j], symbols[j+1])] += freq

        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # Merge the best pair in vocabulary
        vocab_new = {}
        bigram = ' '.join(best_pair)
        replacement = ''.join(best_pair)

        for word in vocab:
            new_word = word.replace(bigram, replacement)
            vocab_new[new_word] = vocab[word]

        vocab = vocab_new

    return merges

# Example usage
corpus = ['low', 'low', 'low', 'low', 'lowest', 'lowest', 'newer', 'newer', 'newer', 'wider']
merges = byte_pair_encoding(corpus, num_merges=10)

# Output (merge sequence):
# 1. ('l', 'o') → 'lo'
# 2. ('lo', 'w') → 'low'
# 3. ('e', 'r') → 'er'
# 4. ('n', 'e') → 'ne'
# 5. ('new', 'er') → 'newer'
# ... etc
```

**Tokenization with Learned BPE**:
```python
def tokenize_with_bpe(word, merges):
    """
    Apply learned BPE merges to tokenize a word

    Args:
        word: String to tokenize
        merges: List of merge operations (from training)

    Returns:
        List of tokens
    """
    word = ' '.join(list(word)) + ' </w>'
    pairs = get_pairs(word)

    if not pairs:
        return [word]

    while True:
        # Find the highest priority merge among current pairs
        bigram = min(pairs, key=lambda p: merges.index(p) if p in merges else float('inf'))

        if bigram not in merges:
            break

        # Apply the merge
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break

            if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        word = new_word
        if len(word) == 1:
            break
        pairs = get_pairs(word)

    return word

def get_pairs(word):
    """Get all adjacent pairs of symbols in word"""
    pairs = set()
    prev = word[0]
    for char in word[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs

# Example
word = "lowest"
tokens = tokenize_with_bpe(word, merges)
# Output: ['low', 'est', '</w>']
```

---

### 2.2 Neural Machine Translation BPE (Sennrich et al., 2016)

**Paper**: "Neural Machine Translation of Rare Words with Subword Units"
**Link**: https://arxiv.org/abs/1508.07909
**Innovation**: Apply BPE to NMT to handle rare words

**Key Modifications**:
1. Operate on characters (not bytes)
2. Add special end-of-word symbol
3. Learn separate vocabularies for source and target languages (or joint)

**Example**:
```python
# English corpus
corpus_en = ["walking", "walked", "walk", "walks", "walker"]

# BPE learns morphology:
# Merge 1: 'w' + 'a' → 'wa'
# Merge 2: 'wa' + 'l' → 'wal'
# Merge 3: 'wal' + 'k' → 'walk'
# Merge 4: 'e' + 'd' → 'ed'
# Merge 5: 'i' + 'n' + 'g' → 'ing'

# Final vocabulary includes:
vocab = ['walk', 'ed', 'ing', 's', 'er', ...]

# Tokenization:
"walking" → ['walk', 'ing']
"walked" → ['walk', 'ed']
"walker" → ['walk', 'er']
"walks" → ['walk', 's']
```

**Benefits for NMT**:
- Handles compound words in German (e.g., "Donaudampfschifffahrt")
- Reduces vocabulary size from millions to ~32K tokens
- Enables translation of rare/unseen words

---

### 2.3 Byte-Level BPE (GPT-2, 2019)

**Paper**: "Language Models are Unsupervised Multitask Learners"
**Innovation**: Operate on bytes instead of characters

**Motivation**:
```
Problem with character-level BPE:
- Large base vocabulary for Unicode (~140K+ characters)
- Special handling for different scripts (Latin, Cyrillic, CJK, etc.)

Solution:
- Work with bytes (0-255) → only 256 base symbols
- Can represent ANY text (universal encoding)
- No special preprocessing needed
```

**Implementation**:
```python
import regex as re

class ByteLevelBPE:
    def __init__(self):
        # Byte to Unicode mapping (for printable representation)
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self):
        """
        Map bytes to Unicode strings.
        Avoids mapping to whitespace/control characters.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1)) +
            list(range(ord("¡"), ord("¬") + 1)) +
            list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def encode(self, text):
        """Convert text to byte sequence"""
        return [self.byte_encoder[b] for b in text.encode('utf-8')]

    def decode(self, tokens):
        """Convert byte tokens back to text"""
        bytes_array = bytes([self.byte_decoder[t] for t in tokens])
        return bytes_array.decode('utf-8', errors='replace')

# Example
bpe = ByteLevelBPE()
text = "Hello 世界"  # Mixed English + Chinese

# Encode to bytes
byte_tokens = bpe.encode(text)
# ['H', 'e', 'l', 'l', 'o', ' ', 'ä¸', '–', 'ç', '•', ...']

# BPE merges operate on these byte-level tokens
# After training: common sequences like "Hello" become single tokens
```

**Advantages**:
- ✅ Universal: handles ANY text (including emoji, rare scripts)
- ✅ Small base vocabulary (256 vs 140K+)
- ✅ No preprocessing needed (no lowercasing, Unicode normalization, etc.)
- ✅ Consistent across languages

**Disadvantages**:
- ❌ Less interpretable tokens
- ❌ Some characters (e.g., Chinese) require multiple tokens
- ❌ Slightly longer sequences for non-English text

---

## 3. WordPiece (Google, 2016)

### 3.1 Algorithm

**Paper**: "Japanese and Korean Voice Search" (Schuster & Nakajima, 2012)
**Used in**: BERT, Gemini

**Key Difference from BPE**: Uses likelihood maximization instead of frequency

**Algorithm**:
```python
def wordpiece_training(corpus, vocab_size):
    """
    Train WordPiece vocabulary

    Instead of merging most frequent pairs,
    merge pairs that maximize likelihood of training data
    """
    import math
    from collections import Counter

    # Initialize with characters
    vocab = set([char for word in corpus for char in word])
    vocab.add('[UNK]')  # Unknown token

    while len(vocab) < vocab_size:
        # Count all possible pairs
        pair_scores = {}

        for word, freq in corpus.items():
            symbols = split_word(word, vocab)

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                merged = symbols[i] + symbols[i+1]

                # Score = P(merged) / (P(symbols[i]) * P(symbols[i+1]))
                # Higher score = better merge (more likely as single unit)
                score = freq / (pair_scores.get(pair, 1e-10))
                pair_scores[pair] = score

        if not pair_scores:
            break

        # Merge pair with highest score
        best_pair = max(pair_scores, key=pair_scores.get)
        vocab.add(''.join(best_pair))

    return vocab

def split_word(word, vocab):
    """
    Split word into longest matching subwords from vocab
    """
    if word in vocab:
        return [word]

    subwords = []
    start = 0

    while start < len(word):
        # Find longest matching subword
        end = len(word)
        found = False

        while start < end:
            substr = word[start:end]
            if substr in vocab or start == 0:
                if start > 0:
                    substr = '##' + substr  # Continuation marker
                if substr in vocab:
                    subwords.append(substr)
                    found = True
                    break
            end -= 1

        if not found:
            subwords.append('[UNK]')
            start += 1
        else:
            start = end

    return subwords
```

**Example**:
```python
# BERT WordPiece tokenization
text = "unaffable"

# Vocabulary (excerpt):
vocab = ["un", "##aff", "##able", ...]

# Tokenization:
tokens = ["un", "##aff", "##able"]
# Note: ## prefix indicates continuation (not word start)
```

**Comparison to BPE**:
```
BPE: Greedy frequency-based merging
  - Faster to train
  - Simpler algorithm
  - May not optimize for language modeling

WordPiece: Likelihood-based merging
  - Optimizes for language modeling objective
  - Slightly more complex
  - Used in BERT (better for understanding tasks)
```

---

## 4. SentencePiece (Google, 2018)

### 4.1 Motivation

**Paper**: "SentencePiece: A simple and language independent text tokenizer"
**Link**: https://arxiv.org/abs/1808.06226
**Authors**: Taku Kudo, John Richardson

**Problem with BPE/WordPiece**:
1. Require pre-tokenization (word splitting)
2. Language-specific rules (e.g., spaces in English, no spaces in Chinese)
3. Irreversible (can't perfectly reconstruct original text)
4. Preprocessing dependencies (lowercasing, Unicode normalization)

**SentencePiece Solution**:
- Treat text as raw byte/character stream
- No pre-tokenization
- Language-agnostic
- Reversible (lossless encoding/decoding)

---

### 4.2 Algorithm

**Two modes**: BPE and Unigram Language Model

#### Mode 1: SentencePiece with BPE
```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='m',
    vocab_size=32000,
    model_type='bpe',  # or 'unigram'
    normalization_rule_name='identity',  # No normalization
    add_dummy_prefix=True,  # Add space at start (for better word boundary)
    byte_fallback=True  # Handle any UTF-8 byte
)

# Load and use
sp = spm.SentencePieceProcessor(model_file='m.model')

# Encode
text = "Hello world!"
tokens = sp.encode(text, out_type=str)
# ['▁Hello', '▁world', '!']
# Note: ▁ (U+2581) represents space

ids = sp.encode(text, out_type=int)
# [123, 456, 789]

# Decode (lossless!)
reconstructed = sp.decode(ids)
# "Hello world!" - exactly the same as input!
```

#### Mode 2: Unigram Language Model

**Key Innovation**: Train tokenizer to maximize likelihood directly

**Algorithm**:
```python
def unigram_tokenization(word, vocab_probs):
    """
    Use dynamic programming to find best tokenization

    Args:
        word: String to tokenize
        vocab_probs: Dict mapping subword → log probability

    Returns:
        Best tokenization (maximizes total probability)
    """
    n = len(word)

    # dp[i] = (best_score, best_segmentation) for word[:i]
    dp = [(-float('inf'), [])] * (n + 1)
    dp[0] = (0.0, [])

    for i in range(1, n + 1):
        for j in range(i):
            subword = word[j:i]
            if subword in vocab_probs:
                score = dp[j][0] + vocab_probs[subword]
                if score > dp[i][0]:
                    dp[i] = (score, dp[j][1] + [subword])

    return dp[n][1]

# Training: Expectation-Maximization algorithm
def train_unigram_model(corpus, vocab_size):
    """
    1. Start with large vocabulary (all substrings up to some length)
    2. Repeat until vocab_size reached:
       a. Compute probability of each subword (E-step)
       b. Remove low-probability subwords (M-step)
    3. Return final vocabulary with probabilities
    """
    # Initialize with all possible substrings
    vocab = initialize_large_vocab(corpus)

    while len(vocab) > vocab_size:
        # E-step: Compute expected counts
        for subword in vocab:
            compute_expected_count(subword, corpus)

        # M-step: Remove worst subwords
        vocab = prune_vocab(vocab, keep_ratio=0.8)

    return vocab
```

**Example**:
```python
# Unigram model learns probabilities:
vocab_probs = {
    'hello': -2.3,      # log P('hello') = -2.3
    'hel': -5.1,
    'lo': -4.2,
    'world': -2.8,
    ...
}

# Tokenization finds highest probability split:
"hello" → ['hello']          # Score: -2.3 (better!)
"hello" → ['hel', 'lo']      # Score: -5.1 + -4.2 = -9.3

# Handles ambiguity optimally
```

---

### 4.3 SentencePiece Features

**Special Tokens**:
```python
# SentencePiece reserves first few IDs
0: <unk>  # Unknown token
1: <s>    # Begin of sentence
2: </s>   # End of sentence
3+: User-defined special tokens

# Example
sp.piece_to_id('<s>')   # → 1
sp.id_to_piece(0)       # → '<unk>'
```

**Whitespace Handling**:
```python
# SentencePiece encodes spaces as '▁' (U+2581)
text = "hello world"
tokens = sp.encode(text, out_type=str)
# ['▁hello', '▁world']

# Decoding is lossless
sp.decode(tokens)  # "hello world"

# Handles leading/trailing spaces correctly
text = "  hello  "
tokens = sp.encode(text, out_type=str)
# ['▁▁hello', '▁▁']
sp.decode(tokens)  # "  hello  " - exactly preserved!
```

---

## 5. Tiktoken (OpenAI, 2022)

### 5.1 Design Goals

**Used in**: GPT-3.5, GPT-4, ChatGPT
**Repository**: https://github.com/openai/tiktoken

**Improvements over GPT-2 BPE**:
1. **Faster**: Rust implementation (10-100× faster than pure Python)
2. **More efficient encoding**: Better compression (fewer tokens)
3. **Special token handling**: Structured support for system/user/assistant roles
4. **Deterministic**: Same input always produces same tokens

---

### 5.2 Implementation

```python
import tiktoken

# Load different tokenizers
enc_gpt2 = tiktoken.get_encoding("gpt2")           # ~50K vocab
enc_gpt35 = tiktoken.get_encoding("cl100k_base")   # ~100K vocab (GPT-3.5, GPT-4)

# Encode text
text = "Hello, world! This is a test."

tokens_gpt2 = enc_gpt2.encode(text)
tokens_gpt35 = enc_gpt35.encode(text)

print(f"GPT-2 tokens: {len(tokens_gpt2)}")    # ~9 tokens
print(f"GPT-3.5 tokens: {len(tokens_gpt35)}")  # ~7 tokens (better compression)

# Decode
reconstructed = enc_gpt35.decode(tokens_gpt35)
assert reconstructed == text  # Lossless!

# Inspect tokens
for token in tokens_gpt35:
    print(f"{token}: {enc_gpt35.decode([token])}")

# Handle special tokens
enc_chat = tiktoken.encoding_for_model("gpt-3.5-turbo")
text_with_special = "<|im_start|>system\nYou are a helpful assistant<|im_end|>"
tokens = enc_chat.encode(text_with_special, allowed_special="all")
```

**Performance Comparison**:
```python
import time

text = "Hello world" * 1000

# GPT-2 BPE (Python implementation)
start = time.time()
for _ in range(100):
    enc_gpt2.encode(text)
python_time = time.time() - start

# Tiktoken (Rust implementation)
start = time.time()
for _ in range(100):
    enc_gpt35.encode(text)
rust_time = time.time() - start

print(f"Python: {python_time:.3f}s")
print(f"Rust: {rust_time:.3f}s")
print(f"Speedup: {python_time / rust_time:.1f}×")
# Typical speedup: 10-50× depending on text
```

---

## 6. LLaMA Tokenization

### 6.1 LLaMA 1 & 2 Tokenizer

**Algorithm**: SentencePiece BPE
**Vocabulary Size**: 32,000 tokens
**Training Data**: Multilingual corpus (English, Spanish, French, German, Italian, Portuguese, Polish, Dutch, Romanian, Czech, Swedish)

**Implementation**:
```python
from sentencepiece import SentencePieceProcessor

# Load LLaMA tokenizer
sp = SentencePieceProcessor(model_file='tokenizer.model')

# Encode
text = "The quick brown fox"
tokens = sp.encode(text, out_type=str)
# ['▁The', '▁quick', '▁brown', '▁fox']

ids = sp.encode(text, out_type=int)
# [450, 4996, 17354, 1701]

# Special tokens
print(sp.bos_id())  # Beginning of sequence: 1
print(sp.eos_id())  # End of sequence: 2
print(sp.unk_id())  # Unknown: 0
```

**Vocabulary Breakdown**:
```
Token ID ranges:
0: <unk>
1: <s>     (BOS)
2: </s>    (EOS)
3-31: Special/reserved
32-255: Byte fallback tokens (for any UTF-8 byte)
256-31999: Learned BPE tokens
```

---

### 6.2 LLaMA 3 Tokenizer

**Key Improvements**:
- **Vocabulary Size**: 128,000 tokens (4× larger than LLaMA 2)
- **Better compression**: ~1.3× fewer tokens for same text
- **Improved multilingual support**: Better for non-English languages
- **Code-aware**: Better handling of programming languages

**Comparison**:
```python
# Same text tokenized with different models
text = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# LLaMA 2 (32K vocab)
tokens_llama2 = tokenizer_llama2.encode(text)
print(f"LLaMA 2: {len(tokens_llama2)} tokens")  # ~45 tokens

# LLaMA 3 (128K vocab)
tokens_llama3 = tokenizer_llama3.encode(text)
print(f"LLaMA 3: {len(tokens_llama3)} tokens")  # ~35 tokens (22% reduction)
```

**Impact on Inference**:
```
Fewer tokens = faster inference (attention is O(n²))
128K context:
- With LLaMA 2 tokenizer: ~128K tokens
- With LLaMA 3 tokenizer: ~98K tokens equivalent (30% more text capacity)
```

---

## 7. Practical Implications for llama.cpp

### 7.1 Tokenizer Detection and Usage

```python
from gguf import GGUFReader

def inspect_tokenizer(model_path):
    """Extract tokenizer information from GGUF model"""
    reader = GGUFReader(model_path)

    tokenizer_model = reader.fields.get('tokenizer.ggml.model', 'unknown')
    vocab_size = len(reader.tensors)  # Vocabulary size

    print(f"Tokenizer: {tokenizer_model}")
    print(f"Vocabulary size: {vocab_size}")

    # Extract special tokens
    bos_token = reader.fields.get('tokenizer.ggml.bos_token_id')
    eos_token = reader.fields.get('tokenizer.ggml.eos_token_id')
    unknown_token = reader.fields.get('tokenizer.ggml.unknown_token_id')

    print(f"BOS token ID: {bos_token}")
    print(f"EOS token ID: {eos_token}")
    print(f"UNK token ID: {unknown_token}")

# Usage
inspect_tokenizer("llama-2-7b.gguf")
# Tokenizer: llama
# Vocabulary size: 32000
# BOS token ID: 1
# EOS token ID: 2
```

### 7.2 Token Counting for Context Management

```python
def estimate_tokens(text, model="llama2"):
    """
    Rough estimation of token count without loading full model

    Rule of thumb:
    - English: ~0.75 tokens per word
    - Code: ~1.2 tokens per word
    - Chinese: ~1.5 tokens per character
    """
    if model == "llama2":
        # 32K vocab - more tokens needed
        return int(len(text.split()) * 0.75)
    elif model == "llama3":
        # 128K vocab - better compression
        return int(len(text.split()) * 0.58)
    elif model == "gpt4":
        # cl100k_base - very efficient
        return int(len(text.split()) * 0.6)

# Example
text = "The quick brown fox jumps over the lazy dog" * 100

print(f"Words: {len(text.split())}")
print(f"LLaMA 2 tokens (estimate): {estimate_tokens(text, 'llama2')}")
print(f"LLaMA 3 tokens (estimate): {estimate_tokens(text, 'llama3')}")
```

### 7.3 Special Token Handling in llama.cpp

```cpp
// llama.cpp - Special token handling
struct llama_context {
    llama_token bos_token = 1;   // <s>
    llama_token eos_token = 2;   // </s>
    llama_token nl_token = 13;   // \n (for chat templates)
};

// Tokenization with special tokens
std::vector<llama_token> tokenize(
    const std::string & text,
    bool add_bos,
    bool special
) {
    std::vector<llama_token> tokens;

    if (add_bos) {
        tokens.push_back(bos_token);
    }

    // ... tokenize text ...

    return tokens;
}
```

**Python Bindings**:
```python
from llama_cpp import Llama

llm = Llama(model_path="llama-2-7b.gguf")

# Tokenize with BOS token
tokens = llm.tokenize(b"Hello world", add_bos=True)
# [1, 15043, 3186]  # 1 = BOS token

# Tokenize without BOS
tokens = llm.tokenize(b"Hello world", add_bos=False)
# [15043, 3186]
```

---

## 8. Key Takeaways

### 8.1 Algorithm Comparison

| Algorithm | Vocab Size | Training | Reversible | Multilingual | Used In |
|-----------|------------|----------|------------|--------------|---------|
| BPE | 32K-50K | Fast (frequency) | No | Fair | GPT-2, RoBERTa |
| Byte-level BPE | 50K | Fast | Yes | Good | GPT-3, GPT-4 |
| WordPiece | 30K | Slow (likelihood) | No | Fair | BERT, Gemini |
| SentencePiece | 32K-100K | Medium | Yes | Excellent | T5, LLaMA, mT5 |
| Unigram LM | Flexible | Slow (EM algorithm) | Yes | Excellent | T5, XLNet |

### 8.2 Best Practices

✅ **Model Selection**:
- English-only: Any algorithm works well
- Multilingual: SentencePiece > BPE
- Code: Larger vocab (128K like LLaMA 3) > smaller vocab

✅ **Vocabulary Size**:
- Smaller (32K): Faster softmax, larger memory for embeddings
- Larger (128K): Better compression, slower softmax
- Sweet spot: 32K-64K for most applications

✅ **Context Management**:
- Monitor token count, not character count
- Different models have different compression rates
- Leave buffer for completion tokens

✅ **Special Tokens**:
- Always check BOS/EOS requirements
- Chat models need special formatting (system/user/assistant)
- Don't forget to decode with special tokens properly

---

## 9. Further Reading

### Essential Papers
1. **BPE for NMT** (Sennrich et al., 2016)
   - https://arxiv.org/abs/1508.07909
   - Foundation of modern tokenization

2. **SentencePiece** (Kudo & Richardson, 2018)
   - https://arxiv.org/abs/1808.06226
   - Best multilingual tokenization

3. **Byte-level BPE** (GPT-2 paper, 2019)
   - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
   - Section on tokenization

### Tools and Libraries
- **SentencePiece**: https://github.com/google/sentencepiece
- **Tiktoken**: https://github.com/openai/tiktoken
- **Hugging Face Tokenizers**: https://github.com/huggingface/tokenizers
- **llama.cpp tokenization**: `llama-tokenize` tool

### Interactive Resources
- **Tiktoken Playground**: https://platform.openai.com/tokenizer
- **Hugging Face Tokenizer Playground**: Try different tokenizers online

---

**Document Information**
- Created: 2025-11-18
- Module: 2 - Understanding LLM Architecture
- Author: Research Coordinator
- Status: Complete
- Module 2 Complete: All 3 papers finished
