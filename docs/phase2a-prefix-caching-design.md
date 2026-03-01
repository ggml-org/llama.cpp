# Phase 2A: Prefix Caching Design

> **Goal**: Block sharing for common prefixes, Copy-on-Write for diverging suffix
> **Reference**: vLLM technical report - "shared prefix directly reuses physical blocks"
> **Target**: Agent/Tool-chain scenario TTFT reduction

---

## 1. Problem Statement

### Current Issue
In Agent/Tool-chain scenarios (like multi-turn conversations with system prompts):
- Each new request processes the same prefix (system prompt, conversation history) from scratch
- Same KV cache values are computed repeatedly
- TTFT (Time To First Token) wasted on redundant prefill

### Solution
**Prefix Caching** - Reuse KV cache blocks for identical prefixes:
1. Hash token sequences to identify reusable blocks
2. Share physical blocks between sequences (increment ref_count)
3. Copy-on-Write when suffix diverges

---

## 2. Architecture

### 2.1 Block Pool Extension

```
┌─────────────────────────────────────────────────────────────────┐
│              Block Pool with Prefix Caching                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Current:                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │ Block 0 │    │ Block 1 │    │ Block N │                    │
│   │ ref_cnt │    │ ref_cnt │    │ ref_cnt │                    │
│   │ seq_id  │    │ seq_id  │    │ seq_id  │                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                                                                 │
│   Extended:                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                      Block Metadata                      │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │  • ref_count     - Reference count (CoW)                 │   │
│   │  • is_free       - Block availability                    │   │
│   │  • seq_id        - Owner sequence (-1 if shared)         │   │
│   │  • logical_block - Position in sequence                  │   │
│   │  • token_hash    - Hash of tokens in this block (NEW)    │   │
│   │  • prefix_ref    - Reference to prefix block (NEW)       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   NEW: Prefix Hash Index                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  std::unordered_map<token_hash, block_id>                │   │
│   │                                                          │   │
│   │  token_hash = hash(token_sequence)                       │   │
│   │  → Find existing block for same prefix                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Token Hash Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Token Hash Strategy                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Option A: Full Block Hash (vLLM approach)                     │
│   ────────────────────────────────────────────────────────────  │
│   • Hash all tokens in a block                                  │
│   • Exact match only                                            │
│   • Simple, deterministic                                       │
│                                                                 │
│   hash = fnv1a_64(tokens[0:block_size])                         │
│                                                                 │
│   Option B: Rolling Hash (more flexible)                        │
│   ────────────────────────────────────────────────────────────  │
│   • Support partial block matching                              │
│   • Better for variable-length prefixes                         │
│   • More complex                                                │
│                                                                 │
│   Option C: Tree-based (advanced)                               │
│   ────────────────────────────────────────────────────────────  │
│   • Radix tree for prefix sharing                               │
│   • Maximum reuse                                               │
│   • Higher memory overhead                                      │
│                                                                 │
│   Decision: Start with Option A (Full Block Hash)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Copy-on-Write Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  Copy-on-Write Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Sequence A: [prefix_block_0] [suffix_A_1]                     │
│   Sequence B: [prefix_block_0] [suffix_B_1]  (share prefix)     │
│                                                                 │
│   Step 1: Sequence A allocates block_0                          │
│           block_0.ref_count = 1                                 │
│           block_0.token_hash = hash(prefix_tokens)              │
│                                                                 │
│   Step 2: Sequence B needs same prefix                          │
│           Found block_0 with matching hash                      │
│           block_0.ref_count = 2  (shared)                       │
│                                                                 │
│   Step 3: Sequence A writes to block_0 (needs modification)     │
│           if block_0.ref_count > 1:                             │
│               new_block = copy(block_0)                         │
│               block_0.ref_count--                               │
│               Sequence A uses new_block                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. API Changes

### 3.1 llama_block_pool.h

```cpp
// NEW: Token hash type
using token_hash_t = uint64_t;

// NEW: Hash function
token_hash_t compute_token_hash(const llama_token * tokens, size_t n_tokens);

struct llama_block_pool {
    // ... existing members ...

    // NEW: Prefix hash index
    std::unordered_map<token_hash_t, int32_t> hash_to_block;

    // NEW: Find block by token hash (for prefix reuse)
    int32_t find_block_by_hash(token_hash_t hash) const;

    // NEW: Share block (increment ref_count, don't allocate new)
    bool share_block(int32_t block_id, llama_seq_id seq_id, int32_t logical_block);

    // NEW: Copy-on-Write - copy block if ref_count > 1
    int32_t cow_block(int32_t block_id);

    // NEW: Register block hash (called when block is filled)
    void register_block_hash(int32_t block_id, token_hash_t hash);
};
```

### 3.2 llama-kv-cache.h

```cpp
class llama_kv_cache {
public:
    // ... existing members ...

    // NEW: Enable/disable prefix caching
    void set_prefix_caching(bool enable);
    bool get_prefix_caching() const { return prefix_caching; }

    // NEW: Try to reuse prefix blocks for a sequence
    // Returns number of blocks reused
    size_t try_reuse_prefix(llama_seq_id seq_id, const llama_token * tokens, size_t n_tokens);

private:
    // NEW: Prefix caching flag
    bool prefix_caching = false;
};
```

---

## 4. Implementation Plan

### Phase 2A.1: Core Infrastructure (Estimated: 200 lines)

1. **Add token hash function**
   - FNV-1a 64-bit hash
   - Hash tokens in block_size chunks

2. **Extend block_info structure**
   - Add `token_hash` field
   - Update constructor/initialization

3. **Add hash_to_block index**
   - std::unordered_map for O(1) lookup
   - Thread-safe access (if needed)

### Phase 2A.2: Block Sharing (Estimated: 150 lines)

4. **Implement find_block_by_hash()**
   - Lookup in hash_to_block
   - Return block_id or -1

5. **Implement share_block()**
   - Increment ref_count
   - Update block table for new sequence
   - Don't allocate new physical block

6. **Implement try_reuse_prefix()**
   - Hash incoming tokens
   - Find matching blocks
   - Share all matching prefix blocks

### Phase 2A.3: Copy-on-Write (Estimated: 100 lines)

7. **Implement cow_block()**
   - Check ref_count
   - If > 1: allocate new block, copy data, decrement old ref_count
   - Return new block_id (or same if no copy needed)

8. **Integrate CoW into write path**
   - Before writing to shared block
   - Ensure exclusive access via CoW

### Phase 2A.4: Testing & Validation (Estimated: 100 lines + test scripts)

9. **Unit tests**
   - Hash computation correctness
   - Block sharing ref_count
   - CoW behavior

10. **Integration tests**
    - Multi-turn conversation TTFT
    - Agent tool-chain scenarios
    - Memory efficiency

---

## 5. Expected Benefits

### 5.1 TTFT Reduction

```
Scenario: Multi-turn conversation with 4K context prefix

Without Prefix Caching:
  Turn 1: prefill 4K + generate → TTFT = X
  Turn 2: prefill 4K + generate → TTFT = X (redundant!)
  Turn 3: prefill 4K + generate → TTFT = X (redundant!)

With Prefix Caching:
  Turn 1: prefill 4K + generate → TTFT = X
  Turn 2: reuse 4K + generate → TTFT = Y (Y << X)
  Turn 3: reuse 4K + generate → TTFT = Y

Expected improvement: 50-90% TTFT reduction for cached prefixes
```

### 5.2 Memory Efficiency

```
Without sharing:
  Sequence A: 4K tokens = 256 blocks (block_size=16)
  Sequence B: 4K tokens = 256 blocks
  Total: 512 blocks

With prefix sharing (2K common prefix):
  Shared prefix: 128 blocks (ref_count=2)
  A suffix: 128 blocks
  B suffix: 128 blocks
  Total: 384 blocks (25% reduction)
```

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Hash collision | Use 64-bit hash, add collision detection |
| Memory overhead | Hash index is small (~8 bytes per block) |
| CoW latency | Only copy when needed, lazy copy |
| Complexity | Incremental implementation, extensive testing |

---

## 7. Success Metrics

| Metric | Target |
|--------|--------|
| TTFT reduction (cached prefix) | >50% |
| Memory savings (shared prefix) | >20% |
| No correctness regression | 100% pass |
| No performance regression (cold) | <5% |

---

## 8. References

1. vLLM Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. SGLang: "RadixAttention" for prefix caching
3. llama.cpp existing block pool implementation
