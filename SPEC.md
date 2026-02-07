# Full Technical Specification — Deterministic Runtime Optimization Layer

All phases, all optimizations. This document is the authoritative specification.

---

## Table of Contents

- [Phase 1: High-Impact Foundations](#phase-1-high-impact-foundations)
  - [1.1 Optimization Infrastructure](#11-optimization-infrastructure-llama-opt)
  - [1.2 Context Block Hashing](#12-context-block-hashing-llama-context-hash)
  - [1.3 Context Block Deduplication](#13-context-block-deduplication-llama-kv-cache-dedup)
  - [1.4 Structural KV Cache Diffing](#14-structural-kv-cache-diffing-llama-kv-cache-diff)
  - [1.5 Schema-Aware Token Skipping](#15-schema-aware-token-skipping-llama-schema-skip)
- [Phase 2: Caching Infrastructure](#phase-2-caching-infrastructure)
  - [2.1 KV Cache Canonicalization](#21-kv-cache-canonicalization)
  - [2.2 Deterministic Precomputation](#22-deterministic-precomputation)
  - [2.3 Persistent Cross-Session KV Cache](#23-persistent-cross-session-kv-cache)
- [Phase 3: Output-Side Optimization](#phase-3-output-side-optimization)
  - [3.1 Token-Level Output Memoization](#31-token-level-output-memoization)
- [Phase 4: Approximate Optimizations (Tier 2)](#phase-4-approximate-optimizations-tier-2)
  - [4.1 Attention Sink Pruning](#41-attention-sink-pruning)
  - [4.2 Sparse Attention over Tool Results](#42-sparse-attention-over-tool-results)
- [Phase 5: Low-Priority Exact Optimizations](#phase-5-low-priority-exact-optimizations)
  - [5.1 Activation Deduplication (CSE)](#51-activation-deduplication-cse)
  - [5.2 No-Op Detection and Elision](#52-no-op-detection-and-elision)
  - [5.3 Exact Algebraic Rewriting](#53-exact-algebraic-rewriting)
- [Phase 6: Stabilization](#phase-6-stabilization)
- [Complete File Manifest](#complete-file-manifest)
- [CMake Feature Flags](#cmake-feature-flags)
- [Runtime Configuration](#runtime-configuration)

---

# Phase 1: High-Impact Foundations

**Tier: Exact (Tier 1) | All optimizations produce bitwise identical outputs.**

## 1.1 Optimization Infrastructure (`llama-opt`)

### Configuration

```cpp
struct llama_opt_config {
    // Block hashing
    uint32_t block_size;          // Token block size for hashing (default: 64)

    // Phase 1 — Context block deduplication
    bool     dedup_enabled;       // Enable context block dedup (default: true)
    uint32_t dedup_pool_max;      // Max entries in the block pool (default: 16384)

    // Phase 1 — Structural KV cache diffing
    bool     diff_enabled;        // Enable KV cache diffing (default: true)
    uint32_t diff_min_unchanged;  // Min unchanged span to skip recompute (default: 8)

    // Phase 1 — Schema-aware token skipping
    bool     schema_skip_enabled; // Enable schema-aware skipping (default: true)

    // Phase 2 — KV cache canonicalization
    bool     canon_enabled;       // Enable KV canonicalization (default: true)

    // Phase 2 — Deterministic precomputation
    bool     precompute_enabled;  // Enable precomputation cache (default: true)
    uint32_t precompute_rope_max; // Max RoPE cache entries (default: 131072)

    // Phase 2 — Persistent cross-session KV cache
    bool        persist_enabled;  // Enable persistent KV (default: false)
    std::string persist_dir;      // Directory for KV cache files

    // Phase 3 — Token-level output memoization
    bool     memo_enabled;        // Enable output memoization (default: true)
    uint32_t memo_trie_max;       // Max trie nodes (default: 65536)

    // Phase 4 — Attention sink pruning (Tier 2)
    bool     attn_prune_enabled;  // Enable attention pruning (default: false)
    uint32_t attn_prune_window;   // Turns of inactivity before eviction (default: 10)
    float    attn_prune_threshold;// Attention weight threshold (default: 0.001)

    // Phase 4 — Sparse attention over tool results (Tier 2)
    bool     sparse_attn_enabled; // Enable sparse tool attention (default: false)
    float    sparse_relevance_threshold; // Min relevance score (default: 0.1)

    // Phase 5 — Activation deduplication
    bool     act_dedup_enabled;   // Enable activation CSE (default: false)

    // Phase 5 — No-op detection
    bool     noop_enabled;        // Enable no-op elision (default: true)

    // Phase 5 — Algebraic rewriting
    bool     algebraic_enabled;   // Enable algebraic rewriting (default: false)

    // Statistics
    bool     stats_enabled;       // Print per-turn statistics (default: false)
};
```

Configuration is initialized from compile-time CMake flags, then overridden by environment variables at runtime.

### Statistics Tracking

```cpp
struct llama_opt_stats {
    // Phase 1 — Context block dedup
    uint64_t dedup_blocks_total;
    uint64_t dedup_blocks_hit;
    uint64_t dedup_tokens_saved;

    // Phase 1 — KV cache diffing
    uint64_t diff_tokens_total;
    uint64_t diff_tokens_unchanged;
    uint64_t diff_tokens_recomputed;

    // Phase 1 — Schema-aware skipping
    uint64_t schema_tokens_total;
    uint64_t schema_tokens_skipped;
    uint64_t schema_tokens_inferred;

    // Phase 2 — KV canonicalization
    uint64_t canon_entries_total;
    uint64_t canon_entries_deduped;
    uint64_t canon_bytes_saved;

    // Phase 2 — Precomputation
    uint64_t precompute_rope_hits;
    uint64_t precompute_norm_hits;
    uint64_t precompute_mask_hits;

    // Phase 2 — Persistent KV
    uint64_t persist_loads;
    uint64_t persist_saves;
    uint64_t persist_tokens_restored;

    // Phase 3 — Output memoization
    uint64_t memo_lookups;
    uint64_t memo_hits;
    uint64_t memo_tokens_skipped;

    // Phase 4 — Attention pruning
    uint64_t attn_blocks_pruned;
    uint64_t attn_tokens_pruned;

    // Phase 5 — Activation dedup
    uint64_t act_lookups;
    uint64_t act_hits;

    // Phase 5 — No-op elision
    uint64_t noop_ops_elided;

    void reset();
    void print() const;
};
```

### Compile-Time Guards

Every optimization wrapped in `#ifdef LLAMA_OPT_<NAME>` and additionally checked at runtime via `opt_config.<name>_enabled`. Fallback is always baseline behavior.

---

## 1.2 Context Block Hashing (`llama-context-hash`)

### Purpose

Deterministic, platform-independent hash function for fixed-size token blocks. Foundation for dedup, canonicalization, and diffing.

### Hash Function: FNV-1a (64-bit)

```cpp
uint64_t llama_opt_hash_block(const llama_token * tokens, uint32_t n_tokens);
uint64_t llama_opt_hash_block_pos(const llama_token * tokens, uint32_t n_tokens, llama_pos pos_start);

struct llama_opt_hasher {
    uint64_t state;
    llama_opt_hasher();
    void     feed(llama_token token);
    void     feed(const llama_token * tokens, uint32_t n);
    uint64_t finalize() const;
    void     reset();
};
```

### Block Segmentation

```
Context: [t0 t1 ... t63] [t64 t65 ... t127] [t128 ... t191] ...
          └─ block 0 ─┘   └─── block 1 ───┘  └── block 2 ──┘
```

- Default block size: 64 tokens (configurable)
- Final partial block hashed with actual size
- Position-independent hashes for dedup; position-dependent hashes available for diff

### Design Decisions

- FNV-1a chosen for zero dependencies, determinism, simplicity
- Token values hashed as raw 32-bit integers
- No SIMD — hashing cost negligible vs GEMM

---

## 1.3 Context Block Deduplication (`llama-kv-cache-dedup`)

### Purpose

Eliminate redundant KV projections for repeated context blocks. Coding agent workloads have 60-80% identical tokens across turns.

### Data Structures

```cpp
struct llama_opt_kv_block {
    uint64_t                 hash;
    uint32_t                 n_tokens;
    std::vector<llama_token> tokens;      // for collision verification
    uint32_t                 ref_count;
    uint64_t                 last_access;  // turn counter for LRU
};

class llama_opt_block_pool {
public:
    llama_opt_block_pool(uint32_t max_blocks);
    const llama_opt_kv_block * lookup(uint64_t hash, const llama_token * tokens, uint32_t n_tokens) const;
    llama_opt_kv_block * insert(uint64_t hash, const llama_token * tokens, uint32_t n_tokens);
    void ref(const llama_opt_kv_block * block);
    void unref(const llama_opt_kv_block * block);
    uint32_t evict_stale(uint64_t min_turn);
    uint32_t size() const;
    uint32_t capacity() const;
private:
    uint32_t max_blocks_;
    uint64_t turn_counter_;
    std::unordered_map<uint64_t, std::vector<llama_opt_kv_block>> blocks_;
};
```

### Integration Flow

1. Before prefill: segment context into blocks, hash each
2. For each block: lookup in pool
   - Hit: mark KV slots as DEDUP (point to canonical buffer)
   - Miss: compute normally, insert into pool
3. After prefill: update ref counts and timestamps

### KV Slot Status

```cpp
struct llama_opt_kv_slot_status {
    enum type { OWNED, DEDUP };
    type                       status;
    const llama_opt_kv_block * dedup_source;
};
```

### Correctness

- Token-level verification on every hit (not just hash)
- Hash collision → treat as miss
- Same position + same tokens → identical K/V (RoPE is deterministic)

### Limitations

- Same-position only (cross-position dedup deferred to Phase 2 with RoPE correction)

---

## 1.4 Structural KV Cache Diffing (`llama-kv-cache-diff`)

### Purpose

Avoid recomputing KV for unchanged tokens between turns, even with mid-context edits. Handles insertions, deletions, and replacements — not just shared prefixes.

### Diff Types

```cpp
enum llama_opt_diff_op { DIFF_KEEP, DIFF_INSERT, DIFF_DELETE, DIFF_REPLACE };

struct llama_opt_diff_span {
    llama_opt_diff_op op;
    uint32_t prev_start, curr_start, length;
};

using llama_opt_diff_result = std::vector<llama_opt_diff_span>;
```

### Algorithm

1. **Fast path** (O(1)): detect append-only (most common)
2. **Block-level LCS**: hash blocks from both contexts, find matching regions
3. **Token-level refinement**: refine block boundaries to exact token matches
4. **Output**: KEEP/INSERT/DELETE/REPLACE span list

Complexity: O(n) common case, O(n*m/B) worst case.

```cpp
class llama_opt_diff_engine {
public:
    llama_opt_diff_engine(uint32_t block_size);
    llama_opt_diff_result compute(
        const llama_token * prev, uint32_t n_prev,
        const llama_token * curr, uint32_t n_curr) const;
private:
    uint32_t block_size_;
    bool try_append_diff(...) const;
    llama_opt_diff_result block_diff(...) const;
    void refine_boundaries(...) const;
};
```

### RoPE Position Correction

When KEEP tokens shift position:

```cpp
void llama_opt_rope_correction(
    float * k_data, uint32_t n_dims,
    llama_pos old_pos, llama_pos new_pos,
    float rope_freq_base, float rope_freq_scale);
```

Exact element-wise rotation — no approximation.

### Context History

```cpp
class llama_opt_context_history {
public:
    void record(const llama_token * tokens, uint32_t n_tokens);
    const llama_token * prev_tokens() const;
    uint32_t prev_n_tokens() const;
    bool has_prev() const;
    void clear();
private:
    std::vector<llama_token> prev_, curr_;
};
```

---

## 1.5 Schema-Aware Token Skipping (`llama-schema-skip`)

### Purpose

Skip forward passes for tokens deterministically forced by output grammar. 30-50% of structured output tokens are predictable.

### API

```cpp
struct llama_opt_skip_result {
    bool        can_skip;
    llama_token forced_token;
};

llama_opt_skip_result llama_opt_schema_query_next(
    const llama_grammar * grammar,
    const llama_vocab   * vocab);

struct llama_opt_skip_sequence {
    std::vector<llama_token> tokens;
    uint32_t                 n_skip;
};

llama_opt_skip_sequence llama_opt_schema_query_sequence(
    const llama_grammar * grammar,
    const llama_vocab   * vocab,
    uint32_t              max_lookahead);
```

### Skip Condition

A token is forced when the grammar's current state allows exactly one valid continuation token. Since the grammar would force this token regardless of model output, skipping the forward pass is mathematically exact.

### Sampling Loop Integration

```
for each generation step:
    if grammar active:
        result = query_next(grammar, vocab)
        if result.can_skip:
            emit forced_token
            advance grammar
            accumulate for deferred KV batch fill
            continue
    // batch-fill any accumulated skipped tokens
    if pending_skipped > 0:
        llama_decode(batch_of_skipped_tokens)
    // normal forward pass
    llama_decode(current_token)
    sample + emit
```

### KV Cache for Skipped Tokens

Phase 1 uses **deferred batch fill**: accumulate skipped tokens, batch-compute their KV projections before the next real forward pass. Exact, simple.

### Forced Token Detection

Naive: scan all vocab tokens against grammar → O(vocab_size).
Optimization (Phase 2+): precompute forced-token table per grammar state → O(1).

---

# Phase 2: Caching Infrastructure

**Tier: Exact (Tier 1)**

## 2.1 KV Cache Canonicalization

### Purpose

Reduce KV cache memory by deduplicating identical K/V tensor entries across different sequence positions and layers.

### Files

```
src/llama-kv-cache-canon.h
src/llama-kv-cache-canon.cpp
```

### Mechanism

```cpp
struct llama_opt_kv_entry_key {
    uint64_t k_hash;  // Hash of the K tensor data
    uint64_t v_hash;  // Hash of the V tensor data
    uint32_t layer;   // Layer index
};

class llama_opt_kv_canon_store {
public:
    // Check if an identical KV entry already exists
    // Returns pointer to canonical entry, or nullptr
    const float * lookup_k(const llama_opt_kv_entry_key & key) const;
    const float * lookup_v(const llama_opt_kv_entry_key & key) const;

    // Insert a canonical KV entry
    void insert(const llama_opt_kv_entry_key & key,
                const float * k_data, uint32_t k_size,
                const float * v_data, uint32_t v_size);

    // Replace KV cache slot with reference to canonical
    void canonicalize_slot(uint32_t slot, const llama_opt_kv_entry_key & key);

    uint64_t bytes_saved() const;

private:
    // Hash → canonical tensor data
    std::unordered_map<uint64_t, std::vector<float>> k_store_;
    std::unordered_map<uint64_t, std::vector<float>> v_store_;
    // Slot → canonical key mapping
    std::unordered_map<uint32_t, llama_opt_kv_entry_key> slot_map_;
};
```

### Integration

After KV projections are computed for a batch:
1. Hash each K and V tensor per slot per layer
2. Check canonical store for duplicates
3. If duplicate found, replace slot's data pointer with canonical reference
4. If new, insert into canonical store

### Correctness

- Exact tensor hash comparison (FNV-1a over raw float bits)
- On hash collision, full byte comparison as fallback
- Attention computation reads from canonical tensors — identical results

### Expected Impact

- 1.2–1.5× RAM reduction when combined with context block dedup
- Near-zero compute overhead (hashing is cheap vs GEMM)

---

## 2.2 Deterministic Precomputation

### Purpose

Cache and reuse deterministic intermediate values that depend only on known parameters (positions, dimensions, model config).

### Files

```
src/llama-opt-precompute.h
src/llama-opt-precompute.cpp
```

### Targets

**RoPE rotation cache:**
```cpp
class llama_opt_rope_cache {
public:
    llama_opt_rope_cache(float freq_base, float freq_scale, uint32_t n_dims, uint32_t max_pos);

    // Get precomputed cos/sin for position
    const float * cos(llama_pos pos) const;
    const float * sin(llama_pos pos) const;

    // Precompute range of positions
    void precompute(llama_pos pos_start, llama_pos pos_end);

private:
    float freq_base_, freq_scale_;
    uint32_t n_dims_;
    std::unordered_map<llama_pos, std::vector<float>> cos_cache_;
    std::unordered_map<llama_pos, std::vector<float>> sin_cache_;
};
```

**Attention mask cache:**
```cpp
class llama_opt_mask_cache {
public:
    // Get or create causal mask for given dimensions
    const float * get_causal_mask(uint32_t n_tokens, uint32_t n_kv) const;

    // Precompute masks for expected batch sizes
    void precompute(uint32_t max_tokens, uint32_t max_kv);

private:
    mutable std::unordered_map<uint64_t, std::vector<float>> masks_;
};
```

**LayerNorm statistics cache:**
```cpp
class llama_opt_norm_cache {
public:
    // Check if we have cached statistics for this input
    bool lookup(uint64_t input_hash, float * mean, float * variance) const;
    void insert(uint64_t input_hash, float mean, float variance);

private:
    std::unordered_map<uint64_t, std::pair<float, float>> cache_;
};
```

### Integration

- RoPE cache: intercept `ggml_rope_ext` calls, serve from cache when position is precomputed
- Mask cache: intercept attention mask creation in graph builder
- Norm cache: intercept LayerNorm/RMSNorm computation, cache by input hash

### Expected Impact

- ~1.05–1.1× on long contexts (RoPE cache most impactful)
- Essentially free — precomputation cost amortized over session

---

## 2.3 Persistent Cross-Session KV Cache

### Purpose

Eliminate cold-start prefill by persisting KV cache to disk and reloading for recurring contexts.

### Files

```
src/llama-kv-cache-persist.h
src/llama-kv-cache-persist.cpp
```

### Serialization Format

```cpp
struct llama_opt_persist_header {
    char     magic[8];         // "LLAMAKV\0"
    uint32_t version;          // Format version
    uint64_t model_hash;       // Hash of model weights (for validation)
    uint64_t context_hash;     // Hash of the full token context
    uint32_t n_tokens;         // Number of tokens in cached context
    uint32_t n_layers;         // Number of layers
    uint32_t n_kv_heads;       // Number of KV heads
    uint32_t kv_dim;           // KV dimension per head
    ggml_type type_k;          // K tensor type
    ggml_type type_v;          // V tensor type
    uint64_t data_offset;      // Offset to tensor data
};
```

### API

```cpp
class llama_opt_kv_persist {
public:
    llama_opt_kv_persist(const std::string & dir);

    // Save current KV cache state to disk
    bool save(const llama_kv_cache & cache,
              const llama_token * tokens, uint32_t n_tokens,
              uint64_t model_hash);

    // Try to load a matching KV cache from disk
    // Returns number of tokens restored (0 if no match)
    uint32_t load(llama_kv_cache & cache,
                  const llama_token * tokens, uint32_t n_tokens,
                  uint64_t model_hash);

    // Load partial match (prefix match)
    uint32_t load_prefix(llama_kv_cache & cache,
                         const llama_token * tokens, uint32_t n_tokens,
                         uint64_t model_hash);

    // Clean up stale cache files
    void cleanup(uint32_t max_age_seconds);

private:
    std::string dir_;
    std::string cache_path(uint64_t context_hash, uint64_t model_hash) const;
};
```

### Session Flow

```
Session start:
    1. Hash current context tokens
    2. Check persist directory for matching cache file
    3a. Full match → load KV, skip prefill entirely
    3b. Prefix match → load prefix KV, compute only new tokens
    3c. No match → full prefill from scratch

Session end (or periodically):
    1. Serialize current KV cache to disk
    2. Tag with context hash + model hash
```

### Validation

- Model hash must match (different model weights → invalidate)
- Context hash must match (different tokens → invalidate)
- KV type/dimensions must match (different quantization → invalidate)
- File integrity check (truncation, corruption → invalidate)

### Expected Impact

- Near-elimination of cold-start prefill for recurring workloads
- Most impactful for large contexts (32k+ tokens)
- First session: no benefit. Second session onward: near-instant start.

---

# Phase 3: Output-Side Optimization

**Tier: Exact (Tier 1)**

## 3.1 Token-Level Output Memoization

### Purpose

Detect when the model is entering a previously generated token sequence and fast-forward through confirmed tokens.

### Files

```
src/llama-opt-memo.h
src/llama-opt-memo.cpp
```

### Data Structure: Memoization Trie

```cpp
struct llama_opt_memo_node {
    llama_token                                       token;
    uint64_t                                          hidden_state_hash; // Hash at this decision point
    float                                             confidence;        // argmax probability when originally generated
    std::unordered_map<llama_token, llama_opt_memo_node *> children;
};

class llama_opt_memo_trie {
public:
    llama_opt_memo_trie(uint32_t max_nodes);

    // Record a generated sequence for future memoization
    void record(const llama_token * tokens, uint32_t n_tokens,
                const uint64_t * hidden_hashes, uint32_t n_hashes);

    // Check if current hidden state matches a known entry point
    struct lookup_result {
        bool        found;
        llama_token next_token;
        uint32_t    sequence_length; // How many tokens can be fast-forwarded
    };
    lookup_result lookup(uint64_t hidden_state_hash) const;

    // Advance along a memoized path
    // Returns false if the path diverges (hidden state doesn't match)
    bool advance(uint64_t hidden_state_hash, llama_token & next_token);

    // Reset cursor to root
    void reset_cursor();

    uint32_t size() const;

private:
    uint32_t max_nodes_;
    llama_opt_memo_node root_;
    llama_opt_memo_node * cursor_;
};
```

### Integration

```
for each generation step:
    compute hidden_state_hash from final hidden layer
    result = trie.lookup(hidden_state_hash)
    if result.found:
        // Verify: run forward pass anyway, check if argmax matches
        logits = llama_decode(...)
        actual_token = argmax(logits)
        if actual_token == result.next_token:
            fast-forward through confirmed sequence
            stats.memo_tokens_skipped += sequence_length
        else:
            // Divergence — exit memoization, use actual_token
    else:
        // Normal generation, record for future memo
```

### Verification Strategy

For Tier 1 exactness, every memoized token must be verified:
- Compute the forward pass
- Check that argmax matches the memoized token
- If match, skip decoding overhead for subsequent tokens in the sequence
- If diverge, exit memoization immediately

The savings come from batching: if we know the next N tokens, we can compute N forward passes in a single batch, which is faster than N sequential passes.

### Expected Impact

- 1.05–1.15× generation speedup for repetitive output patterns
- Most valuable in long sessions with boilerplate (imports, error handling, tool call structures)

---

# Phase 4: Approximate Optimizations (Tier 2)

**All Tier 2 optimizations are OFF by default and require explicit opt-in.**

## 4.1 Attention Sink Pruning

### Purpose

Reduce attention compute by evicting context blocks that no longer receive meaningful attention weight.

### Files

```
src/llama-opt-attn-prune.h
src/llama-opt-attn-prune.cpp
```

### Mechanism

```cpp
struct llama_opt_attn_block_stats {
    uint64_t block_id;
    uint32_t block_start;     // Start token position
    uint32_t block_length;    // Number of tokens
    float    avg_attn_weight;  // Rolling average attention weight
    uint32_t consecutive_dead; // Turns below threshold
    bool     evicted;
};

class llama_opt_attn_pruner {
public:
    llama_opt_attn_pruner(float threshold, uint32_t dead_turns_limit);

    // Update attention statistics after each forward pass
    void update(const float * attn_weights, uint32_t n_tokens, uint32_t n_kv);

    // Get list of blocks to evict
    std::vector<uint32_t> get_eviction_candidates() const;

    // Move block to cold storage (retained for potential retrieval)
    void evict(uint32_t block_id, llama_kv_cache & cache);

    // Restore a block from cold storage if needed
    bool restore(uint32_t block_id, llama_kv_cache & cache);

private:
    float threshold_;
    uint32_t dead_turns_limit_;
    std::vector<llama_opt_attn_block_stats> block_stats_;
    // Cold storage for evicted blocks
    std::unordered_map<uint32_t, std::vector<float>> cold_k_;
    std::unordered_map<uint32_t, std::vector<float>> cold_v_;
};
```

### Integration

After each attention computation:
1. Extract attention weights per context block
2. Update rolling average
3. If a block has been below threshold for N consecutive turns, evict
4. Evicted blocks move to cold storage (CPU RAM, not in attention computation)
5. If attention pattern shifts, restore from cold storage

### Correctness

- **This is an approximation** — outputs may differ from baseline
- The assumption: context blocks with negligible attention weight contribute negligibly to outputs
- Cold storage retrieval adds latency if the model unexpectedly needs evicted context
- Guardrail: never evict system prompt or recent turns

### Expected Impact

- 1.1–1.3× attention speedup in long sessions (50+ turns)
- Scales with session length — more stale context = more pruning opportunity

---

## 4.2 Sparse Attention over Tool Results

### Purpose

Apply full attention only to relevant spans within large tool results (file dumps, terminal output, search results).

### Files

```
src/llama-opt-sparse-attn.h
src/llama-opt-sparse-attn.cpp
```

### Mechanism

```cpp
struct llama_opt_relevance_span {
    uint32_t start;    // Start token position within tool result
    uint32_t length;   // Number of tokens
    float    score;    // Relevance score (0.0 to 1.0)
};

class llama_opt_sparse_scorer {
public:
    llama_opt_sparse_scorer(float threshold);

    // Score relevance of spans within a tool result
    // Uses token overlap with the current query
    std::vector<llama_opt_relevance_span> score(
        const llama_token * tool_result, uint32_t n_tool_tokens,
        const llama_token * query, uint32_t n_query_tokens) const;

    // Create a sparse attention mask based on relevance
    void create_mask(
        float * mask, uint32_t n_tokens, uint32_t n_kv,
        const std::vector<llama_opt_relevance_span> & spans,
        uint32_t tool_result_start, uint32_t tool_result_end) const;

private:
    float threshold_;
};
```

### Relevance Detection

Lightweight relevance scoring (no neural network):
1. Token overlap: count shared tokens between query and each span of the tool result
2. Position recency: spans closer to the end of the tool result score higher
3. Structural heuristic: spans starting with keywords, function names, error messages score higher

### Integration

Before attention computation:
1. Identify tool result spans in the context
2. Score each span's relevance to the current generation context
3. Create a modified attention mask: full attention for relevant spans, zero/reduced for irrelevant
4. Apply modified mask during attention

### Correctness

- **This is an approximation** — relevance detection is imperfect
- If important content is scored as irrelevant, it will be missed
- Guardrail: always include first and last N tokens of any tool result (likely to contain summary/error info)

### Expected Impact

- Meaningful for very large tool results (1000+ tokens)
- Reduces attention compute proportional to the irrelevant fraction
- Needs prototyping to validate relevance scoring quality

---

# Phase 5: Low-Priority Exact Optimizations

**Tier: Exact (Tier 1)**

## 5.1 Activation Deduplication (CSE)

### Purpose

Eliminate redundant computation of identical intermediate tensors using common subexpression elimination (CSE).

### Files

```
src/llama-opt-act-dedup.h
src/llama-opt-act-dedup.cpp
```

### Mechanism

```cpp
struct llama_opt_op_key {
    int      op_type;      // GGML operation type
    uint32_t layer_id;     // Layer index
    uint64_t input_hash;   // Hash of input tensor data (tile-level)
};

class llama_opt_act_cache {
public:
    llama_opt_act_cache(uint32_t max_entries);

    // Check if output for this operation is cached
    bool lookup(const llama_opt_op_key & key, ggml_tensor * output) const;

    // Cache an operation's output
    void insert(const llama_opt_op_key & key, const ggml_tensor * output);

    // Clear between batches (intermediate tensors are batch-specific)
    void clear();

private:
    uint32_t max_entries_;
    std::unordered_map<uint64_t, std::vector<float>> cache_;
};
```

### Target Operations

- `ggml_mul_mat` — Matrix multiplication (most expensive)
- `ggml_add` — Residual connections
- `ggml_mul` — Element-wise multiplication
- `ggml_rms_norm` — RMS normalization

### Tile-Level Hashing

To avoid hashing entire large tensors, use tile-level sampling:
1. Divide tensor into tiles (e.g., 64×64 for matrices)
2. Hash a sample of tiles (e.g., 4 corners + center)
3. Use combined hash as the cache key
4. On hit, verify with full comparison before reuse

### Honest Assessment

In practice, exact bitwise duplication of intermediate tensors is rare during normal inference. The hashing overhead may consume most of the savings. This optimization is speculative and needs profiling before shipping.

### Expected Impact

- ~1.0–1.1× for most workloads
- May have niche value in batch processing with identical sequences
- Profile carefully — overhead may exceed benefit

---

## 5.2 No-Op Detection and Elision

### Purpose

Skip computations mathematically guaranteed to produce no-effect results.

### Files

```
src/llama-opt-noop.h
src/llama-opt-noop.cpp
```

### Detectable No-Ops

```cpp
enum llama_opt_noop_type {
    NOOP_ZERO_GEMM,       // 0 * W = 0
    NOOP_IDENTITY_ADD,    // x + 0 = x
    NOOP_IDENTITY_MUL,    // x * 1 = x
    NOOP_ONEHOT_SOFTMAX,  // softmax reduces to one-hot
    NOOP_ZERO_RESIDUAL,   // residual branch produces zero
};

class llama_opt_noop_detector {
public:
    // Check if an operation is a no-op given its inputs
    bool is_noop(const ggml_tensor * op, llama_opt_noop_type & type) const;

    // Get the result without computing (identity, zero, etc.)
    void get_noop_result(const ggml_tensor * op, llama_opt_noop_type type,
                         ggml_tensor * result) const;

private:
    // Fast checks for common patterns
    bool is_zero_tensor(const ggml_tensor * t) const;
    bool is_identity_tensor(const ggml_tensor * t) const;
    bool is_onehot_softmax(const ggml_tensor * t) const;
};
```

### Integration

Intercept `ggml_compute_forward_*` calls:
1. Before computing, check if inputs trigger a no-op condition
2. If no-op detected, write the result directly (zero, identity copy, etc.)
3. Skip the actual computation

### Honest Assessment

These conditions almost never occur during normal inference:
- Tensors are rarely exactly zero (floating-point noise)
- Softmax rarely produces exact one-hot (except with extreme temperature)
- The detection checks themselves have a cost

### Expected Impact

- ~1.0–1.05× for typical workloads
- Detection overhead is small but so are the savings

---

## 5.3 Exact Algebraic Rewriting

### Purpose

Reduce redundant matrix multiplications by exploiting algebraic structure in the transformer computation.

### Files

```
src/llama-opt-algebraic.h
src/llama-opt-algebraic.cpp
```

### Techniques

**QKV Factorization:**
```
Instead of: Q = x @ Wq, K = x @ Wk, V = x @ Wv  (3 matmuls)
If Wq, Wk, Wv share structure: QKV = x @ W_qkv  (1 matmul, already fused in most models)
```

**Attention Head Equivalence:**
```cpp
class llama_opt_head_analyzer {
public:
    // Analyze weight matrices to find equivalent heads
    struct head_group {
        std::vector<uint32_t> head_indices;
        // Transform to derive group members from the first head
        // e.g., head_3 = permute(head_0, perm_matrix)
        ggml_tensor * transform;
    };

    std::vector<head_group> analyze(const llama_model & model, uint32_t layer) const;
};
```

If two attention heads produce equivalent results (up to a known transform), compute once and derive the second.

**Shared Projection Factorization:**
```cpp
// If attention weights across blocks share a common factor:
// W_block_i = U * diag(sigma_i) * V^T
// Then: x @ W_block_i = (x @ U) * sigma_i @ V^T
// The (x @ U) part is shared across blocks
class llama_opt_shared_projection {
public:
    bool can_factor(const ggml_tensor * w1, const ggml_tensor * w2,
                    float tolerance) const;
    void factor(const ggml_tensor * weights, uint32_t n_blocks,
                ggml_tensor * shared, ggml_tensor * per_block) const;
};
```

### Honest Assessment

- Most models already fuse QKV into a single matmul
- Head equivalence is model-architecture-dependent and may not exist in practice
- Shared projection factorization requires SVD analysis at model load time
- This needs concrete profiling per model architecture before confident estimates

### Expected Impact

- Architecture-dependent, needs investigation
- Potentially 1.0–1.2× for models with exploitable structure
- Likely marginal for well-optimized modern architectures

---

# Phase 6: Stabilization

No new features. Focus on:

### 6.1 Profiling and Overhead Analysis

- Per-optimization overhead measurement (hashing cost, cache lookup latency, diff computation)
- Net impact analysis (benefit minus overhead) per workload type
- Memory overhead of all caches and indexes
- Automated regression detection in CI

### 6.2 Documentation

- Architecture overview with diagrams
- Per-optimization deep-dive documentation
- Configuration guide (which optimizations for which workloads)
- Benchmark methodology and reproducibility guide

### 6.3 Upstream Alignment

- Rebase on latest upstream llama.cpp
- Resolve merge conflicts in modified upstream files
- Verify all optimizations still work after rebase
- Minimize diff against upstream for maintainability

### 6.4 Quality Assurance

- Full test suite passing for all optimization combinations
- Fuzz testing for hash functions and diff algorithms
- Long-running stress tests (1000+ turn sessions)
- Memory leak detection
- Thread safety verification for all caches

---

# Complete File Manifest

| Phase | File | Type | Description |
|---|---|---|---|
| 1 | `src/llama-opt.h` | Header | Config, stats, global init |
| 1 | `src/llama-opt.cpp` | Source | Config loading, env var parsing, stats printing |
| 1 | `src/llama-context-hash.h` | Header | FNV-1a hash, block hasher |
| 1 | `src/llama-context-hash.cpp` | Source | Hash implementation |
| 1 | `src/llama-kv-cache-dedup.h` | Header | Block pool, KV slot status |
| 1 | `src/llama-kv-cache-dedup.cpp` | Source | Dedup implementation |
| 1 | `src/llama-kv-cache-diff.h` | Header | Diff engine, context history, RoPE correction |
| 1 | `src/llama-kv-cache-diff.cpp` | Source | Diff implementation |
| 1 | `src/llama-schema-skip.h` | Header | Schema query, forced token detection |
| 1 | `src/llama-schema-skip.cpp` | Source | Schema-aware skipping |
| 2 | `src/llama-kv-cache-canon.h` | Header | KV entry canonicalization |
| 2 | `src/llama-kv-cache-canon.cpp` | Source | Canonicalization implementation |
| 2 | `src/llama-opt-precompute.h` | Header | RoPE cache, mask cache, norm cache |
| 2 | `src/llama-opt-precompute.cpp` | Source | Precomputation implementation |
| 2 | `src/llama-kv-cache-persist.h` | Header | Persistent KV serialization |
| 2 | `src/llama-kv-cache-persist.cpp` | Source | Persistent KV implementation |
| 3 | `src/llama-opt-memo.h` | Header | Memoization trie |
| 3 | `src/llama-opt-memo.cpp` | Source | Output memoization implementation |
| 4 | `src/llama-opt-attn-prune.h` | Header | Attention sink pruning |
| 4 | `src/llama-opt-attn-prune.cpp` | Source | Pruning implementation |
| 4 | `src/llama-opt-sparse-attn.h` | Header | Sparse attention scoring |
| 4 | `src/llama-opt-sparse-attn.cpp` | Source | Sparse attention implementation |
| 5 | `src/llama-opt-act-dedup.h` | Header | Activation CSE |
| 5 | `src/llama-opt-act-dedup.cpp` | Source | Activation dedup implementation |
| 5 | `src/llama-opt-noop.h` | Header | No-op detection |
| 5 | `src/llama-opt-noop.cpp` | Source | No-op elision implementation |
| 5 | `src/llama-opt-algebraic.h` | Header | Algebraic rewriting |
| 5 | `src/llama-opt-algebraic.cpp` | Source | Algebraic rewriting implementation |
| 1-5 | `tests/test-opt-context-hash.cpp` | Test | Hash function tests |
| 1 | `tests/test-opt-kv-dedup.cpp` | Test | Dedup pool tests |
| 1 | `tests/test-opt-kv-diff.cpp` | Test | Diff algorithm tests |
| 1 | `tests/test-opt-schema-skip.cpp` | Test | Schema skipping tests |
| 2 | `tests/test-opt-kv-canon.cpp` | Test | Canonicalization tests |
| 2 | `tests/test-opt-precompute.cpp` | Test | Precomputation tests |
| 2 | `tests/test-opt-kv-persist.cpp` | Test | Persistent KV tests |
| 3 | `tests/test-opt-memo.cpp` | Test | Memoization trie tests |
| 4 | `tests/test-opt-attn-prune.cpp` | Test | Attention pruning tests |

---

# CMake Feature Flags

```cmake
# Phase 1 — Tier 1 (Exact), ON by default
option(LLAMA_OPT_DEDUP         "llama: context block deduplication"        ON)
option(LLAMA_OPT_KV_DIFF       "llama: structural KV cache diffing"        ON)
option(LLAMA_OPT_SCHEMA_SKIP   "llama: schema-aware token skipping"        ON)

# Phase 2 — Tier 1 (Exact), ON by default
option(LLAMA_OPT_KV_CANON      "llama: KV cache canonicalization"          ON)
option(LLAMA_OPT_PRECOMPUTE    "llama: deterministic precomputation"       ON)
option(LLAMA_OPT_KV_PERSIST    "llama: persistent cross-session KV cache"  OFF)

# Phase 3 — Tier 1 (Exact), ON by default
option(LLAMA_OPT_MEMO          "llama: token-level output memoization"     ON)

# Phase 4 — Tier 2 (Approximate), OFF by default
option(LLAMA_OPT_ATTN_PRUNE    "llama: attention sink pruning (Tier 2)"    OFF)
option(LLAMA_OPT_SPARSE_ATTN   "llama: sparse tool attention (Tier 2)"    OFF)

# Phase 5 — Tier 1 (Exact), mixed defaults
option(LLAMA_OPT_ACT_DEDUP     "llama: activation deduplication (CSE)"     OFF)
option(LLAMA_OPT_NOOP          "llama: no-op detection and elision"        ON)
option(LLAMA_OPT_ALGEBRAIC     "llama: exact algebraic rewriting"          OFF)
```

---

# Runtime Configuration

All environment variables prefixed with `LLAMA_OPT_`:

| Variable | Type | Default | Phase | Description |
|---|---|---|---|---|
| `LLAMA_OPT_BLOCK_SIZE` | int | 64 | 1 | Token block size for hashing |
| `LLAMA_OPT_DEDUP_ENABLED` | bool | 1 | 1 | Enable/disable context block dedup |
| `LLAMA_OPT_DEDUP_POOL_MAX` | int | 16384 | 1 | Max blocks in dedup pool |
| `LLAMA_OPT_DIFF_ENABLED` | bool | 1 | 1 | Enable/disable KV cache diffing |
| `LLAMA_OPT_DIFF_MIN_UNCHANGED` | int | 8 | 1 | Min span size to skip recompute |
| `LLAMA_OPT_SCHEMA_SKIP_ENABLED` | bool | 1 | 1 | Enable/disable schema-aware skipping |
| `LLAMA_OPT_CANON_ENABLED` | bool | 1 | 2 | Enable/disable KV canonicalization |
| `LLAMA_OPT_PRECOMPUTE_ENABLED` | bool | 1 | 2 | Enable/disable precomputation |
| `LLAMA_OPT_PERSIST_ENABLED` | bool | 0 | 2 | Enable/disable persistent KV |
| `LLAMA_OPT_PERSIST_DIR` | string | "" | 2 | Directory for KV cache files |
| `LLAMA_OPT_MEMO_ENABLED` | bool | 1 | 3 | Enable/disable output memoization |
| `LLAMA_OPT_ATTN_PRUNE_ENABLED` | bool | 0 | 4 | Enable/disable attention pruning |
| `LLAMA_OPT_ATTN_PRUNE_WINDOW` | int | 10 | 4 | Turns before eviction |
| `LLAMA_OPT_ATTN_PRUNE_THRESHOLD` | float | 0.001 | 4 | Attention weight threshold |
| `LLAMA_OPT_SPARSE_ATTN_ENABLED` | bool | 0 | 4 | Enable/disable sparse attention |
| `LLAMA_OPT_ACT_DEDUP_ENABLED` | bool | 0 | 5 | Enable/disable activation CSE |
| `LLAMA_OPT_NOOP_ENABLED` | bool | 1 | 5 | Enable/disable no-op elision |
| `LLAMA_OPT_ALGEBRAIC_ENABLED` | bool | 0 | 5 | Enable/disable algebraic rewriting |
| `LLAMA_OPT_STATS` | bool | 0 | all | Print per-turn optimization statistics |
