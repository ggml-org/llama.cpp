// PHANTOM-X speculative decoding for llama.cpp
//
// Single-file implementation of PHANTOM-X hybrid speculative decoding:
//   1. Bloom filter negative bigram tracking
//   2. Dynamic γ (draft length adaptation)
//   3. Auto-fallback with hysteresis
//   4. Scan+patch pipeline (truncate at bad bigrams)
//   5. GhostBuffer — pinned memory ring for zero-copy HSA transport
//
// Wraps ngram_mod internally and adds all 5 components on top.
// Ported from carlosfundora/sglang-1-bit-turbo phantom_worker.py
//
// Usage: --spec-type phantom [--phantom-buffers 2] [--phantom-bloom-bits 16384]

#pragma once

#include "llama.h"
#include "common.h"
#include "ngram-mod.h"
#include "log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#endif

// ============================================================================
// Component 1: Bloom filter for negative bigrams
//
// Tracks (prev_token, rejected_token) pairs. Aged periodically to avoid
// permanent blacklisting. FNV-1a hash with configurable number of probes.
// ============================================================================

struct phantom_bloom {
    std::vector<uint64_t> bits;
    size_t n_bits;
    int    n_hashes;
    size_t n_inserts = 0;

    static constexpr size_t AGE_INTERVAL = 256;

    phantom_bloom(size_t n_bits_ = 16384, int n_hashes_ = 3)
        : n_bits(n_bits_), n_hashes(n_hashes_) {
        bits.resize((n_bits + 63) / 64, 0);
    }

    void reset() {
        std::fill(bits.begin(), bits.end(), 0);
        n_inserts = 0;
    }

    uint64_t hash(llama_token a, llama_token b, int seed) const {
        uint64_t h = 14695981039346656037ULL ^ (uint64_t)seed;
        h ^= (uint64_t)(uint32_t)a; h *= 1099511628211ULL;
        h ^= (uint64_t)(uint32_t)b; h *= 1099511628211ULL;
        return h % n_bits;
    }

    void insert(llama_token a, llama_token b) {
        for (int i = 0; i < n_hashes; i++) {
            uint64_t idx = hash(a, b, i);
            bits[idx / 64] |= (1ULL << (idx % 64));
        }
        if (++n_inserts % AGE_INTERVAL == 0) {
            age();
        }
    }

    bool query(llama_token a, llama_token b) const {
        for (int i = 0; i < n_hashes; i++) {
            uint64_t idx = hash(a, b, i);
            if (!(bits[idx / 64] & (1ULL << (idx % 64)))) {
                return false;
            }
        }
        return true;
    }

    // Clear one quarter of the bit array to prevent saturation
    void age() {
        size_t quarter = bits.size() / 4;
        if (quarter == 0) quarter = 1;
        size_t offset = ((n_inserts / AGE_INTERVAL) % 4) * quarter;
        size_t end = std::min(offset + quarter, bits.size());
        for (size_t i = offset; i < end; i++) {
            bits[i] = 0;
        }
    }

    float occupancy() const {
        size_t set = 0;
        for (auto w : bits) {
            set += __builtin_popcountll(w);
        }
        return (float)set / (float)n_bits;
    }
};

// ============================================================================
// Component 5: GhostBuffer — pinned memory ring for draft tokens
//
// Allocates page-locked memory for zero-copy CPU→GPU transport on ROCm HSA.
// On HSA, the GPU reads pinned system memory directly via PCIe — no memcpy.
// Falls back gracefully to unpinned memory if mlock() or GPU registration fails.
// ============================================================================

struct phantom_ghost_buffer {
    struct slot {
        llama_token * tokens   = nullptr;
        size_t        n_tokens = 0;
        size_t        capacity = 0;
    };

    std::vector<slot> slots;
    size_t write_idx  = 0;
    bool   is_pinned  = false;
    bool   is_gpu_reg = false;
    size_t alloc_size = 0;

    phantom_ghost_buffer() = default;

    // non-copyable due to pinned memory ownership
    phantom_ghost_buffer(const phantom_ghost_buffer &) = delete;
    phantom_ghost_buffer & operator=(const phantom_ghost_buffer &) = delete;

    bool init(int n_buffers, size_t max_tokens) {
        if (n_buffers <= 0 || max_tokens == 0) {
            return false;
        }

        // size must be multiple of alignment for posix_memalign
        alloc_size = ((max_tokens * sizeof(llama_token) + 63) / 64) * 64;

        slots.resize(n_buffers);
        for (auto & s : slots) {
            s.capacity = max_tokens;
            s.n_tokens = 0;
            s.tokens   = nullptr;

#if defined(__linux__)
            void * ptr = nullptr;
            if (posix_memalign(&ptr, 64, alloc_size) == 0) {
                if (mlock(ptr, alloc_size) == 0) {
                    is_pinned = true;
                }
                s.tokens = (llama_token *)ptr;
            }
#else
            s.tokens = (llama_token *)malloc(alloc_size);
#endif
            if (!s.tokens) {
                cleanup();
                return false;
            }
            memset(s.tokens, 0, alloc_size);

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
            if (ggml_backend_cuda_register_host_buffer(s.tokens, alloc_size)) {
                is_gpu_reg = true;
            }
#endif
        }

        LOG_INF("phantom: ghost buffer %d×%zu slots, pinned=%s, gpu_reg=%s\n",
                n_buffers, max_tokens,
                is_pinned  ? "yes" : "no",
                is_gpu_reg ? "yes" : "no");
        return true;
    }

    void write(const llama_token * src, size_t n) {
        if (slots.empty()) return;
        auto & s = slots[write_idx];
        size_t copy_n = std::min(n, s.capacity);
        memcpy(s.tokens, src, copy_n * sizeof(llama_token));
        s.n_tokens = copy_n;
    }

    llama_token * write_ptr(size_t * cap_out) {
        if (slots.empty()) return nullptr;
        auto & s = slots[write_idx];
        if (cap_out) *cap_out = s.capacity;
        return s.tokens;
    }

    const llama_token * read(size_t * n_out) const {
        if (slots.empty()) return nullptr;
        const auto & s = slots[write_idx];
        if (n_out) *n_out = s.n_tokens;
        return s.tokens;
    }

    void advance() {
        if (!slots.empty()) {
            write_idx = (write_idx + 1) % slots.size();
        }
    }

    void cleanup() {
        for (auto & s : slots) {
            if (s.tokens) {
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
                ggml_backend_cuda_unregister_host_buffer(s.tokens);
#endif
#if defined(__linux__)
                munlock(s.tokens, alloc_size);
#endif
                free(s.tokens);
                s.tokens = nullptr;
            }
        }
        slots.clear();
        is_pinned  = false;
        is_gpu_reg = false;
    }

    ~phantom_ghost_buffer() { cleanup(); }
};

// ============================================================================
// PHANTOM-X speculative decode state
//
// Extends ngram_mod with bloom filtering, dynamic γ, auto-fallback,
// scan+patch, and ghost buffer pinned transport.
// ============================================================================

struct common_speculative_state_phantom : public common_speculative_state {
    common_ngram_mod & mod; // shared ngram_mod instance (from params)

    // --- Component 1: Bloom filter ---
    phantom_bloom bloom;

    // --- Component 2: Dynamic γ ---
    float gamma_ema   = 0.5f;
    float gamma_alpha = 0.15f;
    int   gamma_n_cur = 8;

    // --- Component 3: Auto-fallback + hysteresis ---
    int fallback_streak   = 0;
    int fallback_cooldown = 0;

    static constexpr int   FALLBACK_THRESHOLD = 4;
    static constexpr int   COOLDOWN_ROUNDS    = 8;
    static constexpr float ACC_LOW_THRESH     = 0.2f;

    // --- Component 4: Scan+patch tracking ---
    llama_tokens last_draft;
    llama_token  last_id_before  = 0;
    bool         awaiting_accept = false;

    // --- Component 5: Ghost buffer ---
    phantom_ghost_buffer ghost_buf;

    // --- ngram index ---
    size_t i_last = 0;

    // --- Stats ---
    size_t n_bloom_filtered = 0;
    size_t n_fallback_skips = 0;
    size_t n_gamma_adjusts  = 0;

    const bool verbose;

    common_speculative_state_phantom(
            enum common_speculative_type type,
            common_ngram_mod & mod_,
            int32_t bloom_bits    = 16384,
            int32_t ghost_buffers = 2,
            int32_t ghost_cap     = 64)
        : common_speculative_state(type)
        , mod(mod_)
        , bloom((size_t)bloom_bits)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr)
    {
        if (ghost_buffers > 0) {
            ghost_buf.init(ghost_buffers, (size_t)ghost_cap);
        }
    }

    ~common_speculative_state_phantom() override = default;

    // ----------------------------------------------------------------
    // begin() — seed ngram table from prompt
    // ----------------------------------------------------------------
    void begin(const llama_tokens & prompt) override {
        i_last = 0;
        last_draft.clear();
        awaiting_accept = false;

        const size_t n = mod.get_n();
        if (prompt.size() < n) return;

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }
        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: phantom ngram=%zu/%zu (%.2f), bloom=%.2f\n",
                __func__, mod.get_used(), mod.size(), f, bloom.occupancy());

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram occupancy %.2f > %.2f — resetting\n", __func__, f, f_thold);
            mod.reset();
        }
    }

    // ----------------------------------------------------------------
    // draft() — generate draft with all 5 phantom components
    //
    // Uses rolling hash for O(1) per-step n-gram lookup and fused
    // bloom checking (no second scan pass).
    // ----------------------------------------------------------------
    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        // Handle missed accept(0): framework swallows accept(0) at speculative.cpp:1254
        if (awaiting_accept && !last_draft.empty()) {
            handle_implicit_reject();
        }

        // Component 3: Auto-fallback — skip speculation during cooldown
        if (fallback_cooldown > 0) {
            fallback_cooldown--;
            n_fallback_skips++;
            if (verbose) {
                LOG_INF("%s: cooldown (%d remaining)\n", __func__, fallback_cooldown);
            }
            result.clear();
            return;
        }

        const size_t cur_len = prompt_tgt.size();
        const size_t n = mod.get_n();
        if (cur_len < n) {
            result.clear();
            return;
        }

        // Incrementally update ngram table
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }
            i_last = cur_len - n;
        }

        // Component 2: Use adaptive draft length clamped to params
        const int n_draft = std::clamp(gamma_n_cur, params.n_min, params.n_max);

        // Build ngram lookup context: last (n-1) prompt tokens + id_last
        std::vector<llama_token> ctx(n + n_draft);
        for (size_t i = 0; i < n - 1; ++i) {
            ctx[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        ctx[n - 1] = id_last;

        // Fused draft generation with rolling hash + inline bloom check.
        // First window: ctx[0..n-1] — full hash
        uint64_t h = mod.hash_full(ctx.data());
        llama_token prev_tok = id_last;
        int drafted = 0;

        for (int i = 0; i < n_draft; ++i) {
            const llama_token tok = mod.get_by_hash(h);
            if (tok == common_ngram_mod::EMPTY) {
                break;
            }

            // Component 4: Fused bloom check — test bigram before accepting
            if (bloom.query(prev_tok, tok)) {
                n_bloom_filtered++;
                if (verbose) {
                    LOG_INF("%s: bloom blocked (%d→%d) at pos %d\n",
                            __func__, prev_tok, tok, i);
                }
                break;
            }

            ctx[n + i] = tok;
            prev_tok = tok;
            drafted++;

            // Rolling hash for next window (if we'll continue)
            if (i + 1 < n_draft) {
                h = mod.hash_roll(h, ctx[i], tok);
            }
        }

        if (drafted == 0 || drafted < params.n_min) {
            result.clear();
            clear_pending();
            return;
        }

        // Extract drafted tokens
        result.assign(ctx.begin() + (ptrdiff_t)n, ctx.begin() + (ptrdiff_t)(n + drafted));

        // Component 5: Write to ghost buffer (pinned memory)
        if (!ghost_buf.slots.empty()) {
            ghost_buf.write(result.data(), result.size());
            ghost_buf.advance();
        }

        // Save for accept() correlation
        last_draft     = result;
        last_id_before = id_last;
        awaiting_accept = true;
    }

    // ----------------------------------------------------------------
    // accept() — update bloom, γ, fallback from acceptance signal
    // ----------------------------------------------------------------
    void accept(uint16_t n_accepted) override {
        awaiting_accept = false;

        if (last_draft.empty()) return;

        const size_t n_drafted = last_draft.size();
        const float f_acc = (n_drafted > 0)
            ? (float)n_accepted / (float)n_drafted
            : 0.0f;

        if (verbose) {
            LOG_INF("%s: %d/%zu (%.0f%%) γ_ema=%.2f γ_n=%d streak=%d\n",
                    __func__, n_accepted, n_drafted, f_acc * 100.0f,
                    gamma_ema, gamma_n_cur, fallback_streak);
        }

        // Component 1: Learn from FIRST failing transition only
        // Tokens after n_accepted were never tested, not "rejected"
        if (n_accepted < (uint16_t)n_drafted) {
            llama_token prev = (n_accepted == 0)
                ? last_id_before
                : last_draft[n_accepted - 1];
            llama_token bad = last_draft[n_accepted];
            bloom.insert(prev, bad);
        }

        // Component 2: Update dynamic γ EMA
        gamma_ema = gamma_alpha * f_acc + (1.0f - gamma_alpha) * gamma_ema;

        if (gamma_ema > 0.7f && gamma_n_cur < 24) {
            gamma_n_cur++;
            n_gamma_adjusts++;
        } else if (gamma_ema < 0.3f && gamma_n_cur > 2) {
            gamma_n_cur--;
            n_gamma_adjusts++;
        }

        // Component 3: Auto-fallback with hysteresis
        if (f_acc < ACC_LOW_THRESH) {
            fallback_streak++;
            if (fallback_streak >= FALLBACK_THRESHOLD) {
                fallback_cooldown = COOLDOWN_ROUNDS;
                fallback_streak   = 0;
                mod.reset();
                LOG_WRN("%s: fallback cooldown (%d rounds), resetting ngram\n",
                        __func__, COOLDOWN_ROUNDS);
            }
        } else {
            fallback_streak = 0;
        }

        last_draft.clear();
    }

private:
    // Accept(0) is swallowed by the framework. If draft() is called again
    // while awaiting_accept is true, the previous draft was fully rejected.
    void handle_implicit_reject() {
        if (verbose) {
            LOG_INF("%s: implicit reject (accept(0) swallowed)\n", __func__);
        }

        // Learn first failing bigram
        if (!last_draft.empty()) {
            bloom.insert(last_id_before, last_draft[0]);
        }

        // γ update with 0 acceptance
        gamma_ema = (1.0f - gamma_alpha) * gamma_ema;
        if (gamma_n_cur > 2) {
            gamma_n_cur--;
            n_gamma_adjusts++;
        }

        // Fallback streak
        fallback_streak++;
        if (fallback_streak >= FALLBACK_THRESHOLD) {
            fallback_cooldown = COOLDOWN_ROUNDS;
            fallback_streak   = 0;
            mod.reset();
            LOG_WRN("%s: fallback after implicit reject\n", __func__);
        }

        clear_pending();
    }

    void clear_pending() {
        last_draft.clear();
        awaiting_accept = false;
    }
};
