
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <fstream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef _WIN32
#include <unistd.h> // getpid() for per-writer-unique temp filenames (cross-process atomicity)
#else
#include <process.h> // _getpid()
#define getpid _getpid
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

// --- KV restore-reuse: logits sidecar -------------------------------------------
// When a slot's full state is saved to disk (SLOT_SAVE) for a recurrent/hybrid (FULL) model,
// we additionally persist the last decoded token's full-vocab logits in a small sidecar file
// (<state>.logits). On SLOT_RESTORE of an exact-prompt "regenerate" request, those logits let
// the server emit the first token WITHOUT re-decoding into the (un-rewindable) restored
// recurrent state — which would otherwise crash. The sidecar is independent of libllama's
// state-file format (so that format is left untouched) and is purely best-effort: any
// missing/corrupt/vocab-mismatched sidecar degrades gracefully to the existing behavior.
static constexpr uint32_t SLOT_LOGITS_MAGIC   = 0x474C4B4Cu; // "LKLG" (llama kv logits), LE
static constexpr uint32_t SLOT_LOGITS_VERSION = 1u;

static std::string slot_logits_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".logits";
}

// Best-effort "touch": bump the mtime of an auto-cache snapshot's 3-file unit (state + .logits +
// .meta) to now, so a snapshot that is REUSED (read/restored) but never rewritten is treated as
// recently-used by the mtime LRU. Without this, the LRU is least-recently-WRITTEN, which would
// evict a hot base snapshot that N forked requests keep restoring from (it never gets rewritten).
// Never throws and never errors out the caller: every failure is swallowed via error_code (the
// file may have been concurrently evicted by another process; that is harmless here). This only
// runs on a successful restore, off the generation hot path.
static void auto_touch_unit(const std::string & state_filepath) {
    const auto now = std::filesystem::file_time_type::clock::now();
    std::error_code ec;
    std::filesystem::last_write_time(state_filepath, now, ec);
    std::filesystem::last_write_time(slot_logits_sidecar_path(state_filepath), now, ec);
    std::filesystem::last_write_time(state_filepath + ".meta", now, ec);
}

// Best-effort write of the logits sidecar. Returns the number of bytes written (0 on failure or
// when there is nothing valid to write). Never throws. The file is written to a temp path and
// atomically renamed so a partial/interrupted write can never leave a corrupt sidecar in place.
// Fields are serialized byte-by-byte little-endian (not a raw struct fwrite) for portability;
// the float payload is documented LE-only, matching llama.cpp's native-LE state-file contract.
static size_t slot_logits_write(const std::string & state_filepath,
                                const std::vector<float> & logits,
                                int32_t n_vocab,
                                uint32_t n_tokens) {
    if (logits.empty() || (int32_t) logits.size() != n_vocab || n_vocab <= 0) {
        return 0;
    }
    const std::string sidecar = slot_logits_sidecar_path(state_filepath);
    const std::string tmp     = sidecar + ".tmp";

    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        return 0;
    }
    auto put_u32 = [&](uint32_t v) {
        const unsigned char b[4] = {
            (unsigned char)( v        & 0xFF),
            (unsigned char)((v >> 8)  & 0xFF),
            (unsigned char)((v >> 16) & 0xFF),
            (unsigned char)((v >> 24) & 0xFF),
        };
        f.write((const char *) b, 4);
    };
    put_u32(SLOT_LOGITS_MAGIC);
    put_u32(SLOT_LOGITS_VERSION);
    put_u32((uint32_t) n_vocab);
    put_u32(n_tokens);
    f.write((const char *) logits.data(), (std::streamsize) logits.size() * sizeof(float));
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, sidecar, ec); // atomic replace
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    return 16 + logits.size() * sizeof(float);
}

// Read a logits sidecar. Returns true and fills `out` (size n_vocab) iff a valid sidecar exists
// whose vocab matches `expect_n_vocab` AND whose recorded token count matches `expect_n_tokens`
// (the count of the state just restored). The token-count check is AUTHORITATIVE: it binds the
// sidecar to the exact state it was saved against, so a sidecar that was somehow written for a
// different state length can never be reused. Any mismatch / short read / missing file => false
// with `out` cleared, so the caller falls back to existing behavior. Never throws.
static bool slot_logits_read(const std::string & state_filepath,
                             int32_t expect_n_vocab,
                             uint32_t expect_n_tokens,
                             std::vector<float> & out) {
    out.clear();
    if (expect_n_vocab <= 0) {
        return false;
    }
    const std::string sidecar = slot_logits_sidecar_path(state_filepath);
    std::ifstream f(sidecar, std::ios::binary);
    if (!f) {
        return false;
    }
    auto get_u32 = [&](uint32_t & v) -> bool {
        unsigned char b[4];
        f.read((char *) b, 4);
        if (f.gcount() != 4) {
            return false;
        }
        v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
        return true;
    };
    uint32_t magic = 0, version = 0, n_vocab = 0, n_tokens = 0;
    if (!get_u32(magic) || !get_u32(version) || !get_u32(n_vocab) || !get_u32(n_tokens)) {
        return false;
    }
    if (magic != SLOT_LOGITS_MAGIC || version != SLOT_LOGITS_VERSION ||
        (int32_t) n_vocab != expect_n_vocab || n_tokens != expect_n_tokens) {
        return false;
    }
    out.resize(n_vocab);
    const std::streamsize want = (std::streamsize) n_vocab * (std::streamsize) sizeof(float);
    f.read((char *) out.data(), want);
    if (f.gcount() != want) {
        out.clear();
        return false;
    }
    return true;
}

// --- KV restore-reuse: bounded slot-save store ----------------------------------
// One slot-save snapshot = a state file plus its optional <name>.logits sidecar and (for auto-cache
// snapshots) its <name>.meta sidecar; all are always evicted together as a single unit, keyed by
// the state file's mtime (LRU).
struct slot_save_unit {
    std::string state_path;
    std::string sidecar_path; // "<state>.logits", "" if none
    std::string meta_path;    // "<state>.meta",   "" if none (auto disk cache)
    uintmax_t   bytes = 0;
    std::filesystem::file_time_type mtime;
};

// Enforce --slot-save-max-count / --slot-save-max-bytes over `dir` using LRU-by-mtime eviction.
// `just_written` is the state path that was just saved: it is never evicted, but if it ALONE
// exceeds the byte cap it is deleted (with its sidecar) and `oversized` is set so the caller can
// reject the save rather than evict everything else. Operates strictly within `dir`; uses only
// the error_code std::filesystem overloads so it never throws across the server loop.
//
// IMPORTANT: when a cap is set, --slot-save-path is treated as a server-owned store — any regular
// file in it (other than recognized "<X>.logits" sidecars and "*.tmp" temporaries) is an eviction
// candidate. Point --slot-save-max-count/-mb at a DEDICATED directory; do not mix unrelated files
// into the slot-save directory. (With no caps set — the default unless explicitly enabled — nothing
// is ever deleted and the directory is left exactly as before.)
// `just_written` is the exact filepath string the server built as `slot_save_path + filename`;
// directory_iterator(dir) over that same `slot_save_path` yields identically-spelled path strings
// on POSIX (the production target), so raw string equality correctly identifies the just-saved
// unit. (Not used on Windows in practice; if ever needed there, switch to filename comparison.)
static void slot_save_enforce_limits(const std::string & dir,
                                     int32_t max_count, int64_t max_bytes,
                                     const std::string & just_written,
                                     bool & oversized) {
    oversized = false;
    if (max_count <= 0 && max_bytes <= 0) {
        return; // both unlimited
    }

    std::error_code ec;
    std::vector<slot_save_unit> units;
    uintmax_t this_unit_bytes = 0;

    // First pass: enumerate every regular file once and record the full set of paths so we can
    // tell a real sidecar (sibling of a state file we wrote) from a state file a client happened
    // to name "foo.logits". We must NOT blindly skip every "*.logits" — fs_validate_filename
    // allows that suffix, so a state file literally named "foo.logits" would otherwise escape both
    // caps entirely. Only "<X>.logits" where "<X>" also exists is treated as a sidecar.
    std::vector<std::string> all_files;
    {
        std::set<std::string> present;
        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) {
                continue;
            }
            all_files.push_back(it->path().string());
            present.insert(all_files.back());
        }

        for (const std::string & p : all_files) {
            std::error_code fec;
            // in-flight temp files are never counted or evicted (a concurrent save owns them)
            if (p.size() >= 4 && p.compare(p.size() - 4, 4, ".tmp") == 0) {
                continue;
            }
            // a "<X>.logits" file is a sidecar ONLY when its state file "<X>" is also present;
            // accounted together with that state file below, so skip it here.
            if (p.size() >= 7 && p.compare(p.size() - 7, 7, ".logits") == 0 &&
                present.count(p.substr(0, p.size() - 7))) {
                continue;
            }
            // a "<X>.meta" file is the auto disk cache's tokens+fingerprint sidecar; treat it
            // exactly like ".logits" — accounted with its state file below, reaped if orphaned.
            if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".meta") == 0 &&
                present.count(p.substr(0, p.size() - 5))) {
                continue;
            }
            // reap an ORPHANED sidecar (its state file was evicted/lost): otherwise these silently
            // accumulate (we never count them) and eat real on-disk space forever.
            if (p.size() >= 7 && p.compare(p.size() - 7, 7, ".logits") == 0 &&
                !present.count(p.substr(0, p.size() - 7))) {
                std::filesystem::remove(p, fec);
                continue;
            }
            if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".meta") == 0 &&
                !present.count(p.substr(0, p.size() - 5))) {
                std::filesystem::remove(p, fec);
                continue;
            }

            slot_save_unit u;
            u.state_path = p;
            u.bytes = std::filesystem::file_size(p, fec);
            if (fec) {
                continue;
            }
            const std::string side = p + ".logits";
            if (present.count(side)) {
                const auto sb = std::filesystem::file_size(side, fec);
                if (!fec) {
                    u.sidecar_path = side;
                    u.bytes += sb;
                }
            }
            const std::string meta = p + ".meta";
            if (present.count(meta)) {
                const auto mb = std::filesystem::file_size(meta, fec);
                if (!fec) {
                    u.meta_path = meta;
                    u.bytes += mb;
                }
            }
            u.mtime = std::filesystem::last_write_time(p, fec);
            if (fec) {
                continue;
            }

            if (p == just_written) {
                this_unit_bytes = u.bytes;
            }
            units.push_back(std::move(u));
        }
    }

    // a single snapshot larger than the byte cap is rejected: delete only the just-written unit,
    // do NOT cascade-evict every other (valid) snapshot to make room for something that can't fit.
    // NOTE: intentionally a no-op when max_bytes == 0 (byte cap disabled); in count-only mode an
    // individual snapshot's size is never bounded — only --slot-save-max-mb bounds per-snapshot size.
    if (max_bytes > 0 && this_unit_bytes > (uintmax_t) max_bytes) {
        for (const auto & u : units) {
            if (u.state_path == just_written) {
                std::filesystem::remove(u.state_path, ec);
                if (!u.sidecar_path.empty()) {
                    std::filesystem::remove(u.sidecar_path, ec);
                }
                if (!u.meta_path.empty()) {
                    std::filesystem::remove(u.meta_path, ec);
                }
                break;
            }
        }
        oversized = true;
        return;
    }

    std::sort(units.begin(), units.end(),
              [](const slot_save_unit & a, const slot_save_unit & b) { return a.mtime < b.mtime; }); // oldest first

    size_t    count = units.size();
    uintmax_t total = 0;
    for (const auto & u : units) {
        total += u.bytes;
    }
    size_t idx = 0;

    auto evict_oldest = [&]() -> bool {
        while (idx < units.size() && units[idx].state_path == just_written) {
            idx++; // never evict the snapshot we just wrote
        }
        if (idx >= units.size()) {
            return false;
        }
        const auto & u = units[idx];
        std::filesystem::remove(u.state_path, ec);
        if (!u.sidecar_path.empty()) {
            std::filesystem::remove(u.sidecar_path, ec);
        }
        if (!u.meta_path.empty()) {
            std::filesystem::remove(u.meta_path, ec);
        }
        total -= std::min(total, (uintmax_t) u.bytes);
        count = (count > 0) ? count - 1 : 0;
        idx++;
        return true;
    };

    if (max_count > 0) {
        while (count > (size_t) max_count) {
            if (!evict_oldest()) {
                break;
            }
        }
    }
    if (max_bytes > 0) {
        while (total > (uintmax_t) max_bytes) {
            if (!evict_oldest()) {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// --- Auto disk prompt/KV cache (opt-in: --slot-save-auto) ---
//
// Persists per-slot KV snapshots to disk (state file + '.meta' [tokens + fingerprint]
// + '.logits' sidecar) and indexes them by a chained hash over token IDs, so a cold
// process can reuse a warm process's KV with no client/router involvement.
//
// Design invariants (all must hold; comments below reference them by number):
//   1. Off by default: every hook's FIRST statement is auto_cache_enabled(); when
//      false there is no scan, index, hashing, or allocation — behavior is unchanged.
//   2. Never restore on hash alone: the snapshot's token-ID array must byte-compare
//      equal to the request prefix before any restore (collision-safe).
//   3. Model identity: each snapshot carries a fingerprint (model/vocab/ctx/rope/
//      KV-type/FULL-vs-attention/LoRA); a mismatch refuses the restore.
//   4. Fallback totality: any failure (corrupt file, fp/vocab mismatch, IO error,
//      no match) falls back to a normal prefill — never crash, never wrong output.
//   5. Hot-path purity: the multi-GB save runs only on slot release/reassign, never
//      during generation; restore happens once before prefill.
//
// Concurrency: all slot work runs on the single server-loop thread, so the index is
// single-threaded and the mutex below is uncontended today; it becomes load-bearing
// only if the save I/O is later moved to a worker thread (do not make save async
// without keeping the mutex honest). Independent of legacy --prompt-cache and the
// in-memory prefix-reuse path; auto-restore fires only when in-memory reuse is poor.
// ---------------------------------------------------------------------------

static constexpr uint32_t SLOT_META_MAGIC   = 0x544D4B4Cu; // "LKMT" (llama kv meta), LE
static constexpr uint32_t SLOT_META_VERSION = 1u;

// Model/quant/context fingerprint that MUST match for a restore to be sound. All
// fields are stable inference-affecting identity captured once at model load and
// compared by exact equality (pure-CPU int compares). See invariant 3. The blob
// produced by llama_state_seq_save_file is only safe to load into a context with
// identical KV geometry — a Q4_0-KV blob loaded into an F16 ctx, or a different
// rope/yarn scale (positions are baked into the saved state), silently corrupts —
// so cache_type_k/v and rope_scale are NOT optional.
struct model_fp {
    uint64_t fp_model      = 0; // hash of llama_model_desc + size + n_params (+ n_embd/n_layer)
    uint32_t fp_n_vocab    = 0;
    uint32_t fp_n_ctx_train= 0;
    uint32_t fp_n_embd     = 0;
    uint32_t fp_n_layer    = 0;
    uint32_t fp_rope_type  = 0;
    uint32_t fp_cache_k    = 0; // ggml_type of K cache (enum int)
    uint32_t fp_cache_v    = 0; // ggml_type of V cache (enum int)
    uint32_t fp_n_ctx      = 0; // effective per-seq n_ctx
    uint32_t fp_kv_full    = 0; // 1 if COMMON_CONTEXT_SEQ_RM_TYPE_FULL else 0
    uint32_t fp_block      = 0; // slot_save_block this snapshot was hashed with
    uint64_t fp_rope_scale = 0; // bit-pattern of effective rope_freq_scale (position-critical)
    // rope_freq_base and ALL YaRN params also bake positions into the saved KV state exactly as
    // rope_freq_scale does — a same-model run differing only in --rope-freq-base or any --yarn-*
    // flag would otherwise pass the fingerprint and silently restore positionally-corrupt state.
    // All are bit-cast (float->u32) into identity; yarn_orig_ctx is an int. "0/negative = use
    // model-trained value" is normalized in auto_compute_fingerprint so equal effective configs match.
    uint64_t fp_rope_base       = 0; // bit-pattern of effective rope_freq_base
    uint32_t fp_yarn_ext        = 0; // bit-pattern of yarn_ext_factor
    uint32_t fp_yarn_attn       = 0; // bit-pattern of yarn_attn_factor
    uint32_t fp_yarn_beta_fast  = 0; // bit-pattern of yarn_beta_fast
    uint32_t fp_yarn_beta_slow  = 0; // bit-pattern of yarn_beta_slow
    uint32_t fp_yarn_orig_ctx   = 0; // yarn_orig_ctx (int)
    uint64_t fp_lora       = 0; // hash of active LoRA-set ids+scales (0 if none)
    // refuse cross-shape restores: 1 if the server was launched with --mmproj (mctx != nullptr),
    // else 0. The auto-cache only ever persists text-only prefixes, but mmproj-aware rope (M-RoPE)
    // and projector wiring CAN alter the text KV layout, so we conservatively REFUSE to cross-load
    // a text-only-server snapshot into an mmproj server (or vice-versa) — they get disjoint stores.
    // Removing this bit later would require proving the text KV layout is identical across the two
    // deployment shapes.
    uint32_t fp_mmproj_loaded   = 0;

    // exact field-by-field equality (C++17: no defaulted operator==). Any difference REFUSES the
    // restore (invariant 3). Note: fp_block is intentionally part of identity — a snapshot hashed
    // with a different block size cannot be longest-prefix-matched against the current index.
    bool operator==(const model_fp & o) const {
        return fp_model == o.fp_model && fp_n_vocab == o.fp_n_vocab &&
               fp_n_ctx_train == o.fp_n_ctx_train && fp_n_embd == o.fp_n_embd &&
               fp_n_layer == o.fp_n_layer && fp_rope_type == o.fp_rope_type &&
               fp_cache_k == o.fp_cache_k && fp_cache_v == o.fp_cache_v &&
               fp_n_ctx == o.fp_n_ctx && fp_kv_full == o.fp_kv_full &&
               fp_block == o.fp_block && fp_rope_scale == o.fp_rope_scale &&
               fp_rope_base == o.fp_rope_base && fp_yarn_ext == o.fp_yarn_ext &&
               fp_yarn_attn == o.fp_yarn_attn && fp_yarn_beta_fast == o.fp_yarn_beta_fast &&
               fp_yarn_beta_slow == o.fp_yarn_beta_slow && fp_yarn_orig_ctx == o.fp_yarn_orig_ctx &&
               fp_lora == o.fp_lora && fp_mmproj_loaded == o.fp_mmproj_loaded;
    }
};

// 64-bit chained block hash over token IDs. Each token folds via FNV-1a then a
// splitmix avalanche; block k's output seeds block k+1, so the hash at every block
// boundary commits to the ENTIRE prefix [0, (k+1)*B). Collision resistance is only
// a candidate-narrowing accelerator: we NEVER trust it alone (invariant 2) — the
// caller byte-verifies tokens before any restore. Block boundaries are the only
// resumable prefix lengths (vLLM-APC / SGLang-radix granularity).
static inline uint64_t auto_hash_mix(uint64_t h, int32_t tok) {
    h ^= (uint64_t) (uint32_t) tok;
    h *= 0x100000001b3ULL;                                  // FNV-1a 64-bit prime
    h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32; // splitmix64 finalize
    return h;
}

// Returns, for each block boundary b in [1 .. n/B], the cumulative chain hash
// committing to tokens[0 .. b*B). out[k] = hash of prefix length (k+1)*B. The
// chain is salted with `salt` (the model fingerprint hash) so two different models
// can never produce the same boundary hash for identical tokens. A trailing
// partial block is NOT a boundary (only whole-block prefixes are index keys).
static std::vector<uint64_t> auto_block_hashes(const llama_tokens & toks, int B, uint64_t salt) {
    std::vector<uint64_t> out;
    if (B <= 0) {
        return out;
    }
    out.reserve(toks.size() / (size_t) B);
    uint64_t h = 0xcbf29ce484222325ULL ^ salt; // FNV offset basis, fingerprint-salted
    for (size_t i = 0; i < toks.size(); ++i) {
        h = auto_hash_mix(h, toks[i]);
        if ((i + 1) % (size_t) B == 0) {
            out.push_back(h);
        }
    }
    return out;
}

// One in-memory index entry: the longest snapshot that reaches a given block
// boundary. Mirrors slot_save_unit's mtime LRU semantics for reconciliation.
struct auto_cache_entry {
    std::string state_path; // full state file path (sidecars derived via *_path helpers)
    uint32_t    n_tokens = 0;
    model_fp    fp;         // snapshot's fingerprint (must equal the live one to be used)
};

// boundary-hash -> best (longest) entry covering that prefix length. Touched only
// from the single server-loop thread in v1 (mtx documented above). `scanned`
// guards the one-time startup scan; `dir_mtime`/`last_refresh` drive the cheap
// cross-process refresh (see auto_index_refresh): a peer process that writes a new
// snapshot bumps the slot-save directory's mtime, which the next lookup notices and
// re-scans — so a freshly-created cache becomes visible to OTHER processes without a
// restart (no inotify/no background thread; one stat per gated check).
struct auto_cache_index {
    std::mutex mtx;
    std::unordered_map<uint64_t, auto_cache_entry> by_boundary;
    std::unordered_set<std::string> indexed_files;    // state paths already scanned (incremental refresh)
    bool scanned = false;
    std::filesystem::file_time_type dir_mtime{};      // dir mtime as of the last scan
    std::chrono::steady_clock::time_point last_refresh{}; // throttle: skip stat storms in a burst
};

// Cross-process refresh throttle: at most one dir-mtime stat per this interval on the hot lookup
// path (a forced refresh on a lookup miss bypasses it). Sub-second so a peer's new snapshot is
// visible within ~1 prefill of being written — effectively immediate from the user's view.
static constexpr int AUTO_REFRESH_MIN_MS = 1000;

// Sidecar path twins for an auto snapshot's state file. `.logits` is the committed
// (byte-identical) regenerate sidecar; `.meta` is the NEW tokens+fingerprint
// sidecar this feature adds so the startup scan / pre-restore verify reads only a
// tiny file, never the multi-GB state.
static std::string slot_meta_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".meta";
}

// Best-effort atomic write of the .meta sidecar (LE, temp+rename — the exact idiom
// of slot_logits_write). Layout: magic/version, fingerprint fields, tok_count,
// chain_hash, then int32 tokens[tok_count]. Returns true on success. Never throws.
static bool slot_meta_write(const std::string & state_filepath,
                            const model_fp & fp,
                            const llama_tokens & toks,
                            uint64_t chain_hash) {
    const std::string sidecar = slot_meta_sidecar_path(state_filepath);
    const std::string tmp     = sidecar + ".tmp";

    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    auto put_u32 = [&](uint32_t v) {
        const unsigned char b[4] = {
            (unsigned char)( v        & 0xFF),
            (unsigned char)((v >> 8)  & 0xFF),
            (unsigned char)((v >> 16) & 0xFF),
            (unsigned char)((v >> 24) & 0xFF),
        };
        f.write((const char *) b, 4);
    };
    auto put_u64 = [&](uint64_t v) {
        put_u32((uint32_t)(v & 0xFFFFFFFFu));
        put_u32((uint32_t)(v >> 32));
    };
    put_u32(SLOT_META_MAGIC);
    put_u32(SLOT_META_VERSION);
    put_u64(fp.fp_model);
    put_u32(fp.fp_n_vocab);
    put_u32(fp.fp_n_ctx_train);
    put_u32(fp.fp_n_embd);
    put_u32(fp.fp_n_layer);
    put_u32(fp.fp_rope_type);
    put_u32(fp.fp_cache_k);
    put_u32(fp.fp_cache_v);
    put_u32(fp.fp_n_ctx);
    put_u32(fp.fp_kv_full);
    put_u32(fp.fp_block);
    put_u64(fp.fp_rope_scale);
    // rope_freq_base + YaRN fingerprint fields
    put_u64(fp.fp_rope_base);
    put_u32(fp.fp_yarn_ext);
    put_u32(fp.fp_yarn_attn);
    put_u32(fp.fp_yarn_beta_fast);
    put_u32(fp.fp_yarn_beta_slow);
    put_u32(fp.fp_yarn_orig_ctx);
    put_u64(fp.fp_lora);
    // mmproj deployment-shape bit — refuses cross-shape restores.
    put_u32(fp.fp_mmproj_loaded);
    put_u32((uint32_t) toks.size());
    put_u64(chain_hash);
    // token IDs as raw LE int32 (llama_token == int32_t; llama.cpp's on-disk
    // contract is native-LE, matching slot_logits_write's float payload).
    f.write((const char *) toks.data(), (std::streamsize) toks.size() * sizeof(int32_t));
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return false;
    }
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, sidecar, ec); // atomic replace
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

// Read a .meta sidecar. Returns true and fills `fp_out` + `toks_out` iff a valid
// sidecar exists. Any short read / bad magic / version mismatch => false with
// outputs cleared (invariant 4). Never throws. Note: `chain_hash` is recorded for
// debuggability but the authority for reuse is always the byte-compared tokens.
static bool slot_meta_read(const std::string & state_filepath,
                           model_fp & fp_out,
                           llama_tokens & toks_out) {
    fp_out = model_fp{};
    toks_out.clear();
    const std::string sidecar = slot_meta_sidecar_path(state_filepath);
    std::ifstream f(sidecar, std::ios::binary);
    if (!f) {
        return false;
    }
    auto get_u32 = [&](uint32_t & v) -> bool {
        unsigned char b[4];
        f.read((char *) b, 4);
        if (f.gcount() != 4) {
            return false;
        }
        v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
        return true;
    };
    auto get_u64 = [&](uint64_t & v) -> bool {
        uint32_t lo = 0, hi = 0;
        if (!get_u32(lo) || !get_u32(hi)) {
            return false;
        }
        v = (uint64_t) lo | ((uint64_t) hi << 32);
        return true;
    };
    uint32_t magic = 0, version = 0;
    if (!get_u32(magic) || !get_u32(version)) {
        return false;
    }
    if (magic != SLOT_META_MAGIC || version != SLOT_META_VERSION) {
        return false;
    }
    model_fp fp;
    uint32_t tok_count = 0;
    uint64_t chain_hash = 0;
    if (!get_u64(fp.fp_model)         || !get_u32(fp.fp_n_vocab)       || !get_u32(fp.fp_n_ctx_train) ||
        !get_u32(fp.fp_n_embd)        || !get_u32(fp.fp_n_layer)       || !get_u32(fp.fp_rope_type)   ||
        !get_u32(fp.fp_cache_k)       || !get_u32(fp.fp_cache_v)       || !get_u32(fp.fp_n_ctx)       ||
        !get_u32(fp.fp_kv_full)       || !get_u32(fp.fp_block)         || !get_u64(fp.fp_rope_scale)  ||
        // rope_freq_base + YaRN — must be read in the same order slot_meta_write emits.
        !get_u64(fp.fp_rope_base)     || !get_u32(fp.fp_yarn_ext)      || !get_u32(fp.fp_yarn_attn)   ||
        !get_u32(fp.fp_yarn_beta_fast)|| !get_u32(fp.fp_yarn_beta_slow)|| !get_u32(fp.fp_yarn_orig_ctx)||
        // mmproj deployment-shape bit — read in the same order slot_meta_write emits.
        !get_u64(fp.fp_lora)          || !get_u32(fp.fp_mmproj_loaded) ||
        !get_u32(tok_count)           || !get_u64(chain_hash)) {
        return false;
    }
    (void) chain_hash;
    // sanity-bound the count so a corrupt header cannot make us allocate gigabytes.
    if (tok_count > (1u << 28)) {
        return false;
    }
    toks_out.resize(tok_count);
    const std::streamsize want = (std::streamsize) tok_count * (std::streamsize) sizeof(int32_t);
    f.read((char *) toks_out.data(), want);
    if (f.gcount() != want) {
        toks_out.clear();
        return false;
    }
    fp_out = fp;
    return true;
}

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    bool just_restored  = false; // set on disk slot-restore; one-shot, gates restored-slot KV reuse

    // --- KV restore-reuse (logits sidecar) ---
    // Full-vocab logits of this slot's most recently sampled token, captured at sample time
    // (only populated for FULL/recurrent models when --slot-save-path is set). Serialized to the
    // <state>.logits sidecar on SLOT_SAVE so a later exact-prompt "regenerate" can emit the first
    // token WITHOUT re-decoding into the (un-rewindable) restored recurrent state.
    std::vector<float> logits_last;     // size n_vocab when valid, else empty
    // Token count of the prompt-state that `logits_last` corresponds to (i.e. the slot's KV/token
    // length at the moment of capture). Used to BIND the captured distribution to a specific state:
    // a sidecar is only written when this equals the saved snapshot's token_count, so a stale
    // distribution (e.g. left over from a prior task, or skipped on a spec-decode step) can never
    // be serialized against a mismatched state. -1 = no valid capture.
    int32_t logits_last_n_tokens = -1;
    // Logits loaded from a sidecar at SLOT_RESTORE, consumed once by the restore-continue path.
    std::vector<float> restored_logits; // size n_vocab when a valid sidecar was loaded, else empty

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        common_context_seq_rm(ctx_tgt, id, -1, -1);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, id, -1, -1);
        }

        prompt.tokens.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_print_last = 0;
    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // one-shot; never carry restored sidecar logits into a non-restore request.
        // NOTE: logits_last is deliberately NOT cleared here — it is the slot's running
        // "last sampled distribution" and must survive into the idle state so a subsequent
        // SLOT_SAVE can serialize it.
        restored_logits.clear();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
    }

    bool need_embd_pre_norm() const {
        GGML_ASSERT(task);
        return spec && common_speculative_need_embd_pre_norm(spec);
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            for (auto token : spec_draft) {
                common_batch_add(batch, token, pos0++, { this->id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        // Guard against n_prompt_tokens_processed == 0 (e.g. the restore-continue regenerate
        // fast-path, where the entire prompt is reused and zero tokens are re-processed). Without
        // this, the divisions emit inf/NaN which then serialize as invalid JSON in the response's
        // "timings" object.
        timings.prompt_per_token_ms = n_prompt_tokens_processed > 0 ? t_prompt_processing / n_prompt_tokens_processed : 0.0;
        timings.prompt_per_second   = n_prompt_tokens_processed > 0 ? 1e3 / t_prompt_processing * n_prompt_tokens_processed : 0.0;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        t_print_last = t_now;

        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s\n", n_decoded, n_gen_second);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total);
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        common_context_seq_rm(ctx_tgt, other.id,     -1, -1);
        common_context_seq_cp(ctx_tgt, id, other.id, -1, -1);

        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, other.id,     -1, -1);
            common_context_seq_cp(ctx_dft, id, other.id, -1, -1);
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_ui_settings = json::object();    // Primary: new name
    json json_webui_settings = json::object();    // Deprecated: use json_ui_settings instead (kept for compat)

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    // --- auto disk prompt/KV cache (opt-in: --slot-save-auto) ---
    // Default-constructed: empty and untouched when the feature is OFF (invariant 1).
    // `cur_fp` is the live model fingerprint, computed once at load (only when enabled).
    auto_cache_index auto_idx;
    model_fp         cur_fp;

    // The ONE gate for the entire auto disk cache. When false, NO hook below does
    // any work (no scan, no hash, no alloc). This is invariant 1 — the first
    // statement of every auto_* hook is `if (!auto_cache_enabled()) return;`.
    bool auto_cache_enabled() const {
        return params_base.slot_save_auto && !params_base.slot_save_path.empty();
    }

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        llama_batch_free(batch);
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        // Auto-save (write path, invariant 5): persist this slot's KV to disk BEFORE the
        // in-memory cache drops it. This runs at task-launch time for idle slots (off the
        // launching task's generation hot path), reusing the exact SLOT_SAVE machinery.
        // Gate the CALL SITE so the entire call frame is elided when OFF (this is the dominant
        // idle-slot-flush path under cache_idle_slots=true). The callee keeps its own
        // first-statement gate as defense-in-depth.
        if (auto_cache_enabled()) {
            auto_save_slot_if_useful(slot);
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    // ----- auto disk cache: fingerprint, index, restore, save (all gated by auto_cache_enabled()) -----

    // hash of the active LoRA set (ids/scales) for the fingerprint; 0 when no adapter is active.
    static uint64_t auto_lora_hash(const std::vector<common_adapter_lora_info> & lora) {
        uint64_t h = 0xcbf29ce484222325ULL;
        bool any = false;
        for (const auto & a : lora) {
            if (a.scale == 0.0f) {
                continue; // disabled adapter does not affect inference identity
            }
            any = true;
            for (char c : a.path) {
                h ^= (uint64_t) (unsigned char) c; h *= 0x100000001b3ULL;
            }
            uint32_t sc; std::memcpy(&sc, &a.scale, sizeof(sc));
            h = auto_hash_mix(h, (int32_t) sc);
        }
        return any ? h : 0;
    }

    // Compute the live model fingerprint once at load (invariant 3). Pure-CPU; only called from
    // an auto_cache_enabled() branch so it costs nothing when OFF.
    // See README "Automatic disk prompt cache" for which flags invalidate the cache.
    model_fp auto_compute_fingerprint() const {
        model_fp fp;
        // model identity string (arch + params + quant), hardened with size/n_params/n_embd/n_layer.
        char desc[256] = {0};
        llama_model_desc(model_tgt, desc, sizeof(desc));
        uint64_t h = 0xcbf29ce484222325ULL;
        for (const char * p = desc; *p; ++p) {
            h ^= (uint64_t) (unsigned char) *p; h *= 0x100000001b3ULL;
        }
        const uint64_t sz = llama_model_size(model_tgt);
        const uint64_t np = llama_model_n_params(model_tgt);
        h = auto_hash_mix(h, (int32_t) (sz & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (sz >> 32));
        h = auto_hash_mix(h, (int32_t) (np & 0xFFFFFFFFu)); h = auto_hash_mix(h, (int32_t) (np >> 32));

        fp.fp_model       = h;
        fp.fp_n_vocab     = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
        fp.fp_n_ctx_train = (uint32_t) llama_model_n_ctx_train(model_tgt);
        fp.fp_n_embd      = (uint32_t) llama_model_n_embd(model_tgt);
        fp.fp_n_layer     = (uint32_t) llama_model_n_layer(model_tgt);
        fp.fp_rope_type   = (uint32_t) llama_model_rope_type(model_tgt);
        // K/V cache type has no live-ctx getter — capture from the server's own params (the value
        // used to construct ctx_tgt). Blob-layout-critical: a Q4_0-KV blob into an F16 ctx corrupts.
        fp.fp_cache_k     = (uint32_t) params_base.cache_type_k;
        fp.fp_cache_v     = (uint32_t) params_base.cache_type_v;
        fp.fp_n_ctx       = (uint32_t) llama_n_ctx_seq(ctx_tgt);
        fp.fp_kv_full     = (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) ? 1u : 0u;
        fp.fp_block       = (uint32_t) params_base.slot_save_block;
        // effective rope scale (positions are baked into the saved state). rope_freq_scale==0 means
        // "use the model's trained value", so fall back to that for a stable comparison.
        float rs = params_base.rope_freq_scale != 0.0f
                       ? params_base.rope_freq_scale
                       : llama_model_rope_freq_scale_train(model_tgt);
        uint32_t rsb; std::memcpy(&rsb, &rs, sizeof(rsb));
        fp.fp_rope_scale  = (uint64_t) rsb;
        // rope_freq_base + the five YaRN params also bake positions into the saved
        // KV, so they MUST be part of identity. There is no public getter for the model's trained
        // rope base, so we normalize "use-model-default" to a single canonical 0 sentinel: when the
        // operator left the knob at its default (rope_freq_base==0; YaRN floats<0, i.e. -1.0 "auto";
        // yarn_orig_ctx<=0), we store 0. Two runs that both rely on the model default thus match;
        // any explicit override (or two different overrides) yields a different fp and refuses
        // (conservative — a needless miss is safe, a wrong restore is not). yarn_orig_ctx is an int.
        auto bitcast_f = [](float v) -> uint32_t { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; };
        auto norm_yarn = [&](float v) -> uint32_t { return v < 0.0f ? 0u : bitcast_f(v); }; // <0 == model default
        const float rb = params_base.rope_freq_base > 0.0f ? params_base.rope_freq_base : 0.0f; // 0 == model default
        uint32_t rbb; std::memcpy(&rbb, &rb, sizeof(rbb));
        fp.fp_rope_base      = (uint64_t) rbb;
        fp.fp_yarn_ext       = norm_yarn(params_base.yarn_ext_factor);
        fp.fp_yarn_attn      = norm_yarn(params_base.yarn_attn_factor);
        fp.fp_yarn_beta_fast = norm_yarn(params_base.yarn_beta_fast);
        fp.fp_yarn_beta_slow = norm_yarn(params_base.yarn_beta_slow);
        fp.fp_yarn_orig_ctx  = params_base.yarn_orig_ctx > 0 ? (uint32_t) params_base.yarn_orig_ctx : 0u;
        // LoRA: fingerprint the global adapter set so a snapshot under adapter A never restores
        // under B. (Per-request adapter overrides additionally gate at the restore hook.)
        fp.fp_lora        = auto_lora_hash(params_base.lora_adapters);
        // deployment-shape bit (invariant 3): text-only server vs --mmproj server get
        // disjoint stores (mmproj-aware rope/projector wiring can change the text KV layout).
        fp.fp_mmproj_loaded = (mctx != nullptr) ? 1u : 0u;
        return fp;
    }

    // Auto-snapshot filename: model-fp prefix lets the startup scan reject foreign-model files by
    // name before opening anything; chain-hash + token-count make it deterministic across processes
    // (a same-prefix save from another process yields the same name -> atomic-rename-idempotent).
    std::string auto_state_filename(uint64_t chain_hash, size_t n_tokens) const {
        char buf[96]; // "auto-" + 16 hex fp + "-" + 16 hex hash + "-" + up to 20-digit count + ".bin" < 96
        snprintf(buf, sizeof(buf), "auto-%016" PRIx64 "-%016" PRIx64 "-%zu.bin",
                 cur_fp.fp_model, chain_hash, n_tokens);
        return params_base.slot_save_path + std::string(buf);
    }

    // Insert/keep-longer: an entry replaces an existing boundary only if it covers a longer prefix.
    void auto_index_insert_locked(uint64_t boundary, const auto_cache_entry & e) {
        auto it = auto_idx.by_boundary.find(boundary);
        if (it == auto_idx.by_boundary.end() || it->second.n_tokens < e.n_tokens) {
            auto_idx.by_boundary[boundary] = e;
        }
    }

    // Scan the slot-save dir and (re)build index entries from .meta sidecars: header-only reads
    // (never the multi-GB state). Each bad/foreign file is skipped individually (invariant 4);
    // foreign-model files are left on disk (a sibling model may own them). Idempotent: re-running it
    // only ever keep-longer-inserts the same/new entries (auto_index_insert_locked), so it is safe to
    // call repeatedly for the cross-process refresh. Records the dir mtime so a refresh can cheaply
    // tell whether anything changed. CALLER MUST HOLD auto_idx.mtx.
    void auto_index_scan_locked() {
        std::error_code mec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
        if (!mec) {
            auto_idx.dir_mtime = dmt; // snapshot the dir mtime we are scanning at
        }
        std::error_code ec;
        for (std::filesystem::directory_iterator it(params_base.slot_save_path, ec), end;
             !ec && it != end; it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) {
                continue;
            }
            const std::string p = it->path().string();
            // only our own state files: basename "auto-*.bin" (sidecars and temps skipped). Requiring
            // the "auto-" basename prefix rejects foreign/manual .bin files BY NAME before we open any
            // sidecar — the stated scan optimization.
            const std::string base = it->path().filename().string();
            if (base.rfind("auto-", 0) != 0) {
                continue; // not one of ours
            }
            if (p.size() < 4 || p.compare(p.size() - 4, 4, ".bin") != 0) {
                continue;
            }
            // already indexed by a prior scan? cheap skip so a refresh only opens NEW files.
            if (auto_idx.indexed_files.count(p)) {
                continue;
            }
            model_fp fp;
            llama_tokens toks;
            if (!slot_meta_read(p, fp, toks)) {
                continue; // no/short/corrupt meta -> not indexable (invariant 4)
            }
            if (!(fp == cur_fp)) {
                continue; // foreign model / requant / different ctx geometry (invariant 3)
            }
            const auto bhs = auto_block_hashes(toks, params_base.slot_save_block, cur_fp.fp_model);
            auto_cache_entry e{ p, (uint32_t) toks.size(), fp };
            for (uint64_t bh : bhs) {
                auto_index_insert_locked(bh, e);
            }
            auto_idx.indexed_files.insert(p);
        }
    }

    // One-time startup scan: builds the initial index. Invariant 1: only ever called from an
    // auto_cache_enabled() branch.
    void auto_index_scan() {
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        if (auto_idx.scanned) {
            return;
        }
        auto_idx.scanned = true;
        auto_idx.last_refresh = std::chrono::steady_clock::now();
        auto_index_scan_locked();
    }

    // Cross-process refresh: make snapshots that OTHER processes created visible here WITHOUT a
    // restart. Cheap by design: throttled to at most once per AUTO_REFRESH_MIN_MS, and even then it
    // only does one stat of the dir mtime — a full re-scan happens ONLY when the dir actually changed
    // (a peer create/rename/delete bumps the dir mtime) or when `force` is set (a lookup miss, where
    // we are about to pay a cold prefill anyway so the scan is free in comparison). On a change we
    // also drop entries whose files a peer evicted. CALLER MUST HOLD auto_idx.mtx.
    void auto_index_refresh_locked(bool force) {
        const auto now = std::chrono::steady_clock::now();
        if (!force &&
            now - auto_idx.last_refresh < std::chrono::milliseconds(AUTO_REFRESH_MIN_MS)) {
            return; // throttled: avoid a stat storm during a burst of lookups
        }
        auto_idx.last_refresh = now;
        std::error_code ec;
        const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, ec);
        if (!ec && dmt == auto_idx.dir_mtime && !force) {
            return; // nothing changed on disk since the last scan
        }
        // a peer changed the dir (or forced): re-scan for NEW files, then reconcile deletions.
        auto_index_scan_locked();
        auto_index_drop_missing_locked();
    }

    // Longest-prefix lookup over the request tokens. Returns the candidate whose boundary hash is
    // the DEEPEST match with a fingerprint equal to the live one. Verification (byte-compare of the
    // candidate's persisted tokens) is mandatory and done by the caller (invariant 2). O(#blocks).
    std::optional<auto_cache_entry> auto_index_lookup(const llama_tokens & req) {
        if (!auto_cache_enabled()) {
            return std::nullopt; // off by default
        }
        const auto bhs = auto_block_hashes(req, params_base.slot_save_block, cur_fp.fp_model);
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        // Cross-process visibility: cheaply pick up snapshots a peer process created since our last
        // scan (throttled dir-mtime check). Then search; on a MISS, force a re-scan and search again
        // — the force is justified because a miss means we are about to cold-prefill, so the scan
        // cost is negligible against it, and a peer's snapshot written <1s ago (within the throttle
        // window) is still found on this first request rather than only the next one.
        auto_index_refresh_locked(/*force=*/false);
        for (int attempt = 0; attempt < 2; ++attempt) {
            for (size_t k = bhs.size(); k-- > 0; ) { // longest boundary first
                auto it = auto_idx.by_boundary.find(bhs[k]);
                if (it == auto_idx.by_boundary.end()) {
                    continue;
                }
                if (!(it->second.fp == cur_fp)) {
                    continue; // invariant 3
                }
                return it->second;
            }
            if (attempt == 0) {
                auto_index_refresh_locked(/*force=*/true); // miss -> rescan once before giving up
            }
        }
        return std::nullopt;
    }

    // After an LRU eviction (which deletes files silently — ours OR a peer process's), drop index
    // boundaries pointing at files that no longer exist, and forget them in indexed_files so a future
    // re-create can be re-indexed. Cheap stat per unique path; keeps index <-> disk consistent (invariant 4).
    // A lookup that races an eviction and finds a now-deleted file simply fails the load -> prefill.
    // CALLER MUST HOLD auto_idx.mtx.
    void auto_index_drop_missing_locked() {
        std::unordered_set<std::string> gone;
        for (auto it = auto_idx.by_boundary.begin(); it != auto_idx.by_boundary.end(); ) {
            std::error_code ec;
            if (!std::filesystem::exists(it->second.state_path, ec) || ec) {
                gone.insert(it->second.state_path);
                it = auto_idx.by_boundary.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto & p : gone) {
            auto_idx.indexed_files.erase(p);
        }
    }

    void auto_index_drop_missing() {
        std::lock_guard<std::mutex> lk(auto_idx.mtx);
        auto_index_drop_missing_locked();
    }

    // Restore a disk snapshot INTO `slot`, mirroring the SLOT_RESTORE handler body. Returns true on
    // success (slot.prompt.tokens / n_past-equivalent + just_restored + restored_logits are set as
    // for a manual restore). On ANY failure (load <=0, capacity exceeded) the slot seq is left
    // cleared and false is returned so the caller falls through to a normal prefill (invariant 4).
    bool do_slot_restore(server_slot & slot, const std::string & filepath,
                         size_t * out_token_count = nullptr, size_t * out_nread = nullptr) {
        llama_tokens tokens;
        tokens.resize(slot.n_ctx);
        size_t token_count = 0;
        const size_t nread = llama_state_seq_load_file(
            ctx_tgt, filepath.c_str(), slot.id, tokens.data(), tokens.size(), &token_count);
        if (out_nread)       { *out_nread = nread; }
        if (out_token_count) { *out_token_count = token_count; }
        if (nread == 0) {
            slot.prompt.tokens.clear(); // KV may already have been invalidated by the partial load
            return false;
        }
        tokens.resize(token_count);
        slot.prompt.tokens.clear();
        slot.prompt.tokens.insert(tokens);
        slot.just_restored = true;

        // Reconstruct a context checkpoint at the restored position so hybrid/recurrent (and SWA)
        // models — which cannot partially rewind — can reuse this state for the suffix; other
        // models do not need it.
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            const auto ckpt_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
            const auto ckpt_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (ckpt_pos_min >= 0) {
                slot.prompt.checkpoints.clear();
                create_checkpoint(slot, 0, ckpt_pos_min, ckpt_pos_max);
            }
        }

        // The restored state's running distribution is unknown; invalidate any stale capture so a
        // later SLOT_SAVE cannot persist a mismatched sidecar.
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;

        // Load the regenerate logits sidecar (FULL only) so an exact-prompt regenerate can emit the
        // first token without re-decoding into the restored recurrent state.
        slot.restored_logits.clear();
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            if (slot_logits_read(filepath, nv, (uint32_t) token_count, slot.restored_logits)) {
                SLT_INF(slot, "loaded logits sidecar (%d vocab, %zu tokens) — regenerate fast-path armed\n", nv, token_count);
            }
        }
        return true;
    }

    // AUTO-RESTORE wrapper: byte-verify the candidate's persisted tokens against the request prefix
    // (invariant 2), confirm the fingerprint (invariant 3), then restore. Returns the verified prefix length
    // actually restored, or 0 if nothing was restored (caller keeps the in-memory prefill path).
    // `req` is the full request token-ID array; `n_keep_mem` is the in-memory match to beat.
    int auto_restore_into_slot(server_slot & slot, const auto_cache_entry & cand,
                               const llama_tokens & req, int n_keep_mem) {
        // read the small .meta sidecar (tokens + fp) — never opens the multi-GB state file (invariant 5).
        model_fp disk_fp;
        llama_tokens disk_toks;
        if (!slot_meta_read(cand.state_path, disk_fp, disk_toks)) {
            return 0; // invariant 4
        }
        if (!(disk_fp == cur_fp)) {
            return 0; // invariant 3
        }
        // byte-verify: longest common prefix of the persisted tokens and the request (invariant 2).
        const size_t lim = std::min(disk_toks.size(), req.size());
        size_t v = 0;
        while (v < lim && disk_toks[v] == req[v]) {
            ++v;
        }
        // Only WHOLE-block prefixes are valid reuse lengths (hash boundaries).
        const int B = params_base.slot_save_block;
        int n_keep_disk;
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            // a FULL/recurrent/hybrid/SWA state cannot be PARTIALLY rewound —
            // do_slot_restore loads the ENTIRE L-token snapshot, and a later keep_first(n_past<L)
            // would issue a PARTIAL common_context_seq_rm that GGML_ABORTs the server (a FULL model's
            // llama_memory_seq_rm refuses a partial range). So we ONLY auto-restore a FULL snapshot
            // when the request diverges at or beyond the snapshot end (v == disk_toks.size(), i.e.
            // the whole snapshot is a verified prefix of the request). If the request diverges INSIDE
            // the snapshot, refuse and fall back to normal prefill — never restore a FULL snapshot we
            // would have to partially unwind. (No block-boundary clamp for FULL: only the exact whole
            // snapshot is a legal restore length here.)
            if (v != disk_toks.size()) {
                return 0;
            }
            n_keep_disk = (int) disk_toks.size();
        } else {
            // Attention (PART) models support per-token partial seq_rm, so a mid-snapshot divergence
            // is fine: claim the verified prefix clamped down to the last whole block boundary <= v.
            // An exact full-snapshot match keeps the whole snapshot length.
            if (v == disk_toks.size()) {
                n_keep_disk = (int) disk_toks.size();
            } else {
                n_keep_disk = (int) (v - (v % (size_t) B));
            }
        }
        if (n_keep_disk <= 0) {
            return 0;
        }
        // MARGIN gate (invariant 5): only pay a multi-GB load if disk strictly beats the
        // in-memory match by at least one block — never thrash a reload to save a few tokens.
        if (n_keep_disk < n_keep_mem + B) {
            return 0;
        }
        // Clear the slot's resident KV before loading the snapshot (mirror the restore-continue safe
        // fallback): seq removal + token/checkpoint clear so the restore writes into an empty seq.
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
        slot.prompt.tokens.clear();
        slot.prompt.checkpoints.clear();

        if (!do_slot_restore(slot, cand.state_path)) {
            // restore failed -> slot seq already cleared by do_slot_restore; caller reprefills (invariant 4).
            return 0;
        }
        // do_slot_restore loaded the snapshot. For FULL models n_keep_disk == snapshot length (gated
        // above), so the existing regenerate / suffix-reuse path takes over with no partial
        // rewind. For attention models the request may diverge inside the snapshot; keep_first(n_past)
        // + a PARTIAL seq_rm then reprefills the divergent tail (supported for PART). The verified
        // prefix is what we claim as reused.
        // Bump the snapshot's mtime so the LRU treats a reused-but-not-rewritten base snapshot as
        // recently-used (true LRU, not least-recently-written) — critical for the fan-out case where
        // many requests restore one hot base prefix. Best-effort; never errors the restore (invariant 4).
        auto_touch_unit(cand.state_path);
        SLT_INF(slot, "auto-restore: reused %d tokens from disk (in-memory match was %d), file=%s\n",
                n_keep_disk, n_keep_mem, cand.state_path.c_str());
        return n_keep_disk;
    }

    // AUTO-SAVE: persist a slot's KV before it is discarded, keyed by its token-prefix block hash.
    // Skips redundant writes (an equal-or-longer snapshot already covers this prefix), writes the
    // state + .logits + .meta as a 3-file unit (atomically, .meta LAST so a torn write is never
    // indexed), enforces the bounded LRU, then reconciles the index. Invariant 1: first statement
    // is the gate; invariant 5: only called on slot release/reassign, never during generation.
    void auto_save_slot_if_useful(server_slot & slot) {
        if (!auto_cache_enabled()) {
            return; // off by default
        }
        // exclusions reuse the existing guards. NOTE: an idle slot has already been reset(), so
        // `slot.task` is null here — the just-finished task survives as `slot.task_prev`. Use it for
        // the generative check (COMPLETION/INFILL only). Gate on the PER-REQUEST `has_media()` (not
        // the server-wide has_mtmd/mctx) so an --mmproj server still persists its text-only turns;
        // a turn carrying an image (has_media()==true) is skipped — exactly correct, since token-ids
        // alone cannot identify image content.
        const auto & wtask = slot.task ? slot.task : slot.task_prev;
        if (!wtask || !wtask->need_sampling() || slot.prompt.tokens.has_media()) {
            return;
        }
        // The fingerprint captures the GLOBAL LoRA set; refuse to persist a snapshot taken under a
        // per-request adapter override that differs from it (invariant 3). (Conservative: a future version
        // could fold the slot's adapters into the snapshot fingerprint instead.)
        if (!are_lora_equal(slot.lora, params_base.lora_adapters)) {
            return;
        }
        // get_text_tokens() (not get_tokens()): media-safe accessor that never trips the
        // get_tokens() GGML_ASSERT(!has_mtmd) under mmproj. For this no-media prompt (has_media()
        // false, guarded above) it equals the full token-id prefix, so the persisted token stream
        // and the block-hash key are byte-identical to what a text-only server would write.
        const llama_tokens toks = slot.prompt.tokens.get_text_tokens();
        if ((int) toks.size() < params_base.slot_save_block) {
            return; // < 1 block: not worth a multi-GB write
        }
        const auto bhs = auto_block_hashes(toks, params_base.slot_save_block, cur_fp.fp_model);
        if (bhs.empty()) {
            return;
        }
        const uint64_t full_hash = bhs.back(); // commits the whole whole-block prefix
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto it = auto_idx.by_boundary.find(full_hash);
            if (it != auto_idx.by_boundary.end() && it->second.n_tokens >= toks.size()) {
                return; // an equal-or-longer snapshot for this exact prefix already exists
            }
        }

        const std::string fname = auto_state_filename(full_hash, toks.size());
        // cross-process atomicity: the temp path MUST be unique per writer. The final
        // name (fname) is deterministic (fp + chain hash + tok count), so two processes sharing one
        // --slot-save-path would otherwise both stream a multi-GB state into the SAME "<fname>.tmp"
        // and interleave -> a corrupt temp gets renamed over a good final file. We disambiguate the
        // temp with pid + a per-process monotonic counter, so each writer owns its own complete temp
        // and the deterministic-name rename is the ONLY shared, atomic step (idempotent: identical
        // content). The sidecar temps derive from this same unique base so they are unique too.
        // (nonce is atomic so it stays correct if save I/O is later threaded.)
        static std::atomic<uint64_t> s_tmp_nonce{0};
        const uint64_t nonce = s_tmp_nonce.fetch_add(1, std::memory_order_relaxed);
        const std::string tmp = fname + "." + std::to_string((long) getpid()) + "." +
                                std::to_string(nonce) + ".tmp";

        // 1) write the state to a per-writer-unique temp path (atomic via rename below). NOTE:
        //    llama_state_seq_save_file writes in place, so we write to the unique temp then rename — a
        //    crash mid-write never leaves a corrupt state file the index would trust.
        const size_t nwrite = llama_state_seq_save_file(ctx_tgt, tmp.c_str(), slot.id,
                                                        toks.data(), toks.size());
        if (nwrite == 0) {
            std::error_code ec; std::filesystem::remove(tmp, ec);
            return; // invariant 4: disk full / IO error -> generation unaffected
        }
        // 2) regenerate logits sidecar on the temp path (FULL only, and only when the captured
        //    distribution provably belongs to this exact state — the same stamp check SLOT_SAVE uses).
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
            slot.logits_last_n_tokens == (int32_t) toks.size() && !slot.logits_last.empty()) {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
            slot_logits_write(tmp, slot.logits_last, nv, (uint32_t) toks.size());
        }
        // 3) meta sidecar on the temp path (tokens + fingerprint). Written but renamed LAST.
        if (!slot_meta_write(tmp, cur_fp, toks, full_hash)) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            std::filesystem::remove(slot_logits_sidecar_path(tmp), ec);
            std::filesystem::remove(slot_meta_sidecar_path(tmp), ec);
            return; // invariant 4
        }
        // 4) atomic publish: rename state first, then sidecars to their final names. .meta is the
        //    last to appear, so the startup scan (which keys on .meta) never sees a half-written unit.
        std::error_code ec;
        std::filesystem::rename(tmp, fname, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            std::filesystem::remove(slot_logits_sidecar_path(tmp), ec);
            std::filesystem::remove(slot_meta_sidecar_path(tmp), ec);
            return; // invariant 4
        }
        std::filesystem::rename(slot_logits_sidecar_path(tmp), slot_logits_sidecar_path(fname), ec);
        ec.clear();
        std::filesystem::rename(slot_meta_sidecar_path(tmp), slot_meta_sidecar_path(fname), ec);
        // the .meta is the scan key — a unit whose .meta never landed must NOT be
        // published. If the meta rename failed, the .bin is already in place but unindexable, so we
        // unlink the orphan .bin (and any leftover temps) and DO NOT insert into the in-memory index.
        // Leaving the .bin would waste disk and a restart scan would skip it anyway (no .meta).
        if (ec) {
            std::error_code rec;
            std::filesystem::remove(fname, rec);
            std::filesystem::remove(slot_logits_sidecar_path(fname), rec);
            std::filesystem::remove(slot_logits_sidecar_path(tmp), rec);
            std::filesystem::remove(slot_meta_sidecar_path(tmp), rec);
            return; // invariant 4: don't index a unit whose .meta (the scan key) never published
        }

        SLT_INF(slot, "auto-save: persisted %zu tokens to %s\n", toks.size(), fname.c_str());

        // index insert (every boundary -> this snapshot), then bounded-LRU + reconcile.
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto_cache_entry e{ fname, (uint32_t) toks.size(), cur_fp };
            for (uint64_t bh : bhs) {
                auto_index_insert_locked(bh, e);
            }
            auto_idx.indexed_files.insert(fname); // remember our own write so a refresh won't re-open it
        }
        if (params_base.slot_save_max_count > 0 || params_base.slot_save_max_bytes > 0) {
            bool oversized = false;
            slot_save_enforce_limits(params_base.slot_save_path,
                                     params_base.slot_save_max_count,
                                     params_base.slot_save_max_bytes,
                                     fname, oversized);
        }
        // Reconcile index with what the LRU kept (ours or a peer's) AND adopt the post-write dir
        // mtime as our scan baseline — both under ONE lock. Re-baselining here means OUR OWN
        // save+evict does not make the next lookup think a PEER changed the dir (which would force a
        // redundant full re-scan); a real peer write afterwards bumps the mtime again -> still
        // detected. CALLER holds no lock here.
        {
            std::lock_guard<std::mutex> lk(auto_idx.mtx);
            auto_index_drop_missing_locked();
            std::error_code mec;
            const auto dmt = std::filesystem::last_write_time(params_base.slot_save_path, mec);
            if (!mec) {
                auto_idx.dir_mtime = dmt;
            }
        }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;

        std::string & mmproj_path = params_base.mmproj.path;
        bool has_mmproj = !mmproj_path.empty();
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.media_marker     = get_media_marker();
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_INF("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB\n", total / (1024.0 * 1024.0));
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
            const bool has_draft = params_base.speculative.has_dft();

            if (has_draft || spec_mtp) {
                common_params params_dft = params_base;
                bool measure_model_bytes = true;

                if (has_draft) {
                    const auto & params_spec = params_base.speculative.draft;
                    params_dft.devices               = params_spec.devices;
                    params_dft.model                 = params_spec.mparams;
                    params_dft.n_gpu_layers          = params_spec.n_gpu_layers;
                    params_dft.cache_type_k          = params_spec.cache_type_k;
                    params_dft.cache_type_v          = params_spec.cache_type_v;
                    params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
                } else {
                    // MTP draft context lives on the target model, only context+compute are new
                    measure_model_bytes = false;
                }

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                    cparams_dft.type_k   = params_base.speculative.draft.cache_type_k;
                    cparams_dft.type_v   = params_base.speculative.draft.cache_type_v;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes =
                            (measure_model_bytes ? dmd[j].mb.model : 0) +
                            dmd[j].mb.context +
                            dmd[j].mb.compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_INF("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_ERR("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_INF("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            auto cparams = common_context_params_to_llama(params_dft);

            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
            if (spec_mtp) {
                cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }

            // note: for small models maybe we can set this to the maximum possible draft from all speculative types
            //       the extra memory for small models is likely negligible?
            cparams.n_rs_seq = 0;
            ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        } else if (std::find(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                             COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end()) {
            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            auto cparams_mtp = common_context_params_to_llama(params_base);
            cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cparams_mtp.type_k   = params_base.speculative.draft.cache_type_k;
            cparams_mtp.type_v   = params_base.speculative.draft.cache_type_v;
            cparams_mtp.n_rs_seq = 0;

            ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        }

        if (has_mmproj) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (spec) {
            SRV_INF("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_INF("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_INF("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_INF("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_INF("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_INF("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_INF("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_INF("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        // AUTO disk prompt/KV cache (invariant 1): compute the model fingerprint and build the
        // longest-prefix index ONCE, header-only — but ONLY when the feature is enabled. When OFF
        // this is a single boolean test and nothing else (no fingerprint, no scan, no allocation).
        if (auto_cache_enabled()) {
            cur_fp = auto_compute_fingerprint();
            auto_index_scan();
            SRV_INF("auto disk prompt cache enabled: indexed %zu prefix boundaries from %s (block=%d)\n",
                    auto_idx.by_boundary.size(), params_base.slot_save_path.c_str(), params_base.slot_save_block);
        }

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (!params_base.kv_unified) {
                SRV_WRN("%s", "--cache-idle-slots requires --kv-unified, disabling\n");
                params_base.cache_idle_slots = false;
            } else if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                SRV_INF("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        // populate UI settings (from either new ui_config_json or deprecated webui_config_json)
        {
            const std::string & cfg = !params_base.ui_config_json.empty()
                ? params_base.ui_config_json
                : params_base.webui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                    json_webui_settings = json_settings; // deprecated: keep in sync
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            // Second auto-save site for when cache_idle_slots is OFF (the idle-flush path that calls
            // slot_save_and_clear -> auto_save never runs). Here get_available_slot just picked `ret`
            // for a new task and `update_cache` signals its prior KV is about to be discarded, so we
            // persist it before the prompt_save/prompt_load below overwrites it. Mutually exclusive
            // with the primary site via !cache_idle_slots, so no double-save. Reads `update_cache`
            // BEFORE the `&& prompt_cache` narrowing so disk save works without --cache-ram. The
            // callee carries all correctness gates; `ret` is idle so this never stalls generation.
            if (auto_cache_enabled() && !params_base.cache_idle_slots && update_cache) {
                auto_save_slot_if_useful(*ret);
            }

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_INF("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_INF("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // A new task is being assigned to this slot: its prompt may differ from whatever produced
        // the slot's current `logits_last` (even at the same token count). Invalidate the capture so
        // a SLOT_SAVE issued on the new prompt can never serialize a stale, mismatched distribution.
        // The stamp is re-established only by a real decode of the new prompt (the capture point).
        slot.logits_last.clear();
        slot.logits_last_n_tokens = -1;

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                slot_save_and_clear(s);
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    // persist this slot's last-token logits as a sidecar (FULL/recurrent
                    // only). Best-effort — a missing/failed sidecar simply disables the regenerate
                    // fast-path for this snapshot. NOT folded into res->n_bytes (that contract stays
                    // "state-file bytes only").
                    //
                    // CRITICAL consistency guard: only write the sidecar when the captured logits
                    // provably belong to the EXACT state being saved, i.e. logits_last_n_tokens ==
                    // token_count. This blocks every stale-logits path (restore-then-save with no
                    // intervening decode; a spec-decode step that skipped the capture; a distribution
                    // left over from a prior task on this slot object) from persisting a sidecar that
                    // does not match the saved state — which would otherwise emit a wrong first token
                    // on a later regenerate with nothing to catch it.
                    if (nwrite > 0 && ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                        if (slot->logits_last_n_tokens == (int32_t) token_count && !slot->logits_last.empty()) {
                            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                            const size_t nwrite_logits =
                                slot_logits_write(filepath, slot->logits_last, nv, (uint32_t) token_count);
                            if (nwrite_logits == 0) {
                                SLT_WRN(*slot, "%s", "failed to write logits sidecar; regenerate fast-path disabled for this snapshot\n");
                            }
                        } else {
                            SLT_DBG(*slot, "no matching captured logits for this state (stamp=%d, token_count=%zu); sidecar omitted\n",
                                    slot->logits_last_n_tokens, token_count);
                        }
                    }

                    // enforce the bounded slot-save store (LRU by mtime). If this single
                    // snapshot exceeds the byte cap, reject the save instead of evicting everything.
                    if (nwrite > 0 &&
                        (params_base.slot_save_max_count > 0 || params_base.slot_save_max_bytes > 0)) {
                        bool oversized = false;
                        slot_save_enforce_limits(params_base.slot_save_path,
                                                 params_base.slot_save_max_count,
                                                 params_base.slot_save_max_bytes,
                                                 filepath, oversized);
                        if (oversized) {
                            send_error(task,
                                       "slot snapshot exceeds --slot-save-max-mb; save rejected",
                                       ERROR_TYPE_INVALID_REQUEST);
                            break;
                        }
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    // Shared restore body (also used by the transparent auto-restore path): loads the
                    // state file into seq slot->id, sets just_restored + restored_logits, rebuilds the
                    // FULL-model checkpoint. On a load failure the slot seq is cleared and we error.
                    size_t token_count = 0;
                    size_t nread = 0;
                    if (!do_slot_restore(*slot, filepath, &token_count, &nread)) {
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                common_context_seq_rm (ctx_tgt, slot.id, n_keep            , n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                if (ctx_dft) {
                    common_context_seq_rm (ctx_dft.get(), slot.id, n_keep            , n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        }

        // generate the actual drafts (if any)
        {
            common_speculative_draft(spec.get());
        }

        // make checkpoints if needed
        for (auto * slot_ptr : drafting) {
            auto & slot = *slot_ptr;

            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }

                common_context_seq_rm(ctx_dft.get(), slot.id, ckpt.pos_max + 1, -1);
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft.get()));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }
            }
        }

        // update the batch with the sampled/drafted tokens
        for (auto * slot_ptr : generating) {
            auto & slot = *slot_ptr;

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // ===== AUTO-RESTORE: cold/cross-process KV reuse from disk (opt-in) ==========
                                // If the in-memory match (n_past) is POOR and the disk index holds a snapshot
                                // whose persisted tokens are a verified, fingerprint-matching, longer prefix of
                                // this request, restore it INTO the slot and RECOMPUTE n_past so all downstream
                                // machinery runs unchanged — agnostic to HOW the tokens arrived. Gated on the
                                // PER-REQUEST has_media() (not the server-wide has_mtmd) so an --mmproj server
                                // still caches its text-only turns: a no-media prompt has no NULL placeholders,
                                // so get_text_tokens() equals the full token-id prefix and does not trip the
                                // get_tokens() GGML_ASSERT(!has_mtmd); a turn carrying an image (and every turn
                                // after it) is skipped. auto_restore_into_slot byte-verifies tokens + fingerprint
                                // and falls back to a normal prefill on any mismatch/failure (invariants 2/3/4).
                                // media-prefix caching intentionally unsupported: token-ids cannot identify image content.
                                if (auto_cache_enabled()
                                        && slot.task->need_sampling()        // generative only (not embed/rerank)
                                        && !slot.prompt.tokens.has_media()   // no media in THIS request
                                        && slot.alora_invocation_start <= 0      // aLoRA caching bound (mirror below)
                                        && are_lora_equal(slot.lora, params_base.lora_adapters)) { // fp captures global LoRA (invariant 3)
                                    // get_text_tokens() (not get_tokens()): media-safe accessor that never asserts
                                    // under has_mtmd and, for this no-media prompt, equals the full token-id prefix.
                                    const llama_tokens req = input_tokens.get_text_tokens();
                                    if (auto cand = auto_index_lookup(req)) {
                                        // auto_restore_into_slot may CLEAR the slot
                                        // (KV seq + prompt.tokens) and then have do_slot_restore FAIL
                                        // (corrupt/short .bin, KV-capacity exceeded, racing LRU eviction
                                        // deleting the file mid-read). In that case the slot tokens are now
                                        // empty. We therefore RECOMPUTE n_past UNCONDITIONALLY after any
                                        // attempt — not only on success — so a cleared-but-failed restore
                                        // falls back to n_past=0 (clean cold prefill) instead of carrying a
                                        // stale n_keep_mem>0 into keep_first() on an empty token vector
                                        // (which would GGML_ASSERT/abort). The recompute is harmless on the
                                        // early-return-before-clear paths (margin/fp/verify rejects): those
                                        // leave prompt.tokens untouched, so the LCP is identical to before.
                                        auto_restore_into_slot(slot, *cand, req, (int) n_past);
                                        n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
                                    }
                                }
                                // ===== end AUTO-RESTORE =====================================================

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                common_context_seq_rm (ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - 1);

                            // ===== restore-continue (regenerate fast-path) ==============================
                            // A just-restored FULL/recurrent slot receiving the EXACT restored tokens (no
                            // suffix) cannot rewind its memory: the normal path would re-decode into the
                            // already-occupied sequence and crash (the pure-recurrent gate at the relaxed
                            // checkpoint predicate below is FALSE for this case, so just_restored would never
                            // be consumed and [TAG_PROMPT_LOGITS] would decrement n_past and re-decode into
                            // the occupied sequence). Detect that case here, BEFORE the n_past>0 guard, and:
                            //   - if we have the saved next-token logits: emit the first token with NO decode,
                            //     keeping the full restored state, then continue normal autoregression;
                            //   - otherwise: fall back to a SAFE clear-then-reprefill (never crash).
                            // Gated so it is unreachable for non-recurrent models, with-suffix requests,
                            // non-generative slots, and multimodal (already excluded by check_no_mtmd at
                            // save/restore). With-suffix restore reuse is left entirely to the unchanged
                            // relaxed-predicate path below.
                            // NOTE: cache_prompt==false sets n_past=0 above, so this gate (which
                            // requires n_past == task->n_tokens()) is naturally not entered for a
                            // no-cache request; that case safely takes the normal full-clear +
                            // reprefill path (common_context_seq_rm at [p0,-1) empties the restored
                            // sequence first), so regenerate degrades to a cold reprefill, never a crash.
                            // No `n_past < n_ctx` clause: the fast path emits with NO decode so it
                            // needs no free context slot; a full-n_ctx no-suffix restore is handled
                            // here rather than falling through to a zero-token-added crash window.
                            if (slot.just_restored &&
                                ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
                                slot.task->need_sampling() &&
                                slot.alora_invocation_start <= 0 &&
                                n_past == slot.task->n_tokens() &&
                                n_past == (int) slot.prompt.n_tokens()) {

                                slot.just_restored = false; // one-shot consume (this path owns it)

                                if (!slot.restored_logits.empty()) {
                                    // --- fast path: emit first token from saved logits, no decode ---
                                    slot.n_prompt_tokens_cache     = n_past; // entire prompt "reused"
                                    slot.n_prompt_tokens_processed = 0;      // prompt_n = 0 => observable reuse signal

                                    // prime the sampler over the full restored prompt (penalties/grammar
                                    // history), exactly as the normal DONE_PROMPT transition (init_sampler) would.
                                    slot.n_decoded = 0;
                                    slot.init_sampler();

                                    slot.state   = SLOT_STATE_GENERATING;
                                    slot.i_batch = -1;

                                    // rebuild the (cold, unsaved) draft context for the restored prompt,
                                    // mirroring the normal prompt-done transition.
                                    if (slot.can_speculate()) {
                                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                                    }

                                    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                                    const llama_token id = common_sampler_sample_from_logits(
                                            slot.smpl.get(), slot.restored_logits.data(), nv, /*grammar_first=*/false);
                                    slot.restored_logits.clear(); // consumed

                                    common_sampler_accept(slot.smpl.get(), id, true);

                                    // mirror the generation accounting from the normal sample path
                                    const int64_t t_current = ggml_time_us();
                                    slot.n_decoded += 1;
                                    slot.t_start_generation  = t_current;
                                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                                    metrics.on_prompt_eval(slot);
                                    slot.t_token_generation  = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                                    if (slot.task->params.stream) {
                                        // mirror the normal prompt-start streaming signal exactly so a
                                        // return_progress client still gets its initial 0% progress event
                                        if (slot.task->params.return_progress) {
                                            send_partial_response(slot, {}, true);
                                        } else {
                                            // signal HTTP to send the headers (200 status)
                                            send_partial_response(slot, {}, false, true);
                                        }
                                    }

                                    completion_token_output result;
                                    result.tok  = id;
                                    // inline of the accept_special_token lambda (defined later in this method,
                                    // out of scope here): keep special tokens iff the server allows specials
                                    // or the request explicitly preserves this token.
                                    const bool keep_special =
                                        params_base.special ||
                                        slot.task->params.sampling.preserved_tokens.find(result.tok) !=
                                            slot.task->params.sampling.preserved_tokens.end();
                                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, keep_special);
                                    result.prob         = 1.0f;

                                    // First-token logprobs (n_probs>0): the post-sampling variant reads the
                                    // candidate set (cur_p), which common_sampler_sample_from_logits leaves
                                    // populated — so we can serve it exactly as the normal path does. The
                                    // pre-sampling variant reads raw ctx logits at a decode index we bypass
                                    // here; idx=-1 is passed but populate_token_probs() only uses idx in that
                                    // branch, so we restrict the call to post_sampling to stay correct.
                                    if (slot.task->params.sampling.n_probs > 0 && slot.task->params.post_sampling_probs) {
                                        populate_token_probs(slot, result, /*post_sampling=*/true, params_base.special, /*idx=*/-1);
                                    }

                                    if (!process_token(result, slot)) {
                                        slot.print_timings();
                                        send_final_response(slot);
                                        metrics.on_prediction(slot);
                                        slot.release();
                                    }

                                    SLT_INF(slot, "%s", "restore-continue: emitted first token from saved logits (prompt_n=0)\n");
                                    continue; // skip ALL prompt-batch building for this slot this iteration
                                }

                                // --- safe fallback: no valid sidecar -> clear restored seq, then reprefill ---
                                // FULL models support full-sequence removal; clearing first guarantees the
                                // subsequent reprefill writes into an EMPTY sequence instead of re-decoding
                                // into the already-occupied restored state (which is the crash being fixed).
                                SLT_WRN(slot, "%s", "restore-continue: no saved logits; clearing restored state and re-prefilling\n");
                                llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
                                slot.prompt.tokens.clear();
                                slot.prompt.checkpoints.clear();
                                n_past   = 0;
                                pos_next = 0; // mirror the do_reset path; keep the stale full-length value from leaking into the checkpoint-erase loop below
                                // fall through to the normal guard below with an empty sequence (safe)
                            }
                            // ===== end restore-continue =================================================

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    // reuse a tail checkpoint (e.g. restored from disk) when genuinely-new tokens follow,
                                    // which supply the required logits so the >=1-token guarantee still holds
                                    const bool slot_was_restored = slot.just_restored; slot.just_restored = false;
                                    const bool has_new_suffix = (size_t) slot.task->n_tokens() > (size_t) n_past;
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            return cur.pos_min == 0 || cur.pos_min < pos_min_thold || (slot_was_restored && has_new_suffix && cur.pos_min == pos_min_thold);
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    const int64_t t_current = ggml_time_us();
                    slot.t_prompt_processing = (t_current - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    common_context_seq_rm(ctx_tgt, slot.id, p0, -1);
                    if (ctx_dft) {
                        common_context_seq_rm(ctx_dft.get(), slot.id, p0, -1);
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx_tgt, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (ctx_dft) {
                            // TODO: in the future, figure out how to infuse target embeddings to the images
                            //       for now, we skip this for simplicity
                            //       maybe we simply need to call `common_speculative_process()` on the mtmd batches in the `process_chunk` above?
                            res = input_tokens.process_chunk(ctx_dft.get(), mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                            if (res != 0) {
                                GGML_ABORT("failed to process multi-modal data on draft context\n");
                            }
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const int32_t n_before_user = slot.task->params.n_before_user;
                    const bool n_before_user_known = n_before_user > 0;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_pre_norm into ctx_dft.
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // stop the prompt batch exactly before the latest user input, so a checkpoint
                        // can be created after the previous messages
                        if (n_before_user_known &&
                            slot.prompt.n_tokens() == n_before_user) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints
                        if (!n_before_user_known && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // checkpoints are created before the current batch is decoded, so
                    // their token position is the batch start rather than the prompt end
                    const int32_t n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    {
                        const bool is_on_user =
                            n_before_user_known &&
                            n_tokens_start == n_before_user;

                        const bool is_after_user =
                            n_before_user_known &&
                            n_tokens_start > n_before_user;

                        const bool is_allowed =
                            !n_before_user_known ||
                            is_on_user ||
                            (is_after_user && near_prompt_end);

                        if (do_checkpoint && !is_allowed) {
                            do_checkpoint = false;
                        }
                    }

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx_tgt, batch_view);

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            //       for now, always re-evaluate for simplicity
            //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
            //
            // | spec type   | need re-eval |
            // | ---         | ---          |
            // | draft model | no           | because the draft model does not use embeddings from the target
            // | MTP (std)   | yes          |
            // | MTP Gemma4  | no           | because the KV cache is shared
            // | Eagle3      | yes          |
            // | DFlash      | yes          | https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4405406982
            //
            // note: this logic is now moved in `common_speculative_process()`
            //       keeping the sketch here until for a bit, until the logic is finalized
            //
            //if (ctx_dft) {
            //    // TODO: update as needed for MTP, Eagle3, etc.
            //    const bool need_tgt_embd = false;

            //    if (need_tgt_embd) {
            //        llama_synchronize(ctx_tgt);
            //    }

            //    // the logic here varies depending on the speculative decoding method
            //    //  - some draft contexts require embeddings from the target context, others don't
            //    //  - some draft contexts involve an encoder step to transform the target embeddings to draft embeddings
            //    // TODO: extract this in a function ?
            //    {
            //        // TODO: hook the embeddings from the last target batch here
            //        if (llama_model_has_encoder(model_dft.get())) {
            //            //llama_encode(ctx_dft, ...);

            //            GGML_ABORT("not implemented yet\n");
            //        }

            //        const int ret = llama_decode(ctx_dft.get(), batch_view);

            //        if (ret != 0) {
            //            SRV_ERR("failed to decode draft batch, ret = %d\n", ret);

            //            // TODO: handle error
            //            break;
            //        }
            //    }
            //}
            if (!common_speculative_process(spec.get(), batch_view)) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                break;
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx_tgt);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);

                // capture this slot's last-token full-vocab logits for a possible disk
                // save. Cost: one ~n_vocab*4-byte copy per decoded token, incurred ONLY on
                // FULL/recurrent models AND only when slot saving is enabled (--slot-save-path set);
                // attention models and servers without slot-save pay nothing at all. The copy is
                // unavoidable for correctness: ctx logits are overwritten by the next slot's decode,
                // so a lazy read at SLOT_SAVE would be wrong under --parallel>1. Captured per-slot
                // from this slot's own tok_idx so it is correct for any N/interleave (never read
                // from the shared ctx at save time). common_sampler_sample already synchronized the
                // context above, so llama_get_logits_ith is valid here.
                if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL && !params_base.slot_save_path.empty()) {
                    const float * lg = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                    if (lg) {
                        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
                        slot.logits_last.assign(lg, lg + nv);
                        // Stamp the capture with the token count of the state it corresponds to.
                        // At this point process_token() has NOT yet appended the just-sampled token,
                        // so prompt.tokens.size() is exactly the length whose final token produced
                        // these logits — i.e. it matches the token_count a SLOT_SAVE would record.
                        slot.logits_last_n_tokens = (int32_t) slot.prompt.tokens.size();
                    } else {
                        // capture failed -> invalidate so a later save never serializes stale logits
                        slot.logits_last.clear();
                        slot.logits_last_n_tokens = -1;
                    }
                }

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }

                slot.print_timings_tg();
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }

                // save the original draft size
                const size_t n_draft = slot.spec_draft.size();

                GGML_ASSERT(n_draft > 0);

                // verify and try to accept the draft
                {
                    // save the sampler sampler state in case we need to restore it
                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                    auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                    const bool use_ckpt_tgt =
                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                       (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                    // check for partial draft acceptance
                    if (n_rollback > 0) {
                        if (use_ckpt_tgt) {
                            if (trace > 0) {
                                SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                            }

                            // partial acceptance is not supported by the context -> truncate the draft and restore the state
                            slot.spec_draft = std::move(accepted);

                            const auto & ckpt = slot.spec_ckpt;

                            SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                            {
                                ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                            }

                            if (slot.ctx_dft) {
                                ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                            }

                            slot.prompt.tokens.keep_first(ckpt.n_tokens);
                            slot.smpl = std::move(smpl_save);

                            continue;
                        }
                    }

                    if (trace > 0) {
                        SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                    }

                    common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                    slot.spec_draft = std::move(accepted);
                }

                const int64_t t_current = ggml_time_us();

                const auto ids = std::move(slot.spec_draft);

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                // add accepted tokens to the prompt
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                slot.sampled = ids.back(); // last accepted token
                SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
                if (slot.ctx_dft) {
                    common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
                }

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    slot.n_decoded += 1;

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                slot.print_timings_tg();

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* json_webui_settings    */ impl->json_webui_settings,  // Deprecated
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}

// compute the number of tokens before the last user message in the prompt
static int32_t prompt_get_n_before_user(
        const json & message_spans,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const llama_vocab * vocab,
        mtmd_context * mctx) {
    int32_t result = -1;
    int32_t byte_pos = -1;

    for (const auto & span : message_spans) {
        const std::string role = json_value(span, "role", std::string());

        if (role == "user") {
            byte_pos = json_value(span, "pos", -1);
        }
    }

    if (byte_pos >= 0) {
        GGML_ASSERT((size_t) byte_pos <= prompt.size());

        const std::string prefix = prompt.substr(0, (size_t) byte_pos);

        const std::string marker = get_media_marker();
        size_t n_prefix_media = 0;
        for (size_t pos = 0; (pos = prefix.find(marker, pos)) != std::string::npos; pos += marker.size()) {
            n_prefix_media++;
        }

        GGML_ASSERT(n_prefix_media <= files.size());

        if (mctx != nullptr && n_prefix_media > 0) {
            // TODO: this makes a copy - avoid it
            std::vector<raw_buffer> prefix_files(files.begin(), files.begin() + n_prefix_media);

            result = (int32_t) process_mtmd_prompt(mctx, prefix, prefix_files).size();
        } else {
            result = (int32_t) tokenize_input_prompts(vocab, nullptr, prefix, true, true)[0].size();
        }

        SRV_TRC("message_spans: last user message: byte_pos=%d, media=%zu, n_before_user=%d\n",
                byte_pos, n_prefix_media, result);
    }

    return result;
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            const auto message_spans = json_value(data, "message_spans", json::array());
            if (prompt.is_string() && message_spans.is_array()) {
                task.params.n_before_user =
                    prompt_get_n_before_user(
                        message_spans,
                        prompt.get<std::string>(),
                        files,
                        ctx_server.vocab,
                        ctx_server.mctx);
            }

            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            // New keys
            { "ui",                           params.ui },
            { "ui_settings",                  meta->json_ui_settings },
            // Deprecated: use ui/ui_settings instead (kept for backward compat)
            { "webui",                        params.webui },
            { "webui_settings",               meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy || params.webui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}
