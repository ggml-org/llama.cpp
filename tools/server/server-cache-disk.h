#pragma once

#include "server-common.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct common_params;
struct llama_context;

// disk-backed prompt cache: a cold tier below the in-RAM server_prompt_cache
//
// each entry is one file in a flat directory, named after the exact token prefix it contains:
//
//     {compat_hash8}-{n_tokens}-{chain_hash16}.kvc
//
//   - compat_hash: hash of everything that invalidates a KV state (model file, mmproj, loras,
//     cache types, rope params, ...) - see server_cache_disk_compat_hash()
//   - chain_hash: chained hash over the first n_tokens tokens, so a filename identifies an exact
//     prefix and lookup is a single rolling-hash pass over the incoming prompt plus an index probe
//
// file contents mirror what the RAM cache holds for the target context:
//
//     header | server_tokens::serialize() bytes | llama_state_seq_get_data (FLAGS_NONE) bytes

struct server_cache_disk_file {
    std::string name; // filename inside the cache directory

    uint64_t chain_hash = 0;
    uint32_t n_tokens   = 0;
    uint64_t n_bytes    = 0;
    int64_t  mtime      = 0; // only used for relative ordering during eviction
};

struct server_prompt_cache_disk {
    server_prompt_cache_disk(const std::string & dir, uint64_t compat_hash, bool has_mtmd, int32_t limit_mib, bool write_through);

    enum load_status {
        LOAD_OK,             // state restored into the sequence
        LOAD_MISS,           // file unusable (corrupt, collision, ...) - sequence untouched
        LOAD_FAIL_SEQ_DIRTY, // restore failed mid-way - the sequence was cleared and must be re-filled
    };

    // largest exact-prefix hit for the first n_max tokens, or nullptr on miss
    const server_cache_disk_file * lookup(const server_tokens & tokens, size_t n_max) const;

    // restore the state from a file into sequence id_slot of ctx
    // on LOAD_OK, tokens_out receives the cached token list (an exact prefix of tokens_new)
    load_status load(server_cache_disk_file file, const server_tokens & tokens_new, llama_context * ctx, int32_t id_slot, server_tokens & tokens_out);

    // write one entry; deduplicates against existing files and enforces the size budget
    bool store(const server_tokens & tokens, const std::vector<uint8_t> & state_main);

    size_t n_files() const;
    size_t n_bytes_total() const { return total_bytes; }

    const bool write_through;

private:
    void scan_dir();

    server_cache_disk_file * find_file(uint32_t n_tokens, uint64_t chain_hash);

    void touch (const server_cache_disk_file & file);        // bump mtime so eviction treats it as fresh
    void forget(const server_cache_disk_file & file);        // drop from the index without touching the filesystem
    void remove_file(const server_cache_disk_file & file);   // delete from disk and drop from the index

    // delete oldest-mtime files (ours and foreign alike) while over the size budget
    void enforce_budget(const std::string & name_protected);

    const std::string dir;
    const uint64_t    compat_hash;
    const bool        has_mtmd;
    const size_t      limit_bytes; // 0 = no limit

    // n_tokens -> chain_hash -> file, for our compat hash only
    std::map<uint32_t, std::unordered_map<uint64_t, server_cache_disk_file>> index;

    // .kvc files with a different compat hash prefix - never opened, but counted toward the budget
    std::vector<server_cache_disk_file> foreign;

    size_t total_bytes = 0; // ours + foreign

    // (n_tokens, chain_hash) prefixes known to be covered by a file written or loaded this
    // session - lets store() skip prefixes of already-persisted prompts
    std::unordered_set<uint64_t> covered;

    uint32_t tmp_counter = 0;
};

// hash of everything that invalidates a saved KV state for the current server configuration
uint64_t server_cache_disk_compat_hash(const common_params & params);
