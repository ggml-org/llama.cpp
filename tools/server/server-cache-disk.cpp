#include "server-cache-disk.h"

#include "common.h"
#include "llama.h"

#include "xxhash/xxhash.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

namespace {

constexpr uint32_t SERVER_CACHE_DISK_MAGIC   = 0x3143564B; // "KVC1"
constexpr uint32_t SERVER_CACHE_DISK_VERSION = 1;

// seed for the chained prefix hash - changing it invalidates all filenames
constexpr uint64_t SERVER_CACHE_DISK_CHAIN_SEED = 0x6b7663636861696eULL;

struct server_cache_disk_file_header {
    uint32_t magic       = SERVER_CACHE_DISK_MAGIC;
    uint32_t version     = SERVER_CACHE_DISK_VERSION;
    uint64_t compat_hash = 0; // full 64-bit value (the filename only carries the low 32 bits)
    uint64_t chain_hash  = 0;
    uint32_t n_tokens    = 0;
    uint32_t pad         = 0;
    uint64_t tokens_size = 0; // bytes of the server_tokens::serialize() section
    uint64_t state_size  = 0; // bytes of the llama_state_seq_get_data section
};

static_assert(sizeof(server_cache_disk_file_header) == 48, "unexpected header size");

std::string make_filename(uint64_t compat_hash, uint32_t n_tokens, uint64_t chain_hash) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%08x-%u-%016" PRIx64 ".kvc", (uint32_t) compat_hash, n_tokens, chain_hash);
    return buf;
}

bool parse_filename(const std::string & name, uint32_t & compat32, uint32_t & n_tokens, uint64_t & chain_hash) {
    if (sscanf(name.c_str(), "%8x-%u-%16" SCNx64 ".kvc", &compat32, &n_tokens, &chain_hash) != 3) {
        return false;
    }

    // reject padding/case/suffix variations by requiring the canonical spelling
    return name == make_filename(compat32, n_tokens, chain_hash);
}

int64_t file_mtime(const std::filesystem::path & path) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(path, ec);
    return ec ? 0 : (int64_t) t.time_since_epoch().count();
}

uint64_t covered_key(uint32_t n_tokens, uint64_t chain_hash) {
    const uint64_t buf[2] = { n_tokens, chain_hash };
    return XXH64(buf, sizeof(buf), 0);
}

// walk the chained hash over the token list, invoking cb(n, h) at every valid prefix boundary:
// after each text token and after each complete media chunk (never mid-chunk)
// returns true if the walk reached n_max
bool tokens_chain_hash_walk(const server_tokens & tokens, size_t n_max, const std::function<bool(size_t, uint64_t)> & cb) {
    uint64_t h = SERVER_CACHE_DISK_CHAIN_SEED;

    size_t i = 0;

    try {
        while (i < n_max) {
            const llama_token tok = tokens[i];

            if (tok == LLAMA_TOKEN_NULL) {
                // media chunk - fold in its content id instead of the placeholder token ids,
                // otherwise different images would hash identically
                const auto & chunk = tokens.find_chunk(i);

                const char * id    = mtmd_input_chunk_get_id(chunk.get());
                const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());

                if (id == nullptr || id[0] == '\0' || n_tok == 0 || i + n_tok > n_max) {
                    return false;
                }

                std::vector<uint8_t> buf;
                buf.reserve(5 + strlen(id));
                buf.push_back(0x01);
                for (int b = 0; b < 4; ++b) {
                    buf.push_back((uint8_t) (n_tok >> (8*b)));
                }
                buf.insert(buf.end(), id, id + strlen(id));

                h = XXH64(buf.data(), buf.size(), h);

                i += n_tok;
            } else {
                uint8_t buf[5] = { 0x00 };
                memcpy(buf + 1, &tok, sizeof(tok));

                h = XXH64(buf, sizeof(buf), h);

                i += 1;
            }

            if (!cb(i, h)) {
                return false;
            }
        }
    } catch (const std::exception & e) {
        SRV_WRN("failed to hash token list: %s\n", e.what());
        return false;
    }

    return true;
}

} // namespace

server_prompt_cache_disk::server_prompt_cache_disk(const std::string & dir_, uint64_t compat_hash, bool has_mtmd, int32_t limit_mib, bool write_through) :
        write_through(write_through),
        dir(dir_.empty() || dir_.back() == DIRECTORY_SEPARATOR ? dir_ : dir_ + DIRECTORY_SEPARATOR),
        compat_hash(compat_hash),
        has_mtmd(has_mtmd),
        limit_bytes(limit_mib < 0 ? 0 : 1024ull*1024ull*limit_mib) {
    scan_dir();
}

void server_prompt_cache_disk::scan_dir() {
    namespace fs = std::filesystem;

    std::error_code ec;

    for (const auto & ent : fs::directory_iterator(dir, ec)) {
        if (!ent.is_regular_file(ec)) {
            continue;
        }

        const std::string name = ent.path().filename().string();

        // leftover temporary files from a previous crash
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0 && name[0] == '.') {
            fs::remove(ent.path(), ec);
            continue;
        }

        uint32_t compat32  = 0;
        uint32_t n_tokens  = 0;
        uint64_t chain     = 0;

        if (!parse_filename(name, compat32, n_tokens, chain)) {
            continue;
        }

        server_cache_disk_file file;
        file.name       = name;
        file.chain_hash = chain;
        file.n_tokens   = n_tokens;
        file.n_bytes    = ent.file_size(ec);
        file.mtime      = file_mtime(ent.path());

        total_bytes += file.n_bytes;

        if (compat32 == (uint32_t) compat_hash) {
            index[n_tokens][chain] = std::move(file);
        } else {
            foreign.push_back(std::move(file));
        }
    }

    SRV_INF("disk prompt cache '%s': %zu usable entries, %zu from other configurations, %.3f MiB total (budget: %.3f MiB)\n",
            dir.c_str(), n_files(), foreign.size(), total_bytes / (1024.0 * 1024.0), limit_bytes / (1024.0 * 1024.0));
}

size_t server_prompt_cache_disk::n_files() const {
    size_t res = 0;

    for (const auto & [n, files] : index) {
        res += files.size();
    }

    return res;
}

server_cache_disk_file * server_prompt_cache_disk::find_file(uint32_t n_tokens, uint64_t chain_hash) {
    const auto it = index.find(n_tokens);
    if (it == index.end()) {
        return nullptr;
    }

    const auto it_file = it->second.find(chain_hash);

    return it_file == it->second.end() ? nullptr : &it_file->second;
}

const server_cache_disk_file * server_prompt_cache_disk::lookup(const server_tokens & tokens, size_t n_max) const {
    if (index.empty()) {
        return nullptr;
    }

    // no file can be longer than the largest indexed length - cap the walk
    n_max = std::min<size_t>(n_max, index.rbegin()->first);

    const server_cache_disk_file * best = nullptr;

    tokens_chain_hash_walk(tokens, n_max, [&](size_t n, uint64_t h) {
        const auto it = index.find((uint32_t) n);
        if (it != index.end()) {
            const auto it_file = it->second.find(h);
            if (it_file != it->second.end()) {
                best = &it_file->second;
            }
        }

        return true;
    });

    return best;
}

void server_prompt_cache_disk::touch(const server_cache_disk_file & file) {
    std::error_code ec;
    std::filesystem::last_write_time(dir + file.name, std::filesystem::file_time_type::clock::now(), ec);

    if (auto * f = find_file(file.n_tokens, file.chain_hash)) {
        f->mtime = file_mtime(dir + file.name);
    }
}

void server_prompt_cache_disk::forget(const server_cache_disk_file & file) {
    // copy the fields first - the reference may point into the index entry being erased
    const uint32_t n_tokens = file.n_tokens;
    const uint64_t chain    = file.chain_hash;
    const uint64_t n_bytes  = file.n_bytes;

    const auto it = index.find(n_tokens);
    if (it == index.end()) {
        return;
    }

    if (it->second.erase(chain) > 0) {
        total_bytes -= std::min<size_t>(total_bytes, n_bytes);
    }

    if (it->second.empty()) {
        index.erase(it);
    }
}

void server_prompt_cache_disk::remove_file(const server_cache_disk_file & file) {
    SRV_WRN("disk prompt cache: removing '%s'\n", file.name.c_str());

    std::error_code ec;
    std::filesystem::remove(dir + file.name, ec);

    forget(file);
}

void server_prompt_cache_disk::enforce_budget(const std::string & name_protected) {
    if (limit_bytes == 0) {
        return;
    }

    while (total_bytes > limit_bytes) {
        // find the oldest file, ours and foreign alike
        const server_cache_disk_file * oldest = nullptr;
        bool oldest_foreign = false;

        for (const auto & [n, files] : index) {
            for (const auto & [h, file] : files) {
                if (file.name != name_protected && (!oldest || file.mtime < oldest->mtime)) {
                    oldest = &file;
                    oldest_foreign = false;
                }
            }
        }

        for (const auto & file : foreign) {
            if (file.name != name_protected && (!oldest || file.mtime < oldest->mtime)) {
                oldest = &file;
                oldest_foreign = true;
            }
        }

        if (!oldest) {
            break;
        }

        SRV_INF("disk prompt cache: size %.3f MiB over budget %.3f MiB, evicting oldest entry '%s'\n",
                total_bytes / (1024.0 * 1024.0), limit_bytes / (1024.0 * 1024.0), oldest->name.c_str());

        if (oldest_foreign) {
            std::error_code ec;
            std::filesystem::remove(dir + oldest->name, ec);

            total_bytes -= std::min<size_t>(total_bytes, oldest->n_bytes);

            foreign.erase(foreign.begin() + (oldest - foreign.data()));
        } else {
            remove_file(*oldest);
        }
    }
}

bool server_prompt_cache_disk::store(const server_tokens & tokens, const std::vector<uint8_t> & state_main) {
    if (tokens.empty() || state_main.empty()) {
        return false;
    }

    std::vector<std::pair<size_t, uint64_t>> bounds;

    if (!tokens_chain_hash_walk(tokens, tokens.size(), [&](size_t n, uint64_t h) { bounds.emplace_back(n, h); return true; }) ||
        bounds.empty() || bounds.back().first != tokens.size()) {
        SRV_WRN("%s", "disk prompt cache: token list cannot be hashed, skipping\n");
        return false;
    }

    const uint32_t n_tokens = (uint32_t) tokens.size();
    const uint64_t chain    = bounds.back().second;

    if (auto * existing = find_file(n_tokens, chain)) {
        SRV_TRC("disk prompt cache: '%s' already exists, refreshing\n", existing->name.c_str());
        touch(*existing);
        return true;
    }

    if (covered.count(covered_key(n_tokens, chain)) > 0) {
        SRV_TRC(" - prompt with %u tokens is a prefix of an already persisted entry, skipping\n", n_tokens);
        return true;
    }

    std::vector<char> tok_data;
    try {
        tok_data = tokens.serialize();
    } catch (const std::exception & e) {
        SRV_WRN("disk prompt cache: failed to serialize tokens: %s\n", e.what());
        return false;
    }

    server_cache_disk_file_header header;
    header.compat_hash = compat_hash;
    header.chain_hash  = chain;
    header.n_tokens    = n_tokens;
    header.tokens_size = tok_data.size();
    header.state_size  = state_main.size();

    const std::string name = make_filename(compat_hash, n_tokens, chain);

    char tmp_buf[64];
    snprintf(tmp_buf, sizeof(tmp_buf), ".%08x-%u.tmp", (uint32_t) (uintptr_t) this, tmp_counter++);

    const std::string path_tmp = dir + tmp_buf;
    const std::string path     = dir + name;

    {
        std::ofstream out(path_tmp, std::ios::binary | std::ios::trunc);

        out.write((const char *) &header, sizeof(header));
        out.write(tok_data.data(), tok_data.size());
        out.write((const char *) state_main.data(), state_main.size());

        if (!out.good()) {
            SRV_ERR("disk prompt cache: failed to write '%s'\n", path_tmp.c_str());

            out.close();

            std::error_code ec;
            std::filesystem::remove(path_tmp, ec);

            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(path_tmp, path, ec);
    if (ec) {
        SRV_ERR("disk prompt cache: failed to rename '%s' to '%s': %s\n", path_tmp.c_str(), path.c_str(), ec.message().c_str());

        std::filesystem::remove(path_tmp, ec);

        return false;
    }

    server_cache_disk_file file;
    file.name       = name;
    file.chain_hash = chain;
    file.n_tokens   = n_tokens;
    file.n_bytes    = sizeof(header) + tok_data.size() + state_main.size();
    file.mtime      = file_mtime(path);

    total_bytes += file.n_bytes;

    index[n_tokens][chain] = std::move(file);

    for (const auto & [n, h] : bounds) {
        covered.insert(covered_key((uint32_t) n, h));
    }

    SRV_INF("disk prompt cache: saved prompt with %u tokens, %.3f MiB to '%s'\n",
            n_tokens, (sizeof(header) + tok_data.size() + state_main.size()) / (1024.0 * 1024.0), name.c_str());
    SRV_DBG("%s", "__TEST_TAG_CACHE_DISK_STORE__\n");

    enforce_budget(name);

    return true;
}

server_prompt_cache_disk::load_status server_prompt_cache_disk::load(
        server_cache_disk_file file, const server_tokens & tokens_new, llama_context * ctx, int32_t id_slot, server_tokens & tokens_out) {
    const std::string path = dir + file.name;

    std::error_code ec;
    const uint64_t n_bytes = std::filesystem::file_size(path, ec);

    if (ec) {
        // deleted by another process - not an error, just a miss
        forget(file);
        return LOAD_MISS;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        forget(file);
        return LOAD_MISS;
    }

    server_cache_disk_file_header header;
    in.read((char *) &header, sizeof(header));

    if (!in.good() ||
        header.magic    != SERVER_CACHE_DISK_MAGIC   ||
        header.version  != SERVER_CACHE_DISK_VERSION ||
        header.chain_hash  != file.chain_hash ||
        header.n_tokens    != file.n_tokens   ||
        header.tokens_size % sizeof(llama_token) != 0 ||
        sizeof(header) + header.tokens_size + header.state_size != n_bytes) {
        SRV_WRN("disk prompt cache: '%s' is corrupt\n", file.name.c_str());
        remove_file(file);
        return LOAD_MISS;
    }

    if (header.compat_hash != compat_hash) {
        // same low 32 bits, different configuration - leave the file for its owner
        SRV_WRN("disk prompt cache: '%s' belongs to a different configuration, ignoring\n", file.name.c_str());
        forget(file);
        return LOAD_MISS;
    }

    llama_tokens packed(header.tokens_size / sizeof(llama_token));
    in.read((char *) packed.data(), header.tokens_size);

    if (!in.good()) {
        SRV_WRN("disk prompt cache: '%s' is truncated\n", file.name.c_str());
        remove_file(file);
        return LOAD_MISS;
    }

    server_tokens loaded;
    try {
        loaded = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & e) {
        SRV_WRN("disk prompt cache: failed to deserialize tokens from '%s': %s\n", file.name.c_str(), e.what());
        remove_file(file);
        return LOAD_MISS;
    }

    // the filename hash only proves an exact prefix probabilistically - verify against the actual tokens
    if (loaded.size() != file.n_tokens ||
        loaded.get_common_prefix(tokens_new) != file.n_tokens ||
        !loaded.validate(ctx)) {
        SRV_WRN("disk prompt cache: token mismatch in '%s' (hash collision?)\n", file.name.c_str());
        remove_file(file);
        return LOAD_MISS;
    }

    std::vector<uint8_t> state;
    try {
        state.resize(header.state_size);
    } catch (const std::bad_alloc &) {
        SRV_ERR("disk prompt cache: failed to allocate %" PRIu64 " bytes for '%s'\n", header.state_size, file.name.c_str());
        return LOAD_MISS;
    }

    in.read((char *) state.data(), state.size());

    if (!in.good()) {
        SRV_WRN("disk prompt cache: '%s' is truncated\n", file.name.c_str());
        remove_file(file);
        return LOAD_MISS;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, state.data(), state.size(), id_slot, 0);
    if (n != state.size()) {
        SRV_WRN("disk prompt cache: failed to restore state from '%s' (%zu / %zu bytes)\n", file.name.c_str(), n, state.size());

        // the sequence may hold a partial state now - clear it and let the caller recover
        llama_memory_seq_rm(llama_get_memory(ctx), id_slot, -1, -1);

        return LOAD_FAIL_SEQ_DIRTY;
    }

    tokens_out = std::move(loaded);

    covered.insert(covered_key(file.n_tokens, file.chain_hash));

    touch(file);

    SRV_INF("disk prompt cache: restored prompt with %u tokens, %.3f MiB from '%s'\n",
            file.n_tokens, state.size() / (1024.0 * 1024.0), file.name.c_str());
    SRV_DBG("%s", "__TEST_TAG_CACHE_DISK_HIT__\n");

    return LOAD_OK;
}

//
// compat hash
//

namespace {

template <typename T>
void hash_pod(std::string & blob, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "hash_pod requires a POD type");
    blob.append((const char *) &value, sizeof(value));
}

void hash_str(std::string & blob, const std::string & value) {
    blob += value;
    blob += '\0';
}

// path + size + mtime: conservative, but never misses a changed file
void hash_file_meta(std::string & blob, const std::string & path) {
    hash_str(blob, path);

    std::error_code ec;

    const uint64_t size = path.empty() ? 0 : (uint64_t) std::filesystem::file_size(path, ec);
    hash_pod(blob, ec ? (uint64_t) 0 : size);

    hash_pod(blob, path.empty() ? (int64_t) 0 : file_mtime(path));
}

} // namespace

uint64_t server_cache_disk_compat_hash(const common_params & params) {
    std::string blob;

    // format versions
    hash_pod(blob, (uint32_t) SERVER_CACHE_DISK_VERSION);
    hash_pod(blob, (uint32_t) LLAMA_STATE_SEQ_VERSION);
    hash_pod(blob, (uint32_t) server_tokens::SERVER_TOKENS_STATE_VERSION);

    // model identity
    hash_file_meta(blob, params.model.path);
    hash_file_meta(blob, params.mmproj.path);

    for (const auto & la : params.lora_adapters) {
        hash_file_meta(blob, la.path);
        hash_pod(blob, la.scale);
    }

    // KV cache layout
    hash_pod(blob, (int32_t) params.cache_type_k);
    hash_pod(blob, (int32_t) params.cache_type_v);
    hash_pod(blob, (uint8_t) params.swa_full);

    // rope params change the KV content for the same tokens
    hash_pod(blob, params.rope_freq_base);
    hash_pod(blob, params.rope_freq_scale);
    hash_pod(blob, (int32_t) params.rope_scaling_type);
    hash_pod(blob, params.yarn_ext_factor);
    hash_pod(blob, params.yarn_attn_factor);
    hash_pod(blob, params.yarn_beta_fast);
    hash_pod(blob, params.yarn_beta_slow);
    hash_pod(blob, params.yarn_orig_ctx);

    return XXH64(blob.data(), blob.size(), 0);
}
