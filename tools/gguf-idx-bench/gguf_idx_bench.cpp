// Standalone prototype/benchmark: does caching the GGUF metadata+tensor TOC
// in a flat sidecar file meaningfully beat parsing the .gguf's own metadata
// section fresh via gguf_init_from_file()?
//
// This does NOT modify llama.cpp's loading path -- it's an isolated
// experiment against the public gguf.h API to test the hypothesis before
// deciding whether it's worth a real integration.
//
// Usage: gguf_idx_bench <model.gguf>

#include "gguf.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

using clock_t_ = std::chrono::steady_clock;

static double ms_since(clock_t_::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count();
}

struct TensorRec {
    std::string name;
    int32_t     type;
    int64_t     ne[4];
    uint64_t    offset;
};

struct KVRecScalar {
    std::string key;
    int32_t     type; // gguf_type, scalar types only (arrays skipped -- see note below)
    // stored as raw bytes of the widest scalar type we support (f64/u64)
    uint64_t    raw;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    const std::string idx_path = path + ".idxcache";

    // ---- Baseline: fresh gguf_init_from_file() ----
    auto t0 = clock_t_::now();
    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    const double fresh_parse_ms = ms_since(t0);
    if (!ctx) {
        std::fprintf(stderr, "gguf_init_from_file failed\n");
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const int64_t n_kv      = gguf_get_n_kv(ctx);
    std::printf("fresh gguf_init_from_file(): %.3f ms  (%lld kv, %lld tensors)\n",
                fresh_parse_ms, (long long)n_kv, (long long)n_tensors);

    // ---- Extract what we'd cache: tensor TOC + scalar KVs ----
    // (Array-valued KVs -- the big tokenizer.ggml.tokens/merges/token_type
    // arrays -- are a separate downstream cost (tokenizer init, measured at
    // ~34-50ms elsewhere) and deliberately excluded here: this experiment is
    // testing the *model-loading* metadata/TOC path specifically, the direct
    // analog of what HK's fixed-size TOC targets.)
    auto t1 = clock_t_::now();
    std::vector<TensorRec> tensors;
    tensors.reserve(n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        TensorRec r;
        r.name = gguf_get_tensor_name(ctx, i);
        r.type = (int32_t) gguf_get_tensor_type(ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        for (int d = 0; d < 4; d++) r.ne[d] = ne[d];
        r.offset = gguf_get_tensor_offset(ctx, i);
        tensors.push_back(std::move(r));
    }
    std::vector<KVRecScalar> kvs;
    for (int64_t i = 0; i < n_kv; i++) {
        enum gguf_type t = gguf_get_kv_type(ctx, i);
        if (t == GGUF_TYPE_ARRAY || t == GGUF_TYPE_STRING) continue; // handled separately/skipped
        KVRecScalar r;
        r.key  = gguf_get_key(ctx, i);
        r.type = (int32_t) t;
        r.raw  = 0;
        switch (t) {
            case GGUF_TYPE_UINT8:   r.raw = gguf_get_val_u8(ctx, i);  break;
            case GGUF_TYPE_INT8:    r.raw = (uint8_t)  gguf_get_val_i8(ctx, i);  break;
            case GGUF_TYPE_UINT16:  r.raw = gguf_get_val_u16(ctx, i); break;
            case GGUF_TYPE_INT16:   r.raw = (uint16_t) gguf_get_val_i16(ctx, i); break;
            case GGUF_TYPE_UINT32:  r.raw = gguf_get_val_u32(ctx, i); break;
            case GGUF_TYPE_INT32:   r.raw = (uint32_t) gguf_get_val_i32(ctx, i); break;
            case GGUF_TYPE_FLOAT32: { float f = gguf_get_val_f32(ctx, i); std::memcpy(&r.raw, &f, sizeof(f)); break; }
            case GGUF_TYPE_UINT64:  r.raw = gguf_get_val_u64(ctx, i); break;
            case GGUF_TYPE_INT64:   r.raw = (uint64_t) gguf_get_val_i64(ctx, i); break;
            case GGUF_TYPE_FLOAT64: { double d = gguf_get_val_f64(ctx, i); std::memcpy(&r.raw, &d, sizeof(d)); break; }
            case GGUF_TYPE_BOOL:    r.raw = gguf_get_val_bool(ctx, i) ? 1 : 0; break;
            default: continue;
        }
        kvs.push_back(std::move(r));
    }
    const double extract_ms = ms_since(t1);
    std::printf("extract TOC+scalar-KVs from parsed ctx: %.3f ms  (%zu tensors, %zu scalar kv)\n",
                extract_ms, tensors.size(), kvs.size());

    gguf_free(ctx);

    // ---- Write sidecar cache ----
    auto t2 = clock_t_::now();
    {
        std::ofstream out(idx_path, std::ios::binary | std::ios::trunc);
        const uint32_t magic = 0x47494458; // "GIDX"
        const uint32_t version = 1;
        out.write((const char*)&magic, 4);
        out.write((const char*)&version, 4);
        uint64_t nt = tensors.size();
        out.write((const char*)&nt, 8);
        for (auto& r : tensors) {
            uint32_t nlen = (uint32_t) r.name.size();
            out.write((const char*)&nlen, 4);
            out.write(r.name.data(), nlen);
            out.write((const char*)&r.type, 4);
            out.write((const char*)r.ne, sizeof(r.ne));
            out.write((const char*)&r.offset, 8);
        }
        uint64_t nk = kvs.size();
        out.write((const char*)&nk, 8);
        for (auto& r : kvs) {
            uint32_t klen = (uint32_t) r.key.size();
            out.write((const char*)&klen, 4);
            out.write(r.key.data(), klen);
            out.write((const char*)&r.type, 4);
            out.write((const char*)&r.raw, 8);
        }
    }
    const double write_ms = ms_since(t2);
    std::printf("write sidecar cache '%s': %.3f ms\n", idx_path.c_str(), write_ms);

    // ---- Read sidecar cache back (the actual "fast path" being tested) ----
    auto t3 = clock_t_::now();
    std::vector<TensorRec> tensors2;
    std::vector<KVRecScalar> kvs2;
    {
        std::ifstream in(idx_path, std::ios::binary);
        uint32_t magic, version;
        in.read((char*)&magic, 4);
        in.read((char*)&version, 4);
        uint64_t nt;
        in.read((char*)&nt, 8);
        tensors2.resize(nt);
        for (uint64_t i = 0; i < nt; i++) {
            uint32_t nlen;
            in.read((char*)&nlen, 4);
            tensors2[i].name.resize(nlen);
            in.read(&tensors2[i].name[0], nlen);
            in.read((char*)&tensors2[i].type, 4);
            in.read((char*)tensors2[i].ne, sizeof(tensors2[i].ne));
            in.read((char*)&tensors2[i].offset, 8);
        }
        uint64_t nk;
        in.read((char*)&nk, 8);
        kvs2.resize(nk);
        for (uint64_t i = 0; i < nk; i++) {
            uint32_t klen;
            in.read((char*)&klen, 4);
            kvs2[i].key.resize(klen);
            in.read(&kvs2[i].key[0], klen);
            in.read((char*)&kvs2[i].type, 4);
            in.read((char*)&kvs2[i].raw, 8);
        }
    }
    const double cached_read_ms = ms_since(t3);
    std::printf("read sidecar cache (fast path): %.3f ms  (%zu tensors, %zu scalar kv)\n",
                cached_read_ms, tensors2.size(), kvs2.size());

    // ---- Sanity check the cache round-trips correctly ----
    bool ok = tensors2.size() == tensors.size() && kvs2.size() == kvs.size();
    for (size_t i = 0; ok && i < tensors.size(); i++) {
        if (tensors2[i].name != tensors[i].name || tensors2[i].type != tensors[i].type ||
            tensors2[i].offset != tensors[i].offset ||
            memcmp(tensors2[i].ne, tensors[i].ne, sizeof(tensors[i].ne)) != 0) {
            ok = false;
            break;
        }
    }
    std::printf("round-trip correctness: %s\n", ok ? "OK" : "MISMATCH");

    std::printf("\n== Result ==\n");
    std::printf("  fresh parse (gguf_init_from_file)      : %.3f ms\n", fresh_parse_ms);
    std::printf("  cached index read (sidecar -> TOC)     : %.3f ms\n", cached_read_ms);
    std::printf("  delta                                  : %.3f ms (%.1fx)\n",
                fresh_parse_ms - cached_read_ms,
                cached_read_ms > 0.0 ? fresh_parse_ms / cached_read_ms : 0.0);

    return ok ? 0 : 1;
}
