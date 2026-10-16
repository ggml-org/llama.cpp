// MUL_MAT_ID on the CPU backend with weights in an mmap'ed file, as llama.cpp does for lazy / -cmoe experts
// registered without and with GGML_CPU_MOE_PREFETCH=1 (read once at CPU backend init), and with a small
// GGML_CPU_MOE_PREFETCH_WINDOW_MB so that the batch-40 residency case takes the windowed prefetch path
// note: residency is checked after the op, so this proves every piece of every used expert (and nothing else) was
// hinted, not when; the window's timing is only visible in end-to-end runs (page cache refaults, throughput)
// each run checks MUL_MAT_ID against per-expert MUL_MAT on the same data, and one case checks through page cache
// residency that exactly the used experts were prefetched

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static bool g_saw_prefetch_log  = false;
static bool g_residency_skipped = false;

static void log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (strstr(text, "CPU MoE expert prefetch enabled")) {
        g_saw_prefetch_log = true;
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        err += (double) (a[i] - b[i]) * (a[i] - b[i]);
        ref += (double) b[i] * b[i];
    }
    return ref > 0.0 ? err / ref : err;
}

// is the page at addr in the page cache / mapped
static bool page_resident(const void * addr) {
    unsigned char v = 0;
    return mincore(const_cast<void *>(addr), 1, &v) == 0 && (v & 1);
}

struct test_case {
    const char * name;
    ggml_type    type;
    int64_t      k;         // row length
    int64_t      n;         // rows per expert
    int64_t      n_as;      // experts
    int64_t      n_used;    // experts per token
    int64_t      n_tokens;
    bool         mmapped;   // weights in an mmap'ed file at an unaligned offset, else a regular buffer
    bool         residency; // experts strided with an unread gap; check which gaps end up in the page cache
};

struct case_state {
    int                   fd    = -1;
    void *                map   = nullptr;
    size_t                map_size = 0;
    ggml_context *        ctx_w = nullptr;
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf   = nullptr;

    ~case_state() {
        if (buf)   { ggml_backend_buffer_free(buf); }
        if (ctx)   { ggml_free(ctx); }
        if (buf_w) { ggml_backend_buffer_free(buf_w); }
        if (ctx_w) { ggml_free(ctx_w); }
        if (map)   { munmap(map, map_size); }
        if (fd >= 0) { close(fd); }
    }
};

static std::string tmp_dir() {
    // prefer a real filesystem: on tmpfs there is no readahead and the residency check is skipped
    const char * d = getenv("TMPDIR");
    return d && *d ? d : ".";
}

// returns true on success
static bool run_case(ggml_backend_t backend, ggml_backend_dev_t dev, const test_case & tc, bool prefetch) {
    case_state st;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);

    // weights: random floats converted to tc.type, one expert after the other
    const size_t row_size    = ggml_row_size(tc.type, tc.k);
    const size_t expert_size = row_size * tc.n;
    std::vector<float> w_f32(tc.k * tc.n * tc.n_as);
    for (auto & v : w_f32) {
        v = dist(rng);
    }
    std::vector<uint8_t> w_data(expert_size * tc.n_as);
    ggml_quantize_chunk(tc.type, w_f32.data(), w_data.data(), 0, tc.n * tc.n_as, tc.k, nullptr);

    // src0 lives in its own no_alloc context so that it can be placed at an explicit address
    ggml_init_params wp = { ggml_tensor_overhead(), nullptr, true };
    st.ctx_w = ggml_init(wp);
    ggml_tensor * as = ggml_new_tensor_3d(st.ctx_w, tc.type, tc.k, tc.n, tc.n_as);
    if (tc.residency) {
        // a 3-page gap after each expert: MUL_MAT_ID never reads it, the prefetch hints the whole nb02 stride
        as->nb[2] = expert_size + 3*page;
        as->nb[3] = as->nb[2] * tc.n_as;
    }
    const size_t nb2 = as->nb[2];

    const size_t offs = 4096 + 96; // 32-aligned, not page aligned
    bool check_residency = false;

    if (tc.mmapped) {
        std::string path = tmp_dir() + "/test-moe-prefetch-XXXXXX";
        st.fd = mkstemp(path.data());
        if (st.fd < 0) {
            fprintf(stderr, "mkstemp in %s failed\n", tmp_dir().c_str());
            return false;
        }
        unlink(path.c_str());

        // ggml_nbytes stops at the end of the last expert's data; include its gap, which the prefetch hints too
        st.map_size = offs + nb2*tc.n_as;
        std::vector<uint8_t> file(st.map_size, 0);
        for (int64_t e = 0; e < tc.n_as; ++e) {
            memcpy(file.data() + offs + e*nb2, w_data.data() + e*expert_size, expert_size);
        }
        if (pwrite(st.fd, file.data(), file.size(), 0) != (ssize_t) file.size()) {
            fprintf(stderr, "pwrite failed\n");
            return false;
        }
        if (tc.residency) {
            // drop the just-written pages, so that only what is read or prefetched comes back
            fdatasync(st.fd);
            posix_fadvise(st.fd, 0, 0, POSIX_FADV_DONTNEED);
        }
        // read-only shared file mapping, like llama_mmap
        st.map = mmap(nullptr, st.map_size, PROT_READ, MAP_SHARED, st.fd, 0);
        if (st.map == MAP_FAILED) {
            st.map = nullptr;
            fprintf(stderr, "mmap failed\n");
            return false;
        }
        if (tc.residency) {
            // like the lazy ranges of llama_mmap: no fault readahead, faults read only the touched pages
            posix_madvise(st.map, st.map_size, POSIX_MADV_RANDOM);
            check_residency = true;
            for (size_t o = 0; o < st.map_size; o += page) {
                if (page_resident((char *) st.map + o)) {
                    check_residency = false; // e.g. tmpfs: pages cannot be dropped
                    break;
                }
            }
        }
        st.buf_w = ggml_backend_dev_buffer_from_host_ptr(dev, st.map, st.map_size, ggml_nbytes(as));
        if (!st.buf_w || ggml_backend_tensor_alloc(st.buf_w, as, (char *) st.map + offs) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "failed to wrap the mapping\n");
            return false;
        }
    } else {
        st.buf_w = ggml_backend_alloc_ctx_tensors(st.ctx_w, backend);
        if (!st.buf_w) {
            fprintf(stderr, "failed to allocate the weights\n");
            return false;
        }
        for (int64_t e = 0; e < tc.n_as; ++e) {
            ggml_backend_tensor_set(as, w_data.data() + e*expert_size, e*nb2, expert_size);
        }
    }

    // activations, routing and the graph; per reference: 2 views + 1 mul_mat
    const size_t n_nodes = 4 * tc.n_used * tc.n_tokens + 16;
    ggml_init_params gp = { ggml_tensor_overhead() * (4 * n_nodes) + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
    st.ctx = ggml_init(gp);

    ggml_tensor * b   = ggml_new_tensor_3d(st.ctx, GGML_TYPE_F32, tc.k, tc.n_used, tc.n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(st.ctx, GGML_TYPE_I32, tc.n_used, tc.n_tokens);

    ggml_tensor * out = ggml_mul_mat_id(st.ctx, as, b, ids);

    // distinct experts per token, deterministic; batch 1 uses experts {0, 5}, batch 40 uses all of them
    std::vector<int32_t> ids_data(tc.n_used * tc.n_tokens);
    std::vector<bool> used(tc.n_as, false);
    for (int64_t t = 0; t < tc.n_tokens; ++t) {
        for (int64_t i = 0; i < tc.n_used; ++i) {
            const int32_t e = (int32_t) ((t * 3 + i * 5) % tc.n_as);
            ids_data[t * tc.n_used + i] = e;
            used[e] = true;
        }
    }

    // reference: one MUL_MAT per (token, slot) over a 2-D view of the selected expert
    // (built in a separate graph, so that the residency check sees only what MUL_MAT_ID did)
    std::vector<ggml_tensor *> refs;
    for (int64_t t = 0; t < tc.n_tokens; ++t) {
        for (int64_t i = 0; i < tc.n_used; ++i) {
            const int32_t e = ids_data[t * tc.n_used + i];
            ggml_tensor * w  = ggml_view_2d(st.ctx, as, tc.k, tc.n, as->nb[1], e * as->nb[2]);
            ggml_tensor * bi = ggml_view_2d(st.ctx, b, tc.k, 1, b->nb[1], i * b->nb[1] + t * b->nb[2]);
            refs.push_back(ggml_mul_mat(st.ctx, w, bi));
        }
    }

    ggml_cgraph * gf_id  = ggml_new_graph_custom(st.ctx, n_nodes, false);
    ggml_cgraph * gf_ref = ggml_new_graph_custom(st.ctx, n_nodes, false);
    ggml_build_forward_expand(gf_id, out);
    for (auto * r : refs) {
        ggml_build_forward_expand(gf_ref, r);
    }

    st.buf = ggml_backend_alloc_ctx_tensors(st.ctx, backend);
    if (!st.buf) {
        fprintf(stderr, "failed to allocate the graph tensors\n");
        return false;
    }

    std::vector<float> b_data(ggml_nelements(b));
    for (auto & v : b_data) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));

    if (ggml_backend_graph_compute(backend, gf_id) != GGML_STATUS_SUCCESS) {
        printf("  %-40s MUL_MAT_ID compute failed\n", tc.name);
        return false;
    }

    bool ok = true;

    if (tc.residency) {
        // the last page of each expert's gap is never read, so it is resident iff that expert was prefetched
        // WILLNEED readahead is asynchronous: wait for the expected pages up to a deadline
        auto gap_page = [&](int64_t e) {
            const uintptr_t end = (uintptr_t) as->data + (e + 1)*nb2;
            return (const void *) ((end & ~(uintptr_t)(page - 1)) - page);
        };
        if (!check_residency) {
            printf("  %-40s residency check SKIPPED (file pages could not be dropped, tmpfs?)\n", tc.name);
            g_residency_skipped = true;
        } else {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            for (;;) {
                bool all = true;
                for (int64_t e = 0; e < tc.n_as; ++e) {
                    if (prefetch && used[e] && !page_resident(gap_page(e))) {
                        all = false;
                    }
                }
                if (all || std::chrono::steady_clock::now() > deadline) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::string got;
            std::string exp;
            for (int64_t e = 0; e < tc.n_as; ++e) {
                const bool want = prefetch && used[e];
                const bool have = page_resident(gap_page(e));
                got += have ? '1' : '0';
                exp += want ? '1' : '0';
                ok = ok && want == have;
            }
            printf("  %-40s gap pages resident per expert: %s (expected %s) %s\n",
                    tc.name, got.c_str(), exp.c_str(), ok ? "OK" : "FAIL");
        }
    }

    if (ggml_backend_graph_compute(backend, gf_ref) != GGML_STATUS_SUCCESS) {
        printf("  %-40s reference compute failed\n", tc.name);
        return false;
    }

    // out: [n, n_used, n_tokens]
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    std::vector<float> exp(got.size());
    size_t r = 0;
    for (int64_t t = 0; t < tc.n_tokens; ++t) {
        for (int64_t i = 0; i < tc.n_used; ++i, ++r) {
            ggml_backend_tensor_get(refs[r], exp.data() + (t * tc.n_used + i) * tc.n, 0, tc.n * sizeof(float));
        }
    }

    const double e = nmse(got, exp);
    const bool match = std::isfinite(e) && e < 1e-7;
    printf("  %-40s nb02=%9zu nmse=%.3e %s\n", tc.name, nb2, e, match ? "OK" : "FAIL");

    return ok && match;
}

int main() {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    const char * env = getenv("GGML_CPU_MOE_PREFETCH");
    const bool want_prefetch = env != nullptr && atoi(env) == 1;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads) {
        set_n_threads(backend, 4); // exercise the split of the prefetch pieces across threads
    }

    printf("test-moe-prefetch: GGML_CPU_MOE_PREFETCH=%s, tmp dir %s\n", want_prefetch ? "1" : "unset", tmp_dir().c_str());

    const test_case cases[] = {
        // name                                       type             k     n     n_as used tokens mmapped residency
        { "mxfp4 small, mmap, batch 1",               GGML_TYPE_MXFP4,   64,   16, 8,   2,   1,    true,   false },
        { "mxfp4 small, mmap, batch 40",              GGML_TYPE_MXFP4,   64,   16, 8,   2,   40,   true,   false },
        { "f32 small, mmap, batch 1",                 GGML_TYPE_F32,     64,   12, 8,   2,   1,    true,   false },
        { "f32 small, mmap, batch 40",                GGML_TYPE_F32,     64,   12, 8,   2,   40,   true,   false },
        { "mxfp4 2.1 MiB slice (short tail), mmap",   GGML_TYPE_MXFP4, 4096, 1012, 8,   2,   1,    true,   false },
        { "mxfp4 2.1 MiB slice, batch 40, mmap",      GGML_TYPE_MXFP4, 4096, 1012, 8,   2,   40,   true,   false },
        { "mxfp4 2.1 MiB strided, residency",         GGML_TYPE_MXFP4, 4096, 1012, 8,   2,   1,    true,   true  },
        // all 8 experts used (~17.7 MB): beyond a small window, so every piece must still be issued in time
        { "mxfp4 2.1 MiB strided, residency, b40",    GGML_TYPE_MXFP4, 4096, 1012, 8,   2,   40,   true,   true  },
        { "mxfp4 small, regular buffer, batch 40",    GGML_TYPE_MXFP4,   64,   16, 8,   2,   40,   false,  false },
    };

    bool ok = true;
    for (const auto & tc : cases) {
        ok = run_case(backend, dev, tc, want_prefetch) && ok;
    }

    // the env flag must actually reach the CPU backend, or this run tests nothing new
    if (want_prefetch != g_saw_prefetch_log) {
        fprintf(stderr, "prefetch log line %s but GGML_CPU_MOE_PREFETCH is %s\n",
                g_saw_prefetch_log ? "seen" : "missing", want_prefetch ? "1" : "unset");
        ok = false;
    }

    ggml_backend_free(backend);

    // the residency checks are what prove the prefetch issued its pieces: report a skip, not a pass
    if (ok && want_prefetch && g_residency_skipped) {
        printf("SKIPPED (residency not checkable in %s)\n", tmp_dir().c_str());
        return 77;
    }

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
