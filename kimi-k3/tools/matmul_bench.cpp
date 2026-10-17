// matmul_bench: 1-token MUL_MAT of real Kimi-K3 weights on the CPU, in VRAM, zero-copy from pinned host memory and staged
// Part B of PLAN-cpu-gpu-matmul.md. Build with build_bench.sh. Weights are read with pread() from the GGUF shards (O_RDONLY).
//
//   matmul_bench [--model DIR] [--weights a,b@E,c@E:N,...] [--paths a,b,c,c4,d] [--threads 7,14,28] [--reps 20] [--warmup 3]
//                [--expert E] [--json out.json] [--cuda-graphs 0|1] [--inventory] [--load-wait-s 300] [--hugepage] [--force]
// --hugepage: host copies 2 MiB aligned + MADV_HUGEPAGE, touched before cudaHostRegister; AnonHugePages delta in the JSON
//
// paths: a = CPU backend, b = CUDA0 weight in VRAM, c = CUDA0 weight in pinned host memory (zero-copy),
//        c4 = c with 4 CUDA backend instances at once, d = pinned -> VRAM copy, then b
// weights: name = one MUL_MAT; name@E = expert slice E of an _exps tensor (MUL_MAT); name@E:N = experts E..E+N-1 as one
//        MUL_MAT_ID with ids 0..N-1 (src1 broadcast for gate / up, one row per expert for down), as in the decode graph
// OpenMP (the CPU backend) reads OMP_PROC_BIND / OMP_PLACES at load: set them in the environment; the JSON records them

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <fcntl.h>
#include <dirent.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

static const size_t   MIB             = 1024ull * 1024;
static const size_t   GIB             = 1024 * MIB;
static const size_t   VRAM_BUDGET     = 1 * GIB;
static const size_t   PINNED_BUDGET   = 3 * GIB / 2;
static const size_t   MIN_FREE_VRAM   = 3 * GIB;
static const double   NMSE_MAX        = 5e-4;
static const int      N_INSTANCES     = 4;

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { die("%s: %s", #x, cudaGetErrorString(err_)); } } while (0)

static void cleanup();

[[noreturn]] static void die(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
static void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "matmul_bench: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    cleanup();
    exit(1);
}

// ---------------------------------------------------------------- resources freed on any exit

static size_t g_vram_used   = 0;
static size_t g_pinned_used = 0;
static std::vector<void *> g_registered;
static std::vector<void *> g_host_allocs;
static std::vector<ggml_backend_buffer_t> g_buffers;

static void cleanup() {
    for (void * p : g_registered) {
        cudaHostUnregister(p);
    }
    g_registered.clear();
    for (ggml_backend_buffer_t b : g_buffers) {
        ggml_backend_buffer_free(b);
    }
    g_buffers.clear();
    for (void * p : g_host_allocs) {
        free(p);
    }
    g_host_allocs.clear();
}

// async-signal-safe: no CUDA / ggml calls; the driver frees VRAM and pinned memory at process exit
static void on_signal(int sig) {
    static const char msg[] = "matmul_bench: signal, exit\n";
    ssize_t n = write(2, msg, sizeof(msg) - 1);
    (void) n;
    _exit(128 + sig);
}

static ggml_backend_buffer_t buf_alloc(ggml_backend_buffer_type_t buft, size_t size, bool vram) {
    if (vram && g_vram_used + size > VRAM_BUDGET) {
        die("VRAM budget: %.1f MiB in use + %.1f MiB > %.0f MiB", g_vram_used / (double) MIB, size / (double) MIB, VRAM_BUDGET / (double) MIB);
    }
    ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, size);
    if (!b) {
        die("buffer alloc of %zu bytes failed", size);
    }
    if (vram) {
        g_vram_used += ggml_backend_buffer_get_size(b);
    }
    g_buffers.push_back(b);
    return b;
}

static void buf_free(ggml_backend_buffer_t b, bool vram) {
    if (!b) {
        return;
    }
    if (vram) {
        g_vram_used -= ggml_backend_buffer_get_size(b);
    }
    g_buffers.erase(std::find(g_buffers.begin(), g_buffers.end(), b));
    ggml_backend_buffer_free(b);
}

static void * host_alloc(size_t size, bool huge = false) {
    void * p = nullptr;
    const size_t align = huge ? 2 * MIB : 4096;
    if (posix_memalign(&p, align, GGML_PAD(size, align)) != 0) {
        die("posix_memalign %zu", size);
    }
    if (huge && madvise(p, GGML_PAD(size, align), MADV_HUGEPAGE) != 0) {
        die("madvise(MADV_HUGEPAGE): %s", strerror(errno));
    }
    g_host_allocs.push_back(p);
    return p;
}

// AnonHugePages of this process in KiB (THP backed anonymous memory)
static long anon_huge_kib() {
    long v = -1;
    if (FILE * f = fopen("/proc/self/smaps_rollup", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "AnonHugePages: %ld kB", &v) == 1) {
                break;
            }
        }
        fclose(f);
    }
    return v;
}

static void host_free(void * p) {
    g_host_allocs.erase(std::find(g_host_allocs.begin(), g_host_allocs.end(), p));
    free(p);
}

static int g_ro_flag_used = 0;

// direct registration: ggml_backend_cuda_register_host_buffer is a no-op without GGML_CUDA_REGISTER_HOST
static void host_register(void * p, size_t size) {
    if (g_pinned_used + size > PINNED_BUDGET) {
        die("pinned budget: %.1f MiB + %.1f MiB > %.0f MiB", g_pinned_used / (double) MIB, size / (double) MIB, PINNED_BUDGET / (double) MIB);
    }
    cudaError_t err = cudaHostRegister(p, size, cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    g_ro_flag_used = 1;
    if (err != cudaSuccess) {
        cudaGetLastError();
        g_ro_flag_used = 0;
        CK(cudaHostRegister(p, size, cudaHostRegisterPortable | cudaHostRegisterMapped));
    }
    g_registered.push_back(p);
    g_pinned_used += size;
}

static void host_unregister(void * p, size_t size) {
    CK(cudaHostUnregister(p));
    g_registered.erase(std::find(g_registered.begin(), g_registered.end(), p));
    g_pinned_used -= size;
}

// ---------------------------------------------------------------- model

struct shard {
    std::string    path;
    gguf_context * g    = nullptr;
    ggml_context * meta = nullptr;
    int            fd   = -1;
};

struct weight {
    std::string spec;         // name[@expert[:n_exp]]
    std::string name;
    int         expert = -1;
    int         n_exp  = 0;   // > 0: MUL_MAT_ID over n_exp experts
    int         n_b    = 1;   // MUL_MAT_ID src1 rows: 1 (broadcast, gate / up) or n_exp (down)
    ggml_type   type;
    int64_t     ne[4] = {1, 1, 1, 1};
    int         n_dims;
    size_t      nbytes;
    int         fd;
    size_t      file_off;
};

static std::vector<shard> g_shards;
static const weight *     g_cur = nullptr;  // the weight being timed (MUL_MAT_ID shape)

static void open_model(const std::string & dir) {
    std::vector<std::string> files;
    for (const auto & e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() == ".gguf") {
            files.push_back(e.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        die("no .gguf files in %s", dir.c_str());
    }
    for (const auto & f : files) {
        shard s;
        s.path = f;
        gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ &s.meta };
        s.g = gguf_init_from_file(f.c_str(), ip);
        if (!s.g) {
            die("gguf_init_from_file %s", f.c_str());
        }
        s.fd = open(f.c_str(), O_RDONLY);
        if (s.fd < 0) {
            die("open %s", f.c_str());
        }
        g_shards.push_back(s);
    }
}

static const ggml_tensor * find_meta(const std::string & name, const shard ** out) {
    for (const auto & s : g_shards) {
        const ggml_tensor * t = ggml_get_tensor(s.meta, name.c_str());
        if (t) {
            *out = &s;
            return t;
        }
    }
    return nullptr;
}

static weight make_weight(const std::string & spec, int def_expert) {
    weight w;
    w.spec = spec;
    w.name = spec;
    auto at = spec.find('@');
    if (at != std::string::npos) {
        w.name   = spec.substr(0, at);
        w.expert = atoi(spec.c_str() + at + 1);
        auto colon = spec.find(':', at);
        if (colon != std::string::npos) {
            w.n_exp = atoi(spec.c_str() + colon + 1);
            if (w.n_exp < 1) {
                die("%s: bad expert count", spec.c_str());
            }
        }
    }
    const shard * s = nullptr;
    const ggml_tensor * t = find_meta(w.name, &s);
    if (!t) {
        die("tensor %s not found", w.name.c_str());
    }
    w.type     = t->type;
    w.fd       = s->fd;
    w.file_off = gguf_get_data_offset(s->g) + gguf_get_tensor_offset(s->g, gguf_find_tensor(s->g, w.name.c_str()));
    if (w.name.find("_exps") != std::string::npos) {
        if (w.expert < 0) {
            w.expert = def_expert;
        }
        const int n = std::max(1, w.n_exp);
        if (w.expert + n > t->ne[2]) {
            die("%s: experts %d..%d >= %" PRId64, w.name.c_str(), w.expert, w.expert + n - 1, t->ne[2]);
        }
        w.spec     = w.name + "@" + std::to_string(w.expert) + (w.n_exp ? ":" + std::to_string(w.n_exp) : "");
        w.ne[0]    = t->ne[0];
        w.ne[1]    = t->ne[1];
        w.ne[2]    = n;
        w.n_dims   = w.n_exp ? 3 : 2;
        w.nbytes   = (size_t) n * t->nb[2];
        w.file_off += (size_t) w.expert * t->nb[2];
        w.n_b      = w.n_exp && w.name.find("down") != std::string::npos ? w.n_exp : 1;
    } else {
        w.n_dims = ggml_n_dims(t);
        for (int i = 0; i < 4; ++i) {
            w.ne[i] = t->ne[i];
        }
        w.nbytes = ggml_nbytes(t);
        if (w.n_exp) {
            die("%s: an expert count needs an _exps tensor", spec.c_str());
        }
    }
    if (w.ne[3] != 1) {
        die("%s: 4D weights not supported", w.name.c_str());
    }
    return w;
}

// all matmul weights of a layer: 2D+ tensors except the experts and the conv kernels
static std::vector<std::string> dense_of_layer(int il) {
    std::vector<std::string> out;
    std::string pre = "blk." + std::to_string(il) + ".";
    for (const auto & s : g_shards) {
        for (int64_t i = 0; i < gguf_get_n_tensors(s.g); ++i) {
            std::string n = gguf_get_tensor_name(s.g, i);
            if (n.rfind(pre, 0) != 0 || n.find("_exps") != std::string::npos || n.find("conv1d") != std::string::npos) {
                continue;
            }
            const ggml_tensor * t = ggml_get_tensor(s.meta, n.c_str());
            if (ggml_n_dims(t) >= 2) {
                out.push_back(n);
            }
        }
    }
    return out;
}

static void pread_all(int fd, void * dst, size_t size, size_t off) {
    char * p = (char *) dst;
    while (size > 0) {
        ssize_t n = pread(fd, p, std::min(size, (size_t) 1 << 30), off);
        if (n <= 0) {
            die("pread: %s", strerror(errno));
        }
        p += n;
        off += n;
        size -= n;
    }
}

// ---------------------------------------------------------------- reference: CPU f32 dequant x f32 dot (double accumulate)

// x[k]: ne0 * ne2 floats (src1 [ne0, 1, ne2]; MUL_MAT_ID: [ne0, n_b, 1]); ref[k]: ne1 * ne2 floats
static void reference(const weight & w, const void * data, const std::vector<std::vector<float>> & x, std::vector<std::vector<float>> & ref) {
    const int64_t ne0 = w.ne[0], ne1 = w.ne[1], ne2 = w.ne[2];
    const size_t row_size = ggml_row_size(w.type, ne0);
    const auto * tr = ggml_get_type_traits(w.type);
    if (w.type != GGML_TYPE_F32 && !tr->to_float) {
        die("%s: no to_float for %s", w.name.c_str(), ggml_type_name(w.type));
    }
    const int K = (int) x.size();
    ref.assign(K, std::vector<float>(ne1 * ne2));
    const int64_t n_rows = ne1 * ne2;
    const int n_thr = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::vector<std::thread> th;
    for (int t = 0; t < n_thr; ++t) {
        th.emplace_back([&, t]() {
            std::vector<float> row(ne0);
            for (int64_t r = t; r < n_rows; r += n_thr) {
                const char * src = (const char *) data + r * row_size;
                if (w.type == GGML_TYPE_F32) {
                    memcpy(row.data(), src, ne0 * sizeof(float));
                } else {
                    tr->to_float(src, row.data(), ne0);
                }
                const int64_t h = (w.n_exp && w.n_b == 1) ? 0 : r / ne1;
                for (int k = 0; k < K; ++k) {
                    const float * xv = x[k].data() + h * ne0;
                    double acc = 0.0;
                    for (int64_t i = 0; i < ne0; ++i) {
                        acc += (double) row[i] * xv[i];
                    }
                    ref[k][r] = (float) acc;
                }
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
}

// ---------------------------------------------------------------- one graph of K x R MUL_MAT nodes

struct graph_set {
    ggml_context *        ctx = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    bool                  vram = false;
    std::vector<ggml_tensor *> dst;  // index r * K + k
    int K = 0, R = 0;

    void build(const std::vector<ggml_tensor *> & W, const std::vector<std::vector<float>> & X, int R_, ggml_backend_buffer_type_t buft, bool vram_) {
        K    = (int) W.size();
        R    = R_;
        vram = vram_;
        const size_t n_nodes = (size_t) K * R + 2 * K + 16;
        ggml_init_params ip = { ggml_tensor_overhead() * (K + K * R + 16) + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        ctx = ggml_init(ip);
        const bool mmid = g_cur->n_exp > 0;
        std::vector<ggml_tensor *> xs(K);
        for (int k = 0; k < K; ++k) {
            const ggml_tensor * w = W[k];
            xs[k] = mmid ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], g_cur->n_b, 1)
                         : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], 1, w->ne[2]);
        }
        ggml_tensor * ids = mmid ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, g_cur->n_exp, 1) : nullptr;
        gf = ggml_new_graph_custom(ctx, n_nodes, false);
        for (int r = 0; r < R; ++r) {
            for (int k = 0; k < K; ++k) {
                ggml_tensor * d = mmid ? ggml_mul_mat_id(ctx, W[k], xs[k], ids) : ggml_mul_mat(ctx, W[k], xs[k]);
                dst.push_back(d);
                ggml_build_forward_expand(gf, d);
            }
        }
        size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
        if (vram && g_vram_used + need > VRAM_BUDGET) {
            die("VRAM budget (activations): %.1f + %.1f MiB", g_vram_used / (double) MIB, need / (double) MIB);
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            die("activation buffer alloc failed");
        }
        if (vram) {
            g_vram_used += ggml_backend_buffer_get_size(buf);
        }
        g_buffers.push_back(buf);
        for (int k = 0; k < K; ++k) {
            ggml_backend_tensor_set(xs[k], X[k].data(), 0, X[k].size() * sizeof(float));
        }
        if (ids) {
            std::vector<int32_t> iv(g_cur->n_exp);
            for (int i = 0; i < g_cur->n_exp; ++i) {
                iv[i] = i;
            }
            ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
        }
    }

    void release() {
        if (buf) {
            buf_free(buf, vram);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        ctx = nullptr;
        buf = nullptr;
        dst.clear();
    }
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double compute_ms(ggml_backend_t be, ggml_cgraph * gf) {
    const double t0 = now_ms();
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        die("graph compute failed");
    }
    ggml_backend_synchronize(be);
    return now_ms() - t0;
}

struct path_result {
    std::string path;
    int         threads = 0;
    int         K = 0, R = 0;
    double      med_ms = NAN, min_ms = NAN;  // per matmul
    double      gbs = NAN;
    double      nmse = NAN;
    bool        ok = false;
    bool        noisy() const { return med_ms > 1.5 * min_ms; }
    std::string extra;                       // json fragment
    std::vector<float> out0;                 // outputs of r = 0, for the bit compare
};

static double nmse_of(const graph_set & gs, const std::vector<std::vector<float>> & ref, std::vector<float> * out0) {
    double num = 0.0, den = 0.0;
    std::vector<float> y;
    if (out0) {
        out0->clear();
    }
    for (size_t i = 0; i < gs.dst.size(); ++i) {
        const int k = (int) (i % gs.K);
        y.resize(ggml_nelements(gs.dst[i]));
        ggml_backend_tensor_get(gs.dst[i], y.data(), 0, y.size() * sizeof(float));
        for (size_t j = 0; j < y.size(); ++j) {
            const double d = (double) y[j] - ref[k][j];
            num += d * d;
            den += (double) ref[k][j] * ref[k][j];
        }
        if (out0 && (int) i < gs.K) {
            out0->insert(out0->end(), y.begin(), y.end());
        }
    }
    return num / den;
}

struct opts {
    std::string model = "/mnt/k3/kimi-k3-iq1";
    std::vector<std::string> weights;
    std::vector<std::string> paths = {"a", "b", "c", "c4", "d"};
    std::vector<int> threads = {7, 14, 28};
    int    reps = 20;
    int    warmup = 3;
    int    expert = 0;
    double min_graph_ms = 5.0;
    double footprint_mib = 256.0;
    int    cuda_graphs = 0;
    bool   inventory = false;
    bool   force = false;
    bool   hugepage = false;
    double load_wait_s = 300.0;
    std::string json;
};

static int pick_R(ggml_backend_t be, graph_set & gs, const opts & o) {
    compute_ms(be, gs.gf);
    const double t = compute_ms(be, gs.gf);
    int R = (int) std::ceil(o.min_graph_ms / std::max(t, 1e-3));
    R = std::max(1, std::min(R, 64));
    while (R > 1 && (size_t) gs.K * R > 30000) {
        R--;
    }
    return R;
}

static void time_graph(ggml_backend_t be, graph_set & gs, const opts & o, path_result & pr, const std::vector<std::vector<float>> & ref, size_t nbytes) {
    // zero-copy must not copy weights: nsys shows no HtoD memcpy inside the "path_c" range
    const bool mark = pr.path == "c";
    if (mark) {
        nvtxRangePushA("path_c");
    }
    for (int i = 0; i < o.warmup; ++i) {
        compute_ms(be, gs.gf);
    }
    std::vector<double> t;
    for (int i = 0; i < o.reps; ++i) {
        t.push_back(compute_ms(be, gs.gf) / (gs.K * gs.R));
    }
    if (mark) {
        nvtxRangePop();
    }
    std::sort(t.begin(), t.end());
    pr.K      = gs.K;
    pr.R      = gs.R;
    pr.med_ms = t[t.size() / 2];
    pr.min_ms = t[0];
    pr.gbs    = nbytes / (pr.med_ms * 1e6);
    pr.nmse   = nmse_of(gs, ref, &pr.out0);
    pr.ok     = pr.nmse < NMSE_MAX;
}

// builds the graph twice: once to find R so that the K x R graph runs >= min_graph_ms
static void run_path(ggml_backend_t be, ggml_backend_buffer_type_t buft, bool vram, const std::vector<ggml_tensor *> & W,
                     const std::vector<std::vector<float>> & X, const std::vector<std::vector<float>> & ref, size_t nbytes,
                     const opts & o, path_result & pr, int R_fixed = 0) {
    graph_set gs;
    int R = R_fixed;
    if (R == 0) {
        gs.build(W, X, 1, buft, vram);
        R = pick_R(be, gs, o);
        if (R > 1) {
            gs.release();
            gs = graph_set();
        }
    }
    if (!gs.ctx) {
        gs.build(W, X, R, buft, vram);
    }
    time_graph(be, gs, o, pr, ref, nbytes);
    gs.release();
}

static std::string bitcmp(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return "\"n/a\"";
    }
    size_t n_diff = 0;
    double max_d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            n_diff++;
            max_d = std::max(max_d, (double) std::fabs(a[i] - b[i]));
        }
    }
    char s[128];
    snprintf(s, sizeof(s), "{\"identical\": %s, \"n_diff\": %zu, \"max_abs_diff\": %g}", n_diff == 0 ? "true" : "false", n_diff, max_d);
    return s;
}

static void print_result(const weight & w, const path_result & pr) {
    printf("  %-5s %3s K=%-4d R=%-3d  %9.4f ms (min %9.4f)  %8.2f GB/s (at min %8.2f)  NMSE %.2e %s%s%s%s\n", pr.path.c_str(),
           pr.threads ? std::to_string(pr.threads).c_str() : "", pr.K, pr.R, pr.med_ms, pr.min_ms, pr.gbs, w.nbytes / (pr.min_ms * 1e6), pr.nmse,
           pr.ok ? "OK" : "FAIL", pr.noisy() ? " NOISY (median > 1.5 x min)" : "", pr.extra.empty() ? "" : "  ", pr.extra.c_str());
    fflush(stdout);
}

static bool has(const std::vector<std::string> & v, const char * s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) {
            b = s.size();
        }
        if (b > a) {
            out.push_back(s.substr(a, b - a));
        }
        a = b + 1;
    }
    return out;
}

static std::string jstr(const std::string & s) {
    return "\"" + s + "\"";
}

static std::string jnum(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char s[64];
    snprintf(s, sizeof(s), "%.6g", v);
    return s;
}

// ---------------------------------------------------------------- run gate

static double load_avg() {
    double la = 0.0;
    if (FILE * f = fopen("/proc/loadavg", "r")) {
        if (fscanf(f, "%lf", &la) != 1) {
            la = 0.0;
        }
        fclose(f);
    }
    return la;
}

static long long proc_ticks(int pid) {
    char path[64], buf[1024];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE * f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    const char * p = strrchr(buf, ')');
    if (!p) {
        return -1;
    }
    // fields after the comm: state is field 3, utime 14, stime 15
    long long v[13] = {0};
    char st;
    if (sscanf(p + 1, " %c %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", &st, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
               &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]) != 13) {
        return -1;
    }
    return v[10] + v[11];
}

// refuse while a llama-server process exists; else wait until load avg < 2, at most wait_s, then go on with a warning
static std::string run_gate(bool force, double wait_s) {
    std::string warn;
    std::vector<int> pids;
    if (DIR * d = opendir("/proc")) {
        while (dirent * e = readdir(d)) {
            const int pid = atoi(e->d_name);
            if (pid <= 0) {
                continue;
            }
            char path[64], comm[64] = {0};
            snprintf(path, sizeof(path), "/proc/%d/comm", pid);
            if (FILE * f = fopen(path, "r")) {
                if (fgets(comm, sizeof(comm), f)) {
                    comm[strcspn(comm, "\n")] = 0;
                }
                fclose(f);
            }
            if (strcmp(comm, "llama-server") == 0) {
                pids.push_back(pid);
            }
        }
        closedir(d);
    }
    if (!pids.empty()) {
        std::vector<long long> t0;
        for (int pid : pids) {
            t0.push_back(proc_ticks(pid));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        char msg[256];
        for (size_t i = 0; i < pids.size(); ++i) {
            const long long t1 = proc_ticks(pids[i]);
            const double cpu = t0[i] >= 0 && t1 >= 0 ? 100.0 * (t1 - t0[i]) / sysconf(_SC_CLK_TCK) : -1.0;
            snprintf(msg, sizeof(msg), "llama-server pid %d is running (%.0f%% CPU over 1 s)", pids[i], cpu);
            if (!force) {
                die("%s; stop it or use --force", msg);
            }
            warn += std::string(warn.empty() ? "" : "; ") + msg;
        }
    }
    const double t_start = now_ms();
    double la = load_avg();
    while (la >= 2.0 && !force) {
        if (now_ms() - t_start > wait_s * 1000.0) {
            break;
        }
        fprintf(stderr, "load avg %.2f >= 2, waiting (%.0f / %.0f s)\n", la, (now_ms() - t_start) / 1000.0, wait_s);
        std::this_thread::sleep_for(std::chrono::seconds(5));
        la = load_avg();
    }
    if (la >= 2.0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "load avg %.2f >= 2 at start%s", la, force ? " (--force)" : " after the wait");
        fprintf(stderr, "WARNING: %s\n", msg);
        warn += std::string(warn.empty() ? "" : "; ") + msg;
    }
    return warn;
}

static std::string jesc(const std::string & s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
        }
        o += c;
    }
    return o + "\"";
}

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { die("missing value for %s", a.c_str()); } return argv[++i]; };
        if      (a == "--model")         { o.model = next(); }
        else if (a == "--weights")       { o.weights = split(next()); }
        else if (a == "--paths")         { o.paths = split(next()); }
        else if (a == "--threads")       { o.threads.clear(); for (auto & s : split(next())) { o.threads.push_back(atoi(s.c_str())); } }
        else if (a == "--reps")          { o.reps = atoi(next().c_str()); }
        else if (a == "--warmup")        { o.warmup = atoi(next().c_str()); }
        else if (a == "--expert")        { o.expert = atoi(next().c_str()); }
        else if (a == "--min-graph-ms")  { o.min_graph_ms = atof(next().c_str()); }
        else if (a == "--footprint-mib") { o.footprint_mib = atof(next().c_str()); }
        else if (a == "--cuda-graphs")   { o.cuda_graphs = atoi(next().c_str()); }
        else if (a == "--json")          { o.json = next(); }
        else if (a == "--inventory")     { o.inventory = true; }
        else if (a == "--force")         { o.force = true; }
        else if (a == "--hugepage")      { o.hugepage = true; }
        else if (a == "--load-wait-s")   { o.load_wait_s = atof(next().c_str()); }
        else { die("unknown argument %s (see the header of matmul_bench.cpp)", a.c_str()); }
    }
    if (o.reps < 1) {
        die("--reps must be >= 1");
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // the CUDA backend reads this once, at its first graph
    if (o.cuda_graphs) {
        unsetenv("GGML_CUDA_DISABLE_GRAPHS");
    } else {
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    }

    open_model(o.model);
    if (o.weights.empty()) {
        for (int il : {50, 51}) {
            for (auto & n : dense_of_layer(il)) {
                o.weights.push_back(n);
            }
        }
        o.weights.push_back("output.weight");
        o.weights.push_back("blk.50.ffn_gate_exps.weight");
        o.weights.push_back("blk.50.ffn_down_exps.weight");
        o.weights.push_back("blk.5.ffn_down_exps.weight");
        o.weights.push_back("blk.50.ffn_gate_exps.weight@0:16");
        o.weights.push_back("blk.50.ffn_down_exps.weight@0:16");
    }
    std::vector<weight> ws;
    for (auto & s : o.weights) {
        ws.push_back(make_weight(s, o.expert));
    }
    const size_t footprint = (size_t) (o.footprint_mib * MIB);
    auto K_of = [&](const weight & w) { return (int) std::max<size_t>(1, (footprint + w.nbytes - 1) / w.nbytes); };

    printf("weights (%zu), cold footprint %.0f MiB: K = max(1, ceil(footprint / bytes))\n", ws.size(), o.footprint_mib);
    for (auto & w : ws) {
        printf("  %-34s %-7s ne [%" PRId64 ", %" PRId64 ", %" PRId64 "] %12zu B  K=%d\n", w.spec.c_str(), ggml_type_name(w.type), w.ne[0], w.ne[1], w.ne[2], w.nbytes, K_of(w));
    }
    if (o.inventory) {
        return 0;
    }

    const std::string load_warning = run_gate(o.force, o.load_wait_s);
    const double la = load_avg();

    const bool need_cuda = has(o.paths, "b") || has(o.paths, "c") || has(o.paths, "c4") || has(o.paths, "d");
    int can_host_ptr = 0, reg_supported = 0, ro_supported = 0;
    size_t vram_free = 0, vram_total = 0;
    ggml_backend_t cuda = nullptr;
    if (need_cuda) {
        CK(cudaSetDevice(0));
        CK(cudaMemGetInfo(&vram_free, &vram_total));
        if (vram_free < MIN_FREE_VRAM) {
            die("only %.2f GiB VRAM free (< 3 GiB)", vram_free / (double) GIB);
        }
        CK(cudaDeviceGetAttribute(&can_host_ptr, cudaDevAttrCanUseHostPointerForRegisteredMem, 0));
        CK(cudaDeviceGetAttribute(&reg_supported, cudaDevAttrHostRegisterSupported, 0));
        CK(cudaDeviceGetAttribute(&ro_supported, cudaDevAttrHostRegisterReadOnlySupported, 0));
        if (!reg_supported && (has(o.paths, "c") || has(o.paths, "c4") || has(o.paths, "d"))) {
            die("cudaHostRegister not supported on device 0");
        }
        cuda = ggml_backend_cuda_init(0);
        if (!cuda) {
            die("ggml_backend_cuda_init(0) failed");
        }
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t cuda_buft = need_cuda ? ggml_backend_cuda_buffer_type(0) : nullptr;

    printf("load avg %.2f, VRAM free %.2f / %.2f GiB, CUDA graphs %s, host ptr usable %d, register %d, read-only register %d\n", la,
           vram_free / (double) GIB, vram_total / (double) GIB, o.cuda_graphs ? "on" : "off (GGML_CUDA_DISABLE_GRAPHS=1)", can_host_ptr,
           reg_supported, ro_supported);

    std::string js = "{\n \"cuda_graphs\": " + std::to_string(o.cuda_graphs) + ",\n \"reps\": " + std::to_string(o.reps) +
                     ",\n \"warmup\": " + std::to_string(o.warmup) + ",\n \"footprint_mib\": " + jnum(o.footprint_mib) +
                     ",\n \"min_graph_ms\": " + jnum(o.min_graph_ms) + ",\n \"can_use_host_ptr\": " + std::to_string(can_host_ptr) +
                     ",\n \"nmse_max\": " + jnum(NMSE_MAX) + ",\n \"load_avg\": " + jnum(la) + ",\n \"load_warning\": " + jesc(load_warning) +
                     ",\n \"omp_proc_bind\": " + jesc(getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "") +
                     ",\n \"omp_places\": " + jesc(getenv("OMP_PLACES") ? getenv("OMP_PLACES") : "") + ",\n \"weights\": [";
    int n_fail = 0;
    std::map<std::pair<std::string, std::string>, std::vector<double>> per_type;  // (path, type) -> n, bytes, ms

    for (size_t wi = 0; wi < ws.size(); ++wi) {
        const weight & w = ws[wi];
        const int K = K_of(w);
        g_cur = &w;

        // weight tensors: one context for the CPU copies, one for the CUDA copies
        ggml_init_params ip = { ggml_tensor_overhead() * (2 * K + 4), nullptr, true };
        ggml_context * wctx = ggml_init(ip);
        std::vector<ggml_tensor *> Wc(K), Wg(K);
        for (int k = 0; k < K; ++k) {
            Wc[k] = ggml_new_tensor(wctx, w.type, w.n_dims, w.ne);
            Wg[k] = ggml_new_tensor(wctx, w.type, w.n_dims, w.ne);
        }
        size_t alloc = w.nbytes;
        if (cuda_buft) {
            alloc = std::max(alloc, ggml_backend_buft_get_alloc_size(cuda_buft, Wg[0]));
        }
        const size_t stride = GGML_PAD(alloc, 4096);
        const size_t hsize  = stride * K;
        if (hsize > PINNED_BUDGET || (need_cuda && hsize > VRAM_BUDGET)) {
            printf("%s: %zu MiB for K=%d copies exceeds the budget, skipped\n", w.spec.c_str(), hsize / MIB, K);
            ggml_free(wctx);
            continue;
        }

        // K distinct host copies, padding zero
        const long huge0 = anon_huge_kib();
        char * H = (char *) host_alloc(hsize, o.hugepage);
        memset(H, 0, GGML_PAD(hsize, o.hugepage ? 2 * MIB : 4096));  // touch before cudaHostRegister
        const long huge_kib = anon_huge_kib() - huge0;
        printf("host copies: %.1f MiB, %s, AnonHugePages +%ld KiB (%.0f%% of the copies)\n", hsize / (double) MIB,
               o.hugepage ? "2 MiB aligned + MADV_HUGEPAGE" : "4 KiB aligned", huge_kib, 100.0 * huge_kib * 1024 / hsize);
        pread_all(w.fd, H, w.nbytes, w.file_off);
        for (int k = 1; k < K; ++k) {
            memcpy(H + k * stride, H, w.nbytes);
        }

        std::vector<std::vector<float>> X(K), ref;
        for (int k = 0; k < K; ++k) {
            std::mt19937 rng(1234 + k);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            X[k].resize(w.ne[0] * (w.n_exp ? w.n_b : w.ne[2]));
            for (auto & v : X[k]) {
                v = u(rng);
            }
        }
        reference(w, H, X, ref);

        printf("\n%s %s [%" PRId64 ", %" PRId64 ", %" PRId64 "] %.3f MB, K=%d (footprint %.1f MiB)\n", w.spec.c_str(), ggml_type_name(w.type),
               w.ne[0], w.ne[1], w.ne[2], w.nbytes / 1e6, K, hsize / (double) MIB);
        std::vector<path_result> res;

        // (a) CPU, weights in a CPU buffer over the host copies
        ggml_backend_buffer_t cbuf = ggml_backend_cpu_buffer_from_ptr(H, hsize);
        g_buffers.push_back(cbuf);
        for (int k = 0; k < K; ++k) {
            if (ggml_backend_tensor_alloc(cbuf, Wc[k], H + k * stride) != GGML_STATUS_SUCCESS) {
                die("tensor alloc (cpu)");
            }
        }
        if (has(o.paths, "a")) {
            for (int nt : o.threads) {
                ggml_backend_cpu_set_n_threads(cpu, nt);
                path_result pr;
                pr.path = "a";
                pr.threads = nt;
                run_path(cpu, ggml_backend_cpu_buffer_type(), false, Wc, X, ref, w.nbytes, o, pr);
                print_result(w, pr);
                res.push_back(pr);
            }
        }

        ggml_backend_buffer_t vbuf = nullptr;
        path_result rb;
        if (need_cuda) {
            vbuf = buf_alloc(cuda_buft, hsize + 4096, true);
            char * vbase = (char *) GGML_PAD((uintptr_t) ggml_backend_buffer_get_base(vbuf), 4096);
            for (int k = 0; k < K; ++k) {
                if (ggml_backend_tensor_alloc(vbuf, Wg[k], vbase + k * stride) != GGML_STATUS_SUCCESS) {
                    die("tensor alloc (cuda)");
                }
            }
            for (int k = 0; k < K; ++k) {
                ggml_backend_tensor_set(Wg[k], H + k * stride, 0, w.nbytes);
            }
            int R_c = 0;

            // (b) VRAM
            if (has(o.paths, "b") || has(o.paths, "c") || has(o.paths, "d") || has(o.paths, "c4")) {
                rb.path = "b";
                run_path(cuda, cuda_buft, true, Wg, X, ref, w.nbytes, o, rb);
                if (has(o.paths, "b")) {
                    print_result(w, rb);
                    res.push_back(rb);
                }
            }

            if (has(o.paths, "c") || has(o.paths, "c4") || has(o.paths, "d") || has(o.paths, "a")) {
                host_register(H, hsize);
                char * hdev = H;
                if (!can_host_ptr) {
                    void * d = nullptr;
                    CK(cudaHostGetDevicePointer(&d, H, 0));
                    hdev = (char *) d;
                }

                // (a) once more at 14 threads with the weights pinned
                if (has(o.paths, "a")) {
                    ggml_backend_cpu_set_n_threads(cpu, 14);
                    path_result pr;
                    pr.path = "a_pin";
                    pr.threads = 14;
                    run_path(cpu, ggml_backend_cpu_buffer_type(), false, Wc, X, ref, w.nbytes, o, pr);
                    print_result(w, pr);
                    res.push_back(pr);
                }

                // (d) staged: poison VRAM, copy pinned -> VRAM (timed apart), then the VRAM graph
                if (has(o.paths, "d")) {
                    cudaStream_t st;
                    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
                    std::vector<double> tc;
                    for (int i = 0; i < 1 + std::min(o.reps, 5); ++i) {
                        ggml_backend_buffer_clear(vbuf, 0xFF);
                        ggml_backend_synchronize(cuda);
                        CK(cudaDeviceSynchronize());
                        const double t0 = now_ms();
                        CK(cudaMemcpyAsync(vbase, hdev, hsize, cudaMemcpyHostToDevice, st));
                        CK(cudaStreamSynchronize(st));
                        if (i > 0) {
                            tc.push_back(now_ms() - t0);
                        }
                    }
                    CK(cudaStreamDestroy(st));
                    std::sort(tc.begin(), tc.end());
                    path_result pr;
                    pr.path = "d";
                    run_path(cuda, cuda_buft, true, Wg, X, ref, w.nbytes, o, pr, rb.R);
                    const double copy_ms = tc[tc.size() / 2] / K;
                    std::string bc = bitcmp(pr.out0, rb.out0);
                    printf("        d vs b: %s\n", bc.c_str());
                    pr.extra = "\"copy_ms\": " + jnum(copy_ms) + ", \"copy_gbs\": " + jnum(stride / (copy_ms * 1e6)) +
                               ", \"copy_plus_compute_ms\": " + jnum(copy_ms + pr.med_ms) + ", \"vs_b\": " + bc;
                    print_result(w, pr);
                    res.push_back(pr);
                }

                // (c) zero-copy: poison the VRAM backing, show that it fails there, then point the tensors at the host copies
                if (has(o.paths, "c") || has(o.paths, "c4")) {
                    ggml_backend_buffer_clear(vbuf, 0xFF);
                    ggml_backend_synchronize(cuda);
                    path_result ctl;
                    ctl.path = "c_ctl";
                    opts o1 = o;
                    o1.warmup = 0;
                    o1.reps = 1;
                    run_path(cuda, cuda_buft, true, Wg, X, ref, w.nbytes, o1, ctl, 1);
                    const bool ctl_failed = !(ctl.nmse < NMSE_MAX);
                    printf("        poison control (weights = 0xFF VRAM): NMSE %.3g -> %s\n", ctl.nmse,
                           ctl_failed ? "fails as expected" : "PASSES: the poison does not show, the (c) proof is void");
                    for (int k = 0; k < K; ++k) {
                        Wg[k]->data = hdev + k * stride;
                    }
                    path_result pr;
                    pr.path = "c";
                    run_path(cuda, cuda_buft, true, Wg, X, ref, w.nbytes, o, pr);
                    R_c = pr.R;
                    std::string bc = bitcmp(pr.out0, rb.out0);
                    printf("        c vs b: %s\n", bc.c_str());
                    pr.extra = "\"poison_control_nmse\": " + jnum(ctl.nmse) + ", \"poison_control_failed\": " + (ctl_failed ? "true" : "false") +
                               ", \"read_only_register\": " + std::to_string(g_ro_flag_used) + ", \"vs_b\": " + bc;
                    if (!ctl_failed) {
                        pr.ok = false;
                    }
                    if (has(o.paths, "c")) {
                        print_result(w, pr);
                        res.push_back(pr);
                    }

                    // c4: 4 CUDA backend instances, own src1 / dst / graph, the same pinned weights
                    if (has(o.paths, "c4")) {
                        std::vector<ggml_backend_t> be(N_INSTANCES);
                        std::vector<graph_set> gs(N_INSTANCES);
                        be[0] = cuda;
                        for (int i = 1; i < N_INSTANCES; ++i) {
                            be[i] = ggml_backend_cuda_init(0);
                            if (!be[i]) {
                                die("ggml_backend_cuda_init(0) instance %d", i);
                            }
                        }
                        for (int i = 0; i < N_INSTANCES; ++i) {
                            gs[i].build(Wg, X, R_c, cuda_buft, true);
                            for (int j = 0; j < o.warmup; ++j) {
                                compute_ms(be[i], gs[i].gf);
                            }
                        }
                        std::vector<double> t;
                        nvtxRangePushA("path_c4");
                        for (int rep = 0; rep < o.reps; ++rep) {
                            std::vector<double> t0(N_INSTANCES), t1(N_INSTANCES);
                            std::atomic<int> ready{0};
                            std::atomic<int> failed{-1};
                            std::vector<std::thread> th;
                            for (int i = 0; i < N_INSTANCES; ++i) {
                                th.emplace_back([&, i]() {
                                    ready++;
                                    while (ready.load() < N_INSTANCES) {
                                    }
                                    t0[i] = now_ms();
                                    if (ggml_backend_graph_compute(be[i], gs[i].gf) != GGML_STATUS_SUCCESS) {
                                        failed = i;
                                    }
                                    ggml_backend_synchronize(be[i]);
                                    t1[i] = now_ms();
                                });
                            }
                            for (auto & x : th) {
                                x.join();
                            }
                            if (failed >= 0) {
                                die("graph compute failed (instance %d)", failed.load());
                            }
                            t.push_back((*std::max_element(t1.begin(), t1.end()) - *std::min_element(t0.begin(), t0.end())) / (N_INSTANCES * K * R_c));
                        }
                        nvtxRangePop();
                        std::sort(t.begin(), t.end());
                        path_result p4;
                        p4.path = "c4";
                        p4.K = K;
                        p4.R = R_c;
                        p4.med_ms = t[t.size() / 2];
                        p4.min_ms = t[0];
                        p4.gbs = w.nbytes / (p4.med_ms * 1e6);
                        double worst = 0.0;
                        for (int i = 0; i < N_INSTANCES; ++i) {
                            const double e = nmse_of(gs[i], ref, nullptr);
                            worst = std::isfinite(e) ? std::max(worst, e) : INFINITY;
                        }
                        p4.nmse = worst;
                        p4.ok = worst < NMSE_MAX && ctl_failed;
                        p4.extra = "\"instances\": " + std::to_string(N_INSTANCES);
                        print_result(w, p4);
                        res.push_back(p4);
                        for (int i = 0; i < N_INSTANCES; ++i) {
                            gs[i].release();
                        }
                        for (int i = 1; i < N_INSTANCES; ++i) {
                            ggml_backend_free(be[i]);
                        }
                    }
                }
                host_unregister(H, hsize);
            }
        }

        js += std::string(wi ? "," : "") + "\n  {\"name\": " + jstr(w.spec) + ", \"type\": " + jstr(ggml_type_name(w.type)) + ", \"ne\": [" +
              std::to_string(w.ne[0]) + ", " + std::to_string(w.ne[1]) + ", " + std::to_string(w.ne[2]) + "], \"bytes\": " +
              std::to_string(w.nbytes) + ", \"K\": " + std::to_string(K) + ", \"hugepage\": " + (o.hugepage ? "true" : "false") +
              ", \"anon_huge_kib\": " + std::to_string(huge_kib) + ", \"host_kib\": " + std::to_string(hsize / 1024) + ", \"paths\": [";
        for (size_t i = 0; i < res.size(); ++i) {
            const auto & pr = res[i];
            if (pr.ok && std::isfinite(pr.med_ms)) {
                auto & v = per_type[{pr.path + (pr.threads ? "@" + std::to_string(pr.threads) : ""),
                                     std::string(ggml_type_name(w.type)) + (w.n_exp ? " MMID" : "")}];
                v.resize(3);
                v[0] += 1;
                v[1] += w.nbytes;
                v[2] += pr.med_ms;
            }
            if (!pr.ok) {
                n_fail++;
                printf("  FAIL: %s path %s%s NMSE %.3g\n", w.spec.c_str(), pr.path.c_str(), pr.threads ? ("@" + std::to_string(pr.threads)).c_str() : "", pr.nmse);
            }
            js += std::string(i ? "," : "") + "\n   {\"path\": " + jstr(pr.path) + ", \"threads\": " + std::to_string(pr.threads) + ", \"K\": " +
                  std::to_string(pr.K) + ", \"R\": " + std::to_string(pr.R) + ", \"med_ms\": " + jnum(pr.med_ms) + ", \"min_ms\": " + jnum(pr.min_ms) +
                  ", \"gbs\": " + jnum(pr.gbs) + ", \"gbs_at_min\": " + jnum(w.nbytes / (pr.min_ms * 1e6)) + ", \"noisy\": " + (pr.noisy() ? "true" : "false") +
                  ", \"nmse\": " + jnum(pr.nmse) + ", \"ok\": " + (pr.ok ? "true" : "false") +
                  (pr.extra.empty() || pr.extra[0] != '"' ? "" : ", " + pr.extra) + "}";
        }
        js += "]}";

        if (vbuf) {
            buf_free(vbuf, true);
        }
        g_buffers.erase(std::find(g_buffers.begin(), g_buffers.end(), cbuf));
        ggml_backend_buffer_free(cbuf);
        ggml_free(wctx);
        host_free(H);
    }
    // per type and path: bytes-weighted GB/s = sum of bytes / sum of median times, one matmul of each weight
    printf("\nper type and path (GB/s = sum bytes / sum median ms; NMSE-passing paths only)\n");
    js += "\n ],\n \"per_type\": [";
    bool first = true;
    for (const auto & [key, v] : per_type) {
        const double gbs = v[1] / (v[2] * 1e6);
        printf("  %-7s %-13s n=%-3d %10.1f MB %10.4f ms %9.2f GB/s\n", key.first.c_str(), key.second.c_str(), (int) v[0], v[1] / 1e6, v[2], gbs);
        js += std::string(first ? "" : ",") + "\n  {\"path\": " + jstr(key.first) + ", \"type\": " + jstr(key.second) + ", \"n\": " +
              std::to_string((int) v[0]) + ", \"bytes\": " + jnum(v[1]) + ", \"ms\": " + jnum(v[2]) + ", \"gbs\": " + jnum(gbs) + "}";
        first = false;
    }
    js += "\n ],\n \"n_fail\": " + std::to_string(n_fail) + "\n}\n";

    if (!o.json.empty()) {
        FILE * f = fopen(o.json.c_str(), "w");
        if (!f) {
            die("cannot write %s", o.json.c_str());
        }
        fputs(js.c_str(), f);
        fclose(f);
        printf("\nwrote %s\n", o.json.c_str());
    }
    printf("\n%s: %d path(s) failed the NMSE < %g check\n", n_fail ? "FAILED" : "OK", n_fail, NMSE_MAX);

    if (cuda) {
        ggml_backend_free(cuda);
    }
    ggml_backend_free(cpu);
    cleanup();
    for (auto & s : g_shards) {
        close(s.fd);
        gguf_free(s.g);
        ggml_free(s.meta);
    }
    return n_fail ? 1 : 0;
}
