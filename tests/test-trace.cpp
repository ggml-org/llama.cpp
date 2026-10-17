// decode trace (ggml-trace.h) on the CPU backend: a MUL_MAT_ID graph with its weights in an mmap'ed file at an
// unaligned offset, as llama.cpp maps lazily read experts. each case runs in its own process (the expert prefetch
// switch is read once at CPU backend init) and checks the JSON trace it wrote against what the graph did:
//   prefetch   GGML_CPU_MOE_PREFETCH=1, detail 1: TOKEN / OP_CPU / EXPERT / residency / merged I/O slice spans
//   detail2    same at detail 2: per-thread expert spans, per-piece I/O spans and their resolution by cookie
//   cold       no prefetch: the evicted expert is not resident at compute start and stalls more than the hot one
//   state      SKIP / TOKENS accounting, SUSPENDED batches, IDLE issues, duplicates, off-path numerics
//   plan       plan made with the trace off, computed with it on
//   buffer     a 1 MB chunk pool overflowed by 200k records, and a full residency ring: counted, never blocking
// every file is also parsed truncated after each line (a crashed process leaves such a file)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-trace.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

using json = nlohmann::json;

static int g_failures = 0;

#define CHECK(cond, ...) do {                                               \
        if (!(cond)) {                                                      \
            fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
            g_failures++;                                                   \
        }                                                                   \
    } while (0)

static void log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static std::string tmp_dir() {
    // a real filesystem: on tmpfs pages cannot be evicted and the residency checks are skipped
    const char * d = getenv("TMPDIR");
    return d && *d ? d : ".";
}

// TEST_TRACE_KEEP=1 keeps the trace files for inspection
static void remove_trace(const std::string & path) {
    const char * keep = getenv("TEST_TRACE_KEEP");
    if (keep && atoi(keep) == 1) {
        printf("  kept %s\n", path.c_str());
        return;
    }
    unlink(path.c_str());
}

static bool page_resident(const void * addr) {
    unsigned char v = 0;
    return mincore(const_cast<void *>(addr), 1, &v) == 0 && (v & 1);
}

//
// the MUL_MAT_ID graph
//

static constexpr int     N_AS     = 8;
static constexpr int     N_USED   = 2;
static constexpr int     LAYER    = 7;
static constexpr int     HOT      = 3; // touched before the run
static constexpr int     COLD     = 5; // evicted before the run
static constexpr int     N_THREAD = 4;
static constexpr int64_t K        = 4096;
static constexpr int64_t N        = 1012; // MXFP4: 2,202,112 bytes per expert, 2 pieces of <= 2 MiB

struct moe_graph {
    int    fd       = -1;
    void * map      = nullptr;
    size_t map_size = 0;
    size_t page     = 4096;
    bool   evictable = false; // the file pages could be dropped (not tmpfs)

    ggml_context *        ctx_w = nullptr;
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf   = nullptr;

    ggml_tensor * as  = nullptr;
    ggml_tensor * b   = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf  = nullptr;

    std::vector<float> b_data;
    std::vector<int32_t> ids_data = { HOT, COLD };

    ~moe_graph() {
        if (buf)   { ggml_backend_buffer_free(buf); }
        if (ctx)   { ggml_free(ctx); }
        if (buf_w) { ggml_backend_buffer_free(buf_w); }
        if (ctx_w) { ggml_free(ctx_w); }
        if (map)   { munmap(map, map_size); }
        if (fd >= 0) { close(fd); }
    }

    const char * slice(int e) const { return (const char *) as->data + e*as->nb[2]; }

    // the page-aligned pieces of one expert as ggml_cpu_moe_prefetch_piece hints them
    std::vector<std::pair<uintptr_t, size_t>> pieces(int e) const {
        std::vector<std::pair<uintptr_t, size_t>> r;
        const size_t nb02 = as->nb[2];
        for (size_t off = 0; off < nb02; off += 2u*1024*1024) {
            const uintptr_t p   = (uintptr_t) slice(e) + off;
            const uintptr_t n   = std::min<size_t>(2u*1024*1024, nb02 - off);
            const uintptr_t beg = p & ~(uintptr_t) (page - 1);
            const uintptr_t end = (p + n + page - 1) & ~(uintptr_t) (page - 1);
            r.emplace_back(beg, end - beg);
        }
        return r;
    }

    // evict the whole file, then read the hot expert's pieces back (through the fd, not the mapping); pages this
    // process has mapped are not evicted, so its page table entries go first
    void set_cache_state() {
        madvise(map, map_size, MADV_DONTNEED);
        fdatasync(fd);
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        std::vector<char> tmp(4u*1024*1024);
        for (const auto & pc : pieces(HOT)) {
            const off_t off = (off_t) (pc.first - (uintptr_t) map);
            (void) !pread(fd, tmp.data(), std::min(pc.second, tmp.size()), off);
            if (pc.second > tmp.size()) {
                (void) !pread(fd, tmp.data(), pc.second - tmp.size(), off + (off_t) tmp.size());
            }
        }
        evictable = !page_resident(slice(COLD)) && !page_resident(slice(COLD) + as->nb[2] - 1);
    }

    bool init(ggml_backend_t backend, ggml_backend_dev_t dev) {
        page = (size_t) sysconf(_SC_PAGESIZE);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        const size_t row_size    = ggml_row_size(GGML_TYPE_MXFP4, K);
        const size_t expert_size = row_size*N;
        std::vector<float> w_f32(K*N*N_AS);
        for (auto & v : w_f32) {
            v = dist(rng);
        }
        std::vector<uint8_t> w_data(expert_size*N_AS);
        ggml_quantize_chunk(GGML_TYPE_MXFP4, w_f32.data(), w_data.data(), 0, N*N_AS, K, nullptr);

        ggml_init_params wp = { ggml_tensor_overhead(), nullptr, true };
        ctx_w = ggml_init(wp);
        as = ggml_new_tensor_3d(ctx_w, GGML_TYPE_MXFP4, K, N, N_AS);
        ggml_set_name(as, "blk.7.ffn_up_exps.weight");

        std::string path = tmp_dir() + "/test-trace-w-XXXXXX";
        fd = mkstemp(path.data());
        if (fd < 0) {
            fprintf(stderr, "mkstemp in %s failed\n", tmp_dir().c_str());
            return false;
        }
        unlink(path.c_str());

        const size_t offs = 4096 + 96; // 32-aligned, not page aligned: the repack / extra path never applies
        map_size = offs + w_data.size();
        std::vector<uint8_t> file(map_size, 0);
        memcpy(file.data() + offs, w_data.data(), w_data.size());
        if (pwrite(fd, file.data(), file.size(), 0) != (ssize_t) file.size()) {
            fprintf(stderr, "pwrite failed\n");
            return false;
        }
        map = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            map = nullptr;
            fprintf(stderr, "mmap failed\n");
            return false;
        }
        // like the lazy ranges of llama_mmap: faults read only the touched pages
        posix_madvise(map, map_size, POSIX_MADV_RANDOM);
        buf_w = ggml_backend_dev_buffer_from_host_ptr(dev, map, map_size, ggml_nbytes(as));
        if (!buf_w || ggml_backend_tensor_alloc(buf_w, as, (char *) map + offs) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "failed to wrap the mapping\n");
            return false;
        }

        ggml_init_params gp = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(gp);
        b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N_USED, 1);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);
        ggml_set_name(b,   "ffn_inp-7");
        ggml_set_name(ids, "ffn_moe_topk-7");
        ggml_set_input(b);
        ggml_set_input(ids);
        out = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_set_name(out, "ffn_moe_up-7");
        ggml_tensor * y = ggml_scale(ctx, out, 0.5f);
        ggml_set_name(y, "ffn_moe_scaled-7");
        ggml_tensor * z = ggml_add(ctx, y, out);
        ggml_set_name(z, "ffn_moe_out-7");
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, z);

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "failed to allocate the graph tensors\n");
            return false;
        }
        b_data.resize(ggml_nelements(b));
        for (auto & v : b_data) {
            v = dist(rng);
        }
        set_inputs();
        return true;
    }

    void set_inputs() {
        ggml_backend_tensor_set(b,   b_data.data(),   0, ggml_nbytes(b));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));
    }

    // nodes the CPU backend computes: not empty ops, and flagged for compute
    int n_computed() const {
        int n = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * t = ggml_graph_node(gf, i);
            const bool empty = t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
                               t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE;
            n += !empty && (t->flags & GGML_TENSOR_FLAG_COMPUTE);
        }
        return n;
    }

    std::vector<float> result() const {
        std::vector<float> r(ggml_nelements(out));
        ggml_backend_tensor_get(out, r.data(), 0, ggml_nbytes(out));
        return r;
    }

    // until the cold expert's pages are in, so that the poller resolves its pieces before the trace stops
    void wait_resident(int e) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            bool all = true;
            for (const auto & pc : pieces(e)) {
                for (size_t o = 0; o < pc.second && all; o += page) {
                    all = page_resident((const char *) pc.first + o);
                }
            }
            if (all) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        // a few poll intervals more
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
};

static ggml_backend_t make_backend(ggml_backend_dev_t * dev) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        return nullptr;
    }
    *dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(*dev);
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads) {
        set_n_threads(backend, N_THREAD);
    }
    return backend;
}

static ggml_trace_params make_params(const std::string & path, int detail) {
    ggml_trace_params p = {};
    p.path            = path.c_str();
    p.n_tokens        = 0;
    p.n_skip          = 0;
    p.max_batch       = 1;
    p.detail          = detail;
    p.poll_us         = 200;
    p.buffer_mb       = 64;
    p.residency_valid = true;
    p.meta            = "\"model\":\"test-trace\"";
    return p;
}

//
// the trace file
//

struct trace_file {
    json meta;
    std::map<std::pair<int, int>, std::string> thread_names; // (pid, tid) -> name
    std::vector<json> ev; // all but metadata / M records
    size_t n_lines = 0;
    std::vector<std::string> lines;

    std::vector<json> sel(const std::function<bool(const json &)> & f) const {
        std::vector<json> r;
        for (const auto & e : ev) {
            if (f(e)) {
                r.push_back(e);
            }
        }
        return r;
    }
};

static bool is(const json & e, const char * cat, const char * ph) {
    return e.value("cat", "") == cat && e.value("ph", "") == ph;
}

static double ts(const json & e) { return e.at("ts").get<double>(); }
static double te(const json & e) { return ts(e) + e.value("dur", 0.0); }

// parses the file as a whole and truncated after lines; checks the common shape of every event
static bool load_trace(const std::string & path, trace_file & tf) {
    std::ifstream in(path);
    if (!in) {
        CHECK(false, "cannot open %s", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();

    json all;
    try {
        all = json::parse(s);
    } catch (const std::exception & e) {
        CHECK(false, "%s does not parse: %s", path.c_str(), e.what());
        return false;
    }
    CHECK(all.is_array() && !all.empty(), "not a JSON array");

    // a truncated file: the prefix up to any line, without its trailing comma, closed with ']'
    size_t pos = 0;
    std::vector<size_t> ends;
    while ((pos = s.find('\n', pos)) != std::string::npos) {
        ends.push_back(++pos);
    }
    tf.n_lines = ends.size();
    const size_t step = std::max<size_t>(1, ends.size()/400);
    for (size_t i = 1; i + 1 < ends.size(); i += step) {
        std::string pre = s.substr(0, ends[i]);
        while (!pre.empty() && (pre.back() == '\n' || pre.back() == ',')) {
            pre.pop_back();
        }
        pre += "]";
        try {
            const json t = json::parse(pre);
            if (!t.is_array()) {
                CHECK(false, "truncated after line %zu: not an array", i);
            }
        } catch (const std::exception & e) {
            CHECK(false, "truncated after line %zu does not parse: %s", i, e.what());
            break;
        }
    }

    for (const auto & e : all) {
        CHECK(e.contains("ph") && e.contains("pid") && e.contains("name"), "event without ph/pid/name: %s", e.dump().c_str());
        const std::string ph = e.value("ph", "");
        if (ph == "M") {
            if (e.value("name", "") == "thread_name") {
                tf.thread_names[{ e.at("pid").get<int>(), e.at("tid").get<int>() }] = e.at("args").at("name").get<std::string>();
            }
            continue;
        }
        if (e.value("name", "") == "meta") {
            tf.meta = e.at("args");
            continue;
        }
        CHECK(e.contains("ts") && e.contains("tid") && e.contains("args"), "event without ts/tid/args: %s", e.dump().c_str());
        // Perfetto puts async b/e spans on global tracks, outside their process: only complete and instant events
        CHECK(ph == "X" || ph == "i", "unexpected ph %s: %s", ph.c_str(), e.dump().c_str());
        if (ph == "X") {
            CHECK(e.contains("dur") && e.at("dur").get<double>() >= 0.0, "X without dur >= 0: %s", e.dump().c_str());
        }
        tf.ev.push_back(e);
    }
    CHECK(!tf.meta.is_null(), "no metadata record");

    // expert compute / expert I/O: spans on named lanes of their process, never overlapping within a lane
    std::map<std::pair<int, int>, std::vector<std::pair<double, double>>> by_lane;
    for (const auto & e : tf.ev) {
        const int pid = e.at("pid").get<int>();
        const std::string cat = e.value("cat", "");
        if (cat == "expert" || cat == "io") {
            CHECK(pid == (cat == "expert" ? 100 : 101), "%s span on pid %d", cat.c_str(), pid);
            CHECK(e.at("ph") == "X", "%s span not a complete event", cat.c_str());
            by_lane[{ pid, e.at("tid").get<int>() }].emplace_back(ts(e), te(e));
        }
    }
    for (auto & [lane, v] : by_lane) {
        const auto it = tf.thread_names.find(lane);
        CHECK(it != tf.thread_names.end(), "lane %d/%d without thread_name", lane.first, lane.second);
        if (it != tf.thread_names.end()) {
            const std::string & n = it->second;
            const bool ok = lane.first == 100 ? (n.rfind("gate", 0) == 0 || n.rfind("up", 0) == 0 || n.rfind("down", 0) == 0 || n.rfind("other", 0) == 0)
                                              : n.find('/') != std::string::npos;
            CHECK(ok, "lane %d/%d named %s", lane.first, lane.second, n.c_str());
        }
        std::sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); ++i) {
            CHECK(v[i].first >= v[i - 1].second - 1e-3, "lane %d/%d: [%.3f, %.3f] overlaps [%.3f, %.3f]", lane.first, lane.second,
                    v[i].first, v[i].second, v[i - 1].first, v[i - 1].second);
        }
    }
    return true;
}

// the issue -> resident spans on the expert I/O process
static std::vector<json> io_spans(const trace_file & tf) {
    std::vector<json> r;
    for (const auto & e : tf.ev) {
        if (e.value("cat", "") == "io" && e.at("pid") == 101) {
            r.push_back(e);
        }
    }
    return r;
}

// one EXPERT span
struct expert_span {
    json   args;
    double t0, t1;
};

// the EXPERT spans of one step, by expert
static std::map<int, expert_span> experts_of(const trace_file & tf, int step = 0) {
    std::map<int, expert_span> r;
    for (const auto & e : tf.ev) {
        if (is(e, "expert", "X") && e.at("args").at("step") == step) {
            const int x = e.at("args").at("expert").get<int>();
            CHECK(r.count(x) == 0, "two EXPERT spans for expert %d", x);
            r[x] = { e.at("args"), ts(e), te(e) };
        }
    }
    return r;
}

//
// cases
//

static void run_traced_token(moe_graph & g, ggml_backend_t backend, int32_t token, int32_t pos) {
    ggml_trace_batch_begin(1, token, pos);
    g.set_inputs(); // XFER input while traced
    CHECK(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS, "compute failed");
    ggml_trace_batch_end();
}

// common checks of one traced token of the MUL_MAT_ID graph
static void check_token(const trace_file & tf, const moe_graph & g, bool prefetch, int detail) {
    // TOKEN
    const auto tokens = tf.sel([](const json & e) { return is(e, "token", "X"); });
    CHECK(tokens.size() == 1, "TOKEN count %zu", tokens.size());
    if (tokens.size() != 1) {
        return;
    }
    const json & tok = tokens[0];
    CHECK(tok.at("args").at("step") == 0, "TOKEN step");
    CHECK(tok.at("args").at("token") == 1234 && tok.at("args").at("pos") == 41, "TOKEN token/pos %s", tok.dump().c_str());
    CHECK(tok.at("args").at("token_out") == 5678, "TOKEN token_out not stamped: %s", tok.dump().c_str());

    // every event that carries a step belongs to step 0
    for (const auto & e : tf.ev) {
        if (e.at("args").contains("step")) {
            CHECK(e.at("args").at("step") == 0, "event outside step 0: %s", e.dump().c_str());
        }
    }

    // OP_CPU: one per computed node or fused group, on worker 0
    const auto ops = tf.sel([](const json & e) { return is(e, "op", "X") && e.at("pid") == 1; });
    int n_nodes = 0;
    const json * mmid = nullptr;
    for (const auto & o : ops) {
        CHECK(o.at("args").at("n_fused").get<int>() >= 1, "n_fused < 1");
        n_nodes += o.at("args").at("n_fused").get<int>();
        CHECK(o.at("tid") == 100, "OP_CPU not on worker 0");
        CHECK(ts(o) >= ts(tok) && te(o) <= te(tok) + 1e-3, "OP_CPU outside the TOKEN span");
        if (o.at("args").at("op") == "MUL_MAT_ID") {
            CHECK(mmid == nullptr, "two MUL_MAT_ID ops");
            mmid = &o;
        }
    }
    CHECK(n_nodes == g.n_computed(), "OP_CPU covers %d nodes, graph computes %d", n_nodes, g.n_computed());
    CHECK(mmid != nullptr, "no MUL_MAT_ID op");
    if (mmid == nullptr) {
        return;
    }
    CHECK(te(*mmid) > ts(*mmid), "MUL_MAT_ID t0 >= t1");
    CHECK(mmid->at("args").at("layer") == LAYER, "MUL_MAT_ID layer %s", mmid->dump().c_str());
    CHECK(mmid->at("args").at("name") == "ffn_moe_up-7", "MUL_MAT_ID name");

    // XFER of the inputs while traced
    const auto xfer = tf.sel([](const json & e) { return is(e, "xfer", "X"); });
    std::set<std::string> xnames;
    for (const auto & x : xfer) {
        xnames.insert(x.at("args").at("name").get<std::string>());
        CHECK(x.at("args").at("direction") == "input", "XFER direction");
    }
    CHECK(xnames.count("ffn_inp-7") && xnames.count("ffn_moe_topk-7"), "XFER of the inputs missing");

    // EXPERT: exactly the used ones, inside the op
    const auto ex = experts_of(tf);
    CHECK(ex.size() == 2 && ex.count(HOT) && ex.count(COLD), "EXPERT set");
    for (const auto & [x, sp] : ex) {
        CHECK(sp.args.at("layer") == LAYER, "EXPERT layer");
        CHECK(sp.args.at("bytes").get<int64_t>() == (int64_t) g.as->nb[2], "EXPERT bytes");
        CHECK(sp.args.at("stall_ns").get<int64_t>() >= 0, "EXPERT stall < 0");
        CHECK(ts(*mmid) <= sp.t0 + 1e-3 && sp.t0 < sp.t1 && sp.t1 <= te(*mmid) + 1e-3,
                "EXPERT %d [%.3f, %.3f] outside op [%.3f, %.3f]", x, sp.t0, sp.t1, ts(*mmid), te(*mmid));
        CHECK(sp.args.at("n_threads") == N_THREAD, "EXPERT n_threads");
        CHECK(!sp.args.contains("partial"), "EXPERT partial");
        // the residency sample, joined by key
        CHECK(sp.args.at("resident_frac").is_number(), "EXPERT %d without residency", x);
        if (sp.args.at("resident_frac").is_number()) {
            const double rf = sp.args.at("resident_frac").get<double>();
            CHECK(rf >= 0.0 && rf <= 1.0, "resident_frac %f", rf);
            CHECK(sp.args.at("sample_delay_ns").get<int64_t>() >= 0, "sample_delay < 0");
        }
    }
    const auto res = tf.sel([](const json & e) { return is(e, "residency", "i"); });
    std::map<int, int> n_res;
    for (const auto & r : res) {
        n_res[r.at("args").at("expert").get<int>()]++;
        CHECK(r.at("args").at("layer") == LAYER, "residency layer");
        CHECK(r.at("args").at("name") == "ffn_moe_up-7", "residency key name");
    }
    CHECK(n_res.size() == 2 && n_res[HOT] == 1 && n_res[COLD] == 1, "one EXPERT_RESIDENCY per EXPERT");
    if (g.evictable && ex.count(HOT)) {
        CHECK(ex.at(HOT).args.at("resident_frac") == 1.0, "hot expert not resident at compute start");
    }

    // level 2: per-thread spans whose envelope is the EXPERT span
    const auto et = tf.sel([](const json & e) { return is(e, "expert_thread", "X"); });
    if (detail >= 2) {
        CHECK(et.size() == (size_t) N_THREAD*N_USED, "EXPERT_THREAD count %zu", et.size());
        for (const auto & [x, sp] : ex) {
            double t0 = 1e300, t1 = -1e300;
            std::set<int> tids;
            for (const auto & t : et) {
                if (t.at("args").at("expert") == x) {
                    t0 = std::min(t0, ts(t));
                    t1 = std::max(t1, te(t));
                    tids.insert(t.at("tid").get<int>());
                }
            }
            CHECK(tids.size() == (size_t) N_THREAD, "EXPERT_THREAD tracks for %d", x);
            CHECK(std::fabs(t0 - sp.t0) < 2e-3 && std::fabs(t1 - sp.t1) < 2e-3,
                    "EXPERT %d [%.3f, %.3f] != thread envelope [%.3f, %.3f]", x, sp.t0, sp.t1, t0, t1);
        }
    } else {
        CHECK(et.empty(), "EXPERT_THREAD at detail 1");
    }

    // I/O issues of the prefetch on the issuing threads' tracks
    const auto iss = tf.sel([](const json & e) { return is(e, "io_issue", "X"); });
    if (!prefetch) {
        CHECK(iss.empty(), "IO_ISSUE without prefetch");
        return;
    }
    std::map<int, std::vector<json>> by_expert;
    for (const auto & i : iss) {
        const json & a = i.at("args");
        CHECK(a.at("source") == "prefetch" && a.at("layer") == LAYER, "IO_ISSUE source/layer %s", i.dump().c_str());
        CHECK(a.at("name") == "blk.7.ffn_up_exps.weight", "IO_ISSUE name");
        CHECK(te(i) >= ts(i), "IO_ISSUE t0 > t1");
        CHECK(i.at("tid").get<int>() >= 100 && i.at("tid").get<int>() < 100 + N_THREAD, "IO_ISSUE not on a worker track");
        by_expert[a.at("expert").get<int>()].push_back(i);
    }
    CHECK(by_expert.size() == 2 && by_expert.count(HOT) && by_expert.count(COLD), "IO_ISSUE experts");
    for (auto & [x, v] : by_expert) {
        const auto pcs = g.pieces(x);
        CHECK(v.size() == pcs.size(), "expert %d: %zu pieces issued, %zu expected", x, v.size(), pcs.size());
        int64_t sum = 0, want = 0;
        std::set<int> idx;
        for (const auto & i : v) {
            sum += i.at("args").at("bytes").get<int64_t>();
            idx.insert(i.at("args").at("piece").get<int>());
        }
        for (const auto & pc : pcs) {
            want += (int64_t) pc.second;
        }
        CHECK(sum == want, "expert %d: piece bytes %lld, page-rounded pieces %lld", x, (long long) sum, (long long) want);
        CHECK(idx == std::set<int>({ 0, 1 }), "expert %d: piece indices", x);
        if (g.evictable) {
            for (const auto & i : v) {
                const bool cached = i.at("args").at("cached").get<bool>();
                CHECK(cached == (x == HOT), "expert %d piece %d cached=%d", x, i.at("args").at("piece").get<int>(), cached);
            }
        }
    }

    // the I/O tracks: one issue -> resident span per slice (detail 1) or piece (detail 2), ending at residency
    const auto spans = io_spans(tf);
    for (const auto & sp : spans) {
        CHECK(!sp.at("args").contains("partial"), "partial I/O span %s", sp.dump().c_str());
        if (sp.at("args").contains("t_resident_ns")) {
            CHECK(std::fabs(sp.at("args").at("t_resident_ns").get<double>()/1000.0 - te(sp)) < 2e-3 || te(sp) > sp.at("args").at("t_resident_ns").get<double>()/1000.0,
                    "I/O span does not end at residency: %s", sp.dump().c_str());
        }
    }
    if (detail == 1) {
        // one merged slice span per used expert
        CHECK(spans.size() == 2, "I/O slice spans %zu", spans.size());
        for (const auto & b : spans) {
            const int x = b.at("args").at("expert").get<int>();
            CHECK(b.at("args").at("n_pieces") == 2, "slice n_pieces");
            if (g.evictable) {
                CHECK(b.at("args").at("cached").get<bool>() == (x == HOT), "slice %d cached", x);
            }
            if (x == COLD && !b.at("args").at("unresolved").get<bool>()) {
                CHECK(b.at("args").contains("t_prev_poll_ns"), "resolved slice without t_prev_poll");
            }
        }
    } else {
        std::map<int64_t, json> e_by_id;
        for (const auto & sp : spans) {
            CHECK(e_by_id.count(sp.at("args").at("cookie").get<int64_t>()) == 0, "two spans of one cookie");
            e_by_id[sp.at("args").at("cookie").get<int64_t>()] = sp;
        }
        // per piece by cookie: every non-duplicate issue has exactly one resolution
        int n_issued = 0, n_cached = 0, n_resolved = 0, n_unresolved = 0;
        for (const auto & i : iss) {
            const json & a = i.at("args");
            if (a.at("duplicate").get<bool>()) {
                continue;
            }
            n_issued++;
            const int64_t c = a.at("cookie").get<int64_t>();
            CHECK(e_by_id.count(c) == 1, "cookie %lld without its span", (long long) c);
            if (!e_by_id.count(c)) {
                continue;
            }
            const json & e = e_by_id.at(c);
            const std::string r = e.at("args").at("resolution");
            n_cached     += r == "cached";
            n_resolved   += r == "resolved";
            n_unresolved += r == "unresolved";
            CHECK(r == "cached" || te(e) >= te(i) - 1e-3, "resident before issue end: %s", e.dump().c_str());
            if (r == "resolved") {
                CHECK(e.at("args").at("t_prev_poll_ns").get<double>()/1000.0 <= te(e) + 1e-3, "t_prev_poll > t_this_poll");
            }
            if (g.evictable) {
                CHECK((r == "cached") == (a.at("expert") == HOT), "resolution %s for expert %d", r.c_str(), a.at("expert").get<int>());
            }
        }
        CHECK(n_issued == 4, "non-duplicate pieces %d", n_issued);
        CHECK(n_cached + n_resolved + n_unresolved == n_issued, "resolutions %d + %d + %d != %d", n_cached, n_resolved, n_unresolved, n_issued);
    }
}


// the default (simple) layout: rows token / worker N / load N / copy N of process "simple", spans named exactly
// "L<layer> e<expert>" or "L<layer>" (or input / output for copies); the per-step counts are in the token's args
static void check_simple(const trace_file & tf, const moe_graph & g) {
    CHECK(tf.meta.at("layout") == "simple", "layout %s", tf.meta.at("layout").dump().c_str());
    std::map<std::string, std::vector<json>> rows;
    for (const auto & e : tf.ev) {
        if (e.value("cat", "") == "meta") {
            continue;
        }
        CHECK(e.at("ph") == "X" && e.value("cat", "") == "simple" && e.at("pid") == 1, "not a simple span: %s", e.dump().c_str());
        const auto it = tf.thread_names.find({ 1, e.at("tid").get<int>() });
        CHECK(it != tf.thread_names.end(), "span on an unnamed row: %s", e.dump().c_str());
        if (it != tf.thread_names.end()) {
            rows[it->second].push_back(e);
        }
    }
    std::set<std::string> names;
    for (auto & kv : rows) {
        names.insert(kv.first);
        auto & v = kv.second;
        std::sort(v.begin(), v.end(), [](const json & a, const json & b) { return ts(a) < ts(b); });
        for (size_t i = 1; i < v.size(); ++i) {
            CHECK(ts(v[i]) >= te(v[i - 1]) - 1e-3, "row %s: %s overlaps %s", kv.first.c_str(), v[i].dump().c_str(), v[i - 1].dump().c_str());
        }
    }
    std::set<std::string> want = { "token", "copy 0" };
    for (int w = 0; w < N_THREAD; ++w) {
        want.insert("worker " + std::to_string(w));
    }
    if (g.evictable) {
        want.insert("load 0"); // the cold expert is read; the hot one is cached at issue and has no load span
    }
    CHECK(names == want, "rows: got %zu, expected %zu", names.size(), want.size());
    for (const auto & n : names) {
        CHECK(want.count(n), "unexpected row %s", n.c_str());
    }

    // token: the exact counts
    CHECK(rows["token"].size() == 1, "token spans %zu", rows["token"].size());
    if (rows["token"].size() != 1) {
        return;
    }
    const json & tok = rows["token"][0];
    const json & ta  = tok.at("args");
    CHECK(tok.at("name") == "token 0 -> 5678", "token span %s", tok.at("name").dump().c_str());
    CHECK(ta.at("token") == 1234 && ta.at("pos") == 41 && ta.at("token_out") == 5678, "token args");
    CHECK(ta.at("n_experts") == N_USED && ta.at("n_expert_threads") == N_USED*N_THREAD && ta.at("n_residency") == N_USED, "expert counts %s", ta.dump().c_str());
    CHECK(ta.at("n_pieces_prefetch") == 4 && ta.at("n_pieces_dup") == 0, "piece counts %s", ta.dump().c_str());
    if (g.evictable) {
        CHECK(ta.at("n_pieces_cached") == 2 && ta.at("n_loads") == 2 && ta.at("n_loads_cached") == 1, "cached counts %s", ta.dump().c_str());
    }
    // the same transfers as XFER in the detailed layout: the two inputs, 32768 + 8 bytes
    CHECK(ta.at("n_copies") == 2 && ta.at("copy_bytes") == 32768 + 8, "copy counts %s", ta.dump().c_str());

    // workers: each thread computes both experts once; worker 0 also the layer's other ops (scale, add), merged
    std::set<std::string> worker_names;
    for (int w = 0; w < N_THREAD; ++w) {
        std::map<std::string, int> n;
        for (const auto & e : rows["worker " + std::to_string(w)]) {
            n[e.at("name").get<std::string>()]++;
            if (e.at("name").get<std::string>().find(" e") != std::string::npos) {
                worker_names.insert(e.at("name").get<std::string>());
                const json & a = e.at("args");
                CHECK(a.at("layer") == LAYER && a.at("tensor") == "ffn_moe_up-7" && a.at("bytes").get<int64_t>() == (int64_t) g.as->nb[2] &&
                      a.at("stall_ns").get<int64_t>() >= 0 && a.at("n_threads") == N_THREAD, "worker span args %s", a.dump().c_str());
                CHECK(ts(e) >= ts(tok) && te(e) <= te(tok) + 1e-3, "worker span outside the token");
            }
        }
        CHECK(n["L7 e3"] == 1 && n["L7 e5"] == 1, "worker %d expert spans", w);
        if (w == 0) {
            CHECK(n["L7"] == 1 && n.size() == 3, "worker 0 rows: %zu names, %d L7", n.size(), n["L7"]);
        } else {
            CHECK(n.size() == 2, "worker %d: %zu names", w, n.size());
        }
    }

    // loads: the cold expert from issue to resident, named like its compute spans
    std::set<std::string> load_names;
    for (const auto & kv : rows) {
        if (kv.first.rfind("load ", 0) == 0) {
            for (const auto & e : kv.second) {
                load_names.insert(e.at("name").get<std::string>());
                const json & a = e.at("args");
                CHECK(a.at("pieces") == 2 && a.at("sources") == "prefetch" && a.at("tensors") == 1 && a.at("unresolved") == false,
                        "load args %s", a.dump().c_str());
            }
        }
    }
    for (const auto & n : load_names) {
        CHECK(worker_names.count(n), "load %s has no compute span of the same name", n.c_str());
    }
    if (g.evictable) {
        CHECK(load_names == std::set<std::string>({ "L7 e5" }), "load names %zu", load_names.size());
    }

    // copies
    int64_t bytes = 0;
    for (const auto & e : rows["copy 0"]) {
        bytes += e.at("args").at("bytes").get<int64_t>();
        CHECK(e.at("name") == "L7" && e.at("args").at("direction") == "h2d", "copy span %s", e.dump().c_str());
    }
    CHECK(rows["copy 0"].size() == 2 && bytes == 32768 + 8, "copy row: %zu spans, %lld bytes", rows["copy 0"].size(), (long long) bytes);
}

static int case_token(bool prefetch, int detail) {
    ggml_backend_dev_t dev;
    ggml_backend_t backend = make_backend(&dev);
    moe_graph g;
    if (!backend || !g.init(backend, dev)) {
        return 1;
    }
    // untraced reference output first (the trace must not change numerics)
    CHECK(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS, "compute failed");
    const std::vector<float> ref = g.result();
    g.set_cache_state();
    if (!g.evictable) {
        printf("  file pages could not be dropped (tmpfs?): residency checks skipped\n");
    }

    const std::string path = tmp_dir() + "/test-trace-" + std::to_string(getpid()) + ".json";
    ggml_trace_params p = make_params(path, detail);
    CHECK(ggml_trace_start(&p), "start failed");
    CHECK(ggml_trace_enabled() && !ggml_trace_active(), "predicates in IDLE");

    run_traced_token(g, backend, 1234, 41);
    if (prefetch) {
        g.wait_resident(COLD);
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // the residency samples
    }
    CHECK(ggml_trace_stamp_token_out(5678) == 0, "stamp did not hit step 0");
    CHECK(ggml_trace_stamp_token_out(9999) == -1, "second stamp hit a token");
    const std::vector<float> got = g.result();
    ggml_trace_stop();
    CHECK(!ggml_trace_enabled(), "enabled after stop");

    CHECK(got.size() == ref.size() && memcmp(got.data(), ref.data(), ref.size()*sizeof(float)) == 0, "traced result differs");

    trace_file tf;
    if (load_trace(path, tf)) {
        if (detail == 1) {
            check_simple(tf, g);
        } else {
            check_token(tf, g, prefetch, detail);
        }
        CHECK(tf.meta.at("tokens_traced") == 1 && tf.meta.at("dropped") == 0 && tf.meta.at("ring_drops") == 0, "meta counters %s", tf.meta.dump().c_str());
        CHECK(tf.meta.at("config").at("detail") == detail, "meta detail");

        if (!prefetch && g.evictable) {
            // nothing hinted: the cold expert faults in during compute
            const auto ex = experts_of(tf);
            if (ex.count(HOT) && ex.count(COLD)) {
                const double rf = ex.at(COLD).args.at("resident_frac").is_number() ? ex.at(COLD).args.at("resident_frac").get<double>() : 1.0;
                CHECK(rf < 1.0, "cold expert resident at compute start (%f)", rf);
                const int64_t s3 = ex.at(HOT).args.at("stall_ns").get<int64_t>();
                const int64_t s5 = ex.at(COLD).args.at("stall_ns").get<int64_t>();
                CHECK(s5 > s3, "cold stall %lld <= hot stall %lld", (long long) s5, (long long) s3);
                printf("  cold expert: resident %.3f at compute start, stall %.3f ms (hot %.3f ms)\n", rf, s5/1e6, s3/1e6);
            }
        }
    }
    remove_trace(path);
    ggml_backend_free(backend);
    return g_failures ? 1 : (g.evictable ? 0 : 77);
}

static int case_state() {
    ggml_backend_dev_t dev;
    ggml_backend_t backend = make_backend(&dev);
    moe_graph g;
    if (!backend || !g.init(backend, dev)) {
        return 1;
    }

    // off: no file, both predicates false, and the same numerics as traced runs
    CHECK(!ggml_trace_enabled() && !ggml_trace_active(), "predicates without start");
    const std::string path = tmp_dir() + "/test-trace-state-" + std::to_string(getpid()) + ".json";
    unlink(path.c_str());
    CHECK(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS, "compute failed");
    CHECK(access(path.c_str(), F_OK) != 0, "file exists before start");
    const std::vector<float> ref = g.result();
    g.set_cache_state();

    ggml_trace_params p = make_params(path, 2);
    p.n_skip   = 1;
    p.n_tokens = 2;
    CHECK(ggml_trace_start(&p), "start failed");
    CHECK(ggml_trace_start(&p), "second start (refcount) failed");
    ggml_trace_stop(); // still one reference
    CHECK(ggml_trace_enabled(), "refcounted stop closed the trace");

    // skipped decode-size batch, then a prompt batch: neither traced, the prompt batch does not count as a skip
    ggml_trace_batch_begin(1, 1, 0);
    CHECK(!ggml_trace_active() && ggml_trace_enabled(), "skipped batch traced");
    CHECK(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS, "compute failed");
    ggml_trace_batch_end();
    ggml_trace_batch_begin(40, 2, 1);
    CHECK(!ggml_trace_active(), "40-token batch traced");
    CHECK(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS, "compute failed");
    // a predictor hint during a prompt batch is not recorded
    CHECK(!ggml_trace_io_enabled(), "I/O hints enabled while SUSPENDED");
    {
        const auto pc = g.pieces(7);
        const int64_t c = ggml_trace_io_issue(ggml_trace_step(), (const void *) pc[0].first, pc[0].second, GGML_TRACE_IO_PREDICT_LOW,
                0, LAYER, 7, "blk.7.ffn_up_exps.weight", 0, 2, ggml_trace_now_ns(), ggml_trace_now_ns() + 1, 4096, GGML_TRACE_TID_PREDICTOR);
        CHECK(c == 0, "hint during a prompt batch got cookie %lld", (long long) c);
    }
    ggml_trace_batch_end();
    CHECK(ggml_trace_io_enabled(), "I/O hints disabled in IDLE");

    // an issue in IDLE
    CHECK(ggml_trace_step() == -1, "step in IDLE");
    const auto pcs = g.pieces(6); // expert 6 is never used: it stays cold
    int64_t t0 = ggml_trace_now_ns();
    int64_t t1 = ggml_trace_now_ns();
    const int64_t c0 = ggml_trace_io_issue(ggml_trace_step(), (const void *) pcs[0].first, pcs[0].second, GGML_TRACE_IO_PREDICT_LOW, 3, LAYER, 6,
            "blk.7.ffn_up_exps.weight", 0, 2, t0, t1, 4096, GGML_TRACE_TID_PREDICTOR);
    CHECK(c0 > 0, "issue cookie");

    // two traced tokens; the second issue of the same range is a duplicate with the original cookie
    run_traced_token(g, backend, 10, 50);
    CHECK(ggml_trace_last_step() == 0 && ggml_trace_step() == -1, "last step after the first token");
    t0 = ggml_trace_now_ns();
    t1 = ggml_trace_now_ns();
    const int64_t c1 = ggml_trace_io_issue(ggml_trace_step(), (const void *) pcs[0].first, pcs[0].second, GGML_TRACE_IO_PREDICT_HIGH, -1, LAYER, 6,
            "blk.7.ffn_up_exps.weight", 0, 2, t0, t1, 4096, GGML_TRACE_TID_PREDICTOR);
    if (g.evictable) {
        CHECK(c1 == c0, "duplicate cookie %lld != %lld", (long long) c1, (long long) c0);
    }
    CHECK(ggml_trace_stamp_token_out(11) == 0, "stamp of token 0");
    run_traced_token(g, backend, 11, 51);
    CHECK(ggml_trace_stamp_token_out(12) == 1, "stamp of token 1");
    const std::vector<float> got = g.result();

    // TOKENS reached: the next decode closes the file
    ggml_trace_batch_begin(1, 12, 52);
    CHECK(!ggml_trace_enabled() && !ggml_trace_active(), "trace still on after TOKENS");
    CHECK(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS, "compute failed");
    ggml_trace_batch_end();
    ggml_trace_stop();

    CHECK(memcmp(got.data(), ref.data(), ref.size()*sizeof(float)) == 0, "traced result differs from untraced");

    trace_file tf;
    if (load_trace(path, tf)) {
        const auto tokens = tf.sel([](const json & e) { return is(e, "token", "X"); });
        CHECK(tokens.size() == 2, "TOKEN count %zu", tokens.size());
        for (const auto & t : tokens) {
            const int s = t.at("args").at("step").get<int>();
            CHECK((s == 0 && t.at("args").at("token") == 10 && t.at("args").at("pos") == 50 && t.at("args").at("token_out") == 11) ||
                  (s == 1 && t.at("args").at("token") == 11 && t.at("args").at("pos") == 51 && t.at("args").at("token_out") == 12),
                    "TOKEN %s", t.dump().c_str());
        }
        // compute events only for the traced tokens: 2 x the computed nodes
        int n_nodes = 0;
        for (const auto & o : tf.sel([](const json & e) { return is(e, "op", "X"); })) {
            n_nodes += o.at("args").at("n_fused").get<int>();
            CHECK(o.at("args").at("step").get<int>() >= 0, "OP outside a traced step");
        }
        CHECK(n_nodes == 2*g.n_computed(), "OP_CPU nodes %d, expected %d", n_nodes, 2*g.n_computed());
        CHECK(experts_of(tf, 0).size() == 2 && experts_of(tf, 1).size() == 2, "EXPERT spans per traced token");
        CHECK(tf.sel([](const json & e) { return is(e, "expert", "X"); }).size() == 4, "EXPERT spans outside the traced tokens");

        const auto iss = tf.sel([](const json & e) { return is(e, "io_issue", "X"); });
        CHECK(iss.size() == 2, "IO_ISSUE count %zu", iss.size());
        if (!g.evictable) {
            printf("  file pages could not be dropped (tmpfs?): duplicate checks skipped\n");
            remove_trace(path);
            ggml_backend_free(backend);
            return g_failures ? 1 : 77;
        }
        int n_dup = 0;
        for (const auto & i : iss) {
            const json & a = i.at("args");
            CHECK(a.at("cookie").get<int64_t>() == c0, "cookie");
            if (a.at("duplicate").get<bool>()) {
                n_dup++;
                CHECK(a.at("step") == -1 && a.at("source") == "predict_high" && a.at("rank").is_null(), "duplicate issue %s", i.dump().c_str());
            } else {
                CHECK(a.at("step") == -1 && a.at("source") == "predict_low" && a.at("rank") == "r3", "IDLE issue %s", i.dump().c_str());
            }
            CHECK(i.at("tid") == 2, "predictor track");
        }
        CHECK(n_dup == 1, "duplicates %d", n_dup);
        for (const auto & e : tf.ev) {
            CHECK(!(e.at("args").contains("expert") && e.at("args").at("expert") == 7), "event of the prompt-batch hint: %s", e.dump().c_str());
        }
        // one resolution for the cookie: unresolved at stop (the range was never read)
        int n_e = 0;
        for (const auto & e : tf.ev) {
            if (e.at("pid") == 101 && e.value("cat", "") == "io" && e.at("args").at("cookie") == c0) {
                n_e++;
                CHECK(e.at("args").at("resolution") == "unresolved", "resolution %s", e.dump().c_str());
            }
        }
        CHECK(n_e == 1, "resolutions of the cookie %d", n_e);
        CHECK(tf.meta.at("tokens_traced") == 2, "meta tokens_traced");
    }
    remove_trace(path);
    ggml_backend_free(backend);
    return g_failures ? 1 : 0;
}

static int case_plan() {
    ggml_backend_dev_t dev;
    ggml_backend_t backend = make_backend(&dev);
    moe_graph g;
    if (!backend || !g.init(backend, dev)) {
        return 1;
    }
    // planned with the trace off: the first-arrival flags must be reserved anyway
    ggml_cplan plan = ggml_graph_plan(g.gf, N_THREAD, nullptr);
    std::vector<uint8_t> work(plan.work_size + 64);
    plan.work_data = work.data();

    const std::string path = tmp_dir() + "/test-trace-plan-" + std::to_string(getpid()) + ".json";
    ggml_trace_params p = make_params(path, 2);
    CHECK(ggml_trace_start(&p), "start failed");
    ggml_trace_batch_begin(1, 7, 3);
    CHECK(ggml_graph_compute(g.gf, &plan) == GGML_STATUS_SUCCESS, "compute failed");
    ggml_trace_batch_end();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ggml_trace_stop();

    trace_file tf;
    if (load_trace(path, tf)) {
        const auto ex = experts_of(tf);
        CHECK(ex.size() == 2 && ex.count(HOT) && ex.count(COLD), "EXPERT spans with a plan made while off: %zu", ex.size());
        // the trace ran without the token being sampled: emitted at the next begin or at stop, unstamped
        const auto tokens = tf.sel([](const json & e) { return is(e, "token", "X"); });
        CHECK(tokens.size() == 1 && tokens[0].at("args").at("token_out") == -1, "unstamped TOKEN");
    }
    remove_trace(path);
    ggml_backend_free(backend);
    return g_failures ? 1 : 0;
}

static int case_buffer() {
    const std::string path = tmp_dir() + "/test-trace-buffer-" + std::to_string(getpid()) + ".json";
    ggml_trace_params p = make_params(path, 2);
    p.buffer_mb = 1;
    CHECK(ggml_trace_start(&p), "start failed");
    ggml_trace_batch_begin(1, 0, 0);

    // names: a long name keeps head and tail and is cut on UTF-8 boundaries
    {
        ggml_trace_event ev;
        ggml_trace_event_init(&ev, GGML_TRACE_OP_CPU, 0);
        const std::string longname = "blk.12.\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4\xc3\xa4-12";
        ggml_trace_event_set_name(&ev, longname.c_str());
        const std::string got = ev.name;
        CHECK(got.size() < sizeof(ev.name) && got.rfind("blk.12.", 0) == 0 && got.size() >= 3 && got.substr(got.size() - 3) == "-12",
                "long name kept as %s", got.c_str());
        bool utf8_ok = true;
        for (size_t i = 0; i < got.size(); ) {
            const unsigned char c = (unsigned char) got[i];
            const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
            if (n == 0 || i + n > got.size()) {
                utf8_ok = false;
                break;
            }
            for (size_t k = 1; k < n; ++k) {
                utf8_ok = utf8_ok && (((unsigned char) got[i + k]) & 0xC0) == 0x80;
            }
            i += n;
        }
        CHECK(utf8_ok, "truncated name is not valid UTF-8");
    }

    // layers: unnamed nodes inherit the previous node's layer of the emitting thread
    for (const char * name : { "attn_norm-5", "node_17", "ffn_out-6", "node_18" }) {
        ggml_trace_event ev;
        ggml_trace_event_init(&ev, GGML_TRACE_OP_CPU, 0);
        ev.t0_ns = ggml_trace_now_ns();
        ev.t1_ns = ev.t0_ns + 10;
        ggml_trace_event_set_name(&ev, name);
        ggml_trace_emit(&ev);
    }

    const int n_ev = 200000;
    for (int i = 0; i < n_ev; ++i) {
        ggml_trace_event ev;
        ggml_trace_event_init(&ev, GGML_TRACE_PREDICT, GGML_TRACE_TID_PREDICTOR);
        ev.t0_ns = ggml_trace_now_ns();
        ev.t1_ns = ev.t0_ns + 10;
        ev.layer = i % 93;
        ggml_trace_emit(&ev);
    }

    // a full residency ring: requests from 4 threads at once, far faster than the poller samples them
    std::vector<char> mem(1u << 20, 1);
    const int n_req = 50000;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> thr;
    for (int t = 0; t < 4; ++t) {
        thr.emplace_back([&, t] {
            for (int i = 0; i < n_req/4; ++i) {
                ggml_trace_residency_request(LAYER, (t*n_req/4 + i) % N_AS, "ffn_moe_up-7", mem.data() + (i % 256)*4096, 4096, ggml_trace_now_ns());
            }
        });
    }
    for (auto & t : thr) {
        t.join();
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ms < 2000.0, "residency requests blocked (%.1f ms)", ms);

    ggml_trace_batch_end();
    ggml_trace_stop();

    trace_file tf;
    if (load_trace(path, tf)) {
        const int64_t emitted = tf.meta.at("records_emitted").get<int64_t>();
        const int64_t dropped = tf.meta.at("dropped").get<int64_t>();
        const int64_t ring    = tf.meta.at("ring_drops").get<int64_t>();
        const int64_t n_pred  = (int64_t) tf.sel([](const json & e) { return e.value("cat", "") == "predict"; }).size();
        const int64_t n_res   = (int64_t) tf.sel([](const json & e) { return is(e, "residency", "i"); }).size();
        const int64_t n_tok   = (int64_t) tf.sel([](const json & e) { return is(e, "token", "X"); }).size();
        const auto ops = tf.sel([](const json & e) { return is(e, "op", "X"); });
        const int64_t n_ops = (int64_t) ops.size();
        std::map<std::string, std::pair<int, bool>> lay;
        for (const auto & o : ops) {
            lay[o.at("name").get<std::string>()] = { o.at("args").at("layer").get<int>(), o.at("args").value("layer_inferred", false) };
        }
        CHECK(n_ops == 4, "OP events %lld", (long long) n_ops);
        CHECK(lay["attn_norm-5"] == std::make_pair(5, false) && lay["node_17"] == std::make_pair(5, true) &&
              lay["ffn_out-6"] == std::make_pair(6, false) && lay["node_18"] == std::make_pair(6, true), "layer inference");
        const int64_t n_adapt = (int64_t) tf.sel([](const json & e) { return is(e, "meta", "i") && e.value("name", "") == "poll interval"; }).size();
        CHECK(n_adapt == tf.meta.at("poll_adaptations").get<int64_t>() || dropped > 0, "poll adaptations");
        printf("  buffer: %lld records emitted, %lld dropped, %lld PREDICT + %lld residency + %lld TOKEN written, %lld ring drops\n",
                (long long) emitted, (long long) dropped, (long long) n_pred, (long long) n_res, (long long) n_tok, (long long) ring);
        CHECK(dropped > 0, "a 1 MB pool did not overflow");
        const int64_t adapt = tf.meta.at("poll_adaptations").get<int64_t>();
        CHECK(emitted == 4 + n_ev + (n_req - ring) + 1 + adapt, "emitted %lld, expected %lld", (long long) emitted, (long long) (4 + n_ev + n_req - ring + 1 + adapt));
        CHECK(dropped == emitted - (n_ops + n_pred + n_res + n_tok + n_adapt), "dropped %lld != emitted %lld - written %lld",
                (long long) dropped, (long long) emitted, (long long) (n_ops + n_pred + n_res + n_tok + n_adapt));
        CHECK(ring > 0, "the residency ring never filled");
    }
    remove_trace(path);
    return g_failures ? 1 : 0;
}


// lanes are pruned of old intervals: a load that is still pending (written at the end, unresolved) must not land on
// a lane next to an earlier load it overlaps, also when spans more than 10 s later have been written in between
static int case_lanes() {
    const size_t mb = 1u << 20;
    char * buf = (char *) mmap(nullptr, 4*mb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(buf != MAP_FAILED, "mmap");
    memset(buf + 2*mb, 1, 3*4096); // B and C resident, A never touched
    const std::string path = tmp_dir() + "/test-trace-lanes-" + std::to_string(getpid()) + ".json";
    ggml_trace_params p = make_params(path, 1);
    CHECK(ggml_trace_start(&p), "start failed");
    ggml_trace_batch_begin(1, 1, 0);
    const int64_t t = ggml_trace_now_ns();
    const int64_t late = t + 20000000000LL; // a span 20 s later (faked time)
    ggml_trace_io_issue(0, buf,                4096, GGML_TRACE_IO_PREFETCH, -1, 3, 5, "blk.3.ffn_up_exps.weight", 0, 1, t,      t + 1000, 4096, 0);
    ggml_trace_io_issue(0, buf + 2*mb,         4096, GGML_TRACE_IO_PREFETCH, -1, 3, 6, "blk.3.ffn_up_exps.weight", 0, 1, t + 10, t + 1000, 4096, 0);
    ggml_trace_io_issue(0, buf + 2*mb + 8192,  4096, GGML_TRACE_IO_PREFETCH, -1, 3, 7, "blk.3.ffn_up_exps.weight", 0, 1, late,   late + 1000, 4096, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ggml_trace_batch_end();
    ggml_trace_stamp_token_out(2);
    // B and C resolve and are written, then the 20 s span lets the lanes prune; A is written at stop
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    ggml_trace_stop();
    munmap(buf, 4*mb);

    trace_file tf;
    if (load_trace(path, tf)) {
        std::map<std::string, std::vector<json>> rows;
        for (const auto & e : tf.ev) {
            if (e.value("cat", "") == "simple") {
                rows[tf.thread_names[{ 1, e.at("tid").get<int>() }]].push_back(e);
            }
        }
        int n_loads = 0, n_unresolved = 0;
        for (auto & kv : rows) {
            if (kv.first.rfind("load ", 0) != 0) {
                continue;
            }
            auto & v = kv.second;
            std::sort(v.begin(), v.end(), [](const json & a, const json & b) { return ts(a) < ts(b); });
            for (size_t i = 1; i < v.size(); ++i) {
                CHECK(ts(v[i]) >= te(v[i - 1]) - 1e-3, "%s: %s overlaps %s", kv.first.c_str(), v[i].dump().c_str(), v[i - 1].dump().c_str());
            }
            for (const auto & e : v) {
                n_loads++;
                n_unresolved += e.at("args").at("unresolved").get<bool>();
            }
        }
        CHECK(n_loads == 3 && n_unresolved == 1, "loads %d, unresolved %d", n_loads, n_unresolved);
    }
    remove_trace(path);
    return g_failures ? 1 : 0;
}

//
// driver: one process per case
//

static int run_child(const char * self, const char * name, const std::vector<std::string> & env_add) {
    std::vector<std::string> env_s;
    for (char ** e = environ; *e; ++e) {
        if (strncmp(*e, "GGML_CPU_MOE_PREFETCH", 21) != 0) {
            env_s.emplace_back(*e);
        }
    }
    env_s.insert(env_s.end(), env_add.begin(), env_add.end());
    std::vector<char *> envp;
    for (auto & s : env_s) {
        envp.push_back(s.data());
    }
    envp.push_back(nullptr);

    std::string a0 = self, a1 = "--case", a2 = name;
    char * argv[] = { a0.data(), a1.data(), a2.data(), nullptr };
    pid_t pid;
    if (posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, argv, envp.data()) != 0) {
        fprintf(stderr, "posix_spawn failed\n");
        return 1;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

int main(int argc, char ** argv) {
    ggml_log_set(log_cb, nullptr);

    if (argc == 3 && strcmp(argv[1], "--case") == 0) {
        ggml_backend_load_all();
        const std::string c = argv[2];
        int rc = 1;
        if (c == "prefetch") { rc = case_token(true,  1); }
        if (c == "detail2")  { rc = case_token(true,  2); }
        if (c == "cold")     { rc = case_token(false, 2); }
        if (c == "state")    { rc = case_state(); }
        if (c == "plan")     { rc = case_plan(); }
        if (c == "buffer")   { rc = case_buffer(); }
        if (c == "lanes")    { rc = case_lanes(); }
        return rc;
    }

    printf("test-trace: tmp dir %s\n", tmp_dir().c_str());
    struct { const char * name; std::vector<std::string> env; } cases[] = {
        { "prefetch", { "GGML_CPU_MOE_PREFETCH=1" } },
        { "detail2",  { "GGML_CPU_MOE_PREFETCH=1" } },
        { "cold",     { } },
        { "state",    { } },
        { "plan",     { } },
        { "buffer",   { } },
        { "lanes",    { } },
    };
    bool ok = true, skipped = false;
    for (const auto & c : cases) {
        const int rc = run_child(argv[0], c.name, c.env);
        printf("  %-10s %s\n", c.name, rc == 0 ? "OK" : rc == 77 ? "SKIPPED (residency not checkable)" : "FAILED");
        ok      = ok && (rc == 0 || rc == 77);
        skipped = skipped || rc == 77;
    }
    if (ok && skipped) {
        printf("SKIPPED\n");
        return 77;
    }
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
