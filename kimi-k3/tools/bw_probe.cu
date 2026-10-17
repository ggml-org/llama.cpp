// bw_probe: raw read ceilings for PLAN-cpu-gpu-matmul.md (B2)
//   CPU streaming read (7 / 14 / 28 threads), pinned -> VRAM copy, GPU kernel reading pinned host memory (zero-copy;
//   cudaHostAlloc and posix_memalign + cudaHostRegister), GPU kernel reading VRAM. GPU times with cudaEvents.
//   A background thread samples nvidia-smi pcie.link.gen / width per phase (idle shows gen 1).
//
//   bw_probe [--cpu-gb 4] [--gpu-gb 1] [--threads 7,14,28] [--reps 5] [--hold-ms 1500] [--json out.json] [--load-wait-s 300] [--force]
//   refuses while a llama-server runs; waits for load avg < 2 (at most --load-wait-s, then a warning in the JSON)

#include <cuda_runtime.h>

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static const size_t GIB           = 1ull << 30;
static const size_t VRAM_BUDGET   = GIB;
static const size_t PINNED_BUDGET = 3 * GIB / 2;
static const size_t MIN_FREE_VRAM = 3 * GIB;

static std::vector<void *> g_dev, g_hostalloc, g_registered, g_malloc;
static std::atomic<bool>   g_stop{false};

static void cleanup() {
    for (void * p : g_registered) { cudaHostUnregister(p); }
    for (void * p : g_dev)        { cudaFree(p); }
    for (void * p : g_hostalloc)  { cudaFreeHost(p); }
    for (void * p : g_malloc)     { free(p); }
    g_registered.clear(); g_dev.clear(); g_hostalloc.clear(); g_malloc.clear();
}

[[noreturn]] static void die(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "bw_probe: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    g_stop = true;
    cleanup();
    exit(1);
}

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { die("%s: %s", #x, cudaGetErrorString(err_)); } } while (0)

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- PCIe link sampler

struct pcie_stat { int gen_min = 99, gen_max = 0, w_min = 99, w_max = 0, n = 0; };
static std::mutex                       g_mu;
static std::string                      g_phase = "idle";
static std::map<std::string, pcie_stat> g_pcie;

static void sampler() {
    while (!g_stop) {
        FILE * f = popen("nvidia-smi -i 0 --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null", "r");
        int gen = 0, w = 0;
        if (f) {
            if (fscanf(f, "%d , %d", &gen, &w) != 2) {
                gen = w = 0;
            }
            pclose(f);
        }
        if (gen > 0) {
            std::lock_guard<std::mutex> lk(g_mu);
            pcie_stat & s = g_pcie[g_phase];
            s.gen_min = std::min(s.gen_min, gen); s.gen_max = std::max(s.gen_max, gen);
            s.w_min   = std::min(s.w_min, w);     s.w_max   = std::max(s.w_max, w);
            s.n++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

static void set_phase(const char * p) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_phase = p;
}

// ---------------------------------------------------------------- CPU read

static double cpu_read_gbs(const uint64_t * p, size_t n, int nt) {
    std::atomic<int> ready{0};
    std::vector<uint64_t> sink(nt * 8);
    std::vector<std::thread> th;
    double t0 = 0.0;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t]() {
            const size_t a = n * t / nt, b = n * (t + 1) / nt;
            if (++ready == nt) {
                t0 = now_ms();
            }
            while (ready.load() < nt) {
            }
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            size_t i = a;
            for (; i + 4 <= b; i += 4) {
                s0 += p[i]; s1 += p[i + 1]; s2 += p[i + 2]; s3 += p[i + 3];
            }
            for (; i < b; ++i) {
                s0 += p[i];
            }
            sink[t * 8] = s0 + s1 + s2 + s3;
        });
    }
    for (auto & x : th) {
        x.join();
    }
    const double ms = now_ms() - t0;
    uint64_t s = 0;
    for (auto v : sink) {
        s += v;
    }
    if (s == 42) {
        printf("(sink)\n");
    }
    return n * 8.0 / (ms * 1e6);
}

// ---------------------------------------------------------------- GPU read kernel

__global__ void read_kernel(const uint4 * __restrict__ p, size_t n, unsigned int * out) {
    unsigned int acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w + c.x + c.y + c.z + c.w + d.x + d.y + d.z + d.w;
    }
    for (; i < n; i += stride) {
        const uint4 a = p[i];
        acc ^= a.x + a.y + a.z + a.w;
    }
    if (acc == 0x9e3779b9u) {
        out[0] = acc;
    }
}

struct gpu_res { double gbs = 0.0; int mult = 0, streams = 0, block = 0; };

// best GB/s over the grid sweep; time from a start event on stream 0 to the end of all streams
static gpu_res gpu_read_sweep(const void * dptr, size_t bytes, int n_sm, int reps, unsigned int * dout, std::vector<cudaStream_t> & st) {
    gpu_res best;
    cudaEvent_t e0, e1, ee[4];
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (auto & e : ee) {
        CK(cudaEventCreate(&e));
    }
    const size_t n = bytes / sizeof(uint4);
    for (int ns : {1, 4}) {
        for (int mult : {1, 4, 16}) {
            const int block = 256;
            for (int r = 0; r < reps + 1; ++r) {
                CK(cudaEventRecord(e0, st[0]));
                for (int s = 0; s < ns; ++s) {
                    if (s > 0) {
                        CK(cudaStreamWaitEvent(st[s], e0, 0));
                    }
                    const size_t a = n * s / ns, b = n * (s + 1) / ns;
                    read_kernel<<<n_sm * mult, block, 0, st[s]>>>((const uint4 *) dptr + a, b - a, dout);
                    CK(cudaGetLastError());
                    CK(cudaEventRecord(ee[s], st[s]));
                }
                for (int s = 1; s < ns; ++s) {
                    CK(cudaStreamWaitEvent(st[0], ee[s], 0));
                }
                CK(cudaEventRecord(e1, st[0]));
                CK(cudaEventSynchronize(e1));
                float ms = 0.0f;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                const double gbs = bytes / (ms * 1e6);
                if (r > 0 && gbs > best.gbs) {
                    best = { gbs, mult, ns, block };
                }
            }
        }
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    for (auto & e : ee) {
        CK(cudaEventDestroy(e));
    }
    return best;
}

// keep the link busy with the best config for hold_ms so that the sampler sees it under load
static void gpu_hold(const void * dptr, size_t bytes, int n_sm, const gpu_res & cfg, unsigned int * dout, std::vector<cudaStream_t> & st, double hold_ms) {
    const size_t n = bytes / sizeof(uint4);
    const double t0 = now_ms();
    while (now_ms() - t0 < hold_ms) {
        for (int s = 0; s < cfg.streams; ++s) {
            const size_t a = n * s / cfg.streams, b = n * (s + 1) / cfg.streams;
            read_kernel<<<n_sm * cfg.mult, cfg.block, 0, st[s]>>>((const uint4 *) dptr + a, b - a, dout);
        }
        CK(cudaDeviceSynchronize());
    }
}

static std::vector<int> split_ints(const char * s) {
    std::vector<int> v;
    const char * p = s;
    while (*p) {
        v.push_back(atoi(p));
        const char * c = strchr(p, ',');
        if (!c) {
            break;
        }
        p = c + 1;
    }
    return v;
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
    double cpu_gb = 4.0, gpu_gb = 1.0, hold_ms = 1500.0, load_wait_s = 300.0;
    std::vector<int> threads = {7, 14, 28};
    int reps = 5;
    bool force = false;
    const char * json = nullptr;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { die("missing value for %s", a.c_str()); } return argv[++i]; };
        if      (a == "--cpu-gb")  { cpu_gb = atof(next()); }
        else if (a == "--gpu-gb")  { gpu_gb = atof(next()); }
        else if (a == "--threads") { threads = split_ints(next()); }
        else if (a == "--reps")    { reps = atoi(next()); }
        else if (a == "--hold-ms") { hold_ms = atof(next()); }
        else if (a == "--json")    { json = next(); }
        else if (a == "--force")   { force = true; }
        else if (a == "--load-wait-s") { load_wait_s = atof(next()); }
        else { die("unknown argument %s", a.c_str()); }
    }
    const size_t gpu_bytes = ((size_t) (gpu_gb * 1e9)) & ~(size_t) 4095;
    const size_t cpu_bytes = ((size_t) (cpu_gb * 1e9)) & ~(size_t) 4095;
    if (gpu_bytes + 4096 > VRAM_BUDGET || gpu_bytes > PINNED_BUDGET) {
        die("--gpu-gb %.2f exceeds the budget (VRAM 1 GiB, pinned 1.5 GiB)", gpu_gb);
    }

    const std::string load_warning = run_gate(force, load_wait_s);

    std::string js = "{\n \"load_avg\": " + std::to_string(load_avg()) + ",\n \"load_warning\": " + jesc(load_warning) + ",";
    char tmp[512];

    // CPU
    printf("CPU streaming read of %.2f GB\n", cpu_bytes / 1e9);
    uint64_t * hp = nullptr;
    if (posix_memalign((void **) &hp, 4096, cpu_bytes) != 0) {
        die("posix_memalign");
    }
    g_malloc.push_back(hp);
    memset(hp, 1, cpu_bytes);
    js += "\n \"cpu_read_gbs\": {";
    for (size_t ti = 0; ti < threads.size(); ++ti) {
        double best = 0.0;
        for (int r = 0; r < reps; ++r) {
            best = std::max(best, cpu_read_gbs(hp, cpu_bytes / 8, threads[ti]));
        }
        printf("  %2d threads: %6.2f GB/s (best of %d)\n", threads[ti], best, reps);
        snprintf(tmp, sizeof(tmp), "%s\"%d\": %.3f", ti ? ", " : "", threads[ti], best);
        js += tmp;
    }
    js += "}";
    g_malloc.clear();
    free(hp);

    // GPU
    CK(cudaSetDevice(0));
    size_t vfree = 0, vtotal = 0;
    CK(cudaMemGetInfo(&vfree, &vtotal));
    if (vfree < MIN_FREE_VRAM) {
        die("only %.2f GiB VRAM free (< 3 GiB)", vfree / (double) GIB);
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int can_host_ptr = 0, ro = 0;
    CK(cudaDeviceGetAttribute(&can_host_ptr, cudaDevAttrCanUseHostPointerForRegisteredMem, 0));
    CK(cudaDeviceGetAttribute(&ro, cudaDevAttrHostRegisterReadOnlySupported, 0));
    printf("GPU %s, %d SMs, VRAM free %.2f GiB, test size %.2f GB\n", prop.name, prop.multiProcessorCount, vfree / (double) GIB, gpu_bytes / 1e9);

    std::thread smp(sampler);
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    std::vector<cudaStream_t> st(4);
    for (auto & s : st) {
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }
    unsigned int * dout = nullptr;
    CK(cudaMalloc(&dout, 64));
    g_dev.push_back(dout);
    void * dbuf = nullptr;
    CK(cudaMalloc(&dbuf, gpu_bytes));
    g_dev.push_back(dbuf);
    CK(cudaMemset(dbuf, 1, gpu_bytes));

    // pinned (cudaHostAlloc) -> VRAM copy, then zero-copy reads of the same buffer
    void * pin = nullptr;
    CK(cudaHostAlloc(&pin, gpu_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    g_hostalloc.push_back(pin);
    memset(pin, 1, gpu_bytes);

    set_phase("h2d");
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    double h2d = 0.0;
    const double th0 = now_ms();
    for (int r = 0; r < reps + 1 || now_ms() - th0 < hold_ms; ++r) {
        CK(cudaEventRecord(e0, st[0]));
        CK(cudaMemcpyAsync(dbuf, pin, gpu_bytes, cudaMemcpyHostToDevice, st[0]));
        CK(cudaEventRecord(e1, st[0]));
        CK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r > 0) {
            h2d = std::max(h2d, gpu_bytes / (ms * 1e6));
        }
    }
    printf("pinned -> VRAM copy: %.2f GB/s\n", h2d);

    set_phase("zero_copy_hostalloc");
    void * dpin = pin;
    if (!can_host_ptr) {
        CK(cudaHostGetDevicePointer(&dpin, pin, 0));
    }
    gpu_res zc_a = gpu_read_sweep(dpin, gpu_bytes, prop.multiProcessorCount, reps, dout, st);
    gpu_hold(dpin, gpu_bytes, prop.multiProcessorCount, zc_a, dout, st, hold_ms);
    printf("zero-copy read (cudaHostAlloc):   %.2f GB/s (grid %dx SMs, %d stream(s))\n", zc_a.gbs, zc_a.mult, zc_a.streams);
    g_hostalloc.clear();
    CK(cudaFreeHost(pin));

    // posix_memalign + cudaHostRegister, as matmul_bench path (c)
    set_phase("zero_copy_registered");
    void * reg = nullptr;
    if (posix_memalign(&reg, 4096, gpu_bytes) != 0) {
        die("posix_memalign");
    }
    g_malloc.push_back(reg);
    memset(reg, 1, gpu_bytes);
    int ro_used = 1;
    if (cudaHostRegister(reg, gpu_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly) != cudaSuccess) {
        cudaGetLastError();
        ro_used = 0;
        CK(cudaHostRegister(reg, gpu_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
    }
    g_registered.push_back(reg);
    void * dreg = reg;
    if (!can_host_ptr) {
        CK(cudaHostGetDevicePointer(&dreg, reg, 0));
    }
    gpu_res zc_r = gpu_read_sweep(dreg, gpu_bytes, prop.multiProcessorCount, reps, dout, st);
    gpu_hold(dreg, gpu_bytes, prop.multiProcessorCount, zc_r, dout, st, hold_ms);
    printf("zero-copy read (registered%s): %.2f GB/s (grid %dx SMs, %d stream(s))\n", ro_used ? ", read-only" : "", zc_r.gbs, zc_r.mult, zc_r.streams);
    g_registered.clear();
    CK(cudaHostUnregister(reg));
    g_malloc.clear();
    free(reg);

    set_phase("vram");
    gpu_res vr = gpu_read_sweep(dbuf, gpu_bytes, prop.multiProcessorCount, reps, dout, st);
    gpu_hold(dbuf, gpu_bytes, prop.multiProcessorCount, vr, dout, st, hold_ms);
    printf("VRAM read: %.2f GB/s (grid %dx SMs, %d stream(s))\n", vr.gbs, vr.mult, vr.streams);
    set_phase("done");

    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    for (auto & s : st) {
        CK(cudaStreamDestroy(s));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    g_stop = true;
    smp.join();

    printf("PCIe link (nvidia-smi samples per phase):\n");
    js += ",\n \"pcie\": {";
    bool first = true;
    for (const char * ph : {"idle", "h2d", "zero_copy_hostalloc", "zero_copy_registered", "vram"}) {
        auto it = g_pcie.find(ph);
        if (it == g_pcie.end()) {
            printf("  %-22s no sample\n", ph);
            continue;
        }
        const pcie_stat & s = it->second;
        printf("  %-22s gen %d..%d, width x%d..x%d (%d samples)\n", ph, s.gen_min, s.gen_max, s.w_min, s.w_max, s.n);
        snprintf(tmp, sizeof(tmp), "%s\"%s\": {\"gen_min\": %d, \"gen_max\": %d, \"width_min\": %d, \"width_max\": %d, \"n\": %d}",
                 first ? "" : ", ", ph, s.gen_min, s.gen_max, s.w_min, s.w_max, s.n);
        js += tmp;
        first = false;
    }
    js += "}";
    printf("not measured: NVMe DMA and GPU zero-copy share the memory controller\n");

    snprintf(tmp, sizeof(tmp),
             ",\n \"h2d_gbs\": %.3f,\n \"zero_copy_hostalloc\": {\"gbs\": %.3f, \"grid_mult\": %d, \"streams\": %d},"
             "\n \"zero_copy_registered\": {\"gbs\": %.3f, \"grid_mult\": %d, \"streams\": %d, \"read_only\": %d},"
             "\n \"vram_read\": {\"gbs\": %.3f, \"grid_mult\": %d, \"streams\": %d},\n \"gpu_bytes\": %zu, \"cpu_bytes\": %zu, \"sms\": %d\n}\n",
             h2d, zc_a.gbs, zc_a.mult, zc_a.streams, zc_r.gbs, zc_r.mult, zc_r.streams, ro_used, vr.gbs, vr.mult, vr.streams,
             gpu_bytes, cpu_bytes, prop.multiProcessorCount);
    js += tmp;
    if (json) {
        FILE * f = fopen(json, "w");
        if (!f) {
            die("cannot write %s", json);
        }
        fputs(js.c_str(), f);
        fclose(f);
        printf("wrote %s\n", json);
    }
    cleanup();
    return 0;
}
