// Perfetto C++ glue for C-callable trace shims.

#include <atomic>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if !defined(_WIN32)
#include <dlfcn.h>
#include <time.h>
#endif
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "perfetto.h"
#include "llama_perfetto.h"

// TrackEvent setup: define categories first, then static storage.
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("ML").SetDescription("High-level ML ops (CPU)"),
    perfetto::Category("GPU").SetDescription("Vulkan compute dispatches")
);
PERFETTO_TRACK_EVENT_STATIC_STORAGE();

static void llama_perfetto_init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        perfetto::TracingInitArgs args;
        // Ensure an in-process backend is available for local tracing.
        args.backends = perfetto::BackendType::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();
    });
}

extern "C" void llama_perfetto_trace_begin(const char * name) {
    llama_perfetto_init_once();
    if (name == nullptr) name = "op";
    TRACE_EVENT_BEGIN("ML", perfetto::DynamicString(name));
}

extern "C" void llama_perfetto_trace_begin_with_text(const char * name, const char * text) {
    llama_perfetto_init_once();
    if (name == nullptr) name = "op";
    const char * arg = text ? text : "";
    // Attach the token string as an argument named "text".
    TRACE_EVENT_BEGIN("ML", perfetto::DynamicString(name),
                      "text", perfetto::DynamicString(arg));
}

extern "C" void llama_perfetto_trace_end(void) {
    llama_perfetto_init_once();
    TRACE_EVENT_END("ML");
}

extern "C" void llama_perfetto_gpu_begin(const char * name) {
    llama_perfetto_init_once();
    if (name == nullptr) name = "vk_dispatch";
    TRACE_EVENT_BEGIN("GPU", perfetto::DynamicString(name));
}

extern "C" void llama_perfetto_gpu_end(void) {
    llama_perfetto_init_once();
    TRACE_EVENT_END("GPU");
}

// In-process trace session management
static std::unique_ptr<perfetto::TracingSession> g_session;
static std::atomic<bool> g_flush_stop{true};
static std::thread g_flush_thread;
static int g_trace_fd = -1;
static std::string g_trace_path;

// Optional Vulkan stats hooks resolved dynamically when ggml-vulkan is loaded.
using fn_vk_dump_stats_t = bool (*)(int, const char *);
using fn_vk_dump_timeline_t = bool (*)(int, const char *);
using fn_vk_dump_timeline_abs_t = bool (*)(int, const char *);
using fn_vk_get_anchor_mono_ns_t = uint64_t (*)(int);
using fn_vk_get_desc_t   = void (*)(int, char *, size_t);
using fn_vk_get_mem_t    = void (*)(int, size_t *, size_t *);

static std::atomic<bool> g_vk_syms_resolved{false};
static fn_vk_dump_stats_t g_vk_dump_stats = nullptr;
static fn_vk_get_desc_t   g_vk_get_desc   = nullptr;
static fn_vk_get_mem_t    g_vk_get_mem    = nullptr;
static fn_vk_dump_timeline_t g_vk_dump_timeline = nullptr;
static fn_vk_dump_timeline_abs_t g_vk_dump_timeline_abs = nullptr;
static fn_vk_get_anchor_mono_ns_t g_vk_get_anchor_mono_ns = nullptr;

static void llama_perfetto_resolve_vk_syms_once() {
    bool expected = false;
    if (!g_vk_syms_resolved.compare_exchange_strong(expected, true)) return;
#if defined(_WIN32)
    // Best-effort: resolve from current process address space.
    HMODULE self = GetModuleHandleA(nullptr);
    if (self) {
        g_vk_dump_stats = reinterpret_cast<fn_vk_dump_stats_t>(GetProcAddress(self, "ggml_backend_vk_dump_pipeline_stats"));
        g_vk_get_desc   = reinterpret_cast<fn_vk_get_desc_t>(GetProcAddress(self, "ggml_backend_vk_get_device_description"));
        g_vk_get_mem    = reinterpret_cast<fn_vk_get_mem_t>(GetProcAddress(self, "ggml_backend_vk_get_device_memory"));
        g_vk_dump_timeline = reinterpret_cast<fn_vk_dump_timeline_t>(GetProcAddress(self, "ggml_backend_vk_dump_timeline"));
        g_vk_dump_timeline_abs = reinterpret_cast<fn_vk_dump_timeline_abs_t>(GetProcAddress(self, "ggml_backend_vk_dump_timeline_abs"));
        g_vk_get_anchor_mono_ns = reinterpret_cast<fn_vk_get_anchor_mono_ns_t>(GetProcAddress(self, "ggml_backend_vk_get_timeline_anchor_mono_ns"));
    }
#else
    g_vk_dump_stats = reinterpret_cast<fn_vk_dump_stats_t>(dlsym(RTLD_DEFAULT, "ggml_backend_vk_dump_pipeline_stats"));
    g_vk_get_desc   = reinterpret_cast<fn_vk_get_desc_t>(dlsym(RTLD_DEFAULT, "ggml_backend_vk_get_device_description"));
    g_vk_get_mem    = reinterpret_cast<fn_vk_get_mem_t>(dlsym(RTLD_DEFAULT, "ggml_backend_vk_get_device_memory"));
    g_vk_dump_timeline = reinterpret_cast<fn_vk_dump_timeline_t>(dlsym(RTLD_DEFAULT, "ggml_backend_vk_dump_timeline"));
    g_vk_dump_timeline_abs = reinterpret_cast<fn_vk_dump_timeline_abs_t>(dlsym(RTLD_DEFAULT, "ggml_backend_vk_dump_timeline_abs"));
    g_vk_get_anchor_mono_ns = reinterpret_cast<fn_vk_get_anchor_mono_ns_t>(dlsym(RTLD_DEFAULT, "ggml_backend_vk_get_timeline_anchor_mono_ns"));
#endif
}

extern "C" void llama_perfetto_start_trace(const char * path) {
    if (g_session) {
        // Already started; ignore duplicate start.
        return;
    }
    if (!path || !*path) return;
    llama_perfetto_init_once();

    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(1024 * 64);
    auto * ds = cfg.add_data_sources();
    ds->mutable_config()->set_name("track_event");
    // Enable all TrackEvent categories ("*") so our ML/GPU spans are recorded
    perfetto::protos::gen::TrackEventConfig te;
    te.add_enabled_categories("*");
    ds->mutable_config()->set_track_event_config_raw(te.SerializeAsString());

    g_session = perfetto::Tracing::NewTrace();
    // open file and pass fd to Setup so Perfetto writes directly into it
    int fd = -1;
#if defined(_WIN32)
    // Omit Windows fd handling in this patch
#else
    fd = ::open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
#endif
    g_session->Setup(cfg, fd);
    g_session->StartBlocking();

    g_trace_fd = fd;
    g_trace_path = path;
    g_flush_stop = false;
    // Background flusher to minimize data loss on abrupt termination (e.g., SIGKILL).
    g_flush_thread = std::thread([]{
        using namespace std::chrono_literals;
        while (!g_flush_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(200ms);
            if (g_session) {
                g_session->FlushBlocking(0);
            }
            if (g_trace_fd != -1) {
                ::fsync(g_trace_fd);
            }
        }
    });
}

extern "C" void llama_perfetto_stop_flush(void) {
    g_flush_stop = true;
    if (g_flush_thread.joinable()) g_flush_thread.join();
    if (g_session) {
        g_session->FlushBlocking(0);
        g_session->StopBlocking();
        g_session.reset();
    }
    if (g_trace_fd != -1) {
        ::fsync(g_trace_fd);
        ::close(g_trace_fd);
        g_trace_fd = -1;
    }

    // If Vulkan backend is present, dump basic GPU pipeline stats alongside the Perfetto file.
    llama_perfetto_resolve_vk_syms_once();
    if (!g_trace_path.empty() && g_vk_dump_stats) {
        std::string stats_path = g_trace_path + ".vkstats";
        // Use logical device index 0 by default; adjust if multi-GPU selection is exposed.
        (void)g_vk_dump_stats(0, stats_path.c_str());
    }
}

extern "C" void llama_perfetto_flush_dump_stats(void) {
    // Best-effort flush of trace buffers without stopping the session.
    if (g_session) {
        g_session->FlushBlocking(0);
    }
    if (g_trace_fd != -1) {
        ::fsync(g_trace_fd);
    }
    // Emit Vulkan pipeline stats next to the trace file if available.
    llama_perfetto_resolve_vk_syms_once();
    if (!g_trace_path.empty() && g_vk_dump_stats) {
        std::string stats_path = g_trace_path + ".vkstats";
        (void)g_vk_dump_stats(0, stats_path.c_str());
    }
}

static std::string llama_perfetto_tmp_stats_path() {
    // Prefer trace-adjacent path if tracing is active, else temp dir.
    if (!g_trace_path.empty()) {
        return g_trace_path + ".vkstats";
    }
    try {
        auto dir = std::filesystem::temp_directory_path();
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count();
        // Avoid Windows headers; pid is best-effort for uniqueness
#if defined(_WIN32)
        int pid = 0;
#else
        int pid = (int) getpid();
#endif
        return (dir / ("llama_vkstats_" + std::to_string(pid) + "_" + std::to_string(ts) + ".txt")).string();
    } catch (...) {
        return "llama_vkstats.txt"; // fallback to cwd
    }
}

extern "C" void llama_perfetto_print_gpu_stats(void) {
    static std::atomic<bool> warned{false};
    llama_perfetto_resolve_vk_syms_once();
    if (!g_vk_dump_stats) {
        return;
    }
    std::string path = llama_perfetto_tmp_stats_path();
    if (!g_vk_dump_stats(0, path.c_str())) {
        // Feature not supported or failure; print a one-time notice with minimal device info.
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true)) {
            char desc[256] = {0};
            if (g_vk_get_desc) {
                g_vk_get_desc(0, desc, sizeof(desc));
            }
            size_t free_b = 0, total_b = 0;
            if (g_vk_get_mem) {
                g_vk_get_mem(0, &free_b, &total_b);
            }
            fprintf(stdout,
                    "[GPU] Vulkan pipeline statistics not supported on this device%s%s.\n",
                    desc[0] ? ": " : "",
                    desc);
            if (total_b) {
                fprintf(stdout, "[GPU] Reported device-local memory: %.2f GiB.\n", (double) total_b / (1024.0*1024.0*1024.0));
            }
        }
        return;
    }
    std::ifstream ifs(path);
    if (!ifs.good()) {
        return;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        // Print raw line; main uses stdout redirection already
        fprintf(stdout, "[GPU] %s\n", line.c_str());
    }
    ifs.close();
    // Best effort cleanup for temp file when not using the trace-adjacent name
    if (g_trace_path.empty()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
}

extern "C" void llama_perfetto_counter_tokens_per_s(double tokens_per_s) {
    llama_perfetto_init_once();
    // If tracing is not active, this is still a no-op cost-wise
    TRACE_COUNTER("ML", "tokens_per_s", tokens_per_s);
}

extern "C" void llama_perfetto_counter_gpu_busy(double percent) {
    llama_perfetto_init_once();
    TRACE_COUNTER("GPU", "gpu_busy_percent", percent);
}

extern "C" void llama_perfetto_emit_gpu_timeline(void) {
    // Only emit if tracing is active and we can get a timeline snapshot.
    if (!g_session) return;
    llama_perfetto_resolve_vk_syms_once();
    if (!g_vk_dump_timeline) return;

    // Prefer absolute CPU-monotonic-aligned timeline if available
    bool emitted = false;
    if (g_vk_dump_timeline_abs) {
        std::string path_abs = llama_perfetto_tmp_stats_path();
        if (path_abs.size() >= 8) path_abs.replace(path_abs.size() - 7, 7, "vktimeline.abs");
        if (g_vk_dump_timeline_abs(0, path_abs.c_str())) {
            std::ifstream ifsa(path_abs);
            if (ifsa.good()) {
                struct AbsEntry { std::string name; uint64_t s_abs; uint64_t e_abs; };
                std::vector<AbsEntry> aentries; aentries.reserve(256);
                std::string line;
                while (std::getline(ifsa, line)) {
                    size_t c1 = line.find(',');
                    size_t c2 = line.find(',', c1 == std::string::npos ? 0 : c1 + 1);
                    if (c1 == std::string::npos || c2 == std::string::npos) continue;
                    uint64_t s = strtoull(line.substr(0, c1).c_str(), nullptr, 10);
                    uint64_t e = strtoull(line.substr(c1 + 1, c2 - c1 - 1).c_str(), nullptr, 10);
                    std::string name = line.substr(c2 + 1);
                    if (e > s && s != 0ULL) aentries.push_back({std::move(name), s, e});
                }
                ifsa.close();
                { std::error_code ec; std::filesystem::remove(path_abs, ec); }
                if (!aentries.empty()) {
                    // Map CPU monotonic ns -> Perfetto trace ns by sampling both now and using constant offset
                    uint64_t trace_now = perfetto::TrackEvent::GetTraceTimeNs();
#if !defined(_WIN32)
                    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                    uint64_t mono_now = uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
                    int64_t offset = (int64_t)trace_now - (int64_t)mono_now;
#else
                    // Windows: fallback to relative anchoring below
                    int64_t offset = 0;
#endif
                    auto gpu_track = perfetto::Track(0x47505551304ULL);
                    static std::atomic<bool> desc{false};
                    bool expected = false;
                    if (desc.compare_exchange_strong(expected, true)) {
                        perfetto::protos::gen::TrackDescriptor td; td.set_name("GPU Queue 0");
                        perfetto::TrackEvent::SetTrackDescriptor(gpu_track, td);
                    }
                    struct Ev { uint64_t ts; int delta; };
                    std::vector<Ev> evs; evs.reserve(aentries.size()*2);
                    for (const auto & e : aentries) {
                        uint64_t start_ns = (uint64_t)((int64_t)e.s_abs + offset);
                        uint64_t end_ns   = (uint64_t)((int64_t)e.e_abs + offset);
                        TRACE_EVENT_BEGIN("GPU", perfetto::DynamicString(e.name.c_str()), gpu_track, start_ns);
                        TRACE_EVENT_END("GPU", gpu_track, end_ns);
                        evs.push_back({start_ns, +1});
                        evs.push_back({end_ns,   -1});
                    }
                    if (!evs.empty()) {
                        std::sort(evs.begin(), evs.end(), [](const Ev &a, const Ev &b){
                            if (a.ts != b.ts) return a.ts < b.ts;
                            return a.delta < b.delta;
                        });
                        int active = 0;
                        for (const auto & ev : evs) {
                            if (ev.delta < 0) { active += ev.delta; if (active == 0) TRACE_COUNTER("GPU", "gpu_busy_percent", ev.ts, 0.0); }
                            else { if (active == 0) TRACE_COUNTER("GPU", "gpu_busy_percent", ev.ts, 100.0); active += ev.delta; }
                        }
                    }
                    emitted = true;
                }
            }
        }
    }

    if (emitted) return;

    // Fallback: use relative timeline and anchor last-slice end to the fence-return time if available
    std::string path = llama_perfetto_tmp_stats_path();
    if (path.size() >= 8) path.replace(path.size() - 7, 7, "vktimeline");
    if (!g_vk_dump_timeline(0, path.c_str())) return;

    struct Entry { std::string name; uint64_t s_rel; uint64_t e_rel; };
    std::vector<Entry> entries; entries.reserve(256);
    std::ifstream ifs(path); if (!ifs.good()) return;
    std::string line; uint64_t total_span = 0;
    while (std::getline(ifs, line)) {
        size_t c1 = line.find(','); size_t c2 = line.find(',', c1 == std::string::npos ? 0 : c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos) continue;
        uint64_t s = strtoull(line.substr(0, c1).c_str(), nullptr, 10);
        uint64_t e = strtoull(line.substr(c1 + 1, c2 - c1 - 1).c_str(), nullptr, 10);
        std::string name = line.substr(c2 + 1);
        entries.push_back({std::move(name), s, e});
        if (e > total_span) total_span = e;
    }
    ifs.close(); { std::error_code ec; std::filesystem::remove(path, ec); }
    if (entries.empty() || total_span == 0) return;

    // Try to anchor spans to the CPU time when the fence returned to avoid drawing after the wait
    uint64_t anchor_trace = 0;
#if !defined(_WIN32)
    if (g_vk_get_anchor_mono_ns) {
        uint64_t anchor_mono = g_vk_get_anchor_mono_ns(0);
        if (anchor_mono) {
            uint64_t trace_now = perfetto::TrackEvent::GetTraceTimeNs();
            timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
            uint64_t mono_now = uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
            int64_t offset = (int64_t)trace_now - (int64_t)mono_now;
            anchor_trace = (uint64_t)((int64_t)anchor_mono + offset);
        }
    }
#endif
    uint64_t now = anchor_trace ? anchor_trace : perfetto::TrackEvent::GetTraceTimeNs();
    auto gpu_track = perfetto::Track(0x47505551304ULL);
    {
        static std::atomic<bool> desc{false}; bool expected = false;
        if (desc.compare_exchange_strong(expected, true)) {
            perfetto::protos::gen::TrackDescriptor td; td.set_name("GPU Queue 0");
            perfetto::TrackEvent::SetTrackDescriptor(gpu_track, td);
        }
    }
    struct Ev { uint64_t ts; int delta; };
    std::vector<Ev> evs; evs.reserve(entries.size()*2);
    for (const auto & e : entries) {
        uint64_t start_ns = now - (total_span - e.s_rel);
        uint64_t end_ns   = now - (total_span - e.e_rel);
        TRACE_EVENT_BEGIN("GPU", perfetto::DynamicString(e.name.c_str()), gpu_track, start_ns);
        TRACE_EVENT_END("GPU", gpu_track, end_ns);
        if (end_ns > start_ns) { evs.push_back({start_ns, +1}); evs.push_back({end_ns, -1}); }
    }
    if (!evs.empty()) {
        std::sort(evs.begin(), evs.end(), [](const Ev &a, const Ev &b){ if (a.ts != b.ts) return a.ts < b.ts; return a.delta < b.delta; });
        int active = 0; for (const auto & ev : evs) { if (ev.delta < 0) { active += ev.delta; if (active == 0) TRACE_COUNTER("GPU", "gpu_busy_percent", ev.ts, 0.0); } else { if (active == 0) TRACE_COUNTER("GPU", "gpu_busy_percent", ev.ts, 100.0); active += ev.delta; } }
    }
}

extern "C" void llama_perfetto_try_start_from_env(void) {
    const char * path = getenv("LLAMA_PERFETTO_TRACE");
    if (path && *path) {
        llama_perfetto_start_trace(path);
        return;
    }
    // Fallback: if LLAMA_PERFETTO is set (any value), write to default file
    const char * on = getenv("LLAMA_PERFETTO");
    if (on && *on) {
        llama_perfetto_start_trace("llama.perfetto-trace");
    }
}
