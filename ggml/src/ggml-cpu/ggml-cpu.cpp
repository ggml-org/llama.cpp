#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <atomic>
#include <cctype>
#include <string>
#include <vector>

#if defined(__gnu_linux__)
#    include <cerrno>
#    include <cstdio>
#    include <cstring>
#    include <linux/mempolicy.h>
#    include <sched.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation

    int                 numa_node;       // -1 = threads not pinned
    ggml_threadpool_t   numa_threadpool; // owned, threads pinned to numa node
    int                 numa_n_threads;  // thread count of numa_threadpool
    int                 numa_max_threads; // 0 = no limit, else CPUs of the smallest node in TP (comm_init)
    int                 numa_n_nodes;     // NUMA nodes in TP
    int                 numa_cap_node;    // smallest node in TP, it logs the error
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    ggml_threadpool_free(cpu_ctx->numa_threadpool);
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

// threadpool pinned to the node CPUs, rebuilt when n_threads changes. With more threads than node CPUs, all CPUs are allowed
static ggml_threadpool_t ggml_backend_cpu_numa_threadpool(ggml_backend_cpu_context * ctx) {
#if defined(__gnu_linux__)
    if (ctx->numa_threadpool && ctx->numa_n_threads == ctx->n_threads) {
        return ctx->numa_threadpool;
    }
    ggml_threadpool_free(ctx->numa_threadpool);
    ctx->numa_threadpool = NULL;

    ggml_threadpool_params tpp = ggml_threadpool_params_default(ctx->n_threads);
    int n_node_cpus = 0;
    for (int c = 0; c < GGML_MAX_N_THREADS; c++) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpu%d", ctx->numa_node, c);
        if (access(path, F_OK) == 0) {
            tpp.cpumask[c] = true;
            n_node_cpus++;
        }
    }
    if (n_node_cpus == 0) {
        ctx->numa_node = -1;
        return NULL;
    }
    if (ctx->n_threads > n_node_cpus) {
        for (int c = 0; c < GGML_MAX_N_THREADS; c++) {
            char path[256];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", c);
            tpp.cpumask[c] = access(path, F_OK) == 0;
        }
    }

    // one CPU per thread: the pool fills the node, and floating threads can stack on one CPU and stall every barrier
    tpp.strict_cpu = true;

    if (ctx->n_threads <= n_node_cpus) {
        // pin the master first, so the pool and its OpenMP team are first-touched on the node
        // through the affinity helper: a direct sched_setaffinity bypasses its mask cache, and the later pin to one CPU is skipped
        ggml_cpu_thread_set_affinity(tpp.cpumask);
    }

    ctx->numa_threadpool = ggml_threadpool_new_numa(&tpp);
    ctx->numa_n_threads  = ctx->n_threads;
    return ctx->numa_threadpool;
#else
    ctx->numa_node = -1;
    return NULL;
#endif
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    // every TP device checks the same limit, so all of them fail before any thread enters the all-reduce
    if (cpu_ctx->numa_max_threads > 0 && cpu_ctx->n_threads > cpu_ctx->numa_max_threads) {
        if (cpu_ctx->numa_node == cpu_ctx->numa_cap_node) {
            GGML_LOG_ERROR("%s: CPU NUMA TP needs %d threads per NUMA node, but NUMA node %d has only %d CPUs, "
                           "use at most -t %d (%d NUMA nodes x %d)\n", __func__, cpu_ctx->n_threads, cpu_ctx->numa_cap_node,
                           cpu_ctx->numa_max_threads, cpu_ctx->numa_n_nodes * cpu_ctx->numa_max_threads,
                           cpu_ctx->numa_n_nodes, cpu_ctx->numa_max_threads);
        }
        return GGML_STATUS_FAILED;
    }

    ggml_threadpool_t threadpool = cpu_ctx->threadpool;
    if (threadpool == NULL && cpu_ctx->numa_node >= 0) {
        threadpool = ggml_backend_cpu_numa_threadpool(cpu_ctx);
    }

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;
    ctx->numa_node           = -1;
    ctx->numa_threadpool     = NULL;
    ctx->numa_n_threads      = 0;
    ctx->numa_max_threads    = 0;
    ctx->numa_n_nodes        = 0;
    ctx->numa_cap_node       = -1;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// NUMA buffer type: one per NUMA node, memory bound to the node with mbind(), so each device's tensors stay on its node

#if defined(__gnu_linux__)

struct ggml_backend_cpu_numa_buffer_type_context {
    int         node;
    std::string name;
};

// true if node exists in /sys
static bool ggml_backend_cpu_numa_node_real(int node) {
    return node >= 0 && (uint32_t) node < ggml_numa_node_count();
}

static bool ggml_backend_cpu_numa_node_memory(int node, size_t * free, size_t * total) {
    if (!ggml_backend_cpu_numa_node_real(node)) {
        return false;
    }

    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);

    FILE * f = fopen(path, "r");
    if (!f) {
        return false;
    }

    char line[256];
    unsigned long kb_total = 0;
    unsigned long kb_free  = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long v;
        if (sscanf(line, "Node %*d MemTotal: %lu kB", &v) == 1) {
            kb_total = v;
        } else if (sscanf(line, "Node %*d MemFree: %lu kB", &v) == 1) {
            kb_free = v;
        }
    }
    fclose(f);

    if (kb_total == 0) {
        return false;
    }
    *total = (size_t) kb_total * 1024;
    *free  = (size_t) kb_free  * 1024;

    return true;
}

static const char * ggml_backend_cpu_numa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ((ggml_backend_cpu_numa_buffer_type_context *) buft->context)->name.c_str();
}

static void ggml_backend_cpu_numa_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    munmap(buffer->context, buffer->size > 0 ? buffer->size : 1);
}

static ggml_backend_buffer_t ggml_backend_cpu_numa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const int node = ((ggml_backend_cpu_numa_buffer_type_context *) buft->context)->node;

    // mmap rather than ggml_aligned_malloc: mbind needs a page aligned range it fully owns
    const size_t len = size > 0 ? size : 1;

    void * data = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        GGML_LOG_ERROR("%s: mmap of %zu bytes failed: %s\n", __func__, len, strerror(errno));
        return NULL;
    }

    if (ggml_backend_cpu_numa_node_real(node)) {

        // MPOL_BIND has no fallback node: a too large request is not refused here, it is OOM killed on first touch. Warn while we can
        size_t node_free = 0, node_total = 0;
        if (ggml_backend_cpu_numa_node_memory(node, &node_free, &node_total) && len > node_free) {
            GGML_LOG_WARN("%s: %zu MiB requested on node %d, which has %zu MiB free; "
                          "the binding allows no fallback, so this may be OOM killed when first touched\n",
                          __func__, len >> 20, node, node_free >> 20);
        }

        const unsigned long mask = 1UL << node;
        if (syscall(SYS_mbind, data, len, MPOL_BIND, &mask, 8*sizeof(mask), MPOL_MF_STRICT | MPOL_MF_MOVE) != 0) {
            GGML_LOG_ERROR("%s: mbind of %zu bytes to node %d failed: %s\n", __func__, len, node, strerror(errno));
            munmap(data, len);
            return NULL;
        }
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_numa_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cpu_numa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_numa_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_numa_buffer_type_is_numa(ggml_backend_buffer_type_t buft);

static const ggml_backend_buffer_type_i ggml_backend_cpu_numa_buffer_type_i = {
    /* .get_name         = */ ggml_backend_cpu_numa_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cpu_numa_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ ggml_backend_cpu_numa_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ NULL, // defaults to ggml_bytes,
    /* .get_alloc_size_n = */ NULL, // defaults to ggml_bytes,
    /* .is_host          = */ ggml_backend_cpu_numa_buffer_type_is_host,
};

static bool ggml_backend_cpu_numa_buffer_type_is_numa(ggml_backend_buffer_type_t buft) {
    return buft != NULL && buft->iface.get_name == ggml_backend_cpu_numa_buffer_type_get_name;
}


static int ggml_backend_cpu_numa_device_count(void);

// Callers hold raw pointers into these, build once never resize.
static ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(ggml_backend_dev_t dev, int node) {
    const int n_dev = ggml_backend_cpu_numa_device_count();
    GGML_ASSERT(node >= 0 && node < n_dev);

    struct numa_bufts {
        std::vector<ggml_backend_cpu_numa_buffer_type_context> ctxs;
        std::vector<ggml_backend_buffer_type>                  bufts;
    };

    static numa_bufts * nb = [n_dev]() {
        numa_bufts * b = new numa_bufts();
        b->ctxs.resize(n_dev);
        b->bufts.reserve(n_dev);
        for (int i = 0; i < n_dev ; i++) {
            b->ctxs[i].node = i;
            b->ctxs[i].name = "NUMA" + std::to_string(i);
            b->bufts.push_back({
                /* .iface   = */ ggml_backend_cpu_numa_buffer_type_i,
                /* .device  = */ nullptr,
                /* .context = */ &b->ctxs[i],
            });
        }
        return b;
    } ();

    nb->bufts[node].device = dev;

    return &nb->bufts[node];
}

#endif // __gnu_linux__

// device 0 is "CPU", llama adds it by itself. With GGML_CPU_NUMA_DEVICES=1 on a multi-node machine, NUMA0 ... NUMAn-1 follow it for -sm tensor
static int ggml_backend_cpu_numa_device_count(void) {
    static const int n = []() -> int {
        const char * s = getenv("GGML_CPU_NUMA_DEVICES");
        if (!s || atoi(s) == 0) {
            return 0;
        }
        // runs before --numa is parsed, so it must not call ggml_numa_init
        const int n_nodes = (int) ggml_numa_node_count();
        if (n_nodes < 2) {
            return 0;
        }
        return n_nodes > GGML_BACKEND_META_MAX_DEVICES ? GGML_BACKEND_META_MAX_DEVICES : n_nodes;
    }();

    return n;
}

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";
    std::string name = "CPU";
    int         numa_node = -1; // -1 = not node bound

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    // "CPU" for device 0, "NUMA<n>" for the NUMA devices
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#if defined(__gnu_linux__)
    // a NUMA device reports the memory of its node only
    const struct ggml_backend_cpu_device_context * numa_ctx = (const struct ggml_backend_cpu_device_context *)dev->context;
    if (ggml_backend_cpu_numa_node_memory(numa_ctx->numa_node, free, total)) {
        *free = *total;
        return;
    }
#endif
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend != NULL) {
        // ggml_backend_cpu_init always sets device 0, so set the real device and its node here
        backend->device = dev;

        const ggml_backend_cpu_device_context * dev_ctx = (const ggml_backend_cpu_device_context *) dev->context;
        ((ggml_backend_cpu_context *) backend->context)->numa_node = dev_ctx->numa_node;
    }
    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
#if defined(__gnu_linux__)
    // one buffer type per NUMA device, bound to its node: with the shared CPU buffer type the meta backend can't tell the devices apart
    const ggml_backend_cpu_device_context * ctx = (const ggml_backend_cpu_device_context *) dev->context;
    if (ctx->numa_node >= 0) {
        return ggml_backend_cpu_numa_buffer_type(dev, ctx->numa_node);
    }
#endif
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // F16 or BF16 in src1 is widened into the F32 work buffer
            return src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   ((src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16) &&
                    ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SSM_SCAN:
            return ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
#if defined(__gnu_linux__)
    // any CPU can read host memory, but a node's buffer type belongs to its own device only (like a GPU)
    if (ggml_backend_cpu_numa_buffer_type_is_numa(buft)) {
        return buft->device == dev;
    }
#endif
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return (size_t) (1 + ggml_backend_cpu_numa_device_count());

    GGML_UNUSED(reg);
}

// device 0 is "CPU", devices 1..n are NUMA0 ... NUMAn-1. Callers keep raw pointers, so build once and never resize
static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    const int n_dev = 1 + ggml_backend_cpu_numa_device_count();
    GGML_ASSERT(index < (size_t) n_dev);

    struct cpu_devices {
        std::vector<ggml_backend_cpu_device_context>    ctxs;
        std::vector<ggml_backend_device>                devs;
    };

    static cpu_devices * devices = [reg, n_dev]() {
        cpu_devices * d = new cpu_devices();
        d->ctxs.resize(n_dev);
        d->devs.reserve(n_dev);

        for (int i = 0; i < n_dev; i++) {
            if (i > 0) {
                d->ctxs[i].name = "NUMA" + std::to_string(i - 1);
                d->ctxs[i].numa_node = i - 1;
            }
            d->devs.push_back({
                /* .iface = */   ggml_backend_cpu_device_i,
                /* .reg = */     reg,
                /* .context = */ &d->ctxs[i],
            });
        }
        return d;
    }();

    return &devices->devs[index];
}


// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// all-reduce for the meta backend over NUMA devices, run as a graph node by every thread of every device
// thread i sends its slice to the other devices, waits for their thread i, then sums its slice in place
struct ggml_backend_cpu_comm_flag {
    alignas(GGML_CACHE_LINE) std::atomic<int> value[2]; // [face], set to the sense when the slice was sent
    std::atomic<int> nth;

    ggml_backend_cpu_comm_flag() {
        value[0].store(0, std::memory_order_relaxed);
        value[1].store(0, std::memory_order_relaxed);
        nth.store(0, std::memory_order_relaxed);
    }
};

struct ggml_backend_cpu_comm_state {
    alignas(GGML_CACHE_LINE) int face  = 0;
    int             sense = 1;
};

struct ggml_backend_cpu_comm_context;

struct ggml_backend_cpu_comm_op {
    ggml_backend_cpu_comm_context * comm;
    size_t                          rank;
};

struct ggml_backend_cpu_comm_context {
    size_t                                   n_backends;
    std::vector<ggml_backend_t>              backends;
    std::vector<ggml_backend_cpu_comm_op>    ops;
    std::vector<ggml_backend_cpu_comm_flag>  flags;  // [rank*GGML_MAX_N_THREADS + ith]
    std::vector<ggml_backend_cpu_comm_state> states; // [rank*GGML_MAX_N_THREADS + ith]
    std::vector<ggml_backend_buffer_t>       inbox;  // [face*n_backends + dst], n_backends - 1 slots each, on the node of dst
    size_t                                   inbox_size = 0; // bytes of one slot

    ggml_backend_cpu_comm_context(ggml_backend_t * backends_in, size_t n) :
        n_backends(n),
        backends(backends_in, backends_in + n),
        flags(n*GGML_MAX_N_THREADS),
        states(n*GGML_MAX_N_THREADS) {
        for (size_t rank = 0; rank < n; rank++) {
            ops.push_back({ this, rank });
        }
    }

    void free_inbox() {
        for (ggml_backend_buffer_t buf : inbox) {
            ggml_backend_buffer_free(buf);
        }
        inbox.clear();
        inbox_size = 0;
    }

    ~ggml_backend_cpu_comm_context() {
        free_inbox();
    }

    float * slot(int face, size_t dst, size_t src) const {
        char * base = (char *) ggml_backend_buffer_get_base(inbox[face*n_backends + dst]);
        return (float *) (base + (src < dst ? src : src - 1)*inbox_size);
    }
};

static void * ggml_backend_cpu_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cpu(backends[i])) {
            return nullptr;
        }
    }
#if defined(__gnu_linux__)
    // the all-reduce pairs thread i of each node, so every node runs at most the CPU count of the smallest node
    int max_threads = 0;
    int cap_node    = -1;
    for (size_t i = 0; i < n_backends; i++) {
        const int node = ((ggml_backend_cpu_context *) backends[i]->context)->numa_node;
        int n_cpus = 0;
        for (int c = 0; node >= 0 && c < GGML_MAX_N_THREADS; c++) {
            char path[256];
            snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpu%d", node, c);
            n_cpus += access(path, F_OK) == 0;
        }
        if (n_cpus == 0) {
            max_threads = 0; // not node bound, no limit
            break;
        }
        if (i == 0 || n_cpus < max_threads) {
            max_threads = n_cpus;
            cap_node    = node;
        }
    }
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) backends[i]->context;
        ctx->numa_max_threads = max_threads;
        ctx->numa_n_nodes     = (int) n_backends;
        ctx->numa_cap_node    = cap_node;
    }
#endif
    return new ggml_backend_cpu_comm_context(backends, n_backends);
}

static void ggml_backend_cpu_comm_free(void * comm_ctx) {
    delete (ggml_backend_cpu_comm_context *) comm_ctx;
}

// all-reduce runs as a graph node (get_allreduce_op), false makes meta use its fallback. Meta asserts this hook exists
static bool ggml_backend_cpu_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors) {
    GGML_UNUSED(comm_ctx);
    GGML_UNUSED(tensors);
    return false;
}

static void ggml_backend_cpu_comm_allreduce_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_backend_cpu_comm_op * op = (const ggml_backend_cpu_comm_op *) userdata;
    ggml_backend_cpu_comm_context * ctx = op->comm;
    const size_t n    = ctx->n_backends;
    const size_t rank = op->rank;

    // dst is a view of the partial result in src[0], a tensor without compute flag was not computed and counts as zero
    const bool has_compute = (dst->src[0]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    float * data = (float *) dst->data;
    const size_t ne    = ggml_nelements(dst);
    const size_t start = ne*ith/nth;
    const size_t stop  = ne*(ith + 1)/nth;

    ggml_backend_cpu_comm_flag  & flag  = ctx->flags [rank*GGML_MAX_N_THREADS + ith];
    ggml_backend_cpu_comm_state & state = ctx->states[rank*GGML_MAX_N_THREADS + ith];
    const int face  = state.face;
    const int sense = state.sense;

    for (size_t peer = 0; peer < n; peer++) {
        if (peer == rank) {
            continue;
        }
        float * slot = ctx->slot(face, peer, rank);
        if (has_compute) {
            memcpy(slot + start, data + start, (stop - start)*sizeof(float));
        } else {
            memset(slot + start, 0, (stop - start)*sizeof(float));
        }
    }
    flag.nth.store(nth, std::memory_order_relaxed);
    flag.value[face].store(sense, std::memory_order_release);

    for (size_t peer = 0; peer < n; peer++) {
        if (peer == rank) {
            continue;
        }
        const ggml_backend_cpu_comm_flag & other = ctx->flags[peer*GGML_MAX_N_THREADS + ith];
        while (other.value[face].load(std::memory_order_acquire) != sense) {
            ggml_thread_cpu_relax();
        }
        GGML_ASSERT(other.nth.load(std::memory_order_relaxed) == nth);
    }

    // sum in the order of the devices, so all devices get the same result
    const float * srcs[GGML_BACKEND_META_MAX_DEVICES];
    size_t n_srcs = 0;
    for (size_t src = 0; src < n; src++) {
        if (src != rank) {
            srcs[n_srcs++] = ctx->slot(face, rank, src);
        } else if (has_compute) {
            srcs[n_srcs++] = data;
        }
    }
    for (size_t i = start; i < stop; i++) {
        float sum = srcs[0][i];
        for (size_t k = 1; k < n_srcs; k++) {
            sum += srcs[k][i];
        }
        data[i] = sum;
    }

    state.sense = sense ^ face;
    state.face  = face ^ 1;
}

// size is the bytes of the biggest tensor to reduce
static bool ggml_backend_cpu_comm_get_allreduce_op(void * comm_ctx, size_t rank, size_t size, ggml_custom_op_t * fun, void ** userdata) {
    ggml_backend_cpu_comm_context * ctx = (ggml_backend_cpu_comm_context *) comm_ctx;
    if (ctx->n_backends < 2 || ctx->n_backends > GGML_BACKEND_META_MAX_DEVICES) {
        return false;
    }

    size = GGML_PAD(size, 64);
    if (size > ctx->inbox_size) {
        ctx->free_inbox();
        for (size_t face = 0; face < 2; face++) {
            for (size_t dst = 0; dst < ctx->n_backends; dst++) {
                ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(ctx->backends[dst], (ctx->n_backends - 1)*size);
                if (buf == nullptr) {
                    ctx->free_inbox();
                    return false;
                }
                ctx->inbox.push_back(buf);
            }
        }
        ctx->inbox_size = size;
    }

    *fun      = ggml_backend_cpu_comm_allreduce_op;
    *userdata = &ctx->ops[rank];
    return true;
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }
    // similar to ggml_backend_cuda_reg_get_proc_address in ggml-cuda.cu
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cpu_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cpu_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cpu_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_get_allreduce_op") == 0) {
        return (void *)ggml_backend_cpu_comm_get_allreduce_op;
    }
    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
