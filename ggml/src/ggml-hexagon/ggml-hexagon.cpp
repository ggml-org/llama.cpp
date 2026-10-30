#define _USE_MATH_DEFINES

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>
#include <thread>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <queue>
#include <deque>
#include <algorithm>
#include <cmath>
#include <initializer_list>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#else
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-common.h"
#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "htp-opnode.h"
#include "htp/htp-ops.h"
#include "htp_iface.h"
#include "htp-drv.h"
#include "htp-repack.h"
#include "htp-common.h"
#include "htp-kparams.h"
#include "htp-fusion.h"

using intvec  = std::vector<int>;
using uintvec = std::vector<unsigned int>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

struct ggml_hexagon_device_config {
    int         physical_idx = 0;
    int         virtual_idx  = 0;
    int         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

static ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

int    opt_arch       = 0; // autodetect
size_t opt_ndev       = 1;
size_t opt_nhvx       = 0; // use all
int    opt_nhmx       = 1; // when set, enable HMX; when 0, use HVX only
size_t opt_vtcm_size  = 0; // VTCM size override in bytes; 0 = auto-detect
size_t opt_vmem       = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
int    opt_etm        = 0;
int    opt_verbose    = 0;
int    opt_profile    = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
bool   opt_hostbuf    = false;
bool   opt_dma64      = false;
int    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

static const std::pair<const char *, int> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_LOW_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_HIGH_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_LOW_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_HIGH_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_HIGH_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

static int htp_power_mode_from_str(const char * s) {
    for (auto & e : htp_power_mode_names) {
#if defined(_MSC_VER)
        if (_stricmp(s, e.first) == 0) {
#else
        if (strcasecmp(s, e.first) == 0) {
#endif
            return e.second;
        }
    }
    return -1;
}

static const char * htp_power_mode_to_str(int mode) {
    for (auto & e : htp_power_mode_names) {
        if (e.second == mode) {
            return e.first;
        }
    }
    return "unknown";
}

size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

int    opt_mm_select     = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
int    opt_fa_select     = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
int    opt_fa_split      = 1; // 1 = head-split + team-split when n_kv_heads < N_cores (default), 0 = off
int    opt_gdn_select    = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
int    opt_ar_select     = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
int    opt_ar_scatter    = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

int    opt_opbatch   = 2048; // max number of ops in a batch
int    opt_opqueue   = 8;    // max number of pending batches
int    opt_optrace   = 0;    // trace buffer size per thread (0 means default)
int    opt_oppoll    = 0;    // polling for batch completions
int    opt_opfusion  = 1;    // enable/disable op fusion
int    opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

static std::regex* opt_opfilter = NULL; // regex of ops to not claim

static const char * status_to_str(uint32_t status) {
    switch (status) {
        case HTP_STATUS_OK:
            return "OK";
        case HTP_STATUS_NO_SUPPORT:
            return "NO-SUPPORT";
        case HTP_STATUS_INVAL_PARAMS:
            return "INVAL-PARAMS";
        case HTP_STATUS_VTCM_TOO_SMALL:
            return "VTCM-TOO-SMALL";
        case HTP_STATUS_INTERNAL_ERR:
            return "INTERNAL-ERROR";
        default:
            return "UNKNOWN";
    }
}

// ** debug helpers

static void ggml_hexagon_dump_op_exec(const char * sess_name, const htp_opnode & node, const uint32_t req_flags) {
    if (!opt_verbose) return;

    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|flags 0x%x\n",
                   sess_name, htp_format_op(node, HTP_OPFMT_EXEC).c_str(), req_flags);
}

static void ggml_hexagon_dump_op_supp(const char * sess_name, const struct ggml_tensor * op, bool supp) {
    if (!opt_verbose) return;

    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s\n",
                   sess_name, htp_format_op(op, HTP_OPFMT_SUPP).c_str(), supp ? "yes" : "no");
}

static const char * htp_event_name(uint16_t id) {
    switch (id) {
        case HTP_TRACE_EVT_DMA:            return "DMA";
        case HTP_TRACE_EVT_HVX_COMP:       return "HVX_COMP";
        case HTP_TRACE_EVT_HVX_A_QUANT:    return "HVX_A_QUANT";
        case HTP_TRACE_EVT_HVX_A_PREP:     return "HVX_A_PREP";
        case HTP_TRACE_EVT_HVX_W_DEQUANT:  return "HVX_W_DEQUANT";
        case HTP_TRACE_EVT_HVX_W_PREP:     return "HVX_W_PREP";
        case HTP_TRACE_EVT_HVX_O_PROC:     return "HVX_O_PROC";
        case HTP_TRACE_EVT_HVX_FA_QK:      return "HVX_QK_FA";
        case HTP_TRACE_EVT_HVX_FA_SFM:     return "HVX_SFM_FA";
        case HTP_TRACE_EVT_HVX_FA_Q_PREP:  return "HVX_Q_PREP";
        case HTP_TRACE_EVT_HVX_FA_K_PREP:  return "HVX_K_PREP";
        case HTP_TRACE_EVT_HVX_FA_V_PREP:  return "HVX_V_PREP";
        case HTP_TRACE_EVT_HVX_GDN_PREP:   return "HVX_GDN_PREP";
        case HTP_TRACE_EVT_HVX_GDN_SOLVE:  return "HVX_GDN_SOLVE";
        case HTP_TRACE_EVT_HVX_GDN_V_PREP: return "HVX_GDN_V_PREP";
        case HTP_TRACE_EVT_HVX_GDN_D_PREP: return "HVX_GDN_D_PREP";
        case HTP_TRACE_EVT_HVX_GDN_OUT:    return "HVX_GDN_OUT";
        case HTP_TRACE_EVT_HVX_GDN_STATE:  return "HVX_GDN_STATE";
        case HTP_TRACE_EVT_HVX_GDN_REM:    return "HVX_GDN_REM";
        case HTP_TRACE_EVT_HMX_COMP:       return "HMX_COMP";
        case HTP_TRACE_EVT_L2FLUSH:        return "L2FLUSH";
        case HTP_TRACE_EVT_INIT:           return "INIT";
        case HTP_TRACE_EVT_BUFF:           return "BUFF";
        case HTP_TRACE_EVT_FENCE:          return "FENCE";
        default:                           return "UNKNOWN";
    }
}

static void ggml_hexagon_dump_op_prof(const char * sess_name, const htp_opnode & node, const htp_prof_desc & pd) {
    if (!opt_profile) return;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    const uint32_t * pmu = pd.pmu;

    char pmu_str[256] = "";
    if (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, sizeof(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    float mhz = op_usec > 0 ? (float) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|usec %u cycles %u start %u mhz %.1f%s\n",
                   sess_name, htp_format_op(node, HTP_OPFMT_PROF).c_str(), op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

static void ggml_hexagon_dump_batch_prof(const char * sess_name, const htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    float batch_mhz = rsp.usecs > 0 ? (float) batch_cycles / rsp.usecs : 0.0f;

    char evt_str[256] = "----";
    if (opt_profile == 3) {
        snprintf(evt_str, sizeof(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name, rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

static void ggml_hexagon_dump_trace_events(const char * sess_name, const htp_opbatch_rsp & rsp,
                                           const htp_trace_desc * trace_events, uint32_t n_traces) {
    if (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            for (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                const auto & e = trace_events[t * n_traces + idx];
                bool is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name, htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

// ** backend sessions

struct ggml_hexagon_opbatch;
struct ggml_hexagon_opqueue;
struct ggml_hexagon_shared_buffer;
struct ggml_hexagon_fence_buffer;
struct ggml_hexagon_session;
struct ggml_backend_hexagon_device_context;

struct ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

struct ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    volatile uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

struct ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    volatile uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

struct ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    int              phys_idx;
    int              virt_idx;
    bool             valid_session;
    bool             valid_handle;
    bool             valid_queue;
    bool             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<int, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_threads   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_size   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> last_error{HTP_STATUS_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_group                mdev;
    ggml_backend_dev_t                     dev       = nullptr;
    ggml_backend_hexagon_device_context *  dev_ctx   = nullptr;
    ggml_hexagon_fence_buffer *            fence_buf = nullptr;

    ggml_hexagon_session(const ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_count = 0) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    const char* c_name() const { return name.c_str(); }

    void allocate(const ggml_hexagon_device_config & config) noexcept(false);
    void release() noexcept(true);

    uint8_t *   alloc_fence(uint32_t n_slots = 1);
    void        free_fence(void * ptr, uint32_t n_slots = 1);
    ggml_tensor make_fence_tensor(const volatile void * slot, uint32_t n_slots = 1) const;
    uint32_t    next_fence_seq() {
        if (++fence_seq == 0) fence_seq = 1;
        return fence_seq;
    }

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, volatile uint32_t *> cpy_fence_slots;

    void enqueue_mdev_group();
    void enqueue_op(const htp_opnode & node);
    void enqueue_cpy(const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    void enqueue_fence(const ggml_tensor * sync_tensor, uint32_t fence_seq = 0, bool wait = true);
    void enqueue_allreduce(const ggml_tensor * dst, const std::vector<const ggml_tensor *> & src_tensors,
                           const std::vector<const ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    void start_batch();
    void flush_sync(bool all = true);
    void flush_async();
    void flush_batch(size_t min_ops = 1);
    void flush_peers();
    void flush_pending(bool all = true);

    ggml_hexagon_shared_buffer * mmap_tensor(const ggml_tensor * t);
    bool clone_buffer(const ggml_hexagon_shared_buffer*);
    void release_buffer(const ggml_hexagon_shared_buffer*);
    void unclone_buffer(const ggml_hexagon_shared_buffer*);

    void add_peer(ggml_hexagon_session * peer) {
        if (this->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } else {
            phys_peers.insert(peer);
        }
    }
};

const char * htp_session_name(const struct ggml_hexagon_session * sess) { return sess->c_name(); }
size_t       htp_session_vtcm_size(const struct ggml_hexagon_session * sess) { return sess->vtcm_size; }
uint32_t     htp_session_n_threads(const struct ggml_hexagon_session * sess) { return sess->n_threads; }
uint32_t     htp_session_n_hmx(const struct ggml_hexagon_session * sess) { return sess->n_hmx; }
uint32_t     htp_session_mdev_count(const struct ggml_hexagon_session * sess) { return sess->mdev.count; }

// ** backend buffers

struct ggml_backend_hexagon_device_context {
    int                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(int dev_id, const ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    const char * c_name() const { return config.name.c_str(); }

    ggml_hexagon_session * session() {
        if (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        return sess.get();
    }
};

struct ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(const std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        this->dev_ctx = dev_ctx;
        this->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

struct ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    int       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        if (!base) {
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        if (fd < 0) {
            rpcmem_free(base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        this->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        if (base) {
            rpcmem_free(base);
        }
    }
};

struct ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<htp_tensor_extra *>            tensor_extra;
    bool     mapped;
    bool     pinned;
    bool     extended;

    const char * c_name() const { return sess->c_name(); }
    uint8_t *    base()   const { return mem ? mem->base : nullptr; }
    size_t       size()   const { return mem ? mem->size : 0;  }
    int          fd()     const { return mem ? mem->fd   : -1; }

    void mmap(bool extended = false) {
        if (!this->mem)   return;
        if (this->mapped) return;

        GGML_ASSERT(!this->pinned || !extended);

        this->extended = extended;
        fastrpc_map_flags flags = this->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        int err = fastrpc_mmap(sess->domain_id, fd(), (void *) base(), 0, size(), flags);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (void *) base(), size(), fd(), pinned, extended);

        this->mapped = true;
    }

    void unmap() {
        if (!this->mapped) return;

        if (!this->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        if (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (void *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) base(), size(), fd());

        this->mapped = false;
    }

    void alloc(size_t size) {
        if (this->mem) return;

        this->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (void *) base(), this->size(), fd(), (int) pinned);
        if (this->pinned) {
            mmap();
        }
    }

    void free() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (void *) base(), size(), fd());
        this->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, bool pinned = false) {
        this->sess     = sess;
        this->mapped   = false;
        this->pinned   = pinned;
        this->extended = false;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        if (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, const ggml_hexagon_shared_buffer & other) {
        this->sess   = sess;
        this->mem    = other.mem;
        this->mapped = false;
        this->pinned = other.pinned;
        this->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        free();
        for (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

struct ggml_hexagon_fence_buffer : public ggml_hexagon_shared_buffer {
    uint32_t              slot_count = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, false /* pinned */),
          slot_count(size / HTP_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(this);
        backend_buffer.size    = size;
        mmap(false);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        if (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * HTP_FENCE_SLOT_SIZE;
        } else if (slot_head + n_slots <= slot_count) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * HTP_FENCE_SLOT_SIZE;
        }
        if (ptr) {
            memset(ptr, 0, (size_t) n_slots * HTP_FENCE_SLOT_SIZE);
        }
        return ptr;
    }

    void free_slot(void * ptr, uint32_t n_slots = 1) {
        if (!ptr) return;
        uint32_t slot = ((uint8_t *) ptr - base()) / HTP_FENCE_SLOT_SIZE;
        for (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

inline uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    return ptr;
}

inline void ggml_hexagon_session::free_fence(void * ptr, uint32_t n_slots) {
    if (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

inline ggml_tensor ggml_hexagon_session::make_fence_tensor(const volatile void * slot, uint32_t n_slots) const {
    static htp_tensor_extra fence_extra { {}, 0, HTP_TENSOR_FENCE };

    ggml_tensor t {};
    t.buffer = &this->fence_buf->backend_buffer;
    t.extra  = &fence_extra;
    t.data   = const_cast<void *>(slot);
    t.type   = GGML_TYPE_I8;
    t.ne[0]  = HTP_FENCE_SLOT_SIZE;
    t.ne[1]  = (int64_t) n_slots;
    t.ne[2]  = 1;
    t.ne[3]  = 1;
    t.nb[0]  = 1;
    t.nb[1]  = HTP_FENCE_SLOT_SIZE;
    t.nb[2]  = t.nb[1] * t.ne[1];
    t.nb[3]  = t.nb[2];
    t.op     = GGML_OP_NONE;
    return t;
}

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    return sbuf->base();
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (void *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = new htp_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (htp_tensor_extra *)           tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= HTP_TENSOR_WEIGHT;
        if (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= HTP_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, extra->flags);

    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    if (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        return;
    }

    if (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    if (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   tensor,
                                                   void *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (htp_tensor_extra *)           tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, extra->flags);

    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   const struct ggml_tensor * src,
                                                   struct ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    return false;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

static void ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          const void *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (htp_tensor_extra *)           tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= HTP_TENSOR_WEIGHT;
        if (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= HTP_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (int) buffer->usage, extra->flags);

    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
        }
        return;
    }

    if (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    if (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      const ggml_tensor *   tensor,
                                                      void *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (htp_tensor_extra *)           tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (int) buffer->usage);

    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (const uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        return;
    }

    size_t temp_size      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t slice_size     = tensor->ne[1] * ggml_row_size(tensor->type, tensor->ne[0]);
    size_t slice_offset   = offset % slice_size;
    size_t row_size_bytes = ggml_row_size(tensor->type, tensor->ne[0]);

    GGML_ASSERT((slice_offset % row_size_bytes) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((temp_size % row_size_bytes)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((offset + temp_size) <= ggml_nbytes(tensor));

    std::vector<uint8_t> temp_buf(temp_size);
    htp_unpack_tensor_tiled(temp_buf.data(), tensor, offset, temp_size);

    for (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (void *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

// ** backend buffer type

static void ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        const void *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        const ggml_tensor *   tensor,
                                                        void *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

// ** backend buffer type

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    return static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    if (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        return nullptr;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size, false);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        return nullptr;
    }
}

static ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    if (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        return nullptr;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size, false);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        return nullptr;
    }
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;  // HVX alignment
    GGML_UNUSED(buft);
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
    if (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        return std::max(tiled_size, ggml_nbytes(t));
    }
    return ggml_nbytes(t);
}

static size_t parse_size(const char * str, size_t default_unit = 1024 * 1024) {
    if (!str || str[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    double val = strtod(str, &end);
    if (val < 0) {
        return 0;
    }
    if (end && *end) {
        while (*end == ' ') end++;
        if (*end == 'k' || *end == 'K') {
            return (size_t) (val * 1024);
        } else if (*end == 'm' || *end == 'M') {
            return (size_t) (val * 1024 * 1024);
        } else if (*end == 'g' || *end == 'G') {
            return (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    return (size_t) (val * default_unit);
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

struct ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    int    first;
    int    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

static const char * ggml_hexagon_kv_layer_suffix(const struct ggml_tensor * t) {
    if (strncmp(t->name, "cache_", 6) != 0) {
        return NULL;
    }
    const char * p = strstr(t->name, "_l");
    if (!p || !isdigit((unsigned char)p[2])) {
        return NULL;
    }
    return p;
}

struct ggml_backend_hexagon_alloc_unit {
    size_t size;
    int    first;
    int    last;
};

static ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    int i = 0;
    while (i < n_tensors) {
        struct ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        int unit_first = i;
        int unit_last = i + 1;

        if (t->data == NULL && t->view_src == NULL) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        const char * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        while (unit_last < n_tensors) {
            struct ggml_tensor * next = tensors[unit_last];

            if (next->view_src != NULL) {
                unit_last++;
                continue;
            }

            if (layer_suffix != NULL) {
                const char * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                if (next_suffix != NULL && strcmp(layer_suffix, next_suffix) == 0) {
                    if (next->data == NULL) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    continue;
                }
            }

            break;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    int    cur_buf_first = 0;

    for (const auto & unit : units) {
        if (unit.size == 0) {
            continue;
        }

        if (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    return plan;
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const auto & item : plan) {
        total += item.size;
    }
    return total;
}

static ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

static ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(int dev_id, const ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name, this);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", this);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", this);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

static bool ggml_backend_buffer_is_hexagon(const struct ggml_backend_buffer * b) {
    return b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

struct ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<const ggml_tensor*, int> t_map; // tensor ptr  to index
    std::unordered_multimap<void*, int>         d_map; // tensor data to index

    unsigned int n_bufs;     // num buffers in the batch
    unsigned int n_tens;     // num tensors ...
    unsigned int n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned int n_bufs_max;
    unsigned int n_tens_max;
    unsigned int n_ops_max;
    size_t       b_vmem_max;

    void reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size, size_t max_vmem) {
        this->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_size;
        n_tens_max = std::min<size_t>(n_ops_max * 4 + HTP_OP_MAX_INPUTS + HTP_OP_MAX_OUTPUTS, HTP_OP_MAX_TENSORS);

        b_vmem_max = max_vmem;

        ops.resize(n_ops_max);

        h_bufs.resize(n_bufs_max);
        h_tens.resize(n_tens_max);
        h_ops.resize(n_ops_max);

        t_map.reserve(n_tens_max);
        d_map.reserve(n_tens_max);

        GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
                sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);

        reset();
    }

    bool empty() const { return n_ops == 0; }
    bool full()  const { return n_ops >= n_ops_max; }

    int find_buffer(int fd) const {
        for (unsigned int i = 0; i < n_bufs; i++) {
            if ((int) h_bufs[i].fd == fd) {
                return (int) i;
            }
        }
        return -1;
    }

    // add buffer and return its index
    int add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        int bi = find_buffer(sbuf->fd());
        if (bi >= 0) { return bi; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        bi = n_bufs++;

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        if (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (void*) sbuf->base(), (size_t) b.size, b_vmem);

        return bi;
    }

    static void init_tensor_shape(htp_tensor & h, const ggml_tensor * t) {
        auto extra = (htp_tensor_extra *) t->extra;

        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        if (extra->flags & HTP_TENSOR_TILED) {
            if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = sizeof(ggml_fp16_t);
            } else {
                h.nb[0] = t->nb[0];
            }
            const int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            const int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
        } else {
            h.nb[0] = t->nb[0];
            h.nb[1] = t->nb[1];
            h.nb[2] = t->nb[2];
            h.nb[3] = t->nb[3];
            h.size  = ggml_nbytes(t);
        }

        h.flags = extra->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_TILED | HTP_TENSOR_FENCE);
    }

    static bool same_shape(const htp_tensor & a, const htp_tensor & b) {
        return (a.type == b.type) &&
               (a.ne[0] == b.ne[0]) && (a.ne[1] == b.ne[1]) && (a.ne[2] == b.ne[2]) && (a.ne[3] == b.ne[3]) &&
               (a.nb[0] == b.nb[0]) && (a.nb[1] == b.nb[1]) && (a.nb[2] == b.nb[2]) && (a.nb[3] == b.nb[3]);
    }

    // add tensor and return its index
    int add_tensor(const ggml_tensor * t) {
        // Lookup by tensor ptr
        auto it = t_map.find(t);
        if (it != t_map.end()) { return it->second; }

        htp_tensor shape {};
        init_tensor_shape(shape, t);

        // Lookup by tensor data and shape
        auto range = d_map.equal_range(t->data);
        for (auto dit = range.first; dit != range.second; ++dit) {
            if (same_shape(h_tens[dit->second], shape)) {
                t_map.insert({t, dit->second});
                return dit->second;
            }
        }

        auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // Add new tensor to the batch
        int ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->data, ti});

        uint64_t t_offset = (uint8_t *) t->data - sbuf->base();

        htp_tensor & h = h_tens[ti];
        h      = shape;
        h.bi   = add_buffer(sbuf);
        h.ti   = ti;
        h.data = t_offset;

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (void*) t->data, (size_t) t_offset, (size_t) h.size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        return ti;
    }

    bool fit_tensors(const htp_opnode & node) const {
        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        const bool check_tens = (n_tens + node.inputs.size() + node.outputs.size()) > n_tens_max;

        int seen_bufs[HTP_OP_MAX_BUFS];
        int n_seen_bufs = 0;

        auto fit_tensor = [&](const ggml_tensor * t) {
            if (!t) return;
            if (check_tens) {
                if (t_map.count(t)) return;
                extra_tens++;
            }

            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            int fd = sbuf->fd();
            if (find_buffer(fd) < 0) {
                for (int i = 0; i < n_seen_bufs; i++) {
                    if (seen_bufs[i] == fd) return;
                }
                if (n_seen_bufs < HTP_OP_MAX_BUFS) {
                    seen_bufs[n_seen_bufs++] = fd;
                }
                if (!sbuf->extended) {
                    extra_vmem += sbuf->size();
                }
                extra_bufs += 1;
            }
        };

        for (const auto * src : node.get_inputs()) {
            fit_tensor(src);
        }
        for (const auto * output : node.get_outputs()) {
            fit_tensor(output);
        }

        if ((extra_bufs + n_bufs) > n_bufs_max) return false;
        if (check_tens && (extra_tens + n_tens) > n_tens_max) return false;
        if ((extra_vmem + b_vmem) > b_vmem_max) return false;

        return true;
    }

    void sync_op_desc(unsigned int n) {
        const htp_opnode & node = ops[n];
        htp_op_desc & o = h_ops[n];

        memcpy(o.params,        node.params,        sizeof(o.params));
        memcpy(o.kernel_params, node.kernel_params, sizeof(o.kernel_params));
        o.opcode = node.opcode;

        const auto & inputs = node.get_inputs();
        for (unsigned int i = 0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i]) ? add_tensor(inputs[i]) : 0xffff;
        }

        const auto & outputs = node.get_outputs();
        for (unsigned int i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    // caller must ensure the op fits
    void add_op(const htp_opnode & node) {
        unsigned int n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = node;
        h_ops[n].flags = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], h_ops[n].flags);
        sync_op_desc(n);
    }

    void sort_buffers() {
        if (n_bufs <= 1) return;

        std::vector<int> order(n_bufs);
        for (unsigned int i = 0; i < n_bufs; i++) { order[i] = (int) i; }

        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return h_bufs[a].size > h_bufs[b].size;
        });

        bool already_sorted = true;
        for (unsigned int i = 0; i < n_bufs; i++) {
            if (order[i] != (int) i) {
                already_sorted = false;
                break;
            }
        }
        if (already_sorted) return;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        for (unsigned int new_bi = 0; new_bi < n_bufs; new_bi++) {
            int old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        for (unsigned int i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        for (unsigned int i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    void update_mdev_group(uint32_t mdev_idx) {
        if (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    bool try_fuse(const htp_opnode & node) {
        if (n_ops == 0) {
            return false;
        }

        htp_opnode & last_node = ops[n_ops - 1];
        if (!htp_fusion_try_fuse(sess, last_node, node)) {
            return false;
        }

        sync_op_desc(n_ops - 1);
        HEX_VERBOSE("ggml-hex: %s fused %s (N=%zu, #%u)\n",
                    sess->c_name(), last_node.name().c_str(), last_node.fused.size() + 1, n_ops - 1);
        return true;
    }
};

struct ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

struct ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_buffer *shm_buf;
    size_t                      shm_blk_size;
    size_t                      depth;

    using opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_size, size_t depth) : depth(depth) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_size;
        size_t n_tensors = std::min<size_t>(n_ops * 4 + HTP_OP_MAX_INPUTS + HTP_OP_MAX_OUTPUTS, HTP_OP_MAX_TENSORS);

        size_t tr_size = 0;
        if (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * sizeof(htp_trace_desc);
        }

        shm_blk_size = sizeof(htp_buf_desc)  * n_bufs    +
                       sizeof(htp_tensor)    * n_tensors +
                       sizeof(htp_op_desc)   * n_ops     +
                       sizeof(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = new ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, true /* pinned */);

        op_cache.resize(depth);

        if (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() const { return shm_buf ? shm_buf->size() : 0; }

    // push new batch
    bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, const ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(sizeof(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(sizeof(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(sizeof(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(sizeof(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(sizeof(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(sizeof(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        if (seq - shm_buf->sess->batch_rsp_seq > depth) { return false; }

        const uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        if (opt_profile) {
            op_cache[slot].assign(op_batch->ops.begin(), op_batch->ops.begin() + req.n_ops);
        }

        const size_t b_size = sizeof(htp_buf_desc)  * req.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * req.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * req.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        if (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * sizeof(htp_trace_desc);
        } else {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->base() + ((size_t) slot * shm_blk_size);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base();
        dbuf.size     = b_size + t_size + o_size + p_size + tr_size;

        GGML_ASSERT(dbuf.size <= shm_blk_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
        uint8_t * o_ptr = m_ptr;

        memcpy(b_ptr, (void *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (void *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (void *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        if (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            for (unsigned int i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (void *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            for (unsigned int i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        return true;
    }

    void pop(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        const uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        const size_t b_size = sizeof(htp_buf_desc)  * rsp.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * rsp.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * rsp.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        if (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * sizeof(htp_trace_desc);
        }

        const size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        if (rsp.n_ops > 0 && opt_profile) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            const htp_prof_desc * pd = (const htp_prof_desc *) p_ptr;
            const htp_trace_desc * trace_events = nullptr;
            if (opt_profile == 3) {
                trace_events = (const htp_trace_desc *) (p_ptr + p_size);
            }

            ggml_hexagon_dump_batch_prof(shm_buf->sess->c_name(), rsp);

            for (uint32_t i = 0; i < rsp.n_ops; i++) {
                ggml_hexagon_dump_op_prof(shm_buf->sess->c_name(), ops[i], pd[i]);
            }

            ggml_hexagon_dump_trace_events(shm_buf->sess->c_name(), rsp, trace_events, n_traces);
        }
    }
};

void ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    for (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    for (auto * peer : ppeers) {
        peer->flush_async();
    }

    for (auto & sub : this->mdev.sessions) {
        sub->flush_peers();
    }
}

void ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

void ggml_hexagon_session::flush_pending(bool all) {
    for (auto & sub : this->mdev.sessions) {
        sub->flush_pending(all);
        if (sub->last_error > HTP_STATUS_OK) {
            this->last_error = sub->last_error.load();
        }
    }

    while (this->batch_rsp_seq < this->batch_req_seq) {
        struct htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        struct dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        const uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        if (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            continue;
        }

        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        if (rsp_size != sizeof(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", this->c_name(), rsp_size, n_dbufs);
        }

        if (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", this->c_name(), status_to_str(rsp.status));
            this->last_error = rsp.status;
            for (auto & sub : this->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->pop(rsp, dbuf);

        GGML_ASSERT(rsp.seq == this->batch_rsp_seq + 1);
        this->batch_rsp_seq = rsp.seq;

        if (!all) break;
    }
}

void ggml_hexagon_session::flush_sync(bool all) {
    flush_async();
    flush_pending(all);
}

void ggml_hexagon_session::start_batch() {
    if (this->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

void ggml_hexagon_session::flush_batch(size_t min_ops) {
    if (op_batch->n_ops < min_ops) { return; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    const uint64_t seq = ++this->batch_req_seq;

    op_batch->update_mdev_group(this->mdev.idx);

    if (!op_queue->push(req, dbuf, op_batch, seq)) {
        flush_pending(false);
        op_queue->push(req, dbuf, op_batch, seq);
    }

    for (auto & sub : this->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        if (!sub->op_queue->push(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(false);
            sub->op_queue->push(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        int err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, sizeof(sub_req), (const uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        if (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);

    int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
    if (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    auto clone_tensor_buffer = [this](const ggml_tensor * t) {
        auto sbuf = this->mmap_tensor(t);
        if (!sbuf) return;
        if (sbuf->sess != this) {
            this->clone_buffer(sbuf);
        }
        for (auto & sub : this->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    for (auto t : node.get_inputs()) {
        clone_tensor_buffer(t);
    }
    for (auto t : node.get_outputs()) {
        clone_tensor_buffer(t);
    }

    const bool fits_tensors = op_batch->fit_tensors(node);
    if (opt_opfusion && fits_tensors && op_batch->try_fuse(node)) {
        return;
    }

    if (op_batch->full() || !fits_tensors) {
        flush_async();
    }

    if (op_batch->empty()) {
        start_batch();
        if (!op_batch->fit_tensors(node)) {
            GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n", c_name());
        }
    }

    op_batch->add_op(node);
}

void ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_node(HTP_OP_MDEV_GROUP);

    ggml_tensor dummy_t = make_fence_tensor(this->mdev_fence_slot, this->mdev.count);
    dummy_t.op_params[0] = (int32_t) this->mdev.idx;

    ggml_tensor * node = group_node.add_dummy(dummy_t);
    node->src[0] = node;
    group_node.init(node);
    group_node.outputs.clear();

    for (auto & sub : this->mdev.sessions) {
        sub->clone_buffer(this->fence_buf);
    }

    op_batch->add_op(group_node);
}

void ggml_hexagon_session::enqueue_cpy(const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * sync_tensor, uint32_t fence_seq) {
    const bool with_fence = sync_tensor != nullptr;
    htp_opnode cpy_node(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* node = cpy_node.add_dummy(*dst);
    node->op     = GGML_OP_CPY;
    node->src[0] = const_cast<ggml_tensor *>(src);
    node->src[1] = with_fence ? cpy_node.add_dummy(*sync_tensor) : nullptr;
    if (with_fence) {
        node->op_params[0] = (int32_t) fence_seq;
    }

    cpy_node.init(node);
    const bool ok = htp_kparams_solve(this, node, cpy_node.kernel_params);
    if (ok && !with_fence && cpy_node.kparams<htp_copy_kernel_params>()->total_elems == 0) {
        return;
    }
    this->enqueue_op(cpy_node);
}

void ggml_hexagon_session::enqueue_fence(const ggml_tensor * sync_tensor, uint32_t fence_seq, bool wait) {
    htp_opnode sync_node(HTP_OP_FENCE);

    ggml_tensor* node = sync_node.add_dummy(*sync_tensor);
    node->op           = GGML_OP_NONE;
    node->src[0]       = node;
    node->op_params[0] = (int32_t) fence_seq;
    node->op_params[1] = wait ? 0 : 1;

    sync_node.init(node);
    this->enqueue_op(sync_node);
}

void ggml_hexagon_session::enqueue_allreduce(
    const ggml_tensor * dst,
    const std::vector<const ggml_tensor *> & src_tensors,
    const std::vector<const ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_node(HTP_OP_ALLREDUCE);

    ggml_tensor* node = ar_node.add_dummy(*dst);
    node->op           = GGML_OP_NONE;
    node->op_params[0] = (int32_t) fence_seq_entry;
    node->op_params[1] = (int32_t) fence_seq_exit;

    ar_node.init(node);

    ar_node.inputs.clear();
    for (size_t i = 0; i < src_tensors.size(); i++) {
        ar_node.inputs.push_back(src_tensors[i]);
    }
    for (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_node.inputs.push_back(ar_node.add_dummy(*sync_tensors[i]));
    }

    ar_node.outputs.clear();
    for (size_t i = 0; i < src_tensors.size(); i++) {
        ar_node.outputs.push_back(src_tensors[i]);
    }

    htp_kparams_solve_allreduce(
        this, dst, rank, n_ranks, false, false, /*is_shard_ok=*/ false,
        ar_node.kparams<htp_allreduce_kernel_params>());

    this->enqueue_op(ar_node);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(const ggml_tensor * t) {
    if (!t) return nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    if (!sbuf->mapped) {
        const bool is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        const bool extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    return sbuf;
}

bool ggml_hexagon_session::clone_buffer(const ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    if (sbuf->sess == this) return true;

    auto mem = sbuf->mem;
    int   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    if (this->cloned_buffers.find(fd) != this->cloned_buffers.end()) return true;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", this->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(this, *sbuf);
    try {
        clone->mmap(sbuf->extended);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", this->c_name(), exc.what());
        return false;
    }

    this->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(this);
    return true;
}

void ggml_hexagon_session::release_buffer(const ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    int   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = this->cloned_buffers.find(fd);
    if (it != this->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        this->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(this);
}

void ggml_hexagon_session::unclone_buffer(const ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    for (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

static size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    const size_t MiB = 1024 * 1024;
    const size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    try {
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;

        while (1) {
            sbufs.push_back(new ggml_hexagon_shared_buffer(sess, step, true));
            vmem += step;
        }
    } catch (...) { }

    for (auto b : sbufs) { delete b; }

    return vmem - step; // backoff to account for overhead from internal mappings
}

void ggml_hexagon_session::allocate(const ggml_hexagon_device_config & config) noexcept(false) {
    int phys_idx = config.physical_idx;
    int virt_idx = config.virtual_idx;

    this->valid_session = false;
    this->valid_handle  = false;
    this->valid_queue   = false;
    this->valid_iface   = false;

    this->name          = config.name;
    this->phys_idx      = phys_idx;
    this->virt_idx      = virt_idx;
    this->domain_id     = config.domain_id;
    this->session_id    = 0;
    this->batch_req_seq = 0;
    this->batch_rsp_seq = 0;
    this->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", this->name.c_str(), this->domain_id, phys_idx, virt_idx);

    if (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        throw std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    const std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    if (virt_idx > 0) {
        struct remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<char *>(dom_name.c_str());
        n.session_name     = const_cast<char *>(this->name.c_str());
        n.session_name_len = this->name.size();
        n.session_id       = virt_idx;

        int err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (void *) &n, sizeof(n));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           this->c_name(), phys_idx, virt_idx, err);
            throw std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        this->session_id    = n.session_id;
        this->domain_id     = n.effective_domain_id;
        this->valid_session = true;
    } else {
        struct remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<char *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        int err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (void *) &eff, sizeof(eff));
        if (err == AEE_SUCCESS) {
            this->domain_id = eff.effective_domain_id;
        } else {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           this->name.c_str(), err, this->domain_id);
        }
    }

    // Enable unsigned modules
    {
        struct remote_rpc_control_unsigned_module u;
        u.domain = this->domain_id;
        u.enable = 1;
        int err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", this->c_name(), err);
            throw std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    char session_uri[256];
    {
        char htp_uri[256];
        snprintf(htp_uri, sizeof(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        struct remote_rpc_get_uri u = {};
        u.session_id      = this->session_id;
        u.domain_name     = const_cast<char *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<char *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);

        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            snprintf(session_uri, sizeof(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), this->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          this->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    int err = htp_iface_open(session_uri, &this->handle);
    if (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", this->c_name(), session_uri, err);
        throw std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    this->valid_handle = true;

    // Query HW info and resolve session options
    {
        unsigned int hw_n_threads = 0;
        unsigned int hw_n_hvx     = 0;
        unsigned int hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        int hw_err = htp_iface_hwinfo(this->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        if (hw_err == 0) {
            const uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            this->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            this->n_hvx     = this->n_threads;
            this->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            this->vtcm_size = (uint64_t)hw_vtcm_size;
            if (opt_vtcm_size > 0) {
                if (opt_vtcm_size > hw_vtcm_size) {
                    GGML_LOG_WARN("ggml-hex: %s VTCM override %zu exceeds HW size %llu\n", this->c_name(), opt_vtcm_size, hw_vtcm_size);
                }
                this->vtcm_size = opt_vtcm_size;
            }
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          this->c_name(), this->n_threads, this->n_hvx, this->n_hmx,
                          (unsigned long long)(this->vtcm_size / (1024 * 1024)));
        } else {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", this->c_name(), hw_err);
            const uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            this->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            this->n_hvx     = this->n_threads;
            this->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            this->vtcm_size = opt_vtcm_size > 0 ? opt_vtcm_size : 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        struct remote_rpc_control_latency l;
        l.enable = 1;

        int err = remote_handle64_control(this->handle, DSPRPC_CONTROL_LATENCY, (void *) &l, sizeof(l));
        if (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", this->c_name(),
                  this->session_id, this->domain_id, session_uri, (unsigned long) this->handle);

    const size_t req_q_size = (sizeof(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    const size_t rsp_q_size = (sizeof(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(this->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (void *) this,  // Callback context
                          &queue);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", this->name.c_str(), (unsigned) err);
        throw std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    this->valid_queue = true;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &this->queue_id);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        throw std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    if (opt_etm) {
        err = htp_iface_etm(this->handle, 1);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    this->op_queue = new ggml_hexagon_opqueue(this, opt_opbatch, opt_opqueue);

    this->fence_buf = new ggml_hexagon_fence_buffer(this, &dev_ctx->fence_buffer_type, 64 * 1024);
    if (this->mdev.count > 1) {
        this->mdev_fence_slot = this->alloc_fence(this->mdev.count);
    }

    if (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(this);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", this->c_name(), opt_vmem);
    }
    const size_t shm_size = this->op_queue->shm_size();
    this->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch, this->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(this->handle, this->session_id, this->queue_id, this->n_threads, opt_nhmx, opt_vtcm_size, this->max_vmem,
                          (uint32_t) opt_power_mode);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", this->c_name(), (unsigned) err);
        throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    this->valid_iface = true;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", this->c_name(), htp_power_mode_to_str(opt_power_mode));

    if (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(this->handle, opt_profile, &pmu_conf);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

void ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", this->name.c_str());

    this->mdev.sessions.clear();

    int err;

    if (this->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(this->handle);
        if (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete this->op_batch;
    delete this->op_queue;
    for (auto & it : this->cpy_fence_slots) {
        free_fence((void *) it.second, 1);
    }
    this->cpy_fence_slots.clear();

    if (this->fence_buf) {
        unclone_buffer(this->fence_buf);
        delete this->fence_buf;
        this->fence_buf = nullptr;
    }
    while (!this->cloned_buffers.empty()) {
        release_buffer(this->cloned_buffers.begin()->second.get());
    }

    if (opt_etm) {
        err = htp_iface_etm(this->handle, 0);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    if (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(this->handle, 0, &pmu_conf);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    if (this->valid_queue) {
        err = dspqueue_close(queue);
        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    if (this->valid_handle) {
        htp_iface_close(this->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(const ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_count) noexcept(false) {
    this->dev        = dev;
    this->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    this->mdev.idx   = mdev_idx;
    this->mdev.count = mdev_count > 0 ? mdev_count : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)this) & 0xFFFF;

    try {
        allocate(config);
        if (this->mdev.idx == 0 && !config.mdev_group.empty()) {
            for (size_t i = 0; i < config.mdev_group.size(); i++) {
                this->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], this->dev, (uint32_t) (i + 1), this->mdev.count));
            }
        }
    } catch (const std::exception & exc) {
        release();
        throw;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
    release();
}

// ** backend interface

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    return sess->c_name();
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

static inline bool op_is_compute(ggml_tensor *node)
{
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    if (sess->last_error > HTP_STATUS_OK) {
        return GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    const std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    bool cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    if (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } else {
        htp_fusion_tag_graph(graph);

        computed_nodes.reserve(graph->n_nodes);

        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            if (!op_is_compute(n)) {
                continue;
            }

            const htp_op_code opcode = htp_op_remap(n);
            if (opcode == HTP_OP_INVALID) {
                GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(n));
            }

            htp_opnode node(opcode, n);
            if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = node.node->src[0];
                if (htp_needs_repack(src0)) {
                    HEX_VERBOSE("ggml-hex: %s rt-repack %s|size %zu\n", sess->c_name(), htp_format_tensor(src0, HTP_OPFMT_TENSOR).c_str(), ggml_nbytes(src0));
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    memcpy(raw.data(), src0->data, raw.size());
                    htp_repack_tensor_tiled(src0, raw.data(), raw.size());
                    auto * extra = (htp_tensor_extra *) src0->extra;
                    extra->flags |= HTP_TENSOR_TILED;
                }
            }

            GGML_ASSERT(htp_kparams_solve(sess, node));
            if (node.opcode == HTP_OP_CPY && node.kparams<htp_copy_kernel_params>()->total_elems == 0) {
                continue;
            }

            computed_nodes.push_back(std::move(node));
        }

        if (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } else {
            nodes_ptr = &computed_nodes;
        }
    }

    for (const auto & node : *nodes_ptr) {
        sess->enqueue_op(node);
    }

    if (sess->last_error > HTP_STATUS_OK) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    if (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

enum ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

struct ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

struct ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    void reset() {
        ranges.clear();
    }

    void add(const ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    bool check(const ggml_hexagon_mem_range & mr) const {
        for (const auto & cmp : ranges) {
            if (mr.pb != cmp.pb) {
                continue;
            }
            if (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                continue;
            }
            if (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                return false;
            }
        }
        return true;
    }
};

static ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(const ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    const ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    if (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    return mr;
}

struct node_info {
    ggml_tensor * node;
    htp_opvec<ggml_tensor *, HTP_OP_MAX_FUSED> fused;

    const ggml_tensor * dst() const { return fused.empty() ? node : fused.back(); }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};

static void ggml_hexagon_mem_ranges_add_node(ggml_hexagon_mem_ranges & mrs, const node_info & node) {
    if (node.is_empty()) return;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node.node->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(node.node->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    for (const auto * fused : node.fused) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(node.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

static bool ggml_hexagon_mem_ranges_check_node(const ggml_hexagon_mem_ranges & mrs, const node_info & node) {
    if (node.is_empty()) return true;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node.node->src[i]) {
            if (!mrs.check(ggml_hexagon_mem_range_from_tensor(node.node->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                return false;
            }
        }
    }
    for (const auto * fused : node.fused) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (fused->src[i]) {
                if (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    return false;
                }
            }
        }
    }
    return mrs.check(ggml_hexagon_mem_range_from_tensor(node.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

static std::vector<int> ggml_hexagon_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        if (!htp_fusion_can_stack(node0.node)) {
            res.push_back(i0);
            used[i0] = true;
            continue;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr int N_FORWARD = 64;

        htp_opvec<int, N_FORWARD> stack;
        stack.push_back(i0);

        mrs.reset();

        for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            if (used[i1]) {
                continue;
            }

            const auto & node1 = nodes[i1];

            if (htp_fusion_can_stack(node0.node, node1.node) && ggml_hexagon_mem_ranges_check_node(mrs, node1)) {
                stack.push_back(i1);
            } else {
                ggml_hexagon_mem_ranges_add_node(mrs, node1);
            }
        }

        for (int idx : stack) {
            res.push_back(idx);
            used[idx] = true;
        }
    }

    return res;
}

static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    const int n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    for (int i = 0; i < n; i++) {
        node_info node = { gf->nodes[i], {} };

        const int f = htp_fusion_max(gf, i);
        for (int k = 1; k < f; k++) {
            node.add_fused(gf->nodes[++i]);
        }

        nodes.push_back(std::move(node));
    }

    const auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

static uint64_t ggml_hexagon_session_key(const ggml_hexagon_session * sess) {
    return ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

static bool ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    if (!sess_src->clone_buffer(sbuf_dst)) { return false; }

    const uint64_t src_key = ggml_hexagon_session_key(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_key];
    if (!fence_slot) {
        fence_slot = (volatile uint32_t *) sess_dst->alloc_fence(1);
    }

    if (!sess_src->clone_buffer(sess_dst->fence_buf)) { return false; }

    uint32_t fence_seq = sess_dst->next_fence_seq();

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    ggml_tensor fence_tensor = sess_dst->make_fence_tensor(fence_slot);

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ true);

    sess_dst->add_peer(sess_src);

    return true;
}

static bool ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    if (!sess_dst->clone_buffer(sbuf_src)) { return false; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    return true;
}

static bool ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        return false;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    if (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<htp_tensor_extra *>(dst->extra);
    const auto * src_extra = static_cast<const htp_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~HTP_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    if (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        return true;
    }

    if (sess_src->phys_idx != sess_dst->phys_idx)
        return ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    return ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

static ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = new ggml_hexagon_event();
    hex_event->fence_sess   = sess;
    hex_event->sess         = sess;
    hex_event->fence_slot   = (volatile uint32_t *) sess->alloc_fence(1);
    hex_event->fence_tensor = sess->make_fence_tensor(hex_event->fence_slot);

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (void *)hex_event, (void *)hex_event->fence_slot);

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

static void ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    if (hex_event->seq == 0) {
        return;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<const volatile std::atomic<uint32_t> *>(hex_event->fence_slot);

    if ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    while (true) {
        if ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            if (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            break;
        }
        std::this_thread::yield();
    }
}

static void ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (void *)hex_event);
    hex_event->fence_sess->free_fence((void *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

static void ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

static void ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    hex_event->sess = sess;
    hex_event->seq  = sess->next_fence_seq();

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ false);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);
}

static void ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    if (hex_event->seq == 0) {
        return;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    if (sess->phys_idx == hex_event->sess->phys_idx) {
        if (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        return;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ true);
}

static void ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (int) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

static void ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (int) tensor->buffer->usage : -1);
    sess->flush_sync();
    if (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

static void ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     struct ggml_tensor * tensor,
                                                     const void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (int) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static void ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     const struct ggml_tensor * tensor,
                                                     void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (int) tensor->buffer->usage : -1);
    sess->flush_sync();
    if (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static struct ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

static ggml_guid_t ggml_backend_hexagon_guid() {
    static ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    return &guid;
}

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

static ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, const char * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    return new ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return dev_ctx->c_name();

    GGML_UNUSED(dev);
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    return "Hexagon";
    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return &dev_ctx->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    if (!opt_hostbuf) {
        return NULL;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return &dev_ctx->host_buffer_type;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    if (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        return false;
    }

    bool supp = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        default:
            supp = htp_kparams_check(sess, op);
            break;
    }

    ggml_hexagon_dump_op_supp(sess->c_name(), op, supp);
    return supp;
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    bool supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    return supp;
}

static const struct ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ NULL,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    for (size_t i = 0; i < opt_ndev; i++) {
        const auto & cfg = opt_device_configs[i];
        if (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } else {
            std::string peers_str;
            for (const auto & p : cfg.mdev_group) {
                if (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = new ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    for (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

static const char * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    return "HTP";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_hexagon_reg_get_device_count(ggml_backend_reg_t reg) {
    return opt_ndev;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    if (index >= opt_ndev || !hreg->devices[index].context) {
        return nullptr;
    }

    return &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

static void * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || n_backends > 4) {
        return nullptr;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_hexagon(backends[i])) {
            return nullptr;
        }
    }

    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        for (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            if (sess_i->phys_idx == sess_j->phys_idx) {
                return nullptr;
            }
        }
    }

    auto * ctx = new ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i]   = (volatile uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = sess_i->make_fence_tensor(ctx->fence_slots[i]);
    }

    return ctx;
}

static void ggml_backend_hexagon_comm_free(void * comm_ctx_v) {
    if (!comm_ctx_v) return;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    for (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((void *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

static bool ggml_backend_hexagon_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (opt_ar_select == 0 || !comm_ctx_v) return false;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->n_backends;

    if (n_backends < 2 || n_backends > 4) return false;

    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        for (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            if (sess_i->phys_idx == sess_j->phys_idx) {
                return false;
            }
        }
    }

    for (size_t i = 0; i < n_backends; i++) {
        if (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            return false;
        }
        if (tensors[i]->type != tensors[0]->type) {
            return false;
        }
        if (!ggml_is_contiguous(tensors[i])) {
            return false;
        }
        if (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            return false;
        }
    }

    if (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }

    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        if (!htp_kparams_check<htp_kparams_solve_allreduce>(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, false, false, false)) {
            return false;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    for (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        if ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    if (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    if (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<const ggml_tensor *> data_tensors(n_backends);
    std::vector<const ggml_tensor *> sync_tensors(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        for (size_t j = 0; j < n_backends; j++) {
            if (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    return true;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(int main_device, const float * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    if (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    if (!dev) return nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return &dev_ctx->buffer_type;
}

static void * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *) ggml_backend_hexagon_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) ggml_backend_hexagon_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) ggml_backend_hexagon_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    return NULL;
}

template<typename T> std::vector<T> str_to_vec(const char* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    while (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    return v;
}

template<typename T, int BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    for (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    return str;
}

static void ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, bool discovery_supported, const std::unordered_map<int, fastrpc_domain> & cdsp_map) {
    if (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        if (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } else {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } else {
        switch (cfg.physical_idx) {
            case 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                break;
            case 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                break;
            default:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                break;
        }
    }
    for (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
static void ggml_hexagon_discover_devices() {
    std::unordered_map<int, fastrpc_domain> cdsp_map;
    bool discovery_supported = false;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    int err = remote_system_request(&domain_info);
    if (err == AEE_SUCCESS && domain_info.sys.num_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.num_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (int) domains.size();

        err = remote_system_request(&domain_info);
        if (err == AEE_SUCCESS) {
            discovery_supported = true;
            const int n_domains = std::min(domain_info.sys.num_domains, (int) domains.size());
            for (int i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (int) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                if (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (int) domains[i].type);
                    continue;
                }
                if (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    continue;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } else {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } else if (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    for (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

static void ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned int) HTP_TYPE_Q4_0 == (unsigned int) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q4_1 == (unsigned int) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q8_0 == (unsigned int) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_MXFP4 == (unsigned int) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_IQ4_NL == (unsigned int) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q4_K == (unsigned int) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q5_K == (unsigned int) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q6_K == (unsigned int) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q3_K == (unsigned int) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q2_K == (unsigned int) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q1_0 == (unsigned int) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q2_0 == (unsigned int) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_BF16 == (unsigned int) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    const char * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    const char * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    const char * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    const char * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    const char * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    const char * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    const char * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    const char * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    const char * str_etm      = getenv("GGML_HEXAGON_ETM");
    const char * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    const char * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    const char * str_vtcm_size = getenv("GGML_HEXAGON_VTCM_SIZE");
    const char * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    const char * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    const char * str_fa_split  = getenv("GGML_HEXAGON_FA_SPLIT");
    const char * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    const char * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    const char * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    const char * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    const char * str_arch     = getenv("GGML_HEXAGON_ARCH");
    const char * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    const char * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    const char * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    const char * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    const char * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    const char * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    if (!str_arch) {
        int err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } else {
            if (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } else if (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } else {
        if (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, NULL, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, NULL, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, NULL, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, NULL, 0)          : (opt_opbatch * HTP_TRACE_MAX_EVENTS + 131072);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  NULL, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    if (str_power_mode) {
        int m = htp_power_mode_from_str(str_power_mode);
        if (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } else {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, NULL, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_vtcm_size = str_vtcm_size ? parse_size(str_vtcm_size)                 : opt_vtcm_size;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_split  = str_fa_split  ? atoi(str_fa_split)                   : opt_fa_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    if (str_mbuf) {
        const char * p = str_mbuf;
        for (int idx = 0; idx < 3 && p && *p; idx++) {
            while (*p == ' ') p++;
            const char * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            while (len > 0 && p[len - 1] == ' ') len--;
            if (len > 0) {
                std::string token(p, len);
                if (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                if (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                if (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            if (!comma) break;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    const char * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    if (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    if (str_devices && str_devices[0] != '\0') {
        bool is_single_number = true;
        for (int i = 0; str_devices[i] != '\0'; i++) {
            if (!isdigit((unsigned char)str_devices[i])) {
                is_single_number = false;
                break;
            }
        }
        if (is_single_number) {
            int n = atoi(str_devices);
            if (n < 1) n = 1;
            if (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            for (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (int)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } else {
            std::string s_devices(str_devices);
            std::vector<std::string> items;
            std::string curr_item;
            int bracket_depth = 0;
            for (char ch : s_devices) {
                if (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } else if (ch == ']') {
                    if (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } else if (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    if (s != std::string::npos) {
                        items.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } else {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            if (s != std::string::npos) {
                items.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            for (const auto & item : items) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                if (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    int virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    if (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        try {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } catch (...) {
                            virt = 0;
                        }
                    } else {
                        size_t dev_colon = dev_name.find(':');
                        if (dev_colon != std::string::npos) {
                            try {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } catch (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<int> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    while (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        if (ps == std::string::npos) continue;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        if (dash_pos != std::string::npos) {
                            try {
                                int p_start = std::stoi(p_part.substr(0, dash_pos));
                                int p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                for (int p = p_start; p <= p_end; p++) {
                                    if (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } catch (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } else {
                            try {
                                int p = std::stoi(p_part);
                                if (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } catch (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    if (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    if (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        for (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } else {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } else if (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    int phys = 0;
                    int virt = 0;
                    try {
                        if (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } else {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } catch (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        continue;
                    }

                    if (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } else {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } else {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } else {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#if defined(__ANDROID__)
    if (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    if (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            switch (v.size()) {
                case 1:  opt_profile = v[0]; return opt_pmu_evt; // mode with default pmu events
                case 8:  opt_profile = 2;    return v;           // mode with custom  pmu events
                default: opt_profile = 0;    return {};          // garbage input
            }}();
        if (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = new ggml_hexagon_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            auto nErr = htpdrv_init();
            if (nErr != AEE_SUCCESS) {
                return NULL;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
