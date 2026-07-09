#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
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
#include <algorithm>

#ifdef _WIN32
#    include <sal.h>
#else
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "htp-opnode.h"
#include "htp-ops.h"
#include "htp/matmul-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp_iface.h"
#include "htp-drv.h"

using intvec  = std::vector<int>;
using uintvec = std::vector<unsigned int>;
using u32vec  = std::vector<uint32_t>;

static int    opt_arch    = 0; // autodetect
static size_t opt_ndev    = 1;
static size_t opt_nhvx    = 0; // use all
static int    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
static size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
static size_t opt_mbuf    = 1ul * 1024 * 1024 * 1024; // max buffer size
static int    opt_etm     = 0;
static int    opt_verbose = 0;
static int    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
static int    opt_hostbuf = 1; // hostbuf ON by default

static int    opt_mm_select = 3; // 3 = HMX -> Tiled -> Flat -> CPU, 2 = Tiled -> Flat -> CPU, 1 = Flat -> CPU
static int    opt_fa_select = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

// Enable all stages by default
static int opt_opstage  = HTP_OPSTAGE_QUEUE | HTP_OPSTAGE_COMPUTE;
static int opt_opbatch  = 1024; // max number of ops in a batch
static int opt_opqueue  = 16;   // max number of pending batches
static int opt_optrace  = 0;    // trace buffer size per thread (0 means default)
static int opt_oppoll   = 0;    // polling for batch completions
static int opt_opfusion = 1;    // enable/disable op fusion

static std::regex* opt_opfilter = NULL; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

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

static void ggml_hexagon_dump_op_exec(const std::string &sess_name, const htp_opnode & node, const uint32_t req_flags) {
    if (!opt_verbose) return;

    htp_opformat fmt(node);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

static void ggml_hexagon_dump_op_supp(const std::string &sess_name, const struct ggml_tensor * op, bool supp) {
    if (!opt_verbose) return;

    htp_opformat fmt(htp_opformat(htp_opnode{const_cast<ggml_tensor*>(op), {}, HTP_OP_INVALID}));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
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
        case HTP_TRACE_EVT_HMX_COMP:       return "HMX_COMP";
        default:                           return "UNKNOWN";
    }
}

static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const htp_opnode & node,
                                      const htp_prof_desc & pd) {
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

    htp_opformat fmt(node);
    float mhz = op_usec > 0 ? (float) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

// **

static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4;
}

static inline bool ggml_hexagon_is_hmx_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || ggml_hexagon_is_repack_type(type);
}

struct htp_mm_kernel_params;
struct ggml_hexagon_session;
static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_qkv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_ffn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
);

// ** backend sessions

struct ggml_hexagon_opbatch;
struct ggml_hexagon_opqueue;
struct htp_opnode;

struct ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    int              dev_id;
    bool             valid_session;
    bool             valid_handle;
    bool             valid_queue;
    bool             valid_iface;

    std::atomic<int>      op_pending;
    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    ggml_backend_buffer_type buffer_type        = {};
    ggml_backend_buffer_type repack_buffer_type = {};

    uint32_t n_threads = 0;
    uint32_t n_hvx     = 0;
    uint32_t n_hmx     = 0;
    uint64_t vtcm_size = 0;
    size_t   max_vmem  = 0;
    size_t   max_bufsize = 0;

    struct {
        uint64_t uid = 0;
        std::vector<htp_opnode> htp_nodes;
    } cached_graph;

    ggml_hexagon_session(int dev_id, ggml_backend_dev_t dev) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    const char* c_name() const { return name.c_str(); }

    void allocate(int dev_id) noexcept(false);
    void release() noexcept(true);

    void enqueue_op(const htp_opnode & node);
    void flush(bool all = true);

    void flush_pending(bool all = false);
    void flush_batch();
};

// ** backend buffers

struct ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(const std::string & name, ggml_hexagon_session * sess) {
        this->sess = sess;
        this->name = name;
    }

    ggml_hexagon_session * sess;
    std::string            name;
};

struct ggml_hexagon_shared_buffer {
    ggml_hexagon_session * sess;
    uint8_t *              base;
    size_t                 size;
    int                    fd;
    bool                   mapped;
    bool                   pinned;

    void mmap() {
        fastrpc_map_flags flags = this->pinned ? FASTRPC_MAP_FD : FASTRPC_MAP_FD_DELAYED;

        int err = fastrpc_mmap(sess->domain_id, this->fd, (void *) this->base, 0, this->size, flags);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, this->size, this->fd, (unsigned) err);
            throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u\n",
                sess->c_name(), (void *) this->base, this->size, this->fd, pinned);

        this->mapped = true;
    }

    void unmap() {
        if (!this->mapped) return;

        if (!this->pinned) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, this->fd);
        }

        fastrpc_munmap(sess->domain_id, this->fd, (void *) this->base, this->size);

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) this->base, size, this->fd);

        this->mapped = false;
        this->fd     = -1;
    }

    void alloc(size_t size) {
        if (this->base) return;

        this->base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        if (!this->base) {
            GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer : size %zu\n", sess->c_name(), size);
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed (see log for details)");
        }

        this->fd = rpcmem_to_fd(this->base);
        if (this->fd < 0) {
            GGML_LOG_ERROR("ggml-hex: %s failed to get FD for buffer %p\n", sess->c_name(), (void *) this->base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed (see log for details)");
        }
        this->size = size;

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (void *) this->base, this->size, this->fd, (int) pinned);
        mmap();
    }

    void free() {
        if (!this->base) return;

        unmap();
        rpcmem_free(this->base);

        HEX_VERBOSE("ggml-hex: %s freed buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) this->base, size, this->fd);

        this->base = NULL;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, bool pinned = false) {
        this->sess   = sess;
        this->size   = 0;
        this->base   = nullptr;
        this->fd     = -1;
        this->mapped = false;
        this->pinned = pinned;

        alloc(size);
    }

    ~ggml_hexagon_shared_buffer() {
        free();
    }
};

static ggml_hexagon_session * ggml_backend_hexagon_buffer_get_sess(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer->buft->context)->sess;
}

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    delete sbuf;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    return sbuf->base;
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu usage %d\n", sess->c_name(),
                tensor->name, (void *) sbuf->base, tensor->data, ggml_nbytes(tensor), (int) buffer->usage);

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        return GGML_STATUS_SUCCESS; // nothing to do for the view
    }

    return GGML_STATUS_SUCCESS;
}

// ** Repack helpers for tiled quantized weights

static void unpack_q4_0_quants(uint8_t * qs, const block_q4_0 * x, unsigned int bi) {
    static const int qk = QK4_0;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_q4_0_quants(block_q4_0 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_0;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

static void unpack_q4_1_quants(uint8_t * qs, const block_q4_1 * x, unsigned int bi) {
    static const int qk = QK4_1;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_q4_1_quants(block_q4_1 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_1;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

static void unpack_mxfp4_quants(uint8_t * qs, const block_mxfp4 * x, unsigned int bi) {
    static const int qk = QK_MXFP4;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_mxfp4_quants(block_mxfp4 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK_MXFP4;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

// repack q4_0 data into q4_0_tiled tensor
static void repack_q4_0_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_q4_0 * src_matrix = (const block_q4_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_0_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt], 0);
                        } else {
                            memset(tile_quants[row], 8, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }
}

// repack q4_0_tiled tensor into q4_0 data
static void repack_tiled_q4_0(void * data, const ggml_tensor * t, size_t size) {
    block_q4_0 * dst_matrix = (block_q4_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_0 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_0_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row], 0);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[row];
                        }
                    }
                }
            }
        }
    }
}

// repack q4_1 data into q4_1_tiled tensor
static void repack_q4_1_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_q4_1 * src_matrix = (const block_q4_1 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_1 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_1_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt], 0);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            scale_dst[2 * row + 0] = src_expert[r * (ne0 / 32) + kt].d;
                            scale_dst[2 * row + 1] = src_expert[r * (ne0 / 32) + kt].m;
                        } else {
                            scale_dst[2 * row + 0] = 0;
                            scale_dst[2 * row + 1] = 0;
                        }
                    }
                }
            }
        }
    }
}

// repack q4_1_tiled tensor into q4_1 data
static void repack_tiled_q4_1(void * data, const ggml_tensor * t, size_t size) {
    block_q4_1 * dst_matrix = (block_q4_1 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_1 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_1_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row], 0);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[2 * row];
                            dst_expert[r * (ne0 / 32) + kt].m = scale_src[2 * row + 1];
                        }
                    }
                }
            }
        }
    }
}

// repack q8_0 data into q8_0_tiled tensor
static void repack_q8_0_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_q8_0 * src_matrix = (const block_q8_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q8_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        int col0 = cp * 2;
                        int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            int64_t r = ct * 32 + row;
                            const block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_expert[r * (ne0 / 32) + kt] : NULL;
                            tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                            tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 1024);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }
}

// repack q8_0_tiled tensor into q8_0 data
static void repack_tiled_q8_0(void * data, const ggml_tensor * t, size_t size) {
    block_q8_0 * dst_matrix = (block_q8_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q8_0 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        int col0 = cp * 2;
                        int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            int64_t r = ct * 32 + row;
                            if (r < ne1 && kt < ne0 / 32) {
                                block_q8_0 & b = dst_expert[r * (ne0 / 32) + kt];
                                b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                                b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                            }
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 1024);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[row];
                        }
                    }
                }
            }
        }
    }
}

// repack mxfp4 data into mxfp4_tiled tensor
static void repack_mxfp4_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_mxfp4 * src_matrix = (const block_mxfp4 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_mxfp4 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_mxfp4_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt], 0);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    uint8_t * scale_dst = tile_dst + 512;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].e : 0;
                    }
                }
            }
        }
    }
}

// repack mxfp4_tiled tensor into mxfp4 data
static void repack_tiled_mxfp4(void * data, const ggml_tensor * t, size_t size) {
    block_mxfp4 * dst_matrix = (block_mxfp4 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_mxfp4 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_mxfp4_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row], 0);
                        }
                    }

                    const uint8_t * scale_src = tile_src + 512;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].e = scale_src[row];
                        }
                    }
                }
            }
        }
    }
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu\n", sess->c_name(), tensor->name, data, offset, size);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q4_0_tiled(tensor, data, size);
            break;

        case GGML_TYPE_Q4_1:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q4_1_tiled(tensor, data, size);
            break;

        case GGML_TYPE_Q8_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q8_0_tiled(tensor, data, size);
            break;

        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            // IQ4_NL has identical block layout to Q4_0 (ggml_half d + uint8_t qs[16])
            repack_q4_0_tiled(tensor, data, size);
            break;

        case GGML_TYPE_MXFP4:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_mxfp4_tiled(tensor, data, size);
            break;

        default:
            memcpy((char *) tensor->data + offset, data, size);
            break;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   tensor,
                                                   void *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu\n", sess->c_name(), tensor->name, data, offset, size);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, size);
            break;

        case GGML_TYPE_Q4_1:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_1(data, tensor, size);
            break;

        case GGML_TYPE_Q8_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q8_0(data, tensor, size);
            break;

        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, size);
            break;

        case GGML_TYPE_MXFP4:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_mxfp4(data, tensor, size);
            break;

        default:
            memcpy(data, (const char *) tensor->data + offset, size);
            break;
    }
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   const struct ggml_tensor * src,
                                                   struct ggml_tensor *       dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    return false;
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (void *) sbuf->base, sbuf->size);
    memset(sbuf->base, value, sbuf->size);
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
};

// ** backend buffer type

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    return static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
    try {
        size += 4 * 1024;  // guard page
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer context (host): %s\n", sess->c_name(), exc.what());
        return nullptr;
    }
}

static ggml_backend_buffer_t ggml_backend_hexagon_repack_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
    try {
        size += 4 * 1024;  // guard page
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer context (repack): %s\n", sess->c_name(), exc.what());
        return nullptr;
    }
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buffer_type) {
    return 128;  // HVX alignment
    GGML_UNUSED(buffer_type);
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_1 || t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_IQ4_NL || t->type == GGML_TYPE_MXFP4) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        return ggml_row_size(t->type, ne0) * ne1 * ne2 * ne3;
    }
    return ggml_nbytes(t);
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
    auto * context = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context);
    return context->sess->max_bufsize;
}

static bool ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return opt_hostbuf;
    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_repack_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_hexagon_buffer_type_is_host,
};

static ggml_backend_buffer_type_i ggml_backend_hexagon_repack_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_hexagon_repack_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_hexagon_repack_buffer_type_is_host,
};

static bool ggml_backend_buffer_is_hexagon(const struct ggml_backend_buffer * b) {
    return b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

static inline bool ggml_backend_buffer_is_hexagon_repack(const struct ggml_backend_buffer * b) {
    if (!opt_hostbuf) {
        return ggml_backend_buffer_is_hexagon(b);
    }
    return b->buft->iface.alloc_buffer == ggml_backend_hexagon_repack_buffer_type_alloc_buffer;
}

struct ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<int, int>                b_map; // buffer fd   to index
    std::unordered_map<const ggml_tensor*, int> t_map; // tensor ptr  to index
    std::unordered_multimap<void*, int>         d_map; // tensor data to index

    unsigned int n_bufs;     // num buffers in the batch
    unsigned int n_tens;     // num tensors ...
    unsigned int n_ops;      // num ops ...
    size_t       b_vmem;     // sum of all buffer sizes

    unsigned int n_bufs_max;
    unsigned int n_tens_max;
    unsigned int n_ops_max;
    size_t       b_vmem_max;

    void reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size, size_t max_vmem) {
        this->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_size;
        n_tens_max = n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS;

        b_vmem_max = max_vmem;

        ops.resize(n_ops_max);

        h_bufs.resize(n_bufs_max);
        h_tens.resize(n_tens_max);
        h_ops.resize(n_ops_max);

        b_map.reserve(n_bufs_max);
        t_map.reserve(n_tens_max);
        d_map.reserve(n_tens_max);

        GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
                sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);

        reset();
    }

    bool empty() const { return n_ops == 0; }

    // add buffer and return its index
    int add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd);
        if (it != b_map.end()) { return it->second; }

        // Add new buffer to the batch
        int bi = n_bufs++;
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);

        b_map.insert({sbuf->fd, bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base = (uint64_t) sbuf->base;
        b.fd   = sbuf->fd;
        b.size = sbuf->size;

        b_vmem += b.size;

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (void*) sbuf->base, (size_t) b.size, b_vmem);

        return bi;
    }

    bool same_shape(const htp_tensor * h, const ggml_tensor * t) const {
        int64_t ne0 = t->ne[0];
        int64_t ne1 = t->ne[1];
        const bool is_repack = ggml_backend_buffer_is_hexagon_repack(t->buffer) && ggml_hexagon_is_repack_type(t->type);
        if (is_repack) {
            ne0 = hex_round_up(ne0, 32);
            ne1 = hex_round_up(ne1, 32);
        }
        int64_t nb1 = is_repack ? ggml_row_size(t->type, ne0) : t->nb[1];
        int64_t nb2 = is_repack ? nb1 * ne1 : t->nb[2];
        int64_t nb3 = is_repack ? nb2 * t->ne[2] : t->nb[3];

        return (h->ne[0] == ne0) && (h->ne[1] == ne1) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == t->nb[0]) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    int add_tensor(const ggml_tensor * t) {
        auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        for (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            if (same_shape(h, t)) { return it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        if (it != t_map.end()) { return it->second; }

        // Add new tensor to the batch
        int ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->data, ti});

        uint64_t t_offset = (uint8_t *) t->data - sbuf->base;
        size_t   t_size   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.data  = t_offset;
        h.type  = t->type;

        const bool is_repack = ggml_backend_buffer_is_hexagon_repack(t->buffer) && ggml_hexagon_is_repack_type(t->type);
        if (is_repack) {
            h.ne[0] = hex_round_up(t->ne[0], 32);
            h.ne[1] = hex_round_up(t->ne[1], 32);
            h.ne[2] = t->ne[2];
            h.ne[3] = t->ne[3];

            h.nb[0] = t->nb[0];
            h.nb[1] = ggml_row_size(t->type, h.ne[0]);
            h.nb[2] = h.nb[1] * h.ne[1];
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } else {
            h.size  = t_size;
            h.ne[0] = t->ne[0]; h.ne[1] = t->ne[1]; h.ne[2] = t->ne[2]; h.ne[3] = t->ne[3];
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            h.flags |= HTP_TENSOR_COMPUTE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (void*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        return ti;
    }

    bool fit_op(const htp_opnode & node) const {
        if (n_ops >= n_ops_max ) return false;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        auto fit_tensor = [&](const ggml_tensor *t) {
            if (!t) return;
            if (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                if (!b_map.count(sbuf->fd)) {
                    extra_vmem += sbuf->size;
                    extra_bufs += 1;
                }
            }
        };

        for (const auto * src : node.get_inputs()) {
            fit_tensor(src);
        }
        for (const auto * output : node.get_outputs()) {
            fit_tensor(output);
        }

        if ((extra_bufs + n_bufs) > n_bufs_max) return false;
        if ((extra_tens + n_tens) > n_tens_max) return false;
        if ((extra_vmem + b_vmem) > b_vmem_max) return false;

        return true;
    }

    // assumes that fit_op() was called first and returned true
    void add_op(const htp_opnode & node) {
        // Add new op

        unsigned int n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = node;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        node.node->op_params, sizeof(node.node->op_params));
        memcpy(o.kernel_params, node.kernel_params,   sizeof(o.kernel_params));
        o.opcode = node.opcode;
        o.flags  = 0;

        if (!(opt_opstage & HTP_OPSTAGE_COMPUTE)) {
            o.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
        }

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = node.get_inputs();
        for (unsigned int i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = node.get_outputs();
        for (unsigned int i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }
};

struct ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_buffer *shm_buf;
    size_t                      shm_blk_size;

    using opvec = std::vector<htp_opnode>;

    std::queue<unsigned int>    done;           // completed batch ids
    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       start_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_size, size_t depth) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_size;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

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
        start_usec.resize(depth, 0);

        // init done queue
        for (unsigned int i = 0; i < depth; i++) { done.push(i); }

        if (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated op-queue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size, shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    // push new batch
    bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, ggml_hexagon_opbatch* op_batch) {
        static_assert(sizeof(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(sizeof(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(sizeof(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(sizeof(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(sizeof(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(sizeof(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        if (done.empty()) { return false; }

        req.id        = done.front(); done.pop(); // batch id
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[req.id]   = op_batch->ops;
        start_usec[req.id] = ggml_time_us();

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

        dbuf.ptr      = shm_buf->base + (req.id * shm_blk_size);
        dbuf.fd       = shm_buf->fd;
        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base;
        dbuf.size     = b_size + t_size + o_size + p_size + tr_size;

        GGML_ASSERT(dbuf.size <= shm_blk_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
        uint8_t * o_ptr = m_ptr;

        memcpy(b_ptr, (void *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (void *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (void *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s op-queue push batch #%u : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), req.id, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        op_batch->reset();

        if (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            for (unsigned int i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (void *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            for (unsigned int i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %u size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        return true;
    }

    void pop(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        GGML_ASSERT(rsp.id < op_cache.size());

        done.push(rsp.id);

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

        HEX_VERBOSE("ggml-hex: %s op-queue pop batch #%u : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), rsp.id, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        if (opt_profile && rsp.n_ops > 0) {
            auto & ops = op_cache[rsp.id];

            uint64_t batch_usec = ggml_time_us() - start_usec[rsp.id];
            uint32_t htp_usec   = 0;

            GGML_ASSERT(rsp.n_ops <= ops.size());

            const htp_prof_desc * pd = (const htp_prof_desc *) p_ptr;

            const htp_trace_desc * trace_events = nullptr;

            if (opt_profile == 3) {
                trace_events = (const htp_trace_desc *) (p_ptr + p_size);
            }

            uint32_t trace_idx[HTP_MAX_NTHREADS + 1] = {0};
            uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};

            if (opt_profile == 3) {
                for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
                    uint32_t count = rsp.n_traces[t];
                    valid_cnt[t] = count > n_traces ? n_traces : count;
                }
            }

            for (uint32_t i = 0; i < rsp.n_ops; i++) {
                htp_usec += pd[i].usecs;

                ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);

                if (opt_profile == 3) {
                    uint32_t op_duration = pd[i].cycles_stop - pd[i].cycles_start;

                    for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
                        while (trace_idx[t] < valid_cnt[t]) {
                            const auto & e = trace_events[t * n_traces + trace_idx[t]];
                            uint32_t offset = e.cycles - pd[i].cycles_start;
                            if (offset >= 0x80000000) {
                                trace_idx[t]++;
                                continue;
                            }
                            if (offset > op_duration) {
                                break;
                            }
                            bool is_stop = (e.info & 0x8000) != 0;
                            uint16_t info = e.info & 0x7FFF;
                            GGML_LOG_DEBUG("ggml-hex: %s trace-op %s: thread %u event %s info %u %s %u\n",
                                           shm_buf->sess->c_name(), ops[i].op_name().c_str(), t, htp_event_name(e.id), info, is_stop ? "stop" : "start", e.cycles);
                            trace_idx[t]++;
                        }
                    }
                }
            }

            char evt_str[256] = "";
            if (opt_profile == 3) {
                snprintf(evt_str, sizeof(evt_str), " evt [%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u]",
                        rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                        rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                        rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
            }

            GGML_LOG_DEBUG("ggml-hex: %s profile-batch n-ops %u batch-dur-usec %lld htp-ops-usec %u%s\n",
                           shm_buf->sess->c_name(), rsp.n_ops, (long long) batch_usec, htp_usec, evt_str);
        }
    }
};

// Flush HTP response queue i.e wait for all outstanding requests to complete
void ggml_hexagon_session::flush_pending(bool all) {
    while (this->op_pending) {
        struct htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        struct dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        const uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        if (err == AEE_EEXPIRED) {
            continue;
        }

        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        if (rsp_size != sizeof(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", this->c_name(), rsp_size, n_dbufs);
        }

        if (rsp.status != HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp: %s\n", this->c_name(), status_to_str(rsp.status));
            // TODO: handle errors
        }

        op_queue->pop(rsp, dbuf);

        this->op_pending--;  // atomic dec

        if (!all) break;
    }
}

void ggml_hexagon_session::flush_batch() {
    if (op_batch->empty()) { return; }

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    if (!op_queue->push(req, dbuf, op_batch)) {
        flush_pending(false);
        op_queue->push(req, dbuf, op_batch);
    }

    // Bump pending flag (cleared in the session::flush once we get the response)
    this->op_pending++;  // atomic inc

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);

    int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
    if (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
    }
}

void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    if (!op_batch->fit_op(node)) {
        flush_batch();
    }
    op_batch->add_op(node);
}

// Flush HTP response queue i.e wait for all outstanding requests to complete
void ggml_hexagon_session::flush(bool all) {
    flush_batch();
    flush_pending(all);
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

void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
    this->valid_session = false;
    this->valid_handle  = false;
    this->valid_queue   = false;
    this->valid_iface   = false;

    this->domain_id  = 3;  // Default for CDSP, updated after the session is created
    this->session_id = 0;  // Default for CDSP, updated after the session is created
    this->dev_id     = dev_id;
    this->name       = std::string("HTP") + std::to_string(dev_id);

    this->op_pending  = 0;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session\n", this->name.c_str());

    domain * my_domain = get_domain(this->domain_id);
    if (my_domain == NULL) {
        GGML_LOG_ERROR("ggml-hex: unable to get domain struct for CDSP\n");
        throw std::runtime_error("ggml-hex: failed to get CDSP domain (see log for details)");
    }

    // Create new session
    if (dev_id != 0) {
        struct remote_rpc_reserve_new_session n;
        n.domain_name_len  = strlen(CDSP_DOMAIN_NAME);
        n.domain_name      = const_cast<char *>(CDSP_DOMAIN_NAME);
        n.session_name     = const_cast<char *>(this->name.c_str());
        n.session_name_len = this->name.size();

        int err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (void *) &n, sizeof(n));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: failed to reserve new session %d : error 0x%x\n", dev_id, err);
            throw std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        this->session_id    = n.session_id;
        this->domain_id     = n.effective_domain_id;
        this->valid_session = true;
    }

    // Get session URI

    char session_uri[256];
    {
        char htp_uri[256];
        snprintf(htp_uri, sizeof(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        struct remote_rpc_get_uri u = {};
        u.session_id      = this->session_id;
        u.domain_name     = const_cast<char *>(CDSP_DOMAIN_NAME);
        u.domain_name_len = strlen(CDSP_DOMAIN_NAME);
        u.module_uri      = const_cast<char *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);

        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            // fallback to single session uris
            int htp_URI_domain_len = strlen(htp_uri) + MAX_DOMAIN_NAMELEN;

            snprintf(session_uri, htp_URI_domain_len, "%s%s", htp_uri, my_domain->uri);

            GGML_LOG_WARN("ggml-hex: failed to get URI for session %d : error 0x%x. Falling back to single session URI: %s\n", dev_id, err, session_uri);
        }
    }

    // Enable Unsigned PD
    {
        struct remote_rpc_control_unsigned_module u;
        u.domain = this->domain_id;
        u.enable = 1;
        int err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: failed to enable unsigned PD for session %d : error 0x%x\n", dev_id, err);
            throw std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    // Open session
    int err = htp_iface_open(session_uri, &this->handle);
    if (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: failed to open session %d : error 0x%x\n", dev_id, err);
        throw std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    this->valid_handle = true;

    // Query HW info and resolve session options
    this->max_bufsize = opt_mbuf;
    {
        unsigned int hw_n_threads = 0;
        unsigned int hw_n_hvx     = 0;
        unsigned int hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        int hw_err = htp_iface_hwinfo(this->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        if (hw_err == 0) {
            this->n_threads = opt_nhvx > 0 ? (uint32_t)opt_nhvx : (uint32_t)hw_n_threads;
            this->n_hvx     = opt_nhvx > 0 ? (uint32_t)opt_nhvx : (uint32_t)hw_n_hvx;
            this->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            this->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          this->c_name(), this->n_threads, this->n_hvx, this->n_hmx,
                          (unsigned long long)(this->vtcm_size / (1024 * 1024)));
        } else {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", this->c_name(), hw_err);
            this->n_threads = opt_nhvx > 0 ? (uint32_t)opt_nhvx : 8;
            this->n_hvx     = opt_nhvx > 0 ? (uint32_t)opt_nhvx : 8;
            this->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            this->vtcm_size = 8 * 1024 * 1024;
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

    if (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(this->handle, opt_profile, &pmu_conf);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    this->op_queue = new ggml_hexagon_opqueue(this, opt_opbatch, opt_opqueue);

    if (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(this);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", this->c_name(), opt_vmem);
    }
    this->max_vmem = opt_vmem;

    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch, this->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(this->handle, dev_id, this->queue_id, opt_nhvx, opt_nhmx, this->max_vmem);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", this->c_name(), (unsigned) err);
        throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    this->valid_iface = true;
}

void ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", this->name.c_str());

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

ggml_hexagon_session::ggml_hexagon_session(int dev_id, ggml_backend_dev_t dev) noexcept(false) {
    buffer_type.device        = dev;
    repack_buffer_type.device = dev;

    op_batch = nullptr;
    op_queue = nullptr;

    try {
        allocate(dev_id);

        buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
        buffer_type.context = new ggml_backend_hexagon_buffer_type_context(this->name, this);

        repack_buffer_type.iface   = ggml_backend_hexagon_repack_buffer_type_interface;
        repack_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(this->name + "-REPACK", this);
    } catch (const std::exception & exc) {
        release();
        throw;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
    release();

    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(repack_buffer_type.context);
}

// ** backend interface

static bool ggml_hexagon_flash_attn_is_hmx_eligible(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    const struct ggml_tensor * sinks
) {
    if (sess->n_hmx == 0) {
        return false;
    }

    if (opt_fa_select < 2) {
        return false;
    }

    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }

    const uint32_t DK = q->ne[0];
    const uint32_t DV = v->ne[0];

    if (DK % 64 != 0 || DV % 64 != 0) {
        return false;
    }

    // Fall back to HVX for small token counts if head dimension is small (DK <= 128)
    const uint32_t neq1 = q->ne[1];
    if (DK <= 128 && neq1 < 5) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_precompute_flash_attn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fa_kernel_params * kparams
) {
    if (opt_fa_select < 1) {
        return false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * q    = op->src[0];
    const struct ggml_tensor * k    = op->src[1];
    const struct ggml_tensor * v    = op->src[2];
    const struct ggml_tensor * mask = op->src[3];
    const struct ggml_tensor * dst  = op;

    const uint32_t neq0 = q->ne[0];  // head_dim (DK)
    const uint32_t neq1 = q->ne[1];  // n_tokens
    const uint32_t neq2 = q->ne[2];  // n_heads

    const uint32_t nek1 = k->ne[1];  // kv_len

    const uint32_t nev0 = v->ne[0];  // head_dim (DV)

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G          = neq2 / n_kv_heads;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], sizeof(float));
    memcpy(&max_bias,      &op->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &op->op_params[2], sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->is_q_fp32 = (q->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->is_dst_fp32 = (dst->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->G = G;

    const uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    const struct ggml_tensor * sinks = op->src[4];
    if (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        size_t Br = 0, Bc = 0;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, nek1, sess->vtcm_size, sess->n_threads);
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK, DV, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0);

            const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            const size_t m_line_bytes = hex_align_up(Bc * sizeof(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_line_bytes / sizeof(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            return true;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    const size_t size_q_row_padded = hex_round_up(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    const size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, kparams->is_q_fp32 != 0, mask != nullptr, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_values(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    return true;
}

static bool ggml_hexagon_supported_flash_attn_ext(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * src3 = op->src[3];
    const struct ggml_tensor * src4 = op->src[4];
    const struct ggml_tensor * dst  = op;

    // Check for F16 support only as requested
    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) || src1->type != GGML_TYPE_F16 || src2->type != GGML_TYPE_F16) {
        return false;
    }

    if (src3 && src3->type != GGML_TYPE_F16) {  // mask
        return false;
    }

    if (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        return false;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    if (dst->ne[3] != 1) {
        return false;
    }

    struct htp_fa_kernel_params kparams;
    if (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        return false;
    }

    if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_gated_delta_net(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * beta  = op->src[4];
    const struct ggml_tensor * state = op->src[5];
    const struct ggml_tensor * dst   = op;

    if (!q || !k || !v || !g || !beta || !state) {
        return false;
    }

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    if (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        return false;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return false;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return false;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    if (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        return false;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_hexagon_matmul_is_hmx_eligible(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ne01_padded,
    bool is_matmul_id,
    bool is_batched
) {
    const int ne00  = src0->ne[0];
    const int ne11  = src1->ne[1];
    const int ne12  = src1->ne[2];
    const int wtype = src0->type;

    // HMX weight tile requires N to be 32-aligned.
    if (ne01_padded % 32 != 0) {
        return false;
    }

    // HMX supports F16, F32, and repack quantized types.
    if (!ggml_hexagon_is_hmx_weight_type((ggml_type) wtype)) {
        return false;
    }

    // HMX paths require K aligned to 32.
    if (ne00 % 32 != 0) {
        return false;
    }

    // Quantized HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    if (!is_matmul_id && is_batched && wtype != GGML_TYPE_F16) {
        return false;
    }

    // HMX assumes contiguous row-major layout.
    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        return false;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS
    const int m = is_matmul_id ? ne12 : ne11;
    if (m <= HTP_MM_HMX_MIN_NROWS) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_precompute_hmx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne00_padded,
    int ne01_padded,
    int ne02,
    int ne11,
    int ne12,
    int ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    const int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
    const bool pipeline = is_matmul_id ? false : htp_mm_hmx_pipeline(ne11);
    const int n_threads = (int)sess->n_threads;
    const int ne10 = src1->ne[0];

    const bool is_batched_val = is_matmul_id ? false : is_batched;
    const int group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    bool use_grouped = false;
    int act_threads_selected = 0;

    if (is_batched_val && wtype == GGML_TYPE_F16 && group_size > 1) {
        // Try grouped path first
        const bool use_dma_activation = (src1->nb[1]/sizeof(float) > (size_t)ne00_padded);
        size_t best_mblocks = SIZE_MAX;
        int best_act_threads = 0;
        size_t best_m_chunk = 0;
        size_t best_n_chunk = 0;
        size_t best_vtcm_size = 0;

        int act_threads = n_threads;
        while (act_threads >= 1) {
            const size_t f32_scratch_size = use_dma_activation ? hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * ne00_padded * sizeof(float), HTP_MM_HMX_TILE_SIZE) : 0;
            size_t group_overhead = 256 + f32_scratch_size;
            size_t group_size_per_n, group_size_per_m, group_size_per_mn;
            htp_mm_hmx_get_batched_chunk_costs(ne00_padded, group_size, &group_size_per_n, &group_size_per_m, &group_size_per_mn);

            size_t m_chunk_candidate = 0;
            size_t n_chunk_candidate = 0;
            size_t vtcm_size_candidate = 0;

            if (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead, group_size_per_n, group_size_per_m, group_size_per_mn, hex_align_up(ne11, 32), ne01_padded,
                                   (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                   &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
                size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(wtype, ne00_padded, m_chunk_candidate, n_chunk_candidate, group_size, use_dma_activation, pipeline, act_threads);
                if (exact_size <= vtcm_budget) {
                    size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                    if (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                        best_mblocks = mblocks;
                        best_act_threads = act_threads;
                        best_m_chunk = m_chunk_candidate;
                        best_n_chunk = n_chunk_candidate;
                        best_vtcm_size = exact_size;
                    }
                }
            }
            if (act_threads == 1) {
                act_threads = 0;
            } else {
                act_threads /= 2;
            }
        }

        if (best_act_threads > 0) {
            m_chunk = best_m_chunk;
            n_chunk = best_n_chunk;
            vtcm_size = best_vtcm_size;
            act_threads_selected = best_act_threads;
            use_grouped = true;
        }
    }

    if (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        size_t best_mblocks = SIZE_MAX;
        int best_act_threads = 0;
        size_t best_m_chunk = 0;
        size_t best_n_chunk = 0;
        size_t best_vtcm_size = 0;

        // For MUL_MAT_ID the kernel runs one 2D matmul per expert, with M equal to the number of rows routed to that expert.
        // A single expert can receive up to all routed rows (dst->ne[1]*dst->ne[2] = n_expert_used*n_tokens), so size the chunk
        // search for that upper bound rather than ne12 (token positions only).
        // We recompute m_chunk per expert against the actual count in the NPU kernel.
        const int m_id_rows    = (int) ((size_t) dst->ne[1] * dst->ne[2]);
        const int m_for_chunks = is_matmul_id ? hex_align_up(m_id_rows, 32) : ne11_padded;
        const int m_for_cost   = is_matmul_id ? m_id_rows : ne11;

        int act_threads = n_threads;
        while (act_threads >= 1) {
            const size_t act_f32_size = is_matmul_id ? 0 : hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * ne00_padded * sizeof(float), HTP_MM_HMX_TILE_SIZE);
            size_t simple_2d_overhead = 256 + act_f32_size;
            size_t simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn;
            htp_mm_hmx_get_2d_chunk_costs(wtype, ne00_padded, pipeline, aligned_tile_size, &simple_2d_size_per_n, &simple_2d_size_per_m, &simple_2d_size_per_mn);

            size_t m_chunk_candidate = 0;
            size_t n_chunk_candidate = 0;
            size_t vtcm_size_candidate = 0;

            if (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn, m_for_chunks, ne01_padded,
                                   (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                                   &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
                size_t exact_size = htp_mm_hmx_get_2d_vtcm_size(wtype, ne00_padded, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_threads, aligned_tile_size);
                if (exact_size <= vtcm_budget) {
                    size_t mblocks = ((size_t) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                    if (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                        best_mblocks = mblocks;
                        best_act_threads = act_threads;
                        best_m_chunk = m_chunk_candidate;
                        best_n_chunk = n_chunk_candidate;
                        best_vtcm_size = exact_size;
                    }
                }
            }
            if (act_threads == 1) {
                act_threads = 0;
            } else {
                act_threads /= 2;
            }
        }

        if (best_act_threads > 0) {
            m_chunk = best_m_chunk;
            n_chunk = best_n_chunk;
            vtcm_size = best_vtcm_size;
            act_threads_selected = best_act_threads;
        } else {
            return false;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_threads = n_threads;
    kparams->n_act_threads = act_threads_selected;
    kparams->tile_size = htp_mm_get_weight_tile_size(wtype);
    kparams->aligned_tile_size = aligned_tile_size;
    kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    kparams->vtcm_size = vtcm_size;
    kparams->vtcm_src0_size = 0;
    kparams->vtcm_src1_size = 0;
    kparams->vtcm_dst_size = 0;

    if (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_F16_BATCHED;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    return true;
}

static void ggml_hexagon_precompute_hvx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne02,
    int ne03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    bool is_matmul_id,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    kparams->n_hmx = 0;

    const bool is_quant = (wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_F32);
    const int src1_nrows = ne11 * ne12 * ne13;

    if (is_quant) {
        // Quantized HVX
        kparams->tile_size = htp_mm_get_weight_tile_size(wtype);
        kparams->aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);

        const bool k_align = (ne10 % 32 == 0);

        if (is_matmul_id) {
            kparams->kernel_type   = (src1_nrows < (int) sess->n_threads) ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

            struct htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], 0, d, true, false, false
                );
                if (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (best_n_prefetch == 2 && L.total_bytes > vtcm_budget) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], 0, 2, true, false, false
                );
            }
            kparams->n_prefetch = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size  = L.dst_bytes;
        } else {
            bool try_tiled = (k_align && opt_mm_select >= 2);
            if (try_tiled) {
                kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
                if (src1_nrows < (int)sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } else {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                struct htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], d, false, false, false
                    );
                    if (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        break;
                    }
                }
                if (best_n_prefetch == 2 && L.total_bytes > vtcm_budget) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], 2, false, false, false
                    );
                }

                kparams->n_prefetch = best_n_prefetch;

                if (L.total_bytes <= vtcm_budget) {
                    kparams->vtcm_size = L.total_bytes;
                    kparams->vtcm_src0_size = L.src0_bytes;
                    kparams->vtcm_src1_size = L.src1_bytes;
                    kparams->vtcm_dst_size = L.dst_bytes;
                    goto done_quant;
                }
                HEX_VERBOSE("ggml-hex: %s HVX tiled path VTCM size needed (%zu) > budget (%zu), falling back to HVX flat\n", sess->name.c_str(), L.total_bytes, vtcm_budget);
            }

            // Flat HVX fallback
            {
                kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
                kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;

                struct htp_mm_hvx_vtcm_layout L;
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], 16, false, false, false
                );

                kparams->n_prefetch = 16;
                kparams->vtcm_size = L.total_bytes;
                kparams->vtcm_src0_size = L.src0_bytes;
                kparams->vtcm_src1_size = L.src1_bytes;
                kparams->vtcm_dst_size = L.dst_bytes;
            }
        }

    done_quant:;
    } else if (wtype == GGML_TYPE_F16) {
        // F16 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        struct htp_mm_hvx_vtcm_layout L;
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16, false, false, false
        );

        if (!is_batched && !is_permuted && L.total_bytes <= vtcm_budget) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->src1_row_size = hex_round_up(ne10 * 2, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        } else {
            if (src1->type == GGML_TYPE_F32) {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR;
            } else {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_DDR;
            }
            kparams->src1_row_size = src1->nb[1];
            htp_mm_hvx_vtcm_layout_build(
                &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16, false, false, false
            );
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        }
    } else {
        // F32 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        struct htp_mm_hvx_vtcm_layout L;
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16, false, false, false
        );

        if (!is_batched && !is_permuted && L.total_bytes <= vtcm_budget) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->src1_row_size = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        } else {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_DDR;
            kparams->src1_row_size = src1->nb[1];
            htp_mm_hvx_vtcm_layout_build(
                &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16, false, false, false
            );
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        }
    }
}

static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const int ne00_padded = is_repack ? hex_round_up(ne00, 32) : ne00;
    const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : ne01;
    const int ne11_padded = hex_round_up(ne11, 32);

    const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 3);
    if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, ne01_padded, is_matmul_id, is_batched)) {
        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_padded, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, vtcm_budget, kparams)) {
            goto finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne11     = init_fastdiv_values(ne11);
}

static void ggml_hexagon_precompute_fused_qkv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // Wk
    const struct ggml_tensor * src1, // x
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];

    uint32_t best_n_prefetch = 16;

    if (is_repack) {
        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            struct htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
                0, src0_row_size, src1_row_size, d, false, true, false
            );
            if (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                break;
            }
        }
    }

    struct htp_mm_hvx_vtcm_layout L;
    bool try_tiled = (opt_mm_select >= 2);

    // Test tiled first
    htp_mm_hvx_vtcm_layout_build(
        &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
        0, src0_row_size, src1_row_size, best_n_prefetch, false, true, false
    );

    if (try_tiled && L.total_bytes <= sess->vtcm_size) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_src3_size = L.src3_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT, wtype, ne10, src1_nrows, sess->n_threads,
            0, src0_row_size, flat_src1_row_size, best_n_prefetch, false, true, false
        );
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_src3_size = L.src3_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    }
}

static void ggml_hexagon_precompute_fused_ffn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // Wgate
    const struct ggml_tensor * src1, // y
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];

    uint32_t best_n_prefetch = 16;

    if (is_repack) {
        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            struct htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
                0, src0_row_size, src1_row_size, d, false, false, true
            );
            if (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                break;
            }
        }
    }

    struct htp_mm_hvx_vtcm_layout L;
    bool try_tiled = (opt_mm_select >= 2);

    // Test tiled first
    htp_mm_hvx_vtcm_layout_build(
        &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
        0, src0_row_size, src1_row_size, best_n_prefetch, false, false, true
    );

    if (try_tiled && L.total_bytes <= sess->vtcm_size) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT, wtype, ne10, src1_nrows, sess->n_threads,
            0, src0_row_size, flat_src1_row_size, best_n_prefetch, false, false, true
        );
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    }
}

static bool ggml_hexagon_supported_mul_mat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            if (src0->ne[0] % 32) {
                return false;
            }

            // hardcoded limit to refuse the lm-head for now
            if (src0->ne[1] > 32768) {
                return false;
            }

            if (src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;  // no broadcasting (for now)
            }

            // src0 (weights) must be repacked
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            break;

        case GGML_TYPE_F16:
            if (src0->nb[1] < src0->nb[0]) {
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                return false;
            }
            break;

        case GGML_TYPE_F32:
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->nb[1] < src0->nb[0]) {
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                return false;
            }
            break;

        default:
            return false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    if ((size_t)kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_mul_mat_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            if ((src0->ne[0] % 32)) {
                return false;
            }

            // src0 (weights) must be repacked
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            break;

        default:
            return false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    if ((size_t)kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_binary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        if (dst->type != GGML_TYPE_F32) {
            return false;
        }
    }
    else if (src0->type == GGML_TYPE_F16) {
        if (src1->type != GGML_TYPE_F16) {
            return false;
        }
        if (dst->type != GGML_TYPE_F16) {
            return false;
        }
    }
    else {
        return false;
    }

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }
    if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_add_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    // REVISIT: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_unary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    // dst must be contiguous; src0 may be non-contiguous
    if (!ggml_is_contiguous(dst)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_sum_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // TODO: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_activations(const struct ggml_hexagon_session * sess,
                                               const struct ggml_tensor *          op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_are_same_shape(src0, src1)) {
            return false;
        }
        if (!ggml_is_contiguous(src1)) {
            return false;
        }
    }

    return true;
}

static bool ggml_hexagon_supported_softmax(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src2) {
        return false;  // FIXME: add support for sinks
    }

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            return false;
        }
        if (src0->ne[0] != src1->ne[0]) {
            return false;
        }
        if (src1->ne[1] < src0->ne[1]) {
            return false;
        }
        if (src0->ne[2] % src1->ne[2] != 0) {
            return false;
        }
        if (src0->ne[3] % src1->ne[3] != 0) {
            return false;
        }
    }

    if (src1) {
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            return false;
        }
    } else {
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            return false;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    const int64_t ne0 = src0->ne[0];
    if (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        return false;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    if (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_set_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }

    if (dst->type != GGML_TYPE_F16) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_argsort(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * dst  = op;         // indices

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }

    if (dst->type != GGML_TYPE_I32) {
        return false;
    }

    if (src0->ne[0] > (16*1024)) {
        // reject tensors with huge rows for now
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_rope(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const int32_t * op_params = &op->op_params[0];

    int mode = op_params[2];

    // n_dims == ne0/2, so the rotation spans the full row
    if (mode == GGML_ROPE_TYPE_VISION) {
        const int n_dims = op_params[1];
        if (n_dims != (int) (op->src[0]->ne[0] / 2)) {
            return false;
        }
    }
    if (mode & 1) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;  // FIXME: add support for GGML_TYPE_F16 for src0
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->type != GGML_TYPE_I32) {
        return false;
    }
    if (src2) {
        if (src2->type != GGML_TYPE_F32) {
            return false;
        }
        int n_dims = op_params[1];
        if (src2->ne[0] < (n_dims / 2)) {
            return false;
        }
    }

    if (src2) {
        if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(src2)) {
            return false;
        }
    } else {
        if (!ggml_is_contiguous(src1)) {
            return false;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    if (src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] < src0->ne[0] * sizeof(float) || dst->nb[1] < dst->ne[0] * sizeof(float)) {
        return false;
    }
    return true;
}

static bool ggml_hexagon_supported_ssm_conv(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    // Only support FP32 for now
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Check IO tensor shapes and dims
    if (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        return false; // src0 should be effectively 3D
    }

    const int d_conv = src1->ne[0];
    const int d_inner = src0->ne[1];
    const int n_t = dst->ne[1];
    const int n_s = dst->ne[2];

    if (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        return false;
    }
    if (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        return false;
    }
    if (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] != src0->ne[0] * sizeof(float) || src1->nb[1] != src1->ne[0] * sizeof(float)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_pad(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_hexagon_supported_cumsum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_hexagon_supported_diag(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // diag only supports F32 currently
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Input must have ne[1] == 1 (vector input)
    if (src0->ne[1] != 1) {
        return false;
    }

    // Output must be square in first two dimensions
    if (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_hexagon_supported_solve_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // A
    const struct ggml_tensor * src1 = op->src[1]; // B
    const struct ggml_tensor * dst  = op;         // X

    if (!src0 || !src1) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] != src0->ne[1]) {
        return false;
    }

    if (src0->ne[1] != src1->ne[1]) {
        return false;
    }

    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        return false;
    }

    if (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_hexagon_supported_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) { return false; }
    if (dst->type  != GGML_TYPE_F32) { return false; }
    if (!ggml_are_same_shape(src0, dst)) { return false; }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { return false; }

    return true;

    GGML_UNUSED(sess);
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    return sess->c_name();
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

static htp_op_code op_remap_to_htp(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_FLASH_ATTN_EXT:  return HTP_OP_FLASH_ATTN_EXT;
        case GGML_OP_MUL_MAT:         return HTP_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID:      return HTP_OP_MUL_MAT_ID;
        case GGML_OP_MUL:             return HTP_OP_MUL;
        case GGML_OP_ADD:             return HTP_OP_ADD;
        case GGML_OP_ADD_ID:          return HTP_OP_ADD_ID;
        case GGML_OP_SUB:             return HTP_OP_SUB;
        case GGML_OP_DIV:             return HTP_OP_DIV;
        case GGML_OP_CPY:             return HTP_OP_CPY;
        case GGML_OP_CONT:            return HTP_OP_CPY;
        case GGML_OP_GET_ROWS:        return HTP_OP_GET_ROWS;
        case GGML_OP_SET_ROWS:        return HTP_OP_SET_ROWS;
        case GGML_OP_SUM_ROWS:        return HTP_OP_SUM_ROWS;
        case GGML_OP_ARGSORT:         return HTP_OP_ARGSORT;
        case GGML_OP_NORM:            return HTP_OP_NORM;
        case GGML_OP_L2_NORM:         return HTP_OP_L2_NORM;
        case GGML_OP_RMS_NORM:        return HTP_OP_RMS_NORM;
        case GGML_OP_CONCAT:          return HTP_OP_CONCAT;
        case GGML_OP_SCALE:           return HTP_OP_SCALE;
        case GGML_OP_SQR:             return HTP_OP_SQR;
        case GGML_OP_SQRT:            return HTP_OP_SQRT;
        case GGML_OP_SOFT_MAX:        return HTP_OP_SOFTMAX;
        case GGML_OP_SSM_CONV:        return HTP_OP_SSM_CONV;
        case GGML_OP_GATED_DELTA_NET: return HTP_OP_GATED_DELTA_NET;
        case GGML_OP_ROPE:            return HTP_OP_ROPE;
        case GGML_OP_REPEAT:          return HTP_OP_REPEAT;
        case GGML_OP_CUMSUM:          return HTP_OP_CUMSUM;
        case GGML_OP_FILL:            return HTP_OP_FILL;
        case GGML_OP_DIAG:            return HTP_OP_DIAG;
        case GGML_OP_SOLVE_TRI:       return HTP_OP_SOLVE_TRI;
        case GGML_OP_TRI:             return HTP_OP_TRI;
        case GGML_OP_PAD:             return HTP_OP_PAD;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU:       return HTP_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU:       return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_GELU_QUICK: return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_SIGMOID:    return HTP_OP_UNARY_SIGMOID;
                case GGML_UNARY_OP_NEG:        return HTP_OP_UNARY_NEG;
                case GGML_UNARY_OP_EXP:        return HTP_OP_UNARY_EXP;
                case GGML_UNARY_OP_SOFTPLUS:   return HTP_OP_UNARY_SOFTPLUS;
                case GGML_UNARY_OP_TANH:       return HTP_OP_UNARY_TANH;
            default:
                break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(t)) {
                case GGML_GLU_OP_SWIGLU:     return HTP_OP_GLU_SWIGLU;
                case GGML_GLU_OP_SWIGLU_OAI: return HTP_OP_GLU_SWIGLU_OAI;
                case GGML_GLU_OP_GEGLU:      return HTP_OP_GLU_GEGLU;
                default: break;
            }
            break;

        default:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    return HTP_OP_INVALID;
}

static inline bool op_is_compute(ggml_tensor *node)
{
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static bool mm_is_hmx_eligible(const ggml_tensor * t) {
    if (opt_nhmx == 0) { return false; }

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];

    const int wtype = src0->type;
    const bool is_repack    = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const bool is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    const int ne01_padded = is_repack ? hex_round_up(src0->ne[1], 32) : src0->ne[1];

    return ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, ne01_padded, is_matmul_id, is_batched);
}

static bool is_mergeable_mul_mat(const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_MUL_MAT)   return false;
    if (t->src[1]->type != GGML_TYPE_F32) return false;
    return ggml_is_quantized(t->src[0]->type) && !mm_is_hmx_eligible(t);
}

static bool is_mergeable_mul_mat_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) {
        return false;
    }
    if (n1->src[0]->ne[0] != n2->src[0]->ne[0] ||
        n1->src[0]->ne[1] != n2->src[0]->ne[1]) {
        return false;
    }
    if (n1->src[0]->type != n2->src[0]->type) {
        return false;
    }
    return true;
}

static bool is_qkv_mergeable(const ggml_tensor * n_q, const ggml_tensor * n_k, const ggml_tensor * n_v) {
    if (!is_mergeable_mul_mat(n_q) || !is_mergeable_mul_mat(n_k) || !is_mergeable_mul_mat(n_v)) {
        return false;
    }
    if (n_q->src[1] != n_k->src[1] || n_q->src[1] != n_v->src[1]) {
        return false;
    }
    if (n_q->src[0]->type != n_k->src[0]->type || n_q->src[0]->type != n_v->src[0]->type) {
        return false;
    }
    if (n_k->src[0]->ne[0] != n_v->src[0]->ne[0] ||
        n_k->src[0]->ne[1] != n_v->src[0]->ne[1]) {
        return false;
    }
    if (n_q->src[0]->ne[0] != n_k->src[0]->ne[0]) {
        return false;
    }
    return true;
}

static bool try_fuse_node(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int & i, std::vector<htp_opnode> & nodes) {
    if (!opt_opfusion) {
        return false;
    }

    ggml_tensor * n = graph->nodes[i];
    ggml_tensor * next_node = (i + 1 < graph->n_nodes) ? graph->nodes[i + 1] : nullptr;

    if (n->op == GGML_OP_RMS_NORM && next_node) {
        if (next_node->op == GGML_OP_MUL && op_is_compute(next_node) && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            htp_opnode node(n, {}, HTP_OP_RMS_NORM_MUL);
            node.add_fused(next_node);
            nodes.push_back(std::move(node));
            i++; // skip the fused MUL node
            return true;
        }
    }

    if (is_mergeable_mul_mat(n)) {
        ggml_tensor * n1 = (i + 1 < graph->n_nodes) ? graph->nodes[i + 1] : nullptr;
        ggml_tensor * n2 = (i + 2 < graph->n_nodes) ? graph->nodes[i + 2] : nullptr;
        if (is_qkv_mergeable(n, n1, n2)) {
            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_qkv_params(sess, n1->src[0], n1->src[1], &kparams);
            if ((size_t)kparams.vtcm_size <= sess->vtcm_size) {
                // Reorder to KVQ: K (n1), V (n2), Q (n)
                htp_opnode node(n1, {}, HTP_OP_MUL_MAT_QKV);
                node.add_fused(n2, true);
                node.add_fused(n, true);
                memcpy(node.kernel_params, &kparams, sizeof(kparams));
                nodes.push_back(std::move(node));
                i += 2;
                return true;
            } else {
                HEX_VERBOSE("ggml-hex: skip QKV fusion because VTCM needed (%d) > budget (%zu)\n",
                            kparams.vtcm_size, sess->vtcm_size);
            }
        }
        if (is_mergeable_mul_mat_pair(n, n1)) {
            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_ffn_params(sess, n->src[0], n->src[1], &kparams);
            if ((size_t)kparams.vtcm_size <= sess->vtcm_size) {
                htp_opnode node(n, {}, HTP_OP_MUL_MAT_FFN);
                node.add_fused(n1, true);
                memcpy(node.kernel_params, &kparams, sizeof(kparams));
                nodes.push_back(std::move(node));
                i += 1;
                return true;
            } else {
                HEX_VERBOSE("ggml-hex: skip FFN fusion because VTCM needed (%d) > budget (%zu)\n",
                            kparams.vtcm_size, sess->vtcm_size);
            }
        }
    }

    if (n->op == GGML_OP_MUL_MAT && next_node) {
        if (next_node->op == GGML_OP_ADD && op_is_compute(next_node) && ggml_can_fuse(graph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
            if (next_node->src[0] == n || next_node->src[1] == n) {
                struct htp_mm_kernel_params kparams;
                ggml_hexagon_precompute_matmul_params(sess, n->src[0], n->src[1], next_node, &kparams);
                if ((size_t)kparams.vtcm_size <= sess->vtcm_size) {
                    htp_opnode node(n, {}, HTP_OP_MUL_MAT_ADD);
                    node.add_fused(next_node);
                    memcpy(node.kernel_params, &kparams, sizeof(kparams));
                    nodes.push_back(std::move(node));
                    i += 1;
                    return true;
                } else {
                    HEX_VERBOSE("ggml-hex: skip MUL_MAT_ADD fusion because VTCM needed (%d) > budget (%zu)\n",
                                kparams.vtcm_size, sess->vtcm_size);
                }
            }
        }
    }

    return false;
}

static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    const std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    bool cache_hit = (graph->uid != 0 && sess->cached_graph.uid == graph->uid);
    if (cache_hit) {
        nodes_ptr = &sess->cached_graph.htp_nodes;
    } else {
        computed_nodes.reserve(graph->n_nodes);

        // Fuse and finalize
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            if (!op_is_compute(n)) {
                continue;
            }

            if (try_fuse_node(sess, graph, i, computed_nodes)) {
                continue;
            }

            htp_opnode node(n, {}, HTP_OP_INVALID);
            node.opcode = op_remap_to_htp(n);
            if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    node.node->src[0], node.node->src[1], node.node,
                    (struct htp_mm_kernel_params *)node.kernel_params
                );
            } else if (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    node.node,
                    (struct htp_fa_kernel_params *)node.kernel_params
                );
            }
            computed_nodes.push_back(std::move(node));
        }

        if (graph->uid != 0) {
            sess->cached_graph.uid = graph->uid;
            sess->cached_graph.htp_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_graph.htp_nodes;
        } else {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    if (opt_opstage & HTP_OPSTAGE_QUEUE) {
        for (const auto & node : *nodes_ptr) {
            sess->enqueue_op(node);
        }
    }

    // Wait until all pending ops complete
    sess->flush();

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush();
}

static std::vector<int> ggml_hexagon_graph_optimize_reorder(const std::vector<htp_opnode> & nodes) {
    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows use to reuse dynamically quantized src1 in VTCM.

    // TODO: the current version might do incorrect reordering in cases where quantized src0
    //       input is an output of another Op.

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        res.push_back(i0);

        const auto & node0 = nodes[i0];

        if (!node0.stackable()) {
            continue;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr int N_FORWARD = 16;

        for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            if (used[i1]) {
                continue;
            }

            const auto & node1 = nodes[i1];

            if (node1.stackable() && node1.same_input(node0)) {
                res.push_back(i1);
                used[i1] = true;
            }
        }
    }

    return res;
}

static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf) {
    const int n = gf->n_nodes;

    constexpr int MAX_FUSE = 16;

    enum ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        htp_opnode node = {
            /*.node =*/gf->nodes[i],
            /*.fused =*/{},
        };

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
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
}

static struct ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
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
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);

    return new ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);
    return sess->c_name();

    GGML_UNUSED(dev);
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    return "Hexagon";
    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = *free;

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
        /* .host_buffer           = */ (bool) opt_hostbuf,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);
    return &sess->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_repack_buffer_type(ggml_backend_dev_t dev) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);
    return &sess->repack_buffer_type;
}

static bool ggml_hexagon_supported_buffer(ggml_hexagon_session *sess, const struct ggml_tensor * t) {
    if (t && t->buffer) {
        if (ggml_backend_buffer_is_hexagon(t->buffer)      == false) return false; // not our buffer
        if (ggml_backend_hexagon_buffer_get_sess(t->buffer) != sess) return false; // wrong session
    }
    return true;
}

static bool ggml_hexagon_supported_buffers(ggml_hexagon_session *sess, const struct ggml_tensor * t) {
    // all srcs & dsts must be mapped to the same session
    if (!ggml_hexagon_supported_buffer(sess, t)) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (!ggml_hexagon_supported_buffer(sess, t->src[i])) {
            return false;
        }
    }

    return true;
}

static bool ggml_hexagon_supported_cpy(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // for now we can do f32 -> f16 and f16 -> f32 (without reshaping)
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
    if ( dst->type != GGML_TYPE_F32 &&  dst->type != GGML_TYPE_F16) return false;

    const bool sametype   = (src0->type == dst->type);
    const bool transposed = ggml_is_transposed(src0) || ggml_is_transposed(dst);
    const bool sameshape  = !transposed && ggml_are_same_shape(src0, dst);

    // can handle any shape and any same-type (pretty slow if reshaping is required)
    if (sametype) return true;

    // cannot handle re-shaping and type conversion at the same time
    if (!sameshape) return false;

    return true;
}

static bool ggml_hexagon_supported_cont(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src0 = op->src[0];

    // CONT is same-type only, supports f32 and f16
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;

    return true;
}

static bool ggml_hexagon_supported_repeat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // Support f32 and f16
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;

    // src and dst must be the same type
    if (src0->type != dst->type) return false;

    // dst dims must be multiples of src dims
    if (dst->ne[0] % src0->ne[0] != 0) return false;
    if (dst->ne[1] % src0->ne[1] != 0) return false;
    if (dst->ne[2] % src0->ne[2] != 0) return false;
    if (dst->ne[3] % src0->ne[3] != 0) return false;

    // require contiguous tensors (no transposition)
    if (ggml_is_transposed(src0) || ggml_is_transposed(dst)) return false;

    return true;
}

static bool ggml_hexagon_supported_concat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    int dim = ((const int32_t *) op->op_params)[0];
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const struct ggml_tensor * src = op->src[i];
        if (!src) {
            continue;
        }
        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32 && src->type != GGML_TYPE_F16) {
            return false;
        }
    }

    return true;
}

static bool ggml_hexagon_supported_fill(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * dst = op;

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);

    // reject ops that match the filter
    if (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        return false;
    }

    // all srcs & dsts must be mapped to the same session
    if (!ggml_hexagon_supported_buffers(sess, op)) {
        ggml_hexagon_dump_op_supp(sess->name, op, false);
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

        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            supp = ggml_hexagon_supported_binary(sess, op);
            break;

        case GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            break;

        case GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            break;

        case GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            break;

        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            break;

        case GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            break;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_TANH:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    break;
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    break;
                default:
                    break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    break;
                default:
                    break;
            }
            break;

        case GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            break;

        case GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            break;

        case GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            break;

        case GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            break;

        case GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            break;

        case GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            break;

        case GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            break;

        case GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            break;

        case GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            break;

        case GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            break;

        case GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            break;

        case GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            break;

        case GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            break;

        case GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            break;

        case GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            break;

        case GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            break;

        default:
            break;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    return supp;
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_alignment != ggml_backend_hexagon_buffer_type_get_alignment) {
        return false;
    }

    auto s0 = static_cast<ggml_hexagon_session *>(dev->context);
    auto s1 = static_cast<ggml_backend_hexagon_buffer_type_context *>(buft->context)->sess;

    // Need session/domain-id for buffers to be compatible
    bool supp = (s0->session_id == s1->session_id);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s (%d)\n", s0->name.c_str(), s1->name.c_str(), (int) supp);

    return supp;
}

static ggml_backend_buffer_type_t * ggml_backend_hexagon_device_get_extra_buffers_type(ggml_backend_dev_t dev) {
    auto s0 = static_cast<ggml_hexagon_session *>(dev->context);
    HEX_VERBOSE("ggml-hex: device-get-extra-buft : %s \n", s0->name.c_str());

    static ggml_backend_buffer_type_t bufts[2];
    bufts[0] = ggml_backend_hexagon_device_get_repack_buffer_type(dev);
    bufts[1] = NULL;
    return bufts;
}

static const struct ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,  // ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ NULL,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//** backend registry

#define GGML_HEXAGON_MAX_SESSIONS 16

struct ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d\n", opt_arch);

    // Create devices / sessions
    for (size_t i = 0; i < opt_ndev; i++) {
        devices[i].iface = ggml_backend_hexagon_device_i;
        devices[i].reg   = reg;
        try {
            devices[i].context = new ggml_hexagon_session(i, &devices[i]);
        } catch (const std::exception & exc) {
            GGML_LOG_ERROR("ggml-hex: failed to create device/session %zu\n", i);
            devices[i].context = nullptr;
        }
    }
}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices / sessions
    for (size_t i = 0; i < opt_ndev; i++) {
        auto sess = static_cast<ggml_hexagon_session *>(devices[i].context);
        delete sess;
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

static void * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0 && opt_hostbuf) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_hexagon_device_get_extra_buffers_type;
        return (void *) fct;
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

    const char * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    const char * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    const char * str_opstage  = getenv("GGML_HEXAGON_OPSTAGE");
    const char * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    const char * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    const char * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    const char * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    const char * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    const char * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    const char * str_etm      = getenv("GGML_HEXAGON_ETM");
    const char * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    const char * str_use_hmx  = getenv("GGML_HEXAGON_USE_HMX");
    const char * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    const char * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    const char * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    const char * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    const char * str_arch     = getenv("GGML_HEXAGON_ARCH");
    const char * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    const char * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    const char * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");

    // Init Arch first since it affects other defaults
    if (!str_arch) {
        int err = get_hex_arch_ver(CDSP_DOMAIN_ID, &opt_arch);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        }
    } else {
        if (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, NULL, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf)                      : opt_hostbuf;
    opt_opstage   = str_opstage  ? strtoul(str_opstage, NULL, 0)          : opt_opstage;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, NULL, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, NULL, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, NULL, 0)          : (opt_opbatch * 128);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  NULL, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, NULL, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : (str_use_hmx ? atoi(str_use_hmx) : opt_nhmx);
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_ndev      = str_ndev     ? strtoul(str_ndev, NULL, 0)             : opt_ndev;
    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf)                      : opt_hostbuf;
    opt_mbuf      = str_mbuf     ? strtoul(str_mbuf, NULL, 0) * MiB       : opt_mbuf;
    opt_vmem      = str_vmem     ? strtoul(str_vmem, NULL, 0) * MiB       : opt_vmem;

    if (opt_ndev > GGML_HEXAGON_MAX_SESSIONS) {
        opt_ndev = GGML_HEXAGON_MAX_SESSIONS;
    }

#if defined(__ANDROID__)
    if (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

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
