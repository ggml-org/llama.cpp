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
#include <unordered_set>
#include <unordered_map>
#include <regex>

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
#include "op-desc.h"
#include "htp-ops.h"
#include "htp_iface.h"
#include "htp-drv.h"

static size_t opt_ndev         = 1;
static size_t opt_nhvx         = 0; // use all
static int    opt_arch         = 0; // autodetect
static int    opt_etm          = 0;
static int    opt_verbose      = 0;
static int    opt_profile      = 0;
static int    opt_hostbuf      = 1; // hostbuf ON by default
static int    opt_use_hmx      = 1; // when set, enable HMX; when 0, use HVX only

// Enable all stages by default
static int opt_opmask   = HTP_OPMASK_QUEUE | HTP_OPMASK_COMPUTE;
static int opt_opsync   = 0;  // synchronous ops
static int opt_opbatch  = 1024; // max number of ops in a batch
static int opt_opqueue  = 16;   // max number of pending batches
static std::regex* opt_opfilter = NULL; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

static inline uint64_t hex_is_aligned(void * addr, uint32_t align) {
    return ((size_t) addr & (align - 1)) == 0;
}

static inline size_t hex_round_up(size_t n, size_t m) {
    return m * ((n + m - 1) / m);
}

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

static void ggml_hexagon_dump_op_exec(const std::string &sess_name, const ggml_tensor * op, const uint32_t req_flags) {
    if (!opt_verbose) return;

    op_desc desc(op);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s: %s : %s : %s : %s : %s : flags 0x%x\n", sess_name.c_str(),
                ggml_op_desc(op), desc.names, desc.dims, desc.types, desc.strides, desc.buffs, req_flags);
}

static void ggml_hexagon_dump_op_supp(const std::string &sess_name, const struct ggml_tensor * op, bool supp) {
    if (!opt_verbose) return;

    op_desc desc(op);
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s : %s : %s : %s : %s : %s : %s\n", sess_name.c_str(),
                ggml_op_desc(op), desc.names, desc.dims, desc.types, desc.strides, desc.buffs, supp ? "yes" : "no");
}

static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const ggml_tensor * op,
                                      uint32_t op_usec, uint32_t op_cycles, uint32_t op_pkts, uint64_t call_usec) {
    if (!opt_profile) return;

    op_desc desc(op);
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s: %s : %s : %s : %s : %s : op-usec %u op-cycles %u op-pkts %u (%f) call-usec %llu\n", sess_name.c_str(),
                ggml_op_desc(op), desc.names, desc.dims, desc.types, desc.strides, desc.buffs,
                op_usec, op_cycles, op_pkts, (float) op_cycles / op_pkts, (unsigned long long) call_usec);
}

// ** backend sessions

struct ggml_hexagon_opbatch;
struct ggml_hexagon_opshm;

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
    ggml_hexagon_opbatch *op_batch;
    ggml_hexagon_opshm   *op_shm;

    ggml_backend_buffer_type buffer_type        = {};
    ggml_backend_buffer_type repack_buffer_type = {};

    ggml_hexagon_session(int dev_id, ggml_backend_dev_t dev) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    const char* c_name() const { return name.c_str(); }

    void allocate(int dev_id) noexcept(false);
    void release() noexcept(true);

    void enqueue_op(htp_op_code opcode, const ggml_tensor *op);
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

    void mmap(bool pinned = false) {
        int err = fastrpc_mmap(sess->domain_id, this->fd, (void *) this->base, 0, this->size, FASTRPC_MAP_FD_DELAYED);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, this->size, this->fd, (unsigned) err);
            throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        if (pinned) {
            err = htp_iface_mmap(sess->handle, this->fd, this->size, pinned);
            if (err != 0) {
                GGML_LOG_ERROR("ggml-hex: %s buffer pinning failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                        sess->domain_id, this->size, this->fd, (unsigned) err);
                throw std::runtime_error("ggml-hex: htp_iface_mmap failed (see log for details)");
            }
        }

        this->mapped = true;
        this->pinned = pinned;
        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u\n",
                sess->c_name(), (void *) this->base, this->size, this->fd, pinned);
    }

    void unmap() {
        if (!this->mapped) return;

        htp_iface_munmap(sess->handle, this->fd);
        fastrpc_munmap(sess->domain_id, this->fd, (void *) this->base, this->size);

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) this->base, size, this->fd);

        this->mapped = false;
        this->fd     = -1;
    }

    void alloc(size_t size, bool pinned = false) {
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

        mmap(pinned);
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
        size += 4 * 1024;  // extra page for padding

        this->sess   = sess;
        this->size   = 0;
        this->base   = nullptr;
        this->fd     = -1;
        this->mapped = false;

        alloc(size, pinned);
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

// ======== Q4x4x2 ====================
struct x2_q4 {
    int v[2];
};

static x2_q4 unpack_q4(uint8_t v) {
    x2_q4 x = { (int) (v & 0x0f) - 8, (int) (v >> 4) - 8 };
    return x;
}

static void dump_block_q4_0(const block_q4_0 * b, int i) {
    HEX_VERBOSE("ggml-hex: repack q4_0 %d: %d %d %d %d ... %d %d %d %d : %.6f\n", i, unpack_q4(b->qs[0]).v[0],
                unpack_q4(b->qs[1]).v[0], unpack_q4(b->qs[2]).v[0], unpack_q4(b->qs[3]).v[0], unpack_q4(b->qs[12]).v[1],
                unpack_q4(b->qs[13]).v[1], unpack_q4(b->qs[14]).v[1], unpack_q4(b->qs[15]).v[1],
                GGML_FP16_TO_FP32(b->d));
}

static void dump_packed_block_q4x4x2(const uint8_t * v, unsigned int i, size_t k) {
    static const int qk        = QK_Q4_0x4x2;
    const int        dblk_size = 8 * 2;   // 8x __fp16
    const int        qblk_size = qk / 2;  // int4
    const int        qrow_size = k / 2;   // int4 (not padded)

    const uint8_t * v_q = v + 0;          // quants first
    const uint8_t * v_d = v + qrow_size;  // then scales

    const uint8_t *   q = v_q + i * qblk_size;
    const ggml_half * d = (const ggml_half *) (v_d + i * dblk_size);

    HEX_VERBOSE("ggml-hex: repack q4x4x2-%d: %d %d %d %d ... %d %d %d %d ... %d %d %d %d : %.6f %.6f %.6f %.6f\n", i,
                unpack_q4(q[0]).v[0], unpack_q4(q[1]).v[0], unpack_q4(q[2]).v[0], unpack_q4(q[3]).v[0],
                unpack_q4(q[60]).v[0], unpack_q4(q[61]).v[0], unpack_q4(q[62]).v[0], unpack_q4(q[63]).v[0],
                unpack_q4(q[124]).v[0], unpack_q4(q[125]).v[0], unpack_q4(q[126]).v[0], unpack_q4(q[127]).v[0],
                GGML_FP16_TO_FP32(d[0]), GGML_FP16_TO_FP32(d[1]), GGML_FP16_TO_FP32(d[2]), GGML_FP16_TO_FP32(d[3]));

    HEX_VERBOSE("ggml-hex: repack q4x4x2-%d: %d %d %d %d ... %d %d %d %d ... %d %d %d %d : %.6f %.6f %.6f %.6f\n",
                i + 1, unpack_q4(q[0]).v[1], unpack_q4(q[1]).v[1], unpack_q4(q[2]).v[1], unpack_q4(q[3]).v[1],
                unpack_q4(q[60]).v[1], unpack_q4(q[61]).v[1], unpack_q4(q[62]).v[1], unpack_q4(q[63]).v[1],
                unpack_q4(q[124]).v[1], unpack_q4(q[125]).v[1], unpack_q4(q[126]).v[1], unpack_q4(q[127]).v[1],
                GGML_FP16_TO_FP32(d[4]), GGML_FP16_TO_FP32(d[5]), GGML_FP16_TO_FP32(d[6]), GGML_FP16_TO_FP32(d[7]));
}

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

static void repack_row_q4x4x2(uint8_t * y, const block_q4_0 * x, int64_t k) {
    static const int qk = QK_Q4_0x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)
    const int        nloe = k % qk;           // leftovers

    const int dblk_size = 8 * 2;              // 8x __fp16
    const int qblk_size = qk / 2;             // int4
    const int qrow_size = k / 2;              // int4 (not padded to blocks)

    uint8_t * y_q = y + 0;                    // quants first
    uint8_t * y_d = y + qrow_size;            // then scales

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_block_q4_0(&x[i * 8 + 0], 0);
            dump_block_q4_0(&x[i * 8 + 1], 1);
            dump_block_q4_0(&x[i * 8 + 2], 2);
            dump_block_q4_0(&x[i * 8 + 3], 3);
            dump_block_q4_0(&x[i * 8 + 4], 4);
            dump_block_q4_0(&x[i * 8 + 5], 5);
            dump_block_q4_0(&x[i * 8 + 6], 6);
            dump_block_q4_0(&x[i * 8 + 7], 7);
        }
    }

    // Repack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_Q4_0x4x2];  // unpacked quants
        unpack_q4_0_quants(qs, &x[i * 8 + 0], 0);
        unpack_q4_0_quants(qs, &x[i * 8 + 1], 1);
        unpack_q4_0_quants(qs, &x[i * 8 + 2], 2);
        unpack_q4_0_quants(qs, &x[i * 8 + 3], 3);
        unpack_q4_0_quants(qs, &x[i * 8 + 4], 4);
        unpack_q4_0_quants(qs, &x[i * 8 + 5], 5);
        unpack_q4_0_quants(qs, &x[i * 8 + 6], 6);
        unpack_q4_0_quants(qs, &x[i * 8 + 7], 7);

        bool partial = (nloe && i == nb-1);

        uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            q[j] = partial ? (qs[j*2+1] << 4) | qs[j*2+0] : (qs[j+128] << 4) | qs[j+000];
        }
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_Q4_0x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Repack the scales
        ggml_half * d = (ggml_half *) (y_d + i * dblk_size);
        d[0]          = x[i * 8 + 0].d;
        d[1]          = x[i * 8 + 1].d;
        d[2]          = x[i * 8 + 2].d;
        d[3]          = x[i * 8 + 3].d;
        d[4]          = x[i * 8 + 4].d;
        d[5]          = x[i * 8 + 5].d;
        d[6]          = x[i * 8 + 6].d;
        d[7]          = x[i * 8 + 7].d;
    }

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_q4x4x2(y, i, k);
        }
    }
}

static void unpack_row_q4x4x2(block_q4_0 * x, const uint8_t * y, int64_t k) {
    static const int qk = QK_Q4_0x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)
    const int        nloe = k % qk;           // leftovers

    const int dblk_size = 8 * 2;              // 8x __fp16
    const int qblk_size = qk / 2;             // int4
    const int qrow_size = k / 2;              // int4 (not padded to blocks)

    const uint8_t * y_q = y + 0;              // quants first
    const uint8_t * y_d = y + qrow_size;      // then scales

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_q4x4x2(y, i, k);
        }
    }

    // Unpack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_Q4_0x4x2];  // unpacked quants

        bool partial = (nloe && i == nb-1);

        const uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            if (partial) {
                qs[j*2+0] = q[j] & 0xf;
                qs[j*2+1] = q[j] >> 4;
            } else {
                qs[j+000] = q[j] & 0xf;
                qs[j+128] = q[j] >> 4;
            }
        }

        pack_q4_0_quants(&x[i * 8 + 0], qs, 0);
        pack_q4_0_quants(&x[i * 8 + 1], qs, 1);
        pack_q4_0_quants(&x[i * 8 + 2], qs, 2);
        pack_q4_0_quants(&x[i * 8 + 3], qs, 3);
        pack_q4_0_quants(&x[i * 8 + 4], qs, 4);
        pack_q4_0_quants(&x[i * 8 + 5], qs, 5);
        pack_q4_0_quants(&x[i * 8 + 6], qs, 6);
        pack_q4_0_quants(&x[i * 8 + 7], qs, 7);
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_Q4_0x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Unpack the scales
        const ggml_half * d = (const ggml_half *) (y_d + i * dblk_size);
        x[i * 8 + 0].d      = d[0];
        x[i * 8 + 1].d      = d[1];
        x[i * 8 + 2].d      = d[2];
        x[i * 8 + 3].d      = d[3];
        x[i * 8 + 4].d      = d[4];
        x[i * 8 + 5].d      = d[5];
        x[i * 8 + 6].d      = d[6];
        x[i * 8 + 7].d      = d[7];
    }

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_block_q4_0(&x[i * 8 + 0], 0);
            dump_block_q4_0(&x[i * 8 + 1], 1);
            dump_block_q4_0(&x[i * 8 + 2], 2);
            dump_block_q4_0(&x[i * 8 + 3], 3);
            dump_block_q4_0(&x[i * 8 + 4], 4);
            dump_block_q4_0(&x[i * 8 + 5], 5);
            dump_block_q4_0(&x[i * 8 + 6], 6);
            dump_block_q4_0(&x[i * 8 + 7], 7);
        }
    }
}

static void init_row_q4x4x2(block_q4_0 * x, int64_t k) {
    static const int qk = QK_Q4_0x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)

    // Init the quants such that they unpack into zeros
    uint8_t qs[QK_Q4_0x4x2];  // unpacked quants
    memset(qs, 8, sizeof(qs));

    for (int i = 0; i < nb; i++) {
        pack_q4_0_quants(&x[i * 8 + 0], qs, 0);
        pack_q4_0_quants(&x[i * 8 + 1], qs, 1);
        pack_q4_0_quants(&x[i * 8 + 2], qs, 2);
        pack_q4_0_quants(&x[i * 8 + 3], qs, 3);
        pack_q4_0_quants(&x[i * 8 + 4], qs, 4);
        pack_q4_0_quants(&x[i * 8 + 5], qs, 5);
        pack_q4_0_quants(&x[i * 8 + 6], qs, 6);
        pack_q4_0_quants(&x[i * 8 + 7], qs, 7);
    }

    // Init the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_Q4_0x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Unpack the scales
        x[i * 8 + 0].d = 0;
        x[i * 8 + 1].d = 0;
        x[i * 8 + 2].d = 0;
        x[i * 8 + 3].d = 0;
        x[i * 8 + 4].d = 0;
        x[i * 8 + 5].d = 0;
        x[i * 8 + 6].d = 0;
        x[i * 8 + 7].d = 0;
    }
}

// repack q4_0 data into q4x4x2 tensor
static void repack_q4_0_q4x4x2(ggml_tensor * t, const void * data, size_t size) {
    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up(t->ne[0], QK_Q4_0x4x2));  // extra elements for the pad
    size_t row_size_rp = row_size * 2;  // extra space for tmp pad (if any)

    // Ensure we don't try to read more data than is available in the source buffer 'data'
    // or write more than the tensor can hold.
    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    // Calculate how many full rows and how many remaining bytes we need to process.
    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    HEX_VERBOSE("ggml-hex: repack-q4_0-q4x4x2 %s : data %p size %zu dims %ldx%ld row-size %zu\n", t->name, data, size,
                t->ne[0], nrows, row_size);

    init_row_q4x4x2((block_q4_0 *) buf_pd, t->ne[0]);  // init padded buffer to make sure the tail is all zeros

    // 1. Process all the full rows
    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) data + (i * row_size);
        uint8_t *       dst = (uint8_t *) t->data + (i * row_size);

        memcpy(buf_pd, src, row_size);
        repack_row_q4x4x2((uint8_t *) buf_rp, (const block_q4_0 *) buf_pd, t->ne[0]);
        memcpy(dst, buf_rp, row_size);
    }

    // 2. Process the final, potentially partial, row
    if (n_rem_bytes > 0) {
        const int64_t i = n_full_rows;
        const uint8_t * src = (const uint8_t *) data + (i * row_size);
        uint8_t *       dst = (uint8_t *) t->data + (i * row_size);

        // re-init the row because we are potentially copying a partial row
        init_row_q4x4x2((block_q4_0 *) buf_pd, t->ne[0]);

        // Copy only the remaining bytes from the source.
        memcpy(buf_pd, src, n_rem_bytes);

        // Repack the entire buffer
        repack_row_q4x4x2((uint8_t *) buf_rp, (const block_q4_0 *) buf_pd, t->ne[0]);

        // Write only the corresponding remaining bytes to the destination tensor.
        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
}

// repack q4x4x2 tensor into q4_0 data
static void repack_q4x4x2_q4_0(void * data, const ggml_tensor * t, size_t size) {
    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up(t->ne[0], QK_Q4_0x4x2));  // extra elements for the pad
    size_t row_size_rp = row_size * 2;  // extra space for tmp pad (if any)

    // Ensure we don't try to copy more data than the tensor actually contains.
    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    // Calculate how many full rows and how many remaining bytes we need to process.
    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    HEX_VERBOSE("ggml-hex: repack-q4x4x2-q4_0 %s : data %p size %zu dims %ldx%ld row-size %zu\n", t->name, data, size,
                t->ne[0], nrows, row_size);

    memset(buf_pd, 0, row_size_pd);  // clear-out padded buffer to make sure the tail is all zeros

    // 1. Process all the full rows
    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        memcpy(buf_pd, src, row_size);
        unpack_row_q4x4x2((block_q4_0 *) buf_rp, (const uint8_t *) buf_pd, t->ne[0]);
        memcpy(dst, buf_rp, row_size);
    }

    // 2. Process the final, potentially partial, row
    if (n_rem_bytes > 0) {
        const int64_t i = n_full_rows;
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        // We still need to read and unpack the entire source row because quantization is block-based.
        memcpy(buf_pd, src, row_size);
        unpack_row_q4x4x2((block_q4_0 *) buf_rp, (const uint8_t *) buf_pd, t->ne[0]);

        // But we only copy the remaining number of bytes to the destination.
        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
}

// ======== Q8x4x2 ====================
static void dump_block_q8_0(const block_q8_0 * b, int i) {
    HEX_VERBOSE("ggml-hex: repack q8_0 %d: %d %d %d %d ... %d %d %d %d : %.6f\n", i, b->qs[0], b->qs[1], b->qs[2],
                b->qs[3], b->qs[28], b->qs[29], b->qs[30], b->qs[31], GGML_FP16_TO_FP32(b->d));
}

static void dump_packed_block_q8x4x2(const uint8_t * v, unsigned int i, size_t k) {
    static const int qk        = QK_Q8_0x4x2;
    const int        dblk_size = 8 * 2;   // 8x __fp16
    const int        qblk_size = qk;      // int8
    const int        qrow_size = k;       // int8 (not padded)

    const uint8_t * v_q = v + 0;          // quants first
    const uint8_t * v_d = v + qrow_size;  // then scales

    const uint8_t *   q = v_q + i * qblk_size;
    const ggml_half * d = (const ggml_half *) (v_d + i * dblk_size);

    HEX_VERBOSE("ggml-hex: repack q8x4x2-%d: %d %d %d %d ... %d %d %d %d ... %d %d %d %d : %.6f %.6f %.6f %.6f\n", i,
                q[0], q[1], q[2], q[3], q[60], q[61], q[62], q[63], q[124], q[125], q[126], q[127],
                GGML_FP16_TO_FP32(d[0]), GGML_FP16_TO_FP32(d[1]), GGML_FP16_TO_FP32(d[2]), GGML_FP16_TO_FP32(d[3]));

    HEX_VERBOSE("ggml-hex: repack q8x4x2-%d: %d %d %d %d ... %d %d %d %d ... %d %d %d %d : %.6f %.6f %.6f %.6f\n",
                i + 1, q[128], q[129], q[130], q[131], q[192], q[193], q[194], q[195], q[252], q[253], q[254], q[255],
                GGML_FP16_TO_FP32(d[4]), GGML_FP16_TO_FP32(d[5]), GGML_FP16_TO_FP32(d[6]), GGML_FP16_TO_FP32(d[7]));
}

static void unpack_q8_0_quants(uint8_t * qs, const block_q8_0 * x, unsigned int bi) {
    static const int qk = QK8_0;

    for (unsigned int i = 0; i < qk; ++i) {
        qs[bi * qk + i] = x->qs[i];
    }
}

static void pack_q8_0_quants(block_q8_0 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK8_0;

    for (unsigned int i = 0; i < qk; ++i) {
        x->qs[i] = qs[bi * qk + i];
    }
}

static void repack_row_q8x4x2(uint8_t * y, const block_q8_0 * x, int64_t k) {
    static const int qk = QK_Q8_0x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)

    const int dblk_size = 8 * 2;              // 8x __fp16
    const int qblk_size = qk;                 // int8
    const int qrow_size = k;                  // int8 (not padded to blocks)

    uint8_t * y_q = y + 0;                    // quants first
    uint8_t * y_d = y + qrow_size;            // then scales

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_block_q8_0(&x[i * 8 + 0], 0);
            dump_block_q8_0(&x[i * 8 + 1], 1);
            dump_block_q8_0(&x[i * 8 + 2], 2);
            dump_block_q8_0(&x[i * 8 + 3], 3);
            dump_block_q8_0(&x[i * 8 + 4], 4);
            dump_block_q8_0(&x[i * 8 + 5], 5);
            dump_block_q8_0(&x[i * 8 + 6], 6);
            dump_block_q8_0(&x[i * 8 + 7], 7);
        }
    }

    // Repack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_Q8_0x4x2];  // unpacked quants

        unpack_q8_0_quants(qs, &x[i * 8 + 0], 0);
        unpack_q8_0_quants(qs, &x[i * 8 + 1], 1);
        unpack_q8_0_quants(qs, &x[i * 8 + 2], 2);
        unpack_q8_0_quants(qs, &x[i * 8 + 3], 3);
        unpack_q8_0_quants(qs, &x[i * 8 + 4], 4);
        unpack_q8_0_quants(qs, &x[i * 8 + 5], 5);
        unpack_q8_0_quants(qs, &x[i * 8 + 6], 6);
        unpack_q8_0_quants(qs, &x[i * 8 + 7], 7);

        uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk; j++) {
            q[j] = qs[j];
        }
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_Q4_0x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Repack the scales
        ggml_half * d = (ggml_half *) (y_d + i * dblk_size);
        d[0]          = x[i * 8 + 0].d;
        d[1]          = x[i * 8 + 1].d;
        d[2]          = x[i * 8 + 2].d;
        d[3]          = x[i * 8 + 3].d;
        d[4]          = x[i * 8 + 4].d;
        d[5]          = x[i * 8 + 5].d;
        d[6]          = x[i * 8 + 6].d;
        d[7]          = x[i * 8 + 7].d;
    }

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_q8x4x2(y, i, k);
        }
    }
}

static void unpack_row_q8x4x2(block_q8_0 * x, const uint8_t * y, int64_t k) {
    static const int qk = QK_Q8_0x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)

    const int dblk_size = 8 * 2;              // 8x __fp16
    const int qblk_size = qk;                 // int8
    const int qrow_size = k;                  // int8 (not padded to blocks)

    const uint8_t * y_q = y + 0;              // quants first
    const uint8_t * y_d = y + qrow_size;      // then scales

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_q8x4x2(y, i, k);
        }
    }

    // Unpack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_Q4_0x4x2];  // unpacked quants

        const uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk; j++) {
            qs[j] = q[j];
        }

        pack_q8_0_quants(&x[i * 8 + 0], qs, 0);
        pack_q8_0_quants(&x[i * 8 + 1], qs, 1);
        pack_q8_0_quants(&x[i * 8 + 2], qs, 2);
        pack_q8_0_quants(&x[i * 8 + 3], qs, 3);
        pack_q8_0_quants(&x[i * 8 + 4], qs, 4);
        pack_q8_0_quants(&x[i * 8 + 5], qs, 5);
        pack_q8_0_quants(&x[i * 8 + 6], qs, 6);
        pack_q8_0_quants(&x[i * 8 + 7], qs, 7);
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_Q4_0x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Unpack the scales
        const ggml_half * d = (const ggml_half *) (y_d + i * dblk_size);
        x[i * 8 + 0].d      = d[0];
        x[i * 8 + 1].d      = d[1];
        x[i * 8 + 2].d      = d[2];
        x[i * 8 + 3].d      = d[3];
        x[i * 8 + 4].d      = d[4];
        x[i * 8 + 5].d      = d[5];
        x[i * 8 + 6].d      = d[6];
        x[i * 8 + 7].d      = d[7];
    }

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_block_q8_0(&x[i * 8 + 0], 0);
            dump_block_q8_0(&x[i * 8 + 1], 1);
            dump_block_q8_0(&x[i * 8 + 2], 2);
            dump_block_q8_0(&x[i * 8 + 3], 3);
            dump_block_q8_0(&x[i * 8 + 4], 4);
            dump_block_q8_0(&x[i * 8 + 5], 5);
            dump_block_q8_0(&x[i * 8 + 6], 6);
            dump_block_q8_0(&x[i * 8 + 7], 7);
        }
    }
}

static void init_row_q8x4x2(block_q8_0 * x, int64_t k) {
    static const int qk = QK_Q8_0x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)

    // Init the quants such that they unpack into zeros
    uint8_t qs[QK_Q8_0x4x2];  // unpacked quants
    memset(qs, 0, sizeof(qs));

    for (int i = 0; i < nb; i++) {
        pack_q8_0_quants(&x[i * 8 + 0], qs, 0);
        pack_q8_0_quants(&x[i * 8 + 1], qs, 1);
        pack_q8_0_quants(&x[i * 8 + 2], qs, 2);
        pack_q8_0_quants(&x[i * 8 + 3], qs, 3);
        pack_q8_0_quants(&x[i * 8 + 4], qs, 4);
        pack_q8_0_quants(&x[i * 8 + 5], qs, 5);
        pack_q8_0_quants(&x[i * 8 + 6], qs, 6);
        pack_q8_0_quants(&x[i * 8 + 7], qs, 7);
    }

    // Init the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_Q8_0x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Unpack the scales
        x[i * 8 + 0].d = 0;
        x[i * 8 + 1].d = 0;
        x[i * 8 + 2].d = 0;
        x[i * 8 + 3].d = 0;
        x[i * 8 + 4].d = 0;
        x[i * 8 + 5].d = 0;
        x[i * 8 + 6].d = 0;
        x[i * 8 + 7].d = 0;
    }
}

// repack q8_0 data into q8x4x2 tensor
static void repack_q8_0_q8x4x2(ggml_tensor * t, const void * data, size_t size) {
    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up(t->ne[0], QK_Q8_0x4x2));  // extra elements for the pad
    size_t row_size_rp = row_size * 2;  // extra space for tmp pad (if any)

    // Ensure we don't try to read more data than is available in the source buffer 'data'
    // or write more than the tensor can hold.
    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    // Calculate how many full rows and how many remaining bytes we need to process.
    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    HEX_VERBOSE("ggml-hex: repack-q8_0-q8x4x2 %s : data %p size %zu dims %ldx%ld row-size %zu\n", t->name, data, size,
                t->ne[0], nrows, row_size);

    init_row_q8x4x2((block_q8_0 *) buf_pd, t->ne[0]);  // init padded buffer to make sure the tail is all zeros

    // 1. Process all the full rows
    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) data + (i * row_size);
        uint8_t *       dst = (uint8_t *) t->data + (i * row_size);

        memcpy(buf_pd, src, row_size);
        repack_row_q8x4x2((uint8_t *) buf_rp, (const block_q8_0 *) buf_pd, t->ne[0]);
        memcpy(dst, buf_rp, row_size);
    }

    // 2. Process the final, potentially partial, row
    if (n_rem_bytes > 0) {
        const int64_t i = n_full_rows;
        const uint8_t * src = (const uint8_t *) data + (i * row_size);
        uint8_t *       dst = (uint8_t *) t->data + (i * row_size);

        // re-init the row because we are potentially copying a partial row
        init_row_q8x4x2((block_q8_0 *) buf_pd, t->ne[0]);

        // Copy only the remaining bytes from the source.
        memcpy(buf_pd, src, n_rem_bytes);

        // Repack the entire buffer
        repack_row_q8x4x2((uint8_t *) buf_rp, (const block_q8_0 *) buf_pd, t->ne[0]);

        // Write only the corresponding remaining bytes to the destination tensor.
        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
}

// repack q8x4x2 tensor into q8_0 data
static void repack_q8x4x2_q8_0(void * data, const ggml_tensor * t, size_t size) {
    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up(t->ne[0], QK_Q8_0x4x2));  // extra elements for the pad
    size_t row_size_rp = row_size * 2;  // extra space for tmp pad (if any)

    // Ensure we don't try to copy more data than the tensor actually contains.
    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    // Calculate how many full rows and how many remaining bytes we need to process.
    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    HEX_VERBOSE("ggml-hex: repack-q8x4x2-q8_0 %s : data %p size %zu dims %ldx%ld row-size %zu\n", t->name, data, size,
                t->ne[0], nrows, row_size);

    memset(buf_pd, 0, row_size_pd);  // clear-out padded buffer to make sure the tail is all zeros

    // 1. Process all the full rows
    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        memcpy(buf_pd, src, row_size);
        unpack_row_q8x4x2((block_q8_0 *) buf_rp, (const uint8_t *) buf_pd, t->ne[0]);
        memcpy(dst, buf_rp, row_size);
    }

    // 2. Process the final, potentially partial, row
    if (n_rem_bytes > 0) {
        const int64_t i = n_full_rows;
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        // We still need to read and unpack the entire source row because quantization is block-based.
        memcpy(buf_pd, src, row_size);
        unpack_row_q8x4x2((block_q8_0 *) buf_rp, (const uint8_t *) buf_pd, t->ne[0]);

        // But we only copy the remaining number of bytes to the destination.
        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
}

// ======== MXFP4x4x2 ====================
struct x2_mxfp4 {
    int v[2];
};

static x2_mxfp4 unpack_mxfp4(uint8_t v) {
    x2_mxfp4 x;
    x.v[0] = kvalues_mxfp4[(v & 0x0f)];
    x.v[1] = kvalues_mxfp4[(v >> 4)];
    return x;
}

static void dump_block_mxfp4(const block_mxfp4 * b, int i) {
    HEX_VERBOSE("ggml-hex: repack mxfp4 %d: %d %d %d %d ... %d %d %d %d : %.6f\n", i, unpack_mxfp4(b->qs[0]).v[0],
                unpack_mxfp4(b->qs[1]).v[0], unpack_mxfp4(b->qs[2]).v[0], unpack_mxfp4(b->qs[3]).v[0],
                unpack_mxfp4(b->qs[12]).v[1], unpack_mxfp4(b->qs[13]).v[1], unpack_mxfp4(b->qs[14]).v[1],
                unpack_mxfp4(b->qs[15]).v[1], GGML_E8M0_TO_FP32_HALF(b->e));
}

static void dump_packed_block_mxfp4x4x2(const uint8_t * v, unsigned int i, size_t k) {
    static const int qk        = QK_MXFP4x4x2;
    const int        eblk_size = 8 * 1;   // 8x E8M0
    const int        qblk_size = qk / 2;  // int4
    const int        qrow_size = k / 2;   // int4 (not padded)

    const uint8_t * v_q = v + 0;          // quants first
    const uint8_t * v_e = v + qrow_size;  // then scales

    const uint8_t * q = v_q + i * qblk_size;
    const uint8_t * e = (const uint8_t *) (v_e + i * eblk_size);

    HEX_VERBOSE("ggml-hex: repack mxfp4x4x2-%d: %d %d %d %d ... %d %d %d %d ... %d %d %d %d : %.6f %.6f %.6f %.6f\n", i,
                unpack_mxfp4(q[0]).v[0], unpack_mxfp4(q[1]).v[0], unpack_mxfp4(q[2]).v[0], unpack_mxfp4(q[3]).v[0],
                unpack_mxfp4(q[60]).v[0], unpack_mxfp4(q[61]).v[0], unpack_mxfp4(q[62]).v[0], unpack_mxfp4(q[63]).v[0],
                unpack_mxfp4(q[124]).v[0], unpack_mxfp4(q[125]).v[0], unpack_mxfp4(q[126]).v[0],
                unpack_mxfp4(q[127]).v[0], GGML_E8M0_TO_FP32_HALF(e[0]), GGML_E8M0_TO_FP32_HALF(e[1]),
                GGML_E8M0_TO_FP32_HALF(e[2]), GGML_E8M0_TO_FP32_HALF(e[3]));

    HEX_VERBOSE("ggml-hex: repack mxfp4x4x2-%d: %d %d %d %d ... %d %d %d %d ... %d %d %d %d : %.6f %.6f %.6f %.6f\n",
                i + 1, unpack_mxfp4(q[0]).v[1], unpack_mxfp4(q[1]).v[1], unpack_mxfp4(q[2]).v[1],
                unpack_mxfp4(q[3]).v[1], unpack_mxfp4(q[60]).v[1], unpack_mxfp4(q[61]).v[1], unpack_mxfp4(q[62]).v[1],
                unpack_mxfp4(q[63]).v[1], unpack_mxfp4(q[124]).v[1], unpack_mxfp4(q[125]).v[1],
                unpack_mxfp4(q[126]).v[1], unpack_mxfp4(q[127]).v[1], GGML_E8M0_TO_FP32_HALF(e[4]),
                GGML_E8M0_TO_FP32_HALF(e[5]), GGML_E8M0_TO_FP32_HALF(e[6]), GGML_E8M0_TO_FP32_HALF(e[7]));
}

static void unpack_mxfp4_quants(uint8_t * qs, const block_mxfp4 * x, unsigned int bi) {
    static const int qk = QK_MXFP4;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0         = (x->qs[i] & 0x0F);
        const uint8_t x1         = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_mxfp4_quants(block_mxfp4 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_0;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

static void repack_row_mxfp4x4x2(uint8_t * y, const block_mxfp4 * x, int64_t k) {
    static const int qk = QK_MXFP4x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)
    const int        nloe = k % qk;           // leftovers

    const int eblk_size = 8 * 1;              // 8x E8M0
    const int qblk_size = qk / 2;             // int4
    const int qrow_size = k / 2;              // int4 (not padded to blocks)

    uint8_t * y_q = y + 0;                    // quants first
    uint8_t * y_e = y + qrow_size;            // then scales

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_block_mxfp4(&x[i * 8 + 0], 0);
            dump_block_mxfp4(&x[i * 8 + 1], 1);
            dump_block_mxfp4(&x[i * 8 + 2], 2);
            dump_block_mxfp4(&x[i * 8 + 3], 3);
            dump_block_mxfp4(&x[i * 8 + 4], 4);
            dump_block_mxfp4(&x[i * 8 + 5], 5);
            dump_block_mxfp4(&x[i * 8 + 6], 6);
            dump_block_mxfp4(&x[i * 8 + 7], 7);
        }
    }

    // Repack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_MXFP4x4x2];  // unpacked quants

        unpack_mxfp4_quants(qs, &x[i * 8 + 0], 0);
        unpack_mxfp4_quants(qs, &x[i * 8 + 1], 1);
        unpack_mxfp4_quants(qs, &x[i * 8 + 2], 2);
        unpack_mxfp4_quants(qs, &x[i * 8 + 3], 3);
        unpack_mxfp4_quants(qs, &x[i * 8 + 4], 4);
        unpack_mxfp4_quants(qs, &x[i * 8 + 5], 5);
        unpack_mxfp4_quants(qs, &x[i * 8 + 6], 6);
        unpack_mxfp4_quants(qs, &x[i * 8 + 7], 7);

        bool partial = (nloe && i == nb-1);

        uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            q[j] = partial ? (qs[j*2+1] << 4) | qs[j*2+0] : (qs[j+128] << 4) | qs[j+000];
        }
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_MXFP4x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Repack the scales
        uint8_t * e = (uint8_t *) (y_e + i * eblk_size);
        e[0]        = x[i * 8 + 0].e;
        e[1]        = x[i * 8 + 1].e;
        e[2]        = x[i * 8 + 2].e;
        e[3]        = x[i * 8 + 3].e;
        e[4]        = x[i * 8 + 4].e;
        e[5]        = x[i * 8 + 5].e;
        e[6]        = x[i * 8 + 6].e;
        e[7]        = x[i * 8 + 7].e;
    }

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_mxfp4x4x2(y, i, k);
        }
    }
}

static void unpack_row_mxfp4x4x2(block_mxfp4 * x, const uint8_t * y, int64_t k) {
    static const int qk = QK_MXFP4x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)
    const int        nloe = k % qk;           // leftovers

    const int eblk_size = 8 * 1;              // 8x E8M0
    const int qblk_size = qk / 2;             // int4
    const int qrow_size = k / 2;              // int4 (not padded to blocks)

    const uint8_t * y_q = y + 0;              // quants first
    const uint8_t * y_e = y + qrow_size;      // then scales

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_mxfp4x4x2(y, i, k);
        }
    }

    // Unpack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_MXFP4x4x2];  // unpacked quants

        bool partial = (nloe && i == nb-1);

        const uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            if (partial) {
                qs[j*2+0] = q[j] & 0xf;
                qs[j*2+1] = q[j] >> 4;
            } else {
                qs[j+000] = q[j] & 0xf;
                qs[j+128] = q[j] >> 4;
            }
        }

        pack_mxfp4_quants(&x[i * 8 + 0], qs, 0);
        pack_mxfp4_quants(&x[i * 8 + 1], qs, 1);
        pack_mxfp4_quants(&x[i * 8 + 2], qs, 2);
        pack_mxfp4_quants(&x[i * 8 + 3], qs, 3);
        pack_mxfp4_quants(&x[i * 8 + 4], qs, 4);
        pack_mxfp4_quants(&x[i * 8 + 5], qs, 5);
        pack_mxfp4_quants(&x[i * 8 + 6], qs, 6);
        pack_mxfp4_quants(&x[i * 8 + 7], qs, 7);
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_MXFP4_0x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Unpack the scales
        const uint8_t * e = (const uint8_t *) (y_e + i * eblk_size);
        x[i * 8 + 0].e    = e[0];
        x[i * 8 + 1].e    = e[1];
        x[i * 8 + 2].e    = e[2];
        x[i * 8 + 3].e    = e[3];
        x[i * 8 + 4].e    = e[4];
        x[i * 8 + 5].e    = e[5];
        x[i * 8 + 6].e    = e[6];
        x[i * 8 + 7].e    = e[7];
    }

    if (opt_verbose > 2) {
        for (int i = 0; i < nb; i++) {
            dump_block_mxfp4(&x[i * 8 + 0], 0);
            dump_block_mxfp4(&x[i * 8 + 1], 1);
            dump_block_mxfp4(&x[i * 8 + 2], 2);
            dump_block_mxfp4(&x[i * 8 + 3], 3);
            dump_block_mxfp4(&x[i * 8 + 4], 4);
            dump_block_mxfp4(&x[i * 8 + 5], 5);
            dump_block_mxfp4(&x[i * 8 + 6], 6);
            dump_block_mxfp4(&x[i * 8 + 7], 7);
        }
    }
}

static void init_row_mxfp4x4x2(block_mxfp4 * x, int64_t k) {
    static const int qk = QK_MXFP4x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)

    // Init the quants such that they unpack into zeros
    uint8_t qs[QK_MXFP4x4x2];  // unpacked quants
    memset(qs, 0, sizeof(qs));

    for (int i = 0; i < nb; i++) {
        pack_mxfp4_quants(&x[i * 8 + 0], qs, 0);
        pack_mxfp4_quants(&x[i * 8 + 1], qs, 1);
        pack_mxfp4_quants(&x[i * 8 + 2], qs, 2);
        pack_mxfp4_quants(&x[i * 8 + 3], qs, 3);
        pack_mxfp4_quants(&x[i * 8 + 4], qs, 4);
        pack_mxfp4_quants(&x[i * 8 + 5], qs, 5);
        pack_mxfp4_quants(&x[i * 8 + 6], qs, 6);
        pack_mxfp4_quants(&x[i * 8 + 7], qs, 7);
    }

    // Init the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_MXFP4x4x2)
    // the last block is truncated and overridden by the scales.
    for (int i = 0; i < nb; i++) {
        // Unpack the scales
        x[i * 8 + 0].e = 0;
        x[i * 8 + 1].e = 0;
        x[i * 8 + 2].e = 0;
        x[i * 8 + 3].e = 0;
        x[i * 8 + 4].e = 0;
        x[i * 8 + 5].e = 0;
        x[i * 8 + 6].e = 0;
        x[i * 8 + 7].e = 0;
    }
}

// repack mxfp4 data into mxfp4x4x2 tensor
static void repack_mxfp4_mxfp4x4x2(ggml_tensor * t, const void * data, size_t size) {
    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up(t->ne[0], QK_MXFP4x4x2));  // extra elements for the pad
    size_t row_size_rp = row_size * 2;  // extra space for tmp pad (if any)

    // Ensure we don't try to read more data than is available in the source buffer 'data'
    // or write more than the tensor can hold.
    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    // Calculate how many full rows and how many remaining bytes we need to process.
    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    HEX_VERBOSE("ggml-hex: repack-mxfp4-mxfp4x4x2 %s : data %p size %zu dims %ldx%ld row-size %zu\n", t->name, data,
                size, t->ne[0], nrows, row_size);

    init_row_mxfp4x4x2((block_mxfp4 *) buf_pd, t->ne[0]);  // init padded buffer to make sure the tail is all zeros

    // 1. Process all the full rows
    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) data + (i * row_size);
        uint8_t *       dst = (uint8_t *) t->data + (i * row_size);

        memcpy(buf_pd, src, row_size);
        repack_row_mxfp4x4x2((uint8_t *) buf_rp, (const block_mxfp4 *) buf_pd, t->ne[0]);
        memcpy(dst, buf_rp, row_size);
    }

    // 2. Process the final, potentially partial, row
    if (n_rem_bytes > 0) {
        const int64_t i = n_full_rows;
        const uint8_t * src = (const uint8_t *) data + (i * row_size);
        uint8_t *       dst = (uint8_t *) t->data + (i * row_size);

        // re-init the row because we are potentially copying a partial row
        init_row_mxfp4x4x2((block_mxfp4 *) buf_pd, t->ne[0]);

        // Copy only the remaining bytes from the source.
        memcpy(buf_pd, src, n_rem_bytes);

        // Repack the entire buffer (partial data + zero padding).
        repack_row_mxfp4x4x2((uint8_t *) buf_rp, (const block_mxfp4 *) buf_pd, t->ne[0]);

        // Write only the corresponding remaining bytes to the destination tensor.
        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
}

// repack mxfp4x4x2 tensor into mxfp4 data
static void repack_mxfp4x4x2_mxfp4(void * data, const ggml_tensor * t, size_t size) {
    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up(t->ne[0], QK_MXFP4x4x2));  // extra elements for the pad
    size_t row_size_rp = row_size * 2;  // extra space for tmp pad (if any)

    // Ensure we don't try to copy more data than the tensor actually contains.
    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    // Calculate how many full rows and how many remaining bytes we need to process.
    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    HEX_VERBOSE("ggml-hex: repack-mxfp4x4x2-mxfp4 %s : data %p size %zu dims %ldx%ld row-size %zu\n", t->name, data,
                size, t->ne[0], nrows, row_size);

    memset(buf_pd, 0, row_size_pd);  // clear-out padded buffer to make sure the tail is all zeros

    // 1. Process all the full rows
    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        memcpy(buf_pd, src, row_size);
        unpack_row_mxfp4x4x2((block_mxfp4 *) buf_rp, (const uint8_t *) buf_pd, t->ne[0]);
        memcpy(dst, buf_rp, row_size);
    }

    // 2. Process the final, potentially partial, row
    if (n_rem_bytes > 0) {
        const int64_t i = n_full_rows;
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        // We still need to read and unpack the entire source row because the format is block-based.
        memcpy(buf_pd, src, row_size);
        unpack_row_mxfp4x4x2((block_mxfp4 *) buf_rp, (const uint8_t *) buf_pd, t->ne[0]);

        // But we only copy the remaining number of bytes to the destination to respect the size limit.
        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
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
            repack_q4_0_q4x4x2(tensor, data, size);
            break;

        case GGML_TYPE_Q8_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q8_0_q8x4x2(tensor, data, size);
            break;

        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            // IQ4_NL has identical block layout to Q4_0 (ggml_half d + uint8_t qs[16])
            repack_q4_0_q4x4x2(tensor, data, size);
            break;

        case GGML_TYPE_MXFP4:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_mxfp4_mxfp4x4x2(tensor, data, size);
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
            repack_q4x4x2_q4_0(data, tensor, size);
            break;

        case GGML_TYPE_Q8_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q8x4x2_q8_0(data, tensor, size);
            break;

        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q4x4x2_q4_0(data, tensor, size);
            break;

        case GGML_TYPE_MXFP4:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_mxfp4x4x2_mxfp4(data, tensor, size);
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
    return ggml_nbytes(t);
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
    return 1UL * 1024 * 1024 * 1024;  // 1GB per buffer
    GGML_UNUSED(buffer_type);
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

// Backend session implementation

struct ggml_hexagon_opshm {
    ggml_hexagon_shared_buffer *sbuf;

    std::vector<bool> block_mask;
    size_t            block_size;

    uint8_t * base()     const { return this->sbuf->base; }
    int       fd()       const { return this->sbuf->fd;   }
    size_t    n_blocks() const { return this->block_mask.size(); }

    ggml_hexagon_opshm(ggml_hexagon_session *sess, size_t max_batch, size_t max_pending) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = max_batch;
        size_t n_tensors = n_ops + n_ops * HTP_OP_MAX_INPUTS;

        block_mask.resize(max_pending, true);

        block_size = sizeof(htp_buf_desc) * n_bufs    +
                     sizeof(htp_tensor)   * n_tensors +
                     sizeof(htp_op_desc)  * n_ops;

        sbuf = new ggml_hexagon_shared_buffer(sess, block_size * block_mask.size(), true /* pinned */);

        if (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated shared buf %zu : block-size %zu max-batch %zu max-pending %zu\n",
                    sess->c_name(), (size_t) sbuf->size, block_size, max_batch, max_pending);
        }
    }

    ~ggml_hexagon_opshm() {
        delete sbuf;
    }

    uint8_t * allocate() {
        auto it = std::find(block_mask.begin(), block_mask.end(), true);
        if (it == block_mask.end())
            return nullptr;

        unsigned int i = std::distance(block_mask.begin(), it);
        uint8_t*  addr = sbuf->base + (i * block_size);
        block_mask[i]  = false;

        HEX_VERBOSE("ggml-hex: %s allocated op shm #%u %p\n", sbuf->sess->c_name(), i, (void*) addr);
        return addr;
    }

    void release(uint8_t * addr) {
        int i = (addr - sbuf->base) / block_size;
        block_mask[i] = true;
        HEX_VERBOSE("ggml-hex: %s released op shm #%u %p\n", sbuf->sess->c_name(), i, (void*) addr);
    }
};

struct ggml_hexagon_opbatch {
    const char* name;

    std::vector<htp_buf_desc> buffers;
    std::vector<htp_tensor>   tensors;
    std::vector<htp_op_desc>  ops;

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

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t max_batch) {
        name = sess->c_name();

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = max_batch;
        n_tens_max = n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS;

        b_vmem_max = HTP_OP_MAX_VMEM;

        buffers.resize(n_bufs_max);
        tensors.resize(n_tens_max);
        ops.resize(n_ops_max);

        b_map.reserve(n_bufs_max);
        t_map.reserve(n_tens_max);
        d_map.reserve(n_tens_max);

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

        htp_buf_desc &b = buffers[bi];
        b.base = (uint64_t) sbuf->base;
        b.fd   = sbuf->fd;
        b.size = sbuf->size;

        b_vmem += b.size;

        HEX_VERBOSE("ggml-hex: add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", bi, b.fd, (void*) sbuf->base, (size_t) b.size, b_vmem);

        return bi;
    }

    bool same_shape(const htp_tensor * h, const ggml_tensor * t) const {
        return (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == t->nb[0]) && (h->nb[1] == t->nb[1]) && (h->nb[2] == t->nb[2]) && (h->nb[3] == t->nb[3]);
    }

    // add tensor and return its index
    int add_tensor(const ggml_tensor * t) {
        auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        for (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &tensors[it->second];
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

        htp_tensor &h = tensors[ti];
        h.bi    = add_buffer(sbuf);
        h.data  = t_offset;
        h.size  = t_size;
        h.type  = t->type;
        h.ne[0] = t->ne[0]; h.ne[1] = t->ne[1]; h.ne[2] = t->ne[2]; h.ne[3] = t->ne[3];
        h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];

        h.flags = 0;
        if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            h.flags |= HTP_TENSOR_COMPUTE;
        }

        HEX_VERBOSE("ggml-hex: add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n",
                ti, t->name, h.bi, (void*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) t->ne[0], (size_t) t->ne[1], (size_t) t->ne[2], (size_t) t->ne[3]);

        return ti;
    }

    bool fit_op(const struct ggml_tensor *t) const {
        if (n_ops >= n_ops_max ) return false;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        auto fit_tensor = [&](const ggml_tensor *t) {
            if (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                if (!b_map.count(sbuf->fd)) {
                    extra_vmem += sbuf->size;
                    extra_bufs += 1;
                }
            }
        };

        for (unsigned int i=0; i < HTP_OP_MAX_INPUTS && t->src[i]; i++) {
            fit_tensor(t->src[i]);
        }
        fit_tensor(t);

        if ((extra_bufs + n_bufs) > n_bufs_max) return false;
        if ((extra_tens + n_tens) > n_tens_max) return false;
        if ((extra_vmem + b_vmem) > b_vmem_max) return false;

        return true;
    }

    // assumes that fit_op() was called first and returned true
    void add_op(htp_op_code opcode, const struct ggml_tensor * t) {
        // Add new op
        htp_op_desc &o = ops[n_ops++];
        GGML_ASSERT(n_ops <= n_ops_max);

        memcpy(&o.params, &t->op_params, sizeof(t->op_params));
        o.opcode = opcode;
        o.flags  = 0;

        if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
            o.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
        }

        ggml_hexagon_dump_op_exec(name, t, o.flags);

        for (unsigned int i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = t->src[i] ? add_tensor(t->src[i]) : 0xffff;
        }
        o.dst = add_tensor(t);
    }

    size_t flush(uint8_t * mem_addr, size_t mem_size) {
        static_assert(sizeof(htp_buf_desc) % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(sizeof(htp_tensor)   % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(sizeof(htp_op_desc)  % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");

        const size_t b_size = sizeof(htp_buf_desc) * n_bufs;
        const size_t t_size = sizeof(htp_tensor)   * n_tens;
        const size_t o_size = sizeof(htp_op_desc)  * n_ops;

        const size_t m_size = b_size + t_size + o_size;
        GGML_ASSERT(m_size <= mem_size);

        uint8_t * b_ptr = (uint8_t *) mem_addr;
        uint8_t * t_ptr = (uint8_t *) b_ptr + b_size;
        uint8_t * o_ptr = (uint8_t *) t_ptr + t_size;

        memcpy(b_ptr, (void *) buffers.data(), b_size);
        memcpy(t_ptr, (void *) tensors.data(), t_size);
        memcpy(o_ptr, (void *) ops.data(),     o_size);

        HEX_VERBOSE("ggml-hex: %s flush-opbatch : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu\n",
                name, n_bufs, n_tens, n_ops, b_vmem, b_size, t_size, o_size);

        if (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            for (unsigned int i=0; i < n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", name, i,
                            b[i].fd, (void *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            for (unsigned int i=0; i < n_tens; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %u size %u : %zu:%zu:%zu:%zu\n",
                            name, i, t[i].bi, t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        reset();

        return m_size;
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
        int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, DSPQUEUE_TIMEOUT);
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

        op_shm->release((uint8_t*) dbuf.ptr);

        if (rsp.status != HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp: %s\n", this->c_name(), status_to_str(rsp.status));
            // TODO: handle errors
        }

        // FIXME: profile will be per opreq
        // this->prof_usecs  = rsp.prof_usecs;
        // this->prof_cycles = rsp.prof_cycles;
        // this->prof_pkts   = rsp.prof_pkts;

        this->op_pending--;  // atomic dec

        if (!all) break;
    }
}

void ggml_hexagon_session::flush_batch() {
    if (op_batch->empty()) { return; }

    htp_opbatch_req req;
    req.n_bufs    = op_batch->n_bufs;
    req.n_tensors = op_batch->n_tens;
    req.n_ops     = op_batch->n_ops;

    dspqueue_buffer dbuf;
    dbuf.fd     = op_shm->fd();
    dbuf.flags  = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    dbuf.ptr    = op_shm->allocate();
    if (!dbuf.ptr) {
        flush_pending(false);
        dbuf.ptr = op_shm->allocate();
    }

    dbuf.offset = (uint8_t*) dbuf.ptr - (uint8_t*) op_shm->base();
    dbuf.size   = op_batch->flush((uint8_t*) dbuf.ptr, op_shm->block_size);

    // Bump pending flag (cleared in the session::flush once we get the response)
    this->op_pending++;  // atomic inc

    HEX_VERBOSE("ggml-hex: %s: queue-opbatch : %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);

    int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
    if (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
    }
}

void ggml_hexagon_session::enqueue_op(htp_op_code opcode, const ggml_tensor *op) {
    if (!op_batch->fit_op(op)) {
        flush_batch();
    }
    op_batch->add_op(opcode, op);
}

// Flush HTP response queue i.e wait for all outstanding requests to complete
void ggml_hexagon_session::flush(bool all) {
    flush_batch();
    flush_pending(all);
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

    GGML_LOG_INFO("ggml-hex: allocating new session: %s\n", this->name.c_str());

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

    GGML_LOG_INFO("ggml-hex: new session: %s : session-id %d domain-id %d uri %s handle 0x%lx\n", this->name.c_str(),
                  this->session_id, this->domain_id, session_uri, (unsigned long) this->handle);

    // Enable FastRPC QoS mode
    {
        struct remote_rpc_control_latency l;
        l.enable = 1;

        int err = remote_handle64_control(this->handle, DSPRPC_CONTROL_LATENCY, (void *) &l, sizeof(l));
        if (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

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
        err = htp_iface_enable_etm(this->handle);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Start the DSP-side service. We need to pass the queue ID to the
    // DSP in a FastRPC call; the DSP side will import the queue and start
    // listening for packets in a callback.
    err = htp_iface_start(this->handle, dev_id, this->queue_id, opt_nhvx, opt_use_hmx);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: failed to start session: 0x%08x\n", (unsigned) err);
        throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    this->valid_iface = true;

    // Allocate buffers and state for op batching
    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch);
    this->op_shm   = new ggml_hexagon_opshm(this, opt_opbatch, opt_opqueue);
}

void ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", this->name.c_str());

    int err;

    delete this->op_batch;
    delete this->op_shm;

    // Stop the DSP-side service and close the queue
    if (this->valid_iface) {
        err = htp_iface_stop(this->handle);
        if (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    if (opt_etm) {
        err = htp_iface_disable_etm(this->handle);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
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
    op_shm   = nullptr;

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

static bool ggml_backend_buffer_is_hexagon(const struct ggml_backend_buffer * b) {
    return b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

static inline bool ggml_backend_buffer_is_hexagon_repack(const struct ggml_backend_buffer * b) {
    if (!opt_hostbuf) {
        return ggml_backend_buffer_is_hexagon(b);
    }
    return b->buft->iface.alloc_buffer == ggml_backend_hexagon_repack_buffer_type_alloc_buffer;
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

    if (dst->ne[2] != 1 || dst->ne[3] != 1) {
        // FA during prompt still needs work
        return false;
    }

    return true;
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
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            if (src0->ne[0] % 32) {
                return false;
            }

            if (ggml_nrows(src0) > 16 * 1024) {
                return false;  // typically the lm-head which would be too large for VTCM
            }

            if (ggml_nrows(src1) > 1024 || src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;  // no huge batches or broadcasting (for now)
            }

            // src0 (weights) must be repacked
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            break;

        case GGML_TYPE_F16:
            if (src0->nb[1] < src0->nb[0]) {
                GGML_LOG_DEBUG("ggml_hexagon_supported_mul_mat: permuted F16 src0 not supported\n");
                return false;
            }
            if (ggml_nrows(src1) > 1024) {
                return false;  // no huge batches (for now)
            }
            break;

        default:
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

    // TODO: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
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

    if ((mode & GGML_ROPE_TYPE_MROPE) || (mode & GGML_ROPE_TYPE_VISION)) {
        return false;
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
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(src2) ||
            !ggml_is_contiguous(dst)) {
            return false;
        }
    } else {
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            return false;
        }
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

    // TODO: add support for non-contiguous tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

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
        case GGML_OP_FLASH_ATTN_EXT: return HTP_OP_FLASH_ATTN_EXT;
        case GGML_OP_MUL_MAT:        return HTP_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID:     return HTP_OP_MUL_MAT_ID;
        case GGML_OP_MUL:            return HTP_OP_MUL;
        case GGML_OP_ADD:            return HTP_OP_ADD;
        case GGML_OP_ADD_ID:         return HTP_OP_ADD_ID;
        case GGML_OP_SUB:            return HTP_OP_SUB;
        case GGML_OP_DIV:            return HTP_OP_DIV;
        case GGML_OP_CPY:            return HTP_OP_CPY;
        case GGML_OP_CONT:           return HTP_OP_CPY;
        case GGML_OP_GET_ROWS:       return HTP_OP_GET_ROWS;
        case GGML_OP_SET_ROWS:       return HTP_OP_SET_ROWS;
        case GGML_OP_SUM_ROWS:       return HTP_OP_SUM_ROWS;
        case GGML_OP_ARGSORT:        return HTP_OP_ARGSORT;
        case GGML_OP_RMS_NORM:       return HTP_OP_RMS_NORM;
        case GGML_OP_SCALE:          return HTP_OP_SCALE;
        case GGML_OP_SQR:            return HTP_OP_SQR;
        case GGML_OP_SQRT:           return HTP_OP_SQRT;
        case GGML_OP_SOFT_MAX:       return HTP_OP_SOFTMAX;
        case GGML_OP_SSM_CONV:       return HTP_OP_SSM_CONV;
        case GGML_OP_ROPE:           return HTP_OP_ROPE;
        case GGML_OP_REPEAT:         return HTP_OP_REPEAT;
        case GGML_OP_CUMSUM:         return HTP_OP_CUMSUM;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU:     return HTP_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU:     return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_SIGMOID:  return HTP_OP_UNARY_SIGMOID;
                case GGML_UNARY_OP_NEG:      return HTP_OP_UNARY_NEG;
                case GGML_UNARY_OP_EXP:      return HTP_OP_UNARY_EXP;
                case GGML_UNARY_OP_SOFTPLUS: return HTP_OP_UNARY_SOFTPLUS;
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

static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * n = graph->nodes[i];
        if (op_is_compute(n)) {
            sess->enqueue_op(op_remap_to_htp(n), n);
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

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    const ggml_tensor * src0() const {
        return node->src[0];
    }

    const ggml_tensor * src1() const {
        return node->src[1];
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    bool stackable() const {
        switch (this->op()) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                return ggml_is_quantized(this->src0()->type);
            default:
                return false;
        }
    }

    bool same_input(const node_info& n) const {
        return n.src1() == this->src1();
    }
};

static std::vector<int> ggml_hexagon_graph_optimize_reorder(const std::vector<node_info> & nodes) {
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

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        node_info node = {
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
    /* .get_tensor_2d_async     = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
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
                    supp = ggml_hexagon_supported_unary(sess, op);
                    break;
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
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

        case GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
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

    if (!opt_arch) {
        int err = get_hex_arch_ver(CDSP_DOMAIN_ID, &opt_arch);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        }
    }

#if defined(__ANDROID__)
    if (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

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

static void ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned int) HTP_TYPE_Q4_0 == (unsigned int) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q8_0 == (unsigned int) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_MXFP4 == (unsigned int) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_IQ4_NL == (unsigned int) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");

    const char * str_verbose = getenv("GGML_HEXAGON_VERBOSE");
    const char * str_hostbuf = getenv("GGML_HEXAGON_HOSTBUF");
    const char * str_opmask  = getenv("GGML_HEXAGON_OPMASK");
    const char * str_opsync  = getenv("GGML_HEXAGON_OPSYNC");
    const char * str_opbatch = getenv("GGML_HEXAGON_OPBATCH");
    const char * str_opqueue = getenv("GGML_HEXAGON_OPQUEUE");
    const char * str_opfilter= getenv("GGML_HEXAGON_OPFILTER");
    const char * str_profile = getenv("GGML_HEXAGON_PROFILE");
    const char * str_etm     = getenv("GGML_HEXAGON_ETM");
    const char * str_nhvx    = getenv("GGML_HEXAGON_NHVX");
    const char * str_use_hmx = getenv("GGML_HEXAGON_USE_HMX");
    const char * str_ndev    = getenv("GGML_HEXAGON_NDEV");
    const char * str_arch    = getenv("GGML_HEXAGON_ARCH");

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter     = str_opfilter     ? new std::regex(str_opfilter, RE_ICASE) : NULL;
    opt_verbose      = str_verbose ? atoi(str_verbose) : 0;
    opt_hostbuf      = str_hostbuf ? atoi(str_hostbuf) : opt_hostbuf;
    opt_opmask       = str_opmask  ? strtoul(str_opmask, NULL, 0)  : opt_opmask;
    opt_opsync       = str_opsync  ? atoi(str_opsync)              : opt_opsync;
    opt_opbatch      = str_opbatch ? strtoul(str_opbatch, NULL, 0) : opt_opbatch;
    opt_opqueue      = str_opqueue ? strtoul(str_opqueue, NULL, 0) : opt_opqueue;
    opt_profile      = str_profile ? atoi(str_profile) : 0;
    opt_etm          = str_etm     ? atoi(str_etm)     : 0;
    opt_nhvx         = str_nhvx    ? strtoul(str_nhvx, NULL, 0) : opt_nhvx;
    opt_use_hmx      = str_use_hmx ? atoi(str_use_hmx) : opt_use_hmx;
    opt_ndev         = str_ndev    ? strtoul(str_ndev, NULL, 0) : opt_ndev;
    opt_hostbuf      = str_hostbuf ? atoi(str_hostbuf) : opt_hostbuf;

    if (opt_ndev > GGML_HEXAGON_MAX_SESSIONS) {
        opt_ndev = GGML_HEXAGON_MAX_SESSIONS;
    }

    if (str_arch) {
        if (str_arch[0] == 'v') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, NULL, 0);
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
