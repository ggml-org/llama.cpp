#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

#ifdef _WIN32
#    include <sal.h>
#    ifndef _WINDOWS
#        define _WINDOWS
#    endif
#else
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"

#include "htp-utils.h"

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "htp-msg.h"
#include "htp_iface.h"

static size_t opt_ndev         = 1;
static size_t opt_nhvx         = 0;  // use all
static int    opt_arch         = 0;  // autodetect
static int    opt_etm          = 0;
static int    opt_verbose      = 0;
static int    opt_profile      = 0;
static int    opt_hostbuf      = 1;
static int    opt_experimental = 0;

// Enable all stages by default
static int opt_opmask = HTP_OPMASK_QUEUE | HTP_OPMASK_QUANTIZE | HTP_OPMASK_COMPUTE;
static int opt_opsync = 0;  // synchronous ops

#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

#define HEX_PROFILE(...) \
    if (opt_profile) GGML_LOG_INFO(__VA_ARGS__)

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

static inline int hex_format_tensor_dims(char * str, const struct ggml_tensor * t) {
    if (t->ne[2] == 1 && t->ne[3] == 1) {
        return sprintf(str, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
    } else {
        return sprintf(str, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
    }
}

static inline void hex_format_op_dims(char * str, const struct ggml_tensor * t) {
    char * p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
        p += hex_format_tensor_dims(p, t->src[0]);

        for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
            p += sprintf(p, " x ");
            p += hex_format_tensor_dims(p, t->src[i]);
        }

        p += sprintf(p, " -> ");
    }

    // format self dims separately for better visual alignment
    char self[64];
    hex_format_tensor_dims(self, t);

    p += sprintf(p, "%s", self);
}

static inline int hex_format_tensor_strides(char * str, const struct ggml_tensor * t) {
    const char * c = ggml_is_contiguous(t) ? "" : "!";

    if (t->ne[2] == 1 && t->ne[3] == 1) {
        return sprintf(str, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
    } else {
        return sprintf(str, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2],
                       (size_t) t->nb[3], c);
    }
}

static inline void hex_format_op_strides(char * str, const struct ggml_tensor * t) {
    char * p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
        p += hex_format_tensor_strides(p, t->src[0]);

        for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
            p += sprintf(p, " x ");
            p += hex_format_tensor_strides(p, t->src[i]);
        }

        p += sprintf(p, " -> ");
    }

    // format self dims separately for better visual alignment
    char self[64];
    hex_format_tensor_strides(self, t);

    p += sprintf(p, "%s", self);
}

static inline void hex_format_op_types(char * str, const struct ggml_tensor * t) {
    char * p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
        p += sprintf(p, "%s", ggml_type_name(t->src[0]->type));

        for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
            p += sprintf(p, " x ");
            p += sprintf(p, "%s", ggml_type_name(t->src[i]->type));
        }

        p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%s", ggml_type_name(t->type));
}

static inline const char * hex_tensor_buff_name(const struct ggml_tensor * t) {
    if (t->buffer) {
        return ggml_backend_buffer_name(t->buffer);
    }
    return "NONE";
}

static inline void hex_format_op_buffs(char * str, const struct ggml_tensor * t) {
    char * p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
        p += sprintf(p, "%s", hex_tensor_buff_name(t->src[0]));

        for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
            p += sprintf(p, " x ");
            p += sprintf(p, "%s", hex_tensor_buff_name(t->src[i]));
        }

        p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%s", hex_tensor_buff_name(t));
}

static inline void hex_format_op_names(char * str, const struct ggml_tensor * t) {
    char * p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
        p += sprintf(p, "%s", t->src[0]->name);

        for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
            p += sprintf(p, " x ");
            p += sprintf(p, "%s", t->src[i]->name);
        }

        p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%s", t->name);
}

// ** backend sessions

struct ggml_hexagon_session {
    ggml_hexagon_session(int dev_id, ggml_backend_dev_t dev) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    void allocate(int dev_id) noexcept(false);
    void release() noexcept(true);

    void enqueue(struct htp_general_req &req, struct dspqueue_buffer *bufs, uint32_t n_bufs, bool sync = false);
    void flush();

    ggml_backend_buffer_type buffer_type;
    ggml_backend_buffer_type repack_buffer_type;

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
    std::atomic<int> op_pending;
    uint32_t         prof_usecs;
    uint32_t         prof_cycles;
    uint32_t         prof_pkts;
};

void ggml_hexagon_session::enqueue(struct htp_general_req &req, struct dspqueue_buffer *bufs, uint32_t n_bufs, bool sync) {
    // Bump pending flag (cleared in the session::flush once we get the responce)
    this->op_pending++;  // atomic inc

    int err = dspqueue_write(this->queue,
                             0,                       // flags - the framework will autoset this
                             n_bufs,                  // number of buffers
                             bufs,                    // buffer references
                             sizeof(req),
                             (const uint8_t *) &req,  // Message
                             1000000                  // Timeout
    );

    if (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->name.c_str(), (unsigned) err);
    }

    if (sync) {
        flush();
    }
}

// Flush HTP response queue i.e wait for all outstanding requests to complete
void ggml_hexagon_session::flush() {
    dspqueue_t q = this->queue;

    // Repeatedly read packets from the queue until it's empty. We don't
    // necessarily get a separate callback for each packet, and new packets
    // may arrive while we're processing the previous one.

    while (this->op_pending) {
        struct htp_general_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        struct dspqueue_buffer bufs[HTP_MAX_PACKET_BUFFERS];
        uint32_t               n_bufs;

        // Read response packet from queue
        int err = dspqueue_read(q, &flags,
                                   HTP_MAX_PACKET_BUFFERS,  // Maximum number of buffer references
                                   &n_bufs,                 // Number of buffer references
                                   bufs,                    // Buffer references
                                   sizeof(rsp),             // Max message length
                                   &rsp_size,               // Message length
                                   (uint8_t *) &rsp,
                                   1000000);                // Timeout

        if (err == AEE_EEXPIRED) {
            // TODO: might need to bail out if the HTP is stuck on something
            continue;
        }

        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        if (rsp_size != sizeof(rsp)) {
            GGML_ABORT("ggml-hex: dspcall : bad response (size)\n");
        }

        if (rsp.status != HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: dspcall : dsp-rsp: %s\n", status_to_str(rsp.status));
            // TODO: handle errors
        }

        // TODO: update profiling implementation, currently only works for opt_opsync mode
        this->prof_usecs  = rsp.prof_usecs;
        this->prof_cycles = rsp.prof_cycles;
        this->prof_pkts   = rsp.prof_pkts;

        this->op_pending--;  // atomic dec
    }
}

// ** backend buffers

struct ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(const std::string & name, ggml_hexagon_session * sess) {
        this->sess = sess;
        this->name = name;
    }

    ggml_hexagon_session * sess;
    std::string            name;
};

struct ggml_backend_hexagon_buffer_context {
    bool mmap_to(ggml_hexagon_session * s) {
        HEX_VERBOSE("ggml-hex: %s mmaping buffer: base %p domain-id %d session-id %d size %zu fd %d repack %d\n",
                    s->name.c_str(), (void *) this->base, s->domain_id, s->session_id, this->size, this->fd,
                    (int) this->repack);

        int err = fastrpc_mmap(s->domain_id, this->fd, (void *) this->base, 0, this->size, FASTRPC_MAP_FD);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n",
                    s->domain_id, this->size, this->fd, (unsigned) err);
            return false;
        }

        return true;
    }

    bool mmap() {
        if (this->mapped) {
            return true;
        }
        if (!mmap_to(this->sess)) {
            return false;
        }
        this->mapped = true;
        return true;
    }

    void munmap() {
        if (!this->mapped) {
            return;
        }

        fastrpc_munmap(this->sess->domain_id, this->fd, this->base, this->size);
        this->mapped = false;
    }

    ggml_backend_hexagon_buffer_context(ggml_hexagon_session * sess, size_t size, bool repack) {
        size += 4 * 1024;  // extra page for padding

        this->base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG, size);
        if (!this->base) {
            GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer : size %zu\n", sess->name.c_str(), size);
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed (see log for details)");
        }

        this->fd = rpcmem_to_fd(this->base);
        if (this->fd < 0) {
            GGML_LOG_ERROR("ggml-hex: %s failed to get FD for buffer %p\n", sess->name.c_str(), (void *) this->base);
            rpcmem_free(this->base);
            this->base = NULL;
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d repack %d\n", sess->name.c_str(),
                    (void *) this->base, size, this->fd, (int) repack);

        this->sess   = sess;
        this->size   = size;
        this->mapped = false;
        this->repack = repack;
    }

    ~ggml_backend_hexagon_buffer_context() {
        munmap();
        if (this->base) {
            rpcmem_free(this->base);
            this->base = NULL;
        }
    }

    ggml_hexagon_session * sess;  // primary session
    uint8_t *              base;
    size_t                 size;
    int                    fd;
    bool                   mapped;  // mmap is done
    bool                   repack;  // repacked buffer
};

static ggml_hexagon_session * ggml_backend_hexagon_buffer_get_sess(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer->buft->context)->sess;
}

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto ctx = static_cast<ggml_backend_hexagon_buffer_context *>(buffer->context);
    delete ctx;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto ctx = static_cast<ggml_backend_hexagon_buffer_context *>(buffer->context);
    return ctx->base;
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto ctx  = static_cast<ggml_backend_hexagon_buffer_context *>(buffer->context);
    auto sess = ctx->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu usage %d repack %d\n", sess->name.c_str(),
                tensor->name, (void *) ctx->base, tensor->data, ggml_nbytes(tensor), (int) buffer->usage,
                (int) ctx->repack);

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        ; // nothing to do for the view
    } else {
        if (!ctx->mapped) {
            ctx->mmap();
        }
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

        uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            q[j] = (qs[j + 128] << 4) | qs[j];
        }
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_Q4_0x4x2)
    // the last block is truncated and overriden by the scales.
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

    if (opt_verbose > 1) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_q4x4x2(y, i, k);
        }
    }
}

static void unpack_row_q4x4x2(block_q4_0 * x, const uint8_t * y, int64_t k) {
    static const int qk = QK_Q4_0x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)

    const int dblk_size = 8 * 2;              // 8x __fp16
    const int qblk_size = qk / 2;             // int4
    const int qrow_size = k / 2;              // int4 (not padded to blocks)

    const uint8_t * y_q = y + 0;              // quants first
    const uint8_t * y_d = y + qrow_size;      // then scales

    if (opt_verbose > 1) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_q4x4x2(y, i, k);
        }
    }

    // Unpack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_Q4_0x4x2];  // unpacked quants

        const uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            qs[j]       = q[j] & 0xf;
            qs[j + 128] = q[j] >> 4;
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
    // the last block is truncated and overriden by the scales.
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
    // the last block is truncated and overriden by the scales.
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
    // the last block is truncated and overriden by the scales.
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

    if (opt_verbose > 1) {
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

    if (opt_verbose > 1) {
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
    // the last block is truncated and overriden by the scales.
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
    // the last block is truncated and overriden by the scales.
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

        uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            q[j] = (qs[j + 128] << 4) | qs[j];
        }
    }

    // Repack the scales
    // Note: Do not combine with the loop above. For tensor sizes not multiple of 256 (QK_MXFP4x4x2)
    // the last block is truncated and overriden by the scales.
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

    if (opt_verbose > 1) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_mxfp4x4x2(y, i, k);
        }
    }
}

static void unpack_row_mxfp4x4x2(block_mxfp4 * x, const uint8_t * y, int64_t k) {
    static const int qk = QK_MXFP4x4x2;
    const int        nb = (k + qk - 1) / qk;  // number of blocks (padded)

    const int eblk_size = 8 * 1;              // 8x E8M0
    const int qblk_size = qk / 2;             // int4
    const int qrow_size = k / 2;              // int4 (not padded to blocks)

    const uint8_t * y_q = y + 0;              // quants first
    const uint8_t * y_e = y + qrow_size;      // then scales

    if (opt_verbose > 1) {
        for (int i = 0; i < nb; i++) {
            dump_packed_block_mxfp4x4x2(y, i, k);
        }
    }

    // Unpack the quants
    for (int i = 0; i < nb; i++) {
        uint8_t qs[QK_MXFP4x4x2];  // unpacked quants

        const uint8_t * q = y_q + (i * qblk_size);
        for (int j = 0; j < qk / 2; j++) {
            qs[j]       = q[j] & 0xf;
            qs[j + 128] = q[j] >> 4;
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
    // the last block is truncated and overriden by the scales.
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
    // the last block is truncated and overriden by the scales.
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
    auto ctx  = (ggml_backend_hexagon_buffer_context *) buffer->context;
    auto sess = ctx->sess;

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu\n", sess->name.c_str(), tensor->name, data,
                offset, size);

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
    auto ctx  = (ggml_backend_hexagon_buffer_context *) buffer->context;
    auto sess = ctx->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu\n", sess->name.c_str(), tensor->name, data,
                offset, size);

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
    auto ctx  = (ggml_backend_hexagon_buffer_context *) buffer->context;
    auto sess = ctx->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->name.c_str(), (void *) ctx->base, ctx->size);
    memset(ctx->base, value, ctx->size);
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
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
        ggml_backend_hexagon_buffer_context * ctx = new ggml_backend_hexagon_buffer_context(sess, size, false /*repack*/);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, ctx, size);
    } catch (std::exception const &exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer context: %s\n", sess->name.c_str(), exc.what());
        return nullptr;
    }
}

static ggml_backend_buffer_t ggml_backend_hexagon_repack_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
    try {
        ggml_backend_hexagon_buffer_context * ctx = new ggml_backend_hexagon_buffer_context(sess, size, true /*repack*/);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, ctx, size);
    } catch (std::exception const &exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer context: %s\n", sess->name.c_str(), exc.what());
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
    return 1 * 1024 * 1024 * 1024;  // 1GB per buffer
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
    this->prof_usecs  = 0;
    this->prof_cycles = 0;
    this->prof_pkts   = 0;

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
        this->session_id = n.session_id;
        this->domain_id  = n.effective_domain_id;
        this->valid_session = true;
    }

    // Get session URI
    char htp_uri[256];
    sprintf(htp_uri, "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

    char session_uri[256];
    {
        struct remote_rpc_get_uri u;
        u.session_id      = this->session_id;
        u.domain_name     = const_cast<char *>(CDSP_DOMAIN_NAME);
        u.domain_name_len = strlen(CDSP_DOMAIN_NAME);
        u.module_uri      = const_cast<char *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);

        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: failed to get URI for session %d : error 0x%x\n", dev_id, err);
            throw std::runtime_error("ggml-hex: remote_session_control(get-uri) failed (see log for details)");
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

    // Now let's setup the DSP queue
    err = dspqueue_create(this->domain_id,
                          0,              // Flags
                          128 * 1024,     // Request  queue size (in bytes)
                          64 * 1024,      // Response queue size (in bytes)
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
    err = htp_iface_start(this->handle, dev_id, this->queue_id, opt_nhvx);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: failed to start session: 0x%08x\n", (unsigned) err);
        throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    this->valid_iface = true;
}

void ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", this->name.c_str());

    int err;

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
    buffer_type.context        = nullptr;
    repack_buffer_type.context = nullptr;

    buffer_type.device         = dev;
    repack_buffer_type.device  = dev;

    try {
        allocate(dev_id);

        buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
        buffer_type.context = new ggml_backend_hexagon_buffer_type_context(this->name, this);

        repack_buffer_type.iface   = ggml_backend_hexagon_repack_buffer_type_interface;
        repack_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(this->name + "-REPACK", this);
    } catch (std::exception const &exc) {
        release();
        throw;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
    release();

    delete static_cast<ggml_backend_hexagon_buffer_type_context*>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context*>(repack_buffer_type.context);
}

// ** backend interface

static bool ggml_backend_buffer_is_hexagon(const struct ggml_backend_buffer * b) {
    return b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

static inline bool ggml_backend_buffer_is_hexagon_repack(const struct ggml_backend_buffer * b) {
    return b->buft->iface.alloc_buffer == ggml_backend_hexagon_repack_buffer_type_alloc_buffer;
}

static bool hex_supported_dims2(const struct ggml_tensor * x, const struct ggml_tensor * y) {
    if (x->ne[0] != y->ne[0]) {
        return false;
    }
    if (x->ne[1] != y->ne[1]) {
        return false;
    }
    if (x->ne[2] != y->ne[2]) {
        return false;
    }
    if (x->ne[3] != y->ne[3]) {
        return false;
    }

    return true;
}

static bool hex_supported_src0_type(ggml_type t) {
    return t == GGML_TYPE_F32;
}

static bool hex_supported_src1_type(ggml_type t) {
    return t == GGML_TYPE_F32;
}

static bool hex_supported_src2_type(ggml_type t) {
    return t == GGML_TYPE_F32;
}

static bool hex_supported_src1_type2(ggml_type t) {
    return t == GGML_TYPE_F16;
}

static bool hex_supported_src1_type3(ggml_type t) {
    return t == GGML_TYPE_I32;
}

static bool hex_supported_dst_type(ggml_type t) {
    return t == GGML_TYPE_F32;
}

static bool hex_supported_dims(const struct ggml_tensor * x, const struct ggml_tensor * y) {
    // TODO: support broadcast for ne[2 and 3]
    if (x->ne[0] != y->ne[0]) {
        return false;
    }
    if (x->ne[2] != y->ne[2]) {
        return false;
    }
    if (x->ne[3] != y->ne[3]) {
        return false;
    }
    return true;
}

static bool ggml_hexagon_supported_mul_mat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // TODO: add support for non-cont tensors
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            if (src0->ne[0] % 32) {
                return false;
            }

            if (src0->ne[1] > 16 * 1024) {
                return false;  // typically the lm-head which would be too large for VTCM
            }

            // if ((src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3])) return false;
            if ((src1->ne[2] != 1 || src1->ne[3] != 1)) {
                return false;
            }

            // src0 (weights) must be repacked
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            break;

        case GGML_TYPE_F16:
            if (!opt_experimental) {
                return false;
            }
            break;

        default:
            return false;
    }

    // src0 & src1 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (src1->buffer &&
        (!ggml_backend_buffer_is_hexagon(src1->buffer) || ggml_backend_hexagon_buffer_get_sess(src1->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
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
        case GGML_TYPE_MXFP4:
            if ((src0->ne[0] % 32)) {
                return false;
            }

            // src0 (weights) must be repacked
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            break;

        case GGML_TYPE_F16:
            if (!opt_experimental) {
                return false;
            }
            break;

        default:
            return false;
    }

    // TODO: add support for non-cont tensors
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // src0 (weights) must be repacked and mapped to the same session
    // src1 & sr2 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (src1->buffer &&
        (!ggml_backend_buffer_is_hexagon(src1->buffer) || ggml_backend_hexagon_buffer_get_sess(src1->buffer) != sess)) {
        return false;
    }
    if (src2->buffer &&
        (!ggml_backend_buffer_is_hexagon(src2->buffer) || ggml_backend_hexagon_buffer_get_sess(src2->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_binary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (!hex_supported_src0_type(src0->type)) {
        return false;
    }
    if (!hex_supported_src1_type(src1->type)) {
        return false;
    }
    if (!hex_supported_dst_type(dst->type)) {
        return false;
    }
    if (!hex_supported_dims2(src0, dst)) {
        return false;
    }
    if (!ggml_can_repeat(src1, src0)) {
        return false;
    }

    // TODO: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // src0, src1 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (src1->buffer &&
        (!ggml_backend_buffer_is_hexagon(src1->buffer) || ggml_backend_hexagon_buffer_get_sess(src1->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_add_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (!hex_supported_src0_type(src0->type)) {
        return false;
    }
    if (!hex_supported_src1_type(src1->type)) {
        return false;
    }
    if (!hex_supported_dst_type(dst->type)) {
        return false;
    }
    if (!hex_supported_dims2(src0, dst)) {
        return false;
    }

    // REVISIT: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // src0, src1 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (src1->buffer &&
        (!ggml_backend_buffer_is_hexagon(src1->buffer) || ggml_backend_hexagon_buffer_get_sess(src1->buffer) != sess)) {
        return false;
    }
    if (src2->buffer &&
        (!ggml_backend_buffer_is_hexagon(src2->buffer) || ggml_backend_hexagon_buffer_get_sess(src2->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_unary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (!hex_supported_src0_type(src0->type)) {
        return false;
    }
    if (!hex_supported_dst_type(dst->type)) {
        return false;
    }
    if (!hex_supported_dims2(src0, dst)) {
        return false;
    }

    // TODO: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // src0 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_activations(const struct ggml_hexagon_session * sess,
                                               const struct ggml_tensor *          op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (!hex_supported_src0_type(src0->type)) {
        return false;
    }
    if (!hex_supported_dst_type(dst->type)) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    if (src1) {
        if (!hex_supported_src1_type(src1->type)) {
            return false;
        }
        if (!hex_supported_dims2(src0, src1)) {
            return false;
        }
        if (!ggml_is_contiguous(src1)) {
            return false;
        }
    }

    // src0, src1 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (src1 && src1->buffer &&
        (!ggml_backend_buffer_is_hexagon(src1->buffer) || ggml_backend_hexagon_buffer_get_sess(src1->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
        return false;
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

    if (!hex_supported_src0_type(src0->type)) {
        return false;
    }
    if (!hex_supported_dst_type(dst->type)) {
        return false;
    }

    if (src1) {
        if (!hex_supported_src1_type(src1->type) && !hex_supported_src1_type2(src1->type)) {
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

    // src0, src1 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (src1 && src1->buffer &&
        (!ggml_backend_buffer_is_hexagon(src1->buffer) || ggml_backend_hexagon_buffer_get_sess(src1->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_rope(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const int32_t * op_params = &op->op_params[0];

    int mode = op_params[2];

    if ((mode & GGML_ROPE_TYPE_NEOX) || (mode & GGML_ROPE_TYPE_MROPE) || (mode & GGML_ROPE_TYPE_VISION)) {
        return false;
    }
    if (mode & 1) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (!hex_supported_src0_type(src0->type)) {
        return false;  // FIXME: add support for GGML_TYPE_F16 for src0
    }
    if (!hex_supported_dst_type(dst->type)) {
        return false;
    }
    if (!hex_supported_src1_type3(src1->type)) {
        return false;
    }
    if (src2) {
        if (!hex_supported_src2_type(src2->type)) {
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

    // src0, src1, src2 & dst must be mapped to the same session
    if (src0->buffer &&
        (!ggml_backend_buffer_is_hexagon(src0->buffer) || ggml_backend_hexagon_buffer_get_sess(src0->buffer) != sess)) {
        return false;
    }
    if (src1->buffer &&
        (!ggml_backend_buffer_is_hexagon(src1->buffer) || ggml_backend_hexagon_buffer_get_sess(src1->buffer) != sess)) {
        return false;
    }
    if (src2 && src2->buffer &&
        (!ggml_backend_buffer_is_hexagon(src2->buffer) || ggml_backend_hexagon_buffer_get_sess(src2->buffer) != sess)) {
        return false;
    }
    if (dst->buffer &&
        (!ggml_backend_buffer_is_hexagon(dst->buffer) || ggml_backend_hexagon_buffer_get_sess(dst->buffer) != sess)) {
        return false;
    }

    return true;
}

// Init hexagon tensor from GGML tensor and Hexagon buffer
static void init_htp_tensor(htp_tensor * h, const ggml_tensor * t) {
    h->data  = 0;  // updated by the receiver
    h->type  = t->type;
    h->ne[0] = t->ne[0];
    h->ne[1] = t->ne[1];
    h->ne[2] = t->ne[2];
    h->ne[3] = t->ne[3];
    h->nb[0] = t->nb[0];
    h->nb[1] = t->nb[1];
    h->nb[2] = t->nb[2];
    h->nb[3] = t->nb[3];
}

static void hex_dump_dspbuf(const struct ggml_tensor * t, const dspqueue_buffer * d) {
    auto buf  = static_cast<ggml_backend_hexagon_buffer_context *>(t->buffer->context);
    auto sess = buf->sess;

    HEX_VERBOSE("ggml-hex: %s dspqbuf : %s base-addr %p base-size %zu data %p offset %u size %u\n", sess->name.c_str(),
                t->name, (void *) buf->base, buf->size, (void *) d->ptr, (unsigned int) d->offset,
                (unsigned int) d->size);
}

static void ggml_hexagon_mul_mat(const struct ggml_tensor * op, uint32_t flags) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    auto src0_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src0->buffer->context);
    auto src1_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src1->buffer->context);
    auto dst_buf  = static_cast<ggml_backend_hexagon_buffer_context *>(dst->buffer->context);

    uint64_t t1, t2;
    t1 = ggml_time_us();

    // Construct HTP message
    htp_general_req req;
    req.op    = HTP_OP_MUL_MAT;
    req.flags = flags;

    init_htp_tensor(&req.src0, src0);
    init_htp_tensor(&req.src1, src1);
    init_htp_tensor(&req.dst, dst);

    // Use opmask to override flags
    if (!(opt_opmask & HTP_OPMASK_QUANTIZE)) {
        req.flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
    }
    if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
        req.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
    }

    dspqueue_buffer bufs[3];
    memset(bufs, 0, sizeof(bufs));

    // First buffer Weights.
    // The content is static, there is no need to do any cache management
    bufs[0].fd     = src0_buf->fd;
    bufs[0].ptr    = src0->data;
    bufs[0].offset = (uint8_t *) src0->data - src0_buf->base;
    bufs[0].size   = ggml_nbytes(src0);
    bufs[0].flags  = 0;

    // Second buffer Input Activations. This is a buffer that the CPU
    // writes and the DSP reads, so we'll need to flush CPU caches and
    // invalidate DSP ones. On platforms with I/O coherency support the
    // framework will automatically skip cache operations where possible.
    bufs[1].fd     = src1_buf->fd;
    bufs[1].ptr    = src1->data;
    bufs[1].offset = (uint8_t *) src1->data - src1_buf->base;
    bufs[1].size   = ggml_nbytes(src1);
    bufs[1].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP

    // Third buffer Output Activations. We'll handle DSP
    // cache maintenance in the response message but need to flush
    // CPU caches to ensure any previously written dirty lines are
    // written out before writes from the DSP start.
    bufs[2].fd     = dst_buf->fd;
    bufs[2].ptr    = dst->data;
    bufs[2].offset = (uint8_t *) dst->data - dst_buf->base;
    bufs[2].size   = ggml_nbytes(dst);
    bufs[2].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER);

    // Primary DSP session from the src0 (normally weight) tensor
    auto sess = src0_buf->sess;

    if (opt_verbose) {
        char dims[64 * GGML_MAX_SRC];
        char strides[64 * GGML_MAX_SRC];
        char types[16 * GGML_MAX_SRC];
        char buffs[64 * GGML_MAX_SRC];
        char names[64 * GGML_MAX_SRC];

        hex_format_op_dims(dims, op);
        hex_format_op_strides(strides, op);
        hex_format_op_types(types, op);
        hex_format_op_buffs(buffs, op);
        hex_format_op_names(names, op);

        HEX_VERBOSE("ggml-hex: %s %s: %s : %s : %s : %s : %s: flags 0x%x\n", sess->name.c_str(), ggml_op_name(op->op),
                    names, dims, types, strides, buffs, req.flags);
        if (opt_verbose > 1) {
            hex_dump_dspbuf(src0, &bufs[0]);
            hex_dump_dspbuf(src1, &bufs[1]);
            hex_dump_dspbuf(dst, &bufs[2]);
        }
    }

    if ((opt_opmask & HTP_OPMASK_QUEUE)) {
        sess->enqueue(req, bufs, 3, opt_opsync);
    }

    t2 = ggml_time_us();

    HEX_PROFILE(
        "ggml-hex: %s %s %s %u:%u:%u:%u x %s %u:%u:%u:%u -> %s %u:%u:%u:%u : op-usec %u op-cycles %u op-pkts %u (%f) "
        "call-usec %llu\n",
        sess->name.c_str(), ggml_op_name(op->op), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2], (uint32_t) src0->ne[3], src1->name, (uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2], (uint32_t) src1->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
        (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
}

static void ggml_hexagon_mul_mat_id(const struct ggml_tensor * op, uint32_t flags) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    auto src0_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src0->buffer->context);
    auto src1_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src1->buffer->context);
    auto src2_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src2->buffer->context);
    auto dst_buf  = static_cast<ggml_backend_hexagon_buffer_context *>(dst->buffer->context);

    uint64_t t1, t2;
    t1 = ggml_time_us();

    // Construct HTP message
    htp_general_req req;
    req.op    = HTP_OP_MUL_MAT_ID;
    req.flags = flags;

    init_htp_tensor(&req.src0, src0);
    init_htp_tensor(&req.src1, src1);
    init_htp_tensor(&req.src2, src2);
    init_htp_tensor(&req.dst, dst);

    // Use opmask to override flags
    if (!(opt_opmask & HTP_OPMASK_QUANTIZE)) {
        req.flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
    }
    if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
        req.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
    }

    dspqueue_buffer bufs[4];
    memset(bufs, 0, sizeof(bufs));

    // First buffer Weights.
    // The content is static, there is no need to do any cache management
    bufs[0].fd     = src0_buf->fd;
    bufs[0].ptr    = src0->data;
    bufs[0].offset = (uint8_t *) src0->data - src0_buf->base;
    bufs[0].size   = ggml_nbytes(src0);
    bufs[0].flags  = 0;

    // Second buffer Input Activations. This is a buffer that the CPU
    // writes and the DSP reads, so we'll need to flush CPU caches and
    // invalidate DSP ones. On platforms with I/O coherency support the
    // framework will automatically skip cache operations where possible.
    bufs[1].fd     = src1_buf->fd;
    bufs[1].ptr    = src1->data;
    bufs[1].offset = (uint8_t *) src1->data - src1_buf->base;
    bufs[1].size   = ggml_nbytes(src1);
    bufs[1].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP

    // Third buffer expert IDs. This is a buffer that the CPU
    // writes and the DSP reads, so we'll need to flush CPU caches and
    // invalidate DSP ones. On platforms with I/O coherency support the
    // framework will automatically skip cache operations where possible.
    bufs[2].fd     = src2_buf->fd;
    bufs[2].ptr    = src2->data;
    bufs[2].offset = (uint8_t *) src2->data - src2_buf->base;
    bufs[2].size   = ggml_nbytes(src2);
    bufs[2].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP

    // Forth buffer Output Activations. We'll handle DSP
    // cache maintenance in the response message but need to flush
    // CPU caches to ensure any previously written dirty lines are
    // written out before writes from the DSP start.
    bufs[3].fd     = dst_buf->fd;
    bufs[3].ptr    = dst->data;
    bufs[3].offset = (uint8_t *) dst->data - dst_buf->base;
    bufs[3].size   = ggml_nbytes(dst);
    bufs[3].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER);

    // Primary DSP session from the src0 (normally weight) tensor
    auto sess = src0_buf->sess;

    if (opt_verbose) {
        char dims[64 * GGML_MAX_SRC];
        char strides[64 * GGML_MAX_SRC];
        char types[16 * GGML_MAX_SRC];
        char buffs[64 * GGML_MAX_SRC];
        char names[64 * GGML_MAX_SRC];

        hex_format_op_dims(dims, op);
        hex_format_op_types(types, op);
        hex_format_op_buffs(buffs, op);
        hex_format_op_names(names, op);

        HEX_VERBOSE("ggml-hex: %s %s: %s : %s : %s : %s : %s: flags 0x%x\n", sess->name.c_str(), ggml_op_name(op->op),
                    names, dims, types, strides, buffs, req.flags);

        if (opt_verbose > 1) {
            hex_dump_dspbuf(src0, &bufs[0]);
            hex_dump_dspbuf(src1, &bufs[1]);
            hex_dump_dspbuf(src2, &bufs[2]);
            hex_dump_dspbuf(dst, &bufs[3]);
        }
    }

    if ((opt_opmask & HTP_OPMASK_QUEUE)) {
        sess->enqueue(req, bufs, 4, opt_opsync);
    }

    t2 = ggml_time_us();

    HEX_PROFILE(
        "ggml-hex: %s matmul-id %s %u:%u:%u:%u x %s %u:%u:%u:%u (%s %u:%u:%u:%u) -> %s %u:%u:%u:%u : op-usec %u "
        "op-cycles %u op-pkts %u (%f) call-usec %llu\n",
        sess->name.c_str(), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) src0->ne[3], src1->name, (uint32_t) src1->ne[0], (uint32_t) src1->ne[1], (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3], src2->name, (uint32_t) src2->ne[0], (uint32_t) src2->ne[1], (uint32_t) src2->ne[2],
        (uint32_t) src2->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
        (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
}

static void ggml_hexagon_binary(const struct ggml_tensor * op, uint32_t flags) {
    const struct ggml_tensor * node = op;
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const struct ggml_tensor * dst  = node;

    auto src0_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src0->buffer->context);
    auto src1_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src1->buffer->context);
    auto dst_buf  = static_cast<ggml_backend_hexagon_buffer_context *>(dst->buffer->context);

    uint64_t t1 = 0;
    uint64_t t2 = 0;

    t1 = ggml_time_us();

    // Construct HTP message
    htp_general_req req;
    req.flags = flags;

    // Use opmask to override flags
    if (!(opt_opmask & HTP_OPMASK_QUANTIZE)) {
        req.flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
    }
    if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
        req.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
    }

    switch (node->op) {
        case GGML_OP_MUL:
            req.op = HTP_OP_MUL;
            break;
        case GGML_OP_ADD:
            req.op = HTP_OP_ADD;
            break;
        case GGML_OP_SUB:
            req.op = HTP_OP_SUB;
            break;
        default:
            GGML_ABORT("ggml-hex: binary : unsupported op:%d\n", node->op);
    }

    init_htp_tensor(&req.src0, src0);
    init_htp_tensor(&req.src1, src1);
    init_htp_tensor(&req.dst, dst);

    dspqueue_buffer bufs[3];
    memset(bufs, 0, sizeof(bufs));

    // First buffer = First Operand of Binary op
    // This is a buffer that the CPU writes and the DSP reads, so we'll
    // need to flush CPU caches and invalidate DSP ones. On platforms
    // with I/O coherency support the framework will automatically skip
    // cache operations where possible.
    bufs[0].fd     = src0_buf->fd;
    bufs[0].ptr    = src0->data;
    bufs[0].offset = (uint8_t *) src0->data - src0_buf->base;
    bufs[0].size   = ggml_nbytes(src0);
    bufs[0].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP;

    // Second buffer = Second Operand of Binary op
    // This is a buffer that the CPU writes and the DSP reads, so we'll
    // need to flush CPU caches and invalidate DSP ones. On platforms
    // with I/O coherency support the framework will automatically skip
    // cache operations where possible.
    bufs[1].fd     = src1_buf->fd;
    bufs[1].ptr    = src1->data;
    bufs[1].offset = (uint8_t *) src1->data - src1_buf->base;
    bufs[1].size   = ggml_nbytes(src1);
    bufs[1].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP

    // Third buffer = Output Activations. We'll handle DSP
    // cache maintenance in the response message but need to flush
    // CPU caches to ensure any previously written dirty lines are
    // written out before writes from the DSP start.
    bufs[2].fd     = dst_buf->fd;
    bufs[2].ptr    = dst->data;
    bufs[2].offset = (uint8_t *) dst->data - dst_buf->base;
    bufs[2].size   = ggml_nbytes(dst);
    bufs[2].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER);

    // Primary DSP session from the src0 tensor
    ggml_hexagon_session * sess = src0_buf->sess;

    if (opt_verbose) {
        char dims[64 * GGML_MAX_SRC];
        char strides[16 * GGML_MAX_SRC];
        char types[16 * GGML_MAX_SRC];
        char buffs[64 * GGML_MAX_SRC];
        char names[64 * GGML_MAX_SRC];

        hex_format_op_dims(dims, op);
        hex_format_op_strides(strides, op);
        hex_format_op_types(types, op);
        hex_format_op_buffs(buffs, op);
        hex_format_op_names(names, op);

        HEX_VERBOSE("ggml-hex: %s %s : %s : %s : %s : %s : %s : flags 0x%x\n", sess->name.c_str(),
                    ggml_op_name(node->op), names, dims, types, strides, buffs, req.flags);
        if (opt_verbose > 1) {
            hex_dump_dspbuf(src0, &bufs[0]);
            hex_dump_dspbuf(src1, &bufs[1]);
            hex_dump_dspbuf(dst, &bufs[2]);
        }
    }

    if ((opt_opmask & HTP_OPMASK_QUEUE)) {
        sess->enqueue(req, bufs, 3, opt_opsync);
    }

    t2 = ggml_time_us();

    HEX_PROFILE(
        "ggml-hex: %s %s %s %u:%u:%u:%u x %s %u:%u:%u:%u -> %s %u:%u:%u:%u : op-usec %u op-cycles %u op-pkts %u (%f) "
        "call-usec %llu\n",
        sess->name.c_str(), ggml_op_name(node->op), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2], (uint32_t) src0->ne[3], src1->name, (uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2], (uint32_t) src1->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
        (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
}

static void ggml_hexagon_add_id(const struct ggml_tensor * op, uint32_t flags) {
    const struct ggml_tensor * node = op;
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const struct ggml_tensor * src2 = node->src[2];
    const struct ggml_tensor * dst  = node;

    auto src0_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src0->buffer->context);
    auto src1_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src1->buffer->context);
    auto src2_buf = static_cast<ggml_backend_hexagon_buffer_context *>(src2->buffer->context);
    auto dst_buf  = static_cast<ggml_backend_hexagon_buffer_context *>(dst->buffer->context);

    uint64_t t1 = 0;
    uint64_t t2 = 0;

    t1 = ggml_time_us();

    // Construct HTP message
    htp_general_req req;
    req.flags = flags;

    // Use opmask to override flags
    if (!(opt_opmask & HTP_OPMASK_QUANTIZE)) {
        req.flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
    }
    if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
        req.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
    }

    switch (node->op) {
        case GGML_OP_ADD_ID:
            req.op = HTP_OP_ADD_ID;
            break;
        default:
            GGML_ABORT("ggml-hex: unsupported op:%d\n", node->op);
    }

    init_htp_tensor(&req.src0, src0);
    init_htp_tensor(&req.src1, src1);
    init_htp_tensor(&req.src2, src2);
    init_htp_tensor(&req.dst, dst);

    dspqueue_buffer bufs[4];
    memset(bufs, 0, sizeof(bufs));

    // First buffer = input activations
    bufs[0].fd     = src0_buf->fd;
    bufs[0].ptr    = src0->data;
    bufs[0].offset = (uint8_t *) src0->data - src0_buf->base;
    bufs[0].size   = ggml_nbytes(src0);
    bufs[0].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP;

    // Second buffer = experts bias
    bufs[1].fd     = src1_buf->fd;
    bufs[1].ptr    = src1->data;
    bufs[1].offset = (uint8_t *) src1->data - src1_buf->base;
    bufs[1].size   = ggml_nbytes(src1);
    bufs[1].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP

    // Third buffer = activated experts
    bufs[2].fd     = src2_buf->fd;
    bufs[2].ptr    = src2->data;
    bufs[2].offset = (uint8_t *) src2->data - src2_buf->base;
    bufs[2].size   = ggml_nbytes(src2);
    bufs[2].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP

    // Forth buffer = output activations
    bufs[3].fd     = dst_buf->fd;
    bufs[3].ptr    = dst->data;
    bufs[3].offset = (uint8_t *) dst->data - dst_buf->base;
    bufs[3].size   = ggml_nbytes(dst);
    bufs[3].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER);

    // Primary DSP session from the src0 tensor
    ggml_hexagon_session * sess = src0_buf->sess;

    if (opt_verbose) {
        char dims[64 * GGML_MAX_SRC];
        char strides[16 * GGML_MAX_SRC];
        char types[16 * GGML_MAX_SRC];
        char buffs[64 * GGML_MAX_SRC];
        char names[64 * GGML_MAX_SRC];

        hex_format_op_dims(dims, op);
        hex_format_op_strides(strides, op);
        hex_format_op_types(types, op);
        hex_format_op_buffs(buffs, op);
        hex_format_op_names(names, op);

        HEX_VERBOSE("ggml-hex: %s %s : %s : %s : %s : %s : %s : flags 0x%x\n", sess->name.c_str(),
                    ggml_op_name(node->op), names, dims, types, strides, buffs, req.flags);

        if (opt_verbose > 1) {
            hex_dump_dspbuf(src0, &bufs[0]);
            hex_dump_dspbuf(src1, &bufs[1]);
            hex_dump_dspbuf(src2, &bufs[2]);
            hex_dump_dspbuf(dst, &bufs[3]);
        }
    }

    if ((opt_opmask & HTP_OPMASK_QUEUE)) {
        sess->enqueue(req, bufs, 4, opt_opsync);
    }

    t2 = ggml_time_us();

    HEX_PROFILE(
        "ggml-hex: %s %s %s %u:%u:%u:%u x %s %u:%u:%u:%u -> %s %u:%u:%u:%u : op-usec %u op-cycles %u op-pkts %u (%f) "
        "call-usec %llu\n",
        sess->name.c_str(), ggml_op_name(node->op), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2], (uint32_t) src0->ne[3], src1->name, (uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2], (uint32_t) src1->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
        (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
}

static void ggml_hexagon_unary(const struct ggml_tensor * op, uint32_t flags) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    uint64_t t1 = 0;
    uint64_t t2 = 0;

    t1 = ggml_time_us();

    // Construct HTP message
    htp_general_req req;

    memset(&req, 0, sizeof(htp_general_req));
    memcpy(&req.op_params, &op->op_params, sizeof(op->op_params));
    req.flags = flags;

    bool supported = false;

    switch (op->op) {
        case GGML_OP_RMS_NORM:
            req.op    = HTP_OP_RMS_NORM;
            supported = true;
            break;

        case GGML_OP_UNARY:
            if (ggml_get_unary_op(dst) == GGML_UNARY_OP_SILU) {
                req.op    = HTP_OP_UNARY_SILU;
                supported = true;
            }
            break;

        case GGML_OP_GLU:
            if (ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU) {
                req.op    = HTP_OP_GLU_SWIGLU;
                supported = true;
            } else if (ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU_OAI) {
                req.op    = HTP_OP_GLU_SWIGLU_OAI;
                supported = true;
            }
            break;

        case GGML_OP_SOFT_MAX:
            req.op    = HTP_OP_SOFTMAX;
            supported = true;

        default:
            break;
    }

    if (!supported) {
        GGML_ABORT("ggml-hex: unary : unsupported op:%d\n", op->op);
    }

    init_htp_tensor(&req.dst, dst);
    init_htp_tensor(&req.src0, src0);
    if (src1) {
        init_htp_tensor(&req.src1, src1);
    }

    // Use opmask to override flags
    if (!(opt_opmask & HTP_OPMASK_QUANTIZE)) {
        req.flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
    }
    if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
        req.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
    }

    dspqueue_buffer bufs[3];
    int             n_bufs = 0;

    memset(bufs, 0, sizeof(bufs));

    // First buffer = Only Operand of Unary op
    // This is a buffer that the CPU writes and the DSP reads, so we'll
    // need to flush CPU caches and invalidate DSP ones. On platforms
    // with I/O coherency support the framework will automatically skip
    // cache operations where possible.
    auto src0_buf       = static_cast<ggml_backend_hexagon_buffer_context *>(src0->buffer->context);
    bufs[n_bufs].fd     = src0_buf->fd;
    bufs[n_bufs].ptr    = src0->data;
    bufs[n_bufs].offset = (uint8_t *) src0->data - src0_buf->base;
    bufs[n_bufs].size   = ggml_nbytes(src0);
    bufs[n_bufs].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                          DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP;
    ++n_bufs;

    if (src1) {
        // Second buffer = Second Operand of Binary op
        // This is a buffer that the CPU writes and the DSP reads, so we'll
        // need to flush CPU caches and invalidate DSP ones. On platforms
        // with I/O coherency support the framework will automatically skip
        // cache operations where possible.
        auto src1_buf       = static_cast<ggml_backend_hexagon_buffer_context *>(src1->buffer->context);
        bufs[n_bufs].fd     = src1_buf->fd;
        bufs[n_bufs].ptr    = src1->data;
        bufs[n_bufs].offset = (uint8_t *) src1->data - src1_buf->base;
        bufs[n_bufs].size   = ggml_nbytes(src1);
        bufs[n_bufs].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                              DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP
        ++n_bufs;
    }

    // Second or third buffer = Output Activations. We'll handle DSP
    // Second buffer = Output Activations. We'll handle DSP
    // cache maintenance in the response message but need to flush
    // CPU caches to ensure any previously written dirty lines are
    // written out before writes from the DSP start.
    auto dst_buf        = static_cast<ggml_backend_hexagon_buffer_context *>(dst->buffer->context);
    bufs[n_bufs].fd     = dst_buf->fd;
    bufs[n_bufs].ptr    = dst->data;
    bufs[n_bufs].offset = (uint8_t *) dst->data - dst_buf->base;
    bufs[n_bufs].size   = ggml_nbytes(dst);
    bufs[n_bufs].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER);
    ++n_bufs;

    // Primary DSP session from the src0 tensor
    ggml_hexagon_session * sess = src0_buf->sess;

    if (opt_verbose) {
        char dims[64 * GGML_MAX_SRC];
        char strides[64 * GGML_MAX_SRC];
        char types[16 * GGML_MAX_SRC];
        char buffs[64 * GGML_MAX_SRC];
        char names[64 * GGML_MAX_SRC];

        hex_format_op_dims(dims, op);
        hex_format_op_strides(strides, op);
        hex_format_op_types(types, op);
        hex_format_op_buffs(buffs, op);
        hex_format_op_names(names, op);

        HEX_VERBOSE("ggml-hex: %s %s : %s : %s : %s : %s : %s : flags 0x%x\n", sess->name.c_str(), ggml_op_name(op->op),
                    names, dims, types, strides, buffs, req.flags);
        if (opt_verbose > 1) {
            hex_dump_dspbuf(src0, &bufs[0]);
            if (src1) {
                hex_dump_dspbuf(src1, &bufs[1]);
                hex_dump_dspbuf(dst, &bufs[2]);
            } else {
                hex_dump_dspbuf(dst, &bufs[1]);
            }
        }
    }

    if ((opt_opmask & HTP_OPMASK_QUEUE)) {
        sess->enqueue(req, bufs, n_bufs, opt_opsync);
    }

    t2 = ggml_time_us();

    if (src1) {
        HEX_PROFILE(
            "ggml-hex: %s %s %s %u:%u:%u:%u x %s %u:%u:%u:%u -> %s %u:%u:%u:%u : op-usec %u op-cycles %u op-pkts %u "
            "(%f) call-usec %llu\n",
            sess->name.c_str(), ggml_op_name(op->op), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
            (uint32_t) src0->ne[2], (uint32_t) src0->ne[3], src1->name, (uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
            (uint32_t) src1->ne[2], (uint32_t) src1->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
            (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
            (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
    } else {
        HEX_PROFILE(
            "ggml-hex: %s %s %s %u:%u:%u:%u -> %s %u:%u:%u:%u : op-usec %u op-cycles %u op-pkts %u (%f) call-usec "
            "%llu\n",
            sess->name.c_str(), ggml_op_name(op->op), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
            (uint32_t) src0->ne[2], (uint32_t) src0->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
            (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
            (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
    }
}

static void ggml_hexagon_rope(const struct ggml_tensor * op, uint32_t flags) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    uint64_t t1 = 0;
    uint64_t t2 = 0;

    t1 = ggml_time_us();

    // Construct HTP message
    htp_general_req req;

    memset(&req, 0, sizeof(htp_general_req));
    memcpy(&req.op_params, &op->op_params, sizeof(op->op_params));
    req.flags = flags;
    req.op    = HTP_OP_ROPE;

    init_htp_tensor(&req.dst, dst);
    init_htp_tensor(&req.src0, src0);
    init_htp_tensor(&req.src1, src1);
    if (src2) {
        init_htp_tensor(&req.src2, src2);
    }

    // Use opmask to override flags
    if (!(opt_opmask & HTP_OPMASK_QUANTIZE)) {
        req.flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
    }
    if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
        req.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
    }

    dspqueue_buffer bufs[4];
    int             n_bufs = 0;

    memset(bufs, 0, sizeof(bufs));

    // First buffer
    // This is a buffer that the CPU writes and the DSP reads, so we'll
    // need to flush CPU caches and invalidate DSP ones. On platforms
    // with I/O coherency support the framework will automatically skip
    // cache operations where possible.
    auto src0_buf       = static_cast<ggml_backend_hexagon_buffer_context *>(src0->buffer->context);
    bufs[n_bufs].fd     = src0_buf->fd;
    bufs[n_bufs].ptr    = src0->data;
    bufs[n_bufs].offset = (uint8_t *) src0->data - src0_buf->base;
    bufs[n_bufs].size   = ggml_nbytes(src0);
    bufs[n_bufs].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                          DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP;
    ++n_bufs;

    // Second buffer
    // This is a buffer that the CPU writes and the DSP reads, so we'll
    // need to flush CPU caches and invalidate DSP ones. On platforms
    // with I/O coherency support the framework will automatically skip
    // cache operations where possible.
    auto src1_buf       = static_cast<ggml_backend_hexagon_buffer_context *>(src1->buffer->context);
    bufs[n_bufs].fd     = src1_buf->fd;
    bufs[n_bufs].ptr    = src1->data;
    bufs[n_bufs].offset = (uint8_t *) src1->data - src1_buf->base;
    bufs[n_bufs].size   = ggml_nbytes(src1);
    bufs[n_bufs].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                          DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP
    ++n_bufs;

    if (src2) {
        // Third buffer
        // This is a buffer that the CPU writes and the DSP reads, so we'll
        // need to flush CPU caches and invalidate DSP ones. On platforms
        // with I/O coherency support the framework will automatically skip
        // cache operations where possible.
        auto src2_buf       = static_cast<ggml_backend_hexagon_buffer_context *>(src2->buffer->context);
        bufs[n_bufs].fd     = src2_buf->fd;
        bufs[n_bufs].ptr    = src2->data;
        bufs[n_bufs].offset = (uint8_t *) src2->data - src2_buf->base;
        bufs[n_bufs].size   = ggml_nbytes(src2);
        bufs[n_bufs].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |         // Flush CPU
                              DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate DSP
        ++n_bufs;
    }

    // Final buffer = Output Activations. We'll handle DSP
    // Second buffer = Output Activations. We'll handle DSP
    // cache maintenance in the response message but need to flush
    // CPU caches to ensure any previously written dirty lines are
    // written out before writes from the DSP start.
    auto dst_buf        = static_cast<ggml_backend_hexagon_buffer_context *>(dst->buffer->context);
    bufs[n_bufs].fd     = dst_buf->fd;
    bufs[n_bufs].ptr    = dst->data;
    bufs[n_bufs].offset = (uint8_t *) dst->data - dst_buf->base;
    bufs[n_bufs].size   = ggml_nbytes(dst);
    bufs[n_bufs].flags  = (DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER);
    ++n_bufs;

    // Primary DSP session from the src0 tensor
    ggml_hexagon_session * sess = src0_buf->sess;

    if (opt_verbose) {
        char dims[64 * GGML_MAX_SRC];
        char strides[64 * GGML_MAX_SRC];
        char types[16 * GGML_MAX_SRC];
        char buffs[64 * GGML_MAX_SRC];
        char names[64 * GGML_MAX_SRC];

        hex_format_op_dims(dims, op);
        hex_format_op_strides(strides, op);
        hex_format_op_types(types, op);
        hex_format_op_buffs(buffs, op);
        hex_format_op_names(names, op);

        HEX_VERBOSE("ggml-hex: %s %s : %s : %s : %s : %s : %s : flags 0x%x\n", sess->name.c_str(), ggml_op_name(op->op),
                    names, dims, types, strides, buffs, req.flags);
        if (opt_verbose > 1) {
            hex_dump_dspbuf(src0, &bufs[0]);
            if (src1) {
                hex_dump_dspbuf(src1, &bufs[1]);
                hex_dump_dspbuf(dst, &bufs[2]);
            } else {
                hex_dump_dspbuf(dst, &bufs[1]);
            }
        }
    }

    if ((opt_opmask & HTP_OPMASK_QUEUE)) {
        sess->enqueue(req, bufs, n_bufs, opt_opsync);
    }

    t2 = ggml_time_us();

    if (src2) {
        HEX_PROFILE(
            "ggml-hex: %s %s %s %u:%u:%u:%u x %s %u:%u:%u:%u x %s %u:%u:%u:%u -> %s %u:%u:%u:%u : op-usec %u op-cycles "
            "%u op-pkts %u (%f) call-usec %llu\n",
            sess->name.c_str(), ggml_op_name(op->op), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
            (uint32_t) src0->ne[2], (uint32_t) src0->ne[3], src1->name, (uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
            (uint32_t) src1->ne[2], (uint32_t) src1->ne[3], src2->name, (uint32_t) src2->ne[0], (uint32_t) src2->ne[1],
            (uint32_t) src2->ne[2], (uint32_t) src2->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
            (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
            (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
    } else {
        HEX_PROFILE(
            "ggml-hex: %s %s %s %u:%u:%u:%u x %s %u:%u:%u:%u -> %s %u:%u:%u:%u : op-usec %u op-cycles %u op-pkts %u "
            "(%f) call-usec %llu\n",
            sess->name.c_str(), ggml_op_name(op->op), src0->name, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
            (uint32_t) src0->ne[2], (uint32_t) src0->ne[3], src1->name, (uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
            (uint32_t) src1->ne[2], (uint32_t) src1->ne[3], dst->name, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
            (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], sess->prof_usecs, sess->prof_cycles, sess->prof_pkts,
            (float) sess->prof_cycles / sess->prof_pkts, (unsigned long long) t2 - t1);
    }
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    return sess->name.c_str();
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

static inline bool op_reuse_src1(const ggml_tensor * op1, const ggml_tensor * op0) {
    return (op0 && op0->src[1] == op1->src[1]);
}

// scan the graph and figure out last compute op index
static inline int last_compute_op(ggml_cgraph * graph) {
    int last;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GLU:
            case GGML_OP_ADD_ID:
                last = i;
                break;

            default:
                break;
        }
    }

    return last;
}

static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->name.c_str(), graph->n_nodes);

    const int last = last_compute_op(graph);

    const struct ggml_tensor * prev_quant_op = nullptr;  // prev executed op with quantizer

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];

        uint32_t flags = 0;

        // skip quantizer if src1 is reused
        if (op_reuse_src1(node, prev_quant_op)) {
            flags |= HTP_OPFLAGS_SKIP_QUANTIZE;
        }

        // ask for early notification for the last Op
        if (i == last) {
            flags |= HTP_OPFLAGS_EARLY_WAKEUP;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_hexagon_mul_mat(node, flags);
                prev_quant_op = node;
                break;
            case GGML_OP_MUL_MAT_ID:
                ggml_hexagon_mul_mat_id(node, flags);
                prev_quant_op = node;
                break;
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
                ggml_hexagon_binary(node, flags);
                break;
            case GGML_OP_ADD_ID:
                ggml_hexagon_add_id(node, flags);
                break;
            case GGML_OP_RMS_NORM:
                ggml_hexagon_unary(node, flags);
                break;
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
                    ggml_hexagon_unary(node, flags);
                }
                break;
            case GGML_OP_GLU:
                if ((ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU) ||
                    (ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU_OAI)) {
                    ggml_hexagon_unary(node, flags);
                }
                break;
            case GGML_OP_SOFT_MAX:
                ggml_hexagon_unary(node, flags);
                break;

            case GGML_OP_ROPE:
                ggml_hexagon_rope(node, flags);
                break;

            // non-compute ops
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(node));
        }
    }

    // Wait until all pending ops complete
    sess->flush();

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->name.c_str());

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

    // TODO: the current version might do incorrect reodering in cases where quantized src0
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
        constexpr int N_FORWARD = 8;

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
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
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
    return sess->name.c_str();

    GGML_UNUSED(dev);
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    return "Hexagon";
    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // ~2GB per session for now
    *free  = 2ULL * 1024 * 1024 * 1024;
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

static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);

    bool supp = false;

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        case GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            break;

        case GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            break;

        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
            supp = ggml_hexagon_supported_binary(sess, op);
            break;

        case GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            break;

        case GGML_OP_RMS_NORM:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            break;

        case GGML_OP_UNARY:
            if (ggml_get_unary_op(op) == GGML_UNARY_OP_SILU) {
                supp = ggml_hexagon_supported_activations(sess, op);
            }
            break;

        case GGML_OP_GLU:
            if ((ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU) /* || (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) */) {
                supp = ggml_hexagon_supported_activations(sess, op);
            }
            break;

        case GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            break;

        default:
            break;
    }

    if (opt_verbose) {
        char dims[64 * GGML_MAX_SRC];
        char strides[64 * GGML_MAX_SRC];
        char types[16 * GGML_MAX_SRC];
        char buffs[64 * GGML_MAX_SRC];
        char names[64 * GGML_MAX_SRC];

        hex_format_op_dims(dims, op);
        hex_format_op_strides(strides, op);
        hex_format_op_types(types, op);
        hex_format_op_buffs(buffs, op);
        hex_format_op_names(names, op);

        HEX_VERBOSE("ggml-hex: %s device-supports-op %s : %s : %s : %s : %s : %s : (%d)\n", sess->name.c_str(),
                    ggml_op_name(op->op), names, dims, types, strides, buffs, (int) supp);
    }

    return supp;

    GGML_UNUSED(dev);
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

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d\n", opt_arch);

    // Create devices / sessions
    for (size_t i = 0; i < opt_ndev; i++) {
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        try {
            devices[i].context = new ggml_hexagon_session(i, &devices[i]);
        } catch (std::exception const &exc) {
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
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
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

    const char * str_verbose = getenv("GGML_HEXAGON_VERBOSE");
    const char * str_hostbuf = getenv("GGML_HEXAGON_HOSTBUF");

    opt_verbose      = str_verbose ? atoi(str_verbose) : 0;
    opt_profile      = getenv("GGML_HEXAGON_PROFILE") != nullptr;
    opt_etm          = getenv("GGML_HEXAGON_ETM") != nullptr;
    opt_experimental = getenv("GGML_HEXAGON_EXPERIMENTAL") != nullptr;

    const char * str_opmask = getenv("GGML_HEXAGON_OPMASK");
    if (str_opmask != nullptr) {
        opt_opmask = strtoul(str_opmask, NULL, 0);
    }
    opt_opsync = getenv("GGML_HEXAGON_OPSYNC") != nullptr;

    const char * str_ndev = getenv("GGML_HEXAGON_NDEV");
    if (str_ndev) {
        opt_ndev = strtoul(str_ndev, NULL, 0);
        if (opt_ndev > GGML_HEXAGON_MAX_SESSIONS) {
            opt_ndev = GGML_HEXAGON_MAX_SESSIONS;
        }
    }

    const char * str_nhvx = getenv("GGML_HEXAGON_NHVX");
    if (str_nhvx) {
        opt_nhvx = strtoul(str_nhvx, NULL, 0);
    }

    const char * str_arch = getenv("GGML_HEXAGON_ARCH");
    if (str_arch) {
        if (str_arch[0] == 'v') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, NULL, 0);
    }

    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) : 1;

    reg->context = new ggml_hexagon_registry(reg);

    HEX_VERBOSE("ggml-hex: size-of-general-req %zu size-of-general-rsp %zu\n", sizeof(struct htp_general_req),
                sizeof(struct htp_general_rsp));
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
            ggml_hexagon_init(&reg);
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
