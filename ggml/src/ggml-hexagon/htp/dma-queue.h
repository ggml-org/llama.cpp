#ifndef HTP_DMA_H
#define HTP_DMA_H

#include <HAP_farf.h>
#include <hexagon_types.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "hex-utils.h"

#include "hex-profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Define the HW descriptor structs here since the ones in HexSDK are a bit out of date
typedef struct dma_descriptor_1d_s {
    void *   next;
    uint32_t size:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
} dma_descriptor_1d;

#if __HVX_ARCH__ < 75

typedef struct dma_descriptor_2d_s {
    void *   next;
    uint32_t reserved0:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
    uint32_t reserved1:24;
    uint32_t row_size:16;
    uint32_t nrows:16;
    uint32_t src_stride:16;
    uint32_t dst_stride:16;
    uint32_t src_offset:16;
    uint32_t dst_offset:16;
} dma_descriptor_2d;

#else

typedef struct dma_descriptor_2d_s {
    void *   next;
    uint32_t dst_stride:24;
    uint32_t desc_size:2;
    uint32_t dst_comp:1;
    uint32_t src_comp:1;
    uint32_t dst_bypass:1;
    uint32_t src_bypass:1;
    uint32_t order:1;
    uint32_t done:1;
    uint32_t src;
    uint32_t dst;
    uint32_t desc_type:8;
#if __HVX_ARCH__ > 79
    uint32_t src_upper:8;
    uint32_t dst_upper:8;
    uint32_t allocation:2;
    uint32_t reserved0:2;
    uint32_t transform:4;
#else
    uint32_t reserved0:24;
#endif
    uint32_t row_size:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t offset:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#endif

#if __HVX_ARCH__ > 79
typedef uint64_t dma_addr_t;
#else
typedef uint32_t dma_addr_t;
#endif

typedef struct {
    dma_addr_t dst;
    dma_addr_t src;
} dma_data;

// Hardware descriptor field limits
#define DMA_MAX_NROWS          0xFFFFu        // 16-bit HW descriptor limit (65535)
#define DMA_MAX_SIZE_16B       0xFFFFu        // 16-bit HW descriptor limit for row_size (65535)
#define DMA_MAX_STRIDE_16B     0xFFFFu        // 16-bit HW descriptor limit for strides (65535)
#define DMA_MAX_SIZE_24B       0x00FFFFFFu    // 24-bit HW descriptor limit for row_size / 1D size (16MB - 1)
#define DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
#define DMA_SAFE_CHUNK_SIZE    0x00F00000u    // ~15MB safe contiguous chunk size

#define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring

typedef struct dma_ring_s dma_ring;
struct dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_data          * data;      // dst/src data
    uint32_t            push_idx;
    uint32_t            pop_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    struct htp_thread_trace * trace;
};

typedef struct dma_queue_s dma_queue;
typedef dma_queue * dma_queue_t;

struct dma_queue_s {
    dma_ring *          ring0;     // Main descriptor ring state
    dma_ring *          ring1;     // Secondary fallback descriptor ring state
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(void);
dma_queue_t dma_queue_init(void * ptr, size_t capacity, struct htp_thread_trace * trace);
void        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(void);
dma_queue_t dma_queue_alias_init(void * ptr, dma_queue_t main_q);
void        dma_queue_alias_free(dma_queue_t q);

bool        dma_queue_push_fallback_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
bool        dma_queue_push_fallback_contig(dma_queue * q, dma_data ddata, size_t total);
#if __HVX_ARCH__ < 75
bool        dma_queue_push_fallback_1d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows);
#endif

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
static inline void dmstart(void * next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmstart(%0)" : : "r"(next));
}

static inline void dmlink(void * cur, void * next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

static inline unsigned int dmpoll(void) {
    unsigned int ret = 0;
    asm volatile(" %0 = dmpoll" : "=r"(ret) : : "memory");
    return ret;
}

static inline unsigned int dmwait(void) {
    unsigned int ret = 0;
    asm volatile(" %0 = dmwait" : "=r"(ret) : : "memory");
    return ret;
}

static inline dma_data dma_make_data_impl(dma_addr_t dst, dma_addr_t src)
{
    dma_data d = { dst, src };
    return d;
}

#define dma_make_data(dst, src) dma_make_data_impl((dma_addr_t) (dst), (dma_addr_t) (src))

static inline bool dma_ring_push_single_1d(dma_ring * r, dma_data ddata, size_t size) {
#if __HVX_ARCH__ > 79
    assert(!((ddata.src | ddata.dst) >> 32) || size == 0);
#endif

    if (((r->push_idx + 1) & r->idx_mask) == r->pop_idx) {
        return false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->push_idx];
    desc->src  = (uint32_t) ddata.src;
    desc->dst  = (uint32_t) ddata.dst;
    desc->size = size;

    r->data[r->push_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->push_idx);

    if (size) {
        desc->next       = NULL;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = 1;
        desc->dst_bypass = 1;
        desc->order      = 0;
        desc->done       = 0;

        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } else {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->push_idx = (r->push_idx + 1) & r->idx_mask;
    return true;
}

static inline bool dma_ring_push_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
#if __HVX_ARCH__ > 79
    const uint32_t src_hi = (uint32_t) (ddata.src >> 32);
    const uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
    const bool is_ext     = (src_hi | dst_hi) != 0;

    if (is_ext && ((ddata.src >> 40) || (ddata.dst >> 40))) {
        return false;
    }
#endif

    if (((r->push_idx + 1) & r->idx_mask) == r->pop_idx) {
        return false;
    }

    dma_descriptor_2d * desc = &r->desc[r->push_idx];

    desc->next           = NULL;
    desc->reserved1      = 0;
    desc->desc_size      = 1; // 2d mode
    desc->src_bypass     = 1;
    desc->dst_bypass     = 1;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->order          = 0;
    desc->done           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (uint32_t) ddata.src;
    desc->dst            = (uint32_t) ddata.dst;
    desc->row_size       = row_size;

#if __HVX_ARCH__ < 75
    desc->reserved0      = 0;
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#else
#if __HVX_ARCH__ > 79
    desc->src_upper      = src_hi;
    desc->dst_upper      = dst_hi;
    desc->allocation     = 0;
    desc->reserved0      = 0;
    desc->transform      = 0;
    desc->desc_type      = is_ext ? 10 : 9; // 2d 40-bit or 24-bit mode
#else
    desc->reserved0      = 0;
    desc->desc_type      = 9; // 2d (24-bit) mode
#endif
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->data[r->push_idx] = ddata;

    htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->push_idx);

    if (nrows) {
        dmlink(r->tail, desc);
        r->tail = desc;
    } else {
        desc->done = 1;
    }

    r->push_idx = (r->push_idx + 1) & r->idx_mask;
    return true;
}

static inline dma_data dma_ring_pop(dma_ring * r) {
    dma_data ddata = { 0 };

    if (r->push_idx == r->pop_idx) {
        return ddata;
    }

    ddata = r->data[r->pop_idx];

    volatile dma_descriptor_2d * desc = &r->desc[r->pop_idx];

    // Wait for desc to complete
    if (!desc->done) {
        // FARF(ALWAYS, "dma-poll: idx %u dst %p src %p", r->pop_idx, ddata.dst, ddata.src);
        while (!desc->done) {
            dmpoll();
        }
    }

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->pop_idx);

    r->pop_idx = (r->pop_idx + 1) & r->idx_mask;
    return ddata;
}

static inline dma_data dma_ring_pop_nowait(dma_ring * r) {
    dma_data ddata = { 0 };

    if (r->push_idx == r->pop_idx) {
        return ddata;
    }

    ddata = r->data[r->pop_idx];

    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->pop_idx);

    r->pop_idx = (r->pop_idx + 1) & r->idx_mask;
    return ddata;
}

static inline bool dma_ring_empty(dma_ring * r) {
    return r->push_idx == r->pop_idx;
}

static inline void dma_ring_flush(dma_ring * r) {
    while (!dma_ring_empty(r)) {
        dma_ring_pop(r);
    }
}

static inline uint32_t dma_ring_depth(dma_ring * r) {
    return (r->push_idx - r->pop_idx) & r->idx_mask;
}

static inline uint32_t dma_ring_capacity(dma_ring * r) {
    return r->capacity;
}

static inline bool dma_queue_push_single_1d(dma_queue * q, dma_data ddata, size_t size) {
    return dma_ring_push_single_1d(q->ring0, ddata, size);
}

static inline bool dma_queue_push_single_2d(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    return dma_ring_push_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
}

static inline dma_data dma_queue_pop(dma_queue * q) {
    return dma_ring_pop(q->ring0);
}

static inline dma_data dma_queue_pop_nowait(dma_queue * q) {
    return dma_ring_pop_nowait(q->ring0);
}

static inline bool dma_queue_empty(dma_queue * q) {
    return dma_ring_empty(q->ring0);
}

static inline void dma_queue_flush(dma_queue * q) {
    dma_ring_flush(q->ring0);
}

static inline uint32_t dma_queue_depth(dma_queue * q) {
    return dma_ring_depth(q->ring0);
}

static inline uint32_t dma_queue_capacity(dma_queue * q) {
    return dma_ring_capacity(q->ring0);
}

#if __HVX_ARCH__ < 75

static inline bool dma_queue_push(dma_queue *q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    // Fast path: everything fits in 16 bits
    if (nrows == 0 || __builtin_expect(
            nrows      <= DMA_MAX_NROWS &&
            row_size   <= DMA_MAX_SIZE_16B &&
            src_stride <= DMA_MAX_STRIDE_16B &&
            dst_stride <= DMA_MAX_STRIDE_16B, 1)) {
        return dma_ring_push_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // Contiguous block: 1D DMA mode supports up to 24-bit size (16MB)
    if (nrows == 1 || (row_size == src_stride && row_size == dst_stride)) {
        size_t total = row_size * nrows;
        if (total <= DMA_MAX_SIZE_24B) {
            return dma_ring_push_single_1d(q->ring0, ddata, total);
        }
        return dma_queue_push_fallback_contig(q, ddata, total);
    }

    // Row count overflow with 16-bit strides: chunk 2D descriptors via fallback ring
    if (row_size <= DMA_MAX_SIZE_16B && src_stride <= DMA_MAX_STRIDE_16B && dst_stride <= DMA_MAX_STRIDE_16B) {
        return dma_queue_push_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // Stride or row_size overflow: row-by-row 1D via fallback ring
    return dma_queue_push_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
}

#else // HVX_ARCH >= 75

static inline bool dma_queue_push(dma_queue *q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    if (nrows == 0 || __builtin_expect(
            nrows      <= DMA_MAX_NROWS &&
            row_size   <= DMA_MAX_SIZE_24B &&
            src_stride <= DMA_MAX_STRIDE_24B &&
            dst_stride <= DMA_MAX_STRIDE_24B, 1)) {
        return dma_ring_push_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
    }

    // Contiguous block exceeding 24 bits
    if (nrows == 1 || (row_size == src_stride && row_size == dst_stride)) {
        size_t total = row_size * nrows;
        return dma_queue_push_fallback_contig(q, ddata, total);
    }

    return dma_queue_push_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
}

#endif

#define DMA_CACHE_MAX_SIZE 256U

typedef struct {
    uint8_t *base;
    uint32_t line_size;
    uint32_t capacity;
    dma_addr_t src[DMA_CACHE_MAX_SIZE];
    uint16_t age[DMA_CACHE_MAX_SIZE];
} dma_cache;

static inline void dma_cache_init(dma_cache *c, uint8_t *base, uint32_t line_size, uint32_t capacity)
{
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->line_size = line_size;

    for (unsigned i=0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

static inline bool dma_cache_push(dma_queue *q, dma_cache *c, dma_addr_t src_addr, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    for (unsigned i=0; i < c->capacity; i++) {
        if (c->src[i] == src_addr) {
            c->age[i] = 0;
            dst = c->base + (i * c->line_size); nrows = 0; // dummy dma
        } else {
            c->age[i]++;
            if (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    if (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = src_addr;
        dst = c->base + o_idx * c->line_size; // normal nrows dma
        return dma_queue_push(q, dma_make_data(dst, src_addr), dst_stride, src_stride, row_size, nrows);
    }

    return dma_queue_push_single_1d(q, dma_make_data(dst, src_addr), 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
