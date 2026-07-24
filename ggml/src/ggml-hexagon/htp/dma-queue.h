#ifndef HTP_DMA_H
#define HTP_DMA_H

#include <HAP_farf.h>
#include <hexagon_types.h>
#include <stdbool.h>
#include <stdint.h>
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
    void *   src;
    void *   dst;
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
    void *   src;
    void *   dst;
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
    void *   src;
    void *   dst;
    uint32_t desc_type:8;
    uint32_t reserved0:24;
    uint32_t row_size:24;
    uint32_t nrows_lo:8;
    uint32_t nrows_hi:8;
    uint32_t src_stride:24;
    uint32_t offset:24;
    uint32_t reserved1:8;
} dma_descriptor_2d;

#endif

typedef struct {
    void       *dst;
    const void *src;
} dma_ptr;

typedef struct dma_ring_s dma_ring;
struct dma_ring_s {
    dma_descriptor_2d * desc;      // descriptor pointers
    dma_descriptor_2d * tail;      // tail pointer
    dma_ptr           * dptr;      // dst/src pointers
    uint32_t            push_idx;
    uint32_t            pop_idx;
    uint32_t            capacity;
    uint32_t            idx_mask;
    struct htp_thread_trace * trace;
    uintptr_t           vtcm_base;
    uintptr_t           vtcm_end;
};

typedef struct dma_queue_s dma_queue;
typedef dma_queue * dma_queue_t;

struct dma_queue_s {
    dma_ring *          ring;      // Points to the descriptor ring state
    uint8_t             nocache;   // Queue-specific bypass flag
    bool                alias;     // When set, dma_queue_delete will not free the ring
};

void        dma_queue_flush(dma_queue_t q);

size_t      dma_queue_sizeof(size_t capacity);
size_t      dma_queue_alignof(void);
dma_queue_t dma_queue_init(void * ptr, size_t capacity, uintptr_t vtcm_base, size_t vtcm_size, struct htp_thread_trace * trace);
void        dma_queue_free(dma_queue_t q);

size_t      dma_queue_alias_sizeof(void);
dma_queue_t dma_queue_alias_init(void * ptr, dma_queue_t main_q, uint8_t nocache);
void        dma_queue_alias_free(dma_queue_t q);

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

static inline dma_ptr dma_make_ptr(void *dst, const void *src)
{
    dma_ptr p = { dst, src };
    return p;
}

static inline bool dma_is_vtcm(const dma_queue * q, const void * ptr) {
    return (uintptr_t) ptr >= q->ring->vtcm_base && (uintptr_t) ptr < q->ring->vtcm_end;
}

static inline bool dma_queue_push_single_1d(dma_queue * q, dma_ptr dptr, size_t size) {
    dma_ring * r = q->ring;
    if (((r->push_idx + 1) & r->idx_mask) == r->pop_idx) {
        FARF(HIGH, "dma-push: queue full\n");
        return false;
    }

    dma_descriptor_1d * desc = (dma_descriptor_1d *) &r->desc[r->push_idx];
    desc->src  = (void *) dptr.src;
    desc->dst  = (void *) dptr.dst;
    desc->size = size;

    r->dptr[r->push_idx] = dptr;

    if (size) {
        desc->next       = NULL;
        desc->desc_size  = 0; // 1D mode
        desc->src_bypass = dma_is_vtcm(q, dptr.src) ? 1 : q->nocache;
        desc->dst_bypass = dma_is_vtcm(q, dptr.dst) ? 1 : q->nocache;
        desc->order      = 0;
        desc->done       = 0;

        htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->push_idx);
        dmlink(r->tail, desc);
        r->tail = (dma_descriptor_2d *) desc;
    } else {
        desc->desc_size = 0;
        desc->done      = 1;
    }

    r->push_idx = (r->push_idx + 1) & r->idx_mask;
    return true;
}

static inline bool dma_queue_push_single_2d(dma_queue * q, dma_ptr dptr, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    dma_ring * r = q->ring;
    if (((r->push_idx + 1) & r->idx_mask) == r->pop_idx) {
        FARF(HIGH, "dma-push: queue full\n");
        return false;
    }

    dma_descriptor_2d * desc = &r->desc[r->push_idx];

    desc->next           = NULL;
    desc->reserved0      = 0;
    desc->reserved1      = 0;
    desc->desc_size      = 1; // 2d mode
    desc->src_bypass     = dma_is_vtcm(q, dptr.src) ? 1 : q->nocache;
    desc->dst_bypass     = dma_is_vtcm(q, dptr.dst) ? 1 : q->nocache;
    desc->src_comp       = 0;
    desc->dst_comp       = 0;
    desc->order          = 0;
    desc->done           = 0;
    desc->src_stride     = src_stride;
    desc->dst_stride     = dst_stride;
    desc->src            = (void *) dptr.src;
    desc->dst            = (void *) dptr.dst;
    desc->row_size       = row_size;

#if __HVX_ARCH__ < 75
    desc->desc_type      = 0; // 2d (16-bit) mode
    desc->nrows          = nrows;
    desc->src_offset     = 0;
    desc->dst_offset     = 0;
#else
    desc->desc_type      = 9; // 2d (24-bit) mode
    desc->nrows_lo       = (nrows & 0xff);
    desc->nrows_hi       = (nrows >> 8);
    desc->offset         = 0;
#endif

    r->dptr[r->push_idx] = dptr;

    if (nrows) {
        htp_trace_event_start(r->trace, HTP_TRACE_EVT_DMA, r->push_idx);
        dmlink(r->tail, desc);
        r->tail = desc;
    } else {
        desc->done = 1;
    }

    r->push_idx = (r->push_idx + 1) & r->idx_mask;
    return true;
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ring * r = q->ring;
    dma_ptr dptr  = { NULL };

    if (r->push_idx == r->pop_idx) {
        return dptr;
    }

    dma_descriptor_2d * desc = &r->desc[r->pop_idx];

    // Wait for desc to complete
    if (!desc->done) {
        while (!desc->done) {
            dmpoll();
        }
    }
    htp_trace_event_stop(r->trace, HTP_TRACE_EVT_DMA, r->pop_idx);

    dptr = r->dptr[r->pop_idx];

    r->pop_idx = (r->pop_idx + 1) & r->idx_mask;
    return dptr;
}

static inline dma_ptr dma_queue_pop_nowait(dma_queue * q) {
    dma_ring * r = q->ring;
    dma_ptr dptr  = { NULL };

    if (r->push_idx == r->pop_idx) {
        return dptr;
    }

    dptr = r->dptr[r->pop_idx];

    r->pop_idx = (r->pop_idx + 1) & r->idx_mask;
    return dptr;
}

static inline bool dma_queue_empty(dma_queue * q) {
    return q->ring->push_idx == q->ring->pop_idx;
}

static inline uint32_t dma_queue_depth(dma_queue * q) {
    return (q->ring->push_idx - q->ring->pop_idx) & q->ring->idx_mask;
}

static inline uint32_t dma_queue_capacity(dma_queue * q) {
    return q->ring->capacity;
}

#if __HVX_ARCH__ < 75

// Overflow-safe DMA push: all 2d descriptor fields (row_size, nrows, src_stride, dst_stride) are 16-bit, max 65535.
// This version transparently handles values that exceed the 16-bit limit and submits chained DMA transtions.

#define DMA_MAX_FIELD_VAL 65535u

static inline bool dma_queue_push(dma_queue *q, dma_ptr dptr, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    // Fast path: everything fits in 16 bits
    if (nrows == 0 || __builtin_expect(
            row_size   <= DMA_MAX_FIELD_VAL &&
            nrows      <= DMA_MAX_FIELD_VAL &&
            src_stride <= DMA_MAX_FIELD_VAL &&
            dst_stride <= DMA_MAX_FIELD_VAL, 1)) {
        return dma_queue_push_single_2d(q, dptr, dst_stride, src_stride, row_size, nrows);
    }

    // Contiguous block
    // Use 1d DMA mode which supports sizes up to 24-bits (16MB)
    if (nrows == 1 || (row_size == src_stride && row_size == dst_stride)) {
        size_t total = row_size * nrows;
        return dma_queue_push_single_1d(q, dptr, total);
    }

    // Stride overflow - fall back to row-by-row.
    {
        const uint8_t *src = (const uint8_t *) dptr.src;
        uint8_t       *dst = (uint8_t *)       dptr.dst;
        for (size_t r = 0; r < nrows; ++r) {
            dma_ptr p = dma_make_ptr(dst + r * dst_stride, src + r * src_stride);
            if (!dma_queue_push_single_1d(q, p, row_size))
                return false;
            if (r + 1 < nrows)
                dma_queue_pop(q);
        }
        return true;
    }
}

#else // HVX_ARCH >= 75

static inline bool dma_queue_push(dma_queue *q, dma_ptr dptr, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    // On v75 and up we always use 2d 24-bit mode
    return dma_queue_push_single_2d(q, dptr, dst_stride, src_stride, row_size, nrows);
}

#endif

static inline bool dma_queue_push_ddr_to_vtcm(dma_queue * q, dma_ptr dptr, size_t dst_row_size, size_t src_row_size, size_t nrows) {
    return dma_queue_push(q, dptr, dst_row_size, src_row_size, src_row_size, nrows);
}

static inline bool dma_queue_push_vtcm_to_ddr(dma_queue * q, dma_ptr dptr, size_t dst_row_size, size_t src_row_size, size_t nrows) {
    return dma_queue_push(q, dptr, dst_row_size, src_row_size, dst_row_size, nrows);
}

#define DMA_CACHE_MAX_SIZE 256U

typedef struct {
    uint8_t *base;
    uint32_t line_size;
    uint32_t capacity;
    uint32_t src[DMA_CACHE_MAX_SIZE];
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

static inline bool dma_cache_push(dma_queue *q, dma_cache *c, const uint8_t * src, uint32_t dst_stride, uint32_t src_stride, uint32_t row_size, uint32_t nrows)
{
    uint32_t o_idx = 0;
    uint16_t o_age = 0;
    uint8_t *  dst = 0;

    for (unsigned i=0; i < c->capacity; i++) {
        if (c->src[i] == (uint32_t) src) {
            c->age[i] = 0;
            dst = c->base + (i * c->line_size); nrows = 0; // dummy dma
        } else {
            c->age[i]++;
            if (c->age[i] > o_age) { o_age = c->age[i]; o_idx = i; }
        }
    }
    if (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = (uint32_t) src;
        dst = c->base + o_idx * c->line_size; // normal nrows dma
        return dma_queue_push(q, dma_make_ptr(dst, src), dst_stride, src_stride, row_size, nrows);
    }

    return dma_queue_push_single_1d(q, dma_make_ptr(dst, src), 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
