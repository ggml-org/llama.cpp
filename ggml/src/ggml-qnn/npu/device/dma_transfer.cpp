#include "dma_transfer.hpp"

#include <qurt.h>

#include <array>
#include <cstdlib>

namespace {

// From addons/compute/libs/userdma/utils_lib/

#define DM0_STATUS_MASK  0x00000003
#define DM0_STATUS_SHIFT 0
#define DM0_STATUS_IDLE  0
#define DM0_STATUS_RUN   1
#define DM0_STATUS_ERROR 2

#define DM0_DESC_ADDR_MASK  0xFFFFFFF0
#define DM0_DESC_ADDR_SHIFT 4

#define DMA_COMPLETE   1
#define DMA_INCOMPLETE 0

#define DMA_SUCCESS 0
#define DMA_FAIL    -1

#define DMA_DESC_TYPE_1D 0
#define DMA_DESC_TYPE_2D 1

#define DESC_NEXT_MASK  0xFFFFFFFF
#define DESC_NEXT_SHIFT 0

#define DESC_DSTATE_MASK       0x80000000
#define DESC_DSTATE_SHIFT      31
#define DESC_DSTATE_INCOMPLETE 0
#define DESC_DSTATE_COMPLETE   1

#define DESC_ORDER_MASK    0x40000000
#define DESC_ORDER_SHIFT   30
#define DESC_ORDER_NOORDER 0
#define DESC_ORDER_ORDER   1

#define DESC_BYPASSSRC_MASK  0x20000000
#define DESC_BYPASSSRC_SHIFT 29
#define DESC_BYPASSDST_MASK  0x10000000
#define DESC_BYPASSDST_SHIFT 28
#define DESC_BYPASS_OFF      0
#define DESC_BYPASS_ON       1

#define DESC_DESCTYPE_MASK  0x03000000
#define DESC_DESCTYPE_SHIFT 24
#define DESC_DESCTYPE_1D    0
#define DESC_DESCTYPE_2D    1

#define DESC_LENGTH_MASK  0x00FFFFFF
#define DESC_LENGTH_SHIFT 0
#define DESC_SRC_MASK     0xFFFFFFFF
#define DESC_SRC_SHIFT    0
#define DESC_DST_MASK     0xFFFFFFFF
#define DESC_DST_SHIFT    0

#define DESC_CACHEALLOC_MASK      0x03000000
#define DESC_CACHEALLOC_SHIFT     24
#define DESC_CACHEALLOC_NONE      0
#define DESC_CACHEALLOC_WRITEONLY 1
#define DESC_CACHEALLOC_READONLY  2
#define DESC_CACHEALLOC_READWRITE 3

#define DESC_ROIWIDTH_MASK   0x0000FFFF
#define DESC_ROIWIDTH_SHIFT  0
#define DESC_ROIHEIGHT_MASK  0xFFFF0000
#define DESC_ROIHEIGHT_SHIFT 16

#define DESC_SRCSTRIDE_MASK  0x0000FFFF
#define DESC_SRCSTRIDE_SHIFT 0
#define DESC_DSTSTRIDE_MASK  0xFFFF0000
#define DESC_DSTSTRIDE_SHIFT 16

#define DESC_SRCWIDTHOFFSET_MASK  0x0000FFFF
#define DESC_SRCWIDTHOFFSET_SHIFT 0
#define DESC_DSTWIDTHOFFSET_MASK  0xFFFF0000
#define DESC_DSTWIDTHOFFSET_SHIFT 16

/**************************/
/* 1D (linear) descriptor */
/**************************/
typedef struct _dma_desc_1d_t {
    uint32_t next;
    uint32_t dstate_order_bypass_desctype_length;
    uint32_t src;
    uint32_t dst;
} dma_desc_1d_t;

static_assert(sizeof(dma_desc_1d_t) == hexagon::dma::kDmaDescSize1D, "kDmaDescSize1D size incorrect");

/***********************/
/* 2D (box) descriptor */
/***********************/
typedef struct _dma_desc_2d_t {
    uint32_t next;
    uint32_t dstate_order_bypass_desctype_length;
    uint32_t src;
    uint32_t dst;
    uint32_t allocation;
    uint32_t roiheight_roiwidth;
    uint32_t dststride_srcstride;
    uint32_t dstwidthoffset_srcwidthoffset;
} dma_desc_2d_t;

static_assert(sizeof(dma_desc_2d_t) == hexagon::dma::kDmaDescSize2D, "kDmaDescSize2D size incorrect");

inline void dmstart(void * next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmstart(%0)" : : "r"(next));
}

inline void dmlink(void * cur, void * next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

inline unsigned int dmpoll(void) {
    unsigned int ret = 0;
    asm volatile(" %0 = dmpoll" : "=r"(ret) : : "memory");
    return ret;
}

inline unsigned int dmwait(void) {
    unsigned int ret = 0;
    asm volatile(" %0 = dmwait" : "=r"(ret) : : "memory");
    return ret;
}

inline void dma_desc_set_next(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->next) &= ~DESC_NEXT_MASK;
    (((dma_desc_1d_t *) d)->next) |= ((v << DESC_NEXT_SHIFT) & DESC_NEXT_MASK);
}

inline uint32_t dma_desc_get_dstate(void * d) {
    return (((((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) & DESC_DSTATE_MASK) >> DESC_DSTATE_SHIFT);
}

inline void dma_desc_set_dstate(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_DSTATE_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_DSTATE_SHIFT) & DESC_DSTATE_MASK);
}

inline void dma_desc_set_desctype(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_DESCTYPE_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_DESCTYPE_SHIFT) & DESC_DESCTYPE_MASK);
}

inline void dma_desc_set_order(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_ORDER_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_ORDER_SHIFT) & DESC_ORDER_MASK);
}

inline void dma_desc_set_bypasssrc(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_BYPASSSRC_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_BYPASSSRC_SHIFT) & DESC_BYPASSSRC_MASK);
}

inline void dma_desc_set_bypassdst(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_BYPASSDST_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_BYPASSDST_SHIFT) & DESC_BYPASSDST_MASK);
}

inline void dma_desc_set_length(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_LENGTH_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_LENGTH_SHIFT) & DESC_LENGTH_MASK);
}

inline uint32_t dma_desc_get_src(void * d) {
    return (((((dma_desc_1d_t *) d)->src) & DESC_SRC_MASK) >> DESC_SRC_SHIFT);
}

inline void dma_desc_set_src(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->src) &= ~DESC_SRC_MASK;
    (((dma_desc_1d_t *) d)->src) |= ((v << DESC_SRC_SHIFT) & DESC_SRC_MASK);
}

inline void dma_desc_set_dst(void * d, uint32_t v) {
    (((dma_desc_1d_t *) d)->dst) &= ~DESC_DST_MASK;
    (((dma_desc_1d_t *) d)->dst) |= ((v << DESC_DST_SHIFT) & DESC_DST_MASK);
}

inline void dma_desc_set_roiwidth(void * d, uint32_t v) {
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) &= ~DESC_ROIWIDTH_MASK;
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) |= ((v << DESC_ROIWIDTH_SHIFT) & DESC_ROIWIDTH_MASK);
}

inline void dma_desc_set_roiheight(void * d, uint32_t v) {
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) &= ~DESC_ROIHEIGHT_MASK;
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) |= ((v << DESC_ROIHEIGHT_SHIFT) & DESC_ROIHEIGHT_MASK);
}

inline void dma_desc_set_srcstride(void * d, uint32_t v) {
    (((dma_desc_2d_t *) d)->dststride_srcstride) &= ~DESC_SRCSTRIDE_MASK;
    (((dma_desc_2d_t *) d)->dststride_srcstride) |= ((v << DESC_SRCSTRIDE_SHIFT) & DESC_SRCSTRIDE_MASK);
}

inline void dma_desc_set_dststride(void * d, uint32_t v) {
    (((dma_desc_2d_t *) d)->dststride_srcstride) &= ~DESC_DSTSTRIDE_MASK;
    (((dma_desc_2d_t *) d)->dststride_srcstride) |= ((v << DESC_DSTSTRIDE_SHIFT) & DESC_DSTSTRIDE_MASK);
}

inline void dma_desc_set_srcwidthoffset(void * d, uint32_t v) {
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) &= ~DESC_SRCWIDTHOFFSET_MASK;
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) |=
        ((v << DESC_SRCWIDTHOFFSET_SHIFT) & DESC_SRCWIDTHOFFSET_MASK);
}

inline void dma_desc_set_dstwidthoffset(void * d, uint32_t v) {
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) &= ~DESC_DSTWIDTHOFFSET_MASK;
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) |=
        ((v << DESC_DSTWIDTHOFFSET_SHIFT) & DESC_DSTWIDTHOFFSET_MASK);
}

inline void dma_desc_set_cachealloc(void * d, uint32_t v) {
    (((dma_desc_2d_t *) d)->allocation) &= ~DESC_CACHEALLOC_MASK;
    (((dma_desc_2d_t *) d)->allocation) |= ((v << DESC_CACHEALLOC_SHIFT) & DESC_CACHEALLOC_MASK);
}

}  // namespace

namespace hexagon::dma {

dma_transfer::dma_transfer() {
    dma_desc_set_next(_dma_1d_desc0, 0);
    dma_desc_set_dstate(_dma_1d_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_1d_desc0, DMA_DESC_TYPE_1D);
    dma_desc_set_order(_dma_1d_desc0, DESC_ORDER_NOORDER);
    dma_desc_set_bypasssrc(_dma_1d_desc0, DESC_BYPASS_ON);   // for dram
    dma_desc_set_bypassdst(_dma_1d_desc0, DESC_BYPASS_OFF);  // for vtcm
    dma_desc_set_length(_dma_1d_desc0, 0);

    dma_desc_set_next(_dma_1d_desc1, 0);
    dma_desc_set_dstate(_dma_1d_desc1, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_1d_desc1, DMA_DESC_TYPE_1D);
    dma_desc_set_order(_dma_1d_desc1, DESC_ORDER_NOORDER);
    dma_desc_set_bypasssrc(_dma_1d_desc1, DESC_BYPASS_ON);   // for dram
    dma_desc_set_bypassdst(_dma_1d_desc1, DESC_BYPASS_OFF);  // for vtcm
    dma_desc_set_length(_dma_1d_desc1, 0);

    dma_desc_set_next(_dma_2d_desc0, 0);
    dma_desc_set_dstate(_dma_2d_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_2d_desc0, DMA_DESC_TYPE_2D);
    dma_desc_set_order(_dma_2d_desc0, DESC_ORDER_NOORDER);
    dma_desc_set_bypasssrc(_dma_2d_desc0, DESC_BYPASS_ON);   // for dram
    dma_desc_set_bypassdst(_dma_2d_desc0, DESC_BYPASS_OFF);  // for vtcm
    dma_desc_set_cachealloc(_dma_2d_desc0, DESC_CACHEALLOC_NONE);
    dma_desc_set_roiwidth(_dma_2d_desc0, 0);
    dma_desc_set_roiheight(_dma_2d_desc0, 0);
    dma_desc_set_srcstride(_dma_2d_desc0, 0);
    dma_desc_set_dststride(_dma_2d_desc0, 0);
    dma_desc_set_srcwidthoffset(_dma_2d_desc0, 0);
    dma_desc_set_dstwidthoffset(_dma_2d_desc0, 0);
}

dma_transfer::~dma_transfer() {
    wait();
}

bool dma_transfer::submit1d(const uint8_t * src, uint8_t * dst, size_t size) {
    constexpr size_t kMaxDmaTransferSize = DESC_LENGTH_MASK;
    if (size > kMaxDmaTransferSize) {
        // TODO: support chained descriptors for large transfers
        DEVICE_LOG_ERROR("dma_transfer::submit1d, size(%zu) is too large\n", size);
        return false;
    }

    if (!dma_transfer::is_desc_done(_dma_1d_desc0)) {
        DEVICE_LOG_ERROR("Failed to initiate DMA transfer for one or more descriptors\n");
        return false;
    }

    dma_desc_set_next(_dma_1d_desc0, 0);
    dma_desc_set_dstate(_dma_1d_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_src(_dma_1d_desc0, reinterpret_cast<uint32_t>(src));
    dma_desc_set_dst(_dma_1d_desc0, reinterpret_cast<uint32_t>(dst));
    dma_desc_set_length(_dma_1d_desc0, size);

    void * buffs[] = { _dma_1d_desc0 };
    if (!submit_impl(buffs, std::size(buffs))) {
        DEVICE_LOG_ERROR("Failed to submit DMA descriptor\n");
        return false;
    }

    DEVICE_LOG_DEBUG("dma_transfer::submit1d, src(%p), dst(%p), size(%zu), desc(%p)\n", (void *) src, (void *) dst,
                     size, (void *) _dma_1d_desc0);
    return true;
}

bool dma_transfer::submit1d(const uint8_t * src0, uint8_t * dst0, const uint8_t * src1, uint8_t * dst1, size_t size) {
    constexpr size_t kMaxDmaTransferSize = DESC_LENGTH_MASK;
    if (size > kMaxDmaTransferSize) {
        // TODO: support chained descriptors for large transfers
        DEVICE_LOG_ERROR("dma_transfer::submit1d, size(%zu) is too large\n", size);
        return false;
    }

    if (!dma_transfer::is_desc_done(_dma_1d_desc0) || !dma_transfer::is_desc_done(_dma_1d_desc1)) {
        DEVICE_LOG_ERROR("Failed to initiate DMA transfer for one or more descriptors\n");
        return false;
    }

    dma_desc_set_next(_dma_1d_desc0, 0);
    dma_desc_set_dstate(_dma_1d_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_src(_dma_1d_desc0, reinterpret_cast<uint32_t>(src0));
    dma_desc_set_dst(_dma_1d_desc0, reinterpret_cast<uint32_t>(dst0));
    dma_desc_set_length(_dma_1d_desc0, size);

    dma_desc_set_next(_dma_1d_desc1, 0);
    dma_desc_set_dstate(_dma_1d_desc1, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_src(_dma_1d_desc1, reinterpret_cast<uint32_t>(src1));
    dma_desc_set_dst(_dma_1d_desc1, reinterpret_cast<uint32_t>(dst1));
    dma_desc_set_length(_dma_1d_desc1, size);

    void * buffs[] = { _dma_1d_desc0, _dma_1d_desc1 };
    if (!submit_impl(buffs, std::size(buffs))) {
        DEVICE_LOG_ERROR("Failed to submit DMA descriptor\n");
        return false;
    }

    DEVICE_LOG_DEBUG(
        "dma_transfer::submit1d, src0(%p), dst0(%p), src1(%p), dst1(%p), size(%zu), desc0(%p), desc1(%p)\n",
        (void *) src0, (void *) dst0, (void *) src1, (void *) dst1, size, (void *) _dma_1d_desc0,
        (void *) _dma_1d_desc1);
    return true;
}

bool dma_transfer::submit2d(const uint8_t * src,
                            uint8_t *       dst,
                            size_t          width,
                            size_t          height,
                            size_t          src_stride,
                            size_t          dst_stride) {
    // Note that the dma only supports 16-bit width and height for 2D transfer, see also: DESC_ROIWIDTH_MASK
    constexpr size_t kMaxDmaTransferSize = DESC_ROIWIDTH_MASK;
    if (width > kMaxDmaTransferSize || height > kMaxDmaTransferSize || src_stride > kMaxDmaTransferSize ||
        dst_stride > kMaxDmaTransferSize) {
        if (src_stride != dst_stride) {
            // TODO: support chained descriptors for large transfers
            DEVICE_LOG_ERROR("dma_transfer::submit2d, src_stride(%zu) or dst_stride(%zu) is too large\n", src_stride,
                             dst_stride);
            return false;
        }

        DEVICE_LOG_DEBUG("dma_transfer::submit2d, width(%zu) or height(%zu) is too large, fallback to 1D transfer\n",
                         width, height);
        return submit1d(src, dst, src_stride * height);
    }

    if (!dma_transfer::is_desc_done(_dma_2d_desc0)) {
        DEVICE_LOG_ERROR("Failed to initiate DMA transfer for one or more descriptors\n");
        return false;
    }

    dma_desc_set_next(_dma_2d_desc0, 0);
    dma_desc_set_dstate(_dma_2d_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_src(_dma_2d_desc0, reinterpret_cast<uint32_t>(src));
    dma_desc_set_dst(_dma_2d_desc0, reinterpret_cast<uint32_t>(dst));
    dma_desc_set_roiwidth(_dma_2d_desc0, width);
    dma_desc_set_roiheight(_dma_2d_desc0, height);
    dma_desc_set_srcstride(_dma_2d_desc0, src_stride);
    dma_desc_set_dststride(_dma_2d_desc0, dst_stride);

    void * buffs[] = { _dma_2d_desc0 };
    if (!submit_impl(buffs, std::size(buffs))) {
        DEVICE_LOG_ERROR("Failed to submit DMA descriptor\n");
        return false;
    }

    DEVICE_LOG_DEBUG(
        "dma_transfer::submit2d, src(%p), dst(%p), width(%zu), height(%zu), src_stride(%zu), dst_stride(%zu), "
        "desc(%p)\n",
        (void *) src, (void *) dst, width, height, src_stride, dst_stride, (void *) _dma_2d_desc0);
    return true;
}

void dma_transfer::wait() {
    uint32_t dm0_status = dmwait() & DM0_STATUS_MASK;
    if (dm0_status != DM0_STATUS_IDLE) {
        DEVICE_LOG_ERROR("dma_transfer: failed to wait for DMA idle, dm0_status: %d\n", (int) dm0_status);
    }
}

bool dma_transfer::is_desc_done(uint8_t * desc) {
    if (!dma_desc_get_src(desc)) {
        return true;
    }

    if (dma_desc_get_dstate(desc) == DESC_DSTATE_COMPLETE) {
        return true;
    }

    dmpoll();
    return false;
}

bool dma_transfer::submit_impl(void ** desc_batch, size_t batch_len) {
    _dma_desc_mutex.lock();
    for (size_t i = 0; i < batch_len - 1; i++) {
        dma_desc_set_next(desc_batch[i], (uint32_t) desc_batch[i + 1]);
    }

    dma_desc_set_next(desc_batch[batch_len - 1], (uint32_t) nullptr);
    uint32_t dm0_status = dmpoll() & DM0_STATUS_MASK;
    if (dm0_status == DM0_STATUS_IDLE) {
        dmstart(desc_batch[0]);
    } else if (dm0_status == DM0_STATUS_RUN) {
        if (_dma_last_desc == nullptr) {
            _dma_desc_mutex.unlock();
            DEVICE_LOG_ERROR("dma_transfer: last descriptor not found for linking. Submission failed\n");
            return false;
        } else {
            dmlink(_dma_last_desc, desc_batch[0]);
        }
    } else {
        _dma_desc_mutex.unlock();
        DEVICE_LOG_ERROR("dma_transfer: DMA not idle or running. Submission failed\n");
        return false;
    }

    dmpoll();

    _dma_last_desc = (void *) desc_batch[batch_len - 1];

    _dma_desc_mutex.unlock();
    return true;
}

qurt_mutex dma_transfer::_dma_desc_mutex;
void *     dma_transfer::_dma_last_desc = nullptr;

}  // namespace hexagon::dma
