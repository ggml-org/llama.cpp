#include "dma_transfer.hpp"

#include <dma_desc.h>
#include <qurt.h>

#include <array>
#include <cstdlib>

namespace hexagon::dma {

dma_transfer::dma_transfer() {
    dma_desc_set_next(_dma_1d_desc0, 0);
    dma_desc_set_dstate(_dma_1d_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_1d_desc0, DMA_DESC_TYPE_1D);
    dma_desc_set_order(_dma_1d_desc0, DESC_ORDER_ORDER);
    dma_desc_set_bypasssrc(_dma_1d_desc0, DESC_BYPASS_ON);   // for dram
    dma_desc_set_bypassdst(_dma_1d_desc0, DESC_BYPASS_OFF);  // for vtcm
    dma_desc_set_length(_dma_1d_desc0, 0);

    dma_desc_set_next(_dma_1d_desc1, 0);
    dma_desc_set_dstate(_dma_1d_desc1, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_1d_desc1, DMA_DESC_TYPE_1D);
    dma_desc_set_order(_dma_1d_desc1, DESC_ORDER_ORDER);
    dma_desc_set_bypasssrc(_dma_1d_desc1, DESC_BYPASS_ON);   // for dram
    dma_desc_set_bypassdst(_dma_1d_desc1, DESC_BYPASS_OFF);  // for vtcm
    dma_desc_set_length(_dma_1d_desc1, 0);

    dma_desc_set_next(_dma_2d_desc0, 0);
    dma_desc_set_dstate(_dma_2d_desc0, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(_dma_2d_desc0, DMA_DESC_TYPE_2D);
    dma_desc_set_order(_dma_2d_desc0, DESC_ORDER_ORDER);
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
    auto ret = dma_wait_for_idle();
    if (ret != DMA_SUCCESS) {
        DEVICE_LOG_ERROR("dma_transfer: failed to wait for DMA idle: %d\n", ret);
    }
}

bool dma_transfer::is_desc_done(uint8_t * desc) {
    return !dma_desc_get_src(desc) || dma_desc_is_done(desc) == DMA_COMPLETE;
}

bool dma_transfer::submit_impl(void ** desc_batch, int batch_len) {
    _dma_desc_mutex.lock();
    const bool succ = dma_desc_submit(desc_batch, batch_len) == DMA_SUCCESS;
    _dma_desc_mutex.unlock();
    return succ;
}

qurt_mutex dma_transfer::_dma_desc_mutex;

}  // namespace hexagon::dma
