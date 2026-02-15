#pragma once

#include "util.hpp"

namespace hexagon::dma {

constexpr const size_t kDmaDescSize1D = 16;
constexpr const size_t kDmaDescSize2D = 32;

class dma_transfer {
  public:
    dma_transfer();
    ~dma_transfer();

    /**
     * Submits a 1D DMA transfer.
     *
     * Limitations:
     *   - The maximum supported transfer size is kMaxDmaTransferSize (DESC_LENGTH_MASK, 24bit).
     *   - Transfers larger than this size are not supported and will fail.
     *   - Large transfers must be split into multiple smaller transfers by the caller.
     */
    bool submit1d(const uint8_t * src, uint8_t * dst, size_t size);
    bool submit1d(const uint8_t * src0, uint8_t * dst0, const uint8_t * src1, uint8_t * dst1, size_t size);
    bool submit2d(const uint8_t * src,
                  uint8_t *       dst,
                  size_t          width,
                  size_t          height,
                  size_t          src_stride,
                  size_t          dst_stride);
    void wait();

  private:
    static bool       is_desc_done(uint8_t * desc);  // TODO: should we use void * here?
    static qurt_mutex _dma_desc_mutex;
    static void *     _dma_last_desc;

    // TODO: can we avoid the void ** here?
    bool submit_impl(void ** desc_batch, size_t batch_len);

    alignas(kDmaDescSize1D) uint8_t _dma_1d_desc0[kDmaDescSize1D] = {};
    alignas(kDmaDescSize1D) uint8_t _dma_1d_desc1[kDmaDescSize1D] = {};
    alignas(kDmaDescSize2D) uint8_t _dma_2d_desc0[kDmaDescSize2D] = {};

    DISABLE_COPY_AND_MOVE(dma_transfer);
};

}  // namespace hexagon::dma
