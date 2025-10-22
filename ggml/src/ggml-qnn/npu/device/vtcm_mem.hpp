#pragma once

#include "util.hpp"

#include <HAP_compute_res.h>
#include <HAP_vtcm_mgr.h>

namespace hexagon {

class vtcm_mem {
  public:
    explicit vtcm_mem(size_t size, bool single_page) {
        constexpr const unsigned int kTimeoutUs = 10000;  // 10ms timeout

        size_t avail_size = single_page ? get_avail_page_size() : get_avail_block_size();
        if (size > avail_size) {
            DEVICE_LOG_ERROR("Requested VTCM size %zu exceeds available size %zu\n", size, avail_size);
            return;
        }

        compute_res_attr_t compute_res;
        HAP_compute_res_attr_init(&compute_res);
        HAP_compute_res_attr_set_serialize(&compute_res, false);
        HAP_compute_res_attr_set_vtcm_param(&compute_res, size, single_page ? 1 : 0);

        _vtcm_context_id = HAP_compute_res_acquire(&compute_res, kTimeoutUs);  // 10ms timeout
        if (_vtcm_context_id == 0) {
            DEVICE_LOG_ERROR("Failed to acquire VTCM context: %zu bytes, timeout %zu us\n", size, kTimeoutUs);
            return;
        }

        _vtcm_mem = HAP_compute_res_attr_get_vtcm_ptr(&compute_res);
        if (_vtcm_mem == nullptr) {
            DEVICE_LOG_ERROR("Failed to allocate VTCM memory: %zu bytes, timeout %zu us\n", size, kTimeoutUs);
            return;
        }

        _vtcm_size = size;
        DEVICE_LOG_DEBUG("VTCM allocated: %p(%zu), avail: %zu\n", _vtcm_mem, size, avail_size);
    }

    explicit vtcm_mem(size_t size, bool single_page, size_t timeout_us) {
        compute_res_attr_t compute_res;
        HAP_compute_res_attr_init(&compute_res);
        HAP_compute_res_attr_set_serialize(&compute_res, false);
        HAP_compute_res_attr_set_vtcm_param(&compute_res, size, single_page ? 1 : 0);

        _vtcm_context_id = HAP_compute_res_acquire(&compute_res, timeout_us);
        if (_vtcm_context_id == 0) {
            DEVICE_LOG_ERROR("Failed to acquire VTCM context: %zu bytes, timeout %zu us\n", size, timeout_us);
            return;
        }

        _vtcm_mem = HAP_compute_res_attr_get_vtcm_ptr(&compute_res);
        if (_vtcm_mem == nullptr) {
            DEVICE_LOG_ERROR("Failed to allocate VTCM memory: %zu bytes, timeout %zu us\n", size, timeout_us);
            return;
        }

        _vtcm_size = size;
        DEVICE_LOG_DEBUG("VTCM allocated: %p(%zu), avail: %zu\n", _vtcm_mem, size, get_avail_block_size());
    }

    ~vtcm_mem() {
        if (_vtcm_context_id != 0) {
            auto ret = HAP_compute_res_release(_vtcm_context_id);
            if (ret != AEE_SUCCESS) {
                DEVICE_LOG_ERROR("Failed to release VTCM memory: %d\n", ret);
            }
        }

        DEVICE_LOG_DEBUG("VTCM released: %zu bytes at %p\n", _vtcm_size, _vtcm_mem);
    }

    bool is_valid() const { return _vtcm_mem != nullptr && _vtcm_size != 0; }

    uint8_t * get_mem() const { return reinterpret_cast<uint8_t *>(_vtcm_mem); }

    size_t get_size() const { return _vtcm_size; }

    static size_t get_total_size() {
        unsigned int arch_page_aligned_size = 0;
        unsigned int arch_page_count        = 0;
        auto         ret                    = HAP_query_total_VTCM(&arch_page_aligned_size, &arch_page_count);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to query total VTCM: %d\n", ret);
            return 0;
        }

        return arch_page_aligned_size;
    }

    static size_t get_avail_block_size() {
        unsigned int avail_block_size = 0;
        unsigned int avail_page_size  = 0;
        unsigned int num_pages        = 0;
        auto         ret              = HAP_query_avail_VTCM(&avail_block_size, &avail_page_size, &num_pages);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to query available VTCM: %d\n", ret);
            return 0;
        }

        return avail_block_size;
    }

    static size_t get_avail_page_size() {
        unsigned int avail_block_size = 0;
        unsigned int avail_page_size  = 0;
        unsigned int num_pages        = 0;
        auto         ret              = HAP_query_avail_VTCM(&avail_block_size, &avail_page_size, &num_pages);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to query available VTCM: %d\n", ret);
            return 0;
        }

        return avail_page_size;
    }

  private:
    void * _vtcm_mem  = nullptr;
    size_t _vtcm_size = 0;

    unsigned int _vtcm_context_id = 0;

    DISABLE_COPY_AND_MOVE(vtcm_mem);
};

}  // namespace hexagon
