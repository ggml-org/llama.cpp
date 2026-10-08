#pragma once

#include "dispatch/dispatch.h"
#include "ggml.h"
#include "runtime/host-buffer-registry.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

typedef struct hrx_buffer_s * hrx_buffer_t;
typedef struct hrx_device_s * hrx_device_t;
typedef struct hrx_stream_s * hrx_stream_t;

namespace ggml::hrx {

struct HostTransferStats {
    uint64_t uploads        = 0;
    uint64_t downloads      = 0;
    size_t   upload_bytes   = 0;
    size_t   download_bytes = 0;
};

struct HostMappedDownloadLease {
    std::unique_lock<std::mutex> lock;
    void *                       host_data = nullptr;
};

class HostTransferManager {
  public:
    ~HostTransferManager();

    Status upload_synchronous(hrx_stream_t stream,
                              const void * host_source,
                              hrx_buffer_t destination,
                              size_t       offset,
                              size_t       size);
    Status upload_async(hrx_stream_t stream,
                        const void * host_source,
                        hrx_buffer_t destination,
                        size_t       offset,
                        size_t       size);
    Status download_synchronous(hrx_stream_t stream,
                                hrx_buffer_t source,
                                size_t       offset,
                                void *       host_destination,
                                size_t       size);
    Status download_to_mapped_synchronous(hrx_device_t              device,
                                          hrx_stream_t              stream,
                                          hrx_buffer_t              source,
                                          size_t                    offset,
                                          size_t                    size,
                                          HostMappedDownloadLease & lease);

    HostTransferStats stats() const;
    void              clear();

  private:
    mutable std::mutex mutex_;
    HostTransferStats  stats_;
    hrx_buffer_t       mapped_download_buffer_   = nullptr;
    void *             mapped_download_data_     = nullptr;
    size_t             mapped_download_capacity_ = 0;
};

struct HostWeightSource {
    const void * host_data  = nullptr;
    hrx_buffer_t device_buffer       = nullptr;
    uint64_t     identity   = 0;
    uint64_t     generation = 0;
    size_t       capacity   = 0;
    size_t       offset     = 0;
    size_t       length     = 0;
    size_t       materialized_length = 0;
    std::string  layout              = kNativeWeightLayout;
    ggml_type    source_type         = GGML_TYPE_COUNT;
    int64_t      input_size          = 0;
    int64_t      output_size         = 0;
};

struct HostWeightCacheStats {
    uint64_t hits             = 0;
    uint64_t misses           = 0;
    size_t   allocation_count = 0;
    size_t   resident_bytes   = 0;
};

class HostWeightLease {
  public:
    HostWeightLease() = default;

    bool                valid() const;
    hrx_buffer_t        buffer() const;
    size_t              length() const;
    const std::string & layout() const;

  private:
    struct Entry;
    std::shared_ptr<Entry> entry_;

    explicit HostWeightLease(std::shared_ptr<Entry> entry);
    friend class HostWeightCache;
};

struct HostWeightAcquireResult {
    HostWeightLease lease;
    Status          status;

    bool valid() const { return lease.valid() && status.success(); }
};

class HostWeightCache {
  public:
    HostWeightCache() = default;
    ~HostWeightCache();

    HostWeightCache(const HostWeightCache &)             = delete;
    HostWeightCache & operator=(const HostWeightCache &) = delete;

    HostWeightAcquireResult acquire(hrx_device_t             device,
                                    hrx_stream_t             stream,
                                    HostTransferManager &    transfers,
                                    const HostWeightSource & source);
    HostWeightCacheStats    stats() const;
    void                    clear();

  private:
    struct SourceKey {
        uint64_t identity   = 0;
        uint64_t generation = 0;
        size_t   capacity   = 0;
        size_t   offset     = 0;
        size_t   length     = 0;
        size_t      materialized_length = 0;
        std::string layout;
        ggml_type   source_type = GGML_TYPE_COUNT;
        int64_t     input_size  = 0;
        int64_t     output_size = 0;

        bool operator==(const SourceKey & other) const {
            return identity == other.identity && generation == other.generation && capacity == other.capacity &&
                   offset == other.offset && length == other.length &&
                   materialized_length == other.materialized_length && layout == other.layout &&
                   source_type == other.source_type && input_size == other.input_size &&
                   output_size == other.output_size;
        }
    };

    struct SourceKeyHash {
        size_t operator()(const SourceKey & key) const;
    };

    mutable std::mutex                                                                    mutex_;
    std::unordered_map<SourceKey, std::shared_ptr<HostWeightLease::Entry>, SourceKeyHash> entries_;
    HostWeightCacheStats                                                                  stats_;
};

struct HostStagingBuffer {
    HostStagingBuffer() = default;
    ~HostStagingBuffer();

    HostStagingBuffer(HostStagingBuffer && other) noexcept;
    HostStagingBuffer & operator=(HostStagingBuffer && other) noexcept;

    HostStagingBuffer(const HostStagingBuffer &)             = delete;
    HostStagingBuffer & operator=(const HostStagingBuffer &) = delete;

    hrx_buffer_t buffer    = nullptr;
    void *       host_data = nullptr;
    HostBufferRef source_host_buffer;
    int32_t      value     = -1;
    size_t       length    = 0;
    bool         upload    = false;
    bool         download  = false;

    void clear();
};

Status allocate_host_staging_buffer(hrx_device_t device, size_t size, HostStagingBuffer & staging);
Status allocate_mapped_host_staging_buffer(hrx_device_t device,
                                           size_t       size,
                                           hrx_buffer_t & buffer,
                                           void *&      host_data);

}  // namespace ggml::hrx
