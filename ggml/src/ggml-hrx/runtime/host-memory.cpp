#include "host-memory.h"

#include "hrx-interop-utils.h"
#include "hrx_runtime.h"

#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr size_t kMaxInlineUploadBytes = 63 * 1024;

static Status allocate_device_buffer(hrx_device_t device, size_t size, hrx_buffer_t & buffer) {
    Status status;
    if (device == nullptr) {
        status.log("missing HRX device for host memory allocation");
        return status;
    }
    if (size == 0) {
        status.log("cannot allocate an empty host memory device buffer");
        return status;
    }
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    if (ErrorResult error =
            take_status(hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, size, &buffer))) {
        status.log("allocate host memory device buffer: %s", error->c_str());
    }
    return status;
}

}  // namespace

Status HostTransferManager::upload_synchronous(hrx_stream_t stream,
                                               const void * host_source,
                                               hrx_buffer_t destination,
                                               size_t       offset,
                                               size_t       size) {
    Status status;
    if (size == 0) {
        return status;
    }
    if (stream == nullptr || host_source == nullptr || destination == nullptr) {
        status.log("invalid HRX host upload");
        return status;
    }
    hrx_device_t device = nullptr;
    if (ErrorResult error = take_status(hrx_stream_get_device(stream, &device))) {
        status.log("query HRX upload device failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
        status.log("synchronize before HRX host upload failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_synchronous_h2d(device, host_source, destination, offset, size))) {
        status.log("synchronous HRX host upload failed: %s", error->c_str());
        return status;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.uploads;
    stats_.upload_bytes += size;
    return status;
}

Status HostTransferManager::upload_async(hrx_stream_t stream,
                                         const void * host_source,
                                         hrx_buffer_t destination,
                                         size_t       offset,
                                         size_t       size) {
    Status status;
    if (size == 0) {
        return status;
    }
    if (stream == nullptr || host_source == nullptr || destination == nullptr) {
        status.log("invalid HRX host upload");
        return status;
    }

    const uint8_t * host_bytes = static_cast<const uint8_t *>(host_source);
    size_t          uploaded   = 0;
    while (uploaded < size) {
        const size_t remaining  = size - uploaded;
        const size_t chunk_size = remaining < kMaxInlineUploadBytes ? remaining : kMaxInlineUploadBytes;
        if (ErrorResult error = take_status(
                hrx_stream_update_buffer(stream, host_bytes + uploaded, chunk_size, destination, offset + uploaded))) {
            status.log("HRX async host upload failed: %s", error->c_str());
            return status;
        }
        uploaded += chunk_size;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.uploads;
    stats_.upload_bytes += size;
    return status;
}

Status HostTransferManager::download_synchronous(hrx_stream_t stream,
                                                 hrx_buffer_t source,
                                                 size_t       offset,
                                                 void *       host_destination,
                                                 size_t       size) {
    Status status;
    if (size == 0) {
        return status;
    }
    if (stream == nullptr || source == nullptr || host_destination == nullptr) {
        status.log("invalid HRX host download");
        return status;
    }
    hrx_device_t device = nullptr;
    if (ErrorResult error = take_status(hrx_stream_get_device(stream, &device))) {
        status.log("query HRX download device failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
        status.log("synchronize before HRX host download failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_synchronous_d2h(device, source, offset, host_destination, size))) {
        status.log("synchronous HRX host download failed: %s", error->c_str());
        return status;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.downloads;
    stats_.download_bytes += size;
    return status;
}

HostTransferStats HostTransferManager::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void HostTransferManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

struct HostWeightLease::Entry {
    ~Entry() {
        if (buffer != nullptr) {
            hrx_buffer_release(buffer);
        }
    }

    hrx_buffer_t buffer = nullptr;
    size_t       length = 0;
    std::string  layout;
};

HostWeightLease::HostWeightLease(std::shared_ptr<Entry> entry) : entry_(std::move(entry)) {}

bool HostWeightLease::valid() const {
    return entry_ != nullptr && entry_->buffer != nullptr;
}

hrx_buffer_t HostWeightLease::buffer() const {
    return valid() ? entry_->buffer : nullptr;
}

size_t HostWeightLease::length() const {
    return entry_ != nullptr ? entry_->length : 0;
}

const std::string & HostWeightLease::layout() const {
    static const std::string empty;
    return entry_ != nullptr ? entry_->layout : empty;
}

HostWeightCache::~HostWeightCache() {
    clear();
}

size_t HostWeightCache::SourceKeyHash::operator()(const SourceKey & key) const {
    uint64_t hash = UINT64_C(1469598103934665603);
    auto     mix  = [&](uint64_t value) {
        hash ^= value;
        hash *= UINT64_C(1099511628211);
    };
    mix(key.identity);
    mix(key.generation);
    mix(static_cast<uint64_t>(key.capacity));
    mix(static_cast<uint64_t>(key.offset));
    mix(static_cast<uint64_t>(key.length));
    return static_cast<size_t>(hash);
}

HostWeightAcquireResult HostWeightCache::acquire(hrx_device_t             device,
                                                 hrx_stream_t             stream,
                                                 HostTransferManager &    transfers,
                                                 const HostWeightSource & source) {
    HostWeightAcquireResult result;
    if (stream == nullptr) {
        result.status.log("host weight residency requires an HRX stream");
        return result;
    }
    if (source.host_data == nullptr || source.identity == 0 || source.generation == 0 || source.length == 0 ||
        source.offset > source.capacity || source.length > source.capacity - source.offset) {
        result.status.log("invalid host weight source");
        return result;
    }
    if (source.layout.empty()) {
        result.status.log("host weight source has no layout");
        return result;
    }

    const SourceKey key{ source.identity, source.generation, source.capacity, source.offset, source.length };
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = entries_.find(key);
        if (found != entries_.end()) {
            if (found->second->layout != source.layout) {
                ++stats_.layout_conflicts;
                result.status.log("host weight source already has resident layout %s, cannot also materialize %s",
                                  found->second->layout.c_str(), source.layout.c_str());
                return result;
            }
            ++stats_.hits;
            result.lease = HostWeightLease(found->second);
            return result;
        }
    }

    auto entry    = std::make_shared<HostWeightLease::Entry>();
    entry->length = source.length;
    entry->layout = source.layout;
    result.status = allocate_device_buffer(device, source.length, entry->buffer);
    if (!result.status.success()) {
        return result;
    }
    result.status = transfers.upload_synchronous(
        stream, static_cast<const uint8_t *>(source.host_data) + source.offset, entry->buffer, 0, source.length);
    if (!result.status.success()) {
        return result;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  inserted = entries_.emplace(key, entry);
        if (!inserted.second) {
            ++stats_.hits;
            result.lease = HostWeightLease(inserted.first->second);
            return result;
        }
        ++stats_.misses;
        stats_.allocation_count = entries_.size();
        stats_.resident_bytes += source.length;
    }
    result.lease = HostWeightLease(std::move(entry));
    return result;
}

HostWeightCacheStats HostWeightCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void HostWeightCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    stats_ = {};
}

HostStagingBuffer::~HostStagingBuffer() {
    clear();
}

HostStagingBuffer::HostStagingBuffer(HostStagingBuffer && other) noexcept {
    *this = std::move(other);
}

HostStagingBuffer & HostStagingBuffer::operator=(HostStagingBuffer && other) noexcept {
    if (this == &other) {
        return *this;
    }
    clear();
    buffer          = other.buffer;
    host_data       = other.host_data;
    value           = other.value;
    length          = other.length;
    upload          = other.upload;
    download        = other.download;
    other.buffer    = nullptr;
    other.host_data = nullptr;
    other.value     = -1;
    other.length    = 0;
    other.upload    = false;
    other.download  = false;
    return *this;
}

void HostStagingBuffer::clear() {
    if (buffer != nullptr) {
        hrx_buffer_release(buffer);
        buffer = nullptr;
    }
    host_data = nullptr;
    value     = -1;
    length    = 0;
    upload    = false;
    download  = false;
}

Status allocate_host_staging_buffer(hrx_device_t device, size_t size, HostStagingBuffer & staging) {
    staging.clear();
    Status status = allocate_device_buffer(device, size, staging.buffer);
    if (status.success()) {
        staging.length = size;
    }
    return status;
}

}  // namespace ggml::hrx
