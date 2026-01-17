// Probe SYCL pinned host allocation limits and report device capabilities as JSON.
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-pinned-probe --json --max-mb 4096

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kDefaultMaxMb = 4096;
constexpr size_t kStepMb = 64;

struct Attempt {
    size_t mb = 0;
    bool   ok = false;
    std::string error;
};

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

size_t align_down(size_t value, size_t step) {
    return (value / step) * step;
}

bool try_alloc_host(sycl::queue & q, size_t bytes, std::string & err) {
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_host(bytes, q);
    } catch (const sycl::exception & e) {
        err = e.what();
        return false;
    }
    if (!ptr) {
        err = "nullptr";
        return false;
    }
    std::memset(ptr, 0, std::min<size_t>(bytes, 4096));
    sycl::free(ptr, q);
    return true;
}

struct ProbeResult {
    size_t max_success_mb = 0;
    size_t first_failure_mb = 0;
    std::vector<Attempt> attempts;
};

ProbeResult probe_host_allocs(sycl::queue & q, size_t max_mb) {
    ProbeResult result{};
    const size_t step_bytes = kStepMb * 1024ULL * 1024ULL;
    const size_t max_bytes = max_mb * 1024ULL * 1024ULL;

    size_t last_ok = 0;
    size_t size = step_bytes;
    size_t first_fail = 0;

    while (size <= max_bytes) {
        Attempt attempt;
        attempt.mb = size / (1024ULL * 1024ULL);
        attempt.ok = try_alloc_host(q, size, attempt.error);
        result.attempts.push_back(attempt);
        if (attempt.ok) {
            last_ok = size;
            size *= 2;
        } else {
            first_fail = size;
            break;
        }
    }

    if (first_fail == 0) {
        result.max_success_mb = last_ok / (1024ULL * 1024ULL);
        return result;
    }

    size_t low = last_ok;
    size_t high = first_fail;
    while (high > low + step_bytes) {
        size_t mid = align_down((low + high) / 2, step_bytes);
        if (mid <= low || mid >= high) {
            break;
        }
        Attempt attempt;
        attempt.mb = mid / (1024ULL * 1024ULL);
        attempt.ok = try_alloc_host(q, mid, attempt.error);
        result.attempts.push_back(attempt);
        if (attempt.ok) {
            low = mid;
        } else {
            high = mid;
        }
    }

    result.max_success_mb = low / (1024ULL * 1024ULL);
    result.first_failure_mb = high / (1024ULL * 1024ULL);
    return result;
}

size_t parse_max_mb(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--max-mb") == 0) {
            long value = std::strtol(argv[i + 1], nullptr, 10);
            if (value > 0) {
                return static_cast<size_t>(value);
            }
        }
    }
    const char * env = std::getenv("GGML_SYCL_PINNED_PROBE_MAX_MB");
    if (env && env[0] != '\0') {
        long value = std::strtol(env, nullptr, 10);
        if (value > 0) {
            return static_cast<size_t>(value);
        }
    }
    return kDefaultMaxMb;
}

}  // namespace

int main(int argc, char ** argv) {
    const size_t max_mb = parse_max_mb(argc, argv);

    std::vector<sycl::device> devices;
    for (const auto & platform : sycl::platform::get_platforms()) {
        for (const auto & dev : platform.get_devices()) {
            if (dev.is_gpu()) {
                devices.push_back(dev);
            }
        }
    }

    std::printf("{\"probe\":{\"max_mb\":%zu,\"step_mb\":%zu},\"devices\":[", max_mb, kStepMb);
    bool first_dev = true;
    for (size_t idx = 0; idx < devices.size(); ++idx) {
        sycl::device dev = devices[idx];
        sycl::queue  q(dev);

        auto name = dev.get_info<sycl::info::device::name>();
        auto vendor = dev.get_info<sycl::info::device::vendor>();
        auto driver = dev.get_info<sycl::info::device::driver_version>();
        auto max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
        auto global_mem = dev.get_info<sycl::info::device::global_mem_size>();
        auto local_mem = dev.get_info<sycl::info::device::local_mem_size>();

        bool usm_host = dev.has(sycl::aspect::usm_host_allocations);
        bool usm_shared = dev.has(sycl::aspect::usm_shared_allocations);
        bool usm_device = dev.has(sycl::aspect::usm_device_allocations);

        ProbeResult probe = probe_host_allocs(q, max_mb);

        if (!first_dev) {
            std::printf(",");
        }
        first_dev = false;

        std::printf("{\"id\":%zu", idx);
        std::printf(",\"name\":\"%s\"", json_escape(name).c_str());
        std::printf(",\"vendor\":\"%s\"", json_escape(vendor).c_str());
        std::printf(",\"driver\":\"%s\"", json_escape(driver).c_str());
        std::printf(",\"max_mem_alloc_size\":%llu", static_cast<unsigned long long>(max_alloc));
        std::printf(",\"global_mem_size\":%llu", static_cast<unsigned long long>(global_mem));
        std::printf(",\"local_mem_size\":%llu", static_cast<unsigned long long>(local_mem));
        std::printf(",\"usm\":{\"host\":%s,\"shared\":%s,\"device\":%s}",
                    usm_host ? "true" : "false",
                    usm_shared ? "true" : "false",
                    usm_device ? "true" : "false");

        std::printf(",\"pinned_probe\":{\"max_success_mb\":%zu,\"first_failure_mb\":%zu,\"attempts\":[",
                    probe.max_success_mb, probe.first_failure_mb);
        bool first_attempt = true;
        for (const auto & attempt : probe.attempts) {
            if (!first_attempt) {
                std::printf(",");
            }
            first_attempt = false;
            std::printf("{\"mb\":%zu,\"ok\":%s", attempt.mb, attempt.ok ? "true" : "false");
            if (!attempt.ok) {
                std::printf(",\"error\":\"%s\"", json_escape(attempt.error).c_str());
            }
            std::printf("}");
        }
        std::printf("]}");

        std::printf("}");
    }
    std::printf("]}\n");

    return 0;
}
