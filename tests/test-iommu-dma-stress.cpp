// Opt-in SYCL pinned-host DMA stress test for IOMMU mapping issues.
// WARNING: This test can hang or crash the GPU driver. It only runs when
// GGML_SYCL_IOMMU_STRESS is set to a non-zero value.

#include <sycl/sycl.hpp>

#include "ggml-sycl/pinned-pool.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static size_t parse_env_mb(const char * name, size_t def_mb) {
    const char * val = std::getenv(name);
    if (!val || !*val) {
        return def_mb;
    }
    char * end = nullptr;
    long long parsed = std::strtoll(val, &end, 10);
    if (end == val || parsed <= 0) {
        std::fprintf(stderr, "WARN: %s='%s' invalid, using %zu MB\n", name, val, def_mb);
        return def_mb;
    }
    return static_cast<size_t>(parsed);
}

static int parse_env_int(const char * name, int def_val) {
    const char * val = std::getenv(name);
    if (!val || !*val) {
        return def_val;
    }
    char * end = nullptr;
    long parsed = std::strtol(val, &end, 10);
    if (end == val) {
        std::fprintf(stderr, "WARN: %s='%s' invalid, using %d\n", name, val, def_val);
        return def_val;
    }
    return static_cast<int>(parsed);
}

static bool parse_env_bool(const char * name, bool def_val) {
    const char * val = std::getenv(name);
    if (!val || !*val) {
        return def_val;
    }
    if (std::strcmp(val, "0") == 0 || std::strcmp(val, "false") == 0 || std::strcmp(val, "no") == 0) {
        return false;
    }
    return true;
}

static const char * parse_env_mode() {
    const char * mode = std::getenv("GGML_SYCL_IOMMU_MODE");
    if (!mode || !*mode) {
        return "raw";
    }
    return mode;
}

static bool mode_is(const char * mode, const char * expected) {
    return std::strcmp(mode, expected) == 0;
}

int main() {
    const char * enable = std::getenv("GGML_SYCL_IOMMU_STRESS");
    if (!enable || std::strcmp(enable, "0") == 0) {
        std::printf("SKIP: set GGML_SYCL_IOMMU_STRESS=1 to run this test\n");
        return 0;
    }

    const size_t chunk_mb   = parse_env_mb("GGML_SYCL_IOMMU_CHUNK_MB", 1024);
    const int    chunks     = std::max(1, parse_env_int("GGML_SYCL_IOMMU_CHUNKS", 1));
    const int    iters      = std::max(1, parse_env_int("GGML_SYCL_IOMMU_ITERS", 1));
    const size_t work_items = static_cast<size_t>(
        std::max(1, parse_env_int("GGML_SYCL_IOMMU_WORK_ITEMS", 1 << 20)));
    const size_t slice_mb   = parse_env_mb("GGML_SYCL_IOMMU_DMA_SLICE_MB", 0);
    const bool   do_kernel  = parse_env_bool("GGML_SYCL_IOMMU_KERNEL", true);
    const bool   copy_back  = parse_env_bool("GGML_SYCL_IOMMU_COPY_BACK", true);
    const bool   touch_pages = parse_env_bool("GGML_SYCL_IOMMU_TOUCH", true);
    const char * mode       = parse_env_mode();

    std::printf("IOMMU/DMA Stress Test (opt-in)\n");
    std::printf("  chunk_mb=%zu, chunks=%d, iters=%d, work_items=%zu\n", chunk_mb, chunks, iters, work_items);
    std::printf("  kernel=%s, copy_back=%s, mode=%s\n",
                do_kernel ? "on" : "off",
                copy_back ? "on" : "off",
                mode);
    if (slice_mb > 0) {
        std::printf("  dma_slice_mb=%zu\n", slice_mb);
    }

    try {
        sycl::queue q(sycl::default_selector_v);
        const auto dev = q.get_device();
        const auto dev_name = dev.get_info<sycl::info::device::name>();
        std::printf("SYCL device: %s\n", dev_name.c_str());

        const size_t chunk_bytes = chunk_mb * 1024ULL * 1024ULL;
        const size_t page        = 4096;

        const bool mode_cache  = mode_is(mode, "cache") || mode_is(mode, "pinned") || mode_is(mode, "dma");
        const bool mode_pinned = mode_is(mode, "pinned");
        const bool mode_dma    = mode_is(mode, "dma");
        const bool mode_raw    = mode_is(mode, "raw");
        const bool mode_probe  = mode_is(mode, "probe");
        const bool mode_shared = mode_is(mode, "shared");

        if (!mode_cache && !mode_raw && !mode_probe && !mode_shared) {
            std::fprintf(stderr, "FAIL: unknown GGML_SYCL_IOMMU_MODE='%s' (use raw|cache|pinned|dma|probe|shared)\n",
                         mode);
            return 1;
        }

        if (mode_probe) {
            const char * chunk_env = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
            if (!chunk_env || !*chunk_env) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%zu", chunk_mb);
                setenv("GGML_SYCL_PINNED_CHUNK_MB", buf, 1);
            }

            const size_t pool_mb = parse_env_mb("GGML_SYCL_IOMMU_POOL_MB", chunk_mb * static_cast<size_t>(chunks));
            ggml_sycl::pinned_chunk_pool pool(q, pool_mb * 1024ULL * 1024ULL);

            std::vector<void *> ptrs;
            ptrs.reserve(chunks);
            std::printf("  probe-only: pool_mb=%zu, touch=%s\n", pool_mb, touch_pages ? "on" : "off");

            for (int c = 0; c < chunks; ++c) {
                std::printf("  probe: allocating chunk %d/%d (%.1f MB)\n", c + 1, chunks, chunk_bytes / (1024.0 * 1024.0));
                std::fflush(stdout);
                void * ptr = pool.allocate(chunk_bytes);
                if (!ptr) {
                    std::fprintf(stderr, "FAIL: probe allocation failed at chunk %d\n", c);
                    return 1;
                }
                ptrs.push_back(ptr);
                if (touch_pages) {
                    auto * bytes = static_cast<unsigned char *>(ptr);
                    for (size_t off = 0; off < chunk_bytes; off += page) {
                        bytes[off] = static_cast<unsigned char>(off & 0xFF);
                    }
                }
                std::printf("  probe: chunk %d allocated\n", c + 1);
                std::fflush(stdout);
            }

            std::printf("PASS: probe allocated %d chunk(s)\n", chunks);
            return 0;
        }

        if (mode_shared) {
            std::printf("  shared-only: using USM shared allocations\n");
        }

        ggml_sycl::host_cache *    host_cache   = nullptr;
        ggml_sycl::unified_cache * device_cache = nullptr;

        if (mode_cache) {
            const size_t cache_mb =
                parse_env_mb("GGML_SYCL_IOMMU_CACHE_MB", chunk_mb * static_cast<size_t>(chunks) + 256);
            ggml_sycl::set_unified_cache_budget(cache_mb * 1024ULL * 1024ULL);
            ggml_sycl::set_unified_cache_host_budget_pct(90);

            host_cache = ggml_sycl::get_host_cache(q);
            if (!host_cache) {
                std::fprintf(stderr, "SKIP: host cache unavailable\n");
                return 0;
            }
            if (!mode_pinned && !mode_dma) {
                device_cache = ggml_sycl::get_unified_cache(q);
                if (!device_cache) {
                    std::fprintf(stderr, "SKIP: unified cache unavailable\n");
                    return 0;
                }
            }
            std::printf("  cache_mb=%zu\n", cache_mb);
        }

        bool use_device = mode_raw || mode_dma || mode_is(mode, "cache");
        bool use_kernel = do_kernel;
        bool use_copy   = copy_back;

        if (mode_pinned) {
            use_device = false;
            use_kernel = false;
            use_copy   = false;
            std::printf("  pinned-only: skipping device alloc, DMA, and kernel\n");
        } else if (mode_dma) {
            use_kernel = false;
            std::printf("  dma-only: skipping kernel\n");
        } else if (mode_shared) {
            use_device = false;
            use_copy   = false;
            std::printf("  shared-only: skipping device alloc and DMA\n");
        }

        for (int iter = 0; iter < iters; ++iter) {
            std::printf("Iteration %d/%d\n", iter + 1, iters);
            std::vector<void *> host_ptrs;
            std::vector<void *> dev_ptrs;
            std::vector<std::vector<uint8_t>> src_buffers;
            std::vector<uint64_t> keys(static_cast<size_t>(chunks));
            host_ptrs.reserve(chunks);
            dev_ptrs.reserve(chunks);
            src_buffers.reserve(chunks);

            for (int c = 0; c < chunks; ++c) {
                keys[static_cast<size_t>(c)] = static_cast<uint64_t>(c + 1);
                const void * key_ptr = &keys[static_cast<size_t>(c)];
                void * host = nullptr;
                void * dev_buf = nullptr;

                if (mode_shared) {
                    host = sycl::malloc_shared(chunk_bytes, q);
                    if (!host) {
                        std::fprintf(stderr, "FAIL: sycl::malloc_shared(%zu) failed at chunk %d\n", chunk_bytes, c);
                        break;
                    }

                    if (touch_pages) {
                        auto * bytes = static_cast<unsigned char *>(host);
                        for (size_t off = 0; off < chunk_bytes; off += page) {
                            bytes[off] = static_cast<unsigned char>(off & 0xFF);
                        }
                    }
                } else if (mode_cache) {
                    src_buffers.emplace_back(chunk_bytes);
                    auto & src = src_buffers.back();
                    if (touch_pages) {
                        for (size_t off = 0; off < chunk_bytes; off += page) {
                            src[off] = static_cast<uint8_t>(off & 0xFF);
                        }
                    }

                    bool needs_fill = false;
                    bool pinned_alloc = false;
                    ggml_sycl::cache_location location = ggml_sycl::cache_location::HOST_MMAP;
                    host = host_cache->ensure_cached_alloc(
                        key_ptr,
                        src.data(),
                        src.size(),
                        chunk_bytes,
                        ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                        -1,
                        -1,
                        GGML_LAYOUT_AOS,
                        false,
                        &needs_fill,
                        &pinned_alloc,
                        &location,
                        nullptr);
                    if (!host) {
                        std::fprintf(stderr, "FAIL: host_cache alloc failed at chunk %d\n", c);
                        break;
                    }
                    if (needs_fill) {
                        std::memcpy(host, src.data(), chunk_bytes);
                    }
                } else {
                    host = sycl::malloc_host(chunk_bytes, q);
                    if (!host) {
                        std::fprintf(stderr, "FAIL: sycl::malloc_host(%zu) failed at chunk %d\n", chunk_bytes, c);
                        break;
                    }

                    if (touch_pages) {
                        auto * bytes = static_cast<unsigned char *>(host);
                        for (size_t off = 0; off < chunk_bytes; off += page) {
                            bytes[off] = static_cast<unsigned char>(off & 0xFF);
                        }
                    }
                }

                host_ptrs.push_back(host);

                if (mode_shared) {
                    dev_buf = host;
                } else if (use_device && mode_cache && !mode_dma) {
                    bool dev_needs_fill = false;
                    dev_buf = device_cache->ensure_cached_alloc(
                        key_ptr,
                        host,
                        chunk_bytes,
                        chunk_bytes,
                        ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                        -1,
                        -1,
                        GGML_LAYOUT_AOS,
                        false,
                        &dev_needs_fill);
                    if (!dev_buf) {
                        std::fprintf(stderr, "FAIL: device cache alloc failed at chunk %d\n", c);
                        break;
                    }
                    if (dev_needs_fill) {
                        q.memcpy(dev_buf, host, chunk_bytes);
                    }
                } else if (use_device) {
                    size_t dev_bytes = chunk_bytes;
                    if (mode_dma && slice_mb > 0) {
                        const size_t slice_bytes = slice_mb * 1024ULL * 1024ULL;
                        dev_bytes = std::min(slice_bytes, chunk_bytes);
                        std::printf("  dma-only: using device slice buffer %.1f MB\n",
                                    dev_bytes / (1024.0 * 1024.0));
                    }
                    dev_buf = sycl::malloc_device(dev_bytes, q);
                    if (!dev_buf) {
                        std::fprintf(stderr, "FAIL: sycl::malloc_device(%zu) failed at chunk %d\n", dev_bytes, c);
                        break;
                    }
                    if (mode_dma && slice_mb > 0) {
                        const size_t slice_bytes = std::min(dev_bytes, chunk_bytes);
                        for (size_t off = 0; off < chunk_bytes; off += slice_bytes) {
                            const size_t copy_size = std::min(slice_bytes, chunk_bytes - off);
                            q.memcpy(static_cast<char *>(dev_buf),
                                     static_cast<const char *>(host) + off,
                                     copy_size);
                            if (use_copy) {
                                q.memcpy(static_cast<char *>(host) + off,
                                         static_cast<const char *>(dev_buf),
                                         copy_size);
                            }
                        }
                    } else {
                        q.memcpy(dev_buf, host, chunk_bytes);
                    }
                }

                if (use_device) {
                    dev_ptrs.push_back(dev_buf);
                }

                if ((use_device || mode_shared) && use_kernel) {
                    const size_t stride = std::max<size_t>(1, std::min(chunk_bytes, work_items));
                    q.parallel_for(sycl::range<1>(stride), [=](sycl::id<1> idx) {
                        auto * d = static_cast<unsigned char *>(dev_buf);
                        const size_t start = static_cast<size_t>(idx[0]);
                        for (size_t off = start; off < chunk_bytes; off += stride) {
                            d[off] = static_cast<unsigned char>(d[off] + 1);
                        }
                    });
                }
                if (use_device && use_copy && !(mode_dma && slice_mb > 0)) {
                    q.memcpy(host, dev_buf, chunk_bytes);
                }
            }

            try {
                q.wait_and_throw();
            } catch (const std::exception & e) {
                std::fprintf(stderr, "SYCL error: %s\n", e.what());
                for (void * ptr : dev_ptrs) {
                    if (mode_cache) {
                        continue;
                    }
                    sycl::free(ptr, q);
                }
                for (void * ptr : host_ptrs) {
                    if (mode_cache) {
                        continue;
                    }
                    sycl::free(ptr, q);
                }
                return 1;
            }

            // Basic sanity: sample first byte of each host buffer.
            for (size_t i = 0; i < host_ptrs.size(); ++i) {
                const unsigned char * bytes = static_cast<const unsigned char *>(host_ptrs[i]);
                const unsigned char v = bytes[0];
                (void) v;
            }

            for (void * ptr : dev_ptrs) {
                if (mode_cache) {
                    continue;
                }
                sycl::free(ptr, q);
            }
            for (void * ptr : host_ptrs) {
                if (mode_cache) {
                    continue;
                }
                sycl::free(ptr, q);
            }

            if (mode_cache) {
                for (int c = 0; c < static_cast<int>(host_ptrs.size()); ++c) {
                    const void * key_ptr = &keys[static_cast<size_t>(c)];
                    if (!mode_pinned && device_cache) {
                        device_cache->remove(
                            key_ptr,
                            ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                            -1,
                            -1,
                            GGML_LAYOUT_AOS);
                    }
                    host_cache->remove(
                        key_ptr,
                        ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                        -1,
                        -1,
                        GGML_LAYOUT_AOS);
                }
            }
        }

        std::printf("PASS: completed %d iteration(s)\n", iters);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
}
