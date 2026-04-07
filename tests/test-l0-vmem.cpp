// test-l0-vmem.cpp
// Validates: Does BMG compute-runtime implement L0 virtual memory APIs?
// This is the P9 gate test — if this fails, P9 falls back to arena approach.
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -o test-l0-vmem tests/test-l0-vmem.cpp -lze_loader
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-l0-vmem
//
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <chrono>

static const char * ze_result_str(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return "SUCCESS";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_INVALID_SIZE: return "INVALID_SIZE";
        case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "INVALID_ENUMERATION";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "UNSUPPORTED_ALIGNMENT";
        case ZE_RESULT_ERROR_UNKNOWN: return "UNKNOWN";
        default: return "OTHER";
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Level Zero Virtual Memory API Test (P9 Gate)           ║\n");
    printf("║  Does BMG compute-runtime implement zeVirtualMem*?      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // ─── Get L0 handles via SYCL interop ──────────────
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto sycl_dev = q.get_device();
    auto sycl_ctx = q.get_context();

    printf("Device: %s\n", sycl_dev.get_info<sycl::info::device::name>().c_str());
    printf("VRAM: %.1f GB\n\n",
        sycl_dev.get_info<sycl::info::device::global_mem_size>() / 1e9);

    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_dev);
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_ctx);

    ze_result_t r;
    int pass_count = 0;
    int fail_count = 0;
    int total_tests = 0;

    auto check = [&](const char * name, ze_result_t result) {
        total_tests++;
        bool ok = (result == ZE_RESULT_SUCCESS);
        if (ok) pass_count++; else fail_count++;
        printf("  %s: %s (%s)\n", ok ? "PASS" : "FAIL", name, ze_result_str(result));
        return ok;
    };

    // ─── Test 1: Query page size ──────────────────────
    printf("--- Test 1: zeVirtualMemQueryPageSize ---\n");
    size_t page_size = 0;
    r = zeVirtualMemQueryPageSize(ze_ctx, ze_dev, 4 * 1024 * 1024, &page_size);
    if (check("Query page size for 4MB", r)) {
        printf("  Page size: %zu bytes (%zu KB)\n", page_size, page_size / 1024);
    }

    // Also try smaller sizes
    size_t page_64k = 0, page_2m = 0;
    r = zeVirtualMemQueryPageSize(ze_ctx, ze_dev, 64 * 1024, &page_64k);
    check("Query page size for 64KB", r);
    r = zeVirtualMemQueryPageSize(ze_ctx, ze_dev, 2 * 1024 * 1024, &page_2m);
    check("Query page size for 2MB", r);
    if (page_64k) printf("  64KB → page size %zu KB\n", page_64k / 1024);
    if (page_2m)  printf("  2MB  → page size %zu KB\n", page_2m / 1024);

    if (page_size == 0) {
        printf("\n  FATAL: Cannot determine page size. Aborting remaining tests.\n");
        printf("\n=== VERDICT: L0 Virtual Memory NOT SUPPORTED ===\n");
        return 1;
    }

    // ─── Test 2: Reserve virtual address range ────────
    printf("\n--- Test 2: zeVirtualMemReserve ---\n");
    size_t reserve_size = 64 * 1024 * 1024;  // 64 MB
    // Round up to page size
    reserve_size = ((reserve_size + page_size - 1) / page_size) * page_size;

    void * vaddr = nullptr;
    r = zeVirtualMemReserve(ze_ctx, nullptr, reserve_size, &vaddr);
    if (check("Reserve 64 MB virtual range", r)) {
        printf("  Virtual address: %p, size: %zu MB\n", vaddr, reserve_size / (1024*1024));
    } else {
        printf("\n  FATAL: Cannot reserve virtual memory. Aborting remaining tests.\n");
        printf("\n=== VERDICT: L0 Virtual Memory NOT SUPPORTED ===\n");
        return 1;
    }

    // ─── Test 3: Create physical memory ───────────────
    printf("\n--- Test 3: zePhysicalMemCreate ---\n");
    ze_physical_mem_handle_t phys = nullptr;
    ze_physical_mem_desc_t phys_desc = {};
    phys_desc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
    phys_desc.flags = 0;
    phys_desc.size = page_size;  // One page

    r = zePhysicalMemCreate(ze_ctx, ze_dev, &phys_desc, &phys);
    check("Create physical memory (1 page)", r);

    // Create a second page
    ze_physical_mem_handle_t phys2 = nullptr;
    r = zePhysicalMemCreate(ze_ctx, ze_dev, &phys_desc, &phys2);
    check("Create physical memory (2nd page)", r);

    if (!phys || !phys2) {
        printf("\n  FATAL: Cannot create physical memory. Aborting remaining tests.\n");
        if (vaddr) zeVirtualMemFree(ze_ctx, vaddr, reserve_size);
        printf("\n=== VERDICT: L0 Virtual Memory NOT SUPPORTED ===\n");
        return 1;
    }

    // ─── Test 4: Map virtual → physical ───────────────
    printf("\n--- Test 4: zeVirtualMemMap ---\n");
    r = zeVirtualMemMap(ze_ctx, vaddr, page_size, phys, 0,
                         ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    check("Map page 0 → physical", r);

    void * vaddr_page1 = (char *)vaddr + page_size;
    r = zeVirtualMemMap(ze_ctx, vaddr_page1, page_size, phys2, 0,
                         ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    check("Map page 1 → physical2", r);

    // ─── Test 5: GPU kernel reads/writes mapped memory ─
    printf("\n--- Test 5: GPU kernel on mapped virtual memory ---\n");
    {
        int * ptr = static_cast<int *>(vaddr);
        int n_ints = std::min((size_t)1024, page_size / sizeof(int));

        // Write pattern from GPU
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(n_ints), [=](sycl::id<1> i) {
                ptr[i] = static_cast<int>(i[0]) + 42;
            });
        }).wait();

        // Read back and verify via a second kernel
        auto * result = sycl::malloc_host<int>(1, q);
        result[0] = 0;
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(n_ints), [=](sycl::id<1> i) {
                if (ptr[i] != static_cast<int>(i[0]) + 42) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                        sycl::access::address_space::global_space> aref(result[0]);
                    aref.fetch_add(1);
                }
            });
        }).wait();

        total_tests++;
        if (result[0] == 0) {
            pass_count++;
            printf("  PASS: GPU kernel write+read on mapped virtual memory (0 errors)\n");
        } else {
            fail_count++;
            printf("  FAIL: GPU kernel read errors: %d\n", result[0]);
        }
        sycl::free(result, q);
    }

    // ─── Test 6: Unmap and remap to different physical ─
    printf("\n--- Test 6: Unmap + remap to different physical ---\n");
    {
        // Write a known value to page 0
        int * ptr = static_cast<int *>(vaddr);
        q.submit([&](sycl::handler & h) {
            h.single_task([=]() { ptr[0] = 12345; });
        }).wait();

        // Unmap page 0
        r = zeVirtualMemUnmap(ze_ctx, vaddr, page_size);
        check("Unmap page 0", r);

        // Remap page 0 to phys2 (which has page 1's data)
        r = zeVirtualMemMap(ze_ctx, vaddr, page_size, phys2, 0,
                             ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
        check("Remap page 0 → physical2", r);

        // Read from the remapped address — should see phys2's data, NOT 12345
        auto * result = sycl::malloc_host<int>(1, q);
        q.submit([&](sycl::handler & h) {
            h.single_task([=]() { result[0] = ptr[0]; });
        }).wait();

        total_tests++;
        if (result[0] != 12345) {
            pass_count++;
            printf("  PASS: After remap, address reads different physical data (got %d, not 12345)\n",
                result[0]);
        } else {
            // Might still be 12345 if phys2 happened to have that value — check with page 1 pattern
            fail_count++;
            printf("  AMBIGUOUS: Got 12345 — may be coincidence. Check manually.\n");
        }
        sycl::free(result, q);

        // Unmap for cleanup
        zeVirtualMemUnmap(ze_ctx, vaddr, page_size);
        zeVirtualMemUnmap(ze_ctx, vaddr_page1, page_size);
    }

    // ─── Test 7: Large reservation (1 GB) ─────────────
    printf("\n--- Test 7: Large virtual reservation (1 GB) ---\n");
    {
        size_t large_size = 1024ULL * 1024 * 1024;
        large_size = ((large_size + page_size - 1) / page_size) * page_size;
        void * large_vaddr = nullptr;
        r = zeVirtualMemReserve(ze_ctx, nullptr, large_size, &large_vaddr);
        if (check("Reserve 1 GB virtual range", r)) {
            printf("  Virtual address: %p\n", large_vaddr);
            zeVirtualMemFree(ze_ctx, large_vaddr, large_size);
            printf("  Freed 1 GB reservation OK\n");
        }
    }

    // ─── Cleanup ──────────────────────────────────────
    printf("\n--- Cleanup ---\n");
    r = zePhysicalMemDestroy(ze_ctx, phys);
    check("Destroy physical memory 1", r);
    r = zePhysicalMemDestroy(ze_ctx, phys2);
    check("Destroy physical memory 2", r);
    r = zeVirtualMemFree(ze_ctx, vaddr, reserve_size);
    check("Free virtual reservation", r);

    // ─── Verdict ──────────────────────────────────────
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  RESULTS: %d/%d tests passed                             \n", pass_count, total_tests);
    if (fail_count == 0) {
        printf("║  VERDICT: L0 Virtual Memory FULLY SUPPORTED on BMG     ║\n");
        printf("║  → P9 can proceed with zeVirtualMem-based KV cache     ║\n");
    } else {
        printf("║  VERDICT: L0 Virtual Memory PARTIALLY/NOT SUPPORTED    ║\n");
        printf("║  → P9 falls back to arena-based KV (P5 design)        ║\n");
    }
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return fail_count > 0 ? 1 : 0;
}
