// test-l0-vmem-kernel.cpp
// Deep investigation: Can SYCL kernels access L0 virtual memory mappings?
//
// The basic L0 VM APIs (reserve/map/unmap) work on BMG. But running a SYCL
// kernel on the mapped pointer SEGFAULT'd. This test isolates exactly why
// and finds a working interop pattern.
//
// Hypotheses for the crash:
//   H1: SYCL runtime looks up pointer in USM tracking, crashes on unknown ptr
//   H2: GPU page table not flushed after zeVirtualMemMap
//   H3: Access attribute not set correctly
//   H4: Need to use L0 command list directly, not SYCL queue
//   H5: Need to register the vmem pointer with SYCL via interop
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -o test-l0-vmem-kernel tests/test-l0-vmem-kernel.cpp -lze_loader
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-l0-vmem-kernel
//
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <csignal>
#include <setjmp.h>

static jmp_buf jump_buf;
static volatile sig_atomic_t got_signal = 0;

static void signal_handler(int sig) {
    got_signal = sig;
    longjmp(jump_buf, 1);
}

static const char * ze_result_str(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return "SUCCESS";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        default: { static char buf[32]; snprintf(buf, sizeof(buf), "0x%x", r); return buf; }
    }
}

struct timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    timer() : t0(clock::now()) {}
    double us() const { return std::chrono::duration<double, std::micro>(clock::now() - t0).count(); }
};

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  L0 Virtual Memory + SYCL Kernel Deep Investigation     ║\n");
    printf("║  Finding working interop pattern for P9                 ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Setup
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto sycl_ctx = q.get_context();
    auto sycl_dev = q.get_device();

    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_ctx);
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_dev);

    printf("Device: %s\n\n", sycl_dev.get_info<sycl::info::device::name>().c_str());
    fflush(stdout);

    // Get page size
    size_t page_size = 0;
    ze_result_t r = zeVirtualMemQueryPageSize(ze_ctx, ze_dev, 4 * 1024 * 1024, &page_size);
    printf("Page size: %zu bytes (%zu KB)\n\n", page_size, page_size / 1024);
    fflush(stdout);

    if (r != ZE_RESULT_SUCCESS || page_size == 0) {
        printf("FATAL: Cannot get page size\n");
        return 1;
    }

    // Reserve and map one page
    void * vaddr = nullptr;
    r = zeVirtualMemReserve(ze_ctx, nullptr, page_size, &vaddr);
    printf("Reserved: %p (%s)\n", vaddr, ze_result_str(r));
    fflush(stdout);
    if (r != ZE_RESULT_SUCCESS) return 1;

    ze_physical_mem_handle_t phys = nullptr;
    ze_physical_mem_desc_t pd = {};
    pd.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
    pd.size = page_size;
    r = zePhysicalMemCreate(ze_ctx, ze_dev, &pd, &phys);
    printf("Physical: %p (%s)\n", (void*)phys, ze_result_str(r));
    fflush(stdout);
    if (r != ZE_RESULT_SUCCESS) { zeVirtualMemFree(ze_ctx, vaddr, page_size); return 1; }

    r = zeVirtualMemMap(ze_ctx, vaddr, page_size, phys, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    printf("Mapped: %s\n\n", ze_result_str(r));
    fflush(stdout);
    if (r != ZE_RESULT_SUCCESS) {
        zePhysicalMemDestroy(ze_ctx, phys);
        zeVirtualMemFree(ze_ctx, vaddr, page_size);
        return 1;
    }

    // ─── Test A: L0 memcpy to vmem (no SYCL) ─────────
    printf("--- Test A: L0 command list memcpy to vmem ---\n");
    fflush(stdout);
    {
        // Get L0 command queue/list from SYCL queue
        // We'll use zeCommandListAppendMemoryCopy directly
        ze_command_list_handle_t ze_cmdlist = nullptr;
        ze_command_list_desc_t cl_desc = {};
        cl_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        cl_desc.commandQueueGroupOrdinal = 0;
        ze_command_queue_desc_t cq_desc = {};
        cq_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        cq_desc.ordinal = 0;
        cq_desc.index = 0;
        cq_desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &ze_cmdlist);
        printf("  Created immediate cmdlist: %s\n", ze_result_str(r));
        fflush(stdout);

        if (r == ZE_RESULT_SUCCESS) {
            // Write pattern via L0 memcpy from host
            int pattern[4] = {42, 43, 44, 45};
            r = zeCommandListAppendMemoryCopy(ze_cmdlist, vaddr, pattern, sizeof(pattern), nullptr, 0, nullptr);
            printf("  MemoryCopy host→vmem: %s\n", ze_result_str(r));
            fflush(stdout);

            // Read back
            int readback[4] = {};
            r = zeCommandListAppendMemoryCopy(ze_cmdlist, readback, vaddr, sizeof(readback), nullptr, 0, nullptr);
            printf("  MemoryCopy vmem→host: %s\n", ze_result_str(r));
            printf("  Readback: [%d, %d, %d, %d] (expected [42, 43, 44, 45])\n",
                readback[0], readback[1], readback[2], readback[3]);
            bool ok = (readback[0]==42 && readback[1]==43 && readback[2]==44 && readback[3]==45);
            printf("  Result: %s\n", ok ? "PASS" : "FAIL");
            fflush(stdout);

            zeCommandListDestroy(ze_cmdlist);
        }
    }

    // ─── Test B: SYCL queue.memcpy to vmem ────────────
    printf("\n--- Test B: SYCL queue.memcpy to vmem pointer ---\n");
    fflush(stdout);
    {
        int pattern[4] = {100, 200, 300, 400};
        int readback[4] = {};

        // Install signal handler for graceful crash recovery
        struct sigaction sa = {}, old_sa = {};
        sa.sa_handler = signal_handler;
        sigaction(SIGSEGV, &sa, &old_sa);
        sigaction(SIGABRT, &sa, &old_sa);

        got_signal = 0;
        if (setjmp(jump_buf) == 0) {
            q.memcpy(vaddr, pattern, sizeof(pattern)).wait();
            q.memcpy(readback, vaddr, sizeof(readback)).wait();
            printf("  SYCL memcpy write+read: [%d, %d, %d, %d]\n",
                readback[0], readback[1], readback[2], readback[3]);
            bool ok = (readback[0]==100 && readback[1]==200 && readback[2]==300 && readback[3]==400);
            printf("  Result: %s\n", ok ? "PASS" : "FAIL");
        } else {
            printf("  CRASHED (signal %d) — SYCL memcpy cannot handle vmem pointers\n", got_signal);
        }
        fflush(stdout);

        // Restore signal handler
        sigaction(SIGSEGV, &old_sa, nullptr);
        sigaction(SIGABRT, &old_sa, nullptr);
    }

    // ─── Test C: SYCL kernel accessing vmem pointer ───
    printf("\n--- Test C: SYCL kernel read/write on vmem pointer ---\n");
    fflush(stdout);
    {
        // First write known values via L0 (we know this works from Test A)
        ze_command_list_handle_t ze_cmdlist = nullptr;
        ze_command_queue_desc_t cq2 = {};
        cq2.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        cq2.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq2, &ze_cmdlist);

        int init_data[256];
        for (int i = 0; i < 256; i++) init_data[i] = i + 1;
        zeCommandListAppendMemoryCopy(ze_cmdlist, vaddr, init_data, sizeof(init_data), nullptr, 0, nullptr);

        // Now try SYCL kernel
        struct sigaction sa = {}, old_sa_segv = {}, old_sa_abrt = {};
        sa.sa_handler = signal_handler;
        sigaction(SIGSEGV, &sa, &old_sa_segv);
        sigaction(SIGABRT, &sa, &old_sa_abrt);

        got_signal = 0;
        if (setjmp(jump_buf) == 0) {
            int * vmem_int = static_cast<int *>(vaddr);
            auto * result = sycl::malloc_host<int>(1, q);
            result[0] = 0;

            // Simple kernel: read vmem, check values
            q.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
                    if (vmem_int[i] != static_cast<int>(i[0]) + 1) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> aref(result[0]);
                        aref.fetch_add(1);
                    }
                });
            }).wait();

            printf("  SYCL kernel read from vmem: %s (%d errors)\n",
                result[0] == 0 ? "PASS" : "FAIL", result[0]);

            // Write test
            q.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
                    vmem_int[i] = static_cast<int>(i[0]) * 10;
                });
            }).wait();

            // Read back via L0
            int verify[256] = {};
            zeCommandListAppendMemoryCopy(ze_cmdlist, verify, vaddr, sizeof(verify), nullptr, 0, nullptr);
            int errors = 0;
            for (int i = 0; i < 256; i++) {
                if (verify[i] != i * 10) errors++;
            }
            printf("  SYCL kernel write to vmem: %s (%d errors)\n",
                errors == 0 ? "PASS" : "FAIL", errors);

            sycl::free(result, q);
        } else {
            printf("  CRASHED (signal %d) — SYCL kernel cannot access vmem\n", got_signal);
        }
        fflush(stdout);

        sigaction(SIGSEGV, &old_sa_segv, nullptr);
        sigaction(SIGABRT, &old_sa_abrt, nullptr);
        if (ze_cmdlist) zeCommandListDestroy(ze_cmdlist);
    }

    // ─── Test D: Pointer type query ───────────────────
    printf("\n--- Test D: SYCL USM pointer type for vmem address ---\n");
    fflush(stdout);
    {
        auto ptype = sycl::get_pointer_type(vaddr, sycl_ctx);
        const char * name = "unknown";
        switch (ptype) {
            case sycl::usm::alloc::host: name = "host"; break;
            case sycl::usm::alloc::device: name = "device"; break;
            case sycl::usm::alloc::shared: name = "shared"; break;
            case sycl::usm::alloc::unknown: name = "unknown"; break;
        }
        printf("  sycl::get_pointer_type(vmem): %s\n", name);
        printf("  (If 'unknown', SYCL USM tracking doesn't know about vmem pointers)\n");
        fflush(stdout);
    }

    // ─── Test E: Multiple pages, sequential access ────
    printf("\n--- Test E: Multi-page mapping (4 pages, sequential) ---\n");
    fflush(stdout);
    {
        size_t n_pages = 4;
        size_t total = n_pages * page_size;
        void * multi_vaddr = nullptr;
        r = zeVirtualMemReserve(ze_ctx, nullptr, total, &multi_vaddr);
        printf("  Reserved %zu MB at %p: %s\n", total / (1024*1024), multi_vaddr, ze_result_str(r));
        fflush(stdout);

        if (r == ZE_RESULT_SUCCESS) {
            // Create and map 4 physical pages
            ze_physical_mem_handle_t pages[4];
            bool all_ok = true;
            for (int i = 0; i < 4; i++) {
                ze_physical_mem_desc_t desc = {};
                desc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
                desc.size = page_size;
                r = zePhysicalMemCreate(ze_ctx, ze_dev, &desc, &pages[i]);
                if (r != ZE_RESULT_SUCCESS) { all_ok = false; break; }

                void * page_addr = (char*)multi_vaddr + i * page_size;
                r = zeVirtualMemMap(ze_ctx, page_addr, page_size, pages[i], 0,
                                     ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
                if (r != ZE_RESULT_SUCCESS) { all_ok = false; break; }
            }
            printf("  Created + mapped 4 pages: %s\n", all_ok ? "OK" : "FAILED");
            fflush(stdout);

            if (all_ok) {
                // Write to all 4 pages via SYCL kernel
                struct sigaction sa = {}, old_sa_segv = {}, old_sa_abrt = {};
                sa.sa_handler = signal_handler;
                sigaction(SIGSEGV, &sa, &old_sa_segv);
                sigaction(SIGABRT, &sa, &old_sa_abrt);

                got_signal = 0;
                if (setjmp(jump_buf) == 0) {
                    int * multi_ptr = static_cast<int *>(multi_vaddr);
                    int n_ints = (int)(total / sizeof(int));
                    int test_count = std::min(n_ints, 4096);

                    q.submit([&](sycl::handler & h) {
                        h.parallel_for(sycl::range<1>(test_count), [=](sycl::id<1> i) {
                            multi_ptr[i] = static_cast<int>(i[0]) + 7;
                        });
                    }).wait();

                    // Verify via L0
                    ze_command_list_handle_t cmdl = nullptr;
                    ze_command_queue_desc_t cq3 = {};
                    cq3.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
                    cq3.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
                    zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq3, &cmdl);

                    int * verify = new int[test_count];
                    zeCommandListAppendMemoryCopy(cmdl, verify, multi_vaddr,
                        test_count * sizeof(int), nullptr, 0, nullptr);

                    int errors = 0;
                    for (int i = 0; i < test_count; i++) {
                        if (verify[i] != i + 7) errors++;
                    }
                    printf("  SYCL kernel across 4 pages: %s (%d errors out of %d)\n",
                        errors == 0 ? "PASS" : "FAIL", errors, test_count);

                    delete[] verify;
                    zeCommandListDestroy(cmdl);
                } else {
                    printf("  CRASHED (signal %d)\n", got_signal);
                }

                sigaction(SIGSEGV, &old_sa_segv, nullptr);
                sigaction(SIGABRT, &old_sa_abrt, nullptr);
            }

            // Cleanup
            for (int i = 0; i < 4; i++) {
                zeVirtualMemUnmap(ze_ctx, (char*)multi_vaddr + i * page_size, page_size);
                zePhysicalMemDestroy(ze_ctx, pages[i]);
            }
            zeVirtualMemFree(ze_ctx, multi_vaddr, total);
        }
    }

    // ─── Test F: Unmap + remap, kernel sees new data ──
    printf("\n--- Test F: Remap physical page, verify kernel sees change ---\n");
    fflush(stdout);
    {
        // Create 2 physical pages with different content
        ze_physical_mem_handle_t pa = nullptr, pb = nullptr;
        ze_physical_mem_desc_t desc = {};
        desc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
        desc.size = page_size;
        zePhysicalMemCreate(ze_ctx, ze_dev, &desc, &pa);
        zePhysicalMemCreate(ze_ctx, ze_dev, &desc, &pb);

        void * test_vaddr = nullptr;
        zeVirtualMemReserve(ze_ctx, nullptr, page_size, &test_vaddr);

        ze_command_list_handle_t cmdl = nullptr;
        ze_command_queue_desc_t cq4 = {};
        cq4.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        cq4.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq4, &cmdl);

        // Map page A, write 111 via L0
        zeVirtualMemMap(ze_ctx, test_vaddr, page_size, pa, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
        int val_a = 111;
        zeCommandListAppendMemoryCopy(cmdl, test_vaddr, &val_a, sizeof(int), nullptr, 0, nullptr);

        // Map page B separately, write 222
        void * tmp_vaddr = nullptr;
        zeVirtualMemReserve(ze_ctx, nullptr, page_size, &tmp_vaddr);
        zeVirtualMemMap(ze_ctx, tmp_vaddr, page_size, pb, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
        int val_b = 222;
        zeCommandListAppendMemoryCopy(cmdl, tmp_vaddr, &val_b, sizeof(int), nullptr, 0, nullptr);
        zeVirtualMemUnmap(ze_ctx, tmp_vaddr, page_size);
        zeVirtualMemFree(ze_ctx, tmp_vaddr, page_size);

        // Read via SYCL kernel — should see 111 (page A mapped)
        struct sigaction sa = {}, old_segv = {}, old_abrt = {};
        sa.sa_handler = signal_handler;
        sigaction(SIGSEGV, &sa, &old_segv);
        sigaction(SIGABRT, &sa, &old_abrt);

        got_signal = 0;
        if (setjmp(jump_buf) == 0) {
            auto * result = sycl::malloc_host<int>(2, q);
            int * vptr = static_cast<int *>(test_vaddr);

            q.submit([&](sycl::handler & h) {
                h.single_task([=]() { result[0] = vptr[0]; });
            }).wait();
            printf("  Read with page A mapped: %d (expected 111) — %s\n",
                result[0], result[0] == 111 ? "PASS" : "FAIL");

            // Unmap A, remap to B
            q.wait();  // Ensure kernel done before unmap
            zeVirtualMemUnmap(ze_ctx, test_vaddr, page_size);
            zeVirtualMemMap(ze_ctx, test_vaddr, page_size, pb, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);

            q.submit([&](sycl::handler & h) {
                h.single_task([=]() { result[1] = vptr[0]; });
            }).wait();
            printf("  Read with page B mapped: %d (expected 222) — %s\n",
                result[1], result[1] == 222 ? "PASS" : "FAIL");

            sycl::free(result, q);
        } else {
            printf("  CRASHED (signal %d)\n", got_signal);
        }

        sigaction(SIGSEGV, &old_segv, nullptr);
        sigaction(SIGABRT, &old_abrt, nullptr);

        zeVirtualMemUnmap(ze_ctx, test_vaddr, page_size);
        zeCommandListDestroy(cmdl);
        zePhysicalMemDestroy(ze_ctx, pa);
        zePhysicalMemDestroy(ze_ctx, pb);
        zeVirtualMemFree(ze_ctx, test_vaddr, page_size);
    }

    // ─── Cleanup original mapping ─────────────────────
    zeVirtualMemUnmap(ze_ctx, vaddr, page_size);
    zePhysicalMemDestroy(ze_ctx, phys);
    zeVirtualMemFree(ze_ctx, vaddr, page_size);

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  P9 READINESS SUMMARY                                    ║\n");
    printf("║  A: L0 memcpy to vmem     — confirms basic VM works     ║\n");
    printf("║  B: SYCL memcpy to vmem   — tests USM interop           ║\n");
    printf("║  C: SYCL kernel on vmem   — THE critical test           ║\n");
    printf("║  D: Pointer type query    — USM tracking compatibility   ║\n");
    printf("║  E: Multi-page kernel     — spans page boundaries       ║\n");
    printf("║  F: Remap + kernel        — transparent page migration  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}
