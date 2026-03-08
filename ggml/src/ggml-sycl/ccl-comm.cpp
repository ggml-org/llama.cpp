#include "ccl-comm.hpp"
#include "common.hpp"

#if GGML_SYCL_CCL

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <chrono>

// File-based KVS address sharing for multi-process CCL
static const char* KVS_ADDR_FILE = "/tmp/ggml_sycl_ccl_kvs_address.bin";

// Global CCL context
ggml_sycl_ccl_context g_ccl_ctx;

// Debug flag for CCL operations
static int g_ccl_debug = -1;

static bool ccl_debug_enabled() {
    if (g_ccl_debug < 0) {
        const char* env = std::getenv("GGML_SYCL_CCL_DEBUG");
        g_ccl_debug = (env && std::string(env) == "1") ? 1 : 0;
    }
    return g_ccl_debug == 1;
}

void ggml_sycl_ccl_init(int world_size, queue_ptr* queues) {
    if (g_ccl_ctx.initialized) {
        return;  // Already initialized
    }

    if (world_size < 2) {
        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: world_size=%d, skipping CCL init (need at least 2)\n", world_size);
        }
        return;
    }

    // NOTE: oneCCL is designed for multi-process (MPI) execution where each GPU
    // runs in a separate process. In our single-process multi-device model,
    // CCL communicator creation will hang waiting for other processes.
    //
    // CCL initialization is DISABLED until we implement multi-process execution
    // (Phase 5 of the oneCCL TP plan). For now, we continue using the manual
    // malloc_shared + GPU kernel ALL_REDUCE approach.
    //
    // To enable CCL (requires multi-process launcher like mpirun):
    //   export GGML_SYCL_CCL_ENABLE=1

    const char* ccl_enable = std::getenv("GGML_SYCL_CCL_ENABLE");
    if (!ccl_enable || std::string(ccl_enable) != "1") {
        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Skipping CCL init (single-process mode, set GGML_SYCL_CCL_ENABLE=1 with mpirun)\n");
        }
        GGML_UNUSED(queues);
        return;
    }

    try {
        // Initialize CCL
        ccl::init();

        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Initializing with world_size=%d\n", world_size);
        }

        // Create KVS (Key-Value Store) for process coordination
        // In single-process multi-device mode, we use the main KVS
        ccl::shared_ptr_class<ccl::kvs> kvs = ccl::create_main_kvs();

        // Create communicators and streams for each device
        g_ccl_ctx.comms.reserve(world_size);
        g_ccl_ctx.streams.reserve(world_size);

        for (int rank = 0; rank < world_size; rank++) {
            if (!queues[rank]) {
                fprintf(stderr, "SYCL CCL: ERROR - null queue for rank %d\n", rank);
                continue;
            }

            // Create CCL device and context from SYCL queue
            auto sycl_dev = queues[rank]->get_device();
            auto sycl_ctx = queues[rank]->get_context();

            ccl::device ccl_dev = ccl::create_device(sycl_dev);
            ccl::context ccl_ctx = ccl::create_context(sycl_ctx);

            // Create communicator for this rank
            // world_size = total devices, rank = this device's ID
            auto comm = ccl::create_communicator(world_size, rank, ccl_dev, ccl_ctx, kvs);
            g_ccl_ctx.comms.push_back(std::move(comm));

            // Create CCL stream wrapping the SYCL queue
            auto stream = ccl::create_stream(*queues[rank]);
            g_ccl_ctx.streams.push_back(std::move(stream));

            if (ccl_debug_enabled()) {
                fprintf(stderr, "SYCL CCL: Created communicator for rank %d/%d\n", rank, world_size);
            }
        }

        g_ccl_ctx.world_size = world_size;
        g_ccl_ctx.initialized = true;

        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Initialized successfully with %d devices\n", world_size);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "SYCL CCL: ERROR during initialization: %s\n", e.what());
        g_ccl_ctx.initialized = false;
    }
}

void ggml_sycl_ccl_init_multiprocess(int rank, int world_size, queue_ptr queue) {
    if (g_ccl_ctx.initialized) {
        return;  // Already initialized
    }

    if (world_size < 2 || rank < 0 || rank >= world_size) {
        fprintf(stderr, "SYCL CCL: Invalid multi-process config (rank=%d, world_size=%d)\n",
                rank, world_size);
        return;
    }

    if (!queue) {
        fprintf(stderr, "SYCL CCL: ERROR - null queue for rank %d\n", rank);
        return;
    }

    // Check if CCL should be enabled (default: enabled in multi-process mode)
    const char* ccl_disable = std::getenv("GGML_SYCL_CCL_DISABLE");
    if (ccl_disable && std::string(ccl_disable) == "1") {
        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Multi-process CCL disabled by GGML_SYCL_CCL_DISABLE=1\n");
        }
        return;
    }

    if (ccl_debug_enabled()) {
        fprintf(stderr, "SYCL CCL: Starting multi-process init - rank %d/%d\n", rank, world_size);
    }

    try {
        // Initialize CCL
        ccl::init();

        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Rank %d/%d - ccl::init() complete\n", rank, world_size);
        }

        // Create KVS (Key-Value Store) for inter-process coordination
        // Use file-based KVS address sharing:
        // - Rank 0 creates main KVS and writes address to file
        // - Other ranks read address from file and connect to the same KVS
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type kvs_addr;

        if (rank == 0) {
            // Rank 0: Create main KVS and write address to file
            kvs = ccl::create_main_kvs();
            kvs_addr = kvs->get_address();

            // Write address to file for other ranks
            std::ofstream ofs(KVS_ADDR_FILE, std::ios::binary);
            if (!ofs.good()) {
                fprintf(stderr, "SYCL CCL: ERROR - cannot write KVS address file\n");
                return;
            }
            ofs.write(reinterpret_cast<const char*>(kvs_addr.data()), kvs_addr.size());
            ofs.close();

            if (ccl_debug_enabled()) {
                fprintf(stderr, "SYCL CCL: Rank 0 created main KVS, address written to %s\n", KVS_ADDR_FILE);
            }
        } else {
            // Other ranks: Wait for KVS address file and read it
            if (ccl_debug_enabled()) {
                fprintf(stderr, "SYCL CCL: Rank %d waiting for KVS address file...\n", rank);
            }

            // Wait for file to exist (with timeout)
            int wait_count = 0;
            const int max_wait = 100;  // 10 second timeout
            while (wait_count < max_wait) {
                std::ifstream test(KVS_ADDR_FILE);
                if (test.good()) {
                    test.close();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                wait_count++;
            }

            if (wait_count >= max_wait) {
                fprintf(stderr, "SYCL CCL: ERROR - Rank %d timeout waiting for KVS address file\n", rank);
                return;
            }

            // Small delay to ensure file is fully written
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Read KVS address from file
            std::ifstream ifs(KVS_ADDR_FILE, std::ios::binary);
            if (!ifs.good()) {
                fprintf(stderr, "SYCL CCL: ERROR - cannot read KVS address file\n");
                return;
            }
            ifs.read(reinterpret_cast<char*>(kvs_addr.data()), 256);
            ifs.close();

            // Create KVS from address
            kvs = ccl::create_kvs(kvs_addr);

            if (ccl_debug_enabled()) {
                fprintf(stderr, "SYCL CCL: Rank %d connected to KVS from file\n", rank);
            }
        }

        // Create CCL device and context from this process's SYCL queue
        auto sycl_dev = queue->get_device();
        auto sycl_ctx = queue->get_context();

        ccl::device ccl_dev = ccl::create_device(sycl_dev);
        ccl::context ccl_ctx = ccl::create_context(sycl_ctx);

        // Create single communicator for this process
        // Each process has exactly one communicator with its rank
        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Rank %d creating communicator...\n", rank);
        }

        auto comm = ccl::create_communicator(world_size, rank, ccl_dev, ccl_ctx, kvs);
        g_ccl_ctx.comms.clear();
        g_ccl_ctx.comms.push_back(std::move(comm));

        // Create CCL stream wrapping this process's SYCL queue
        auto stream = ccl::create_stream(*queue);
        g_ccl_ctx.streams.clear();
        g_ccl_ctx.streams.push_back(std::move(stream));

        g_ccl_ctx.world_size = world_size;
        g_ccl_ctx.rank = rank;
        g_ccl_ctx.initialized = true;

        // Register atexit handler to ensure CCL resources are freed before MPI finalization
        // This is crucial because CCL resources must persist across backend lifetimes
        // (the "fit params" step creates/destroys temporary backends)
        static bool atexit_registered = false;
        if (!atexit_registered) {
            std::atexit(ggml_sycl_ccl_free);
            atexit_registered = true;
        }

        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Multi-process initialized - rank %d/%d ready\n",
                    rank, world_size);
        }

        // Cleanup KVS file after all ranks have connected
        // (rank 0 does this after a delay to ensure all ranks have read)
        if (rank == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::remove(KVS_ADDR_FILE);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "SYCL CCL: ERROR during multi-process init: %s\n", e.what());
        g_ccl_ctx.initialized = false;
    }
}

void ggml_sycl_ccl_allreduce_sum_f32(float* buf, size_t count, int device) {
    // In-place version: send_buf == recv_buf
    ggml_sycl_ccl_allreduce_sum_f32(buf, buf, count, device);
}

void ggml_sycl_ccl_allreduce_sum_f32(const float* send_buf, float* recv_buf, size_t count, int device) {
    if (!g_ccl_ctx.initialized) {
        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: allreduce called but CCL not initialized\n");
        }
        return;
    }

    // In multi-process mode, each process has only ONE communicator at index 0
    // The 'device' parameter is ignored since this process owns only one device
    int comm_idx = 0;
    if (g_ccl_ctx.comms.size() > 1) {
        // Single-process multi-device mode (legacy, not currently used)
        comm_idx = device;
        if (device < 0 || device >= (int)g_ccl_ctx.comms.size()) {
            fprintf(stderr, "SYCL CCL: ERROR - invalid device %d (num_comms=%zu)\n",
                    device, g_ccl_ctx.comms.size());
            return;
        }
    }

    try {
        auto& comm = g_ccl_ctx.comms[comm_idx];
        auto& stream = g_ccl_ctx.streams[comm_idx];

        // Get the SYCL queue from CCL stream to check pointer types
        // CCL requires device memory for GPU allreduce
        // CRITICAL: Stage ALL non-device memory due to Level Zero driver bug that reports
        // mmap'd memory as "shared" (type=3) instead of "unknown" (type=0), causing DEVICE_LOST
        sycl::queue& q = stream.get_native();
        sycl::context ctx = q.get_context();

        auto send_ptr_type = ggml_sycl_get_alloc_type(send_buf);
        auto recv_ptr_type = ggml_sycl_get_alloc_type(recv_buf);

        bool need_staging = (send_ptr_type != sycl::usm::alloc::device ||
                            recv_ptr_type != sycl::usm::alloc::device);

        if (ccl_debug_enabled()) {
            const char* send_type_str = (send_ptr_type == sycl::usm::alloc::device) ? "device" :
                                        (send_ptr_type == sycl::usm::alloc::shared) ? "shared" :
                                        (send_ptr_type == sycl::usm::alloc::host) ? "host" : "unknown";
            const char* recv_type_str = (recv_ptr_type == sycl::usm::alloc::device) ? "device" :
                                        (recv_ptr_type == sycl::usm::alloc::shared) ? "shared" :
                                        (recv_ptr_type == sycl::usm::alloc::host) ? "host" : "unknown";
            fprintf(stderr, "SYCL CCL: allreduce count=%zu send=%p(%s) recv=%p(%s) staging=%d\n",
                    count, (void*)send_buf, send_type_str, (void*)recv_buf, recv_type_str, need_staging);
        }

        if (need_staging) {
            // Allocate shared memory staging buffer
            size_t buf_size = count * sizeof(float);
            float* staging = ggml_sycl_malloc_shared_t<float>(count, q, "ccl_allreduce_staging");
            if (!staging) {
                fprintf(stderr, "SYCL CCL: ERROR - failed to allocate staging buffer (%zu bytes)\n", buf_size);
                return;
            }

            // Copy input to staging buffer
            q.memcpy(staging, send_buf, buf_size).wait();

            // Debug: show values before allreduce
            if (ccl_debug_enabled()) {
                static int val_dbg = 0;
                if (val_dbg++ < 5) {
                    fprintf(stderr, "SYCL CCL: rank=%d BEFORE allreduce: [%.4f,%.4f,%.4f,%.4f]\n",
                            g_ccl_ctx.rank, staging[0], staging[1], staging[2], staging[3]);
                }
            }

            // Perform in-place allreduce on staging buffer
            ccl::allreduce(
                staging,
                staging,
                count,
                ccl::datatype::float32,
                ccl::reduction::sum,
                comm,
                stream
            ).wait();

            // Debug: show values after allreduce
            if (ccl_debug_enabled()) {
                static int val_dbg2 = 0;
                if (val_dbg2++ < 5) {
                    fprintf(stderr, "SYCL CCL: rank=%d AFTER allreduce: [%.4f,%.4f,%.4f,%.4f]\n",
                            g_ccl_ctx.rank, staging[0], staging[1], staging[2], staging[3]);
                }
            }

            // Copy result back to recv buffer
            q.memcpy(recv_buf, staging, buf_size).wait();

            // Free staging buffer
            sycl::free(staging, q);
        } else {
            // Direct allreduce - buffers already in device/shared memory
            ccl::allreduce(
                send_buf,
                recv_buf,
                count,
                ccl::datatype::float32,
                ccl::reduction::sum,
                comm,
                stream
            ).wait();
        }

        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: allreduce completed\n");
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "SYCL CCL: ERROR during allreduce: %s\n", e.what());
    }
}

bool ggml_sycl_ccl_is_initialized() {
    return g_ccl_ctx.initialized;
}

void ggml_sycl_ccl_free() {
    if (!g_ccl_ctx.initialized) {
        return;
    }

    // Mark as not initialized immediately to prevent further use
    g_ccl_ctx.initialized = false;

    if (ccl_debug_enabled()) {
        fprintf(stderr, "SYCL CCL: Rank %d freeing resources\n", g_ccl_ctx.rank);
    }

    // ROOT CAUSE FIX: The crash was caused by CCL destructors running during
    // global object destruction (after MPI starts finalizing). The vectors
    // containing ccl::communicator and ccl::stream objects were not being
    // cleared, so their destructors ran at program exit and crashed.
    //
    // The fix is to explicitly clear the vectors NOW (before MPI finalization)
    // so that CCL objects are destroyed in the correct order.
    //
    // Order matters:
    // 1. Clear streams first (they may reference communicators)
    // 2. Clear communicators
    // 3. CCL internal cleanup happens automatically

    try {
        // Wait for any pending operations to complete
        for (auto& stream : g_ccl_ctx.streams) {
            try {
                // Get the underlying SYCL queue and wait
                stream.get_native().wait();
            } catch (...) {
                // Ignore errors during cleanup
            }
        }

        // Clear streams first (must happen before communicators)
        g_ccl_ctx.streams.clear();

        // Clear communicators
        g_ccl_ctx.comms.clear();

        if (ccl_debug_enabled()) {
            fprintf(stderr, "SYCL CCL: Rank %d - CCL resources freed successfully\n", g_ccl_ctx.rank);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "SYCL CCL: Warning - exception during cleanup: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "SYCL CCL: Warning - unknown exception during cleanup\n");
    }

    g_ccl_ctx.world_size = 0;
    g_ccl_ctx.rank = 0;
}

#endif // GGML_SYCL_CCL
