// Regression tests for UMA zero-copy async buffer transfers.
// Validates set_tensor_async -> event_record -> synchronize -> get round-trip.
// Without the deferred memcpy drain in event_record, UMA zero-copy writes
// are silently dropped on context reset.

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool find_vulkan_device(ggml_backend_dev_t * dev_out) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strstr(ggml_backend_dev_name(dev), "Vulkan") != nullptr) {
            *dev_out = dev;
            return true;
        }
    }
    return false;
}

// Async set -> event_record -> synchronize -> get round-trip.
static bool test_async_event_round_trip(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_async_event_round_trip: ");

    const int n_elements = 1024;
    const size_t data_size = n_elements * sizeof(float);

    // Prepare source data with a recognisable pattern.
    std::vector<float> src(n_elements);
    for (int i = 0; i < n_elements; i++) {
        src[i] = (float)(i * 7 + 13);
    }

    // Allocate a tensor on the backend.
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        return false;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("SKIP (buffer alloc failed)\n");
        ggml_free(ctx);
        return true;  // SKIP is not a failure
    }

    // Create an event.
    ggml_backend_event_t event = ggml_backend_event_new(dev);
    if (!event) {
        printf("SKIP (events not supported)\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    }

    // Model-loader pattern: async write -> event_record -> synchronize.
    ggml_backend_tensor_set_async(backend, t, src.data(), 0, data_size);
    ggml_backend_event_record(event, backend);
    ggml_backend_event_synchronize(event);

    // Read back and verify.
    std::vector<float> dst(n_elements, 0.0f);
    ggml_backend_tensor_get(t, dst.data(), 0, data_size);

    bool ok = true;
    for (int i = 0; i < n_elements; i++) {
        if (dst[i] != src[i]) {
            printf("FAIL (mismatch at [%d]: expected %.1f, got %.1f)\n",
                   i, src[i], dst[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("OK\n");
    }

    ggml_backend_event_free(event);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Multiple async sets interleaved with event records.
static bool test_async_event_multiple(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_async_event_multiple:   ");

    const int n_elements = 512;
    const size_t data_size = n_elements * sizeof(float);
    const int n_rounds = 4;

    ggml_init_params params = {
        /* .mem_size   = */ n_rounds * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        return false;
    }

    ggml_tensor * tensors[n_rounds];
    for (int r = 0; r < n_rounds; r++) {
        tensors[r] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("SKIP (buffer alloc failed)\n");
        ggml_free(ctx);
        return true;
    }

    ggml_backend_event_t event = ggml_backend_event_new(dev);
    if (!event) {
        printf("SKIP (events not supported)\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    }

    // Write different patterns to each tensor, recording an event after each.
    std::vector<std::vector<float>> src_data(n_rounds);
    for (int r = 0; r < n_rounds; r++) {
        src_data[r].resize(n_elements);
        for (int i = 0; i < n_elements; i++) {
            src_data[r][i] = (float)(r * 1000 + i);
        }
        ggml_backend_tensor_set_async(backend, tensors[r], src_data[r].data(), 0, data_size);
        ggml_backend_event_record(event, backend);
        ggml_backend_event_synchronize(event);
    }

    // Verify all tensors.
    bool ok = true;
    for (int r = 0; r < n_rounds && ok; r++) {
        std::vector<float> dst(n_elements, 0.0f);
        ggml_backend_tensor_get(tensors[r], dst.data(), 0, data_size);
        for (int i = 0; i < n_elements && ok; i++) {
            if (dst[i] != src_data[r][i]) {
                printf("FAIL (round %d, mismatch at [%d]: expected %.1f, got %.1f)\n",
                       r, i, src_data[r][i], dst[i]);
                ok = false;
            }
        }
    }
    if (ok) {
        printf("OK\n");
    }

    ggml_backend_event_free(event);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// event_record with no pending memcpys (empty deferred queues).
static bool test_empty_queue_event_record(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_empty_queue_event:       ");

    const int n_elements = 256;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        return false;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("SKIP (buffer alloc failed)\n");
        ggml_free(ctx);
        return true;
    }

    ggml_backend_event_t event = ggml_backend_event_new(dev);
    if (!event) {
        printf("SKIP (events not supported)\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    }

    // Write data synchronously first (not async — bypasses deferred memcpy path).
    std::vector<float> src(n_elements);
    for (int i = 0; i < n_elements; i++) {
        src[i] = (float)(i * 3 + 1);
    }
    ggml_backend_tensor_set(t, src.data(), 0, n_elements * sizeof(float));

    // event_record with empty deferred queues — must not crash or leak.
    ggml_backend_event_record(event, backend);
    ggml_backend_event_synchronize(event);

    // Synchronize should be a no-op for the deferred copy path.
    ggml_backend_synchronize(backend);

    // Verify data is still intact (event_record didn't corrupt anything).
    std::vector<float> dst(n_elements, 0.0f);
    ggml_backend_tensor_get(t, dst.data(), 0, n_elements * sizeof(float));

    bool ok = true;
    for (int i = 0; i < n_elements; i++) {
        if (dst[i] != src[i]) {
            printf("FAIL (mismatch at [%d]: expected %.1f, got %.1f)\n",
                   i, src[i], dst[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("OK\n");
    }

    ggml_backend_event_free(event);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// D2H deferred copies survive event_record context reset.
static bool test_out_memcpy_persistence(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_out_memcpy_persistence:  ");

    const int n_elements = 512;
    const size_t data_size = n_elements * sizeof(float);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        return false;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("SKIP (buffer alloc failed)\n");
        ggml_free(ctx);
        return true;
    }

    ggml_backend_event_t event = ggml_backend_event_new(dev);
    if (!event) {
        printf("SKIP (events not supported)\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    }

    // Phase 1: Write known data to the tensor (synchronous — data is committed).
    std::vector<float> src(n_elements);
    for (int i = 0; i < n_elements; i++) {
        src[i] = (float)(i * 11 + 7);
    }
    ggml_backend_tensor_set(t, src.data(), 0, data_size);

    // Queue async read — deferred until after fence wait.
    std::vector<float> dst(n_elements, 0.0f);
    ggml_backend_tensor_get_async(backend, t, dst.data(), 0, data_size);

    // event_record resets context — out_memcpys must survive.
    ggml_backend_event_record(event, backend);
    ggml_backend_event_synchronize(event);

    // Phase 4: synchronize drains the restored out_memcpys.
    ggml_backend_synchronize(backend);

    // Verify the async read completed correctly.
    bool ok = true;
    for (int i = 0; i < n_elements; i++) {
        if (dst[i] != src[i]) {
            printf("FAIL (mismatch at [%d]: expected %.1f, got %.1f)\n",
                   i, src[i], dst[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("OK\n");
    }

    ggml_backend_event_free(event);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main(void) {
    ggml_backend_load_all();

    ggml_backend_dev_t dev;
    if (!find_vulkan_device(&dev)) {
        printf("No Vulkan device found -- SKIP\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        printf("Failed to init Vulkan backend -- SKIP\n");
        return 0;
    }

    printf("Vulkan UMA async regression tests (%s):\n", ggml_backend_dev_description(dev));

    bool all_ok = true;
    all_ok &= test_async_event_round_trip(backend, dev);
    all_ok &= test_async_event_multiple(backend, dev);
    all_ok &= test_empty_queue_event_record(backend, dev);
    all_ok &= test_out_memcpy_persistence(backend, dev);

    ggml_backend_free(backend);

    if (!all_ok) {
        printf("FAILED\n");
        return 1;
    }

    printf("All tests passed.\n");
    return 0;
}
