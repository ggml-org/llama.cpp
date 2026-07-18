#include <ggml-backend.h>
#include "../ggml/src/ggml-backend-impl.h"
#include <ggml-cpp.h>
#include <ggml-cuda.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

static constexpr int64_t N = 4096;
static constexpr int N_ITERATIONS = 100;
static constexpr size_t MAX_VRAM_DRIFT = 16u * 1024u * 1024u;

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        abort();
    }
}

static void check_output(const std::vector<float> & weights, const std::vector<float> & output) {
    for (int64_t i = 0; i < N; ++i) {
        const float expected = weights[i] + 4.0f * weights[i] * weights[i];
        const float error = std::fabs(output[i] - expected);
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
        if (error > tolerance) {
            fprintf(stderr, "FAIL: output[%lld] = %.9g, expected %.9g, error %.9g\n",
                    (long long) i, output[i], expected, error);
            abort();
        }
    }
}

static void check_metrics(
        ggml_backend_sched_t sched, ggml_backend_t cuda, ggml_backend_t cpu,
        size_t logical_bytes, size_t backend_bytes, size_t weight_window, size_t safety_reserve, int executions) {
    ggml_backend_sched_transient_metrics snapshot = {};
    require(ggml_backend_sched_get_transient_metrics(sched, &snapshot), "metric snapshot query failed");
    require(snapshot.n_backends == 2, "metric snapshot backend count mismatch");
    require(snapshot.current_transient_bytes == 0 && snapshot.current_transient_records == 0,
            "global live ledger did not drain outside compute");
    require(snapshot.peak_transient_bytes == backend_bytes && snapshot.peak_transient_records == 1,
            "global peak ledger is inconsistent with one transient split at a time");
    require(snapshot.graph_compute_count == (uint64_t) executions, "graph compute count mismatch");
    require(snapshot.graph_compute_failure_count == 0 && snapshot.callback_early_stop_count == 0,
            "unexpected graph failure or callback stop metric");
    require(snapshot.ledger_mismatch_count == 0 && snapshot.counter_overflow_count == 0,
            "unexpected ledger mismatch or metric overflow");

    const auto & cuda_row = snapshot.backends[0];
    require(cuda_row.backend_index == 0 && cuda_row.backend == cuda, "CUDA metric row identity mismatch");
    require(cuda_row.current_transient_bytes == 0 && cuda_row.current_transient_records == 0,
            "CUDA live ledger did not drain outside compute");
    require(cuda_row.weight_window_configured && cuda_row.weight_window_memory_valid,
            "CUDA weight window is not valid");
    require(cuda_row.weight_window_limit_bytes == weight_window &&
            cuda_row.weight_window_safety_reserve_bytes == safety_reserve,
            "CUDA weight-window snapshot mismatch");
    require(cuda_row.peak_transient_bytes == backend_bytes && cuda_row.peak_transient_records == 1,
            "CUDA peak ledger mismatch");
    require(cuda_row.peak_transient_bytes <= cuda_row.weight_window_limit_bytes,
            "CUDA peak transient bytes exceeded admitted window");
    require(cuda_row.allocation_requested_bytes == 2u * backend_bytes * executions &&
            cuda_row.allocation_admitted_bytes == 2u * backend_bytes * executions &&
            cuda_row.allocation_rejected_bytes == 0 && cuda_row.allocation_count == (uint64_t) (2 * executions),
            "CUDA allocation metrics mismatch");
    require(cuda_row.allocation_failure_count == 0 && cuda_row.allocation_limit_rejection_count == 0 &&
            cuda_row.oversized_tensor_rejection_count == 0, "unexpected CUDA allocation rejection metric");
    require(cuda_row.upload_count == (uint64_t) (2 * executions) &&
            cuda_row.uploaded_logical_bytes == 2u * logical_bytes * executions &&
            cuda_row.uploaded_backend_bytes == 2u * backend_bytes * executions,
            "CUDA upload metrics mismatch");
    require(cuda_row.shared_reload_count == (uint64_t) executions, "CUDA tied-weight reload count mismatch");
    require(cuda_row.splits_seen_count == (uint64_t) (2 * executions) &&
            cuda_row.transient_split_count == (uint64_t) (2 * executions), "CUDA split metrics mismatch");
    require(cuda_row.transfer_completion_wait_count == (uint64_t) (2 * executions) &&
            cuda_row.compute_completion_wait_count == (uint64_t) (2 * executions),
            "CUDA completion-wait metrics mismatch");
    require(cuda_row.drain_count[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_NORMAL] == (uint64_t) (2 * executions),
            "CUDA normal drain count mismatch");
    for (int reason = GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ALLOCATION_FAILURE;
            reason < GGML_BACKEND_SCHED_TRANSIENT_DRAIN_REASON_COUNT; ++reason) {
        require(cuda_row.drain_count[reason] == 0, "unexpected non-normal CUDA drain metric");
    }

    const auto & cpu_row = snapshot.backends[1];
    require(cpu_row.backend_index == 1 && cpu_row.backend == cpu, "CPU metric row identity mismatch");
    require(cpu_row.current_transient_bytes == 0 && cpu_row.peak_transient_bytes == 0 &&
            cpu_row.current_transient_records == 0 && cpu_row.peak_transient_records == 0,
            "CPU row reported transient ownership");
    require(cpu_row.allocation_count == 0 && cpu_row.upload_count == 0 && cpu_row.transient_split_count == 0 &&
            cpu_row.shared_reload_count == 0, "CPU row reported transient operations");
    require(cpu_row.splits_seen_count == (uint64_t) executions,
            "CPU non-transient split participation count mismatch");
}

int main() {
    const char * benchmark_mode = std::getenv("GGML_CUDA_SCHED_BENCHMARK");
    const bool benchmark = benchmark_mode != nullptr;
    const bool benchmark_query = benchmark && std::atoi(benchmark_mode) != 0;
    const int iterations = benchmark ? 1000 : N_ITERATIONS;
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (cuda == nullptr) {
        printf("SKIP: no CUDA backend is available\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(cpu != nullptr, "CPU backend initialization failed");

    ggml_backend_sched_t sched = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    ggml_context_ptr weights_ctx;
    ggml_context_ptr graph_ctx;

    const ggml_init_params weights_params = { 2 * ggml_tensor_overhead(), nullptr, true };
    weights_ctx.reset(ggml_init(weights_params));
    require(weights_ctx != nullptr, "weight context initialization failed");
    ggml_tensor * tied = ggml_new_tensor_1d(weights_ctx.get(), GGML_TYPE_F32, N);
    weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx.get(), cpu);
    require(weights_buffer != nullptr, "host weight allocation failed");
    require(ggml_backend_buft_is_host(ggml_backend_buffer_get_type(weights_buffer)), "weight buffer is not host-backed");
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const ggml_init_params graph_params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    graph_ctx.reset(ggml_init(graph_params));
    require(graph_ctx != nullptr, "graph context initialization failed");
    ggml_tensor * first = ggml_add(graph_ctx.get(), tied, tied);
    ggml_tensor * cpu_barrier = ggml_sqr(graph_ctx.get(), first);
    ggml_tensor * second = ggml_add(graph_ctx.get(), tied, cpu_barrier);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, second);

    ggml_backend_t backends[] = { cuda, cpu };
    ggml_backend_buffer_type_t bufts[] = {
        ggml_backend_get_default_buffer_type(cuda),
        ggml_backend_get_default_buffer_type(cpu),
    };
    sched = ggml_backend_sched_new(backends, bufts, 2, 64, false, true);
    require(sched != nullptr, "scheduler initialization failed");
    ggml_backend_sched_set_force_weight_offload(sched, true);
    ggml_backend_sched_set_async_weight_prefetch(sched, false);
    require(ggml_backend_sched_get_n_copies(sched) == 1, "scheduler is not sequential");
    ggml_backend_sched_set_tensor_backend(sched, first, cuda);
    ggml_backend_sched_set_tensor_backend(sched, cpu_barrier, cpu);
    ggml_backend_sched_set_tensor_backend(sched, second, cuda);
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation failed");
    require(ggml_backend_sched_get_n_splits(sched) == 3, "expected CUDA -> CPU -> CUDA split topology");
    ggml_backend_sched_synchronize(sched);
    size_t post_reservation_free = 0;
    size_t post_reservation_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(cuda), &post_reservation_free, &post_reservation_total);
    size_t weight_window = 0;
    size_t safety_reserve = 0;
    require(ggml_backend_sched_set_weight_window(sched, cuda, post_reservation_free, post_reservation_total,
            SIZE_MAX, &weight_window, &safety_reserve), "CUDA post-reservation memory sample is invalid");
    ggml_backend_sched_set_weight_residency(sched, cuda, true);

    ggml_tensor * tied_copy = first->src[0];
    require(tied_copy == first->src[1] && tied_copy == second->src[0], "tied descriptor identity was not preserved");
    require(tied_copy != tied, "scheduler did not create a device descriptor");
    require((tied_copy->flags & GGML_TENSOR_FLAG_NO_ALLOC) != 0, "tied descriptor is not NO_ALLOC");
    require(tied_copy->buffer == nullptr && tied_copy->data == nullptr, "tied descriptor is attached outside compute");
    const size_t logical_bytes = ggml_nbytes(tied_copy);
    const size_t backend_bytes = ggml_backend_buft_get_alloc_size(bufts[0], tied_copy);
    require(backend_bytes >= logical_bytes && backend_bytes > 0, "invalid CUDA allocation capacity");

    std::vector<float> weights(N);
    std::vector<float> output(N);
    auto run = [&](int iteration, int executions) {
        for (int64_t i = 0; i < N; ++i) {
            weights[i] = 0.001f * (float) ((i % 97) - 48);
        }
        if (iteration == 0) {
            ggml_backend_tensor_set(tied, weights.data(), 0, ggml_nbytes(tied));
        }
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS, "scheduler compute failed");
        ggml_backend_sched_synchronize(sched);
        require(tied_copy->buffer != nullptr && tied_copy->data != nullptr, "resident tied descriptor was detached after compute");
        ggml_backend_tensor_get(second, output.data(), 0, ggml_nbytes(second));
        check_output(weights, output);
        if (!benchmark || benchmark_query) {
            if (executions == 1) {
                ggml_backend_sched_transient_metrics snapshot = {};
                require(ggml_backend_sched_get_transient_metrics(sched, &snapshot), "residency metric snapshot failed");
                require(snapshot.backends[0].residency_upload_count == 1, "cold CUDA residency upload count mismatch");
                require(snapshot.backends[0].current_resident_bytes == backend_bytes, "CUDA resident byte count mismatch");
                require(snapshot.backends[0].peak_manually_owned_bytes <= weight_window, "CUDA ownership exceeded window");
            }
        }
    };

    run(0, 1);
    ggml_backend_sched_synchronize(sched);
    size_t baseline_free = 0;
    size_t total = 0;
    ggml_backend_cuda_get_device_memory(0, &baseline_free, &total);
    size_t minimum_free = baseline_free;
    size_t previous_free = baseline_free;
    bool monotonically_decreasing = true;

    const auto measured_start = std::chrono::steady_clock::now();
    for (int iteration = 1; iteration <= iterations; ++iteration) {
        run(iteration, iteration + 1);
        size_t current_free = 0;
        ggml_backend_cuda_get_device_memory(0, &current_free, &total);
        minimum_free = std::min(minimum_free, current_free);
        monotonically_decreasing = monotonically_decreasing && current_free < previous_free;
        previous_free = current_free;
    }
    const auto measured_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - measured_start).count();

    const size_t worst_drift = baseline_free > minimum_free ? baseline_free - minimum_free : 0;
    require(!monotonically_decreasing, "free VRAM decreased monotonically after warm-up");
    require(worst_drift <= MAX_VRAM_DRIFT, "post-warm-up free VRAM drift exceeded 16 MiB");

    ggml_backend_sched_transient_metrics final_snapshot = {};
    require(ggml_backend_sched_get_transient_metrics(sched, &final_snapshot), "final metric snapshot query failed");
    const auto & final_cuda_row = final_snapshot.backends[0];
    require(final_cuda_row.residency_upload_count == 1, "resident weight was uploaded more than once");
    require(final_cuda_row.residency_miss_count == 1, "CUDA residency cold-miss count mismatch");
    require(final_cuda_row.residency_hit_count == (uint64_t) (2 * (iterations + 1) - 1),
            "CUDA residency warm-hit count mismatch");
    require(final_cuda_row.residency_eviction_count == 0, "unexpected CUDA residency eviction");
    require(final_cuda_row.current_resident_bytes == backend_bytes && final_cuda_row.current_resident_records == 1,
            "CUDA resident ledger mismatch before disable");
    require(final_cuda_row.peak_resident_bytes == backend_bytes && final_cuda_row.peak_resident_records == 1,
            "CUDA peak resident ledger mismatch");
    require(final_cuda_row.current_transient_bytes == 0 && final_cuda_row.current_transient_records == 0,
            "CUDA transient ledger did not drain");
    require(final_cuda_row.peak_manually_owned_bytes == backend_bytes,
            "CUDA combined resident/transient peak mismatch");
    require(final_cuda_row.peak_manually_owned_bytes <= weight_window, "peak CUDA ownership exceeded window");

    const uint64_t cold_misses = final_cuda_row.residency_miss_count;
    const uint64_t resident_uploads = final_cuda_row.residency_upload_count;
    const uint64_t warm_hits = final_cuda_row.residency_hit_count;
    const uint64_t evictions = final_cuda_row.residency_eviction_count;
    const size_t peak_resident = final_cuda_row.peak_resident_bytes;
    const size_t peak_transient = final_cuda_row.peak_transient_bytes;
    const size_t peak_total = final_cuda_row.peak_manually_owned_bytes;
    const uint64_t allocations = final_cuda_row.allocation_count;

    ggml_backend_sched_set_weight_residency(sched, cuda, false);
    require(tied_copy->buffer == nullptr && tied_copy->data == nullptr, "disable did not drain resident descriptor");
    require(ggml_backend_sched_get_transient_metrics(sched, &final_snapshot), "post-disable metric snapshot failed");
    require(final_snapshot.current_resident_bytes == 0 && final_snapshot.current_resident_records == 0,
            "post-disable resident ledger did not drain");

    ggml_backend_sched_free(sched);
    sched = nullptr;
    ggml_backend_buffer_free(weights_buffer);
    weights_buffer = nullptr;
    graph_ctx.reset();
    weights_ctx.reset();
    ggml_backend_synchronize(cuda);
    size_t cleanup_free = 0;
    ggml_backend_cuda_get_device_memory(0, &cleanup_free, &total);
    ggml_backend_free(cpu);
    ggml_backend_free(cuda);

    printf("PASS: iterations=%d executions=%d logical_bytes=%zu backend_bytes=%zu weight_window=%zu safety_reserve=%zu "
           "cold_misses=%llu resident_uploads=%llu warm_hits=%llu evictions=%llu allocations=%llu "
           "peak_resident=%zu peak_transient=%zu peak_total=%zu post_disable_resident_bytes=%zu "
           "post_disable_resident_records=%zu post_disable_transient_bytes=%zu post_disable_transient_records=%zu "
           "detached=1 output_correct=1 measured_us=%lld benchmark_query=%d baseline_free=%zu minimum_free=%zu "
           "worst_drift=%zu cleanup_free=%zu total=%zu\n",
           iterations, iterations + 1, logical_bytes, backend_bytes, final_cuda_row.weight_window_limit_bytes,
           final_cuda_row.weight_window_safety_reserve_bytes,
           (unsigned long long) cold_misses, (unsigned long long) resident_uploads,
           (unsigned long long) warm_hits, (unsigned long long) evictions, (unsigned long long) allocations,
           peak_resident, peak_transient, peak_total,
           final_snapshot.current_resident_bytes, final_snapshot.current_resident_records,
           final_snapshot.current_transient_bytes, final_snapshot.current_transient_records,
           (long long) measured_us, benchmark_query ? 1 : 0,
           baseline_free, minimum_free, worst_drift, cleanup_free, total);
    return 0;
}
