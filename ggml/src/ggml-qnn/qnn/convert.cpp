
#include "convert.hpp"

#include "logger.hpp"

namespace {

size_t get_convert_buffer_size(const qnn::ggml_dimension_array_t & dimensions, ggml_type dst_type) {
    GGML_ASSERT(ggml_blck_size(dst_type) == 1);
    size_t nbytes = ggml_type_size(dst_type);
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        nbytes *= dimensions[i];  // tight packing
    }

    return nbytes;
}

// from ggml_backend_blas_mul_mat, when omp available, use it otherwise will fall back to standard lib solution
// TODO: remove this when we can fall back the convert to blas backend
#ifdef GGML_USE_OPENMP

void convert_tensor_impl(const ggml_tensor * src, int max_threads,
                         std::shared_ptr<qnn::qnn_mem_buffer_slice> & output_buffer) {
    const auto ne03                = src->ne[3];
    const auto ne02                = src->ne[2];
    const auto ne01                = src->ne[1];
    const auto ne00                = src->ne[0];
    const auto ne_plane            = ne01 * ne00;
    const auto nb03                = src->nb[3];
    const auto nb02                = src->nb[2];
    const auto nb01                = src->nb[1];
    const int  min_cols_per_thread = 4096;
    void *     wdata               = output_buffer->get_buffer();
    const auto to_float            = ggml_get_type_traits(src->type)->to_float;
    GGML_ASSERT(to_float);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const void *  x      = (char *) src->data + i02 * nb02 + i03 * nb03;
            float * const wplane = (float *) wdata + i02 * ne_plane + i03 * ne02 * ne_plane;

            const int min_rows_per_thread = std::max((int) (min_cols_per_thread / ne00), 1);
            const int n_threads           = std::max(std::min(max_threads, (int) (ne01 / min_rows_per_thread)), 1);

#    pragma omp parallel for num_threads(n_threads)
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                to_float((const char *) x + i01 * nb01, wplane + i01 * ne00, ne00);
            }
        }
    }

    return output_buffer;
}

#else

void convert_tensor_impl(const ggml_tensor * src, int max_threads, std::vector<std::future<void>> & tasks,
                         std::shared_ptr<qnn::qnn_mem_buffer_slice> & output_buffer) {
    const auto ne03                = src->ne[3];
    const auto ne02                = src->ne[2];
    const auto ne01                = src->ne[1];
    const auto ne00                = src->ne[0];
    const auto ne_plane            = ne01 * ne00;
    const auto nb03                = src->nb[3];
    const auto nb02                = src->nb[2];
    const auto nb01                = src->nb[1];
    const int  min_cols_per_thread = 4096;
    void *     wdata               = output_buffer->get_buffer();
    const auto to_float            = ggml_get_type_traits(src->type)->to_float;
    GGML_ASSERT(to_float);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const void *  x      = (char *) src->data + i02 * nb02 + i03 * nb03;
            float * const wplane = (float *) wdata + i02 * ne_plane + i03 * ne02 * ne_plane;

            const int min_rows_per_thread = std::max((int) (min_cols_per_thread / ne00), 1);
            const int n_threads           = std::max(std::min(max_threads, (int) (ne01 / min_rows_per_thread)), 1);

            for (int i = 1; i < n_threads; i++) {
                const int64_t start = i * ne01 / n_threads;
                const int64_t end   = (i + 1) * ne01 / n_threads;
                if (start < end) {
                    tasks.push_back(std::async(std::launch::async, [=]() {
                        for (int64_t i01 = start; i01 < end; i01++) {
                            to_float((const char *) x + i01 * nb01, wplane + i01 * ne00, ne00);
                        }
                    }));
                }
            }
            {
                // reuse the current thread for the first task
                const int64_t start = 0;
                const int64_t end   = ne01 / n_threads;
                for (int64_t i01 = start; i01 < end; i01++) {
                    to_float((const char *) x + i01 * nb01, wplane + i01 * ne00, ne00);
                }
            }
        }
    }

    // wait for all tasks to finish
    for (auto & task : tasks) {
        task.get();
    }
    tasks.clear();
}

#endif

}  // namespace

namespace qnn {

std::vector<qnn::qnn_buffer_ptr> convert(std::shared_ptr<qnn_convert_context_t> convert_context,
                                         const ggml_tensor_array_t & tensors, ggml_type target_data_type) {
    convert_context->buffers.resize(tensors.size());
    std::vector<qnn::qnn_buffer_ptr> output_buffers(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor * src = tensors[i];
        if (src->type == target_data_type) {
            continue;
        }

        auto &     data_buffer = convert_context->buffers[i];
        const auto dst_size    = get_convert_buffer_size(src->ne, target_data_type);
        if (!data_buffer || data_buffer->get_size() < dst_size) {
#ifndef NDEBUG
            auto old_size = data_buffer ? data_buffer->get_size() : 0;
            QNN_LOG_DEBUG("create buffer[%d] for tensor %s(%s), old_size: %d, new_size: %d\n", (int) i,
                          ggml_get_name(src), ggml_type_name(src->type), (int) old_size, (int) dst_size);
#endif
            data_buffer = std::make_shared<qnn::qnn_mem_buffer>(dst_size);
        }

        // TODO: add more restrictions to the buffer slice here
        std::shared_ptr<qnn::qnn_mem_buffer_slice> output_buffer =
            std::make_shared<qnn::qnn_mem_buffer_slice>(data_buffer->get_buffer(), dst_size);

        QNN_LOG_DEBUG("convert tensor(%s) from %s to %s, size: %d, n_threads: %d\n", ggml_get_name(src),
                      ggml_type_name(src->type), ggml_type_name(target_data_type), (int) dst_size,
                      convert_context->n_threads);

#ifdef GGML_USE_OPENMP
        convert_tensor_impl(src, convert_context->n_threads, output_buffer);
#else
        convert_tensor_impl(src, convert_context->n_threads, convert_context->tasks, output_buffer);
#endif
        output_buffers[i] = output_buffer;
    }

    return output_buffers;
}

}  // namespace qnn
