#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_indexed_bias(ggml_backend_t backend, int width, bool use_rows, int n_query, int cells = 768) {
    constexpr int heads = 6, kv_heads = 2, streams = 2, bins = 17;
    const int cols = use_rows ? std::min(512, cells) : cells;

    ggml_context * ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, n_query, heads, streams);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, width, cells, kv_heads, use_rows ? 1 : streams);
    auto * v = ggml_dup_tensor(ctx, k);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cols, n_query, 1, streams);
    auto * rows = use_rows ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cols, streams) : nullptr;
    auto * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, heads, bins, n_query*streams);
    auto * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cols, n_query*streams);
    auto * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    auto * out = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, 1.0f/sqrtf(width), 0, 0);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    const bool supports_plain = ggml_backend_supports_op(backend, out);
    ggml_flash_attn_ext_add_indexed_bias(out, table, indices);
    GGML_ASSERT(ggml_flash_attn_ext_has_indexed_bias(out));
    ggml_flash_attn_ext_add_indexed_bias(out, nullptr, nullptr);
    GGML_ASSERT(!ggml_flash_attn_ext_has_indexed_bias(out) && !out->src[6] && !out->src[7]);
    GGML_ASSERT(ggml_backend_supports_op(backend, out) == supports_plain);
    ggml_flash_attn_ext_add_indexed_bias(out, table, indices);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return;
    }

    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);
    ggml_backend_buffer_clear(buf, 0);

    std::vector<int32_t> row(cols*streams), idx(ggml_nelements(indices));
    std::vector<ggml_fp16_t> visibility(idx.size()), values(ggml_nelements(v));
    std::vector<ggml_fp16_t> keys(ggml_nelements(k));
    std::vector<float> queries(ggml_nelements(q));
    float sink_logits[heads];
    for (int h = 0; h < heads; ++h) {
        sink_logits[h] = float(2*h - 3);
    }
    for (size_t i = 0; i < queries.size(); i += width) {
        queries[i] = 4.0f;
    }
    std::vector<float> biases(ggml_nelements(table)), expected(n_query*streams*heads), output(ggml_nelements(out));
    for (int s = 0; s < v->ne[3]; ++s) {
        for (int h = 0; h < kv_heads; ++h) {
            for (int c = 0; c < cells; ++c) {
                const float x = float((c + 3*h + 5*s) % 23 - 11)/16;
                const int offset = ((s*kv_heads + h)*cells + c)*width;
                std::fill_n(values.data() + offset, width, ggml_fp32_to_fp16(x));
                keys[offset] = ggml_fp32_to_fp16(float((7*c + 2*h + 3*s) % 29 - 14)/4);
            }
        }
    }
    for (int s = 0; s < streams; ++s) {
        for (int c = 0; c < cols; ++c) {
            row[s*cols + c] = use_rows ? (c % 19 == 0 ? -1 : (13*c + 73*s) % cells) : c;
        }
        for (int j = 0; j < n_query; ++j) {
            const int query = s*n_query + j;
            for (int h = 0; h < heads; ++h) {
                for (int b = 0; b < bins; ++b) {
                    biases[(query*bins + b)*heads + h] = float((b*(3 + h) + query) % 13 - 6)/2.9f;
                }
                double sum = exp(double(sink_logits[h])), weighted = 0.0;
                for (int c = 0; c < cols; ++c) {
                    const int cell = row[s*cols + c];
                    const int i = query*cols + c;
                    idx[i] = (c + 3*query) % bins;
                    const bool visible = (s != 1 || j != 0) && cell >= 0 && (c + j) % 7 != 0;
                    visibility[i] = ggml_fp32_to_fp16(visible ? 0.25f : -INFINITY);
                    if (visible) {
                        const int vk = (use_rows ? 0 : s)*kv_heads + h/(heads/kv_heads);
                        const double qk = 4.0*ggml_fp16_to_fp32(keys[(vk*cells + cell)*width])/sqrt(width);
                        const float m = 0.25f + biases[(query*bins + idx[i])*heads + h];
                        const double weight = exp(qk + ggml_fp16_to_fp32(ggml_fp32_to_fp16(m)));
                        weighted += weight*ggml_fp16_to_fp32(values[(vk*cells + cell)*width]);
                        sum += weight;
                    }
                }
                expected[query*heads + h] = weighted/sum;
            }
        }
    }
    ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, keys.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, values.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(mask, visibility.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(table, biases.data(), 0, ggml_nbytes(table));
    ggml_backend_tensor_set(sinks, sink_logits, 0, sizeof(sink_logits));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    if (rows) {
        ggml_backend_tensor_set(rows, row.data(), 0, ggml_nbytes(rows));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
    float max_error = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        GGML_ASSERT(std::isfinite(output[i]));
        max_error = std::max(max_error, fabsf(output[i] - expected[i/width]));
    }
    printf("%s: width=%d, rows=%d, queries=%d, cells=%d, max error=%g\n", ggml_backend_name(backend), width, use_rows, n_query, cells, max_error);
    GGML_ASSERT(max_error < 0.001f);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend);
        for (int width : {64, 128}) {
            test_indexed_bias(backend, width, false, 7, 16);
            test_indexed_bias(backend, width, true, 7, 16);
            for (int n_query : {1, 7, 65}) {
                test_indexed_bias(backend, width, false, n_query);
                test_indexed_bias(backend, width, true, n_query);
            }
        }
        ggml_backend_free(backend);
    }
}
