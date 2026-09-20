#include "llama-io.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

void llama_io_tensor_converter::set_tensor(ggml_tensor * tensor, const void * data, size_t offset, size_t size, llama_io_tensor_conversion conversion) {
    if (conversion.type_src == GGML_TYPE_COUNT || (conversion.type_src == tensor->type && conversion.n_rot == 0)) {
        ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }

    const size_t n_per_row = tensor->ne[0];
    const size_t src_row_size = ggml_row_size(conversion.type_src, n_per_row);
    const size_t dst_row_size = ggml_row_size(tensor->type, n_per_row);
    GGML_ASSERT(offset % dst_row_size == 0 && size % src_row_size == 0);
    GGML_ASSERT(conversion.n_rot == 0 || ((conversion.n_rot & (conversion.n_rot - 1)) == 0 && n_per_row % conversion.n_rot == 0));
    const auto to_float = ggml_get_type_traits(conversion.type_src)->to_float;
    GGML_ASSERT(to_float || conversion.type_src == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_quantize_requires_imatrix(tensor->type));

    const size_t chunk_rows = std::min<size_t>(256, size/src_row_size);
    data_src.resize(chunk_rows*src_row_size);
    data_f32.resize(chunk_rows*n_per_row);
    data_dst.resize(chunk_rows*dst_row_size);

    for (size_t pos = 0; pos < size; pos += chunk_rows*src_row_size) {
        const size_t n_rows = std::min(chunk_rows, (size - pos)/src_row_size);
        const size_t n = n_rows*n_per_row;
        // Serialized tensor data need not be aligned.
        memcpy(data_src.data(), (const uint8_t *) data + pos, n_rows*src_row_size);
        if (to_float) {
            to_float(data_src.data(), data_f32.data(), n);
        } else {
            memcpy(data_f32.data(), data_src.data(), n*sizeof(float));
        }

        if (conversion.n_rot) {
            const size_t nr = conversion.n_rot;
            const float scale = 1.0f/std::sqrt(float(nr));
            for (size_t i = 0; i < n; i += nr) {
                float * row = data_f32.data() + i;
                for (size_t j = 0; j < nr; ++j) {
                    row[j] *= scale;
                }
                for (size_t len = 1; len < nr; len *= 2) {
                    for (size_t j = 0; j < nr; j += 2*len) {
                        for (size_t k = 0; k < len; ++k) {
                            const float u = row[j + k];
                            const float v = row[j + k + len];
                            row[j + k]       = u + v;
                            row[j + k + len] = u - v;
                        }
                    }
                }
            }
        }

        const size_t nbytes = ggml_quantize_chunk(tensor->type, data_f32.data(), data_dst.data(), 0, n_rows, n_per_row, nullptr);
        ggml_backend_tensor_set(tensor, data_dst.data(), offset + pos/src_row_size*dst_row_size, nbytes);
    }
}

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}
