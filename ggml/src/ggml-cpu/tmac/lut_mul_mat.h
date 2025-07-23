#pragma once

/* Please do not include this header file outside ggml-cpu/tmac */

#include "lut_ctor.h"
#include "tbl.h"
#include "ggml-cpu-traits.h"

#include <unordered_map>

static const int GGML_TMAC_MAX_NODES = 8192;
struct tmac_tensor_extra {
    int lut_scales_size;
    int scales_size;
    int n_tile_num;
    uint8_t * qweights;
    tmac_float_type * scales;
};

namespace ggml::cpu::tmac {
    class tensor_traits : public ggml::cpu::tensor_traits {
        std::unordered_map<std::string, tmac_tensor_extra *> tmac_tensor_extra;
        // struct tmac_tensor_extra * tmac_tensor_extra = nullptr;

        bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override;
        bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override;

public:
        struct tmac_tensor_extra * get_tmac_tensor_extra(std::string tensor_name) {
            if (tmac_tensor_extra.find(tensor_name) == tmac_tensor_extra.end()) {
                return nullptr;
            }
            return tmac_tensor_extra[tensor_name];
        }
        void set_tmac_tensor_extra(std::string tensor_name, struct tmac_tensor_extra * extra) {
            // if (tmac_tensor_extra.find(tensor_name) != tmac_tensor_extra.end()) {
            //     GGML_LOG_WARN("tmac_tensor_extra already exists for tensor %s. Overriding the data!\n", tensor_name.c_str());
            // }
            tmac_tensor_extra[tensor_name] = extra;
        }
    };
}  // namespace ggml::cpu::tmac


#ifdef  __cplusplus
extern "C" {
#endif

void tmac_init(void);

bool is_tmac_type(enum ggml_type type);

bool is_type_supported(enum ggml_type type);

bool is_embedding_weight(const struct ggml_tensor * tensor);

size_t ggml_backend_tmac_desired_wsize(const struct ggml_tensor * dst);

size_t ggml_backend_tmac_get_alloc_size(const struct ggml_tensor * tensor);

size_t ggml_tmac_get_nbytes(const struct ggml_tensor * tensor);

void ggml_backend_tmac_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);

bool ggml_tmac_can_mul_mat(const struct ggml_tensor * dst);

void ggml_backend_tmac_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
