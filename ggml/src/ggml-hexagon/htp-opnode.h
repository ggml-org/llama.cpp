#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "htp/htp-ops.h"

#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#define HTP_OP_MAX_FUSED 16

// local std::inplace_vector replacement
template <typename T, size_t MAX_N>
struct htp_opvec {
    T      data[MAX_N] {};
    size_t n = 0;

    htp_opvec() = default;
    htp_opvec(std::initializer_list<T> init) {
        GGML_ASSERT(init.size() <= MAX_N);
        for (const auto & v : init) {
            data[n++] = v;
        }
    }

    size_t    size()  const { return n; }
    bool      empty() const { return n == 0; }
    void      clear()       { n = 0; }
    T *       begin()       { return data; }
    T *       end()         { return data + n; }
    const T * begin() const { return data; }
    const T * end()   const { return data + n; }
    T &       back()        { return data[n - 1]; }
    const T & back()  const { return data[n - 1]; }
    T &       operator[](size_t i)       { return data[i]; }
    const T & operator[](size_t i) const { return data[i]; }

    void push_back(T val) {
        GGML_ASSERT(n < MAX_N);
        data[n++] = val;
    }

    void insert(T * pos, T val) {
        GGML_ASSERT(n < MAX_N);
        for (T * p = data + n; p > pos; --p) {
            *p = *(p - 1);
        }
        *pos = val;
        n++;
    }
};

static inline bool htp_is_hadamard(const struct ggml_tensor * t) {
    return t && t->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD;
}

struct htp_opnode {
    const ggml_tensor * node   { nullptr };
    htp_op_code         opcode { HTP_OP_INVALID };
    int32_t             params[HTP_OP_MAX_PARAMS] {0};
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<std::shared_ptr<ggml_tensor>>          dummy;

    htp_opvec<const ggml_tensor *, HTP_OP_MAX_FUSED>   fused;
    htp_opvec<const ggml_tensor *, HTP_OP_MAX_INPUTS>  inputs;
    htp_opvec<const ggml_tensor *, HTP_OP_MAX_OUTPUTS> outputs;

    int n_active_src(const ggml_tensor * t) const {
        if (!t) return 0;
        for (int i = GGML_MAX_SRC - 1; i >= 0; i--) {
            if (t->src[i]) {
                return i + 1;
            }
        }
        return 0;
    }

    void init(const ggml_tensor * node) {
        static_assert(sizeof(this->params) == sizeof(this->node->op_params), "params size mismatch");

        this->node = node;
        if (!node) return;

        memcpy(this->params, this->node->op_params, sizeof(this->params));

        // Build inputs (preserving optional nullptrs)
        if (htp_is_hadamard(this->node)) {
            this->inputs.push_back(this->node->src[1]);
        } else {
            int n_inputs = n_active_src(this->node);
            for (int i = 0; i < n_inputs; i++) {
                this->inputs.push_back(this->node->src[i]);
            }
        }

        // Build outputs
        this->outputs.push_back(this->dst());
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, const ggml_tensor * node = nullptr) : opcode(opcode) {
        init(node);
    }

    htp_opnode(const ggml_tensor * node) : opcode(HTP_OP_INVALID) {
        init(node);
    }

    const ggml_tensor * src0() const { return node->src[0]; }
    const ggml_tensor * src1() const { return node->src[1]; }
    const ggml_tensor * dst()  const { return outputs.empty() ? node : outputs.back(); }

    ggml_tensor * add_dummy(const ggml_tensor & t) {
        dummy.push_back(std::make_shared<ggml_tensor>(t));
        return dummy.back().get();
    }

    const htp_opvec<const ggml_tensor *, HTP_OP_MAX_INPUTS> & get_inputs() const {
        return inputs;
    }

    const htp_opvec<const ggml_tensor *, HTP_OP_MAX_OUTPUTS> & get_outputs() const {
        return outputs;
    }

    template <typename T>
    T * kparams() {
        static_assert(sizeof(T) <= sizeof(kernel_params), "kparams size exceeds HTP_OP_MAX_KERN_PARAMS");
        static_assert(alignof(T) <= alignof(int32_t), "kparams alignment exceeds int32_t");
        return reinterpret_cast<T *>(kernel_params);
    }

    template <typename T>
    const T * kparams() const {
        static_assert(sizeof(T) <= sizeof(kernel_params), "kparams size exceeds HTP_OP_MAX_KERN_PARAMS");
        static_assert(alignof(T) <= alignof(int32_t), "kparams alignment exceeds int32_t");
        return reinterpret_cast<const T *>(kernel_params);
    }

    template <typename T>
    void set_kparams(const T & kp) {
        static_assert(sizeof(T) <= sizeof(kernel_params), "kparams size exceeds HTP_OP_MAX_KERN_PARAMS");
        static_assert(alignof(T) <= alignof(int32_t), "kparams alignment exceeds int32_t");
        memcpy(kernel_params, &kp, sizeof(T));
    }

    std::string name() const;
};

enum htp_opformat_flags {
    HTP_OPFMT_OP      = (1 << 0),
    HTP_OPFMT_NAMES   = (1 << 1),
    HTP_OPFMT_DIMS    = (1 << 2),
    HTP_OPFMT_TYPES   = (1 << 3),
    HTP_OPFMT_STRIDES = (1 << 4),
    HTP_OPFMT_BUFFS   = (1 << 5),
    HTP_OPFMT_KPARAMS = (1 << 6),

    HTP_OPFMT_EXEC    = HTP_OPFMT_OP | HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES | HTP_OPFMT_BUFFS | HTP_OPFMT_KPARAMS,
    HTP_OPFMT_SUPP    = HTP_OPFMT_OP | HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES | HTP_OPFMT_BUFFS,
    HTP_OPFMT_PROF    = HTP_OPFMT_OP | HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES | HTP_OPFMT_KPARAMS,
    HTP_OPFMT_TENSOR  = HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES,
};

std::string htp_format_op(const htp_opnode & node, uint32_t flags = HTP_OPFMT_EXEC);
std::string htp_format_tensor(const ggml_tensor * t, uint32_t flags = HTP_OPFMT_TENSOR);

#endif // HTP_OPNODE_H
