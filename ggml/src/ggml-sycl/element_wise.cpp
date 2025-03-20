#include "common.hpp"
#include "element_wise.hpp"

static void acc_f32(const float * x, const float * y, float * dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset, const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    } else {
        dst[i] = x[i];
    }
}

static void gelu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f * xi *
             (1.0f +
              sycl::tanh(SQRT_2_OVER_PI * xi * (1.0f + GELU_COEF_A * xi * xi)));
}

static void silu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + sycl::native::exp(-x[i]));
}

static void gelu_quick_f32(const float *x, float *dst, int k,
                           const sycl::nd_item<3> &item_ct1) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (1.0f / (1.0f + sycl::native::exp(GELU_QUICK_COEF * x[i])));
}

static void tanh_f32(const float *x, float *dst, int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::tanh((float)(x[i]));
}

static void relu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((float)(x[i]), (float)0);
}

static void sigmoid_f32(const float * x, float * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = 1.0f / (1.0f + sycl::native::exp(-x[i]));
}

static void sqrt_f32(const float * x, float * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::sqrt(x[i]);
}

static void sin_f32(const float * x, float * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::sin(x[i]);
}

static void cos_f32(const float * x, float * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::cos(x[i]);
}

static void hardsigmoid_f32(const float * x, float * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmin(1.0f, sycl::fmax(0.0f, (x[i] + 3.0f) / 6.0f));
}

static void hardswish_f32(const float * x, float * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * sycl::fmin(1.0f, sycl::fmax(0.0f, (x[i] + 3.0f) / 6.0f));
}

static void exp_f32(const float * x, float * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::exp(x[i]);
}

static void log_f32(const float * x, float * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    float xi = x[i];
    if (xi <= 0) {
        dst[i] = -INFINITY;
    } else {
        dst[i] = sycl::log(xi);
    }
}

static void neg_f32(const float * x, float * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = -x[i];
}

static void step_f32(const float * x, float * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] > 0.0f;
}

static void leaky_relu_f32(const float *x, float *dst, const int k, const float negative_slope,
                           const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((float)(x[i]), (float)0) +
             sycl::fmin((float)(x[i]), 0.0f) * negative_slope;
}

static void sqr_f32(const float * x, float * dst, const int k,
                    const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

static void upscale_f32(const float  *x, float *dst, const int nb00, const int nb01,
                        const int nb02, const int nb03, const int ne10, const int ne11,
                        const int ne12, const int ne13, const float sf0, const float sf1,
                        const float sf2, const float sf3, const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_local_id(0) +
               item_ct1.get_group(0) * item_ct1.get_local_range(0);
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }
    // operation
    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    int i00 = i10 / sf0;
    int i01 = i11 / sf1;
    int i02 = i12 / sf2;
    int i03 = i13 / sf3;

    dst[index] = *(const float *)((const char *)x + i03 * nb03 + i02 * nb02 + i01 * nb01 + i00 * nb00);
}

static void pad_f32(const float  *x, float *dst, const int ne0, const int ne00, const int ne01, const int ne02,
                    const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (nidx < ne00 && item_ct1.get_group(1) < (size_t) ne01 && item_ct1.get_group(0) < (size_t) ne02) {
        int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                         item_ct1.get_group(0) * ne00 * ne01;
            dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = 0.0f;
    }
}



static void acc_f32_sycl(const float *x, const float *y, float *dst,
                         const int n_elements, const int ne10, const int ne11,
                         const int ne12, const int nb1, const int nb2,
                         const int offset, queue_ptr stream) {
    int num_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            acc_f32(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset,
                    item_ct1);
        });
}

static void gelu_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_f32(x, dst, k, item_ct1);
        });
}

static void silu_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_SILU_BLOCK_SIZE - 1) / SYCL_SILU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            silu_f32(x, dst, k, item_ct1);
        });
}

static void gelu_quick_f32_sycl(const float *x, float *dst, const int k,
                                queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_quick_f32(x, dst, k, item_ct1);
        });
}

static void tanh_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_TANH_BLOCK_SIZE - 1) / SYCL_TANH_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            tanh_f32(x, dst, k, item_ct1);
        });
}

static void relu_f32_sycl(const float *x, float *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            relu_f32(x, dst, k, item_ct1);
        });
}

static void hardsigmoid_f32_sycl(const float *x, float *dst, const int k,
                                 queue_ptr stream) {
    const int num_blocks = (k + SYCL_HARDSIGMOID_BLOCK_SIZE - 1) / SYCL_HARDSIGMOID_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_HARDSIGMOID_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_HARDSIGMOID_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            hardsigmoid_f32(x, dst, k, item_ct1);
        });
}

static void hardswish_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_HARDSWISH_BLOCK_SIZE - 1) / SYCL_HARDSWISH_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_HARDSWISH_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_HARDSWISH_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            hardswish_f32(x, dst, k, item_ct1);
        });
}

static void exp_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_EXP_BLOCK_SIZE - 1) / SYCL_EXP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            exp_f32(x, dst, k, item_ct1);
        });
}

static void log_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_EXP_BLOCK_SIZE - 1) / SYCL_EXP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            log_f32(x, dst, k, item_ct1);
        });
}

static void neg_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            neg_f32(x, dst, k, item_ct1);
        });
}

static void step_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            step_f32(x, dst, k, item_ct1);
        });
}

static void sigmoid_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SIGMOID_BLOCK_SIZE - 1) / SYCL_SIGMOID_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SIGMOID_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SIGMOID_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sigmoid_f32(x, dst, k, item_ct1);
        });
}

static void sqrt_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SQRT_BLOCK_SIZE - 1) / SYCL_SQRT_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SQRT_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SQRT_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sqrt_f32(x, dst, k, item_ct1);
        });
}

static void sin_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SIN_BLOCK_SIZE - 1) / SYCL_SIN_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sin_f32(x, dst, k, item_ct1);
        });
}

static void cos_f32_sycl(const float *x, float *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SIN_BLOCK_SIZE - 1) / SYCL_SIN_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            cos_f32(x, dst, k, item_ct1);
        });
}

static void leaky_relu_f32_sycl(const float *x, float *dst, const int k,
                                const float negative_slope,
                                queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            leaky_relu_f32(x, dst, k, negative_slope, item_ct1);
        });
}

static void sqr_f32_sycl(const float *x, float *dst, const int k,
                         queue_ptr stream) {
    const int num_blocks = (k + SYCL_SQR_BLOCK_SIZE - 1) / SYCL_SQR_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sqr_f32(x, dst, k, item_ct1);
        });
}

static void upscale_f32_sycl(const float *x, float *dst, const int nb00, const int nb01,
                             const int nb02, const int nb03, const int ne10, const int ne11,
                             const int ne12, const int ne13, const float sf0, const float sf1,
                             const float sf2, const float sf3, queue_ptr stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;
    sycl::range<1> gridDim(num_blocks * SYCL_UPSCALE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(gridDim, sycl::range<1>(SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            upscale_f32(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3, item_ct1);
        });
}

static void pad_f32_sycl(const float *x, float *dst, const int ne00,
                         const int ne01, const int ne02, const int ne0,
                         const int ne1, const int ne2, queue_ptr stream) {
    int num_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(x, dst, ne0, ne00, ne01, ne02, item_ct1);
        });
}

inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}
inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_quick_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    tanh_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    relu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1, ggml_tensor *dst,
                                     const float *src0_dd, const float *src1_dd,
                                     float *dst_dd,
                                     const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardsigmoid_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardswish_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    exp_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    log_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sigmoid_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqrt_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sin_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    cos_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_step(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    step_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    neg_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    leaky_relu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), negative_slope, main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqr_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const float *src0_dd, const float *src1_dd,
                                 float *dst_dd,
                                 const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float sf0 = (float)dst->ne[0]/src0->ne[0];
    const float sf1 = (float)dst->ne[1]/src0->ne[1];
    const float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    upscale_f32_sycl(src0_dd, dst_dd, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                     dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3,
                     main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    pad_f32_sycl(src0_dd, dst_dd,
        src0->ne[0], src0->ne[1], src0->ne[2],
        dst->ne[0], dst->ne[1], dst->ne[2], main_stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_dd);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[3] == 1); // just 3D tensors supported

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    acc_f32_sycl(src0_dd, src1_dd, dst_dd, ggml_nelements(dst), src1->ne[0], src1->ne[1], src1->ne[2], nb1, nb2, offset, main_stream);

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

inline void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}


void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_sqrt);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_sin);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_cos);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_acc);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_gelu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_silu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_gelu_quick);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_tanh);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_relu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_sigmoid);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_hardsigmoid);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_hardswish);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}


void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_exp);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_log);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_neg);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_step);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_leaky_relu);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_sqr);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_upscale);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_pad);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}



void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_add);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_sub);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_mul);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(ctx, dst->src[0], dst->src[1], dst, ggml_sycl_op_div);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}
