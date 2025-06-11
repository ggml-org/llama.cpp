/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  exp_gemvq.hpp
 *
 *  Description:
 *     Exp gemv public API
 **************************************************************************/

#pragma once

#include "dpct/helper.hpp"

void ggml_sycl_op_mul_mat_exp_gemvq(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                    ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                    const int64_t row_high, const int64_t src1_ncols,
                                    const int64_t src1_padded_col_size, const dpct::queue_ptr & stream);
