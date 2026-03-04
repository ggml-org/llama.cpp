#include "common.h"
#include "llama.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#define DEBUG_POS 5

static void print_debug_tensor(struct ggml_tensor * t, bool with_data = true) {
    printf("%s: %s (%s): [%d, %d]\n", __func__, t->name, ggml_type_name(t->type), (int) t->ne[0], (int) t->ne[1]);
    if (!with_data) return;
    printf("%s: %s[0] = [", __func__, t->name);
    for (size_t i = 0; i <= DEBUG_POS; i++) {
        printf(" %f,", ggml_get_f32_nd(t, i, 0, 0, 0));
    }
    printf(" ... ]\n");
}

namespace PCA {

// input params for PCA computations
struct pca_params {
    int n_threads = 1;
    int n_batch = 20; // number of iterations do to in one batch. larger the batch, more memory is used
    int n_iterations = 1000;
    float tolerance = 1e-7;

    // for debugging
    int i_layer = 0;
    int n_layers = 0;
};

// result from each iteration
struct pca_result {
    struct ggml_tensor * calculated_square = NULL;
    std::vector<struct ggml_tensor *> eigenvectors;
    std::vector<float> distances;
};

struct pca_model {
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;      // context to compute graph on target device
    struct ggml_context * ctx_host; // host context to store results

    // tensors on target device
    struct ggml_tensor * dev_input;
    struct ggml_tensor * dev_square;
    struct ggml_tensor * dev_eigenvector;

    pca_model(struct ggml_tensor * t_input) {
#ifdef GGML_USE_CUDA
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        backend = ggml_backend_cuda_init(0); // init device 0
        if (!backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
#endif

// TODO: enable Metal support when support for GGML_OP_SQRT is added
// #ifdef GGML_USE_METAL
//         fprintf(stderr, "%s: using Metal backend\n", __func__);
//         backend = ggml_backend_metal_init();
//         if (!backend) {
//             fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
//         }
// #endif

        // if there aren't GPU Backends fallback to CPU backend
        if (!backend) {
            backend = ggml_backend_cpu_init();
        }

        const int num_tensors = 4;
        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        auto n_samples = t_input->ne[0];
        auto n_embd    = t_input->ne[1];

        dev_input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, n_embd);
        dev_square      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,    n_embd);
        dev_eigenvector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        ggml_set_name(dev_input,       "dev_input");
        ggml_set_name(dev_square,      "dev_square");
        ggml_set_name(dev_eigenvector, "dev_eigenvector");
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(dev_input, t_input->data, 0, ggml_nbytes(t_input));

        // initialize eigenvector to random normalized vector
        {
            std::vector<float> random_vec(ggml_nelements(dev_eigenvector), 0.0);
            std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            float sum_sqr = 0.0; // for normalizing random_vec
            for (size_t i = 0; i < random_vec.size(); ++i) {
                float f = distribution(generator);
                sum_sqr += f * f;
                random_vec[i] = f;
            }
            // normalize it
            float random_vec_norm = std::sqrt(sum_sqr);
            for (size_t i = 0; i < random_vec.size(); ++i) {
                random_vec[i] /= random_vec_norm;
            }
            ggml_backend_tensor_set(dev_eigenvector, random_vec.data(), 0, ggml_nbytes(dev_eigenvector));
        }
    }

    ~pca_model() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
    }
};

static struct ggml_cgraph * build_graph_piter(
        const struct pca_params & params,
        const pca_model & model,
        bool calc_square = false) {
    GGML_ASSERT(params.n_batch > 0);
    // TODO: buf_size must be able to scale with params.n_batch
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // turn v_diff_original into square matrix if needed
    struct ggml_tensor * tmp_square;
    if (calc_square) {
        tmp_square = ggml_mul_mat(ctx0, model.dev_input, model.dev_input);
        ggml_set_name(tmp_square, "tmp_square");
    }

    struct ggml_tensor * b_tensor;
    struct ggml_tensor * distance;
    struct ggml_tensor * old_eigen    = model.dev_eigenvector;
    struct ggml_tensor * input_square = calc_square ? tmp_square : model.dev_square;

    for (int i = 0; i < params.n_batch; ++i) {
        // b_tensor = square * eigenvector^T
        b_tensor = ggml_mul_mat(ctx0, input_square, old_eigen);
        ggml_set_name(b_tensor, "b_tensor");

        // normalize
        b_tensor = ggml_div_inplace(ctx0,
            b_tensor,
            ggml_sqrt_inplace(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, b_tensor)))
        );
        ggml_format_name(b_tensor, "b_tensor_norm_%d", i);

        // calculate distance(new eigenvector - old eigenvector)
        // we don't use ggml_sub because it may not be implemented on GPU backend
        struct ggml_tensor * new_sub_old = ggml_add(ctx0, old_eigen, ggml_scale(ctx0, b_tensor, -1));
        distance = ggml_sqrt_inplace(ctx0,
            ggml_sum_rows(ctx0, ggml_sqr_inplace(ctx0, new_sub_old)));
        ggml_format_name(distance, "distance_%d", i);

        old_eigen = b_tensor;

        // build operations nodes
        ggml_build_forward_expand(gf, distance);
    }

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

static ggml_status compute_piter(
        const struct pca_params & params,
        const pca_model & model,
        struct ggml_cgraph * gf,
        ggml_gallocr_t allocr,
        struct pca_result & result) {
    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);
    }

    ggml_status res = ggml_backend_graph_compute(model.backend, gf);
    if (res == GGML_STATUS_SUCCESS) {
        auto extract_i = [](std::string prefix, std::string str) -> int {
            int i = -1;
            if (str.rfind(prefix, 0) == 0) {
                sscanf(str.c_str(), (prefix + "%d").c_str(), &i);
            }
            return i;
        };
        result.calculated_square = NULL;
        result.eigenvectors.clear();
        result.distances.clear();
        result.eigenvectors.resize(params.n_batch);
        result.distances.resize(params.n_batch);
        // get output nodes
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            auto node = ggml_graph_node(gf, i);
            int iter = -1;
            // find b_tensor (without copying data from device)
            if ((iter = extract_i("b_tensor_norm_", node->name)) > -1) {
                result.eigenvectors[iter] = node;
            }
            // find distances, then copy data from device
            if ((iter = extract_i("distance_", node->name)) > -1) {
                float d;
                ggml_backend_tensor_get(node, &d, 0, sizeof(float));
                result.distances[iter] = d;
                // std::cout << node->name << " = " << d << "\n";
            }
            // find tmp_square if it exists (without copying data from device)
            if (std::string(node->name) == "tmp_square") {
                result.calculated_square = node;
            }
        }
    }
    return res;
}

static void power_iteration(
        const struct pca_params & params,
        struct ggml_tensor * input, // shape of input: [n_samples, n_embd]
        struct ggml_tensor * output) {
    //printf("in power iteration\n");
    struct pca_model model(input);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    struct pca_result result;
    struct ggml_tensor * last_eigenvector = NULL;

    int n_iters = params.n_iterations / params.n_batch; // more batch, fewer iterations
    for (int iter = 0; iter < n_iters; ++iter) {
        bool calc_square = (iter == 0); // only need to calculate square for first iteration
        struct ggml_cgraph * gf = build_graph_piter(params, model, calc_square);
        // ggml_graph_dump_dot(gf, nullptr, "/tmp/_cgraph.dot");
        compute_piter(params, model, gf, allocr, result);

        for (size_t k = 0; k < result.distances.size(); ++k) {
            last_eigenvector = result.eigenvectors[k];
            if (result.distances[k] < params.tolerance) {
                break; // done
            }
        }

        if (calc_square) {
            // copy and store the square matrix if needed
            GGML_ASSERT(result.calculated_square != NULL);
            ggml_backend_tensor_copy(result.calculated_square, model.dev_square);
        }

        {
            // copy last eigen vector and store as input for next iteration
            GGML_ASSERT(last_eigenvector != NULL);
            ggml_backend_tensor_copy(last_eigenvector, model.dev_eigenvector);
        }

        printf("%s: layer %d/%d, iteration: %d / total: %d (batch = %d) ...\n",
            __func__, params.i_layer+1, params.n_layers, iter+1, n_iters, params.n_batch);
    }

    // get output tensor
    GGML_ASSERT(last_eigenvector);
    ggml_backend_tensor_get(last_eigenvector, output->data, 0, ggml_nbytes(last_eigenvector));
    //print_debug_tensor(output);
    ggml_gallocr_free(allocr);

    // TODO @ngxson : The output vector is randomly inverted
    // Solution: https://github.com/ggml-org/llama.cpp/pull/8069#issuecomment-2185328171
}

static void run_pca(
        struct pca_params & params,
        const std::vector<struct ggml_tensor *> & v_input, // shape of v_input[0]: [n_samples, n_embd]
        const std::vector<struct ggml_tensor *> & v_output) {
    printf("%s: Running PCA...\n", __func__);
    for (size_t il = 0; il < v_input.size(); ++il) {

        // prepare output vector
        struct ggml_tensor * ctrl_out = v_output[il];
        ggml_format_name(ctrl_out, "direction.%zu", il+1);

        // run power_iteration
        params.i_layer = il;
        params.n_layers = v_input.size();
        power_iteration(params, v_input[il], ctrl_out);
        printf("%s: Done layer %d / %d\n", __func__, (int) il+1, (int) v_input.size());
    }
}

}
