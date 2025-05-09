#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"

#include <cmath>
#include <cinttypes>
#include <random>
#include <string>
#include <thread>
#include <vector>

static bool almost_equal(const double a, const double b, const double atol) {
    return fabs(a - b) < atol;
}

constexpr int64_t ne_datapoint = 2;
constexpr int64_t ne_label     = 1;
constexpr int64_t ndata        = 6;

struct helper_ctx_data {
    std::vector<ggml_opt_dataset_t>   datasets_supervised;
    std::vector<struct ggml_tensor *> data_batch;
    std::vector<struct ggml_tensor *> labels_batch;

    ggml_opt_dataset_t       dataset_unsupervised;
    struct ggml_context    * ctx_static;
    struct ggml_context    * ctx_compute;
    struct ggml_opt_params   opt_params;
    ggml_opt_context_t       opt_ctx;
    struct ggml_tensor     * inputs;
    struct ggml_tensor     * weights;
    struct ggml_tensor     * outputs;
    ggml_backend_buffer_t    buf;
    ggml_opt_result_t        result;
    ggml_opt_result_t        result2;
};

// These default values make it easier to check optimization results vs. expected values.
static ggml_opt_optimizer_params helper_get_test_opt_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);
    result.adamw.alpha = 1.0f;
    result.adamw.beta1 = 0.0f;
    result.adamw.beta2 = 0.0f;
    result.adamw.eps   = 0.0f;
    return result;
}

static helper_ctx_data helper_get_ctx_data(
        ggml_backend_sched_t    backend_sched,
        ggml_backend_t          backend,
        const bool              init_opt_ctx       = true,
        const bool              optimizer_defaults = true,
        int64_t                 nbatch_logical     = 1,
        int64_t                 nbatch_physical    = 1,
        enum ggml_opt_loss_type loss_type          = GGML_OPT_LOSS_TYPE_SUM) {
    std::vector<ggml_opt_dataset_t> datasets(ndata);
    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        ggml_opt_dataset_t dataset = ggml_opt_dataset_init(ne_datapoint, ne_label, ndata, ndata_shard);

        float * data   = ggml_get_data_f32(ggml_opt_dataset_data(  dataset));
        float * labels = ggml_get_data_f32(ggml_opt_dataset_labels(dataset));

        for (int64_t idata = 0; idata < ndata; ++idata) {
            for (int64_t id = 0; id < ne_datapoint; ++id) {
                data[  idata*ne_datapoint + id] =     16*idata + id;
            }
            for (int64_t il = 0; il < ne_label;     ++il) {
                labels[idata*ne_label     + il] = 16*(16*idata + il);
            }
        }

        datasets[ndata_shard-1] = dataset;
    }

    ggml_opt_dataset_t dataset_unsupervised = ggml_opt_dataset_init(1, 0, ndata, /*ndata_shard =*/ 1);

    float * data = ggml_get_data_f32(ggml_opt_dataset_data(dataset_unsupervised));

    for (int64_t idata = 0; idata < ndata; ++idata) {
        data[idata] = idata;
    }

    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ (2*ndata + 2)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_static = ggml_init(params);
    }
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_compute = ggml_init(params);
    }

    std::vector<struct ggml_tensor *>   data_batch(ndata);
    std::vector<struct ggml_tensor *> labels_batch(ndata);
    for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
        data_batch[ndata_batch-1]   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, ndata_batch*ne_datapoint);
        labels_batch[ndata_batch-1] = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, ndata_batch*ne_label);
    }

    struct ggml_tensor * inputs = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, nbatch_physical);
    ggml_set_name(inputs, "inputs");

    struct ggml_tensor * weights = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(weights, "weights");
    ggml_set_param(ctx_static, weights);

    struct ggml_tensor * intermediary = ggml_add(ctx_compute, inputs, weights);

    struct ggml_tensor * outputs = ggml_scale(ctx_compute, intermediary, 1.0f);
    ggml_set_name(outputs, "outputs");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    const float w0 = float(ndata)/2;
    ggml_backend_tensor_set(weights, &w0, 0, sizeof(float));

    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);
    const int32_t opt_period = nbatch_logical / nbatch_physical;

    struct ggml_opt_params opt_params = ggml_opt_default_params(backend_sched, ctx_compute, inputs, outputs, loss_type);
    opt_params.opt_period = opt_period;
    if (!optimizer_defaults) {
        opt_params.get_opt_pars = helper_get_test_opt_pars;
    }
    ggml_opt_context_t opt_ctx = init_opt_ctx ? ggml_opt_init(opt_params) : nullptr;

    ggml_opt_result_t result  = ggml_opt_result_init();
    ggml_opt_result_t result2 = ggml_opt_result_init();

    return {datasets, data_batch, labels_batch, dataset_unsupervised, ctx_static, ctx_compute, opt_params, opt_ctx, inputs, weights, outputs, buf, result, result2};
}

static void helper_free_ctx_data(struct helper_ctx_data ctx_data) {
    ggml_opt_result_free(ctx_data.result);
    ggml_opt_result_free(ctx_data.result2);
    ggml_opt_free(ctx_data.opt_ctx);
    ggml_backend_buffer_free(ctx_data.buf);
    ggml_free(ctx_data.ctx_static);
    ggml_free(ctx_data.ctx_compute);
    for (ggml_opt_dataset_t dataset : ctx_data.datasets_supervised) {
        ggml_opt_dataset_free(dataset);
    }
    ggml_opt_dataset_free(ctx_data.dataset_unsupervised);
}

static void helper_after_test(
        const char * func, const bool high_level, const std::string options,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    printf("  %s(high_level=%s%s, subtest=%s): ",
           func, high_level ? "yes" : "no", options.c_str(), subtest.c_str());
    if (subtest_ok) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;
}

static std::pair<int, int> test_dataset(ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool shuffle) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend);

    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        ggml_opt_dataset_t dataset = cd.datasets_supervised[ndata_shard-1];

        if (shuffle) {
            ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        }

        for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
            if (ndata_batch % ndata_shard != 0) {
                continue;
            }
            bool subtest_ok = true;

            struct ggml_tensor *   data_batch =   cd.data_batch[ndata_batch-1];
            struct ggml_tensor * labels_batch = cd.labels_batch[ndata_batch-1];

            std::vector<float>   data(ggml_nelements(  data_batch));
            std::vector<float> labels(ggml_nelements(labels_batch));

            std::vector<int64_t> idata_shuffled;
            const int64_t nbatches = ndata / ndata_batch;
            for (int64_t ibatch = 0; ibatch < nbatches; ++ibatch) {
                ggml_opt_dataset_get_batch(dataset, data_batch, labels_batch, ibatch);

                ggml_backend_tensor_get(  data_batch,   data.data(), 0, ggml_nbytes(  data_batch));
                ggml_backend_tensor_get(labels_batch, labels.data(), 0, ggml_nbytes(labels_batch));

                for (int64_t idata_batch = 0; idata_batch < ndata_batch; ++idata_batch) {
                    const int64_t idata = ibatch*ndata_batch + idata_batch;
                    const int64_t idata_found = data[idata_batch*ne_datapoint] / 16;
                    subtest_ok = subtest_ok && (shuffle || idata_found == idata);
                    idata_shuffled.push_back(idata_found);

                    for (int64_t id = 0; id < ne_datapoint; ++id) {
                        if (data[  idata_batch*ne_datapoint + id] != 16*idata_found + id) {
                            subtest_ok = false;
                        }
                    }
                    for (int64_t il = 0; il < ne_label;     ++il) {
                        if (labels[idata_batch*ne_label     + il] != 16*(16*idata_found + il)) {
                            subtest_ok = false;
                        }
                    }
                }
            }

            if (!shuffle || ndata % ndata_batch == 0) {
                const int ndata_max = (ndata / ndata_batch) * ndata_batch;

                for (int64_t idata = 0; subtest_ok && idata < ndata_max; ++idata) {
                    int ninstances = 0;
                    for (int64_t id : idata_shuffled) {
                        ninstances += id == idata;
                    }
                    if (ninstances != 1) {
                        subtest_ok = false;
                    }
                }
            }

            printf("  %s(shuffle=%s, ndata_shard=%" PRId64 ", ndata_batch=%" PRId64 "): ",
                   __func__, shuffle ? "yes" : "no", ndata_shard, ndata_batch);
            if (subtest_ok) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_grad(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false,
    /*nbatch_logical =*/ 999999, /*nbatch_physical =*/ 1);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    for (int idata = 0; idata < ndata; ++idata) {
        const float idataf = idata;
        ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
        ggml_opt_forward_backward(cd.opt_ctx, cd.result);
        ggml_backend_tensor_get(ggml_opt_grad_acc(cd.opt_ctx, cd.weights), grad_history.data() + idata, 0, sizeof(float));
    }

    {
        bool subtest_ok = true;
        for (int idata = 0; idata < ndata; ++idata) {
            if (grad_history[idata] != idata + 1) {
                subtest_ok = false;
            }
        }
        printf("  %s(): ", __func__);
        if (subtest_ok) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static void helper_after_test_forward_backward(
        const char * func, const bool high_level, const bool shuffle,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    std::string options = ", shuffle=";
    options += shuffle ? "yes" : "no";
    helper_after_test(func, high_level, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_forward_backward(
        ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool high_level, const bool shuffle) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor * loss = ggml_opt_loss(cd.opt_ctx);

    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    {
        int64_t ndata;
        ggml_opt_result_ndata(cd.result, &ndata);
        double loss;
        double loss_unc;
        ggml_opt_result_loss(cd.result, &loss, &loss_unc);
        double accuracy;
        double accuracy_unc;
        ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
        const bool subtest_ok = ndata == 0 && loss == 0.0 && std::isnan(loss_unc) && std::isnan(accuracy) && std::isnan(accuracy_unc);
        helper_after_test_forward_backward(__func__, high_level, shuffle, "results_initial", subtest_ok, ntest, npass);
    }

    if (high_level) {
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;
        if (shuffle) {
            ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        }
        ggml_opt_epoch(cd.opt_ctx, dataset, nullptr, cd.result, 0, nullptr, nullptr);
    } else {
        for (int idata = 0; idata < ndata; ++idata) {
            const float idataf = idata;
            ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
            ggml_opt_forward(cd.opt_ctx, cd.result);
            ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
        }
    }

    {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = weights == ndata/2;
        helper_after_test_forward_backward(__func__, high_level, shuffle, "weights_after_forward", subtest_ok, ntest, npass);
    }
    {
        int64_t ndata;
        ggml_opt_result_ndata(cd.result, &ndata);
        bool subtest_ok = ndata == 6;

        double loss;
        double loss_unc;
        ggml_opt_result_loss(cd.result, &loss, &loss_unc);
        subtest_ok = subtest_ok && loss == 33.0 && almost_equal(loss_unc, sqrt(3.5), 1e-10);

        double accuracy;
        double accuracy_unc;
        ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

        helper_after_test_forward_backward(__func__, high_level, shuffle, "results_after_forward", subtest_ok, ntest, npass);
    }

    float w0;
    ggml_backend_tensor_get(cd.weights, &w0, 0, sizeof(float));
    for (int i = 0; i < 10; ++i) {
        ggml_opt_forward_backward(cd.opt_ctx, nullptr);
    }
    ggml_backend_tensor_set(cd.weights, &w0, 0, sizeof(float));

    ggml_opt_reset(cd.opt_ctx, /*optimizer =*/ false);
    ggml_opt_result_reset(cd.result);

    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    if (high_level) {
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;
        if (shuffle) {
            ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        }
        ggml_opt_epoch(cd.opt_ctx, dataset, cd.result, nullptr, ndata, nullptr, nullptr);
    } else {
        for (int idata = 0; idata < ndata; ++idata) {
            const float idataf = idata;
            ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
            ggml_opt_forward_backward(cd.opt_ctx, cd.result);
            ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
        }
    }

    {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = weights == -ndata/2;
        helper_after_test_forward_backward(__func__, high_level, shuffle, "weights_after_forward_backward", subtest_ok, ntest, npass);
    }
    {
        int64_t ndata;
        ggml_opt_result_ndata(cd.result, &ndata);
        bool subtest_ok = ndata == 6;

        double loss;
        double loss_unc;
        ggml_opt_result_loss(cd.result, &loss, &loss_unc);
        subtest_ok = subtest_ok && loss == 18.0 && (shuffle || loss_unc == 0.0);

        double accuracy;
        double accuracy_unc;
        ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

        helper_after_test_forward_backward(__func__, high_level, shuffle, "result_after_forward_backward", subtest_ok, ntest, npass);
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_epoch_vs_fit(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    float weights_epoch;
    float weights_fit;

    {
        struct helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true);
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;

        ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        ggml_opt_epoch(cd.opt_ctx, dataset, cd.result, nullptr, ndata, nullptr, nullptr);

        ggml_backend_tensor_get(cd.weights, &weights_epoch, 0, ggml_nbytes(cd.weights));
        helper_free_ctx_data(cd);
    }
    {
        struct helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ false);
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;

        ggml_opt_fit(backend_sched, cd.ctx_compute, cd.inputs, cd.outputs, dataset,
            GGML_OPT_LOSS_TYPE_SUM, ggml_opt_get_default_optimizer_params, 1, 1, 0.0f, true);

        ggml_backend_tensor_get(cd.weights, &weights_fit, 0, ggml_nbytes(cd.weights));
        helper_free_ctx_data(cd);
    }

    const bool subtest_ok = weights_epoch == weights_fit;

    printf("  %s(): ", __func__);
    if (subtest_ok) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    return std::make_pair(npass, ntest);
}

static void helper_after_test_idata_split(
        const char * func, const bool high_level, const int epoch,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    std::string options = ", epoch=";
    options += std::to_string(epoch);
    helper_after_test(func, high_level, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_idata_split(ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool high_level) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor * loss = ggml_opt_loss(cd.opt_ctx);
    const int idata_split = ndata * 2/3;

    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    for (int epoch = 1; epoch <= 4; ++epoch) {
        if (high_level) {
            ggml_opt_epoch(cd.opt_ctx, cd.dataset_unsupervised, cd.result, cd.result2, idata_split, nullptr, nullptr);
        } else {
            int idata = 0;
            for (; idata < idata_split; ++idata) {
                const float idataf = idata;
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
                ggml_opt_forward_backward(cd.opt_ctx, cd.result);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
            for (; idata < ndata; ++idata) {
                const float idataf = idata;
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
                ggml_opt_forward(cd.opt_ctx, cd.result2);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
        }

        {
            float weights;
            ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
            const bool subtest_ok = weights == ndata/2 - epoch*idata_split;
            helper_after_test_idata_split(__func__, high_level, epoch, "weights", subtest_ok, ntest, npass);
        }
        {
            int64_t ndata_result;
            ggml_opt_result_ndata(cd.result, &ndata_result);
            bool subtest_ok = ndata_result == idata_split;

            double loss;
            double loss_unc;
            ggml_opt_result_loss(cd.result, &loss, &loss_unc);
            subtest_ok = subtest_ok && loss == 28.0 - epoch*16.0 && loss_unc == 0.0;

            double accuracy;
            double accuracy_unc;
            ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_idata_split(__func__, high_level, epoch, "results_backward", subtest_ok, ntest, npass);
        }
        {
            int64_t ndata_result;
            ggml_opt_result_ndata(cd.result2, &ndata_result);
            bool subtest_ok = ndata_result == ndata - idata_split;

            double loss;
            double loss_unc;
            ggml_opt_result_loss(cd.result2, &loss, &loss_unc);
            subtest_ok = subtest_ok && loss == 15.0 - epoch*8 && almost_equal(loss_unc, sqrt(0.5), 1e-10);

            double accuracy;
            double accuracy_unc;
            ggml_opt_result_accuracy(cd.result2, &accuracy, &accuracy_unc);
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_idata_split(__func__, high_level, epoch, "results_forward", subtest_ok, ntest, npass);
        }

        ggml_opt_result_reset(cd.result);
        ggml_opt_result_reset(cd.result2);
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static void helper_after_test_gradient_accumulation(
        const char * func, const int nbatch_physical, const enum ggml_opt_loss_type loss_type, const int epoch,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    std::string options = ", nbatch_physical=";
    options += std::to_string(nbatch_physical);
    options += ", loss_type=";
    options += loss_type == GGML_OPT_LOSS_TYPE_MEAN ? "mean" : "sum";
    options += ", epoch=";
    options += std::to_string(epoch);
    helper_after_test(func, false, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_gradient_accumulation(
        ggml_backend_sched_t backend_sched, ggml_backend_t backend, const int32_t nbatch_physical, const enum ggml_opt_loss_type loss_type) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(
        backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false, /*nbatch_logical =*/ 6, nbatch_physical, loss_type);
    struct ggml_tensor * loss = ggml_opt_loss(cd.opt_ctx);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    for (int epoch = 1; epoch <= 4; ++epoch) {
        if (nbatch_physical == 1) {
            for (int idata = 0; idata < ndata; ++idata) {
                const float idataf = idata;
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, 1*sizeof(float));
                ggml_opt_forward_backward(cd.opt_ctx, cd.result);
                ggml_backend_tensor_get(ggml_opt_grad_acc(cd.opt_ctx, cd.weights), grad_history.data() + idata, 0, 1*sizeof(float));
            }
        } else if (nbatch_physical == 2) {
            for (int idata = 0; idata < ndata; idata += 2) {
                const float idataf[2] = {float(idata + 0), float(idata + 1)};
                ggml_backend_tensor_set(cd.inputs, idataf, 0, 2*sizeof(float));
                ggml_opt_forward_backward(cd.opt_ctx, cd.result);

                grad_history[idata + 0] = 0.0f;
                ggml_backend_tensor_get(ggml_opt_grad_acc(cd.opt_ctx, cd.weights), grad_history.data() + idata + 1, 0, 1*sizeof(float));
            }
        } else {
            GGML_ASSERT(false);
        }

        {
            GGML_ASSERT(ndata == 6);
            constexpr double atol = 1e-6;
            bool subtest_ok = true;
            if (loss_type == GGML_OPT_LOSS_TYPE_SUM) {
                if (nbatch_physical == 1) {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 1.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 3.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 5.0, atol);
                } else {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 0.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 0.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 0.0, atol);
                }
                subtest_ok = subtest_ok && almost_equal(grad_history[1], 2.0, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[3], 4.0, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[5], 0.0, atol);
            } else if (loss_type == GGML_OPT_LOSS_TYPE_MEAN) {
                if (nbatch_physical == 1) {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 1.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 3.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 5.0/ndata, atol);
                } else {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 0.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 0.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 0.0/ndata, atol);
                }
                subtest_ok = subtest_ok && almost_equal(grad_history[1], 2.0/ndata, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[3], 4.0/ndata, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[5], 0.0/ndata, atol);
            } else {
                GGML_ASSERT(false);
            }
            helper_after_test_gradient_accumulation(__func__, nbatch_physical, loss_type, epoch, "grads", subtest_ok, ntest, npass);
        }
        {
            float weights;
            ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
            const bool subtest_ok = weights == (ndata/2) - epoch;
            helper_after_test_gradient_accumulation(__func__, nbatch_physical, loss_type, epoch, "weights", subtest_ok, ntest, npass);
        }
        {
            int64_t ndata_result;
            ggml_opt_result_ndata(cd.result, &ndata_result);
            bool subtest_ok = ndata_result == ndata/nbatch_physical;

            double loss;
            ggml_opt_result_loss(cd.result, &loss, /*loss_unc =*/ nullptr);
            if (loss_type == GGML_OPT_LOSS_TYPE_SUM) {
                subtest_ok = subtest_ok && loss == (39.0 - epoch*6.0);
            } else if (loss_type == GGML_OPT_LOSS_TYPE_MEAN) {
                subtest_ok = subtest_ok && almost_equal(loss, (39.0 - epoch*6.0) / ndata, 1e-6);
            } else {
                GGML_ASSERT(false);
            }

            double accuracy;
            double accuracy_unc;
            ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_gradient_accumulation(__func__, nbatch_physical, loss_type, epoch, "results", subtest_ok, ntest, npass);
        }

        ggml_opt_result_reset(cd.result);
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static ggml_opt_optimizer_params helper_get_regression_opt_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);
    result.adamw.alpha = 0.1f;
    return result;
}

static std::pair<int, int> test_regression(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    // Test for simple regression with f(x) = a*x + b

    constexpr int64_t ndata_regression = 201;
    constexpr float a_true = 1.2f;
    constexpr float b_true = 3.4f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> nd{0.0f, 0.1f};

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(1, 1, ndata_regression, ndata_regression);

    float * data   = ggml_get_data_f32(ggml_opt_dataset_data(  dataset));
    float * labels = ggml_get_data_f32(ggml_opt_dataset_labels(dataset));

    constexpr float x_min = -100.0f;
    constexpr float x_max =  100.0f;

    for (int64_t idata = 0; idata < ndata_regression; ++idata) {
        const float x = x_min + (x_max - x_min) * idata/(ndata_regression-1);
        const float y = a_true*x + b_true + nd(gen);

        data[idata]   = x;
        labels[idata] = y;
    }

    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 3*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_static = ggml_init(params);
    }
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_compute = ggml_init(params);
    }

    // The first dimension is the dimension of the datapoints, the second dimension is the number of datapoints.
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 1, ndata_regression);
    ggml_set_name(x, "x");

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(a, "a");
    ggml_set_param(ctx_static, a);

    struct ggml_tensor * b = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(b, "b");
    ggml_set_param(ctx_static, b);

    struct ggml_tensor * f = ggml_add(ctx_compute, ggml_mul(ctx_compute, x, a), b);
    ggml_set_name(f, "f");
    ggml_set_param(ctx_static, f);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    const float a0 = 1.0f;
    const float b0 = 3.0f;
    ggml_backend_tensor_set(a, &a0, 0, sizeof(float));
    ggml_backend_tensor_set(b, &b0, 0, sizeof(float));

    ggml_opt_fit(backend_sched, ctx_compute, x, f, dataset, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
        helper_get_regression_opt_pars, 100, ndata_regression, 0.0f, true);

    {
        float a_fit;
        ggml_backend_tensor_get(a, &a_fit, 0, sizeof(float));
        float b_fit;
        ggml_backend_tensor_get(b, &b_fit, 0, sizeof(float));
        const bool subtest_ok = almost_equal(a_fit, a_true, 1e-2) && almost_equal(b_fit, b_true, 1e-2);
        printf("  %s(subtest=weights): ", __func__);
        if (subtest_ok) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_opt_dataset_free(dataset);

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_backend(ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int npass = 0;
    int ntest = 0;

    for (bool shuffle : {false, true}) {
        std::pair<int, int> partial = test_dataset(backend_sched, backend, shuffle);
        npass += partial.first;
        ntest += partial.second;
    }
    {
        std::pair<int, int> partial = test_grad(backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}){
        for (bool shuffle : {false, true}) {
            if (!high_level && shuffle) {
                continue;
            }

            std::pair<int, int> partial = test_forward_backward(backend_sched, backend, high_level, shuffle);
            npass += partial.first;
            ntest += partial.second;
        }
    }
    {
        std::pair<int, int> partial = test_epoch_vs_fit(backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}){
        std::pair<int, int> partial = test_idata_split(backend_sched, backend, high_level);
        npass += partial.first;
        ntest += partial.second;
    }
    for (int32_t nbatch_physical : {2, 1}) {
        for (enum ggml_opt_loss_type loss_type : {GGML_OPT_LOSS_TYPE_SUM, GGML_OPT_LOSS_TYPE_MEAN}) {
            std::pair<int, int> partial = test_gradient_accumulation(backend_sched, backend, nbatch_physical, loss_type);
            npass += partial.first;
            ntest += partial.second;
        }
    }
    {
        std::pair<int, int> partial = test_regression(backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }

    return std::make_pair(npass, ntest);
}

int main(void) {
    const size_t dev_count = ggml_backend_dev_count();
    printf("Testing %zu devices\n\n", dev_count);
    size_t n_ok = 0;

    std::vector<ggml_backend_dev_t> devs;
    std::vector<ggml_backend_t>     backends;

    for (size_t i = 0; i < dev_count; ++i) {
        devs.push_back(ggml_backend_dev_get(i));

        ggml_backend_t backend = ggml_backend_dev_init(devs[i], NULL);
        GGML_ASSERT(backend != NULL);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, std::thread::hardware_concurrency() / 2);
        }

        backends.push_back(backend);
    }

    for (size_t i = 0; i < dev_count; ++i) {
        // Put the backend to be tested in front so that it's prioritized:
        std::vector<ggml_backend_t> backends_modded = {backends[i]};
        backends_modded.insert(backends_modded.end(), backends.begin(), backends.end());

        ggml_backend_sched_t backend_sched = ggml_backend_sched_new(
            backends_modded.data(), nullptr, backends_modded.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);

        printf("Backend %zu/%zu: %s\n", i + 1, dev_count, ggml_backend_dev_name(devs[i]));
        printf("  Device description: %s\n", ggml_backend_dev_description(devs[i]));
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(devs[i], &free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");

        std::pair<int, int> result = test_backend(backend_sched, backends[i]);

        printf("  %d/%d tests passed\n", result.first, result.second);
        printf("  Backend %s: ", ggml_backend_name(backends[i]));
        if (result.first == result.second) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_sched_free(backend_sched);
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    printf("%zu/%zu backends passed\n", n_ok, dev_count);
    if (n_ok != dev_count) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }
    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
