#pragma once

#include "ggml-cuda.h"
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <array>
#include <fstream>

namespace ggml::vendors
{

constexpr int MAX_WORKSPACE_SIZE = 4194304; // 4MB
constexpr int BLOCK_SIZE = 64;
constexpr char USE_HIPBLASLT_GROUPED_GEMM_ENV[] = "USE_HIPBLASLT_GROUPED_GEMM";
constexpr char HIPBLASLT_GROUPED_GEMM_FILE_ENV[] = "HIPBLASLT_GROUPED_GEMM_FILE";
constexpr char HIPBLASLT_DEFAULT_BENCH_RESULTS_FILE[] = "hipblaslt_bench_results.txt";
constexpr int HIPBLASLT_GROUPED_GEMM_DISABLE = 0;
constexpr int HIPBLASLT_GROUPED_GEMM_ALL_ALGO = 1;
constexpr int HIPBLASLT_GROUPED_GEMM_BENCH = 2;
constexpr int HIPBLASLT_GROUPED_GEMM_SOL_FILE = 3;

#if !(HIP_VERSION >= 60500000)
inline hipDataType getHipblasLtDatatype(cudaDataType_t dt){
    switch (dt){
        case CUDA_R_16F:
            return HIP_R_16F;
        case CUDA_R_16BF:
            return HIP_R_16BF;
        case CUDA_R_32F:
            return HIP_R_32F;
        default:
            break;
    }
    GGML_LOG_WARN("Unsupported hipblas datatype %d, defaulting to HIP_R_16F\n", dt);
    return HIP_R_16F; // default to 16F
}

inline hipblasComputeType_t getHipblasLtComputeType(cublasComputeType_t ct){
    switch (ct){
        case CUBLAS_COMPUTE_16F:
            return HIPBLAS_COMPUTE_16F;
        case CUBLAS_COMPUTE_32F:
            return HIPBLAS_COMPUTE_32F;
        default:
            break;
    }
    GGML_LOG_WARN("Unsupported hipblas compute type %d, defaulting to HIPBLAS_COMPUTE_16F\n", ct);
    return HIPBLAS_COMPUTE_16F; // default to 16F
}
#endif


static int getHipblasltBatchedGemmEnvVal(){
    static int hipblaslt_env_var = -1;
    if(hipblaslt_env_var >= 0){
        return hipblaslt_env_var;
    }
    const char* hipblasltEnvVal = std::getenv(USE_HIPBLASLT_GROUPED_GEMM_ENV);
    if(hipblasltEnvVal == nullptr){
        hipblaslt_env_var = HIPBLASLT_GROUPED_GEMM_DISABLE; // default to 0 if not set
        return hipblaslt_env_var;
    }
    if(strncmp(hipblasltEnvVal, "1", 1) == 0)
    {
        hipblaslt_env_var = HIPBLASLT_GROUPED_GEMM_ALL_ALGO;
    }
    else if(strncmp(hipblasltEnvVal, "2", 1) == 0)
    {
        hipblaslt_env_var = HIPBLASLT_GROUPED_GEMM_BENCH;
    }
    else if(strncmp(hipblasltEnvVal, "3", 1) == 0)
    {
        hipblaslt_env_var = HIPBLASLT_GROUPED_GEMM_SOL_FILE;
    }
    else
    {
        hipblaslt_env_var = HIPBLASLT_GROUPED_GEMM_DISABLE;
    }
    return hipblaslt_env_var;
}

inline int DIVUP(int a, int b) {
    return (a + b - 1) / b;
}
class OfflineBenchResultsWriter{
public:
    static OfflineBenchResultsWriter& getInstance() {
        static OfflineBenchResultsWriter instance;
        return instance;
    }

    void setAlgoIndex(const std::string& problem_hash, int algo_index) {
        std::lock_guard<std::mutex> lock(mutex_);
        problem_hash_to_algo_index_[problem_hash] = algo_index;
    }

    ~OfflineBenchResultsWriter(){
        for(const auto& pair : problem_hash_to_algo_index_){
            printf("%s,%d\n", pair.first.c_str(), pair.second);
        }
        problem_hash_to_algo_index_.clear();
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, int> problem_hash_to_algo_index_;
};

class OfflineBenchResultsReader{
public:
    static OfflineBenchResultsReader& getInstance() {
        static OfflineBenchResultsReader instance;
        return instance;
    }

    using CSVRow = std::vector<std::string>;
    using CSVData = std::vector<CSVRow>;

    bool read(){
        std::lock_guard<std::mutex> lock(mutex_);
        if(initialized_) {
            return true;
        }
        std::string filename = HIPBLASLT_DEFAULT_BENCH_RESULTS_FILE;
        const char* hipblaslt_grouped_gemm_file = std::getenv(HIPBLASLT_GROUPED_GEMM_FILE_ENV);
        if(hipblaslt_grouped_gemm_file != nullptr){
            filename = std::string(hipblaslt_grouped_gemm_file);
        }
        std::ifstream file(filename);
        if (!file.is_open()) {
            GGML_LOG_WARN("Failed to open %s\n", filename.c_str());
            return false;
        }
        CSVData data;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string field;
            CSVRow row;
            while (std::getline(iss, field, ',')) {
                row.push_back(field);
            }
            data.push_back(row);
        }
        file.close();
        for(const auto& row : data){
            if(row.size() != 2){
                continue;
            }
            std::string problem_hash = row[0];
            int algo_index = std::stoi(row[1]);
            problem_hash_to_algo_index_[problem_hash] = algo_index;
        }

        initialized_ = true;
        return true;
    }

    int getAlgoIndex(const std::string& problem_hash) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(problem_hash_to_algo_index_.count(problem_hash) == 0) {
            return -1;
        }
        return problem_hash_to_algo_index_[problem_hash];
    }

private:
    std::mutex mutex_;
    bool initialized_ = false;
    std::unordered_map<std::string, int> problem_hash_to_algo_index_;
};

static __global__ void k_hipblaslt_args(hipblaslt_ext::UserArguments *d_userArgs,
        const void **a, const void **b, void **c,
        const uint32_t m, const uint32_t n, const uint32_t k,
        const uint32_t stride_a, const uint32_t stride_b, const uint32_t stride_c,
        const uint32_t stride_a2, const uint32_t stride_b2, const uint32_t stride_c2,
        const uint32_t total_batch) {
    int target_inx = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_inx >= total_batch) {
        return;
    }

    d_userArgs[target_inx].m = m;
    d_userArgs[target_inx].n = n;
    d_userArgs[target_inx].k = k;
    d_userArgs[target_inx].strideA1 = stride_a;
    d_userArgs[target_inx].strideB1 = stride_b;
    d_userArgs[target_inx].strideC1 = stride_c;
    d_userArgs[target_inx].strideD1 = stride_c;
    d_userArgs[target_inx].strideA2 = stride_a2;
    d_userArgs[target_inx].strideB2 = stride_b2;
    d_userArgs[target_inx].strideC2 = stride_c2;
    d_userArgs[target_inx].strideD2 = stride_c2;
    d_userArgs[target_inx].batch = 1;
    d_userArgs[target_inx].a = (void*) (((const char **)a)[target_inx]);
    d_userArgs[target_inx].b = (void*) (((const char **)b)[target_inx]);
    d_userArgs[target_inx].c = (void*) (((      char **)c)[target_inx]);
    d_userArgs[target_inx].d = (void*) (((      char **)c)[target_inx]);
}

struct HipblasltBatchedGemmCtxData{
    hipblasLtHandle_t handle{nullptr};
    void* workspace{nullptr};
    size_t workspace_size{0};
    hipStream_t stream{nullptr};
};

struct GemmProblemDesc {
    hipblasOperation_t op_a;
    hipblasOperation_t op_b;
    hipDataType type_a;
    hipDataType type_b;
    hipDataType type_c;
    hipblasComputeType_t type_compute;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t lda = 0;
    uint32_t ldb = 0;
    uint32_t ldc = 0;
    uint32_t stride_a = 0;
    uint32_t stride_b = 0;
    uint32_t stride_c = 0;
    uint32_t batch_count = 0;
    std::string hash;
    void* alpha = nullptr;
    void* beta = nullptr;

    inline void processHash(){
        std::stringstream ss;
        ss << op_a << "|" << op_b << "|" << type_a << "|" << type_b << "|" << type_c << "|" << type_compute
           << "|" << m << "|" << n << "|" << k << "|" << lda << "|" << ldb << "|" << ldc
           << "|" << batch_count;
        hash = ss.str();
    }
};

class IHipblasltBatchedGemm{
public:
    virtual ~IHipblasltBatchedGemm() = default;
    virtual hipblasStatus_t init(const GemmProblemDesc& prob, const HipblasltBatchedGemmCtxData& ctx) = 0;
    virtual hipblasStatus_t runGemm(GemmProblemDesc& prob, const void** a, const void** b, void** c, hipStream_t stream) = 0;
};


class CHipblasltBatchedGemmBase : public IHipblasltBatchedGemm{
public:
    CHipblasltBatchedGemmBase() : IHipblasltBatchedGemm() {}

    ~CHipblasltBatchedGemmBase() override = default;

    hipblasStatus_t init(const GemmProblemDesc& prob, const HipblasltBatchedGemmCtxData& ctx) override {
        if (initialized_) {
            return HIPBLAS_STATUS_SUCCESS;
        }
        user_arg_size_in_bytes_ = prob.batch_count * sizeof(hipblaslt_ext::UserArguments);
        if (ctx.handle == nullptr || ctx.workspace == nullptr || ctx.stream == nullptr || ctx.workspace_size < user_arg_size_in_bytes_) {
            GGML_LOG_WARN("Invalid context data\n");
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        d_userArgs_ = (hipblaslt_ext::UserArguments*) ctx.workspace;
        workspace_left_ = (void*) ((char*) ctx.workspace + user_arg_size_in_bytes_);
        workspace_left_size_ = ctx.workspace_size - user_arg_size_in_bytes_;
        gemm_pref_ = std::make_unique<hipblaslt_ext::GemmPreference>();
        gemm_pref_->setMaxWorkspaceBytes(workspace_left_size_);
        problem_type_ = std::make_unique<hipblaslt_ext::GemmProblemType>();
#if HIP_VERSION >= 60500000
        problem_type_->setOpA(prob.op_a);
        problem_type_->setOpB(prob.op_b);
        problem_type_->setTypeA(prob.type_a);
        problem_type_->setTypeB(prob.type_b);
        problem_type_->setTypeC(prob.type_c);
        problem_type_->setTypeD(prob.type_c);
        problem_type_->setTypeCompute(prob.type_compute);
#else
        problem_type_->op_a = prob.op_a;
        problem_type_->op_b = prob.op_b;
        problem_type_->type_a = prob.type_a;
        problem_type_->type_b = prob.type_b;
        problem_type_->type_c = prob.type_c;
        problem_type_->type_d = prob.type_c;
        problem_type_->type_compute = prob.type_compute;
#endif
        grouped_gemm_ = std::make_unique<hipblaslt_ext::GroupedGemm>(ctx.handle, prob.op_a, prob.op_b, 
            prob.type_a, prob.type_b, prob.type_c, prob.type_c, prob.type_compute);

        std::vector<int64_t>                     Ms(prob.batch_count, prob.m);
        std::vector<int64_t>                     Ns(prob.batch_count, prob.n);
        std::vector<int64_t>                     Ks(prob.batch_count, prob.k);
        std::vector<int64_t>                     ldas(prob.batch_count, prob.lda);
        std::vector<int64_t>                     ldbs(prob.batch_count, prob.ldb);
        std::vector<int64_t>                     ldcs(prob.batch_count, prob.ldc);
        std::vector<int64_t>                     ldds(prob.batch_count, prob.ldc);
        std::vector<int64_t>                     strideas(prob.batch_count, prob.stride_a);
        std::vector<int64_t>                     stridebs(prob.batch_count, prob.stride_b);
        std::vector<int64_t>                     stridecs(prob.batch_count, prob.stride_c);
        std::vector<int64_t>                     strideds(prob.batch_count, prob.stride_c);
        std::vector<int64_t>                     batch_counts(prob.batch_count, 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(prob.batch_count);
        std::vector<hipblaslt_ext::GemmInputs> inputs(prob.batch_count);
        for(int batch =0; batch < prob.batch_count; ++batch) {
#if HIP_VERSION >= 60500000
            inputs[batch].setA(workspace_left_);
            inputs[batch].setB(workspace_left_);
            inputs[batch].setC(workspace_left_);
            inputs[batch].setD(workspace_left_);
            inputs[batch].setAlpha(prob.alpha);
            inputs[batch].setBeta(prob.beta);
#else
            inputs[batch].a = workspace_left_;
            inputs[batch].b = workspace_left_;
            inputs[batch].c = workspace_left_;
            inputs[batch].d = workspace_left_;
            inputs[batch].alpha = prob.alpha;
            inputs[batch].beta = prob.beta;
#endif
        }
        grouped_gemm_->setProblem(Ms,
                        Ns,
                        Ks,
                        batch_counts,
                        ldas,
                        ldbs,
                        ldcs,
                        ldds,
                        strideas,
                        stridebs,
                        stridecs,
                        strideds,
                        epilogues,
                        inputs,
                        *problem_type_);

        hipblasStatus_t status = initProblem(prob, ctx);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            //GGML_LOG_DEBUG("Failed to initialize batched matrix multiplication problem, status: %d\n", status);
            return status;
        }

        hipblaslt_ext::UserArguments* userArgs{nullptr};
        CUDA_CHECK(hipHostMalloc(&userArgs, user_arg_size_in_bytes_));
        status = grouped_gemm_->getDefaultValueForDeviceUserArguments(userArgs);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_WARN("Failed to get default user arguments for batched matrix multiplication, status: %d\n", status);
            CUDA_CHECK(hipHostFree(userArgs));
            return status;
        }
        CUDA_CHECK(hipMemcpyAsync(d_userArgs_, userArgs, user_arg_size_in_bytes_, hipMemcpyHostToDevice, ctx.stream));
        CUDA_CHECK(hipStreamSynchronize(ctx.stream));
        CUDA_CHECK(hipHostFree(userArgs));

        initialized_ = true;
        return HIPBLAS_STATUS_SUCCESS;
    }

protected:
    virtual hipblasStatus_t initProblem(const GemmProblemDesc& prob, const HipblasltBatchedGemmCtxData& ctx) = 0;
    bool initialized_ = false;
    hipblaslt_ext::UserArguments* d_userArgs_{nullptr};
    size_t user_arg_size_in_bytes_{0};
    void* workspace_left_{nullptr};
    size_t workspace_left_size_{0};
    std::unique_ptr<hipblaslt_ext::GemmPreference> gemm_pref_;
    std::unique_ptr<hipblaslt_ext::GemmProblemType> problem_type_;
    std::unique_ptr<hipblaslt_ext::GroupedGemm> grouped_gemm_;
};


class CHipblasltBatchedGemmBenchmark : public CHipblasltBatchedGemmBase{
public:
    CHipblasltBatchedGemmBenchmark() : CHipblasltBatchedGemmBase() {}

    ~CHipblasltBatchedGemmBenchmark() override = default;

    hipblasStatus_t runGemm(GemmProblemDesc& prob, const void** a, const void** b, void** c, hipStream_t stream) override {
        if(!initialized_) {
            GGML_LOG_WARN("Gemm not initialized for %s\n", prob.hash.c_str());
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        }
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize(DIVUP(prob.batch_count, BLOCK_SIZE));

        k_hipblaslt_args<<<gridSize, blockSize, 0, stream>>>(d_userArgs_,
                a, b, c,
                prob.m, prob.n, prob.k,
                prob.lda, prob.ldb, prob.ldc,
                prob.stride_a, prob.stride_b, prob.stride_c,
                prob.batch_count);
        if (hipGetLastError() != hipSuccess) {
            GGML_LOG_WARN("Failed to launch kernel k_hipblaslt_args: %s\n", hipGetErrorString(hipGetLastError()));
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        float min_event_ms = FLT_MAX;
        int cur_algo_best_index = -1;
        for(int i=0; i< valid_idx_.size(); i++) {
            int idx = valid_idx_[i];
            hipblasStatus_t status = grouped_gemm_->initialize(heuristic_result_[idx].algo, workspace_left_);
            if(status != HIPBLAS_STATUS_SUCCESS) {
                GGML_LOG_WARN("Failed to initialize for idx: %d\n, workspace_left_: %ld", idx, workspace_left_size_);
                return status;
            }
            float event_ms{};
            hipEvent_t start, stop;
            static_cast<void>(hipEventCreate(&start));
            static_cast<void>(hipEventCreate(&stop));
            static_cast<void>(hipEventRecord(start, stream));

            status = grouped_gemm_->run(d_userArgs_, stream);
            if(status != HIPBLAS_STATUS_SUCCESS) {
                GGML_LOG_WARN("Failed to run batched matrix multiplication, status: %d\n", status);
                return status;
            }
            static_cast<void>(hipEventRecord(stop, stream));
            static_cast<void>(hipEventSynchronize(stop));
            static_cast<void>(hipEventElapsedTime(&event_ms, start, stop));
            static_cast<void>(hipEventDestroy(start));
            static_cast<void>(hipEventDestroy(stop));
            if(event_ms < min_event_ms) {
                min_event_ms = event_ms;
                cur_algo_best_index = idx;
            }
        }
        if(cur_algo_best_index == -1){
            GGML_LOG_WARN("No valid algorithm found for batched matrix multiplication\n");
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }

        int algo_index = hipblaslt_ext::getIndexFromAlgo(heuristic_result_[valid_idx_[cur_algo_best_index]].algo);
        OfflineBenchResultsWriter::getInstance().setAlgoIndex(prob.hash, algo_index);
        return HIPBLAS_STATUS_SUCCESS;
    }
protected:
    hipblasStatus_t initProblem(const GemmProblemDesc& prob, const HipblasltBatchedGemmCtxData& ctx) override{
        hipblaslt_ext::GemmType gemmType = hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM;
        hipblasStatus_t status = hipblaslt_ext::getAllAlgos(ctx.handle,
                                                     gemmType,
                                                     prob.op_a,
                                                     prob.op_b,
                                                     prob.type_a,
                                                     prob.type_b,
                                                     prob.type_c,
                                                     prob.type_c,
                                                     prob.type_compute,
                                                     heuristic_result_);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_WARN("Failed to get all algorithms for batched matrix multiplication %d\n", status);
            return status;
        }
        if (heuristic_result_.empty()) {
            GGML_LOG_DEBUG("No heuristic results found for batched matrix multiplication, hash: %s\n", prob.hash.c_str());
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }

        for(size_t i = 0; i < heuristic_result_.size(); ++i){
            size_t cur_workspace_size = 0;
            if(grouped_gemm_->isAlgoSupported(heuristic_result_[i].algo, cur_workspace_size) == HIPBLAS_STATUS_SUCCESS){
                if(cur_workspace_size <= workspace_left_size_){
                    valid_idx_.push_back(i);   
                }
            }
        }

        if(valid_idx_.empty()){
            GGML_LOG_DEBUG("No valid heuristic results found for batched matrix multiplication, hash: %s\n", prob.hash.c_str());
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result_;
    std::vector<int> valid_idx_;
};


class CHipblasltBatchedGemmAllAlgos : public CHipblasltBatchedGemmBase{
public:
    CHipblasltBatchedGemmAllAlgos() : CHipblasltBatchedGemmBase() {}

    ~CHipblasltBatchedGemmAllAlgos() override = default;

    hipblasStatus_t runGemm(GemmProblemDesc& prob, const void** a, const void** b, void** c, hipStream_t stream) override {
        if(!initialized_) {
            GGML_LOG_WARN("Gemm not initialized\n");
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        }
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize(DIVUP(prob.batch_count, BLOCK_SIZE));

        k_hipblaslt_args<<<gridSize, blockSize, 0, stream>>>(d_userArgs_,
                a, b, c,
                prob.m, prob.n, prob.k,
                prob.lda, prob.ldb, prob.ldc,
                prob.stride_a, prob.stride_b, prob.stride_c,
                prob.batch_count);

        if (hipGetLastError() != hipSuccess) {
            GGML_LOG_WARN("Failed to launch kernel k_hipblaslt_args: %s\n", hipGetErrorString(hipGetLastError()));
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }

        hipblasStatus_t status = grouped_gemm_->run(d_userArgs_, stream);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_WARN("Failed to run batched matrix multiplication, status %d\n", status);
            return status;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
protected:
    hipblasStatus_t initProblem(const GemmProblemDesc& prob, const HipblasltBatchedGemmCtxData& ctx) override{
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
        hipblaslt_ext::GemmType gemmType = hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM;
        hipblasStatus_t status = hipblaslt_ext::getAllAlgos(ctx.handle,
                                                     gemmType,
                                                     prob.op_a,
                                                     prob.op_b,
                                                     prob.type_a,
                                                     prob.type_b,
                                                     prob.type_c,
                                                     prob.type_c,
                                                     prob.type_compute,
                                                     heuristic_result);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_WARN("Failed to get all algorithms for batched matrix multiplication\n");
            return status;
        }
        if (heuristic_result.empty()) {
            GGML_LOG_WARN("No heuristic results found for batched matrix multiplication, hash: %s\n", prob.hash.c_str());
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
        
        int first_index = -1;
        for(size_t i = 0; i < heuristic_result.size(); ++i){
            size_t cur_workspace_size = 0;
            if(grouped_gemm_->isAlgoSupported(heuristic_result[i].algo, cur_workspace_size) == HIPBLAS_STATUS_SUCCESS){
                if(cur_workspace_size <= workspace_left_size_){
                    first_index = i;
                    break;
                }
            }
        }

        if(first_index == -1){
            //GGML_LOG_DEBUG("No valid heuristic results found for batched matrix multiplication, hash: %s\n", prob.hash.c_str());
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }

        status = grouped_gemm_->initialize(heuristic_result[first_index].algo, workspace_left_);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_DEBUG("Failed to initialize %d: \n", first_index);
            return status;
        }

        return HIPBLAS_STATUS_SUCCESS;
    }
};

class CHipblasltBatchedGemmSolution : public CHipblasltBatchedGemmBase{
public:
    CHipblasltBatchedGemmSolution() : CHipblasltBatchedGemmBase() {}

    ~CHipblasltBatchedGemmSolution() override = default;

    hipblasStatus_t runGemm(GemmProblemDesc& prob, const void** a, const void** b, void** c, hipStream_t stream) override {
        if(!initialized_) {
            GGML_LOG_WARN("Gemm not initialized\n");
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        }
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize(DIVUP(prob.batch_count, BLOCK_SIZE));

        k_hipblaslt_args<<<gridSize, blockSize, 0, stream>>>(d_userArgs_,
                a, b, c,
                prob.m, prob.n, prob.k,
                prob.lda, prob.ldb, prob.ldc,
                prob.stride_a, prob.stride_b, prob.stride_c,
                prob.batch_count);

        if (hipGetLastError() != hipSuccess) {
            GGML_LOG_WARN("Failed to launch kernel k_hipblaslt_args: %s\n", hipGetErrorString(hipGetLastError()));
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
        hipblasStatus_t status = grouped_gemm_->run(d_userArgs_, stream);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_WARN("Failed to run batched matrix multiplication, status: %d\n", status);
            return status;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

protected:
    hipblasStatus_t initProblem(const GemmProblemDesc& prob, const HipblasltBatchedGemmCtxData& ctx) override {
        auto& offline_bench_result = OfflineBenchResultsReader::getInstance();
        int algo_index = offline_bench_result.getAlgoIndex(prob.hash);
        if(algo_index == -1){
            GGML_LOG_WARN("No algo index found for problem hash: %s\n", prob.hash.c_str());
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        }
        std::vector<int> algo_index_list;
        algo_index_list.push_back(algo_index);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_results;
        hipblasStatus_t status = hipblaslt_ext::getAlgosFromIndex(ctx.handle, algo_index_list, heuristic_results);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_WARN("Failed to get algorithm from index %d for batched matrix multiplication\n", algo_index);
            return status;
        }
        if (heuristic_results.empty()) {
            GGML_LOG_WARN("No heuristic results found for batched matrix multiplication, hash: %s\n", prob.hash.c_str());
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
        hipblasLtMatmulHeuristicResult_t heuristicResult = heuristic_results[0];
        status = grouped_gemm_->initialize(heuristicResult.algo, workspace_left_);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            GGML_LOG_WARN("Failed to initialize batched matrix multiplication with heuristic result\n");
            return status;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }
};

class HipblasltBatchedGemmSingleton{
public:
    static HipblasltBatchedGemmSingleton& getInstance() {
        static HipblasltBatchedGemmSingleton instance;
        return instance;
    }

    hipblasStatus_t hipblasLtSetStream(int device_id, hipStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_id < 0 || device_id >= GGML_CUDA_MAX_DEVICES) {
            GGML_LOG_WARN("Invalid device ID: %d\n", device_id);
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        if(hipblas_lt_handles_[device_id] == nullptr) {
            hipblasLtHandle_t handle;
            if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
                GGML_LOG_WARN("Failed to create hipblasLt handle for device %d\n", device_id);
                return HIPBLAS_STATUS_ALLOC_FAILED;
            }
            hipblas_lt_handles_[device_id] = handle;
        }
        // make sure only 1 thread is running for 1 device
        streams_[device_id] = stream;
        if(workspace_ptrs_[device_id].count(stream) == 0) {
            void *workspace_ptr{nullptr};
            if(hipMalloc(&workspace_ptr, MAX_WORKSPACE_SIZE) != hipSuccess) {
                GGML_LOG_WARN("Failed to allocate workspace for stream %p\n", stream);
                return HIPBLAS_STATUS_ALLOC_FAILED;
            }
            workspace_ptrs_[device_id][stream] = workspace_ptr;
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t getContextData(int device_id, HipblasltBatchedGemmCtxData& ctx_data){
        std::lock_guard<std::mutex> lock(mutex_);
        if(device_id < 0 || device_id >= GGML_CUDA_MAX_DEVICES) {
            GGML_LOG_WARN("Invalid device ID: %d\n", device_id);
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        if(hipblas_lt_handles_[device_id] == nullptr) {
            GGML_LOG_WARN("hipblasLt handle for device %d is not initialized\n", device_id);
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        }
        ctx_data.handle = hipblas_lt_handles_[device_id];
        if (streams_[device_id] == nullptr) {
            GGML_LOG_WARN("Stream for device %d is not set\n", device_id);
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        }
        ctx_data.stream = streams_[device_id];
        if(workspace_ptrs_[device_id].count(ctx_data.stream) == 0 || workspace_ptrs_[device_id][ctx_data.stream] == nullptr) {
            GGML_LOG_WARN("Workspace pointer for stream %p is not set\n", ctx_data.stream);
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        }
        ctx_data.workspace = workspace_ptrs_[device_id][ctx_data.stream];
        ctx_data.workspace_size = MAX_WORKSPACE_SIZE;
        return HIPBLAS_STATUS_SUCCESS;
    }

    void hipblasltDestroy(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_id < 0 || device_id >= GGML_CUDA_MAX_DEVICES) {
            GGML_LOG_WARN("Invalid device ID: %d\n", device_id);
            return;
        }
        batched_gemm_map_[device_id].clear();
        for(auto& pair : workspace_ptrs_[device_id]) {
            if(pair.second != nullptr) {
                (void) hipFree(pair.second);
                pair.second = nullptr;
            }
        }
        workspace_ptrs_[device_id].clear();
        if(streams_[device_id] != nullptr) {
            streams_[device_id] = nullptr;
        }
        if(hipblas_lt_handles_[device_id] != nullptr) {
            hipblasLtDestroy(hipblas_lt_handles_[device_id]);
            hipblas_lt_handles_[device_id] = nullptr;
        }
    }

    std::shared_ptr<IHipblasltBatchedGemm> getGem(int device_id, hipStream_t stream, const std::string& problem_hash){
        if (device_id < 0 || device_id >= GGML_CUDA_MAX_DEVICES) {
            GGML_LOG_WARN("Invalid device ID: %d\n", device_id);
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto& cur_stream_batched_gemm_map = batched_gemm_map_[device_id];
        if(cur_stream_batched_gemm_map.count(stream) == 0) {
            return nullptr;
        }
        auto& gemms = cur_stream_batched_gemm_map[stream];
        if(gemms.count(problem_hash) == 0) {
            return nullptr;
        }
        return gemms[problem_hash];
    }

    void setGem(int device_id, const std::string& problem_hash, hipStream_t stream, std::shared_ptr<IHipblasltBatchedGemm> gemm) {
        if (device_id < 0 || device_id >= GGML_CUDA_MAX_DEVICES || gemm == nullptr) {
            GGML_LOG_WARN("Invalid device ID: %d\n", device_id);
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto& cur_stream_batched_gemm_map = batched_gemm_map_[device_id];
        if(cur_stream_batched_gemm_map.count(stream) == 0) {
            cur_stream_batched_gemm_map[stream] = std::unordered_map<std::string, std::shared_ptr<IHipblasltBatchedGemm>>();
        }
        cur_stream_batched_gemm_map[stream][problem_hash] = std::move(gemm);
    }

private:
    using stream_batched_gemm_map = std::unordered_map<hipStream_t, std::unordered_map<std::string, std::shared_ptr<IHipblasltBatchedGemm>>>;
    HipblasltBatchedGemmSingleton() = default;
    ~HipblasltBatchedGemmSingleton(){}
    std::mutex mutex_;
    std::array<hipblasLtHandle_t, GGML_CUDA_MAX_DEVICES> hipblas_lt_handles_ = {nullptr};
    std::array<hipStream_t, GGML_CUDA_MAX_DEVICES> streams_ = {nullptr};
    std::array<std::unordered_map<hipStream_t, void*>, GGML_CUDA_MAX_DEVICES> workspace_ptrs_ = {{}};
    std::array<stream_batched_gemm_map, GGML_CUDA_MAX_DEVICES> batched_gemm_map_ = {{}};
};

hipblasStatus_t hipblasLtGroupedGemm(cublasHandle_t handle,
                                    cublasOperation_t transA,
                                    cublasOperation_t transB,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const void*        alpha,
                                    const void*        A[],
                                    cudaDataType_t  aType,
                                    int                lda,
                                    const void*        B[],
                                    cudaDataType_t  bType,
                                    int                ldb,
                                    const void*        beta,
                                    void*              C[],
                                    cudaDataType_t  cType,
                                    int                ldc,
                                    int                batchCount,
                                    cublasComputeType_t  computeType,
                                    hipblasGemmAlgo_t  algo){

    if(handle == nullptr || A == nullptr || B == nullptr || C == nullptr || batchCount <= 0 || algo != HIPBLAS_GEMM_DEFAULT){
        GGML_LOG_WARN("Invalid arguments to hipblasLtGroupedGemm\n");
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    GemmProblemDesc prob{
        .op_a = transA,
        .op_b = transB,
#if HIP_VERSION >= 60500000
        .type_a = aType,
        .type_b = bType,
        .type_c = cType,
        .type_compute = computeType,
#else
        .type_a = getHipblasLtDatatype(aType),
        .type_b = getHipblasLtDatatype(bType),
        .type_c = getHipblasLtDatatype(cType),
        .type_compute = getHipblasLtComputeType(computeType),
#endif
        .m = static_cast<uint32_t>(m),
        .n = static_cast<uint32_t>(n),
        .k = static_cast<uint32_t>(k),
        .lda = static_cast<uint32_t>(lda),
        .ldb = static_cast<uint32_t>(ldb),
        .ldc = static_cast<uint32_t>(ldc),
        .stride_a = static_cast<uint32_t>(lda * m),
        .stride_b = static_cast<uint32_t>(ldb * n),
        .stride_c = static_cast<uint32_t>(ldc * n),
        .batch_count = static_cast<uint32_t>(batchCount),
        .alpha = (void*) alpha,
        .beta = (void*) beta
    };
    prob.processHash();
    auto& gemm_singleton = HipblasltBatchedGemmSingleton::getInstance();    

    int device_id = -1;
    if(hipGetDevice(&device_id) != hipSuccess) {
        GGML_LOG_WARN("Failed to get current device\n");
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    HipblasltBatchedGemmCtxData prob_ctx_data;
    hipblasStatus_t status = gemm_singleton.getContextData(device_id, prob_ctx_data);
    if(status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("Failed to get context data for device %d: %d\n", device_id, status);
        return status;
    }

    auto gemm_ptr = gemm_singleton.getGem(device_id, prob_ctx_data.stream, prob.hash);
    if(gemm_ptr == nullptr) { 
        int hipblaslt_env_var = getHipblasltBatchedGemmEnvVal();
        std::shared_ptr<IHipblasltBatchedGemm> new_gemm;
        if(hipblaslt_env_var == 1) {
            new_gemm = std::make_shared<CHipblasltBatchedGemmAllAlgos>();
        }else if(hipblaslt_env_var == 2) {
            new_gemm = std::make_shared<CHipblasltBatchedGemmBenchmark>();
        }else if(hipblaslt_env_var == 3){
            if(!OfflineBenchResultsReader::getInstance().read()){
                GGML_LOG_WARN("Cannot locate HIPBLASLT_GROUPED_GEMM_FILE\n");
                return HIPBLAS_STATUS_NOT_INITIALIZED;
            }
            new_gemm = std::make_shared<CHipblasltBatchedGemmSolution>();
        }else{
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
        status = new_gemm->init(prob, prob_ctx_data);
        if(status != HIPBLAS_STATUS_SUCCESS) {
            //GGML_LOG_DEBUG("Failed to initialize batched gemm for device %d: %d\n", device_id, status);
            return status;
        }
        gemm_singleton.setGem(device_id, prob.hash, prob_ctx_data.stream, new_gemm);
        gemm_ptr = new_gemm;
    }
    status = gemm_ptr->runGemm(prob, A, B, C, prob_ctx_data.stream);
    if(status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_DEBUG("Failed to run batched gemm for device %d: %d\n", device_id, status);
        return status;
    }
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasGemmBatchedEx(cublasHandle_t handle,
                                    cublasOperation_t transA,
                                    cublasOperation_t transB,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const void*        alpha,
                                    const void*        A[],
                                    cudaDataType_t  aType,
                                    int                lda,
                                    const void*        B[],
                                    cudaDataType_t  bType,
                                    int                ldb,
                                    const void*        beta,
                                    void*              C[],
                                    cudaDataType_t  cType,
                                    int                ldc,
                                    int                batchCount,
                                    cublasComputeType_t  computeType,
                                    hipblasGemmAlgo_t  algo){
    int cur_dev_id = ggml_cuda_get_device();
    const int cur_dev_cc = ggml_cuda_info().devices[cur_dev_id].cc;
    if(getHipblasltBatchedGemmEnvVal() != 0 && GGML_CUDA_CC_IS_CDNA3(cur_dev_cc)){
        hipblasStatus_t status = hipblasLtGroupedGemm(handle,
                                    transA,
                                    transB,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    A,
                                    aType,
                                    lda,
                                    B,
                                    bType,
                                    ldb,
                                    beta,
                                    C,
                                    cType,
                                    ldc,
                                    batchCount,
                                    computeType,
                                    algo);
        if(status == HIPBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    CUBLAS_CHECK(::cublasGemmBatchedEx(handle,
                        transA,
                        transB,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        aType,
                        lda,
                        B,
                        bType,
                        ldb,
                        beta,
                        C,
                        cType,
                        ldc,
                        batchCount,
                        computeType,
                        algo));
    return HIPBLAS_STATUS_SUCCESS;
}

} // namespace ggml::vendors
