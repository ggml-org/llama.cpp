/**
 * @file llama-cpp-engine.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-12-21
 *
 * @copyright Copyright (c) 2025
 *
 */
//#include "llama-cpp.h"
#include <string>
#include <vector>


#ifdef LLAMA_BUILD
#define ENGINE_API __declspec(dllexport)
#else
#define ENGINE_API __declspec(dllimport)
#endif // !LLAMA_BUILD



namespace llama_engine{

struct Status{
    int code;
    std::string message;
    Status(int c=0, const std::string& _msg="") : code(c), message(_msg) {}
    operator bool() const{
        return code == 0;
    }
};

struct InferInput{
    // img data(original data like jpg binary data file), expect height of img size: 48
    std::vector<unsigned char> img_bufs;
};

struct InferResult{
    std::string result;
    struct TokenInfo{
        char character;
        float prob;
    };
    std::vector<TokenInfo> details;
};

//level {
//    GGML_LOG_LEVEL_NONE = 0,
//    GGML_LOG_LEVEL_DEBUG = 1,
//    GGML_LOG_LEVEL_INFO = 2,
//    GGML_LOG_LEVEL_WARN = 3,
//    GGML_LOG_LEVEL_ERROR = 4,
//    GGML_LOG_LEVEL_CONT = 5, // continue previous log
//};
typedef void (*engine_log_callback)(int level, const std::string& text);

struct EngineConfigParam {
    std::vector<size_t> gpu_devices{0};
    int32_t main_gpu = 0;
    int32_t gpu_layer_count = 999;
    bool fit_param = true;
    int32_t max_predict_token_count = -1; // -1 == nolimit; max new token to generate
    bool need_prob{true};
    bool keep_space = false;
    //int log_level;
    //enum ggml_log_level {
    //    GGML_LOG_LEVEL_NONE = 0,
    //    GGML_LOG_LEVEL_DEBUG = 1,
    //    GGML_LOG_LEVEL_INFO = 2,
    //    GGML_LOG_LEVEL_WARN = 3,
    //    GGML_LOG_LEVEL_ERROR = 4,
    //    GGML_LOG_LEVEL_CONT = 5, // continue previous log
    //};
    //log close
    engine_log_callback log_call_back{nullptr};
};



class ENGINE_API InferEngine{
public:
    InferEngine();
    virtual ~InferEngine();
    InferEngine(const InferEngine& rhs) = delete;
    InferEngine(InferEngine&& rhs)noexcept;
    InferEngine& operator=(const InferEngine&&rhs) = delete;
    InferEngine& operator=(InferEngine&& rhs) noexcept;

    Status set_config_param(const EngineConfigParam& config_param);

    Status load_model_from_file(const std::string &vlm_model_path,
                                const std::string& mmproj_model_path);
    Status load_model_from_buffer(const char* vlm_model_buf, size_t vlm_model_buf_size,
                                const char* mmproj_model_buf, size_t mmproj_model_buf_size);
    /**
     * @brief try to release model from gpu
     *
     * @return Status
     */
    Status release_gpu();

    /**
     * @brief try to load model to gpu, may be failed, if failed, please call load_model_from_file or load_model_from_buffer again
     *
     * @return Status
     */
    Status to_gpu();


    Status infer(const InferInput& input, InferResult& result);
    Status infer_batch(const std::vector<InferInput>& inputs, std::vector<InferResult>& results);

private:
    struct Impl;
    Impl *impl_;
};


}
