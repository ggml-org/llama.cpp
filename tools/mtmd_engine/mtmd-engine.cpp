# include "mtmd-engine.hpp"
#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
//#include "ggml-backend.h"

#include <vector>
#include <limits.h>
#include <cinttypes>
#include <utility>
#include <exception>
#include <mutex>
#include <iostream>
#include <thread>
#include <chrono>

/*限定模型只使用cpu的核心参数：
1. -dev none: 禁止VLM模型使用gpu
2. --no-mmproj-offload: 禁止视觉投影模型占用gpu


*/
//int external_log_to_internal(int level) {
//    switch (level) {
//    case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;
//    case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_INFO;
//    case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;
//    case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;
//    case 5: return -1;
//    default:
//        return LOG_LEVEL_OUTPUT;
//    }
//}

class Timer {
public:
    Timer()=default;

    void enable_print(bool _enable) {enable_print_ = _enable;}

    void start() {
        last = std::chrono::system_clock::now();
    }

    void print_time(const std::string& msg) {
        if(!enable_print_) return;
        auto cur = std::chrono::system_clock::now();
        std::cout << msg << "(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(cur - last).count() << std::endl;
        last = std::chrono::system_clock::now();
    }
  private :
    std::chrono::system_clock::time_point last = std::chrono::system_clock::now();
    bool enable_print_{false};
};



enum LogLevel : int {
    kLogTrace = 0,
    kLogDebug,
    kLogInfo,
    kLogWarn,
    kLogError,
    kLogCritical,
    kLogOff
};

void llama_log_callback(enum ggml_log_level level, const char* text, void* user_data){
    // ignore all log.
}

namespace llama_engine{

struct InferEngine::Impl{
    Timer timer;

    EngineConfigParam config_param;
    std::mutex infer_mutex;
    bool use_gpu = true;
    const std::string prompt = "Read the characters in the image.";

    mtmd::context_ptr ctx_vision{nullptr};
    common_init_result_ptr llama_init{nullptr};

    llama_model* model{nullptr};
    llama_context* lctx{nullptr};
    const llama_vocab* vocab{nullptr};
    common_sampler* smpl{nullptr};
    llama_batch         batch{};
    int                 n_batch{2048};

    mtmd::bitmaps bitmaps;

    // chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;
    // TODO: support for --system-prompt with /clear command

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    //int n_threads = 5;
    llama_pos n_past = 0;

    Impl() {
        // disable llama-cpp log
        llama_log_set(llama_log_callback, nullptr);
        mtmd_helper_log_set(llama_log_callback, nullptr);
        common_log_pause(common_log_main());
    }

    void reset() {
        model = nullptr;
        lctx = nullptr;
        vocab = nullptr;
        smpl = nullptr;
        llama_batch_free(batch);
        common_sampler_free(smpl);
        ctx_vision.reset();
        llama_init.reset();
    }

    Status write_log(int level, const char* file, int line, const char* text) {
        std::string log_level;
        switch (level) {
            case LogLevel::kLogCritical: log_level = {"Critical"}; break;
            case LogLevel::kLogError: log_level = {"Error"}; break;
            case LogLevel::kLogDebug: log_level = {"Debug"}; break;
            case LogLevel::kLogInfo: log_level = {"Info"}; break;
            case LogLevel::kLogOff: log_level = {""}; break;
            case LogLevel::kLogTrace: log_level = {"Trace"}; break;
            case LogLevel::kLogWarn: log_level = {"Warn"}; break;
            default:
                log_level = {""};
        };
        if (config_param.log_call_back) {
            auto msg = string_format("%s:[%s][%d]: %s", log_level.c_str(), file, line, text);
            config_param.log_call_back(LogLevel::kLogWarn, msg);
            return {-level, msg};
        }
        return {};
    }

    Status config_param_to_llama_param(common_params& params) {
        auto ctx_arg = common_params_parser_init(params, LLAMA_EXAMPLE_MTMD,nullptr);
        const auto params_org = ctx_arg.params;

        ctx_arg.params.system_prompt.clear();

        // model
        ctx_arg.params.fit_params = config_param.fit_param;
        ctx_arg.params.n_gpu_layers = config_param.gpu_layer_count;
        ctx_arg.params.n_predict = config_param.max_predict_token_count;
        ctx_arg.params.n_ctx = config_param.n_ctx;
        ctx_arg.params.mmproj_use_gpu = true;

        // sampler
        ctx_arg.params.sampling.temp = 0;
        ctx_arg.params.sampling.penalty_repeat = 1.0;

        // devices
        ctx_arg.params.main_gpu = config_param.main_gpu;
        ctx_arg.params.devices.clear();
        std::vector<ggml_backend_dev_t> devices;
        auto device_ids = config_param.gpu_devices;

        if (device_ids.empty()) {
            // 为空时使用cpu
            auto* dev = ggml_backend_dev_by_name("CPU");
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                return write_log(LogLevel::kLogError, __FILE__, __LINE__, "Config CPU device failed.");
            }
            devices.push_back(dev);
            use_gpu = false;
        } else {
            for (const auto i : device_ids) {
                std::string dev_name = "CUDA" + std::to_string(i);
                auto* dev = ggml_backend_dev_by_name(dev_name.c_str());
                if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    write_log(LogLevel::kLogWarn, __FILE__, __LINE__,
                        string_format("Config cuda device %d is not valid, ignore this device.", i).c_str());
                    continue;
                }
                devices.push_back(dev);
            }
            use_gpu = true;
        }

        if (devices.empty()) {
            auto msg = "Setted gpu devices all invalid.";
            return write_log(LogLevel::kLogError, __FILE__, __LINE__, msg);
        }
        // 因为后续遍历时使用的指针列表，所以需要添加一个nullptr哨兵
        devices.push_back(nullptr);
        ctx_arg.params.devices = devices;

        return {};
    }

    Status load_model_helper(common_params& llama_param) {
        llama_init = common_init_from_params(llama_param);
        model = llama_init->model();
        lctx = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, llama_param.sampling);
        //n_threads = llama_param.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1); // batch for next token generation
        n_batch = llama_param.n_batch;

        if (!model || !lctx) {
            return write_log(LogLevel::kLogCritical, __FILE__, __LINE__, "Fail to load model.");
        }

        if (!llama_model_chat_template(model, nullptr) && llama_param.chat_template.empty()) {
            return write_log(LogLevel::kLogCritical, __FILE__, __LINE__, "Model file is in wrong format, please check whether the model path is correct.");
        }

        tmpls = common_chat_templates_init(model, llama_param.chat_template);
        use_jinja = llama_param.use_jinja;
        chat_history.clear();

        auto status = init_vision_context(llama_param);

        return status;
    }

    Status load_model_from_file(const std::string& vlm_model_path, const std::string& mmproj_model_path) {
        reset();
        common_params llama_param;
        auto status = config_param_to_llama_param(llama_param);
        if (!status) {
            return status;
        }
        llama_param.model.path = vlm_model_path;
        llama_param.mmproj.path = mmproj_model_path;
        llama_param.use_mmap = true;

        return load_model_helper(llama_param);
    }

    Status load_model_from_buffer(const char* vlm_model_buf,
        size_t vlm_model_buf_size,
        const char* mmproj_model_buf,
        size_t mmproj_model_buf_size) {
        reset();
        common_params llama_param;
        auto status = config_param_to_llama_param(llama_param);
        if (!status) {
            return status;
        }
        llama_param.model.model_buf = const_cast<char*>(vlm_model_buf);
        llama_param.model.model_buf_size = vlm_model_buf_size;
        llama_param.mmproj.model_buf = const_cast<char*>(mmproj_model_buf);
        llama_param.mmproj.model_buf_size = mmproj_model_buf_size;
        llama_param.use_mmap = false;

        return load_model_helper(llama_param);
    }

    Status init_vision_context(common_params& params) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = use_gpu;
        mparams.n_threads = std::max(1, config_param.max_cpu_threads);
        mparams.print_timings = true;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;

        const char* clip_path = params.mmproj.path.c_str();
        ctx_vision.reset(mtmd_init_from_file(clip_path, params.mmproj.model_buf, params.mmproj.model_buf_size,  model, mparams));
        if (!ctx_vision.get()) {
            return write_log(LogLevel::kLogCritical, __FILE__, __LINE__, "Failed to load vision model.");
        }
        return {};
    }

    Status release_gpu() {
        // 改为外部提供buffer
        reset();
        return {};
    }

    Status to_gpu() {
        return {-LogLevel::kLogCritical, "Not implemented, please reload the model."};
    }

    void clear_ctx_history() {
        n_past = 0;
        chat_history.clear();
        llama_memory_clear(llama_get_memory(lctx), true);
        write_log(LogLevel::kLogInfo, __FILE__, __LINE__, "Clear status of model.");
    }

    Status load_image(const std::vector<unsigned char>& buf) {
        bitmaps.entries.clear();
        auto bmp = mtmd_helper_bitmap_init_from_buf(ctx_vision.get(), buf.data(), buf.size());
        if (!bmp) {
            return {-LogLevel::kLogError, "Failed to decode image buffer."};
        }
        bitmaps.entries.push_back(std::move(bmp));
        return {};
    }

    std::string chat_add_and_format(common_chat_msg& new_msg) {
        auto formatted = common_chat_format_single(tmpls.get(), chat_history,
            new_msg, new_msg.role == "user",use_jinja);
        chat_history.push_back(new_msg);
        return formatted;
    }

    Status eval_message(common_chat_msg& msg) {
        timer.start();
        llama_batch_free(batch);
        batch = llama_batch_init(n_batch, 0, 1); // batch for next token generation
        timer.print_time("init_batch");
        bool add_bos = chat_history.empty();
        auto formatted_chat = chat_add_and_format(msg);
        //LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());
        timer.print_time("chat and format");
        mtmd_input_text text;
        text.text = formatted_chat.c_str();
        text.add_special = add_bos;
        text.parse_special = true;

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        timer.print_time("mtmd input chunks init");
        auto bitmaps_c_ptr = bitmaps.c_ptr();
        int32_t res = mtmd_tokenize(ctx_vision.get(),
            chunks.ptr.get(), // output
            &text, // text
            bitmaps_c_ptr.data(),
            bitmaps_c_ptr.size());
        if (res != 0) {
            return write_log(LogLevel::kLogError, __FILE__, __LINE__, "Unable to Tokenize prompt");
        }

        bitmaps.entries.clear();
        timer.print_time("mtmd_tokenize");
        llama_pos new_n_past;
        if (mtmd_helper_eval_chunks(ctx_vision.get(),
                                    lctx, // lctx
                                    chunks.ptr.get(), // chunks
                                    n_past, // n_past
                                    0, // seq_id
                                    n_batch, // n_batch
                                    true, // logits_last
                                    &new_n_past)) {
            return { -LogLevel::kLogError, "Fail to eval input." };
        }
        timer.print_time("mtmd_helper_eval_chunks");
        n_past = new_n_past;

        return {};
    }

    Status generate_response(std::string& text, std::vector<float>& probs) {
        llama_tokens generated_tokens;
        //// 因为一个token可能对应一个单词（多个字符），这里将这类单词中的所有字符按同一概率值设置
        //std::vector<llama_tokens> each_genetared_token;
        probs.clear();
        bool stop_normally=true;
        std::vector<float> tmp_probs;
        int max_token = config_param.max_predict_token_count > 0 ? config_param.max_predict_token_count : INT_MAX;
        for (int i = 0; i < max_token; i++) {
            if (i == max_token - 1) {
                write_log(kLogWarn, __FILE__, __LINE__, "Generated count reached the max limition.");
                stop_normally = false;
                break;
            }

            auto [token_id, prob]= common_sampler_sample_with_prob(smpl, lctx, -1);
            generated_tokens.push_back(token_id);
            //each_genetared_token.emplace_back().push_back(token_id);
            tmp_probs.push_back(prob);

            common_sampler_accept(smpl, token_id, true);

            if (llama_vocab_is_eog(vocab, token_id)) {
                write_log(kLogDebug, __FILE__, __LINE__, "Generated with eos.");
                break; // end of generation
            }

            // eval the token
            common_batch_clear(batch);
            common_batch_add(batch, token_id, n_past++, { 0 }, true);
            if (llama_decode(lctx, batch)) {
                return write_log(kLogError, __FILE__, __LINE__, "Failed to decode token.");
            }
        }
        text.clear();
        text.reserve(32);
        if (stop_normally && !generated_tokens.empty()) {
            generated_tokens.pop_back();
        }
        for (size_t i=0; i < generated_tokens.size(); i++) {
            auto tmp_txt = common_token_to_piece(lctx, generated_tokens[i]);
            for (const auto& c : tmp_txt) {
                if(!config_param.keep_space && c == ' ') continue;
                text.push_back(c);
                probs.push_back(tmp_probs[i]);
            }
        }
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = text;
        chat_history.push_back(std::move(msg));
        return {};
    }

    Status infer(const InferInput& input, InferResult& result) {
        timer.start();
        std::unique_lock<std::mutex> infer_lock(infer_mutex);
        result.details.clear();
        clear_ctx_history();
        timer.print_time("clear ctx");
        auto status = load_image(input.img_bufs);
        if (!status) {
            return status;
        }
        timer.print_time("load image");
        std::string infer_prompt = mtmd_default_marker() + prompt;
        timer.print_time("infer_prompt");
        common_chat_msg infer_msg;
        infer_msg.role = "user";
        infer_msg.content = infer_prompt;

        status = eval_message(infer_msg);
        if (!status) {
            return status;
        }
        timer.print_time("eval message");
        std::string text;
        std::vector<float> probs;
        status = generate_response(text, probs);
        timer.print_time("gen response");
        if (!status ) {
            return status;
        }
        result.result = text;
        result.details.reserve(text.size());
        for (size_t i = 0; i < text.size(); i++) {
            InferResult::TokenInfo info;
            info.character = text[i];
            info.prob = probs[i];
            result.details.emplace_back(std::move(info));
        }
        return status;
    }


    Status infer_batch(const std::vector<InferInput>& inputs, std::vector<InferResult>& results) {

        return {};
    }
};

InferEngine::InferEngine(): impl_ (new Impl()){

}
InferEngine::~InferEngine(){
    delete impl_;
    impl_ = nullptr;
}

InferEngine::InferEngine(InferEngine&& rhs)noexcept:impl_(rhs.impl_){

}

InferEngine& InferEngine::operator=(InferEngine&& rhs) noexcept{
    if(this != &rhs){
        std::swap(impl_, rhs.impl_);
    }
    return *this;
}

Status InferEngine::set_config_param(const EngineConfigParam& config_param) {
    impl_->config_param = config_param;

    return {};
}

Status InferEngine::load_model_from_file(const std::string &vlm_model_path, const std::string& mmproj_model_path){

    return impl_->load_model_from_file(vlm_model_path, mmproj_model_path);
}

Status InferEngine::load_model_from_buffer(const char* vlm_model_buf, size_t vlm_model_buf_size,
                            const char* mmproj_model_buf, size_t mmproj_model_buf_size){
    return impl_->load_model_from_buffer(vlm_model_buf, vlm_model_buf_size, mmproj_model_buf, mmproj_model_buf_size);
}

Status InferEngine::release_gpu(){

    return impl_->release_gpu();
}

Status InferEngine::to_gpu(){
    return impl_->to_gpu();
}

Status InferEngine::infer(const InferInput& input, InferResult& result){
    //impl_->n_batch = 1;
    return impl_->infer(input, result);
}


Status InferEngine::infer_batch(const std::vector<InferInput>& inputs, std::vector<InferResult>& results){
    //impl_->n_batch = inputs.size();
    return impl_->infer_batch(inputs, results);
}
}
