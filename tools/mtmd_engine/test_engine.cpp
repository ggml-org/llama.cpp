#include "mtmd-engine.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <fcntl.h>
#include <Windows.h>
#include <io.h>
#include <chrono>
#include <thread>

std::vector<unsigned char> load_img_file(const std::string path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) {
        return {};
    }
    fin.seekg(0, std::ios::end);
    auto s = fin.tellg();
    std::vector<unsigned char> res;
    res.resize(s);
    fin.seekg(0, std::ios::beg);
    fin.read((char*)res.data(),s);
    fin.close();
    return res;
}

void print_time(const std::chrono::system_clock::time_point& t1, const std::chrono::system_clock::time_point& t2) {
    auto count = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout<< count <<"ms" << std::endl;
}

void log_call(int level, const std::string& msg) {
    std::cout<<"log(" << level << "): " << msg << std::endl;
}

void engine2() {
    //test_mem_file();
    llama_engine::InferInput input;
    //input.img_bufs = load_img_file("e:/wafer.png");
    input.img_bufs = load_img_file("G:/model_test/xin_shang_wei_zhuang/20201016065520_S[25]_waferid.jpg");

    llama_engine::InferResult res;

    llama_engine::EngineConfigParam param;
    //param.gpu_devices.clear();
    param.gpu_layer_count         = 99;
    param.log_call_back           = log_call;
    param.max_predict_token_count = 32;
    param.fit_param               = false;

    llama_engine::InferEngine engine;
    auto                      status = engine.set_config_param(param);
    auto                      start  = std::chrono::system_clock::now();
    //auto model_buf = load_img_file("d:/qwen/official_gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf");
    ////auto model_buf = load_img_file("d:/qwen/o_gen/mmproj-Qwen3-VL-2B-Instruct-F16.gguf");
    ////auto mmproj_buf = load_img_file("d:/qwen/official_gguf/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
    //auto mmproj_buf = load_img_file("d:/qwen/o_gen/mmproj-Qwen3-VL-2B-Instruct-F16.gguf");

    status = engine.load_model_from_file(
        //"d:/qwen/Qwen3VL-2B-Instruct-F16.gguf",
        //"d:/qwen/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
        //"d:/qwen/official_gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        //"d:/qwen/official_gguf/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
        "g:/model_test/qwen/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "g:/model_test/qwen/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
    //status = engine.load_model_from_buffer((char*)model_buf.data(), model_buf.size(), (char*)mmproj_buf.data(), mmproj_buf.size());
    auto start1 = std::chrono::system_clock::now();
    status      = engine.infer(input, res);
    auto end    = std::chrono::system_clock::now();
    std::cout << "============= result =======================" << std::endl;
    std::cout << res.result << std::endl;
    std::cout << "every character results:/nc, prob/n";
    for (const auto & ch : res.details) {
        std::cout << ch.character << ": " << ch.prob << std::endl;
    }
    while (true) {
        auto end1 = std::chrono::system_clock::now();
        status    = engine.infer(input, res);
        auto end2 = std::chrono::system_clock::now();
        print_time(end1, end2);
        std::cout << "result from thread 2: " << res.result << std::endl;
    }
    auto end1 = std::chrono::system_clock::now();
    status    = engine.infer(input, res);
    auto end2 = std::chrono::system_clock::now();
    print_time(start, start1);
    print_time(start1, end);
    print_time(end, end1);
    print_time(end1, end2);

    std::cout << "============= result =======================" << std::endl;
    std::cout << res.result << std::endl;
    std::cout << "every character results:/nc, prob/n";
    for (const auto & ch : res.details) {
        std::cout << ch.character << ": " << ch.prob << std::endl;
    }
}


int main() {
    std::thread task2(engine2);


    //test_mem_file();
    llama_engine::InferInput input;
    input.img_bufs = load_img_file("e:/wafer.png");
    //input.img_bufs = load_img_file("G:/model_test/xin_shang_wei_zhuang/20201016065520_S[25]_waferid.jpg");

    llama_engine::InferResult res;

    llama_engine::EngineConfigParam param;
    //param.gpu_devices.clear();
    param.gpu_layer_count = 99;
    param.log_call_back = log_call;
    param.max_predict_token_count=32;
    param.fit_param = false;

    llama_engine::InferEngine engine;
    auto status = engine.set_config_param(param);
    auto start = std::chrono::system_clock::now();
    //auto model_buf = load_img_file("d:/qwen/official_gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf");
    ////auto model_buf = load_img_file("d:/qwen/o_gen/mmproj-Qwen3-VL-2B-Instruct-F16.gguf");
    ////auto mmproj_buf = load_img_file("d:/qwen/official_gguf/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
    //auto mmproj_buf = load_img_file("d:/qwen/o_gen/mmproj-Qwen3-VL-2B-Instruct-F16.gguf");


    status = engine.load_model_from_file(
        //"d:/qwen/Qwen3VL-2B-Instruct-F16.gguf",
        //"d:/qwen/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
        //"d:/qwen/official_gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        //"d:/qwen/official_gguf/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
        "g:/model_test/qwen/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "g:/model_test/qwen/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    );
    //status = engine.load_model_from_buffer((char*)model_buf.data(), model_buf.size(), (char*)mmproj_buf.data(), mmproj_buf.size());
    auto start1 = std::chrono::system_clock::now();
    status = engine.infer(input, res);
    auto end    = std::chrono::system_clock::now();
    std::cout << "============= result =======================" << std::endl;
    std::cout << res.result << std::endl;
    std::cout << "every character results:/nc, prob/n";
    for (const auto& ch : res.details) {
        std::cout << ch.character << ": " << ch.prob << std::endl;
    }
    while (true) {
        auto end1 = std::chrono::system_clock::now();
        status    = engine.infer(input, res);
        auto end2 = std::chrono::system_clock::now();
        print_time(end1, end2);
        std::cout << "result: " << res.result << std::endl;
    }
    auto end1 = std::chrono::system_clock::now();
    status = engine.infer(input, res);
    auto end2 = std::chrono::system_clock::now();
    print_time(start, start1);
    print_time(start1, end);
    print_time(end, end1);
    print_time(end1, end2);


    std::cout<< "============= result =======================" << std::endl;
    std::cout<< res.result << std::endl;
    std::cout<<"every character results:/nc, prob/n";
    for (const auto& ch : res.details) {
        std::cout<< ch.character <<": "<< ch.prob<<std::endl;
    }

    task2.join();
}
