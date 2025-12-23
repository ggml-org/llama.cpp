#include "mtmd-engine.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <fcntl.h>
#include <Windows.h>
#include <io.h>
#include <chrono>

//
//FILE* create_memory_file(const void *data, size_t size) {
//    // 创建内存映射文件
//    HANDLE hMap = CreateFileMapping(
//        INVALID_HANDLE_VALUE,
//        NULL,
//        PAGE_READWRITE,
//        0,
//        size,
//        NULL
//    );
//
//    if (!hMap) return NULL;
//
//    // 映射到内存
//    void* pData = MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, size);
//    if (!pData) {
//        CloseHandle(hMap);
//        return NULL;
//    }
//
//    // 复制数据
//    if (data) {
//        memcpy(pData, data, size);
//    }
//
//    // 创建文件描述符
//    int fd = _open_osfhandle((intptr_t)hMap, _O_RDWR | _O_CREAT);
//    if (fd == -1) {
//        UnmapViewOfFile(pData);
//        CloseHandle(hMap);
//        return NULL;
//    }
//
//    // 创建FILE*指针
//    FILE* fp = _fdopen(fd, "rb+");
//    if (!fp) {
//        _close(fd);
//        UnmapViewOfFile(pData);
//        CloseHandle(hMap);
//    }
//
//    return fp;
//}

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

void test_mem_file() {
    std::vector<unsigned char> data = load_img_file("e:/wafer.jpg");
    auto mmf = MemoryMappedFile::CreateWithData(data.data(), data.size());
    FILE* fp = mmf->GetFilePointer();
    std::vector<unsigned char> new_data;
    new_data.resize(data.size());
    fread(new_data.data(), 1, data.size(), fp);
    std::ofstream new_f("e:/new_wafer.jpg", std::ios::binary);
    new_f.write((char*)new_data.data(), new_data.size());
}




int main() {
    //test_mem_file();
    llama_engine::InferInput input;
    input.img_bufs = load_img_file("e:/wafer.jpg");

    llama_engine::InferResult res;

    llama_engine::EngineConfigParam param;
    param.gpu_devices.clear();
    param.log_call_back = log_call;
    //param.fit_param = false;

    llama_engine::InferEngine engine;
    auto status = engine.set_config_param(param);
    auto start = std::chrono::system_clock::now();
    auto model_buf = load_img_file("d:/qwen/Qwen3VL-2B-Instruct-Q4_K_M.gguf");
    auto mmproj_buf = load_img_file("d:/qwen/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");


    //status = engine.load_model_from_file(
    //    //"d:/qwen/Qwen3VL-2B-Instruct-F16.gguf",
    //    //"d:/qwen/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    //    "d:/qwen/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
    //    "d:/qwen/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    //);
    status = engine.load_model_from_buffer((char*)model_buf.data(), model_buf.size(), (char*)mmproj_buf.data(), mmproj_buf.size());
    auto start1 = std::chrono::system_clock::now();
    status = engine.infer(input, res);
    std::cout << "============= result =======================" << std::endl;
    std::cout << res.result << std::endl;
    std::cout << "every character results:\nc, prob\n";
    for (const auto& ch : res.details) {
        std::cout << ch.character << ": " << ch.prob << std::endl;
    }
    auto end = std::chrono::system_clock::now();
    status = engine.infer(input, res);
    auto end2 = std::chrono::system_clock::now();
    print_time(start, start1);
    print_time(start1, end);
    print_time(end, end2);


    std::cout<< "============= result =======================" << std::endl;
    std::cout<< res.result << std::endl;
    std::cout<<"every character results:\nc, prob\n";
    for (const auto& ch : res.details) {
        std::cout<< ch.character <<": "<< ch.prob<<std::endl;
    }




}
