#include "mtmd-ios.h"
#include <iostream>
#include <string>
#include <cstdlib>

void example_multiple_images_progressive() {
    mtmd_ios_params params = mtmd_ios_params_default();
    params.model_path = "/Users/tianchi/code/project/4o/3b/MiniCPM-4v-3b/model/ggml-model-Q4_0.gguf";
    params.mmproj_path = "/Users/tianchi/code/project/4o/3b/MiniCPM-4v-3b/mmproj-model-f16.gguf";
    params.n_predict = 100;  // 增加生成长度
    params.temperature = 0.6f;
    
    mtmd_ios_context* ctx = mtmd_ios_init(&params);
    if (!ctx) {
        std::cerr << "Failed to initialize context\n";
        return;
    }
    
    std::cout << "=== 多轮多模态对话示例 ===\n";
    std::cout << "命令说明：\n";
    std::cout << "  /image <路径>  - 添加图片\n";
    std::cout << "  /text <内容>   - 添加文本\n";
    std::cout << "  /generate     - 生成响应\n";
    std::cout << "  /quit         - 退出\n";
    std::cout << "=============================\n\n";
    
    std::string input;
    bool has_content = false;  // 跟踪是否有内容可以生成
    
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, input);
        
        if (input.empty()) {
            continue;
        }
        
        if (input == "/quit") {
            break;
        }
        
        if (input == "/generate") {
            if (!has_content) {
                std::cout << "请先添加图片或文本内容\n";
                continue;
            }
            
            std::cout << "Assistant: ";
            int token_count = 0;
            while (true) {
                mtmd_ios_token result = mtmd_ios_loop(ctx);
                
                if (result.is_end) {
                    std::cout << "\n[生成完成 - " << token_count << " tokens]\n\n";
                    break;
                }
                
                if (result.token) {
                    std::cout << result.token;
                    std::cout.flush();
                    mtmd_ios_string_free(result.token);
                    token_count++;
                }
            }  
            
            has_content = false;  // 重置内容标志
            continue;
        }
        
        if (input.find("/image ") == 0) {
            std::string image_path = input.substr(7);
            if (image_path.empty()) {
                std::cout << "请提供图片路径\n";
                continue;
            }
            
            std::cout << "正在加载图片: " << image_path << "\n";
            if (mtmd_ios_prefill_image(ctx, image_path.c_str()) != 0) {
                std::cerr << "Failed to load image: " << mtmd_ios_get_last_error(ctx) << "\n";
            } else {
                std::cout << "图片加载成功\n";
                has_content = true;
            }
            continue;
        }
        
        if (input.find("/text ") == 0) {
            std::string text = input.substr(6);
            if (text.empty()) {
                std::cout << "请提供文本内容\n";
                continue;
            }
            
            std::cout << "正在添加文本: " << text << "\n";
            if (mtmd_ios_prefill_text(ctx, text.c_str(), "user") != 0) {
                std::cerr << "Failed to add text: " << mtmd_ios_get_last_error(ctx) << "\n";
            } else {
                std::cout << "文本添加成功\n";
                has_content = true;
            }
            continue;
        }
        
        // 如果不是命令，当作文本处理
        std::cout << "正在添加文本: " << input << "\n";
        if (mtmd_ios_prefill_text(ctx, input.c_str(), "user") != 0) {
            std::cerr << "Failed to add text: " << mtmd_ios_get_last_error(ctx) << "\n";
        } else {
            std::cout << "文本添加成功\n";
            has_content = true;
        }
    }
    
    std::cout << "对话结束\n";
    mtmd_ios_free(ctx);
}

int main() {
    example_multiple_images_progressive(); 
    return 0;
} 