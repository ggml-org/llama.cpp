#include <cmath>
#include <vector>
#include <cstdio>

int main(int argc, char ** argv) {
    // ... 初始化部分保持不变 ...

    // 运行自定义的flash-decoding实现
    struct ggml_tensor * out_custom = ggml_flash_attn_custom(ctx, q, k, v, true, false);
    ggml_build_forward_expand(gf, out_custom);
    ggml_graph_compute(ctx, gf);

    // 保存自定义op结果
    std::vector<float> custom_res(ggml_nelements(out_custom));
    ggml_backend_tensor_get(out_custom, custom_res.data(), 0, ggml_nbytes(out_custom));

    // 运行标准flash-attn
    struct ggml_tensor * out_standard = ggml_flash_attn(ctx, q, k, v, true, false);
    ggml_build_forward_expand(gf, out_standard);
    ggml_graph_compute(ctx, gf);

    // 保存标准结果
    std::vector<float> standard_res(ggml_nelements(out_standard));
    ggml_backend_tensor_get(out_standard, standard_res.data(), 0, ggml_nbytes(out_standard));

    // 结果对比
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int count = 0;
    for (size_t i = 0; i < standard_res.size(); ++i) {
        float diff = fabs(standard_res[i] - custom_res[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
        count++;
        
        // 打印前10个元素的对比
        if (i < 10) {
            printf("Element %zu: std=%.6f custom=%.6f diff=%.6f\n", 
                   i, standard_res[i], custom_res[i], diff);
        }
    }
    avg_diff /= count;

    // 设置误差容忍度
    const float eps = 1e-3;
    bool pass = max_diff < eps && avg_diff < eps/10;

    printf("\nResult comparison:\n");
    printf("Max difference: %.6f\n", max_diff);
    printf("Avg difference: %.6f\n", avg_diff);
    printf("Tolerance: < %.6f (max), < %.6f (avg)\n", eps, eps/10);
    printf("Test %s\n", pass ? "PASSED" : "FAILED");

    // 清理资源
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    
    return pass ? 0 : 1;
} 