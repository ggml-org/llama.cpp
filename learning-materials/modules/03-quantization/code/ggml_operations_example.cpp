/**
 * GGML Operations Example
 *
 * Demonstrates core GGML tensor operations for LLM inference.
 * Compile: g++ -std=c++17 -I../../../include ggml_operations_example.cpp -L../../../build -lggml -o ggml_example
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "ggml.h"

using namespace std;

class GGMLExample {
private:
    ggml_context* ctx;
    size_t mem_size;

public:
    GGMLExample(size_t mem_size_mb = 128) :
        mem_size(mem_size_mb * 1024 * 1024) {

        // Initialize GGML context
        struct ggml_init_params params = {
            /*.mem_size   =*/ mem_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };

        ctx = ggml_init(params);
        if (!ctx) {
            throw runtime_error("Failed to initialize GGML context");
        }

        cout << "✅ Initialized GGML context with " << mem_size_mb << " MB" << endl;
    }

    ~GGMLExample() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    // Example 1: Basic Matrix Multiplication
    void example_matmul() {
        cout << "\n" << string(60, '=') << endl;
        cout << "Example 1: Matrix Multiplication" << endl;
        cout << string(60, '=') << endl;

        const int M = 512, N = 512, K = 512;

        // Create matrices
        auto* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);  // M x K
        auto* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);  // K x N
        ggml_set_name(A, "matrix_A");
        ggml_set_name(B, "matrix_B");

        // Initialize with random values
        fill_random(A);
        fill_random(B);

        // Create computation graph
        auto* C = ggml_mul_mat(ctx, A, B);  // M x N
        ggml_set_name(C, "matrix_C");

        // Build and execute graph
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, C);

        auto start = chrono::high_resolution_clock::now();
        ggml_graph_compute_with_ctx(ctx, gf, 4);  // 4 threads
        auto end = chrono::high_resolution_clock::now();

        double elapsed_ms = chrono::duration<double, milli>(end - start).count();
        double gflops = (2.0 * M * N * K) / (elapsed_ms * 1e6);

        cout << "Matrix dimensions: " << M << "x" << K << " × " << K << "x" << N << endl;
        cout << "Time: " << elapsed_ms << " ms" << endl;
        cout << "Performance: " << gflops << " GFLOPS" << endl;

        // Verify result (check first element)
        float* c_data = (float*)C->data;
        cout << "C[0,0] = " << c_data[0] << endl;
    }

    // Example 2: RMS Normalization (used in LLaMA)
    void example_rms_norm() {
        cout << "\n" << string(60, '=') << endl;
        cout << "Example 2: RMS Normalization" << endl;
        cout << string(60, '=') << endl;

        const int n_elements = 4096;
        const float eps = 1e-5f;

        // Create input tensor
        auto* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        ggml_set_name(x, "input");
        fill_random(x);

        // Apply RMS normalization
        auto* x_norm = ggml_rms_norm(ctx, x, eps);
        ggml_set_name(x_norm, "normalized");

        // Build and execute
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, x_norm);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        // Verify: Calculate actual RMS
        float* x_data = (float*)x->data;
        float* norm_data = (float*)x_norm->data;

        float rms = 0.0f;
        for (int i = 0; i < n_elements; i++) {
            rms += x_data[i] * x_data[i];
        }
        rms = sqrt(rms / n_elements + eps);

        cout << "Input RMS: " << rms << endl;
        cout << "First 5 values (original): ";
        for (int i = 0; i < 5; i++) {
            cout << x_data[i] << " ";
        }
        cout << endl;

        cout << "First 5 values (normalized): ";
        for (int i = 0; i < 5; i++) {
            cout << norm_data[i] << " ";
        }
        cout << endl;
    }

    // Example 3: Element-wise operations
    void example_elementwise() {
        cout << "\n" << string(60, '=') << endl;
        cout << "Example 3: Element-wise Operations" << endl;
        cout << string(60, '=') << endl;

        const int n = 1024;

        // Create tensors
        auto* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        fill_random(a);
        fill_random(b);

        // Element-wise operations
        auto* add = ggml_add(ctx, a, b);           // a + b
        auto* mul = ggml_mul(ctx, a, b);           // a * b
        auto* gelu = ggml_gelu(ctx, a);            // GELU activation
        auto* silu = ggml_silu(ctx, b);            // SiLU activation

        // Build graph
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, add);
        ggml_build_forward_expand(gf, mul);
        ggml_build_forward_expand(gf, gelu);
        ggml_build_forward_expand(gf, silu);

        // Execute
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        cout << "Computed element-wise operations:" << endl;
        cout << "  - Addition (a + b)" << endl;
        cout << "  - Multiplication (a * b)" << endl;
        cout << "  - GELU activation" << endl;
        cout << "  - SiLU activation" << endl;

        // Show sample results
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* add_data = (float*)add->data;
        float* mul_data = (float*)mul->data;

        cout << "\nSample results (first element):" << endl;
        cout << "  a[0] = " << a_data[0] << endl;
        cout << "  b[0] = " << b_data[0] << endl;
        cout << "  (a + b)[0] = " << add_data[0] << endl;
        cout << "  (a * b)[0] = " << mul_data[0] << endl;
    }

    // Example 4: Attention-like computation
    void example_attention() {
        cout << "\n" << string(60, '=') << endl;
        cout << "Example 4: Scaled Dot-Product Attention" << endl;
        cout << string(60, '=') << endl;

        const int seq_len = 64;
        const int d_k = 128;

        // Create Q, K, V tensors
        auto* Q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_k, seq_len);
        auto* K = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_k, seq_len);
        auto* V = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_k, seq_len);

        fill_random(Q);
        fill_random(K);
        fill_random(V);

        // Attention computation:
        // 1. QK^T / sqrt(d_k)
        auto* QK = ggml_mul_mat(ctx, K, Q);  // seq_len x seq_len
        auto* scale = ggml_new_f32(ctx, 1.0f / sqrt((float)d_k));
        auto* QK_scaled = ggml_scale(ctx, QK, scale);

        // 2. Softmax
        auto* attention_weights = ggml_soft_max(ctx, QK_scaled);

        // 3. Multiply by V
        auto* output = ggml_mul_mat(ctx, V, attention_weights);

        // Build and execute
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, output);

        auto start = chrono::high_resolution_clock::now();
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        auto end = chrono::high_resolution_clock::now();

        double elapsed_ms = chrono::duration<double, milli>(end - start).count();

        cout << "Sequence length: " << seq_len << endl;
        cout << "Dimension: " << d_k << endl;
        cout << "Attention computation time: " << elapsed_ms << " ms" << endl;

        // Verify attention weights sum to 1
        float* attn_data = (float*)attention_weights->data;
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += attn_data[i];
        }
        cout << "Attention weights sum (should be ~1.0): " << sum << endl;
    }

    // Example 5: Quantization/Dequantization
    void example_quantization() {
        cout << "\n" << string(60, '=') << endl;
        cout << "Example 5: Quantization Operations" << endl;
        cout << string(60, '=') << endl;

        const int n = 1024;

        // Create FP32 tensor
        auto* x_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        fill_random(x_f32);

        // Create Q8_0 tensor (quantized)
        auto* x_q8 = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, n);

        // Quantize (in practice, done during model loading)
        // Here we just demonstrate the data types
        cout << "Original tensor type: F32" << endl;
        cout << "Quantized tensor type: Q8_0" << endl;

        size_t f32_size = ggml_nbytes(x_f32);
        size_t q8_size = ggml_nbytes(x_q8);

        cout << "FP32 size: " << f32_size << " bytes" << endl;
        cout << "Q8_0 size: " << q8_size << " bytes" << endl;
        cout << "Compression ratio: " << (float)f32_size / q8_size << "x" << endl;
    }

    // Example 6: Graph optimization
    void example_graph_optimization() {
        cout << "\n" << string(60, '=') << endl;
        cout << "Example 6: Computation Graph" << endl;
        cout << string(60, '=') << endl;

        const int n = 512;

        // Build a more complex computation graph
        auto* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        fill_random(x);

        // Graph: y = GELU(W1 * x + b1)
        //        z = W2 * y + b2
        auto* W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        auto* b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto* W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        auto* b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

        fill_random(W1);
        fill_random(b1);
        fill_random(W2);
        fill_random(b2);

        // Build graph
        auto* h1 = ggml_mul_mat(ctx, W1, x);
        auto* h2 = ggml_add(ctx, h1, b1);
        auto* y = ggml_gelu(ctx, h2);
        auto* h3 = ggml_mul_mat(ctx, W2, y);
        auto* z = ggml_add(ctx, h3, b2);

        // Create and execute graph
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, z);

        cout << "Graph nodes: " << gf->n_nodes << endl;
        cout << "Graph leafs: " << gf->n_leafs << endl;

        auto start = chrono::high_resolution_clock::now();
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        auto end = chrono::high_resolution_clock::now();

        double elapsed_ms = chrono::duration<double, milli>(end - start).count();
        cout << "Graph execution time: " << elapsed_ms << " ms" << endl;
    }

private:
    void fill_random(ggml_tensor* tensor) {
        static random_device rd;
        static mt19937 gen(rd());
        static normal_distribution<float> dist(0.0f, 1.0f);

        int n_elements = ggml_nelements(tensor);
        float* data = (float*)tensor->data;

        for (int i = 0; i < n_elements; i++) {
            data[i] = dist(gen);
        }
    }
};


int main(int argc, char** argv) {
    cout << "GGML Operations Example" << endl;
    cout << "======================" << endl << endl;

    try {
        GGMLExample example(256);  // 256 MB context

        // Run examples
        example.example_matmul();
        example.example_rms_norm();
        example.example_elementwise();
        example.example_attention();
        example.example_quantization();
        example.example_graph_optimization();

        cout << "\n✅ All examples completed successfully!" << endl;

        return 0;
    }
    catch (const exception& e) {
        cerr << "❌ Error: " << e.what() << endl;
        return 1;
    }
}
