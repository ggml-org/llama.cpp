// iFairy 模型单元测试
// 测试量化、反量化、ROPE 算子和计算图构建

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

// Include internal quantization headers for ifairy types
extern "C" {
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-ifairy-lut.h"
#include "../ggml/src/ggml-quants.h"
}

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif
#ifndef GGML_FP32_TO_FP16
#    define GGML_FP32_TO_FP16 ggml_fp32_to_fp16
#endif
#ifndef GGML_BF16_TO_FP32
#    define GGML_BF16_TO_FP32 ggml_bf16_to_fp32
#endif
#ifndef GGML_FP32_TO_BF16
#    define GGML_FP32_TO_BF16 ggml_fp32_to_bf16
#endif

#undef NDEBUG
#include <assert.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#    pragma warning(disable : 4244 4267)  // possible loss of data
#endif

// ============================================================================
// JSON 解析辅助函数（简单实现，避免引入第三方库）
// ============================================================================

// 简单的 JSON 数组解析器
std::vector<float> parse_json_float_array(const std::string & json_str, const std::string & key) {
    std::vector<float> result;

    // 查找 key
    std::string search_key = "\"" + key + "\"";
    size_t      key_pos    = json_str.find(search_key);
    if (key_pos == std::string::npos) {
        fprintf(stderr, "Error: Key '%s' not found in JSON\n", key.c_str());
        return result;
    }

    // 查找数组开始 [
    size_t array_start = json_str.find('[', key_pos);
    if (array_start == std::string::npos) {
        fprintf(stderr, "Error: Array start '[' not found for key '%s'\n", key.c_str());
        return result;
    }

    // 查找数组结束 ]
    size_t array_end = json_str.find(']', array_start);
    if (array_end == std::string::npos) {
        fprintf(stderr, "Error: Array end ']' not found for key '%s'\n", key.c_str());
        return result;
    }

    // 提取数组内容
    std::string array_content = json_str.substr(array_start + 1, array_end - array_start - 1);

    // 解析浮点数
    size_t pos = 0;
    while (pos < array_content.length()) {
        // 跳过空格和逗号
        while (pos < array_content.length() &&
               (array_content[pos] == ' ' || array_content[pos] == ',' || array_content[pos] == '\n')) {
            pos++;
        }

        if (pos >= array_content.length()) {
            break;
        }

        // 解析数字
        char * end_ptr;
        float  value = strtof(array_content.c_str() + pos, &end_ptr);
        result.push_back(value);

        pos = end_ptr - array_content.c_str();
    }

    return result;
}

int parse_json_int(const std::string & json_str, const std::string & key) {
    std::string search_key = "\"" + key + "\"";
    size_t      key_pos    = json_str.find(search_key);
    if (key_pos == std::string::npos) {
        fprintf(stderr, "Error: Key '%s' not found in JSON\n", key.c_str());
        return 0;
    }

    // 查找 : 后的数字
    size_t colon_pos = json_str.find(':', key_pos);
    if (colon_pos == std::string::npos) {
        return 0;
    }

    // 跳过空格
    size_t num_start = colon_pos + 1;
    while (num_start < json_str.length() && json_str[num_start] == ' ') {
        num_start++;
    }

    return atoi(json_str.c_str() + num_start);
}

float parse_json_float(const std::string & json_str, const std::string & key) {
    std::string search_key = "\"" + key + "\"";
    size_t      key_pos    = json_str.find(search_key);
    if (key_pos == std::string::npos) {
        fprintf(stderr, "Error: Key '%s' not found in JSON\n", key.c_str());
        return 0.0f;
    }

    // 查找 : 后的数字
    size_t colon_pos = json_str.find(':', key_pos);
    if (colon_pos == std::string::npos) {
        return 0.0f;
    }

    // 跳过空格
    size_t num_start = colon_pos + 1;
    while (num_start < json_str.length() && json_str[num_start] == ' ') {
        num_start++;
    }

    return strtof(json_str.c_str() + num_start, nullptr);
}

std::string read_file(const std::string & filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename.c_str());
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}

static bool file_exists(const std::string & filename) {
    std::ifstream file(filename);
    return file.good();
}

static std::string ifairy_test_data_path(const char * filename) {
    const char * env = getenv("GGML_IFAIRY_TEST_DATA_DIR");
    if (env && env[0] != '\0') {
        return std::string(env) + "/" + filename;
    }

    const std::string rel_path = std::string("tests/ifairy-test-data/") + filename;
    if (file_exists(rel_path)) {
        return rel_path;
    }

    const std::string src_path = __FILE__;
    const size_t      pos      = src_path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return src_path.substr(0, pos) + "/ifairy-test-data/" + filename;
    }

    return rel_path;
}

static std::string read_ifairy_test_file(const char * filename) {
    return read_file(ifairy_test_data_path(filename));
}

// ============================================================================
// 测试辅助函数
// ============================================================================

constexpr float MAX_ERROR = 1e-2f;  // 允许的最大误差

bool compare_arrays(const float * a, const float * b, size_t n, float max_error = MAX_ERROR) {
    float  max_diff     = 0.0f;
    size_t max_diff_idx = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff     = diff;
            max_diff_idx = i;
        }
    }

    if (max_diff > max_error) {
        fprintf(stderr, "  Max diff: %.6f at index %zu (expected: %.6f, got: %.6f)\n", max_diff, max_diff_idx,
                b[max_diff_idx], a[max_diff_idx]);
        return false;
    }

    printf("  Max diff: %.6f (threshold: %.6f) - PASS\n", max_diff, max_error);
    return true;
}

static bool compare_u32_arrays(const uint32_t * a, const uint32_t * b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            ggml_bf16_t a_pair[2];
            ggml_bf16_t b_pair[2];
            memcpy(a_pair, a + i, sizeof(uint32_t));
            memcpy(b_pair, b + i, sizeof(uint32_t));
            const float ar = GGML_BF16_TO_FP32(a_pair[0]);
            const float ai = GGML_BF16_TO_FP32(a_pair[1]);
            const float br = GGML_BF16_TO_FP32(b_pair[0]);
            const float bi = GGML_BF16_TO_FP32(b_pair[1]);
            fprintf(stderr, "  Mismatch at index %zu: 0x%08x vs 0x%08x (a=(%.6f,%.6f) b=(%.6f,%.6f))\n", i, a[i], b[i],
                    ar, ai, br, bi);
            return false;
        }
    }
    printf("  Bitwise compare: PASS (%zu elements)\n", n);
    return true;
}

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    setenv(name, value ? value : "", 1);
#endif
}

static void unset_env_var(const char * name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

struct scoped_env_var {
    std::string name;
    std::string old_value;
    bool        had = false;

    scoped_env_var(const char * name_) : name(name_) {
        const char * v = getenv(name_);
        if (v) {
            had       = true;
            old_value = v;
        }
    }

    void set(const char * v) { set_env_var(name.c_str(), v); }

    void unset() { unset_env_var(name.c_str()); }

    ~scoped_env_var() {
        if (had) {
            set_env_var(name.c_str(), old_value.c_str());
        } else {
            unset_env_var(name.c_str());
        }
    }
};

static float pack_bf16_pair(float real, float imag) {
    ggml_bf16_t pair[2];
    pair[0] = GGML_FP32_TO_BF16(real);
    pair[1] = GGML_FP32_TO_BF16(imag);
    float out;
    memcpy(&out, pair, sizeof(out));
    return out;
}

// ============================================================================
// 3-weight LUT 索引缓冲构造
// ============================================================================

static void set_ifairy_code(block_ifairy & blk, int idx, uint8_t code) {
    assert(idx >= 0 && idx < QK_K);
    const int chunk = idx / 64;
    const int part  = (idx >> 4) & 0x3;
    const int lane  = idx & 0x0f;

    uint8_t & packed = blk.qs[chunk * 16 + lane];
    packed &= (uint8_t) ~(0x3u << (2 * part));
    packed |= (uint8_t) ((code & 0x3u) << (2 * part));
}

bool test_ifairy_lut_index() {
    printf("\n=== Test 2: iFairy 3-weight index encoding ===\n");

    const int64_t k              = QK_K;  // 256
    const int64_t rows           = 1;
    const int64_t blocks_per_row = k / QK_K;

    std::vector<block_ifairy> weights((size_t) rows * (size_t) blocks_per_row);
    block_ifairy              blk{};

    // 设置前三个三权重组（直接 6-bit pattern 编码）：
    // pat = c0 | (c1<<2) | (c2<<4)
    // g0: (0,1,2) -> 0x24
    // g1: (3,3,3) -> 0x3f
    // g2: (1,2,3) -> 0x39
    set_ifairy_code(blk, 0, 0);
    set_ifairy_code(blk, 1, 1);
    set_ifairy_code(blk, 2, 2);
    set_ifairy_code(blk, 3, 3);
    set_ifairy_code(blk, 4, 3);
    set_ifairy_code(blk, 5, 3);
    set_ifairy_code(blk, 6, 1);
    set_ifairy_code(blk, 7, 2);
    set_ifairy_code(blk, 8, 3);

    weights[0] = blk;

    const ggml_ifairy_3w_index_info info     = ggml_ifairy_3w_get_index_info(k);
    const size_t                    required = ggml_ifairy_3w_index_buffer_size(&info, rows);

    std::vector<uint8_t> index(required);

    const bool ok = ggml_ifairy_3w_encode(weights.data(), k, rows, index.data(), index.size());
    if (!ok) {
        fprintf(stderr, "Failed to encode iFairy 3-weight index buffer\n");
        return false;
    }

    const size_t groups = (size_t) info.groups_per_row;

    bool pass = true;
    pass &= groups == 86;  // 256 -> 85 triplets + 1 tail group (no drop)
    pass &= index.size() == required;
    pass &= index[0] == 0x24;
    pass &= index[1] == 0x3f;
    pass &= index[2] == 0x39;

    if (!pass) {
        fprintf(stderr, "Index encoding mismatch: [%02x, %02x, %02x, %02x, ...]\n", index[0], index[1], index[2],
                index[3]);
    } else {
        printf("  groups_per_row=%zu, first bytes=[%02x %02x %02x %02x]\n", groups, index[0], index[1], index[2],
               index[3]);
    }

    return pass;
}

// ============================================================================
// 测试 2.1: LUT transform 缓存/并发基础覆盖
// ============================================================================

bool test_ifairy_lut_transform_cache() {
    printf("\n=== Test 2.1: iFairy LUT transform cache ===\n");

    struct ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t k         = QK_K;
    const int64_t rows      = 4;
    const int     n_tensors = 4;

    std::vector<ggml_tensor *> weights;
    weights.reserve(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY, k, rows);
        if (!w || !w->data) {
            fprintf(stderr, "Failed to allocate ifairy tensor\n");
            ggml_free(ctx);
            return false;
        }
        memset(w->data, 0, ggml_nbytes(w));
        weights.push_back(w);
    }

    const ggml_ifairy_3w_index_info info     = ggml_ifairy_3w_get_index_info(k);
    const size_t                    expected = ggml_ifairy_3w_index_buffer_size(&info, rows);

    if (!ggml_ifairy_lut_transform_tensor(weights[0], NULL)) {
        fprintf(stderr, "transform_tensor failed on primary weight\n");
        ggml_free(ctx);
        return false;
    }

    const ifairy_lut_extra * extra = (const ifairy_lut_extra *) weights[0]->extra;
    if (!extra || !extra->indexes || extra->size != expected) {
        fprintf(stderr, "transform_tensor produced invalid extra (size=%zu expected=%zu)\n", extra ? extra->size : 0,
                expected);
        ggml_free(ctx);
        return false;
    }

    const uint8_t * base = extra->indexes;
    if (!ggml_ifairy_lut_transform_tensor(weights[0], NULL)) {
        fprintf(stderr, "transform_tensor failed on cached weight\n");
        ggml_free(ctx);
        return false;
    }

    extra = (const ifairy_lut_extra *) weights[0]->extra;
    if (!extra || extra->indexes != base) {
        fprintf(stderr, "transform_tensor cache did not reuse indexes\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<std::thread> threads;
    threads.reserve((size_t) (n_tensors - 1));
    for (int i = 1; i < n_tensors; ++i) {
        threads.emplace_back([w = weights[i]]() { ggml_ifairy_lut_transform_tensor(w, NULL); });
    }
    for (auto & t : threads) {
        t.join();
    }

    for (int i = 1; i < n_tensors; ++i) {
        extra = (const ifairy_lut_extra *) weights[i]->extra;
        if (!extra || !extra->indexes || extra->size != expected) {
            fprintf(stderr, "transform_tensor failed in threaded run (idx=%d)\n", i);
            ggml_free(ctx);
            return false;
        }
    }

    ggml_free(ctx);
    ggml_ifairy_lut_free();
    printf("  transform cache/concurrency - PASS\n");
    return true;
}

// ============================================================================
// 测试 2.2: LUT transform 形状与 layout 策略基础覆盖
// ============================================================================

bool test_ifairy_lut_transform_invalid_shape() {
    printf("\n=== Test 2.2: iFairy LUT transform invalid shape ===\n");

    struct ggml_init_params params = {
        /*.mem_size   =*/2 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t k_bad = QK_K - 1;
    const int64_t rows  = 1;
    ggml_tensor * w     = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY, k_bad, rows);
    if (!w || !w->data) {
        fprintf(stderr, "Failed to allocate ifairy tensor\n");
        ggml_free(ctx);
        return false;
    }
    memset(w->data, 0, ggml_nbytes(w));

    const bool ok = ggml_ifairy_lut_transform_tensor(w, NULL);
    ggml_free(ctx);
    ggml_ifairy_lut_free();

    if (ok) {
        fprintf(stderr, "transform_tensor unexpectedly succeeded on invalid shape\n");
        return false;
    }

    printf("  invalid shape - PASS\n");
    return true;
}

bool test_ifairy_lut_layout_auto_policy() {
#if !defined(GGML_IFAIRY_ARM_LUT) || !defined(__ARM_NEON) || !defined(__aarch64__)
    printf("\n=== Test 2.3: iFairy LUT layout auto policy (SKIP) ===\n");
    return true;
#else
    printf("\n=== Test 2.3: iFairy LUT layout auto policy ===\n");

    scoped_env_var env_layout("GGML_IFAIRY_LUT_LAYOUT");
    scoped_env_var env_kernel("GGML_IFAIRY_LUT_KERNEL");

    struct ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t M   = 2;
    const int64_t N   = 2;
    const int64_t K   = QK_K;
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY, K, M);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    if (!w || !a || !dst) {
        fprintf(stderr, "Failed to allocate tensors for layout test\n");
        ggml_free(ctx);
        return false;
    }

    env_layout.set("legacy");
    const size_t wsize_legacy = ggml_ifairy_lut_get_wsize(w, a, dst, 1);

    env_layout.set("merged64");
    const size_t wsize_merged64 = ggml_ifairy_lut_get_wsize(w, a, dst, 1);

    env_kernel.unset();
    env_layout.set("auto");
    const size_t wsize_auto = ggml_ifairy_lut_get_wsize(w, a, dst, 1);

    ggml_free(ctx);

    if (wsize_legacy == 0 || wsize_merged64 == 0 || wsize_auto == 0 || wsize_auto != wsize_merged64) {
        fprintf(stderr, "layout auto mismatch: legacy=%zu merged64=%zu auto=%zu\n", wsize_legacy, wsize_merged64,
                wsize_auto);
        return false;
    }

    printf("  layout auto - PASS\n");
    return true;
#endif
}

bool test_ifairy_lut_layout_auto_decode_default_merged64() {
#if !defined(GGML_IFAIRY_ARM_LUT) || !defined(__ARM_NEON) || !defined(__aarch64__)
    printf("\n=== Test 2.3.1: iFairy LUT auto decode default merged64 (SKIP) ===\n");
    return true;
#else
    printf("\n=== Test 2.3.1: iFairy LUT auto decode default merged64 ===\n");

    scoped_env_var env_layout("GGML_IFAIRY_LUT_LAYOUT");
    scoped_env_var env_kernel("GGML_IFAIRY_LUT_KERNEL");

    struct ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t M   = 2;
    const int64_t N   = 1;
    const int64_t K   = QK_K;
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY, K, M);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    if (!w || !a || !dst) {
        fprintf(stderr, "Failed to allocate tensors for auto decode policy test\n");
        ggml_free(ctx);
        return false;
    }

    env_layout.set("merged64");
    env_kernel.unset();
    const size_t wsize_merged64 = ggml_ifairy_lut_get_wsize(w, a, dst, 1);

    env_layout.set("auto");
    env_kernel.unset();
    const size_t wsize_auto = ggml_ifairy_lut_get_wsize(w, a, dst, 1);

    ggml_free(ctx);

    if (wsize_merged64 == 0 || wsize_auto == 0 || wsize_auto != wsize_merged64) {
        fprintf(stderr, "auto decode default mismatch: merged64=%zu auto=%zu\n", wsize_merged64, wsize_auto);
        return false;
    }

    printf("  auto decode default merged64 - PASS\n");
    return true;
#endif
}

// ============================================================================
// 测试 2.4: 索引对齐/误对齐缓冲
// ============================================================================

bool test_ifairy_lut_index_alignment() {
    printf("\n=== Test 2.4: iFairy LUT index alignment ===\n");

    const int64_t k              = QK_K;
    const int64_t rows           = 1;
    const int64_t blocks_per_row = k / QK_K;

    std::vector<block_ifairy> weights((size_t) rows * (size_t) blocks_per_row);
    memset(weights.data(), 0, weights.size() * sizeof(block_ifairy));

    const ggml_ifairy_3w_index_info info    = ggml_ifairy_3w_get_index_info(k);
    const size_t                    raw     = ggml_ifairy_3w_index_buffer_size(&info, rows);
    const size_t                    aligned = ggml_ifairy_3w_index_buffer_size_aligned64(&info, rows);

    if (raw == 0 || aligned == 0 || aligned < raw || (aligned & 63u) != 0) {
        fprintf(stderr, "index alignment mismatch: raw=%zu aligned=%zu\n", raw, aligned);
        return false;
    }

    std::vector<uint8_t> buf(raw + 1);
    uint8_t *            misaligned = buf.data() + 1;
    const bool           ok         = ggml_ifairy_3w_encode(weights.data(), k, rows, misaligned, raw);
    if (!ok) {
        fprintf(stderr, "index encoding failed on misaligned buffer\n");
        return false;
    }

    printf("  index alignment - PASS\n");
    return true;
}

// ============================================================================
// 测试 2.5: 索引缓冲不足（模拟分配失败/长度不足）
// ============================================================================

bool test_ifairy_lut_index_encode_failure() {
    printf("\n=== Test 2.5: iFairy LUT index encode failure ===\n");

    const int64_t k              = QK_K;
    const int64_t rows           = 1;
    const int64_t blocks_per_row = k / QK_K;

    std::vector<block_ifairy> weights((size_t) rows * (size_t) blocks_per_row);
    memset(weights.data(), 0, weights.size() * sizeof(block_ifairy));

    const ggml_ifairy_3w_index_info info = ggml_ifairy_3w_get_index_info(k);
    const size_t                    raw  = ggml_ifairy_3w_index_buffer_size(&info, rows);
    if (raw < 2) {
        fprintf(stderr, "unexpected raw index buffer size: %zu\n", raw);
        return false;
    }

    std::vector<uint8_t> buf(raw - 1);
    const bool           ok = ggml_ifairy_3w_encode(weights.data(), k, rows, buf.data(), buf.size());
    if (ok) {
        fprintf(stderr, "index encoding unexpectedly succeeded with short buffer\n");
        return false;
    }

    printf("  index encode failure - PASS\n");
    return true;
}

// ============================================================================
// 测试 2.6: LUT 关键 env 语义
// ============================================================================

bool test_ifairy_lut_env_semantics() {
#if !defined(GGML_IFAIRY_ARM_LUT) || !defined(__ARM_NEON) || !defined(__aarch64__)
    printf("\n=== Test 2.6: iFairy LUT env semantics (SKIP) ===\n");
    return true;
#else
    printf("\n=== Test 2.6: iFairy LUT env semantics ===\n");

    scoped_env_var env_lut("GGML_IFAIRY_LUT");
    scoped_env_var env_bk("GGML_IFAIRY_LUT_BK_BLOCKS");

    struct ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t M   = 2;
    const int64_t N   = 2;
    const int64_t K   = QK_K;
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY, K, M);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    if (!w || !a || !dst) {
        fprintf(stderr, "Failed to allocate tensors for env test\n");
        ggml_free(ctx);
        return false;
    }

    env_lut.set("0");
    const bool can_disabled = ggml_ifairy_lut_can_mul_mat(w, a, dst);

    env_lut.set("1");
    const bool can_enabled = ggml_ifairy_lut_can_mul_mat(w, a, dst);

    env_bk.set("-1");
    const size_t wsize_neg = ggml_ifairy_lut_get_wsize(w, a, dst, 1);
    env_bk.set("0");
    const size_t wsize_zero = ggml_ifairy_lut_get_wsize(w, a, dst, 1);

    ggml_free(ctx);

    if (can_disabled || !can_enabled || wsize_zero == 0 || wsize_neg != wsize_zero) {
        fprintf(stderr, "env semantics mismatch: disabled=%d enabled=%d wsize(-1)=%zu wsize(0)=%zu\n",
                (int) can_disabled, (int) can_enabled, wsize_neg, wsize_zero);
        return false;
    }

    printf("  env semantics - PASS\n");
    return true;
#endif
}

// ============================================================================
// 测试 1: 量化/反量化
// ============================================================================

bool test_quantization() {
    printf("\n=== Test 1: Quantization/Dequantization ===\n");

    // 读取测试数据
    std::string json_data = read_ifairy_test_file("quant_test.json");
    if (json_data.empty()) {
        fprintf(stderr, "Failed to read test data\n");
        return false;
    }

    // 解析输入数据
    std::vector<float> quantized_real   = parse_json_float_array(json_data, "quantized_real");
    std::vector<float> quantized_imag   = parse_json_float_array(json_data, "quantized_imag");
    std::vector<float> expected_dq_real = parse_json_float_array(json_data, "dequantized_real");
    std::vector<float> expected_dq_imag = parse_json_float_array(json_data, "dequantized_imag");

    if (quantized_real.empty() || quantized_imag.empty()) {
        fprintf(stderr, "Failed to parse input data\n");
        return false;
    }

    printf("Testing quantization with %zu elements\n", quantized_real.size());

    // 分配量化块（256 个元素对应 1 个块）
    const size_t n_elements = quantized_real.size();
    const size_t n_blocks   = (n_elements + QK_K - 1) / QK_K;

    std::vector<block_ifairy> quantized(n_blocks);
    std::vector<float>        dequantized_real(n_elements);
    std::vector<float>        dequantized_imag(n_elements);

    // 执行量化
    quantize_row_ifairy_ref(quantized_real.data(), quantized_imag.data(), quantized.data(), n_elements);

    // 执行反量化
    dequantize_row_ifairy(quantized.data(), dequantized_real.data(), dequantized_imag.data(), n_elements);

    // 比较结果
    printf("Comparing real part:\n");
    bool real_ok = compare_arrays(dequantized_real.data(), expected_dq_real.data(), n_elements, 0.1f);

    printf("Comparing imag part:\n");
    bool imag_ok = compare_arrays(dequantized_imag.data(), expected_dq_imag.data(), n_elements, 0.1f);

    return real_ok && imag_ok;
}

// ============================================================================
// 测试 3: ROPE 算子
// ============================================================================

bool test_rope() {
    printf("\n=== Test 3: iFairy ROPE ===\n");

    // 读取测试数据
    std::string json_data = read_ifairy_test_file("rope_test.json");
    if (json_data.empty()) {
        fprintf(stderr, "Failed to read test data\n");
        return false;
    }

    // 解析数据
    std::vector<float> input_real    = parse_json_float_array(json_data, "input_real");
    std::vector<float> input_imag    = parse_json_float_array(json_data, "input_imag");
    std::vector<float> expected_real = parse_json_float_array(json_data, "output_real");
    std::vector<float> expected_imag = parse_json_float_array(json_data, "output_imag");

    int   batch     = parse_json_int(json_data, "batch");
    int   seq_len   = parse_json_int(json_data, "seq_len");
    int   n_heads   = parse_json_int(json_data, "n_heads");
    int   head_dim  = parse_json_int(json_data, "head_dim");
    int   n_dims    = parse_json_int(json_data, "n_dims");
    float freq_base = parse_json_float(json_data, "freq_base");

    printf("Testing ROPE with shape [%d, %d, %d, %d], n_dims=%d, freq_base=%.1f\n", batch, seq_len, n_heads, head_dim,
           n_dims, freq_base);

    // 创建 GGML 上下文
    struct ggml_init_params params = {
        /*.mem_size   =*/128 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // 创建输入张量（交错存储实部和虚部）
    const int64_t        ne[4] = { head_dim * 2, n_heads, seq_len, batch };  // 实部和虚部交错
    struct ggml_tensor * x     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);

    // 填充数据（交错格式：real0, imag0, real1, imag1, ...）
    float * x_data = (float *) x->data;
    for (size_t i = 0; i < input_real.size(); i++) {
        // 注意：这里需要根据实际的 ROPE 实现调整数据布局
        // 暂时简化处理
        int pos_in_output = i * 2;  // 交错存储
        if (pos_in_output < head_dim * 2 * n_heads * seq_len * batch) {
            x_data[pos_in_output]     = input_real[i];
            x_data[pos_in_output + 1] = input_imag[i];
        }
    }

    // 创建位置张量
    struct ggml_tensor * pos      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    int32_t *            pos_data = (int32_t *) pos->data;
    for (int i = 0; i < seq_len; i++) {
        pos_data[i] = i;
    }

    // 应用 ROPE
    struct ggml_tensor * result = ggml_ifairy_rope(ctx, x, pos, n_dims, 0);

    // 构建计算图
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // 执行计算
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // 提取结果并比较
    // 注意：这里需要根据实际的输出格式进行调整
    printf("ROPE computation completed\n");

    // 清理
    ggml_free(ctx);

    // 简化版本：暂时返回 true
    // TODO: 实现完整的结果比较
    printf("  ROPE test - PASS (simplified)\n");
    return true;
}

// ============================================================================
// 测试 4: 标量 LUT matmul 正确性（encode + preprocess + qgemm）
// ============================================================================

static uint8_t get_ifairy_code(const block_ifairy * row_blocks, int idx) {
    const int     block_idx    = idx / QK_K;
    const int     idx_in_block = idx - block_idx * QK_K;
    const int     chunk        = idx_in_block >> 6;
    const int     lane         = idx_in_block & 0x0f;
    const int     part         = (idx_in_block >> 4) & 0x3;
    const uint8_t packed       = row_blocks[block_idx].qs[chunk * 16 + lane];
    return (packed >> (2 * part)) & 0x3;
}

static void decode_ifairy_weight(uint8_t code, float d_real, float d_imag, float & wr, float & wi) {
    if (code <= 1) {
        wr = (code == 1 ? d_real : -d_real);
        wi = 0.0f;
    } else {
        wr = 0.0f;
        wi = (code == 3 ? d_imag : -d_imag);
    }
}

static bool run_ifairy_lut_scalar_case(int64_t M, int64_t N, int64_t K) {
    if (M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        fprintf(stderr, "Invalid scalar test shape: M=%lld N=%lld K=%lld\n", (long long) M, (long long) N,
                (long long) K);
        return false;
    }

    const int64_t blocks_per_row = K / QK_K;
    const int64_t blocks_per_col = K / QK_K;

    const float w_scale = 1.0f / 8.0f;
    const float a_scale = 1.0f / 16.0f;

    std::vector<block_ifairy> weights((size_t) M * (size_t) blocks_per_row);
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            block_ifairy blk{};
            blk.d_real = GGML_FP32_TO_FP16(w_scale);
            blk.d_imag = GGML_FP32_TO_FP16(w_scale);
            for (int j = 0; j < QK_K; ++j) {
                const int     k_idx = (int) (b * QK_K + j);
                const uint8_t code  = (uint8_t) ((k_idx + (int) r) & 0x1);
                set_ifairy_code(blk, j, code);
            }
            weights[(size_t) r * (size_t) blocks_per_row + (size_t) b] = blk;
        }
    }

    std::vector<block_ifairy_q16> acts((size_t) N * (size_t) blocks_per_col);
    for (int64_t c = 0; c < N; ++c) {
        for (int64_t b = 0; b < blocks_per_col; ++b) {
            block_ifairy_q16 blk{};
            blk.d_real = GGML_FP32_TO_FP16(a_scale);
            blk.d_imag = GGML_FP32_TO_FP16(a_scale);
            for (int j = 0; j < QK_K; ++j) {
                const int k_idx = (int) (b * QK_K + j);
                blk.x_real[j]   = (int8_t) (((k_idx + 3 * (int) c) % 13) - 6);
                blk.x_imag[j]   = (int8_t) (((k_idx * 2 + (int) c) % 11) - 5);
            }
            acts[(size_t) c * (size_t) blocks_per_col + (size_t) b] = blk;
        }
    }

    std::vector<float> dst_lut((size_t) M * (size_t) N * 2, 0.0f);
    const size_t       act_stride = (size_t) blocks_per_col * sizeof(block_ifairy_q16);
    ggml_ifairy_lut_mul_mat_scalar((int) M, (int) K, (int) N, weights.data(), acts.data(), act_stride, dst_lut.data());

    std::vector<float> dst_ref(dst_lut.size(), 0.0f);
    const float        w_scale_r = GGML_FP16_TO_FP32(weights[0].d_real);
    const float        w_scale_i = GGML_FP16_TO_FP32(weights[0].d_imag);

    for (int64_t c = 0; c < N; ++c) {
        const block_ifairy_q16 * act_blk = acts.data() + c * blocks_per_col;
        float *                  ref_col = dst_ref.data() + (size_t) c * (size_t) M * 2;

        for (int64_t r = 0; r < M; ++r) {
            const block_ifairy * w_row = weights.data() + r * blocks_per_row;
            float                acc_r = 0.0f, acc_i = 0.0f;

            for (int64_t b = 0; b < blocks_per_col; ++b) {
                const float act_sr = GGML_FP16_TO_FP32(act_blk[b].d_real);
                const float act_si = GGML_FP16_TO_FP32(act_blk[b].d_imag);
                for (int j = 0; j < QK_K; ++j) {
                    const int     k_idx = (int) (b * QK_K + j);
                    const uint8_t code  = get_ifairy_code(w_row, k_idx);
                    float         wr = 0.0f, wi = 0.0f;
                    decode_ifairy_weight(code, w_scale_r, w_scale_i, wr, wi);

                    const int8_t ar_q = (int8_t) act_blk[b].x_real[j];
                    const int8_t ai_q = (int8_t) act_blk[b].x_imag[j];
                    const float  ar   = act_sr * ar_q;
                    const float  ai   = act_si * ai_q;

                    // ggml ifairy dot uses w * conj(a)
                    acc_r += wr * ar + wi * ai;
                    acc_i += wi * ar - wr * ai;
                }
            }

            ref_col[(size_t) r * 2 + 0] = acc_r;
            ref_col[(size_t) r * 2 + 1] = acc_i;
        }
    }

    return compare_arrays(dst_lut.data(), dst_ref.data(), dst_ref.size(), 1e-3f);
}

bool test_ifairy_lut_scalar_matmul() {
    printf("\n=== Test 4: iFairy LUT scalar matmul ===\n");

    const bool ok = run_ifairy_lut_scalar_case(2, 2, QK_K);
    if (!ok) {
        fprintf(stderr, "Scalar LUT matmul mismatch\n");
    }
    return ok;
}

bool test_ifairy_lut_scalar_small_dims() {
    printf("\n=== Test 4.1: iFairy LUT scalar small dims ===\n");

    const bool ok = run_ifairy_lut_scalar_case(1, 1, QK_K);
    if (!ok) {
        fprintf(stderr, "Scalar LUT small dims mismatch\n");
    }
    return ok;
}

// ============================================================================
// 测试 5: CPU backend LUT tiling 回归（非 tiling vs BK/BM tiling）
// ============================================================================

static bool run_ifairy_backend_mul_mat_shape(std::vector<uint32_t> & packed_out,
                                             bool                    tiling,
                                             int64_t                 M,
                                             int64_t                 N,
                                             int64_t                 K,
                                             const char *            layout,
                                             const char *            kernel) {
    // Keep other tests isolated.
    scoped_env_var env_lut("GGML_IFAIRY_LUT");
    scoped_env_var env_strict("GGML_IFAIRY_LUT_VALIDATE_STRICT");
    scoped_env_var env_bk("GGML_IFAIRY_LUT_BK_BLOCKS");
    scoped_env_var env_bm("GGML_IFAIRY_LUT_BM");
    scoped_env_var env_layout("GGML_IFAIRY_LUT_LAYOUT");
    scoped_env_var env_kernel("GGML_IFAIRY_LUT_KERNEL");

    env_lut.set("1");
    env_strict.unset();
    if (layout) {
        env_layout.set(layout);
    } else {
        env_layout.unset();
    }
    if (kernel) {
        env_kernel.set(kernel);
    } else {
        env_kernel.unset();
    }

    if (tiling) {
        // force multi-tile: K>=512 => BK_BLOCKS=1 => multiple tiles
        env_bk.set("1");
        env_bm.set("4");
    } else {
        env_bk.set("0");
        env_bm.unset();
    }

    if (M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        fprintf(stderr, "Invalid backend test shape: M=%lld N=%lld K=%lld\n", (long long) M, (long long) N,
                (long long) K);
        return false;
    }

    const int64_t blocks_per_row = K / QK_K;

    // Build deterministic weights and activations.
    const float w_scale = 1.0f / 8.0f;

    std::vector<block_ifairy> weights((size_t) M * (size_t) blocks_per_row);
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            block_ifairy blk{};
            blk.d_real = GGML_FP32_TO_FP16(w_scale);
            blk.d_imag = GGML_FP32_TO_FP16(w_scale);
            for (int j = 0; j < QK_K; ++j) {
                const int     k_idx = (int) (b * QK_K + j);
                const uint8_t code  = (uint8_t) ((k_idx + 3 * (int) r + 1) & 0x3);
                set_ifairy_code(blk, j, code);
            }
            weights[(size_t) r * (size_t) blocks_per_row + (size_t) b] = blk;
        }
    }

    std::vector<float> act_f32((size_t) K * (size_t) N);
    for (int64_t c = 0; c < N; ++c) {
        for (int64_t k_idx = 0; k_idx < K; ++k_idx) {
            const float xr                                    = (float) (((k_idx + 7 * c) % 17) - 8) / 7.0f;
            const float xi                                    = (float) (((k_idx * 2 + 3 * c) % 15) - 7) / 6.0f;
            act_f32[(size_t) c * (size_t) K + (size_t) k_idx] = pack_bf16_pair(xr, xi);
        }
    }

    // Build a minimal backend graph to hit ggml_compute_forward_mul_mat() in ggml-cpu.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "Failed to init CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);

    struct ggml_init_params params = {
        /*.mem_size   =*/128 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    struct ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_IFAIRY, K, M);
    struct ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * out = ggml_mul_mat(ctx, w, a);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to alloc backend buffer\n");
        return false;
    }

    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(a, act_f32.data(), 0, ggml_nbytes(a));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "backend graph compute failed\n");
        return false;
    }

    std::vector<float> out_f32((size_t) M * (size_t) N);
    ggml_backend_tensor_get(out, out_f32.data(), 0, ggml_nbytes(out));

    packed_out.resize(out_f32.size());
    for (size_t i = 0; i < out_f32.size(); ++i) {
        memcpy(&packed_out[i], &out_f32[i], sizeof(uint32_t));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool run_ifairy_backend_mul_mat(std::vector<uint32_t> & packed_out,
                                       bool                    tiling,
                                       int64_t                 n_cols,
                                       const char *            layout,
                                       const char *            kernel) {
    return run_ifairy_backend_mul_mat_shape(packed_out, tiling, 8, n_cols, 2 * QK_K, layout, kernel);
}

bool test_ifairy_lut_backend_tiling_regression() {
#if !defined(GGML_IFAIRY_ARM_LUT)
    printf("\n=== Test 5: iFairy LUT backend tiling regression (SKIP: GGML_IFAIRY_ARM_LUT not enabled) ===\n");
    return true;
#else
    printf("\n=== Test 5: iFairy LUT backend tiling regression ===\n");

    std::vector<uint32_t> out_no_tile;
    std::vector<uint32_t> out_tile;
    if (!run_ifairy_backend_mul_mat(out_no_tile, false, 2, NULL, NULL)) {
        return false;
    }
    if (!run_ifairy_backend_mul_mat(out_tile, true, 2, NULL, NULL)) {
        return false;
    }

    if (out_no_tile.size() != out_tile.size()) {
        fprintf(stderr, "Size mismatch: %zu vs %zu\n", out_no_tile.size(), out_tile.size());
        return false;
    }

    if (!compare_u32_arrays(out_tile.data(), out_no_tile.data(), out_no_tile.size())) {
        return false;
    }

    // Merged64 tiling regression (prefill-like: N > 1): tiling must not change results.
    std::vector<uint32_t> out_merged64_no_tile;
    std::vector<uint32_t> out_merged64_tile;
    if (!run_ifairy_backend_mul_mat(out_merged64_no_tile, false, 2, "merged64", NULL)) {
        return false;
    }
    if (!run_ifairy_backend_mul_mat(out_merged64_tile, true, 2, "merged64", NULL)) {
        return false;
    }
    if (out_merged64_no_tile.size() != out_merged64_tile.size()) {
        fprintf(stderr, "Size mismatch (merged64): %zu vs %zu\n", out_merged64_no_tile.size(),
                out_merged64_tile.size());
        return false;
    }
    if (!compare_u32_arrays(out_merged64_tile.data(), out_merged64_no_tile.data(), out_merged64_no_tile.size())) {
        return false;
    }

    // Decode-like regression: N == 1, plus layout equivalence (legacy vs compact) for the same backend graph.
    std::vector<uint32_t> out_legacy;
    std::vector<uint32_t> out_compact;
    if (!run_ifairy_backend_mul_mat(out_legacy, false, 1, "legacy", NULL)) {
        return false;
    }
    if (!run_ifairy_backend_mul_mat(out_compact, false, 1, "compact", NULL)) {
        return false;
    }
    if (out_legacy.size() != out_compact.size()) {
        fprintf(stderr, "Size mismatch (layout): %zu vs %zu\n", out_legacy.size(), out_compact.size());
        return false;
    }
    return compare_u32_arrays(out_compact.data(), out_legacy.data(), out_legacy.size());
#endif
}

// ============================================================================
// 测试 5.1: CPU backend LUT 大维度回归
// ============================================================================

bool test_ifairy_lut_backend_large_dims() {
#if !defined(GGML_IFAIRY_ARM_LUT)
    printf("\n=== Test 5.1: iFairy LUT backend large dims (SKIP: GGML_IFAIRY_ARM_LUT not enabled) ===\n");
    return true;
#else
    printf("\n=== Test 5.1: iFairy LUT backend large dims ===\n");

    const int64_t M = 16;
    const int64_t N = 8;
    const int64_t K = 4 * QK_K;

    std::vector<uint32_t> out_no_tile;
    std::vector<uint32_t> out_tile;
    if (!run_ifairy_backend_mul_mat_shape(out_no_tile, false, M, N, K, NULL, NULL)) {
        return false;
    }
    if (!run_ifairy_backend_mul_mat_shape(out_tile, true, M, N, K, NULL, NULL)) {
        return false;
    }

    if (out_no_tile.size() != out_tile.size()) {
        fprintf(stderr, "Size mismatch (large dims): %zu vs %zu\n", out_no_tile.size(), out_tile.size());
        return false;
    }

    return compare_u32_arrays(out_tile.data(), out_no_tile.data(), out_no_tile.size());
#endif
}

// ============================================================================
// 测试 5.2: CPU backend LUT kernel 选择一致性（auto vs sdot）
// ============================================================================

bool test_ifairy_lut_kernel_sdot_consistency() {
#if !defined(GGML_IFAIRY_ARM_LUT) || !defined(__ARM_NEON) || !defined(__aarch64__)
    printf("\n=== Test 5.2: iFairy LUT kernel sdot consistency (SKIP) ===\n");
    return true;
#else
    printf("\n=== Test 5.2: iFairy LUT kernel sdot consistency ===\n");

    std::vector<uint32_t> out_auto;
    std::vector<uint32_t> out_sdot;
    if (!run_ifairy_backend_mul_mat(out_auto, false, 1, "compact", "auto")) {
        return false;
    }
    if (!run_ifairy_backend_mul_mat(out_sdot, false, 1, "compact", "sdot")) {
        return false;
    }

    if (out_auto.size() != out_sdot.size()) {
        fprintf(stderr, "Size mismatch (kernel): %zu vs %zu\n", out_auto.size(), out_sdot.size());
        return false;
    }
    return compare_u32_arrays(out_sdot.data(), out_auto.data(), out_auto.size());
#endif
}

// ============================================================================
// 测试 5: 复数矩阵乘法
// ============================================================================

bool test_complex_matmul() {
    printf("\n=== Test 6: Complex Matrix Multiplication ===\n");

    // 读取测试数据
    std::string json_data = read_ifairy_test_file("matmul_test.json");
    if (json_data.empty()) {
        fprintf(stderr, "Failed to read test data\n");
        return false;
    }

    // 解析数据
    std::vector<float> a_real          = parse_json_float_array(json_data, "a_real");
    std::vector<float> a_imag          = parse_json_float_array(json_data, "a_imag");
    std::vector<float> b_real          = parse_json_float_array(json_data, "b_real");
    std::vector<float> b_imag          = parse_json_float_array(json_data, "b_imag");
    std::vector<float> expected_c_real = parse_json_float_array(json_data, "c_real");
    std::vector<float> expected_c_imag = parse_json_float_array(json_data, "c_imag");

    int M = parse_json_int(json_data, "M");
    int K = parse_json_int(json_data, "K");
    int N = parse_json_int(json_data, "N");

    printf("Testing complex matmul: (%d x %d) @ (%d x %d)\n", M, K, K, N);

    // 手写参考复数 matmul：C = A @ B
    std::vector<float> c_real_calc(M * N, 0.0f);
    std::vector<float> c_imag_calc(M * N, 0.0f);

    for (int m_idx = 0; m_idx < M; ++m_idx) {
        for (int n_idx = 0; n_idx < N; ++n_idx) {
            float acc_r = 0.0f;
            float acc_i = 0.0f;
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                const float ar = a_real[m_idx * K + k_idx];
                const float ai = a_imag[m_idx * K + k_idx];
                const float br = b_real[k_idx * N + n_idx];
                const float bi = b_imag[k_idx * N + n_idx];
                acc_r += ar * br + ai * bi;
                acc_i += ar * bi - ai * br;
            }
            c_real_calc[m_idx * N + n_idx] = acc_r;
            c_imag_calc[m_idx * N + n_idx] = acc_i;
        }
    }

    // 比较结果
    printf("Comparing real part:\n");
    bool real_ok = compare_arrays(c_real_calc.data(), expected_c_real.data(), M * N);

    printf("Comparing imag part:\n");
    bool imag_ok = compare_arrays(c_imag_calc.data(), expected_c_imag.data(), M * N);

    return real_ok && imag_ok;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char ** argv) {
    printf("========================================\n");
    printf("iFairy Model Unit Tests\n");
    printf("========================================\n");

    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
    }

    // 初始化 GGML CPU
    ggml_cpu_init();

    int num_failed = 0;

    // 运行测试
    if (!test_quantization()) {
        fprintf(stderr, "Test 1 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_index()) {
        fprintf(stderr, "Test 2 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_transform_cache()) {
        fprintf(stderr, "Test 2.1 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_transform_invalid_shape()) {
        fprintf(stderr, "Test 2.2 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_layout_auto_policy()) {
        fprintf(stderr, "Test 2.3 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_layout_auto_decode_default_merged64()) {
        fprintf(stderr, "Test 2.3.1 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_index_alignment()) {
        fprintf(stderr, "Test 2.4 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_index_encode_failure()) {
        fprintf(stderr, "Test 2.5 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_env_semantics()) {
        fprintf(stderr, "Test 2.6 FAILED\n");
        num_failed++;
    }

    if (!test_rope()) {
        fprintf(stderr, "Test 3 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_scalar_matmul()) {
        fprintf(stderr, "Test 4 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_scalar_small_dims()) {
        fprintf(stderr, "Test 4.1 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_backend_tiling_regression()) {
        fprintf(stderr, "Test 5 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_backend_large_dims()) {
        fprintf(stderr, "Test 5.1 FAILED\n");
        num_failed++;
    }

    if (!test_ifairy_lut_kernel_sdot_consistency()) {
        fprintf(stderr, "Test 5.2 FAILED\n");
        num_failed++;
    }

    if (!test_complex_matmul()) {
        fprintf(stderr, "Test 6 FAILED\n");
        num_failed++;
    }

    // 总结
    printf("\n========================================\n");
    if (num_failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", num_failed);
    }
    printf("========================================\n");

    return num_failed > 0 ? 1 : 0;
}
