#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

constexpr int offset_has_kv      = 1000;
constexpr int offset_has_tensors = 2000;
constexpr int offset_has_data    = 3000;

enum handcrafted_file_type {
    HANDCRAFTED_HEADER_BAD_MAGIC           =  10,
    HANDCRAFTED_HEADER_BAD_VERSION_1       =  20,
    HANDCRAFTED_HEADER_BAD_VERSION_FUTURE  =  30,
    HANDCRAFTED_HEADER_BAD_N_TENSORS       =  40,
    HANDCRAFTED_HEADER_BAD_N_KV            =  50,
    HANDCRAFTED_HEADER_EMPTY               = 800,

    HANDCRAFTED_KV_BAD_KEY_SIZE            =  10 + offset_has_kv,
    HANDCRAFTED_KV_BAD_TYPE                =  20 + offset_has_kv,
    // HANDCRAFTED_KV_BAD_VALUE_SIZE          =  30 + offset_has_kv, // removed because it can result in allocations > 1 TB (default sanitizer limit)
    HANDCRAFTED_KV_DUPLICATE_KEY           =  40 + offset_has_kv,
    HANDCRAFTED_KV_BAD_ALIGN               =  50 + offset_has_kv,
    HANDCRAFTED_KV_SUCCESS                 = 800 + offset_has_kv,

    HANDCRAFTED_TENSORS_BAD_NAME_SIZE      =  10 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_N_DIMS         =  20 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_SHAPE          =  30 + offset_has_tensors,
    HANDCRAFTED_TENSORS_NE_TOO_BIG         =  40 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_TYPE           =  50 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_OFFSET         =  60 + offset_has_tensors,
    HANDCRAFTED_TENSORS_DUPLICATE_NAME     =  70 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_ALIGN          =  75 + offset_has_tensors,
    HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN =  80 + offset_has_tensors,
    HANDCRAFTED_TENSORS_SUCCESS            = 800 + offset_has_tensors,
    HANDCRAFTED_TENSORS_CUSTOM_ALIGN       = 810 + offset_has_tensors,

    HANDCRAFTED_DATA_NOT_ENOUGH_DATA       =  10 + offset_has_data,
    HANDCRAFTED_DATA_BAD_ALIGN             =  15 + offset_has_data,
    HANDCRAFTED_DATA_INCONSISTENT_ALIGN    =  20 + offset_has_data,
    HANDCRAFTED_DATA_SUCCESS               = 800 + offset_has_data,
    HANDCRAFTED_DATA_CUSTOM_ALIGN          = 810 + offset_has_data,
};

std::string handcrafted_file_type_name(const enum handcrafted_file_type hft) {
    switch (hft) {
        case HANDCRAFTED_HEADER_BAD_MAGIC:           return "HEADER_BAD_MAGIC";
        case HANDCRAFTED_HEADER_BAD_VERSION_1:       return "HEADER_BAD_VERSION_1";
        case HANDCRAFTED_HEADER_BAD_VERSION_FUTURE:  return "HEADER_BAD_VERSION_FUTURE";
        case HANDCRAFTED_HEADER_BAD_N_KV:            return "HEADER_BAD_N_KV";
        case HANDCRAFTED_HEADER_BAD_N_TENSORS:       return "HEADER_BAD_N_TENSORS";
        case HANDCRAFTED_HEADER_EMPTY:               return "HEADER_EMPTY";

        case HANDCRAFTED_KV_BAD_KEY_SIZE:            return "KV_BAD_KEY_SIZE";
        case HANDCRAFTED_KV_BAD_TYPE:                return "KV_BAD_TYPE";
        case HANDCRAFTED_KV_DUPLICATE_KEY:           return "KV_DUPLICATE_KEY";
        case HANDCRAFTED_KV_BAD_ALIGN:               return "KV_BAD_ALIGN";
        case HANDCRAFTED_KV_SUCCESS:                 return "KV_RANDOM_KV";

        case HANDCRAFTED_TENSORS_BAD_NAME_SIZE:      return "TENSORS_BAD_NAME_SIZE";
        case HANDCRAFTED_TENSORS_BAD_N_DIMS:         return "TENSORS_BAD_N_DIMS";
        case HANDCRAFTED_TENSORS_BAD_SHAPE:          return "TENSORS_BAD_SHAPE";
        case HANDCRAFTED_TENSORS_NE_TOO_BIG:         return "TENSORS_NE_TOO_BIG";
        case HANDCRAFTED_TENSORS_BAD_TYPE:           return "TENSORS_BAD_TYPE";
        case HANDCRAFTED_TENSORS_BAD_OFFSET:         return "TENSORS_BAD_OFFSET";
        case HANDCRAFTED_TENSORS_DUPLICATE_NAME:     return "TENSORS_DUPLICATE_NAME";
        case HANDCRAFTED_TENSORS_BAD_ALIGN:          return "TENSORS_BAD_ALIGN";
        case HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN: return "TENSORS_INCONSISTENT_ALIGN";
        case HANDCRAFTED_TENSORS_SUCCESS:            return "TENSORS_SUCCESS";
        case HANDCRAFTED_TENSORS_CUSTOM_ALIGN:       return "TENSORS_CUSTOM_ALIGN";

        case HANDCRAFTED_DATA_NOT_ENOUGH_DATA:       return "DATA_NOT_ENOUGH_DATA";
        case HANDCRAFTED_DATA_BAD_ALIGN:             return "DATA_BAD_ALIGN";
        case HANDCRAFTED_DATA_INCONSISTENT_ALIGN:    return "DATA_INCONSISTENT_ALIGN";
        case HANDCRAFTED_DATA_SUCCESS:               return "DATA_SUCCESS";
        case HANDCRAFTED_DATA_CUSTOM_ALIGN:          return "DATA_CUSTOM_ALIGN";
    }
    GGML_ABORT("fatal error");
}

static bool expect_context_not_null(const enum handcrafted_file_type hft) {
    if (hft < offset_has_kv) {
        return hft >= HANDCRAFTED_HEADER_EMPTY;
    }
    if (hft < offset_has_tensors) {
        return hft >= HANDCRAFTED_KV_SUCCESS;
    }
    if (hft < offset_has_data) {
        return hft >= HANDCRAFTED_TENSORS_SUCCESS;
    }
    return hft >= HANDCRAFTED_DATA_SUCCESS;
}

typedef std::pair<enum ggml_type, std::array<int64_t, GGML_MAX_DIMS>> tensor_config_t;

std::vector<tensor_config_t> get_tensor_configs(std::mt19937 & rng) {
    std::vector<tensor_config_t> tensor_configs;
    tensor_configs.reserve(100);

    for (int i = 0; i < 100; ++i) {
        const enum ggml_type type = ggml_type(rng() % GGML_TYPE_COUNT);
        if (ggml_type_size(type) == 0) {
            continue;
        }

        std::array<int64_t, GGML_MAX_DIMS> shape = {1, 1, 1, 1};
        shape[0] = (1 + rng() % 10) * ggml_blck_size(type);
        const int n_dims = 1 + rng() % GGML_MAX_DIMS;
        for (int i = 1; i < n_dims; ++i) {
            shape[i] = 1 + rng() % 10;
        }

        tensor_configs.push_back(std::make_pair(type, shape));
    }

    return tensor_configs;
}

std::vector<std::pair<enum gguf_type, enum gguf_type>> get_kv_types(std::mt19937 rng) {
    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types;
    kv_types.reserve(100);

    for (int i = 0; i < 100; ++i) {
        const gguf_type type = gguf_type(rng() % GGUF_TYPE_COUNT);

        if (type == GGUF_TYPE_ARRAY) {
            const gguf_type type_arr = gguf_type(rng() % GGUF_TYPE_COUNT);
            if (type_arr == GGUF_TYPE_ARRAY) {
                continue;
            }
            kv_types.push_back(std::make_pair(type, type_arr));
            continue;
        }

        kv_types.push_back(std::make_pair(type, gguf_type(-1)));
    }
    std::shuffle(kv_types.begin(), kv_types.end(), rng);

    return kv_types;
}

template <typename T>
static void helper_write(FILE * file, const T & val) {
    GGML_ASSERT(fwrite(&val, 1, sizeof(val), file) == sizeof(val));
}

static void helper_write(FILE * file, const void * data, const size_t nbytes) {
    GGML_ASSERT(fwrite(data, 1, nbytes, file) == nbytes);
}

static FILE * get_handcrafted_file(const unsigned int seed, const enum handcrafted_file_type hft, const int extra_bytes = 0) {
    FILE * file = tmpfile();

    if (!file) {
        return file;
    }

    std::mt19937 rng(seed);
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;

    if (hft == HANDCRAFTED_HEADER_BAD_MAGIC) {
        const char bad_magic[4] = {'F', 'U', 'G', 'G'};
        helper_write(file, bad_magic, sizeof(bad_magic));
    } else {
        helper_write(file, GGUF_MAGIC, 4);
    }

    if (hft == HANDCRAFTED_HEADER_BAD_VERSION_1) {
        const uint32_t version = 1;
        helper_write(file, version);
    } else if (hft == HANDCRAFTED_HEADER_BAD_VERSION_FUTURE) {
        const uint32_t version = GGUF_VERSION + 1;
        helper_write(file, version);
    } else {
        const uint32_t version = GGUF_VERSION;
        helper_write(file, version);
    }

    std::vector<tensor_config_t> tensor_configs;
    if (hft >= offset_has_tensors) {
        tensor_configs = get_tensor_configs(rng);
    }

    if (hft == HANDCRAFTED_HEADER_BAD_N_TENSORS) {
        const uint64_t n_tensors = -1;
        helper_write(file, n_tensors);
    } else {
        const uint64_t n_tensors = tensor_configs.size();
        helper_write(file, n_tensors);
    }

    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types;
    if (hft >= offset_has_kv) {
        kv_types = get_kv_types(rng);
    }
    {
        uint64_t n_kv = kv_types.size();
        if (hft == HANDCRAFTED_KV_BAD_ALIGN      ||
            hft == HANDCRAFTED_TENSORS_BAD_ALIGN || hft == HANDCRAFTED_TENSORS_CUSTOM_ALIGN ||
            hft == HANDCRAFTED_DATA_BAD_ALIGN    || hft == HANDCRAFTED_DATA_CUSTOM_ALIGN) {

            n_kv += 1;
        } else if (hft == HANDCRAFTED_HEADER_BAD_N_KV) {
            n_kv = -1;
        }
        helper_write(file, n_kv);
    }

    if (hft < offset_has_kv) {
        while (ftell(file) % alignment != 0) {
            const char pad = 0;
            helper_write(file, pad);
        }

        for (int i = 0; i < extra_bytes; ++i) {
            const char tmp = 0;
            helper_write(file, tmp);
        }
        rewind(file);
        return file;
    }

    for (int i = 0; i < int(kv_types.size()); ++i) {
        const enum gguf_type type     = gguf_type(hft == HANDCRAFTED_KV_BAD_TYPE ? GGUF_TYPE_COUNT : kv_types[i].first);
        const enum gguf_type type_arr = gguf_type(hft == HANDCRAFTED_KV_BAD_TYPE ? GGUF_TYPE_COUNT : kv_types[i].second);

        const std::string key = "my_key_" + std::to_string((hft == HANDCRAFTED_KV_DUPLICATE_KEY ? i/2 : i));

        if (hft == HANDCRAFTED_KV_BAD_KEY_SIZE) {
            const uint64_t n = -1;
            helper_write(file, n);
        } else {
            const uint64_t n = key.length();
            helper_write(file, n);
        }
        helper_write(file, key.data(), key.length());

        {
            const int32_t type32 = int32_t(type);
            helper_write(file, type32);
        }

        uint32_t data[16];
        for (int j = 0; j < 16; ++j) {
            data[j] = rng();
            if (type == GGUF_TYPE_STRING || type_arr == GGUF_TYPE_STRING) {
                data[j] |= 0x01010101; // avoid random null-termination of string
            }
        }

        if (type == GGUF_TYPE_STRING) {
            const uint64_t n = rng() % sizeof(data);
            helper_write(file, n);
            helper_write(file, data, n);
            continue;
        }

        if (type == GGUF_TYPE_ARRAY) {
            {
                const int32_t type32 = int32_t(type_arr);
                helper_write(file, type32);
            }
            if (type_arr == GGUF_TYPE_STRING) {
                const uint64_t nstr = rng() % (16 + 1);
                helper_write(file, nstr);
                for (uint64_t istr = 0; istr < nstr; ++istr) {
                    const uint64_t n = rng() % (sizeof(uint32_t) + 1);
                    helper_write(file, n);
                    helper_write(file, &data[istr], n);
                }
                continue;
            }
            const size_t type_size = gguf_type_size(type_arr);
            const uint64_t n = (rng() % sizeof(data)) / type_size;
            helper_write(file, n);
            helper_write(file, &data, n*type_size);
            continue;
        }

        helper_write(file, data, hft == HANDCRAFTED_KV_BAD_TYPE ? 1 : gguf_type_size(type));
    }

    if (hft == HANDCRAFTED_KV_BAD_ALIGN      ||
        hft == HANDCRAFTED_TENSORS_BAD_ALIGN || hft == HANDCRAFTED_TENSORS_CUSTOM_ALIGN ||
        hft == HANDCRAFTED_DATA_BAD_ALIGN    || hft == HANDCRAFTED_DATA_CUSTOM_ALIGN) {

        const uint64_t n = strlen(GGUF_KEY_GENERAL_ALIGNMENT);
        helper_write(file, n);
        helper_write(file, GGUF_KEY_GENERAL_ALIGNMENT, n);

        const int32_t type = gguf_type(GGUF_TYPE_UINT32);
        helper_write(file, type);

        alignment = expect_context_not_null(hft) ? 1 : 13;
        helper_write(file, alignment);
    }

    if (hft < offset_has_tensors) {
        while (ftell(file) % alignment != 0) {
            const char pad = 0;
            helper_write(file, pad);
        }

        for (int i = 0; i < extra_bytes; ++i) {
            const char tmp = 0;
            helper_write(file, tmp);
        }
        rewind(file);
        return file;
    }

    if (hft == HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN || hft == HANDCRAFTED_DATA_INCONSISTENT_ALIGN) {
        alignment = 1;
    }

    uint64_t offset = 0;
    for (int i = 0; i < int(tensor_configs.size()); ++i) {
        const ggml_type                          type  = tensor_configs[i].first;
        const std::array<int64_t, GGML_MAX_DIMS> shape = tensor_configs[i].second;

        std::string name = "my_tensor";
        if (hft != HANDCRAFTED_TENSORS_DUPLICATE_NAME) {
            name += "_" + std::to_string(i);
        }
        if (hft == HANDCRAFTED_TENSORS_BAD_NAME_SIZE) {
            name += "_with_a_very_long_name_which_is_longer_than_what_is_allowed_for_ggml_tensors";
            GGML_ASSERT(name.length() >= GGML_MAX_NAME);
        }
        {
            const uint64_t n = name.length();
            helper_write(file, n);
        }
        helper_write(file, name.data(), name.length());

        uint32_t n_dims = hft == HANDCRAFTED_TENSORS_NE_TOO_BIG ? 2 : 1;
        for (int i = GGML_MAX_DIMS-1; i >= 1; --i) {
            if (shape[i] != 1) {
                n_dims = i + 1;
                break;
            }
        }
        if (hft == HANDCRAFTED_TENSORS_BAD_N_DIMS) {
            const uint32_t n_dims_bad = GGML_MAX_DIMS + 1;
            helper_write(file, n_dims_bad);
        } else {
            helper_write(file, n_dims);
        }

        if (hft == HANDCRAFTED_TENSORS_BAD_SHAPE) {
            for (uint32_t j = 0; j < n_dims; ++j) {
                const int64_t bad_dim = -1;
                helper_write(file, bad_dim);
            }
        } else if (hft == HANDCRAFTED_TENSORS_NE_TOO_BIG){
            for (uint32_t j = 0; j < n_dims; ++j) {
                const int64_t big_dim = 4*int64_t(INT32_MAX);
                helper_write(file, big_dim);
            }
        } else {
            helper_write(file, shape.data(), n_dims*sizeof(int64_t));
        }

        {
            const int32_t type32 = hft == HANDCRAFTED_TENSORS_BAD_TYPE ? GGML_TYPE_COUNT : int32_t(type);
            helper_write(file, type32);
        }

        if (hft == HANDCRAFTED_TENSORS_BAD_OFFSET) {
            const uint64_t bad_offset = -1;
            helper_write(file, bad_offset);
        } else {
            helper_write(file, offset);
        }

        int64_t ne = shape[0];
        for (uint32_t i = 1; i < n_dims; ++i) {
            ne *= shape[i];
        }
        offset += GGML_PAD(ggml_row_size(type, ne), alignment);
    }

    while (ftell(file) % alignment != 0) {
        const char pad = 0;
        helper_write(file, pad);
    }

    if (hft >= offset_has_data) {
        rng.seed(seed + 1);
        uint64_t nbytes = offset;
        if (hft == HANDCRAFTED_DATA_NOT_ENOUGH_DATA) {
            nbytes -= 1;
        }
        for (uint64_t i = 0; i < nbytes; ++i) {
            const uint8_t random_byte = i % 256;
            helper_write(file, random_byte);
        }
    }

    for (int i = 0; i < extra_bytes; ++i) {
        const char tmp = 0;
        helper_write(file, tmp);
    }
    rewind(file);
    return file;
}

static bool handcrafted_check_header(const gguf_context * gguf_ctx, const unsigned int seed, const bool has_kv, const bool has_tensors, const bool alignment_defined) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs;
    if (has_tensors) {
        tensor_configs = get_tensor_configs(rng);
    }
    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types;
    if (has_kv) {
        kv_types = get_kv_types(rng);
    }

    bool ok = true;

    if (gguf_get_version(gguf_ctx) != GGUF_VERSION) {
        ok = false;
    }
    if (gguf_get_n_tensors(gguf_ctx) != int(tensor_configs.size())) {
        ok = false;
    }
    if (gguf_get_n_kv(gguf_ctx) != int(alignment_defined ? kv_types.size() + 1 : kv_types.size())) {
        ok = false;
    }

    return ok;
}

static bool handcrafted_check_kv(const gguf_context * gguf_ctx, const unsigned int seed, const bool has_tensors, const bool alignment_defined) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs;
    if (has_tensors) {
        tensor_configs = get_tensor_configs(rng);
    }

    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types = get_kv_types(rng);

    bool ok = true;

    for (int i = 0; i < int(kv_types.size()); ++i) {
        const enum gguf_type type     = gguf_type(kv_types[i].first);
        const enum gguf_type type_arr = gguf_type(kv_types[i].second);

        const std::string key = "my_key_" + std::to_string(i);

        uint32_t data[16];
        for (int j = 0; j < 16; ++j) {
            data[j] = rng();
            if (type == GGUF_TYPE_STRING || type_arr == GGUF_TYPE_STRING) {
                data[j] |= 0x01010101; // avoid random null-termination of string
            }
        }

        const char * data8 = reinterpret_cast<const char *>(data);
        const int id = gguf_find_key(gguf_ctx, key.c_str());

        if (type == GGUF_TYPE_STRING) {
            const char * str = gguf_get_val_str(gguf_ctx, id);
            const uint64_t n = strlen(str);
            const uint64_t n_expected = rng() % sizeof(data);
            if (n != n_expected) {
                ok = false;
                continue;
            }
            if (!std::equal(str, str + n, data8)) {
                ok = false;
            }
            continue;
        }

        if (type == GGUF_TYPE_ARRAY) {
            const size_t type_size = gguf_type_size(type_arr);
            const uint64_t arr_n = gguf_get_arr_n(gguf_ctx, id);

            if (type_arr == GGUF_TYPE_STRING) {
                const uint64_t nstr_expected = rng() % (16 + 1);
                if (arr_n != nstr_expected) {
                    ok = false;
                    continue;
                }
                for (uint64_t istr = 0; istr < nstr_expected; ++istr) {
                    const char * str = gguf_get_arr_str(gguf_ctx, id, istr);
                    const uint64_t n = strlen(str);
                    const uint64_t n_expected = rng() % (sizeof(uint32_t) + 1);

                    if (n != n_expected) {
                        ok = false;
                        continue;
                    }
                    const char * str_expected = reinterpret_cast<const char *>(&data[istr]);
                    if (strncmp(str, str_expected, n) != 0) {
                        ok = false;
                        continue;
                    }
                }
                continue;
            }

            const uint64_t arr_n_expected = (rng() % sizeof(data)) / type_size;
            if (arr_n != arr_n_expected) {
                ok = false;
                continue;
            }

            const char * data_gguf = reinterpret_cast<const char *>(gguf_get_arr_data(gguf_ctx, id));

            if (type_arr == GGUF_TYPE_BOOL) {
                for (size_t arr_i = 0; arr_i < arr_n; ++arr_i) {
                    if (bool(data8[arr_i]) != bool(data_gguf[arr_i])) {
                        ok = false;
                    }
                }
                continue;
            }

            if (!std::equal(data8, data8 + arr_n*type_size, data_gguf)) {
                ok = false;
            }
            continue;
        }

        const char * data_gguf = reinterpret_cast<const char *>(gguf_get_val_data(gguf_ctx, id));

        if (type == GGUF_TYPE_BOOL) {
            if (bool(*data8) != bool(*data_gguf)) {
                ok = false;
            }
            continue;
        }

        if (!std::equal(data8, data8 + gguf_type_size(type), data_gguf)) {
            ok = false;
        }
    }

    const uint32_t expected_alignment = alignment_defined ? 1 : GGUF_DEFAULT_ALIGNMENT;
    if (gguf_get_alignment(gguf_ctx) != expected_alignment) {
        ok = false;
    }

    return ok;
}

static bool handcrafted_check_tensors(const gguf_context * gguf_ctx, const unsigned int seed) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs = get_tensor_configs(rng);

    // Call get_kv_types to get the same RNG state:
    get_kv_types(rng);

    bool ok = true;

    const int id_alignment = gguf_find_key(gguf_ctx, GGUF_KEY_GENERAL_ALIGNMENT);
    const uint32_t alignment = id_alignment >= 0 ? gguf_get_val_u32(gguf_ctx, id_alignment) : GGUF_DEFAULT_ALIGNMENT;

    uint64_t expected_offset = 0;
    for (int i = 0; i < int(tensor_configs.size()); ++i) {
        const ggml_type                          type  = tensor_configs[i].first;
        const std::array<int64_t, GGML_MAX_DIMS> shape = tensor_configs[i].second;

        const std::string name = "my_tensor_" + std::to_string(i);
        const int id = gguf_find_tensor(gguf_ctx, name.c_str());

        if (id >= 0) {
            if (std::string(gguf_get_tensor_name(gguf_ctx, id)) != name) {
                ok = false;
            }

            if (gguf_get_tensor_type(gguf_ctx, id) != type) {
                ok = false;
            }
        } else {
            ok = false;
            continue;
        }

        const size_t offset = gguf_get_tensor_offset(gguf_ctx, id);

        if (offset != expected_offset) {
            ok = false;
        }

        int64_t ne = shape[0];
        for (size_t j = 1; j < GGML_MAX_DIMS; ++j) {
            ne *= shape[j];
        }
        expected_offset += GGML_PAD(ggml_row_size(type, ne), alignment);
    }

    return ok;
}

static bool handcrafted_check_tensor_data(const gguf_context * gguf_ctx, const unsigned int seed, FILE * file) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs = get_tensor_configs(rng);

    bool ok = true;

    const uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;

    for (int i = 0; i < int(tensor_configs.size()); ++i) {
        const ggml_type                          type  = tensor_configs[i].first;
        const std::array<int64_t, GGML_MAX_DIMS> shape = tensor_configs[i].second;

        int64_t ne = shape[0];
        for (size_t j = 1; j < GGML_MAX_DIMS; ++j) {
            ne *= shape[j];
        }
        const size_t size = ggml_row_size(type, ne);

        const std::string name = "my_tensor_" + std::to_string(i);
        const size_t offset = gguf_get_tensor_offset(gguf_ctx, gguf_find_tensor(gguf_ctx, name.c_str()));

        std::vector<uint8_t> data(size);
        GGML_ASSERT(fseek(file, gguf_get_data_offset(gguf_ctx) + offset, SEEK_SET) == 0);
        GGML_ASSERT(fread(data.data(), 1, data.size(), file) == data.size());

        for (size_t j = 0; j < size; ++j) {
            const uint8_t expected_byte = (j + offset) % 256;
            if (data[j] != expected_byte) {
                ok = false;
            }
        }
    }

    return ok;
}

static std::pair<int, int> test_handcrafted_file(const unsigned int seed) {
    int npass = 0;
    int ntest = 0;

    const std::vector<handcrafted_file_type> hfts = {
        HANDCRAFTED_HEADER_BAD_MAGIC,
        HANDCRAFTED_HEADER_BAD_VERSION_1,
        HANDCRAFTED_HEADER_BAD_VERSION_FUTURE,
        HANDCRAFTED_HEADER_BAD_N_KV,
        HANDCRAFTED_HEADER_BAD_N_TENSORS,
        HANDCRAFTED_HEADER_EMPTY,

        HANDCRAFTED_KV_BAD_KEY_SIZE,
        HANDCRAFTED_KV_BAD_TYPE,
        HANDCRAFTED_KV_DUPLICATE_KEY,
        HANDCRAFTED_KV_BAD_ALIGN,
        HANDCRAFTED_KV_SUCCESS,

        HANDCRAFTED_TENSORS_BAD_NAME_SIZE,
        HANDCRAFTED_TENSORS_BAD_N_DIMS,
        HANDCRAFTED_TENSORS_BAD_SHAPE,
        HANDCRAFTED_TENSORS_NE_TOO_BIG,
        HANDCRAFTED_TENSORS_BAD_TYPE,
        HANDCRAFTED_TENSORS_BAD_OFFSET,
        HANDCRAFTED_TENSORS_DUPLICATE_NAME,
        HANDCRAFTED_TENSORS_BAD_ALIGN,
        HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN,
        HANDCRAFTED_TENSORS_SUCCESS,
        HANDCRAFTED_TENSORS_CUSTOM_ALIGN,

        HANDCRAFTED_DATA_NOT_ENOUGH_DATA,
        HANDCRAFTED_DATA_BAD_ALIGN,
        HANDCRAFTED_DATA_INCONSISTENT_ALIGN,
        HANDCRAFTED_DATA_SUCCESS,
        HANDCRAFTED_DATA_CUSTOM_ALIGN,
    };

    for (enum handcrafted_file_type hft : hfts) {
        printf("%s: handcrafted_file_type=%s\n", __func__, handcrafted_file_type_name(hft).c_str());
        FILE * file = get_handcrafted_file(seed, hft);

#ifdef _WIN32
        if (!file) {
            printf("%s: failed to create tmpfile(), needs elevated privileges on Windows");
            printf("%s: skipping tests");
            continue;
        }
#else
        GGML_ASSERT(file);
#endif // _WIN32

        struct ggml_context * ctx = nullptr;
        struct gguf_init_params gguf_params = {
            /*no_alloc =*/ false,
            /*ctx      =*/ hft >= offset_has_data ? &ctx : nullptr,
        };

        struct gguf_context * gguf_ctx = gguf_init_from_file_impl(file, gguf_params);

        if (expect_context_not_null(hft)) {
            printf("%s:   - context_not_null: ", __func__);
        } else {
            printf("%s:   - context_null: ", __func__);
        }
        if (bool(gguf_ctx) == expect_context_not_null(hft)) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;

        if (hft >= offset_has_data && !expect_context_not_null(hft)) {
            printf("%s:   - no_dangling_ggml_context_pointer: ", __func__);
            if (ctx) {
                printf("\033[1;31mFAIL\033[0m\n");
            } else {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            }
            ntest++;
        }

        const bool alignment_defined = hft == HANDCRAFTED_TENSORS_CUSTOM_ALIGN || hft == HANDCRAFTED_DATA_CUSTOM_ALIGN;

        if (expect_context_not_null(hft)) {
            printf("%s:   - check_header: ", __func__);
            if (handcrafted_check_header(gguf_ctx, seed, hft >= offset_has_kv, hft >= offset_has_tensors, alignment_defined)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        if (expect_context_not_null(hft) && hft >= offset_has_kv) {
            printf("%s:   - check_kv: ", __func__);
            if (handcrafted_check_kv(gguf_ctx, seed, hft >= offset_has_tensors, alignment_defined)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        if (expect_context_not_null(hft) && hft >= offset_has_tensors) {
            printf("%s:   - check_tensors: ", __func__);
            if (handcrafted_check_tensors(gguf_ctx, seed)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        if (expect_context_not_null(hft) && hft >= offset_has_data) {
            printf("%s:   - check_tensor_data: ", __func__);
            if (handcrafted_check_tensor_data(gguf_ctx, seed, file)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        fclose(file);
        if (gguf_ctx) {
            ggml_free(ctx);
            gguf_free(gguf_ctx);
        }
        printf("\n");
    }


    return std::make_pair(npass, ntest);
}

struct random_gguf_context_result {
    struct gguf_context * gguf_ctx;
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

static struct random_gguf_context_result get_random_gguf_context(ggml_backend_t backend, const unsigned int seed) {
    std::mt19937 rng(seed);

    struct gguf_context * gguf_ctx = gguf_init_empty();

    for (int i = 0; i < 256; ++i) {
        const std::string key = "my_key_" + std::to_string(rng() % 1024);
        const enum gguf_type type = gguf_type(rng() % GGUF_TYPE_COUNT);

        switch (type) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (gguf_ctx, key.c_str(), rng() % (1 <<  7));             break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (gguf_ctx, key.c_str(), rng() % (1 <<  7) - (1 <<  6)); break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (gguf_ctx, key.c_str(), rng() % (1 << 15));             break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (gguf_ctx, key.c_str(), rng() % (1 << 15) - (1 << 14)); break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (gguf_ctx, key.c_str(), rng());                         break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (gguf_ctx, key.c_str(), rng()             - (1 << 30)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (gguf_ctx, key.c_str(), rng() % 1024      - 512);       break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(gguf_ctx, key.c_str(), rng() % 2 == 0);                break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (gguf_ctx, key.c_str(), std::to_string(rng()).c_str()); break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (gguf_ctx, key.c_str(), rng());                         break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (gguf_ctx, key.c_str(), rng()             - (1 << 30)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f32 (gguf_ctx, key.c_str(), rng() % 1024      - 512);       break;
            case GGUF_TYPE_ARRAY: {
                const enum gguf_type type_arr = gguf_type(rng() % GGUF_TYPE_COUNT);
                const uint64_t ne = rng() % 1024;

                switch (type_arr) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16:
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_FLOAT32:
                    case GGUF_TYPE_BOOL:
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64:
                    case GGUF_TYPE_FLOAT64: {
                        const size_t nbytes = ne*gguf_type_size(type_arr);
                        std::vector<uint32_t> random_data((nbytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
                        for (size_t j = 0; j < random_data.size(); ++j) {
                            random_data[j] = rng();
                            if (type_arr == GGUF_TYPE_BOOL) {
                                random_data[j] &= 0x01010101; // the sanitizer complains if booleans are not 0 or 1
                            }
                        }
                        gguf_set_arr_data(gguf_ctx, key.c_str(), type_arr, random_data.data(), ne);
                    } break;
                    case GGUF_TYPE_STRING: {
                        std::vector<std::string>  data_cpp(ne);
                        std::vector<const char *> data_c(ne);
                        for (size_t j = 0; j < data_cpp.size(); ++j) {
                            data_cpp[j] = std::to_string(rng());
                            data_c[j]   = data_cpp[j].c_str();
                        }
                        gguf_set_arr_str(gguf_ctx, key.c_str(), data_c.data(), ne);
                    } break;
                    case GGUF_TYPE_ARRAY: {
                        break; // not supported
                    }
                    case GGUF_TYPE_COUNT:
                    default: {
                        GGML_ABORT("fatal error");
                    } break;
                }
            } break;
            case GGUF_TYPE_COUNT:
            default: {
                GGML_ABORT("fatal error");
            } break;
        }
    }

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 256*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    for (int i = 0; i < 256; ++i) {
        const std::string name = "my_tensor_" + std::to_string(i);
        const enum ggml_type type = ggml_type(rng() % GGML_TYPE_COUNT);
        const size_t type_size = ggml_type_size(type);

        if (type_size == 0) {
            continue;
        }

        const int n_dims = 1 + rng() % GGML_MAX_DIMS;
        int64_t ne[GGML_MAX_DIMS];
        ne[0] = (1 + rng() % 10) * ggml_blck_size(type);
        for (int j = 1; j < n_dims; ++j) {
            ne[j] = 1 + rng() % 10;
        }

        struct ggml_tensor * tensor = ggml_new_tensor(ctx, type, n_dims, ne);
        ggml_set_name(tensor, name.c_str());
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint32_t> random_data((nbytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        for (size_t j = 0; j < random_data.size(); ++j) {
            random_data[j] = rng();
        }
        ggml_backend_tensor_set(t, random_data.data(), 0, nbytes);

        gguf_add_tensor(gguf_ctx, t);
    }

    return {gguf_ctx, ctx, buf};
}

static bool all_kv_in_other(const gguf_context * ctx, const gguf_context * other) {
    bool ok = true;

    const int n_kv = gguf_get_n_kv(ctx);
    for (int id = 0; id < n_kv; ++id) {
        const char * name = gguf_get_key(ctx, id);

        const int idx_other = gguf_find_key(other, name);
        if (idx_other < 0) {
            ok = false;
            continue;
        }

        const gguf_type type = gguf_get_kv_type(ctx, id);
        if (type != gguf_get_kv_type(other, idx_other)) {
            ok = false;
            continue;
        }

        if (type == GGUF_TYPE_ARRAY) {
            const int arr_n = gguf_get_arr_n(ctx, id);
            if (arr_n != gguf_get_arr_n(other, idx_other)) {
                ok = false;
                continue;
            }

            const gguf_type type_arr = gguf_get_arr_type(ctx, id);
            if (type_arr != gguf_get_arr_type(other, idx_other)) {
                ok = false;
                continue;
            }

            if (type_arr == GGUF_TYPE_BOOL) {
                const int8_t * data       = reinterpret_cast<const int8_t *>(gguf_get_arr_data(ctx,   id));
                const int8_t * data_other = reinterpret_cast<const int8_t *>(gguf_get_arr_data(other, idx_other));
                for (int arr_i = 0; arr_i < arr_n; ++arr_i) {
                    if (bool(data[arr_i]) != bool(data_other[arr_i])) {
                        ok = false;
                    }
                }
                continue;
            }

            if (type_arr == GGUF_TYPE_STRING) {
                for (int arr_i = 0; arr_i < arr_n; ++arr_i) {
                    const std::string str       = gguf_get_arr_str(ctx,   id,       arr_i);
                    const std::string str_other = gguf_get_arr_str(other, idx_other, arr_i);
                    if (str != str_other) {
                        ok = false;
                    }
                }
                continue;
            }

            const int8_t * data       = reinterpret_cast<const int8_t *>(gguf_get_arr_data(ctx,   id));
            const int8_t * data_other = reinterpret_cast<const int8_t *>(gguf_get_arr_data(other, idx_other));
            if (!std::equal(data, data + arr_n*gguf_type_size(type_arr), data_other)) {
                ok = false;
            }
            continue;
        }

        if (type == GGUF_TYPE_STRING) {
            const std::string str       = gguf_get_val_str(ctx,   id);
            const std::string str_other = gguf_get_val_str(other, idx_other);
            if (str != str_other) {
                ok = false;
            }
            continue;
        }

        const char * data       = reinterpret_cast<const char *>(gguf_get_val_data(ctx,   id));
        const char * data_other = reinterpret_cast<const char *>(gguf_get_val_data(other, idx_other));
        if (!std::equal(data, data + gguf_type_size(type), data_other)) {
            ok = false;
        }
    }

    return ok;
}

static bool all_tensors_in_other(const gguf_context * ctx, const gguf_context * other) {
    bool ok = true;

    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int id = 0; id < n_tensors; ++id) {
        const std::string name = gguf_get_tensor_name(ctx, id);

        const int idx_other = gguf_find_tensor(other, name.c_str());
        if (id != idx_other) {
            ok = false;
            if (idx_other < 0) {
                continue;
            }
        }

        const ggml_type type = gguf_get_tensor_type(ctx, id);
        if (type != gguf_get_tensor_type(other, id)) {
            ok = false;
        }

        const size_t offset = gguf_get_tensor_offset(ctx, id);
        if (offset != gguf_get_tensor_offset(other, id)) {
            ok = false;
        }
    }

    return ok;
}

static bool same_tensor_data(const struct ggml_context * orig, const struct ggml_context * read) {
    bool ok = true;

    struct ggml_tensor * t_orig = ggml_get_first_tensor(orig);
    struct ggml_tensor * t_read = ggml_get_first_tensor(read);
    while (t_orig) {
        if (!t_read) {
            ok = false;
            break;
        }

        const size_t nbytes = ggml_nbytes(t_orig);
        if (ggml_nbytes(t_read) != nbytes) {
            ok = false;
            break;
        }
        std::vector<char> data_orig(nbytes);
        ggml_backend_tensor_get(t_orig, data_orig.data(), 0, nbytes);
        if (!std::equal(data_orig.data(), data_orig.data() + nbytes, reinterpret_cast<const char *>(t_read->data))) {
            ok = false;
        }

        t_orig = ggml_get_next_tensor(orig, t_orig);
        t_read = ggml_get_next_tensor(orig, t_read);
    }
    if (t_read) {
        ok = false;
    }

    return true;
}

static std::pair<int, int> test_roundtrip(ggml_backend_dev_t dev, const unsigned int seed, const bool only_meta) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("%s: device=%s, backend=%s, only_meta=%s\n",
        __func__, ggml_backend_dev_description(dev), ggml_backend_name(backend), only_meta ? "yes" : "no");

    int npass = 0;
    int ntest = 0;

    struct gguf_context * gguf_ctx_0;
    struct ggml_context * ctx_0;
    ggml_backend_buffer_t bbuf;
    {
        struct random_gguf_context_result result = get_random_gguf_context(backend, seed);
        gguf_ctx_0 = result.gguf_ctx;
        ctx_0      = result.ctx;
        bbuf       = result.buffer;
    }

    FILE * file = tmpfile();

#ifdef _WIN32
    if (!file) {
        printf("%s: failed to create tmpfile(), needs elevated privileges on Windows");
        printf("%s: skipping tests");
        return std::make_pair(0, 0);
    }
#else
    GGML_ASSERT(file);
#endif // _WIN32

    {
        std::vector<int8_t> buf;
        gguf_write_to_buf(gguf_ctx_0, buf, only_meta);
        GGML_ASSERT(fwrite(buf.data(), 1, buf.size(), file) == buf.size());
        rewind(file);
    }

    struct ggml_context * ctx_1 = nullptr;
    struct gguf_init_params gguf_params = {
        /*no_alloc =*/ false,
        /*ctx      =*/ only_meta ? nullptr : &ctx_1,
    };
    struct gguf_context * gguf_ctx_1 = gguf_init_from_file_impl(file, gguf_params);

    printf("%s: same_version: ", __func__);
    if (gguf_get_version(gguf_ctx_0) == gguf_get_version(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: same_n_kv: ", __func__);
    if (gguf_get_n_kv(gguf_ctx_0) == gguf_get_n_kv(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: same_n_tensors: ", __func__);
    if (gguf_get_n_tensors(gguf_ctx_0) == gguf_get_n_tensors(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_orig_kv_in_read: ", __func__);
    if (all_kv_in_other(gguf_ctx_0, gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_read_kv_in_orig: ", __func__);
    if (all_kv_in_other(gguf_ctx_1, gguf_ctx_0)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_orig_tensors_in_read: ", __func__);
    if (all_tensors_in_other(gguf_ctx_0, gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_read_tensors_in_orig: ", __func__);
    if (all_tensors_in_other(gguf_ctx_1, gguf_ctx_0)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    if (!only_meta) {
        printf("%s: same_tensor_data: ", __func__);
        if (same_tensor_data(ctx_0, ctx_1)) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;
    }

    ggml_backend_buffer_free(bbuf);
    ggml_free(ctx_0);
    ggml_free(ctx_1);
    gguf_free(gguf_ctx_0);
    gguf_free(gguf_ctx_1);
    ggml_backend_free(backend);
    fclose(file);

    printf("\n");
    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_gguf_set_kv(ggml_backend_dev_t dev, const unsigned int seed) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("%s: device=%s, backend=%s\n", __func__, ggml_backend_dev_description(dev), ggml_backend_name(backend));

    int npass = 0;
    int ntest = 0;

    struct gguf_context * gguf_ctx_0;
    struct ggml_context * ctx_0;
    ggml_backend_buffer_t bbuf_0;
    {
        struct random_gguf_context_result result = get_random_gguf_context(backend, seed);
        gguf_ctx_0 = result.gguf_ctx;
        ctx_0      = result.ctx;
        bbuf_0     = result.buffer;
    }

    struct gguf_context * gguf_ctx_1;
    struct ggml_context * ctx_1;
    ggml_backend_buffer_t bbuf_1;
    {
        struct random_gguf_context_result result = get_random_gguf_context(backend, seed + 1);
        gguf_ctx_1 = result.gguf_ctx;
        ctx_1      = result.ctx;
        bbuf_1     = result.buffer;
    }

    struct gguf_context * gguf_ctx_2 = gguf_init_empty();

    gguf_set_kv(gguf_ctx_1, gguf_ctx_0);
    gguf_set_kv(gguf_ctx_2, gguf_ctx_0);

    printf("%s: same_n_kv: ", __func__);
    if (gguf_get_n_kv(gguf_ctx_0) == gguf_get_n_kv(gguf_ctx_2)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_kv_0_in_1: ", __func__);
    if (all_kv_in_other(gguf_ctx_0, gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_kv_0_in_2: ", __func__);
    if (all_kv_in_other(gguf_ctx_0, gguf_ctx_2)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    gguf_set_kv(gguf_ctx_0, gguf_ctx_1);

    printf("%s: same_n_kv_after_double_copy: ", __func__);
    if (gguf_get_n_kv(gguf_ctx_0) == gguf_get_n_kv(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_kv_1_in_0_after_double_copy: ", __func__);
    if (all_kv_in_other(gguf_ctx_1, gguf_ctx_0)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    ggml_backend_buffer_free(bbuf_0);
    ggml_backend_buffer_free(bbuf_1);
    ggml_free(ctx_0);
    ggml_free(ctx_1);
    gguf_free(gguf_ctx_0);
    gguf_free(gguf_ctx_1);
    gguf_free(gguf_ctx_2);
    ggml_backend_free(backend);

    printf("\n");
    return std::make_pair(npass, ntest);
}

static void print_usage() {
    printf("usage: test-gguf [seed]\n");
    printf("  if no seed is unspecified then a random seed is used\n");
}

int main(int argc, char ** argv) {
    if (argc > 2) {
        print_usage();
        return 1;
    }

    std::random_device rd;
    const unsigned int seed = argc < 2 ? rd() : std::stoi(argv[1]);

    // Initialize ggml backends early so the prints aren't interleaved with the test results:
    ggml_backend_dev_count();
    fprintf(stderr, "\n");

    int npass = 0;
    int ntest = 0;
    {
        std::pair<int, int> result = test_handcrafted_file(seed);
        npass += result.first;
        ntest += result.second;
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        for (bool only_meta : {true, false}) {
            std::pair<int, int> result = test_roundtrip(dev, seed, only_meta);
            npass += result.first;
            ntest += result.second;
        }

        {
            std::pair<int, int> result = test_gguf_set_kv(dev, seed);
            npass += result.first;
            ntest += result.second;
        }
    }

    printf("%d/%d tests passed\n", npass, ntest);
    if (npass != ntest) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }
    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
