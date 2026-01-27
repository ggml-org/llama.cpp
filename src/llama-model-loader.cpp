#include "llama-model-loader.h"

#include "ggml.h"
#include "gguf.h"

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <future>

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2";
        case GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:         return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:      return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:     return "BF16";
        case LLAMA_FTYPE_MOSTLY_Q4_0:     return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:     return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:     return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:     return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0:     return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return "MXFP4 MoE";
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:   return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:   return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:   return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:   return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:   return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:   return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:   return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:   return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:     return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_TQ1_0:    return "TQ1_0 - 1.69 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_TQ2_0:    return "TQ2_0 - 2.06 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:  return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:   return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:    return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:    return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:   return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:    return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:    return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:   return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";

        default: return "unknown, may not work";
    }
}

// return a list of splits for a given path
// for example, given "<name>-00002-of-00004.gguf", returns list of all 4 splits
static std::vector<std::string> llama_get_list_splits(const std::string & path, const int idx, const int n_split) {
    std::vector<std::string> paths;
    std::string split_prefix;
    std::vector<char> buf(llama_path_max(), 0);

    {
        int ret = llama_split_prefix(buf.data(), buf.size(), path.c_str(), idx, n_split);
        if (!ret) {
            throw std::runtime_error(format("invalid split file name: %s", path.c_str()));
        }
        split_prefix = std::string(buf.data(), ret);
    }

    if (split_prefix.empty()) {
        throw std::runtime_error(format("invalid split file: %s", path.c_str()));
    }

    for (int idx = 0; idx < n_split; ++idx) {
        int ret = llama_split_path(buf.data(), buf.size(), split_prefix.c_str(), idx, n_split);
        paths.push_back(std::string(buf.data(), ret));
    }

    return paths;
}

namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx, k);
            return ArrayInfo {
                arr_type,
                size_t(gguf_get_arr_n(ctx, k)),
                arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(const std::string & key, T & result, bool required) {
        const int kid = gguf_find_key(meta.get(), key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(enum llm_kv kid, T & result, bool required) {
        return get_arr_n(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr_n(enum llm_kv kid, uint32_t & result, bool required);

    template<typename T>
    bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, bool required) {
        const gguf_context * ctx = meta.get();
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        switch (arr_info.gt) {
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same<T,     int32_t>::value) ||
                                                (std::is_same<T,    uint32_t>::value)); break;
            case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T,       float>::value)); break;
            case GGUF_TYPE_STRING:  GGML_ASSERT((std::is_same<T, std::string>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32 array", key.c_str()));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);
            result.clear();

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result.emplace_back(value);
            }
        } else {
            result.resize(arr_info.length);
            result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);
        }

        return true;
    }

    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required) {
        const gguf_context * ctx = meta.get();
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        switch (arr_info.gt) {
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same<T,     int32_t>::value) ||
                                                (std::is_same<T,    uint32_t>::value)); break;
            case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T,       float>::value)); break;
            case GGUF_TYPE_STRING:  GGML_ASSERT((std::is_same<T, std::string>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32 array", key.c_str()));
        }

        if (arr_info.length > N_MAX) {
            throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result[i] = value;
            }
        } else {
            std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_arr(enum llm_kv kid, T & result, bool required) {
        return get_arr(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr<std::vector<std::string>>(enum llm_kv kid, std::vector<std::string> & result, bool required);

    template<typename T>
    bool llama_model_loader::get_key(const std::string & key, T & result, bool required) {
        auto it = kv_overrides.find(key);

        const struct llama_model_kv_override * override =
            it != kv_overrides.end() ? &it->second : nullptr;

        const bool found = GGUFMeta::GKV<T>::set(meta.get(), key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool llama_model_loader::get_key(enum llm_kv kid, T & result, bool required) {
        return get_key(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
    template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
    template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
    template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

    template<>
    bool llama_model_loader::get_key(enum llm_kv kid, enum llama_pooling_type & result, bool required) {
        uint32_t tmp;
        const bool found = get_key(kid, tmp, required);
        if (found) {
            result = (enum llama_pooling_type) tmp;
        } else {
            result = LLAMA_POOLING_TYPE_UNSPECIFIED;
        }
        return found;
    }

    // get array of n <= N_MAX elements, or a single element repeated n times
    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required) {
        const int kid = gguf_find_key(meta.get(), key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        if (n > N_MAX) {
            throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", (uint32_t) n, (uint32_t) N_MAX, key.c_str()));
        }

        if (gguf_get_kv_type(meta.get(), kid) == GGUF_TYPE_ARRAY) {
            struct GGUFMeta::ArrayInfo arr_info =
                GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

            if (n != arr_info.length) {
                throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
            }

            return get_arr(key, result, required);
        }

        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required) {
        return get_key_or_arr(llm_kv(kid), result, n, required);
    }

    // TODO: this is not very clever - figure out something better
    template bool llama_model_loader::get_key_or_arr<std::array<int, 4>>(enum llm_kv kid, std::array<int, 4> & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<float, 512>>(enum llm_kv kid, std::array<float, 512> & result, uint32_t n, bool required);


llama_model_loader::llama_model_loader(
        const std::string & fname,
        std::vector<std::string> & splits,
        bool use_mmap,
        bool check_tensors,
        bool no_alloc,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p) {
#ifdef GGML_USE_SYCL
    static std::atomic<uint64_t> g_sycl_model_id{ 1 };
    model_id = g_sycl_model_id.fetch_add(1, std::memory_order_relaxed);
#endif
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    // Load the main GGUF
    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    meta.reset(gguf_init_from_file(fname.c_str(), params));
    if (!meta) {
        throw std::runtime_error(format("%s: failed to load model from %s", __func__, fname.c_str()));
    }

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));

    files.emplace_back(new llama_file(fname.c_str(), "rb"));
    contexts.emplace_back(ctx);

    // Save tensors data offset of the main file.
    // For subsidiary files, `meta` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string tensor_name = std::string(cur->name);
        // make sure there is no duplicated tensor names
        if (weights_map.find(tensor_name) != weights_map.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
        }
        n_elements += ggml_nelements(cur);
        n_bytes    += ggml_nbytes(cur);
        weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, meta.get(), cur));
    }
    uint16_t n_split = 0;
    get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

    // Load additional GGML contexts
    if (n_split > 1) {
        // make sure the main file is loaded first
        uint16_t idx = 0;
        const std::string kv_split_no = llm_kv(LLM_KV_SPLIT_NO);
        get_key(kv_split_no, idx);
        if (idx != 0) {
            throw std::runtime_error(format("illegal split file idx: %d (file: %s), model must be loaded with the first split", idx, fname.c_str()));
        }

        // generate list of splits if needed
        if (splits.empty()) {
            splits = llama_get_list_splits(fname, idx, n_split);
        }

        // in case user give a custom list of splits, check if it matches the expected number
        if (n_split != (uint16_t)splits.size()) {
            throw std::runtime_error(format("invalid split count, given: %zu splits, but expected %d", splits.size(), n_split));
        }

        if (trace > 0) {
            LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
        }

        // load other splits
        for (idx = 1; idx < n_split; idx++) {
            const char * fname_split = splits[idx].c_str();

            struct gguf_init_params split_params = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &ctx,
            };
            gguf_context_ptr ctx_gguf { gguf_init_from_file(fname_split, split_params) };
            if (!ctx_gguf) {
                throw std::runtime_error(format("%s: failed to load GGUF split from %s", __func__, fname_split));
            }

            // check idx
            {
                const int kid = gguf_find_key(ctx_gguf.get(), kv_split_no.c_str());
                if (kid < 0) {
                    throw std::runtime_error(format("missing key %s in GGUF split %s", kv_split_no.c_str(), fname_split));
                }
                int idx_gguf = gguf_get_val_u16(ctx_gguf.get(), kid);
                if (idx_gguf != idx) {
                    throw std::runtime_error(format("invalid split file idx: %d (file: %s), expected %d", idx_gguf, fname_split, idx));
                }
            }

            files.emplace_back(new llama_file(fname_split, "rb"));
            contexts.emplace_back(ctx);

            // Save tensors data offset info of the shard.
            for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                std::string tensor_name = std::string(cur->name);
                // make sure there is no duplicated tensor names
                if (weights_map.find(tensor_name) != weights_map.end()) {
                    throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
                }
                n_elements += ggml_nelements(cur);
                n_bytes    += ggml_nbytes(cur);
                weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur));
            }
        }

        get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

        // sanity check
        {
            const int n_tensors_loaded = (int) weights_map.size();
            if (n_tensors != n_tensors_loaded) {
                throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
            }
        }

        LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
    }

    n_kv      = gguf_get_n_kv(meta.get());
    n_tensors = weights_map.size();

#ifdef GGML_USE_SYCL
    for (const auto & it : weights_map) {
        const llama_tensor_weight & w      = it.second;
        const ggml_tensor *         tensor = w.tensor;
        ggml_backend_sycl_register_weight_identity(
            tensor, w.idx, w.offs, ggml_nbytes(tensor), model_id);
    }
#endif

    fver = (enum llama_fver) gguf_get_version(meta.get());

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (const auto & it : weights_map) {
            const llama_tensor_weight & w = it.second;
            const ggml_tensor * tensor = w.tensor;

            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = w.idx;
                LLAMA_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__,
                        sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            default:
                {
                    LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                    ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                ftype = (llama_ftype) ftype_val;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(meta.get(), i);
            const enum gguf_type type   = gguf_get_kv_type(meta.get(), i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta.get(), i)), gguf_get_arr_n(meta.get(), i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(meta.get(), i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }

    this->use_mmap = use_mmap;
    this->check_tensors = check_tensors;
    this->no_alloc = no_alloc;
}

std::string llama_model_loader::get_arch_name() const {
    return arch_name;
}

enum llm_arch llama_model_loader::get_arch() const {
    return llm_kv.arch;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    auto pos = weights_map.find(name);
    if (pos != weights_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

struct ggml_tensor * llama_model_loader::require_tensor_meta(const std::string & name) const {
    struct ggml_tensor * tensor = get_tensor_meta(name.c_str());
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }
    return tensor;
}

const struct ggml_tensor * llama_model_loader::check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required, int flags) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                // For TP-sharded tensors, allow dimensions that are factors of the file dimensions
                bool is_tp_sharded = false;
                if (i < ne.size()) {
                    // Column-parallel: output dimension (ne[1]) is sharded
                    if ((flags & TENSOR_TP_COL_PARALLEL) && i == 1 && cur->ne[i] % ne[i] == 0) {
                        is_tp_sharded = true;
                    }
                    // Row-parallel: input dimension (ne[0]) is sharded
                    if ((flags & TENSOR_TP_ROW_PARALLEL) && i == 0 && cur->ne[i] % ne[i] == 0) {
                        is_tp_sharded = true;
                    }
                }
                if (!is_tp_sharded) {
                    is_ok = false;
                    break;
                }
            }
        }
        if (!is_ok) {
            throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        llama_format_tensor_shape(ne).c_str(),
                        llama_format_tensor_shape(cur).c_str()));
        }
    }

    return cur;
}

struct ggml_tensor * llama_model_loader::create_tensor(struct ggml_context * ctx, const std::string & name, const std::initializer_list<int64_t> & ne, int flags) {
    LLAMA_LOG_DEBUG("%s: loading tensor %s\n", __func__, name.c_str());
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED), flags);

    if (cur == NULL) {
        return NULL;
    }

    bool duplicated = flags & TENSOR_DUPLICATED;
    bool is_tp_sharded = (flags & TENSOR_TP_COL_PARALLEL) || (flags & TENSOR_TP_ROW_PARALLEL);

    struct ggml_tensor * tensor;
    if (is_tp_sharded) {
        // For TP-sharded tensors, create tensor with requested (sharded) dimensions
        std::vector<int64_t> ne_vec(ne);
        tensor = ggml_new_tensor(ctx, cur->type, ne_vec.size(), ne_vec.data());
    } else {
        // Normal case: create tensor with same dimensions as GGUF file
        tensor = ggml_dup_tensor(ctx, cur);
    }
    ggml_set_name(tensor, ggml_get_name(cur));

    if (duplicated) {
        size_data += ggml_nbytes(cur);
    } else {
        n_created++;
    }

    return tensor;

}

struct ggml_tensor * llama_model_loader::create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base, const std::string & name, const std::initializer_list<int64_t> & ne, size_t offset, bool required) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
    }

    std::array<int64_t, GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne.begin()[i] : 1;
    }

    struct ggml_tensor * tensor = ggml_view_4d(ctx, base,
                                    dims[0], dims[1], dims[2], dims[3],
                                    cur->nb[1], cur->nb[2], cur->nb[3],
                                    offset);

    ggml_set_name(tensor, name.c_str());

    n_created++;

    return tensor;
}

void llama_model_loader::done_getting_tensors() const {
    if (n_created != n_tensors) {
        throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            bool is_numa = false;

            auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (dev) {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                auto * is_numa_fn = (decltype(ggml_is_numa) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_is_numa");
                if (is_numa_fn) {
                    is_numa = is_numa_fn();
                }
            }

            std::unique_ptr<llama_mmap> mapping = std::make_unique<llama_mmap>(file.get(), prefetch ? -1 : 0, is_numa);
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (const auto & it : weights_map) {
        size_data += ggml_nbytes(it.second.tensor);
    }
}

void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr = mapping->addr();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        const auto * weight = get_weight(ggml_get_name(tensor));
        if (!weight || weight->idx != idx) {
            continue;
        }
        *first = std::min(*first, weight->offs);
        *last  = std::max(*last,  weight->offs + ggml_nbytes(tensor));
    }
}

static uint64_t llama_fnv1a64(const uint8_t * data, size_t len) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void llama_maybe_log_canonical_checksum(const ggml_tensor * tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    const char * name = ggml_get_name(tensor);
    if (!name || name[0] == '\0') {
        return;
    }
    static int enabled = -1;
    static std::string filter;
    static size_t bytes_limit = 4096;
    if (enabled < 0) {
        const char * env = std::getenv("LLAMA_CANONICAL_CHECKSUM_TENSOR");
        enabled           = (env && env[0] != '\0') ? 1 : 0;
        if (enabled) {
            filter = env;
        }
        if (const char * bytes_env = std::getenv("LLAMA_CANONICAL_CHECKSUM_BYTES")) {
            const long long v = std::atoll(bytes_env);
            if (v > 0) {
                bytes_limit = static_cast<size_t>(v);
            }
        }
    }
    if (!enabled) {
        return;
    }
    if (filter[0] != '\0' && std::strstr(name, filter.c_str()) == nullptr) {
        return;
    }
    const size_t nbytes = ggml_nbytes(tensor);
    const size_t bytes  = std::min(bytes_limit, nbytes);
    const auto checksum = llama_fnv1a64(static_cast<const uint8_t *>(tensor->data), bytes);
    std::fprintf(stderr,
                 "[LLAMA-CANONICAL-CHECKSUM] tensor=%s bytes=%zu checksum=0x%016llx\n",
                 name, bytes, (unsigned long long) checksum);
}

void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, ggml_nbytes(cur));
        }
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
    }

    if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
    }

    llama_maybe_log_canonical_checksum(cur);
}

#ifdef GGML_USE_SYCL
// Check if tensor is row-parallel (needs strided loading)
static bool is_row_parallel_tensor(const char * tensor_name) {
    if (!tensor_name) return false;
    return (strstr(tensor_name, "attn_output.weight") != nullptr ||
            strstr(tensor_name, "ffn_down.weight") != nullptr);
}

// Perform strided copy for row-parallel tensor loading
// Copies the rank's portion from memory-mapped data with proper striding
static void copy_row_parallel_data(
    uint8_t * dst,
    const uint8_t * src,
    const struct ggml_tensor * tensor,
    const struct gguf_context * gguf_ctx) {

    int world_size = ggml_backend_sycl_get_tp_world_size();
    int rank = ggml_backend_sycl_get_tp_rank();

    // Get tensor info from GGUF to determine original dimensions
    const int tensor_idx = gguf_find_tensor(gguf_ctx, ggml_get_name(tensor));
    if (tensor_idx < 0) {
        // Tensor not found, shouldn't happen
        LLAMA_LOG_ERROR("[TP] Row-parallel tensor '%s' not found in GGUF\n", ggml_get_name(tensor));
        return;
    }

    ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, tensor_idx);

    // Current (sharded) dimensions
    int64_t ne0 = tensor->ne[0];  // Sharded input dimension
    int64_t ne1 = tensor->ne[1];  // Full output dimension

    // Original dimensions
    int64_t orig_ne0 = ne0 * world_size;

    // For quantized types, calculate in blocks
    int64_t block_size = ggml_blck_size(tensor_type);
    size_t type_size = ggml_type_size(tensor_type);

    // Row size in bytes (original full row)
    size_t orig_row_size = ggml_row_size(tensor_type, orig_ne0);

    // Sharded row size
    size_t shard_row_size = ggml_row_size(tensor_type, ne0);

    // Offset within each row for this rank's shard
    size_t row_offset = rank * shard_row_size;

    LLAMA_LOG_INFO("[TP LOAD] Row-parallel strided copy: tensor='%s' rank=%d ne0=%lld orig_ne0=%lld row_offset=%zu orig_row_size=%zu shard_row_size=%zu\n",
                   ggml_get_name(tensor), rank, (long long)ne0, (long long)orig_ne0, row_offset, orig_row_size, shard_row_size);

    // Copy each row's shard
    for (int64_t row = 0; row < ne1; row++) {
        const uint8_t * src_row = src + row * orig_row_size + row_offset;
        uint8_t * dst_row = dst + row * shard_row_size;
        memcpy(dst_row, src_row, shard_row_size);
    }

    // Debug: print source and destination bytes for first two columns
    if (ne1 > 1 && strstr(ggml_get_name(tensor), "blk.0.attn_output")) {
        // Column 0 (row 0 in loop)
        const uint8_t * src_col0 = src + 0 * orig_row_size + row_offset;
        const uint8_t * src_col1 = src + 1 * orig_row_size + row_offset;
        LLAMA_LOG_INFO("[TP LOAD DEBUG] tensor='%s' rank=%d\n", ggml_get_name(tensor), rank);
        LLAMA_LOG_INFO("[TP LOAD DEBUG]   src_col0 @ offset=%zu: [%02x,%02x,%02x,%02x,...]\n",
                       (size_t)(0 * orig_row_size + row_offset), src_col0[0], src_col0[1], src_col0[2], src_col0[3]);
        LLAMA_LOG_INFO("[TP LOAD DEBUG]   src_col1 @ offset=%zu: [%02x,%02x,%02x,%02x,...]\n",
                       (size_t)(1 * orig_row_size + row_offset), src_col1[0], src_col1[1], src_col1[2], src_col1[3]);
        LLAMA_LOG_INFO("[TP LOAD DEBUG]   dst_col0: [%02x,%02x,%02x,%02x,...]\n",
                       dst[0], dst[1], dst[2], dst[3]);
        LLAMA_LOG_INFO("[TP LOAD DEBUG]   dst_col1 @ offset=%zu: [%02x,%02x,%02x,%02x,...]\n",
                       shard_row_size, dst[shard_row_size], dst[shard_row_size+1], dst[shard_row_size+2], dst[shard_row_size+3]);
    }
}

// Helper function to get TP data offset for multi-process TP mode
// Returns the byte offset within the tensor for this rank's shard
// This uses GGUF metadata to get original tensor dimensions
static size_t get_tp_data_offset_from_gguf(
    const struct gguf_context * gguf_ctx,
    const struct ggml_tensor * tensor) {

    if (!ggml_backend_sycl_is_multiprocess_tp()) {
        return 0;
    }

    // Find tensor in GGUF to get original dimensions
    const int tensor_idx = gguf_find_tensor(gguf_ctx, ggml_get_name(tensor));
    if (tensor_idx < 0) {
        return 0;  // Tensor not found, no offset
    }

    // Get original tensor size from GGUF
    size_t orig_size = gguf_get_tensor_size(gguf_ctx, tensor_idx);
    ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, tensor_idx);

    // Infer original dimensions from size
    // For 2D tensors: orig_size = ggml_row_size(type, ne0) * ne1
    // We need to reverse-engineer ne0 and ne1 from the current tensor
    // knowing that only one dimension is sharded

    // Get current (possibly sharded) dimensions
    int64_t cur_ne0 = tensor->ne[0];
    int64_t cur_ne1 = tensor->ne[1];
    size_t cur_size = ggml_nbytes(tensor);

    // Calculate original dimensions based on which dimension was sharded
    int world_size = ggml_backend_sycl_get_tp_world_size();
    int64_t orig_ne[4] = {cur_ne0, cur_ne1, tensor->ne[2], tensor->ne[3]};

    // If current size * world_size roughly equals original size,
    // we can determine the original dimensions
    if (world_size > 1 && cur_size > 0) {
        // Check if ne[1] was sharded (column-parallel)
        size_t col_parallel_orig_size = ggml_row_size(tensor_type, cur_ne0) * cur_ne1 * world_size;
        if (col_parallel_orig_size == orig_size) {
            // Column-parallel: ne[1] was divided
            orig_ne[1] = cur_ne1 * world_size;
        }
        // Check if ne[0] was sharded (row-parallel)
        else {
            size_t row_parallel_orig_size = ggml_row_size(tensor_type, cur_ne0 * world_size) * cur_ne1;
            if (row_parallel_orig_size == orig_size) {
                // Row-parallel: ne[0] was divided
                orig_ne[0] = cur_ne0 * world_size;
            }
        }
    }

    return ggml_backend_sycl_get_tp_data_offset(
        ggml_get_name(tensor),
        orig_ne,
        tensor_type);
}
#endif

bool llama_model_loader::load_all_data(
        struct ggml_context * ctx,
        llama_buf_map & bufs,
        llama_mlocks * lmlocks,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

#ifdef GGML_USE_SYCL
    // Signal SYCL backend that we're in model load phase
    // This disables weight caching to avoid OOM on large models
    ggml_backend_sycl_set_model_loading(true);
    struct sycl_load_guard {
        ~sycl_load_guard() { ggml_backend_sycl_set_model_loading(false); }
    } guard;
#endif

    std::vector<no_init<uint8_t>> read_buf;
    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers = 4;
    constexpr size_t buffer_size = 1 * 1024 * 1024; // 1MB

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<ggml_backend_event_t> events;
    std::vector<void *> host_ptrs;
    size_t buffer_idx = 0; // buffer to use for async loads
    ggml_backend_t upload_backend = [&](const char * func) -> ggml_backend_t {
        if (use_mmap || check_tensors) {
            return nullptr;
        }
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
        auto * dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
            LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

            auto * event = ggml_backend_event_new(dev);
            if (!event) {
                LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            events.emplace_back(event);
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        return backend;
    }(__func__);

    if (upload_backend) {
        LLAMA_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
            ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),
            ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),
            ggml_backend_name(upload_backend));
    }

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        // Calculate TP data offset for multi-process tensor parallelism
        // This allows each rank to read only its portion of sharded weights
        size_t tp_offset = 0;
        bool needs_row_parallel_copy = false;
#ifdef GGML_USE_SYCL
        if (ggml_backend_sycl_is_multiprocess_tp() && ggml_backend_sycl_get_tp_world_size() > 1) {
            if (is_row_parallel_tensor(ggml_get_name(cur))) {
                // Row-parallel tensors need strided copy, not simple offset
                needs_row_parallel_copy = true;
            } else {
                tp_offset = get_tp_data_offset_from_gguf(meta.get(), cur);
                if (tp_offset > 0) {
                    LLAMA_LOG_INFO("[TP LOAD] tensor '%s' rank=%d offset=%zu (+%zu)\n",
                                  ggml_get_name(cur), ggml_backend_sycl_get_tp_rank(),
                                  weight->offs + tp_offset, tp_offset);
                }
            }
        }
#endif
        size_t read_offs = weight->offs + tp_offset;

        // For row-parallel tensors in TP mode, skip mmap and use file-based loading
        // because mmap is read-only and we need to do strided copies
        bool use_mmap_for_this_tensor = use_mmap;
#ifdef GGML_USE_SYCL
        if (needs_row_parallel_copy && use_mmap) {
            use_mmap_for_this_tensor = false;
            LLAMA_LOG_DEBUG("[TP LOAD] Skipping mmap for row-parallel tensor '%s'\n", ggml_get_name(cur));
        }
#endif

        if (use_mmap_for_this_tensor) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs.count(weight->idx)) {
                buf_mmap = bufs.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + read_offs;

            {
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                    }));
                }

                GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
                if (buf_mmap && cur->data == nullptr) {
                    ggml_backend_tensor_alloc(buf_mmap, cur, data);
                    if (lmlocks) {
                        const auto & lmlock = lmlocks->at(weight->idx);
                        lmlock->grow_to(read_offs + n_size);
                    }

                    auto & mmap_used = mmaps_used[weight->idx];
                    mmap_used.first  = std::min(mmap_used.first,  read_offs);
                    mmap_used.second = std::max(mmap_used.second, read_offs + n_size);
                } else {
                    ggml_backend_tensor_set(cur, data, 0, n_size);
                }
            }
        } else {
            const auto & file = files.at(weight->idx);
#ifdef GGML_USE_SYCL
            if (needs_row_parallel_copy) {
                // Row-parallel: need strided file reads
                int world_size = ggml_backend_sycl_get_tp_world_size();
                int rank = ggml_backend_sycl_get_tp_rank();

                int64_t ne0 = cur->ne[0];  // Sharded
                int64_t ne1 = cur->ne[1];  // Full

                ggml_type tensor_type = cur->type;
                size_t orig_row_size = ggml_row_size(tensor_type, ne0 * world_size);
                size_t shard_row_size = ggml_row_size(tensor_type, ne0);
                size_t row_offset = rank * shard_row_size;

                LLAMA_LOG_INFO("[TP LOAD] Row-parallel file read: tensor='%s' rank=%d ne0=%lld row_offset=%zu\n",
                              ggml_get_name(cur), rank, (long long)ne0, row_offset);

                // Read each row's shard
                read_buf.resize(n_size);
                for (int64_t row = 0; row < ne1; row++) {
                    size_t file_pos = weight->offs + row * orig_row_size + row_offset;
                    file->seek(file_pos, SEEK_SET);
                    file->read_raw(read_buf.data() + row * shard_row_size, shard_row_size);
                }

                // Debug: print first bytes of first two columns for blk.0.attn_output
                if (strstr(ggml_get_name(cur), "blk.0.attn_output")) {
                    LLAMA_LOG_INFO("[TP LOAD DEBUG FILE] tensor='%s' rank=%d col0=[%02x,%02x,%02x,%02x] col1=[%02x,%02x,%02x,%02x]\n",
                                   ggml_get_name(cur), rank,
                                   read_buf[0], read_buf[1], read_buf[2], read_buf[3],
                                   read_buf[shard_row_size], read_buf[shard_row_size+1],
                                   read_buf[shard_row_size+2], read_buf[shard_row_size+3]);
                }

                bool is_host_buf = ggml_backend_buffer_is_host(cur->buffer);
                if (strstr(ggml_get_name(cur), "blk.0.attn_output")) {
                    LLAMA_LOG_INFO("[TP LOAD DEBUG] tensor='%s' rank=%d is_host=%d buffer=%p data=%p n_size=%zu\n",
                                   ggml_get_name(cur), rank, is_host_buf, (void*)cur->buffer, cur->data, n_size);
                }
                if (is_host_buf) {
                    memcpy(cur->data, read_buf.data(), n_size);
                } else {
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                }
            } else
#endif
            if (ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(read_offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (upload_backend) {
                    file->seek(read_offs, SEEK_SET);

                    size_t bytes_read = 0;

                    while (bytes_read < n_size) {
                        size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                        ggml_backend_event_synchronize(events[buffer_idx]);
                        file->read_raw(host_ptrs[buffer_idx], read_iteration);
                        ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], bytes_read, read_iteration);
                        ggml_backend_event_record(events[buffer_idx], upload_backend);

                        bytes_read += read_iteration;
                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    read_buf.resize(n_size);
                    file->seek(read_offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }
            }
        }

        size_done += n_size;
    }

    // free temporary resources used for async uploads
    for (auto * event : events) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
    }
    for (auto * buf : host_buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_backend_free(upload_backend);

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

std::string llama_model_loader::ftype_name() const {
    return llama_model_ftype_name(ftype);
}

void llama_model_loader::print_info() const {
    LLAMA_LOG_INFO("%s: file format = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: file type   = %s\n", __func__, llama_model_ftype_name(ftype).c_str());
    if (n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0,        n_bytes*8.0/n_elements);
    } else {
        LLAMA_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0/1024.0, n_bytes*8.0/n_elements);
    }
}
