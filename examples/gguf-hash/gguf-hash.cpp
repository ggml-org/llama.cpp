#include "ggml.h"

#include <cstdlib>   /* abort() */
#include <cstddef>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#include <sstream>
#include <fstream>

#ifdef __cplusplus
extern "C" {
#endif

#include "xxhash/xxhash.h"
#include "sha1/sha1.h"
#include "sha256/sha256.h"

#ifdef __cplusplus
}
#endif


// uuid.uuid5(uuid.NAMESPACE_URL, 'en.wikipedia.org/wiki/Llama.cpp')
#define UUID_NAMESPACE_LLAMA_CPP "ef001206-dadc-5f6d-a15f-3359e577d4e5"
#define UUID_NAMESPACE_LLAMA_CPP_HEX 0xef, 0x00, 0x12, 0x06, 0xda, 0xdc, 0x5f, 0x6d, 0xa1, 0x5f, 0x33, 0x59, 0xe5, 0x77, 0xd4, 0xe5


#define HASH_TYPE_SHA256_STR "sha256"
#define HASH_TYPE_SHA1_STR   "sha1"
#define HASH_TYPE_XXH64_STR  "xxh64"
#define HASH_TYPE_UUID_STR   "uuid"


typedef enum {
    HASH_EXIT_SUCCESS = 0, // All hash has been generated or validated
    HASH_EXIT_FAILURE = 1, // Generic Failure
    HASH_EXIT_MISMATCH = 2, // Hash mismatched during validation
    HASH_EXIT_MANIFEST_MISSING_ENTRY = 3, // Hash attempted validation but missing entry in manifest
    HASH_EXIT_MANIFEST_UNKNOWN_HASH = 4, // Manifest is present, but we do not know any hash format within it
    HASH_EXIT_MANIFEST_FILE_ERROR = 5 // Manifest is either missing or not a known format
} hash_exit_code_t;


typedef enum {
    HASH_MANIFEST_NOT_FOUND,
    HASH_MANIFEST_MISMATCH,
    HASH_MANIFEST_OK,
} hash_manifest_result_t;


struct hash_params {
    std::string input;
    bool xxh64 = false;
    bool sha1 = false;
    bool sha256 = false;
    bool uuid = false;

    bool no_layer = false;

    bool manifest_is_usable = false;
    std::string manifest_file;
};

struct manifest_check_params {
    bool xxh64 = false;
    bool sha1 = false;
    bool sha256 = false;
    bool uuid = false;
};

static char const * hash_manifest_result_to_str(hash_manifest_result_t value) {
    switch (value) {
        case HASH_MANIFEST_NOT_FOUND: return "Not Found";
        case HASH_MANIFEST_MISMATCH: return "Mismatch";
        case HASH_MANIFEST_OK: return "Ok";
    }
    return "?";
}

static char const * hash_exit_code_to_str(hash_exit_code_t value) {
    switch (value) {
        case HASH_EXIT_SUCCESS: return "Success";
        case HASH_EXIT_FAILURE: return "Failure";
        case HASH_EXIT_MISMATCH: return "Mismatch";
        case HASH_EXIT_MANIFEST_MISSING_ENTRY: return "Manifest Missing Entry";
        case HASH_EXIT_MANIFEST_UNKNOWN_HASH: return "Manifest Unknown Hash";
        case HASH_EXIT_MANIFEST_FILE_ERROR: return "Manifest File Error";
    }
    return "?";
}

static void hash_print_usage(const char * executable) {
    const hash_params default_params;
    printf("\n");
    printf("usage: %s [options] GGUF_IN\n", executable);
    printf("\n");
    printf("Hash a GGUF file");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help              show this help message and exit\n");
    printf("      --xxh64             use xxh64 hash\n");
    printf("      --sha1              use sha1 hash\n");
    printf("      --sha256            use sha256 hash\n");
    printf("      --all               use all hash\n");
    printf("      --no-layer          exclude per layer hash\n");
    printf("      --uuid              generate UUIDv5 ID\n");
    printf("  -c, --check <manifest>  verify against a manifest\n");
    printf("\n");
}

static void hash_params_parse_ex(int argc, const char ** argv, hash_params & params) {
    std::string arg;
    bool invalid_param = false;
    const std::string arg_prefix = "--";

    int arg_idx = 1;
    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        arg = argv[arg_idx];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        bool arg_found = false;
        if (arg == "-h" || arg == "--help") {
            hash_print_usage(argv[0]);
            exit(0);
        }

        if (arg == "--xxh64") {
            arg_found = true;
            params.xxh64 = true;
        }

        if (arg == "--sha1") {
            arg_found = true;
            params.sha1 = true;
        }

        if (arg == "--uuid") {
            arg_found = true;
            params.uuid = true;
        }

        if (arg == "--sha256") {
            arg_found = true;
            params.sha256 = true;
        }

        if (arg == "--all") {
            arg_found = true;
            params.sha256 = true;
            params.sha1 = true;
            params.xxh64 = true;
        }

        if (arg == "--no-layer") {
            arg_found = true;
            params.no_layer = true;
        }

        if (arg == "-c" || arg == "--check") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            params.manifest_file = argv[arg_idx];
        }

        if (!arg_found) {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
    }

    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument:" + arg);
    }

    if (argc - arg_idx < 1) {
        throw std::invalid_argument("error: bad arguments");
    }

    params.input = argv[arg_idx++];
}

static bool hash_params_parse(int argc, const char ** argv, hash_params & params) {
    bool result = true;
    try {
        hash_params_parse_ex(argc, argv, params);
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        hash_print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return result;
}

static bool manifest_type(const std::string & manifest_file, manifest_check_params & manifest_check) {
    if (manifest_file.empty()) {
        return false;
    }

    std::ifstream file(manifest_file);
    if (!file.is_open()) {
        return false;
    }

    std::string manifest_entry_line;
    while (getline(file, manifest_entry_line)) {
        // hash_type_str hash_str tensor_name
        // e.g. 'xxh64     f66e9cd66a4396a0  test.gguf:tensor_0'
        std::istringstream line_stream(manifest_entry_line);
        std::string file_hash_type;
        if (line_stream >> file_hash_type) {
            if (file_hash_type == HASH_TYPE_SHA256_STR) {
                manifest_check.sha256 = true;
            } else if (file_hash_type == HASH_TYPE_SHA1_STR) {
                manifest_check.sha1 = true;
            } else if (file_hash_type == HASH_TYPE_XXH64_STR) {
                manifest_check.xxh64 = true;
            } else if (file_hash_type == HASH_TYPE_UUID_STR) {
                manifest_check.uuid = true;
            }
        }
    }

    return true;
}

static hash_manifest_result_t manifest_verify(const std::string& manifest_file, const std::string& hash_type_str, const std::string& hash_str, const std::string& tensor_name) {
    if (manifest_file.empty()) {
        return HASH_MANIFEST_NOT_FOUND;
    }

    std::ifstream file(manifest_file);
    if (!file.is_open()) {
        return HASH_MANIFEST_NOT_FOUND;
    }

    std::string manifest_entry_line;
    while (getline(file, manifest_entry_line)) {
        std::istringstream line_stream(manifest_entry_line);
        std::string file_hash_type;
        std::string file_hash;
        std::string file_tensor_name;
        if (line_stream >> file_hash_type >> file_hash >> file_tensor_name) {
            // Line parsed. Check hash validity

            if (file_hash_type != hash_type_str) {
                continue;
            }

            if (file_tensor_name != tensor_name) {
                continue;
            }

            return (file_hash == hash_str) ? HASH_MANIFEST_OK : HASH_MANIFEST_MISMATCH;
        }
    }

    return HASH_MANIFEST_NOT_FOUND;
}

static void generate_uuidv5(const unsigned char sha1_digest[20], unsigned char uuid[16]) {
    // Ref: https://www.rfc-editor.org/rfc/rfc9562.html#section-5.5
    // Assumes that digest was processed correctly with the expected namespace
    for (int i = 0; i < 16; i++) {
        uuid[i] = sha1_digest[i];
    }

    // Set bits corresponding to UUID ver 5
    uuid[ 6] &= ~(0xF << 4);
    uuid[ 6] |= (5 << 4);

    // Set bits corresponding to UUID variant 0b10XX
    uuid[ 8] &= ~(0xc << 4);
    uuid[ 8] |= (0x8 << 4);
}

static hash_exit_code_t gguf_hash(const hash_params & hash_params) {
    const std::string & fname = hash_params.input;
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    // xxh64 init
    XXH64_state_t* xxh64_model_hash_state = NULL;
    if (hash_params.xxh64) {
        xxh64_model_hash_state = XXH64_createState();
        if (xxh64_model_hash_state==NULL) {
            abort();
        }

        XXH64_hash_t const seed = 0;
        if (XXH64_reset(xxh64_model_hash_state, seed) == XXH_ERROR) {
            abort();
        }
    }

    // sha1 init
    SHA1_CTX sha1_model_hash_ctx;
    if (hash_params.sha1) {
        SHA1Init(&sha1_model_hash_ctx);
    }

    // sha256 init
    sha256_t sha256_model_hash_ctx;
    if (hash_params.sha256) {
        sha256_init(&sha256_model_hash_ctx);
    }

    // sha1 for uuid init
    SHA1_CTX sha1_for_uuid_ctx;
    if (hash_params.uuid) {
        unsigned char const uuidv5_namespace[] = {UUID_NAMESPACE_LLAMA_CPP_HEX};
        SHA1Init(&sha1_for_uuid_ctx);
        SHA1Update( &sha1_for_uuid_ctx, (unsigned char const *)uuidv5_namespace, sizeof(uuidv5_namespace));
    }

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    const int n_tensors = gguf_get_n_tensors(ctx);
    bool tensor_layer_in_manifest = false;
    bool model_in_manifest = false;
    bool tensor_layer_has_mismatch = false;
    bool model_has_mismatch = false;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        auto n_bytes = ggml_nbytes(cur);
        auto *raw_data = cur->data;
        const std::string tensor_layer_name = fname + ":" + name;

        if (hash_params.xxh64) {

            if (!hash_params.no_layer) {
                // Per Layer Hash
                XXH64_hash_t hash = XXH64(raw_data, n_bytes, 0);

                char hex_result[17];
                for (int  offset = 0; offset < 8; offset++) {
                    unsigned int shift_bits_by = (8 * (8 - offset - 1));
                    snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", (unsigned char) (hash >> shift_bits_by)&0xff);
                }

                if (hash_params.manifest_is_usable) {
                    hash_manifest_result_t verify_result = manifest_verify(hash_params.manifest_file, HASH_TYPE_XXH64_STR, hex_result, tensor_layer_name);

                    switch (verify_result) {
                        case HASH_MANIFEST_NOT_FOUND:
                            break;
                        case HASH_MANIFEST_MISMATCH:
                            tensor_layer_in_manifest = true;
                            tensor_layer_has_mismatch = true;
                            break;
                        case HASH_MANIFEST_OK:
                            tensor_layer_in_manifest = true;
                            break;
                    }

                    printf("%-8s  %-s  %s  -  %s\n", HASH_TYPE_XXH64_STR, hex_result, tensor_layer_name.c_str(), hash_manifest_result_to_str(verify_result));
                } else {
                    printf("%-8s  %-s  %s\n", HASH_TYPE_XXH64_STR, hex_result, tensor_layer_name.c_str());
                }
            }

            // Overall Model Hash
            if (XXH64_update(xxh64_model_hash_state, raw_data, n_bytes) == XXH_ERROR) abort();
        }

        if (hash_params.sha1) {

            if (!hash_params.no_layer) {
                // Per Layer Hash
                char result[21]; // sha1 outputs 20 bytes
                SHA1( result, (const char *)raw_data, n_bytes);

                char hex_result[41] = {0};
                for (int  offset = 0; offset < 20; offset++) {
                    snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", result[offset]&0xff);
                }

                if (hash_params.manifest_is_usable) {
                    hash_manifest_result_t verify_result = manifest_verify(hash_params.manifest_file, HASH_TYPE_SHA1_STR, hex_result, tensor_layer_name);

                    switch (verify_result) {
                        case HASH_MANIFEST_NOT_FOUND:
                            break;
                        case HASH_MANIFEST_MISMATCH:
                            tensor_layer_in_manifest = true;
                            tensor_layer_has_mismatch = true;
                            break;
                        case HASH_MANIFEST_OK:
                            tensor_layer_in_manifest = true;
                            break;
                    }

                    printf("%-8s  %-s  %s  -  %s\n", HASH_TYPE_SHA1_STR, hex_result, tensor_layer_name.c_str(), hash_manifest_result_to_str(verify_result));
                } else {
                    printf("%-8s  %-s  %s\n", HASH_TYPE_SHA1_STR, hex_result, tensor_layer_name.c_str());
                }
            }

            // Overall Model Hash
            SHA1Update( &sha1_model_hash_ctx, (unsigned char const *)raw_data, n_bytes);
        }

        if (hash_params.sha256) {

            if (!hash_params.no_layer) {
                // Per Layer Hash
                unsigned char result[SHA256_DIGEST_SIZE]; // sha256 outputs 32 bytes
                sha256_hash((unsigned char*) result, (const unsigned char *)raw_data, n_bytes);

                char hex_result[SHA256_DIGEST_SIZE * 2 + 1] = {0};
                for (int  offset = 0; offset < SHA256_DIGEST_SIZE; offset++) {
                    snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", result[offset]&0xff);
                }

                if (hash_params.manifest_is_usable) {
                    hash_manifest_result_t verify_result = manifest_verify(hash_params.manifest_file, HASH_TYPE_SHA256_STR, hex_result, tensor_layer_name);

                    switch (verify_result) {
                        case HASH_MANIFEST_NOT_FOUND:
                            break;
                        case HASH_MANIFEST_MISMATCH:
                            tensor_layer_in_manifest = true;
                            tensor_layer_has_mismatch = true;
                            break;
                        case HASH_MANIFEST_OK:
                            tensor_layer_in_manifest = true;
                            break;
                    }

                    printf("%-8s  %-s  %s  -  %s\n", HASH_TYPE_SHA256_STR, hex_result, tensor_layer_name.c_str(), hash_manifest_result_to_str(verify_result));
                } else {
                    printf("%-8s  %-s  %s\n", HASH_TYPE_SHA256_STR, hex_result, tensor_layer_name.c_str());
                }
            }

            // Overall Model Hash
            sha256_update( &sha256_model_hash_ctx, (unsigned char const *)raw_data, n_bytes);
        }

        if (hash_params.uuid) {
            SHA1Update( &sha1_for_uuid_ctx, (unsigned char const *)raw_data, n_bytes);
        }
    }

    if (hash_params.xxh64) {
        XXH64_hash_t const hash = XXH64_digest(xxh64_model_hash_state);

        char hex_result[17];
        for (int  offset = 0; offset < 8; offset++) {
            unsigned int shift_bits_by = (8 * (8 - offset - 1));
            snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", (unsigned char) (hash >> shift_bits_by)&0xff);
        }

        if (hash_params.manifest_is_usable) {
            hash_manifest_result_t verify_result = manifest_verify(hash_params.manifest_file, HASH_TYPE_XXH64_STR, hex_result, fname);

            switch (verify_result) {
                case HASH_MANIFEST_NOT_FOUND:
                    break;
                case HASH_MANIFEST_MISMATCH:
                    model_in_manifest = true;
                    model_has_mismatch = true;
                    break;
                case HASH_MANIFEST_OK:
                    model_in_manifest = true;
                    break;
            }

            printf("%-8s  %-s  %s  -  %s\n", HASH_TYPE_XXH64_STR, hex_result, fname.c_str(), hash_manifest_result_to_str(verify_result));
        } else {
            printf("%-8s  %-s  %s\n", HASH_TYPE_XXH64_STR, hex_result, fname.c_str());
        }
    }

    if (hash_params.sha1) {
        unsigned char result[21];
        SHA1Final(result, &sha1_model_hash_ctx);

        char hex_result[41];
        for (int  offset = 0; offset < 20; offset++) {
            snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", result[offset]&0xff);
        }

        if (hash_params.manifest_is_usable) {
            hash_manifest_result_t verify_result = manifest_verify(hash_params.manifest_file, HASH_TYPE_SHA1_STR, hex_result, fname);

            switch (verify_result) {
                case HASH_MANIFEST_NOT_FOUND:
                    break;
                case HASH_MANIFEST_MISMATCH:
                    model_in_manifest = true;
                    model_has_mismatch = true;
                    break;
                case HASH_MANIFEST_OK:
                    model_in_manifest = true;
                    break;
            }

            printf("%-8s  %-s  %s  -  %s\n", HASH_TYPE_SHA1_STR, hex_result, fname.c_str(), hash_manifest_result_to_str(verify_result));
        } else {
            printf("%-8s  %-s  %s\n", HASH_TYPE_SHA1_STR, hex_result, fname.c_str());
        }
    }

    if (hash_params.sha256) {
        unsigned char result[SHA256_DIGEST_SIZE]; // sha256 outputs 32 bytes
        sha256_final( &sha256_model_hash_ctx,  result);

        char hex_result[SHA256_DIGEST_SIZE * 2 + 1] = {0};
        for (int  offset = 0; offset < SHA256_DIGEST_SIZE; offset++) {
            snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", result[offset]&0xff);
        }

        if (hash_params.manifest_is_usable) {
            hash_manifest_result_t verify_result = manifest_verify(hash_params.manifest_file, HASH_TYPE_SHA256_STR, hex_result, fname);

            switch (verify_result) {
                case HASH_MANIFEST_NOT_FOUND:
                    break;
                case HASH_MANIFEST_MISMATCH:
                    model_in_manifest = true;
                    model_has_mismatch = true;
                    break;
                case HASH_MANIFEST_OK:
                    model_in_manifest = true;
                    break;
            }

            printf("%-8s  %-s  %s  -  %s\n", HASH_TYPE_SHA256_STR, hex_result, fname.c_str(), hash_manifest_result_to_str(verify_result));
        } else {
            printf("%-8s  %-s  %s\n", HASH_TYPE_SHA256_STR, hex_result, fname.c_str());
        }
    }

    if (hash_params.uuid) {
        unsigned char result[21];
        SHA1Final(result, &sha1_for_uuid_ctx);

        unsigned char uuid[16];
        generate_uuidv5(result, uuid);

        char string_buffer[37] = {0};
        snprintf(string_buffer, sizeof(string_buffer), "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            uuid[0], uuid[1], uuid[2], uuid[3],
            uuid[4], uuid[5], uuid[6], uuid[7],
            uuid[8], uuid[9], uuid[10], uuid[11],
            uuid[12], uuid[13], uuid[14], uuid[15]);

        if (hash_params.manifest_is_usable) {
            hash_manifest_result_t verify_result = manifest_verify(hash_params.manifest_file, HASH_TYPE_SHA256_STR, string_buffer, fname);

            switch (verify_result) {
                case HASH_MANIFEST_NOT_FOUND:
                    break;
                case HASH_MANIFEST_MISMATCH:
                    model_in_manifest = true;
                    model_has_mismatch = true;
                    break;
                case HASH_MANIFEST_OK:
                    model_in_manifest = true;
                    break;
            }

            printf("%-8s  %-s  %s  -  %s\n", HASH_TYPE_UUID_STR, string_buffer, fname.c_str(), hash_manifest_result_to_str(verify_result));
        } else {
            printf("%-8s  %-s  %s\n", HASH_TYPE_UUID_STR, string_buffer, fname.c_str());
        }
    }


    ggml_free(ctx_data);
    gguf_free(ctx);


    if (hash_params.manifest_is_usable) {
        // In hash verification mode

        if (!model_in_manifest) {
            // model missing in manifest?

            // Check tensor layer...
            if (!tensor_layer_in_manifest) {
                // Still missing? Maybe we are reading the wrong manifest.
                return HASH_EXIT_MANIFEST_MISSING_ENTRY;
            }

            if (tensor_layer_has_mismatch) {
                // Per tensor check found error
                return HASH_EXIT_FAILURE;
            }

            // All per tensor layer checks passed? Sounds good enough.
            return HASH_EXIT_SUCCESS;
        }

        // Overall model check passed, but let's check per layer just in case
        // If missing, we don't care too much as the overall model checked
        if (tensor_layer_in_manifest && tensor_layer_has_mismatch) {
            return HASH_EXIT_FAILURE;
        }

        if (model_has_mismatch) {
            // model has failed hash somewhere in the model
            return HASH_EXIT_FAILURE;
        }

        // All checks appears to be fine
        return HASH_EXIT_SUCCESS;
    }

    // In hash generation mode
    return HASH_EXIT_SUCCESS;
}

int main(int argc, const char ** argv) {
    hash_params params;
    manifest_check_params manifest_check;
    hash_params_parse(argc, argv, params);

    if (!params.manifest_file.empty()) {
        if (!manifest_type(params.manifest_file, manifest_check)) {
            printf("ERROR cannot open manifest %s", params.manifest_file.c_str());
            return HASH_EXIT_MANIFEST_FILE_ERROR;
        }

        if (!manifest_check.sha256 && !manifest_check.sha1 && !manifest_check.xxh64 && !manifest_check.uuid) {
            printf("ERROR manifest does not have any known hash format in %s", params.manifest_file.c_str());
            return HASH_EXIT_MANIFEST_UNKNOWN_HASH;
        }

        printf("manifest  %s", params.manifest_file.c_str());

        if (manifest_check.sha256) {
            printf("  sha256");
        }

        if (manifest_check.sha1) {
            printf("  sha1");
        }

        if (manifest_check.xxh64) {
            printf("  xxh64");
        }

        if (manifest_check.uuid) {
            printf("  uuid");
        }

        printf("\n");

        // Autoselect the highest security hash if manifest is provided but
        // the user has not specifically defined the hash they care about
        if (!params.xxh64 && !params.sha1 && !params.uuid && !params.sha256) {
            // User has not selected a specific value, pick most secure hash
            if (manifest_check.sha256) {
                params.sha256 = true;
            } else if (manifest_check.sha1) {
                params.sha1 = true;
            } else if (manifest_check.xxh64) {
                params.xxh64 = true;
            } else if (manifest_check.uuid) {
                params.uuid = true;
            }
        }

        params.manifest_is_usable = true;
    }

    // By default if no swich argument provided, assume xxh64
    if (!params.xxh64 && !params.sha1 && !params.uuid && !params.sha256) {
        params.xxh64 = true;
    }

    hash_exit_code_t exit_code = gguf_hash(params);

    if (params.manifest_is_usable) {
        printf("\nVerification results for %s - %s\n", params.manifest_file.c_str(), hash_exit_code_to_str(exit_code));
    }

    return exit_code;
}
