#include "cogagent_util.h"

void print_dims(struct ggml_tensor * input_tensor, const char * name) {
    printf("Tensor %s has shape %ld x %ld x %ld x %ld and data type %d\n", name,
        input_tensor->ne[0], input_tensor->ne[1], input_tensor->ne[2],
        input_tensor->ne[3], input_tensor->type);
}

struct ggml_tensor * get_tensor(struct ggml_context * dst_ctx, struct ggml_context * src_ctx, std::string tensor_name, int &count_failed) {
    struct ggml_tensor * cur_tensor = ggml_get_tensor(src_ctx, tensor_name.c_str());
    if (!cur_tensor) {
        printf("Retrieval of tensor %s from model context failed\n", tensor_name.c_str());
        count_failed++;
        return nullptr;
    }
    struct ggml_tensor * new_tensor = ggml_dup_tensor(dst_ctx, cur_tensor);
    ggml_set_name(new_tensor, cur_tensor->name);
    return new_tensor;
}

void save_tensor_filename(struct ggml_tensor * input_tensor, std::string filename) {
    std::string prefix = "/home/tianyue/myworkspace/";
    filename = prefix + filename;
    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "model.architecture", "cogagent");
    gguf_set_val_u32(gguf_ctx, "general.file_type", GGML_TYPE_F32);

    struct ggml_init_params params = {
        ggml_nbytes(input_tensor) + 1000000,  // Memory to allocate
        nullptr,  // Buffer location
        false,  // no_alloc=false, so that tensor data is allocated
    };
    struct ggml_context * tensor_ctx = ggml_init(params);
    struct ggml_tensor * tensor_with_data = ggml_dup(tensor_ctx, input_tensor);
    ggml_backend_tensor_get(input_tensor, tensor_with_data->data,
        0, ggml_nbytes(input_tensor));

    ggml_set_name(tensor_with_data, "output_tensor");
    gguf_add_tensor(gguf_ctx, tensor_with_data);
    gguf_write_to_file(gguf_ctx, filename.c_str(), false);
    gguf_free(gguf_ctx);
    ggml_free(tensor_ctx);
}

void save_tensor_from_data(std::vector<float> tensor_data, int* dims, std::string filename) {
    std::string prefix = "/home/tianyue/myworkspace/";
    filename = prefix + filename;
    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "model.architecture", "cogagent");
    gguf_set_val_u32(gguf_ctx, "general.file_type", GGML_TYPE_F32);

    struct ggml_init_params params = {
        tensor_data.size() * sizeof(float) + 1000000,  // Memory to allocate
        nullptr,  // Buffer location
        false,  // Allocate tensor data
    };
    struct ggml_context * tensor_ctx = ggml_init(params);
    struct ggml_tensor * tensor_with_data = ggml_new_tensor_3d(tensor_ctx,
        GGML_TYPE_F32, dims[0], dims[1], dims[2]);
    // copy the data
    memcpy(tensor_with_data->data, tensor_data.data(), ggml_nbytes(tensor_with_data));

    ggml_set_name(tensor_with_data, "output_tensor");
    gguf_add_tensor(gguf_ctx, tensor_with_data);
    gguf_write_to_file(gguf_ctx, filename.c_str(), false);
    gguf_free(gguf_ctx);
    ggml_free(tensor_ctx);
}

// Function that loads data from the GGUF file to a temporary buffer
// and then from the temporary buffer to the GGML backend buffer
// Copied from the GGML MNIST example
bool load_from_gguf(const char * fname, struct ggml_context * ctx_ggml, struct gguf_context * ctx_gguf) {
    FILE * f = ggml_fopen(fname, "rb");
    if (!f) {
        return false;
    }

    const size_t buf_size = 4*1024*1024;
    void * buf = malloc(buf_size);

    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);

        struct ggml_tensor * tensor = ggml_get_tensor(ctx_ggml, name);
        if (!tensor) {
            // We get here if there is a tensor in the file
            // that is not being requested for the context
            // that we are loading into
            continue;
        }

        const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

        if (fseek(f, offs, SEEK_SET) != 0) {
            fclose(f);
            free(buf);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t nbytes_cpy = buf_size < nbytes - pos ? buf_size : nbytes - pos;

            if (fread(buf, 1, nbytes_cpy, f) != nbytes_cpy) {
                fclose(f);
                free(buf);
                return false;
            }

            ggml_backend_tensor_set(tensor, buf, pos, nbytes_cpy);
        }
    }

    fclose(f);
    free(buf);
    return true;
}

int get_input(
    std::vector<float> &input_data,
    const char * filename
) {
    struct ggml_context * meta_info;

    struct gguf_init_params gguf_params = {
        true, &meta_info,
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(filename, gguf_params);
    if (!gguf_ctx) {
        printf("Failed to initialize GGUF context. Check filename.\n");
        return false;
    }

    // I don't know how to set tensor name when writing a GGUF file
    // in cpp, so the cross vision output tensor doesn't appear
    // to have a name when reading the GGUF file
    struct ggml_tensor * meta_tensor = ggml_get_first_tensor(meta_info);
    if (meta_tensor->type != GGML_TYPE_F32) {
        printf("Expected the input image datatype to be float 32.\n");
        printf("Image loading failed because the datatype is actually %d\n",
            meta_tensor->type);
        return -1;
    }

    size_t tensor_size = ggml_nbytes(meta_tensor);
    printf("Input tensor size is %ld bytes\n", tensor_size);

    input_data.resize(meta_tensor->ne[0] * meta_tensor->ne[1] *
        meta_tensor->ne[2]);
    int num_tokens = meta_tensor->ne[1];

    std::ifstream input_file = std::ifstream(filename, std::ios::binary);
    const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, 0);
    input_file.seekg(offset, std::ios::beg);
    printf("Seeked to input tensor GGUF position %ld\n", offset);
    input_file.read(reinterpret_cast<char *>(input_data.data()), tensor_size);
    input_file.close();
    ggml_free(meta_info);
    return num_tokens;
}