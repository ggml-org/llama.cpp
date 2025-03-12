#include "cogagent.h"
#include "vision_encoder.h"

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cassert>

bool vision_encoder_init_load(const char * filename) {
    vision_encoder_ctx &model_ctx = cogagent_global.vision_encoder;
    vision_encoder &model = cogagent_global.vision_encoder.model;

    model_ctx.backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(model_ctx.backend, 16);

    // Initialize the GGML contexts
    struct ggml_init_params weight_params {
        10000000,  // Memory size
        NULL,  // Memory buffer
        true,  // Don't allocate tensor data
    };

    struct ggml_init_params compute_params {
        6400*ggml_tensor_overhead() + ggml_graph_overhead(),
        NULL,
        true,
    };

    model_ctx.ctx_weight = ggml_init(weight_params);
    model_ctx.ctx_compute = ggml_init(compute_params);

    // Load the model weights
    struct ggml_context * meta;
    struct gguf_init_params gguf_params = {
        true, &meta,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(filename, gguf_params);
    if (!gguf_ctx) {
        printf("Failed to initialize GGUF context. Check filename.\n");
        return false;
    }

    // Calculate the ctx size based on tensors in the GGUF file
    size_t ctx_size = 0;
    int num_tensors = gguf_get_n_tensors(gguf_ctx);
    printf("There are %d tensors in the GGUF file\n", num_tensors);
    for (int i=0; i<num_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * cur_tensor = ggml_get_tensor(meta, name);
        ctx_size += ggml_tensor_overhead();
        ctx_size += ggml_nbytes_pad(cur_tensor);
    }
    printf("The total size of all weight tensors is %ld\n", ctx_size);

    std::ifstream input_file = std::ifstream(filename, std::ios::binary);

    int failed_count = 0;
    // D x 1
    model.cls_embed = get_tensor(model_ctx.ctx_weight, meta, "vision.patch_embedding.cls_embedding", failed_count);

    model.patch_conv_w = get_tensor(model_ctx.ctx_weight, meta, "vision.patch_embedding.proj.weight", failed_count);
    model.patch_conv_b = get_tensor(model_ctx.ctx_weight, meta, "vision.patch_embedding.proj.bias", failed_count);
    model.patch_conv_b = ggml_reshape_3d(model_ctx.ctx_weight, model.patch_conv_b, 1, 1, model.hidden_size);

    // D x L
    model.position_embed_1 = get_tensor(model_ctx.ctx_weight, meta, "vision.patch_embedding.position_embedding.weight", failed_count);

    // D x L
    model.position_embed_2 = get_tensor(model_ctx.ctx_weight, meta, "vision.pos_embed", failed_count);

    model.linear_proj_w = get_tensor(model_ctx.ctx_weight, meta, "vision.linear_proj.linear_proj.weight", failed_count);
    model.linear_proj_norm_w = get_tensor(model_ctx.ctx_weight, meta, "vision.linear_proj.norm1.weight", failed_count);
    model.linear_proj_norm_b = get_tensor(model_ctx.ctx_weight, meta, "vision.linear_proj.norm1.bias", failed_count);
    model.gate_proj_w = get_tensor(model_ctx.ctx_weight, meta, "vision.linear_proj.gate_proj.weight", failed_count);
    model.dense_h_to_4h_w = get_tensor(model_ctx.ctx_weight, meta, "vision.linear_proj.dense_h_to_4h.weight", failed_count);
    model.dense_4h_to_h_w = get_tensor(model_ctx.ctx_weight, meta, "vision.linear_proj.dense_4h_to_h.weight", failed_count);

    model.boi = get_tensor(model_ctx.ctx_weight, meta, "vision.boi", failed_count);
    model.eoi = get_tensor(model_ctx.ctx_weight, meta, "vision.eoi", failed_count);

    for (int i=0; i<model.num_layers; i++) {
        std::string layer_prefix = "vision.transformer.layers." + std::to_string(i) + ".";
        model.transformer_layers.emplace_back();
        vision_encoder_layer &cur_layer = model.transformer_layers.back();
        cur_layer.qkv_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attention.query_key_value.weight", failed_count);
        cur_layer.qkv_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attention.query_key_value.bias", failed_count);
        cur_layer.attn_dense_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attention.dense.weight", failed_count);
        cur_layer.attn_dense_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attention.dense.bias", failed_count);
        cur_layer.input_norm_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "input_layernorm.weight", failed_count);
        cur_layer.input_norm_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "input_layernorm.bias", failed_count);
        cur_layer.fc1_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.fc1.weight", failed_count);
        cur_layer.fc1_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.fc1.bias", failed_count);
        cur_layer.fc2_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.fc2.weight", failed_count);
        cur_layer.fc2_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.fc2.bias", failed_count);
        cur_layer.post_attention_norm_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "post_attention_layernorm.weight", failed_count);
        cur_layer.post_attention_norm_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "post_attention_layernorm.bias", failed_count);
    }

    if (failed_count > 0) {
        printf("%d tensors could not be found in the model context. Model loading failed.\n", failed_count);
        return false;
    }

    model_ctx.weight_data = ggml_backend_alloc_ctx_tensors(model_ctx.ctx_weight, model_ctx.backend);

    if (!load_from_gguf(filename, model_ctx.ctx_weight, gguf_ctx)) {
        printf("Loading data from GGUF file failed\n");
        return false;
    }

    ggml_free(meta);

    return true;
}

// This is not declared in the header because it is only intended
// to be called from run_vision_encoder
static struct ggml_cgraph * vision_encoder_graph() {
    struct ggml_context * ctx = cogagent_global.vision_encoder.ctx_compute;
    vision_encoder &model = cogagent_global.vision_encoder.model;

    // Set flag to tell allocator to not overwrite the input tensor
    // before it is done with computation
    ggml_set_input(model.input_image);

    // Assuming input: h, w, 3, b
    // This is different from PyTorch, which is b, 3, h, w
    // Confirm later that the height and width are not swapped
    // It would appear that the number of out channels in the ggml_conv_2d function
    // is implied from the kernel
    // 3 -> 1792
    struct ggml_tensor * patch_embedding = ggml_conv_2d(ctx, model.patch_conv_w, model.input_image,
        14, 14, 0, 0, 1, 1);  // Don't actually know dilation does :) Matches the torch defaults
    // When adding together float16 and float32, float16 has to be first
    patch_embedding = ggml_add(ctx, ggml_repeat(ctx, model.patch_conv_b, patch_embedding), patch_embedding);
    // after conv: w, h, d, b
    // after flatten: w x h = l, d, b
    // after transpose: d, l, b
    // cls token shape: d, 1, b
    // after concatenation: d, l+1, b
    patch_embedding = ggml_reshape_3d(ctx, patch_embedding, patch_embedding->ne[0] * patch_embedding->ne[1], patch_embedding->ne[2], patch_embedding->ne[3]);  // Flatten
    patch_embedding = ggml_cont(ctx, patch_embedding);
    // d, l, b shape at this point
    // Most layer weights will need reshaping
    patch_embedding = ggml_transpose(ctx, patch_embedding);
    patch_embedding = ggml_cont(ctx, patch_embedding);
    // Assume cls_embed and position_embed are expanded to have 1 more dimension
    // than original. Assume that these operations can broadcast automatically
    struct ggml_tensor * cls_embed_shape = ggml_new_tensor_3d(ctx, model.cls_embed->type,
        patch_embedding->ne[0], 1, patch_embedding->ne[2]);
    patch_embedding = ggml_concat(ctx, ggml_repeat(ctx, model.cls_embed, cls_embed_shape), patch_embedding, 1);
    patch_embedding = ggml_cont(ctx, patch_embedding);
    // num_positions in the config is 257, which is 256 + 1
    // 224 x 224 becomes 16 x 16 after convolution with a kernel of 14 x 14
    // 16 x 16 + 1 = 257
    patch_embedding = ggml_add(ctx, ggml_repeat(ctx, model.position_embed_1, patch_embedding), patch_embedding);

    // Original PatchEmbedding complete at this point
    // Loop through the transformer layers
    struct ggml_tensor * layer_in = patch_embedding;
    for (int i=0; i<model.num_layers; i++) {
        vision_encoder_layer &cur_layer = model.transformer_layers[i];

        struct ggml_tensor * qkv = ggml_mul_mat(ctx, cur_layer.qkv_w, layer_in);
        // 3D x L x B
        qkv = ggml_add(ctx, ggml_repeat(ctx, cur_layer.qkv_b, qkv), qkv);
        // D x L x B
        struct ggml_tensor * qt = ggml_view_3d(ctx, qkv, qkv->ne[0] / 3, qkv->ne[1], qkv->ne[2],
            qkv->nb[1], qkv->nb[2], 0);
        struct ggml_tensor * kt = ggml_view_3d(ctx, qkv, qkv->ne[0] / 3, qkv->ne[1], qkv->ne[2],
            qkv->nb[1], qkv->nb[2], qkv->ne[0] / 3 * qkv->nb[0]);
        struct ggml_tensor * vt = ggml_view_3d(ctx, qkv, qkv->ne[0] / 3, qkv->ne[1], qkv->ne[2],
            qkv->nb[1], qkv->nb[2], 2 * qkv->ne[0] / 3 * qkv->nb[0]);
        qt = ggml_cont(ctx, qt);
        kt = ggml_cont(ctx, kt);
        vt = ggml_cont(ctx, vt);
        qt = ggml_scale(ctx, qt, model.attn_scale);
        // Separate into heads
        // K x H x L x B
        qt = ggml_view_4d(ctx, qt, qt->ne[0] / model.num_heads, model.num_heads, qt->ne[1], qt->ne[2],
            qt->ne[0] / model.num_heads * qt->nb[0], qt->nb[1], qt->nb[2], 0);
        kt = ggml_view_4d(ctx, kt, kt->ne[0] / model.num_heads, model.num_heads, kt->ne[1], kt->ne[2],
            kt->ne[0] / model.num_heads * kt->nb[0], kt->nb[1], kt->nb[2], 0);
        vt = ggml_view_4d(ctx, vt, vt->ne[0] / model.num_heads, model.num_heads, vt->ne[1], vt->ne[2],
            vt->ne[0] / model.num_heads * vt->nb[0], vt->nb[1], vt->nb[2], 0);
        qt = ggml_cont(ctx, qt);
        kt = ggml_cont(ctx, kt);
        vt = ggml_cont(ctx, vt);
        // Switch order of dimensions
        // K x L x H x B
        qt = ggml_permute(ctx, qt, 0, 2, 1, 3);
        kt = ggml_permute(ctx, kt, 0, 2, 1, 3);
        qt = ggml_cont(ctx, qt);
        kt = ggml_cont(ctx, kt);
        // L x K x H x B
        struct ggml_tensor * v = ggml_permute(ctx, vt, 1, 2, 0, 3);
        v = ggml_cont(ctx, v);
        // L x L x H x B
        struct ggml_tensor * attnt = ggml_mul_mat(ctx, kt, qt);
        attnt = ggml_soft_max(ctx, attnt);  // Should be on the first dimension
        // attnt = vt x attnt, but we need to give it v instead of vt
        // K x L x H x B
        attnt = ggml_mul_mat(ctx, v, attnt);
        // Switch the dimensions back
        attnt = ggml_permute(ctx, attnt, 0, 2, 1, 3);  // K, H, L, B
        attnt = ggml_cont(ctx, attnt);
        attnt = ggml_view_3d(ctx, attnt, attnt->ne[0] * attnt->ne[1], attnt->ne[2], attnt->ne[3],
            attnt->nb[2], attnt->nb[3], 0);  // D, L, B
        attnt = ggml_mul_mat(ctx, cur_layer.attn_dense_w, attnt);
        attnt = ggml_add(ctx, ggml_repeat(ctx, cur_layer.attn_dense_b, attnt), attnt);
        // Attention calculation is now complete
        attnt = ggml_norm(ctx, attnt, model.layernorm_eps);  // Config value is 0.000001
        attnt = ggml_mul(ctx, ggml_repeat(ctx,  cur_layer.input_norm_w, attnt), attnt);
        attnt = ggml_add(ctx, ggml_repeat(ctx, cur_layer.input_norm_b, attnt), attnt);
        layer_in = ggml_add(ctx, layer_in, attnt);  // D, L, B
        // Perform MLP calculation
        // Weight has shape intermediate size x D
        struct ggml_tensor * fc1 = ggml_mul_mat(ctx, cur_layer.fc1_w, layer_in);
        fc1 = ggml_add(ctx, ggml_repeat(ctx, cur_layer.fc1_b, fc1), fc1);
        struct ggml_tensor * gelu = ggml_gelu(ctx, fc1);
        struct ggml_tensor * fc2 = ggml_mul_mat(ctx, cur_layer.fc2_w, gelu);
        fc2 = ggml_add(ctx, ggml_repeat(ctx, cur_layer.fc2_b, fc2), fc2);
        fc2 = ggml_norm(ctx, fc2, model.layernorm_eps);
        fc2 = ggml_mul(ctx, ggml_repeat(ctx,  cur_layer.post_attention_norm_w, fc2), fc2);
        fc2 = ggml_add(ctx, ggml_repeat(ctx, cur_layer.post_attention_norm_b, fc2), fc2);
        layer_in = ggml_add(ctx, layer_in, fc2);
    }
    struct ggml_tensor * transformer_output = layer_in;
    // Drop class embedding
    {
        long d = transformer_output->ne[0];
        long l = transformer_output->ne[1];
        long b = transformer_output->ne[2];
        transformer_output = ggml_view_3d(ctx, transformer_output, d, l-1, b,
            transformer_output->nb[1], transformer_output->nb[2], transformer_output->nb[1]);
    }
    // Linear projection
    struct ggml_tensor * linear_proj = ggml_add(ctx, ggml_repeat(ctx, model.position_embed_2, transformer_output), transformer_output);
    struct ggml_tensor * linear_proj_tmp = ggml_mul_mat(ctx, model.linear_proj_w, linear_proj);
    linear_proj = ggml_norm(ctx, linear_proj_tmp, model.layernorm_eps);
    linear_proj = ggml_mul(ctx, ggml_repeat(ctx, model.linear_proj_norm_w, linear_proj), linear_proj);
    linear_proj = ggml_add(ctx, ggml_repeat(ctx, model.linear_proj_norm_b, linear_proj), linear_proj);
    linear_proj = ggml_gelu(ctx, linear_proj);
    struct ggml_tensor * gate_proj = ggml_mul_mat(ctx, model.gate_proj_w, linear_proj);
    gate_proj = ggml_silu(ctx, gate_proj);
    struct ggml_tensor * h_4h = ggml_mul_mat(ctx, model.dense_h_to_4h_w, linear_proj);
    linear_proj = ggml_mul(ctx, gate_proj, h_4h);
    linear_proj = ggml_mul_mat(ctx, model.dense_4h_to_h_w, linear_proj);
    // GLU complete
    struct ggml_tensor * expanded_size = ggml_new_tensor_3d(ctx, linear_proj->type, linear_proj->ne[0], 1, linear_proj->ne[2]);
    model.output_tensor = ggml_concat(ctx, ggml_repeat(ctx, model.boi, expanded_size), linear_proj, 1);
    model.output_tensor = ggml_concat(ctx, model.output_tensor, ggml_repeat(ctx, model.eoi, expanded_size), 1);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, model.output_tensor);
    return gf;
}

void run_vision_encoder(std::vector<float> img_data) {
    vision_encoder_ctx &model_ctx = cogagent_global.vision_encoder;
    vision_encoder &model = cogagent_global.vision_encoder.model;

    // Declare the input image tensor
    model.input_image = ggml_new_tensor_3d(cogagent_global.vision_encoder.ctx_compute, GGML_TYPE_F32,
        cogagent_global.vision_encoder_img_size, cogagent_global.vision_encoder_img_size, 3);

    struct ggml_cgraph * gf = vision_encoder_graph();
    ggml_graph_print(gf);
    printf("Number of nodes in the vision encoder graph is %d\n", ggml_graph_n_nodes(gf));

    model_ctx.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_ctx.backend));
    ggml_gallocr_reserve(model_ctx.allocr, gf);
    size_t compute_size = ggml_gallocr_get_buffer_size(model_ctx.allocr, 0);
    printf("Allocated %ld bytes of space for graph computation.\n", compute_size);
    ggml_gallocr_alloc_graph(model_ctx.allocr, gf);

    ggml_backend_tensor_set(model.input_image, img_data.data(), 0, ggml_nbytes(model.input_image));

    // Computation result is at model.output_tensor
    ggml_backend_graph_compute(model_ctx.backend, gf);

    cogagent_global.vision_encoder_image.resize(model.output_tensor->ne[0] *
        model.output_tensor->ne[1]);
    ggml_backend_tensor_get(model.output_tensor, cogagent_global.vision_encoder_image.data(),
        0, ggml_nbytes(model.output_tensor));
    // Added for debugging implementation of encoders
    save_tensor_filename(model.output_tensor, "cogagent_encoders/vision_encoder_output.gguf");
}

void free_vision_encoder_ctx() {
    vision_encoder_ctx &model_ctx = cogagent_global.vision_encoder;

    ggml_gallocr_free(model_ctx.allocr);
    ggml_backend_buffer_free(model_ctx.weight_data);
    ggml_free(model_ctx.ctx_weight);
    ggml_free(model_ctx.ctx_compute);
}