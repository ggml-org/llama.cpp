#include "cogagent.h"
#include "cross_vision.h"

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cassert>

bool cross_vision_init_load(const char * filename) {
    cross_vision_ctx &model_ctx = cogagent_global.cross_vision;
    cross_vision &model = cogagent_global.cross_vision.model;

    model_ctx.backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(model_ctx.backend, 16);

    struct ggml_init_params weight_params {
        // Counted 515 tensors in cross vision encoder save file
        10000000,  // Memory size
        NULL,  // Memory buffer
        true,  // Don't allocate tensor data
    };

    struct ggml_init_params compute_params {
        GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + ggml_graph_overhead(),
        NULL,
        true,
    };

    model_ctx.ctx_weight = ggml_init(weight_params);
    model_ctx.ctx_compute = ggml_init(compute_params);

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

    model.patch_conv_w = get_tensor(model_ctx.ctx_weight, meta, "cross_vision.vit.model.patch_embed.proj.weight", failed_count);
    model.patch_conv_b = get_tensor(model_ctx.ctx_weight, meta, "cross_vision.vit.model.patch_embed.proj.bias", failed_count);
    model.patch_conv_b = ggml_reshape_3d(model_ctx.ctx_weight, model.patch_conv_b, 1, 1, model.patch_conv_b->ne[0]);

    model.cls_embed = get_tensor(model_ctx.ctx_weight, meta, "cross_vision.vit.model.cls_token", failed_count);
    model.pos_embed_1 = get_tensor(model_ctx.ctx_weight, meta, "cross_vision.vit.model.pos_embed", failed_count);

    model.rope_freqs_cos = get_tensor(model_ctx.ctx_weight, meta, "cross_vision.vit.model.rope.freqs_cos", failed_count);
    model.rope_freqs_sin = get_tensor(model_ctx.ctx_weight, meta, "cross_vision.vit.model.rope.freqs_sin", failed_count);

    for (int i=0; i<24; i++) {
        std::string layer_prefix = "cross_vision.vit.model.blocks." + std::to_string(i) + ".";
        model.transformer_layers.emplace_back();
        cross_vision_layer &cur_layer = model.transformer_layers.back();
        cur_layer.norm1_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "norm1.weight", failed_count);
        cur_layer.norm1_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "norm1.bias", failed_count);
        cur_layer.q_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.q_proj.weight", failed_count);
        cur_layer.q_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.q_bias", failed_count);
        cur_layer.k_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.k_proj.weight", failed_count);
        cur_layer.v_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.v_proj.weight", failed_count);
        cur_layer.v_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.v_bias", failed_count);
        cur_layer.attn_ln_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.inner_attn_ln.weight", failed_count);
        cur_layer.attn_ln_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.inner_attn_ln.bias", failed_count);
        cur_layer.attn_linear_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.proj.weight", failed_count);
        cur_layer.attn_linear_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "attn.proj.bias", failed_count);
        cur_layer.norm2_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "norm2.weight", failed_count);
        cur_layer.norm2_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "norm2.bias", failed_count);
        cur_layer.mlp_linear1_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.w1.weight", failed_count);
        cur_layer.mlp_linear1_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.w1.bias", failed_count);
        cur_layer.mlp_linear2_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.w2.weight", failed_count);
        cur_layer.mlp_linear2_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.w2.bias", failed_count);
        cur_layer.mlp_ln_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.ffn_ln.weight", failed_count);
        cur_layer.mlp_ln_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.ffn_ln.bias", failed_count);
        cur_layer.mlp_linear3_w = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.w3.weight", failed_count);
        cur_layer.mlp_linear3_b = get_tensor(model_ctx.ctx_weight, meta, layer_prefix + "mlp.w3.bias", failed_count);
    }

    model.pos_embed_2 = get_tensor(model_ctx.ctx_weight, meta, "cross_vision.pos_embed", failed_count);

    if (failed_count > 0) {
        printf("%d tensors could not be found in the model context. Model loading failed.\n", failed_count);
        return false;
    }

    // Allocate data storage for the tensors on the backend
    model_ctx.weight_data = ggml_backend_alloc_ctx_tensors(model_ctx.ctx_weight, model_ctx.backend);

    if (!load_from_gguf(filename, model_ctx.ctx_weight, gguf_ctx)) {
        printf("Loading data from GGUF file failed\n");
        return false;
    }

    ggml_free(meta);

    return true;
}

static struct ggml_tensor * compute_rope(cross_vision_ctx &model_ctx, struct ggml_tensor *input_tensor) {
    struct ggml_context * ctx = model_ctx.ctx_compute;
    cross_vision model = model_ctx.model;
    // Don't really think this should be necessary
    // The ggml_is_contiguous_n code in ggml.c doesn't even seem to be using the n variable
    input_tensor = ggml_cont(ctx, input_tensor);
    struct ggml_tensor * cos = ggml_mul(ctx, input_tensor, model.rope_freqs_cos);
    if (!ggml_is_contiguous(input_tensor)) {printf("Not contiguous input tensor\n");}
    struct ggml_tensor * rotate_half = ggml_reshape_4d(ctx, input_tensor, 2, input_tensor->ne[0] / 2,
        input_tensor->ne[1], input_tensor->ne[2]);
    rotate_half = ggml_permute(ctx, rotate_half, 3, 1, 2, 0);
    rotate_half = ggml_cont(ctx, rotate_half);
    struct ggml_tensor * positive = ggml_view_4d(ctx, rotate_half, rotate_half->ne[0], rotate_half->ne[1],
        rotate_half->ne[2], 1, rotate_half->nb[1], rotate_half->nb[2], rotate_half->nb[3], 0);
    struct ggml_tensor * negative = ggml_view_4d(ctx, rotate_half, rotate_half->ne[0], rotate_half->ne[1],
        rotate_half->ne[2], 1, rotate_half->nb[1], rotate_half->nb[2], rotate_half->nb[3], rotate_half->nb[3]);
    negative = ggml_scale(ctx, negative, -1);
    rotate_half = ggml_concat(ctx, negative, positive, 3);
    rotate_half = ggml_permute(ctx, rotate_half, 3, 1, 2, 0);
    rotate_half = ggml_cont(ctx, rotate_half);
    rotate_half = ggml_reshape_3d(ctx, rotate_half, 2 * rotate_half->ne[1], rotate_half->ne[2], rotate_half->ne[3]);
    struct ggml_tensor * sin = ggml_mul(ctx, rotate_half, model.rope_freqs_sin);
    return ggml_add(ctx, cos, sin);
}

static struct ggml_cgraph * cross_vision_graph() {
    struct ggml_context * ctx = cogagent_global.cross_vision.ctx_compute;
    cross_vision_ctx &model_ctx = cogagent_global.cross_vision;
    cross_vision &model = cogagent_global.cross_vision.model;

    // Set flag to tell allocator to not overwrite the input tensor
    // before it is done with computation
    ggml_set_input(model.input_image);

    ggml_tensor * patch_embedding = ggml_conv_2d(ctx, model.patch_conv_w, model.input_image,
        14, 14, 0, 0, 1, 1);
    // ggml_repeat should be automatically applied
    // It is required that the tensor to be repeated is the second one
    patch_embedding = ggml_add(ctx, patch_embedding, model.patch_conv_b);
    // From w x h x d x b to l x d x b
    patch_embedding = ggml_reshape_3d(ctx, patch_embedding, patch_embedding->ne[0] * patch_embedding->ne[1],
        patch_embedding->ne[2], patch_embedding->ne[3]);
    patch_embedding = ggml_transpose(ctx, patch_embedding);  // From l x d x b to d x l x b
    patch_embedding = ggml_cont(ctx, patch_embedding);
    // Concatenate the class embedding
    struct ggml_tensor * cls_embed_shape = ggml_new_tensor_3d(ctx, model.cls_embed->type,
        patch_embedding->ne[0], 1, patch_embedding->ne[2]);
    // ggml_concat only supports F32
    patch_embedding = ggml_concat(ctx, ggml_repeat(ctx, model.cls_embed, cls_embed_shape), patch_embedding, 1);
    patch_embedding = ggml_add(ctx, patch_embedding, model.pos_embed_1);

    struct ggml_tensor * layer_in = patch_embedding;
    for (int i=0; i<23; i++) {
        // d x l x b
        cross_vision_layer &cur_layer = model.transformer_layers[i];
        struct ggml_tensor * attention_input = ggml_norm(ctx, layer_in, model.layernorm_eps);
        attention_input = ggml_mul(ctx, attention_input, cur_layer.norm1_w);
        attention_input = ggml_add(ctx, attention_input, cur_layer.norm1_b);
        struct ggml_tensor * qt = ggml_mul_mat(ctx, cur_layer.q_w, attention_input);
        qt = ggml_add(ctx, qt, cur_layer.q_b);
        struct ggml_tensor * kt = ggml_mul_mat(ctx, cur_layer.k_w, attention_input);
        struct ggml_tensor * v = ggml_mul_mat(ctx, cur_layer.v_w, attention_input);
        v = ggml_add(ctx, v, cur_layer.v_b);
        // reshape and permute to k x l x h x b
        qt = ggml_reshape_4d(ctx, qt, qt->ne[0] / model.num_heads, model.num_heads,
            qt->ne[1], qt->ne[2]);
        qt = ggml_permute(ctx, qt, 0, 2, 1, 3);
        qt = ggml_cont(ctx, qt);
        kt = ggml_reshape_4d(ctx, kt, kt->ne[0] / model.num_heads, model.num_heads,
            kt->ne[1], kt->ne[2]);
        kt = ggml_permute(ctx, kt, 0, 2, 1, 3);
        kt = ggml_cont(ctx, kt);
        // for v, reshape and permute to l x k x h x b
        v = ggml_reshape_4d(ctx, v, v->ne[0] / model.num_heads, model.num_heads,
            v->ne[1], v->ne[2]);
        v = ggml_permute(ctx, v, 1, 2, 0, 3);
        v = ggml_cont(ctx, v);

        // At this point, qt and kt are k x l x h x b
        // v is l x k x h x b

        // Process rope for Q and K
        // Remove class embedding before computing rope
        struct ggml_tensor * qtrope = ggml_view_4d(ctx, qt, qt->ne[0], qt->ne[1] - 1,
            qt->ne[2], qt->ne[3], qt->nb[1], qt->nb[2], qt->nb[3], qt->nb[1]);
        struct ggml_tensor * ktrope = ggml_view_4d(ctx, kt, kt->ne[0], kt->ne[1] - 1,
            kt->ne[2], kt->ne[3], kt->nb[1], kt->nb[2], kt->nb[3], kt->nb[1]);
        struct ggml_tensor * qtcls = ggml_view_4d(ctx, qt, qt->ne[0], 1, qt->ne[2],
            qt->ne[3], qt->nb[1], qt->nb[2], qt->nb[3], 0);
        struct ggml_tensor * ktcls = ggml_view_4d(ctx, kt, kt->ne[0], 1, kt->ne[2],
            kt->ne[3], kt->nb[1], kt->nb[2], kt->nb[3], 0);
        qtrope = compute_rope(model_ctx, qtrope);
        ktrope = compute_rope(model_ctx, ktrope);
        qtrope = ggml_concat(ctx, qtcls, qtrope, 1);
        ktrope = ggml_concat(ctx, ktcls, ktrope, 1);

        struct ggml_tensor * qtscale = ggml_scale(ctx, qtrope, model.attn_scale);

        // Calculate attention score
        // L x L x H x B
        struct ggml_tensor * attnt = ggml_mul_mat(ctx, ktrope, qtscale);
        attnt = ggml_soft_max(ctx, attnt);
        attnt = ggml_mul_mat(ctx, v, attnt);  // k x l x h x b
        attnt = ggml_permute(ctx, attnt, 0, 2, 1, 3);  // k x h x l x b
        attnt = ggml_cont(ctx, attnt);
        attnt = ggml_reshape_3d(ctx, attnt, attnt->ne[0] * attnt->ne[1],
            attnt->ne[2], attnt->ne[3]);
        attnt = ggml_norm(ctx, attnt, model.layernorm_eps);
        attnt = ggml_mul(ctx, attnt, cur_layer.attn_ln_w);
        attnt = ggml_add(ctx, attnt, cur_layer.attn_ln_b);
        attnt = ggml_mul_mat(ctx, cur_layer.attn_linear_w, attnt);
        attnt = ggml_add(ctx, attnt, cur_layer.attn_linear_b);

        layer_in = ggml_add(ctx, layer_in, attnt);

        // MLP calculation
        struct ggml_tensor * mlp_tensor = ggml_norm(ctx, layer_in, model.layernorm_eps);
        mlp_tensor = ggml_mul(ctx, mlp_tensor, cur_layer.norm2_w);
        mlp_tensor = ggml_add(ctx, mlp_tensor, cur_layer.norm2_b);
        struct ggml_tensor * w1 = ggml_mul_mat(ctx, cur_layer.mlp_linear1_w, mlp_tensor);
        w1 = ggml_add(ctx, w1, cur_layer.mlp_linear1_b);
        struct ggml_tensor * w2 = ggml_mul_mat(ctx, cur_layer.mlp_linear2_w, mlp_tensor);
        w2 = ggml_add(ctx, w2, cur_layer.mlp_linear2_b);
        w1 = ggml_silu(ctx, w1);
        mlp_tensor = ggml_mul(ctx, w1, w2);  // MLP hidden size is 2730
        mlp_tensor = ggml_norm(ctx, mlp_tensor, model.layernorm_eps);
        mlp_tensor = ggml_mul(ctx, mlp_tensor, cur_layer.mlp_ln_w);
        mlp_tensor = ggml_add(ctx, mlp_tensor, cur_layer.mlp_ln_b);
        mlp_tensor = ggml_mul_mat(ctx, cur_layer.mlp_linear3_w, mlp_tensor);
        mlp_tensor = ggml_add(ctx, mlp_tensor, cur_layer.mlp_linear3_b);

        layer_in = ggml_add(ctx, layer_in, mlp_tensor);
    }

    model.output_tensor = ggml_view_3d(ctx, layer_in, layer_in->ne[0], layer_in->ne[1] - 1,
        layer_in->ne[2], layer_in->nb[1], layer_in->nb[2], layer_in->nb[1]);
    model.output_tensor = ggml_add(ctx, model.output_tensor, model.pos_embed_2);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, model.output_tensor);

    // Copy the output tensor to the token context
    ggml_build_forward_expand(gf, ggml_cpy(ctx, model.output_tensor,
        cogagent_global.cross_vision_image_tensor));

    return gf;
}

void run_cross_vision(std::vector<float> img_data) {
    cross_vision_ctx &model_ctx = cogagent_global.cross_vision;
    cross_vision &model = cogagent_global.cross_vision.model;

    // Declare the input image tensor
    model.input_image = ggml_new_tensor_3d(cogagent_global.cross_vision.ctx_compute, GGML_TYPE_F32,
        cogagent_global.cross_vision_img_size, cogagent_global.cross_vision_img_size, 3);

    struct ggml_cgraph * gf = cross_vision_graph();
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

    save_tensor_filename(model.output_tensor, "cogagent_encoders/cross_vision_output.gguf");
}

void free_cross_vision_ctx() {
    cross_vision_ctx &model_ctx = cogagent_global.cross_vision;

    ggml_gallocr_free(model_ctx.allocr);
    ggml_backend_buffer_free(model_ctx.weight_data);
    ggml_free(model_ctx.ctx_weight);
    ggml_free(model_ctx.ctx_compute);
}