// the clip engine (src/clip/) inside the clip arch: weights through llama_model_loader, graphs inside llm_graph_context

#include "models.h"

#include "llama-impl.h"
#include "llama-media.h"
#include "llama-model-loader.h"

#include "../clip/clip-graph.h"
#include "../clip/clip-model.h"
#include "../clip/clip-subcontext.h"

#include <algorithm>
#include <map>
#include <stdexcept>

llama_model_clip::llama_model_clip(const struct llama_model_params & params) : llama_model_base(params) {}

llama_model_clip::~llama_model_clip() = default;

const clip_model * llama_model_clip::engine_vision() const { return engine ? engine->v.get()     : nullptr; }
const clip_model * llama_model_clip::engine_audio()  const { return engine ? engine->a.get()     : nullptr; }
const clip_model * llama_model_clip::engine_gen()    const { return engine ? engine->gen_a.get() : nullptr; }

const clip_model * llama_model_clip::engine_for_ctx(enum llama_context_type type) const {
    if (type == LLAMA_CONTEXT_TYPE_ENCODER) {
        return engine_vision() ? engine_vision() : engine_audio();
    }
    if (type == LLAMA_CONTEXT_TYPE_GEN_CODE || type == LLAMA_CONTEXT_TYPE_GEN_WAV) {
        return engine_gen();
    }
    return nullptr;
}

bool llama_model_clip::engine_flash_attn(const clip_model & m, const llama_cparams & cparams) {
    return cparams.flash_attn && (m.flash_attn_auto || !cparams.auto_fa_user);
}

uint32_t llama_model_clip::graph_max_nodes_hint() const {
    // native graphs: vision towers are wide (mtmd used 8192), the code predictor unrolls 15 steps with a sampler chain each
    uint32_t res = vhp.has_vision ? 8192 : 32768;
    for (const clip_model * m : { engine_vision(), engine_audio(), engine_gen() }) {
        if (m) {
            res = std::max(res, (uint32_t) m->max_nodes);
        }
    }
    return res;
}

// llama_hparams of a file whose main tower runs on the engine: they only place the weights and size the context
void llama_model_clip::load_engine_hparams(const clip_model & main) {
    const auto & hp = main.hparams;

    hparams.n_embd      = std::max(hp.n_embd, 1);
    hparams.n_layer_all = std::clamp<uint32_t>(hp.n_layer, 1, LLAMA_MAX_LAYERS);
    hparams.f_norm_eps     = hp.eps;
    hparams.f_norm_rms_eps = hp.eps;

    const uint32_t n_head    = std::max(hp.n_head, 1);
    const uint32_t n_head_kv = hp.n_head_kv > 0 ? hp.n_head_kv : n_head;
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.begin()    + hparams.n_layer_all, n_head);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.begin() + hparams.n_layer_all, n_head_kv);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.begin()      + hparams.n_layer_all, (uint32_t) std::max(hp.n_ff, 1));

    const uint32_t head_dim = hp.n_embd_head > 0 ? hp.n_embd_head : std::max<uint32_t>(hparams.n_embd / n_head, 1);
    hparams.n_embd_head_k_full = head_dim;
    hparams.n_embd_head_v_full = head_dim;
    hparams.n_rot_full         = 0;
}

void llama_model_clip::load_engine_tensors(llama_model_loader & ml) {
    // the weights of the native graphs are shared with the engine: same name, same tensor
    std::map<std::string, ggml_tensor *> created;
    for (const auto & it : ml.ctx_map) {
        for (ggml_tensor * t = ggml_get_first_tensor(it.second.get()); t; t = ggml_get_next_tensor(it.second.get(), t)) {
            created[ggml_get_name(t)] = t;
        }
    }

    auto create = [&](const std::string & name, bool required) -> ggml_tensor * {
        auto it = created.find(name);
        if (it != created.end()) {
            return it->second;
        }
        const ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        if (meta == nullptr) {
            if (required) {
                throw std::runtime_error(format("clip: missing tensor '%s'", name.c_str()));
            }
            return nullptr;
        }
        const auto tn = LLM_TN_IMPL::named(LLM_ARCH_CLIP, name);
        ggml_tensor * t = nullptr;
        switch (ggml_n_dims(meta)) {
            case 1:  t = create_tensor(tn, {meta->ne[0]}, 0); break;
            case 2:  t = create_tensor(tn, {meta->ne[0], meta->ne[1]}, 0); break;
            case 3:  t = create_tensor(tn, {meta->ne[0], meta->ne[1], meta->ne[2]}, 0); break;
            default: t = create_tensor(tn, {meta->ne[0], meta->ne[1], meta->ne[2], meta->ne[3]}, 0); break;
        }
        created[name] = t;
        return t;
    };

    if (engine->v) {
        clip_model_load_tensors(ml, *engine->v, create);
    }
    if (engine->a) {
        clip_model_load_tensors(ml, *engine->a, create);
    }
    if (engine->gen_a) {
        clip_model_load_tensors(ml, *engine->gen_a, create);
    }
}

//
// graph
//

// sets the inputs of an engine graph by name, see clip_subcontext::set_inputs()
class llm_graph_input_clip : public llm_graph_input_i {
public:
    llm_graph_input_clip(clip_subcontext * sub, clip_subctx_encode_params * req, ggml_cgraph * gf) : sub(sub), req(req), gf(gf) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        sub->input_ok = sub->set_inputs(gf, req);
    }

    clip_subcontext           * sub;
    clip_subctx_encode_params * req;
    ggml_cgraph               * gf;
};

struct llama_model_clip::graph_engine : public llm_graph_context {
    graph_engine(const llama_model_clip & clip, const llm_graph_params & params) : llm_graph_context(params) {
        const llama_media_input * mi = params.ubatch.media;

        clip_subcontext           * sub = mi ? mi->clip_sub : nullptr;
        clip_subctx_encode_params * req = mi ? mi->clip_req : nullptr;

        // reservation: the graph of the largest input of the main tower of this context
        std::unique_ptr<clip_subcontext> sub_reserve;
        if (sub == nullptr) {
            const clip_model * cm = clip.engine_for_ctx(cparams.ctx_type);
            GGML_ASSERT(cm != nullptr);
            sub_reserve = std::make_unique<clip_subcontext>(*cm, engine_flash_attn(*cm, cparams));
            sub = sub_reserve.get();
        }

        sub->flash_attn_type = engine_flash_attn(sub->model, cparams) ? CLIP_FLASH_ATTN_TYPE_ENABLED : CLIP_FLASH_ATTN_TYPE_DISABLED;
        sub->ctx0 = ctx0;
        sub->gf   = gf;

        ggml_cgraph * g = nullptr;
        if (req) {
            g = sub->build(req);
        } else {
            const clip_image_f32_batch batch = sub->get_dummy_batch();
            g = clip_get_graph_builder(sub, batch, nullptr)->build();
        }
        GGML_ASSERT(g == gf);

        sub->ctx0 = nullptr;
        sub->gf   = nullptr;

        // the last node is the embedding output, read it here, before scheduling may reorder the nodes
        // graphs with alternative branches (ggml_build_forward_select) have no embedding output in the other branches
        if (req == nullptr || req->out_embd != nullptr) {
            res->t_embd = ggml_graph_node(gf, -1);
        }

        if (req) {
            res->add_input(std::make_unique<llm_graph_input_clip>(sub, req, gf));
        }

        // flash attention nodes, for the auto-detection of llama_context (layers share one device)
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                res->add_fused_node({ LLM_FUSED_OP_FLASH_ATTN, node, 0 });
            }
        }
    }
};

std::unique_ptr<llm_graph_context> llama_model_clip_build_engine_graph(const llama_model_clip & clip, const llm_graph_params & params) {
    return std::make_unique<llama_model_clip::graph_engine>(clip, params);
}
