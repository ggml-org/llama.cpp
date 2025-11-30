#include "llamax.h"

#include <format>
#include <functional>
#include <string>

#include "llama.h"

using namespace llamax;

uint32_t llamax::default_seed() {
    return LLAMA_DEFAULT_SEED;
}

//      _       _
//   __| | __ _| |_ __ _
//  / _` |/ _` | __/ _` |
// | (_| | (_| | || (_| |
//  \__,_|\__,_|\__\__,_|

struct model_params::data {
    data() : model_params(llama_model_default_params()) {}

    llama_model_params model_params;
};

struct model::data {
    ~data() { llama_free_model(model); }

    llama_model * model = nullptr;
};

struct iterator::data {
    data(const context & _ctx) : ctx(_ctx) {}

    context ctx;
};

struct context_params::data {
    data() : ctx_params(llama_context_default_params()) {}

    llama_context_params ctx_params;
};

struct sampler_builder::data {
    ~data() {}

    std::vector<std::function<llama_sampler *(llama_model *)>>   builders;
    std::optional<std::function<llama_sampler *(llama_model *)>> grammar_builder;
    bool                                                         grammar_first = false;
};

struct context::data {
    ~data() {
        llama_free(ctx);
        llama_sampler_free(sampler);
    }

    llamax::model        model;
    llama_context_params ctx_params;
    llama_context *      ctx;
    llama_sampler *      sampler;
    llama_sampler *      grammar_sampler;
    bool                 grammar_first = false;
};

struct chat_template::data {
    llamax::model     model;
    std::vector<char> buffer;
};

//                      _      _
//  _ __ ___   ___   __| | ___| |     _ __   __ _ _ __ __ _ _ __ ___  ___
// | '_ ` _ \ / _ \ / _` |/ _ \ |    | '_ \ / _` | '__/ _` | '_ ` _ \/ __|
// | | | | | | (_) | (_| |  __/ |    | |_) | (_| | | | (_| | | | | | \__ \
// |_| |_| |_|\___/ \__,_|\___|_|____| .__/ \__,_|_|  \__,_|_| |_| |_|___/
//                             |_____|_|

model_params model_params::default_params() {
    model_params p;
    p.d = std::make_unique<model_params::data>();
    return p;
}

model_params & model_params::set_n_gpu_layers(unsigned _n_gpu_layers) {
    d->model_params.n_gpu_layers = _n_gpu_layers;
    return *this;
}

//                  _            _
//   ___ ___  _ __ | |_ _____  __ |_      _ __   __ _ _ __ __ _ _ __ ___  ___
//  / __/ _ \| '_ \| __/ _ \ \/ / __|    | '_ \ / _` | '__/ _` | '_ ` _ \/ __|
// | (_| (_) | | | | ||  __/>  <| |_     | |_) | (_| | | | (_| | | | | | \__ \
//  \___\___/|_| |_|\__\___/_/\_\\__|____| .__/ \__,_|_|  \__,_|_| |_| |_|___/
//                                 |_____|_|

context_params context_params::default_params() {
    context_params p;
    p.d                     = std::make_unique<context_params::data>();
    p.d->ctx_params.no_perf = false;
    return p;
}

context_params & context_params::set_context_size(unsigned _context_size) {
    d->ctx_params.n_ctx = _context_size;
    return *this;
}

context_params & context_params::set_batch_size(unsigned _batch_size) {
    d->ctx_params.n_batch = _batch_size;
    return *this;
}

//                            _            _           _ _     _
//  ___  __ _ _ __ ___  _ __ | | ___ _ __ | |__  _   _(_) | __| | ___ _ __
// / __|/ _` | '_ ` _ \| '_ \| |/ _ \ '__|| '_ \| | | | | |/ _` |/ _ \ '__|
// \__ \ (_| | | | | | | |_) | |  __/ |   | |_) | |_| | | | (_| |  __/ |
// |___/\__,_|_| |_| |_| .__/|_|\___|_|___|_.__/ \__,_|_|_|\__,_|\___|_|
//                     |_|           |_____|

sampler_builder::sampler_builder() : d(std::make_unique<data>()) {}

sampler_builder::~sampler_builder() {}

sampler_builder & sampler_builder::top_k(int32_t _k) {
    d->builders.push_back([_k](llama_model *) { return llama_sampler_init_top_k(_k); });
    return *this;
}

sampler_builder & sampler_builder::top_p(float p, size_t min_keep) {
    d->builders.push_back([p, min_keep](llama_model *) { return llama_sampler_init_top_p(p, min_keep); });
    return *this;
}

sampler_builder & sampler_builder::min_p(float p, size_t min_keep) {
    d->builders.push_back([p, min_keep](llama_model *) { return llama_sampler_init_min_p(p, min_keep); });
    return *this;
}

sampler_builder & sampler_builder::grammar(const std::string & _grammar, const std::string & _root) {
    d->builders.push_back([_grammar, _root](llama_model * _model) {
        const llama_vocab * vocab = llama_model_get_vocab(_model);
        return llama_sampler_init_grammar(vocab, _grammar.c_str(), _root.c_str());
    });
    return *this;
}

sampler_builder & sampler_builder::temp(float t) {
    d->builders.push_back([t](llama_model *) { return llama_sampler_init_temp(t); });
    return *this;
}

sampler_builder & sampler_builder::greedy() {
    d->builders.push_back([](llama_model *) { return llama_sampler_init_greedy(); });
    return *this;
}

sampler_builder & sampler_builder::dist(uint32_t seed) {
    d->builders.push_back([seed](llama_model *) { return llama_sampler_init_dist(seed); });
    return *this;
}

//                      _      _
//  _ __ ___   ___   __| | ___| |
// | '_ ` _ \ / _ \ / _` |/ _ \ |
// | | | | | | (_) | (_| |  __/ |
// |_| |_| |_|\___/ \__,_|\___|_|

model model::load_from_file(const std::string & _name, const model_params & _params) {
    model m;
    m.d        = std::make_shared<model::data>();
    m.d->model = llama_model_load_from_file(_name.c_str(), _params.d->model_params);
    if (m.d->model) {
        return m;
    } else {
        throw exception("Unable to load model from file: " + _name);
    }
}

context model::create_context(const context_params & _context_params, const sampler_builder & _sampler_builder) const {
    context ctx;
    ctx.d             = std::make_shared<context::data>();
    ctx.d->model      = *this;
    ctx.d->ctx_params = _context_params.d->ctx_params;
    ctx.d->ctx        = llama_new_context_with_model(d->model, ctx.d->ctx_params);

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf                    = false;
    ctx.d->sampler                     = llama_sampler_chain_init(sparams);

    for (const std::function<llama_sampler *(llama_model *)> & f : _sampler_builder.d->builders) {
        llama_sampler_chain_add(ctx.d->sampler, f(d->model));
    }

    return ctx;
}

chat_template model::create_chat_template(bool _add_assistant) const {
    chat_template ct;
    ct.d        = std::make_shared<chat_template::data>();
    ct.d->model = *this;
    return ct;
}

//  _ _                 _
// (_) |_ ___ _ __ __ _| |_ ___  _ __
// | | __/ _ \ '__/ _` | __/ _ \| '__|
// | | ||  __/ | | (_| | || (_) | |
// |_|\__\___|_|  \__,_|\__\___/|_|

iterator::iterator() : d(nullptr) {}

iterator::iterator(iterator && _rhs) : d(std::move(_rhs.d)) {}

iterator::~iterator() {}

std::optional<std::string> iterator::next() {
    llama_token         new_token_id = llama_sampler_sample(d->ctx.d->sampler, d->ctx.d->ctx, -1);
    const llama_vocab * vocab        = llama_model_get_vocab(d->ctx.d->model.d->model);
    if (llama_vocab_is_eog(vocab, new_token_id)) {
        return std::nullopt;
    }
    char buf[128];
    int  n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        throw exception("Failed to convert token " + std::to_string(new_token_id) + " to piece.");
    }
    std::string s(buf, n);

    // prepare the next batch with the sampled token
    auto    batch         = llama_batch_get_one(&new_token_id, 1);
    int32_t decode_result = llama_decode(d->ctx.d->ctx, batch);

    if (decode_result) {
        throw exception("Failed to eval, return code: " + std::to_string(decode_result));
    }
    return s;
}

//                  _            _
//   ___ ___  _ __ | |_ _____  __ |_
//  / __/ _ \| '_ \| __/ _ \ \/ / __|
// | (_| (_) | | | | ||  __/>  <| |_
//  \___\___/|_| |_|\__\___/_/\_\\__|

iterator context::prompt(const std::string & _prompt) {
    const llama_vocab * vocab = llama_model_get_vocab(d->model.d->model);

    //// First tokenize the prompt
    // Get the number of tokens
    const int n_prompt = -llama_tokenize(vocab, _prompt.c_str(), _prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, _prompt.c_str(), _prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) <
        0) {
        throw exception("Failed to tokenize the prompt: " + _prompt);
    }

    //// Prepare batch
    const unsigned batch_size = d->ctx_params.n_batch;

    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    //// Consume the prompt
    for (int32_t batch_first_token = 0; batch_first_token < n_prompt; batch_first_token += batch_size) {
        const int32_t remaining     = n_prompt - batch_first_token;
        int32_t       count_to_eval = std::min<int32_t>(remaining, batch_size);

        llama_batch batch = llama_batch_get_one(&prompt_tokens[batch_first_token], count_to_eval);

        int32_t decode_result = llama_decode(d->ctx, batch);

        if (decode_result) {
            throw exception("Failed to eval, return code: " + std::to_string(decode_result));
        }
    }

    iterator it;
    it.d = std::make_unique<iterator::data>(*this);
    return it;
}

//       _           _      _                       _       _
//   ___| |__   __ _| |_   | |_ ___ _ __ ___  _ __ | | __ _| |_ ___
//  / __| '_ \ / _` | __|  | __/ _ \ '_ ` _ \| '_ \| |/ _` | __/ _ \
// | (__| | | | (_| | |_   | ||  __/ | | | | | |_) | | (_| | ||  __/
//  \___|_| |_|\__,_|\__|___\__\___|_| |_| |_| .__/|_|\__,_|\__\___|
//                     |_____|               |_|

std::string chat_template::generate(const std::vector<chat_message> & _messages) {
    std::vector<llama_chat_message> messages;
    for (const chat_message & msg : _messages) {
        const char * role = nullptr;
        switch (msg.role) {
            case chat_message_role::assistant:
                role = "assistant";
                break;
            case chat_message_role::system:
                role = "system";
                break;
            case chat_message_role::user:
                role = "user";
                break;
        }
        if (not role) {
            throw exception("Unknown role.");
        }
        messages.push_back({ role, msg.content.c_str() });
    }
    const char * tmpl = llama_model_chat_template(d->model.d->model, /* name */ nullptr);

    int new_len =
        llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, d->buffer.data(), d->buffer.size());
    if (new_len > (int) d->buffer.size()) {
        d->buffer.resize(new_len);
        new_len =
            llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, d->buffer.data(), d->buffer.size());
    }
    if (new_len < 0) {
        throw exception("Failed to apply template to messages.");
    }

    return std::string(d->buffer.data(), d->buffer.size());
}
