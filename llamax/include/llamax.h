#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llamax {
uint32_t default_seed();

class context;
class chat_template;
class model;
class sampler_builder;

/// Exception class for errors in llamax
class exception : public std::exception {
    friend class chat_template;
    friend class context;
    friend class iterator;
    friend class model;

    exception(const std::string & what) : m_what(what) {}
  public:
    const char * what() const noexcept override { return m_what.c_str(); }
  private:
    std::string m_what;
};

/// Parameters for a llama models
class model_params {
    friend class model;
  public:
    static model_params default_params();
    /// Set the number of layers offset to a GPU
    model_params &      set_n_gpu_layers(unsigned _n_gpu_layers);
  private:
    struct data;
    std::shared_ptr<data> d;
};

/// Parameters for the context
class context_params {
    friend class model;
  public:
    static context_params default_params();
    /// Set the context size
    context_params &      set_context_size(unsigned _context_size);
    // batch_size is the maximum number of tokens that can be processed in a single call to
    // llama_decode
    context_params &      set_batch_size(unsigned _batch_size);
  private:
    struct data;
    std::shared_ptr<data> d;
};

class sampler_builder {
    friend class model;
  public:
    sampler_builder();
    ~sampler_builder();
    sampler_builder & top_k(int32_t _k);
    sampler_builder & top_p(float p, size_t min_keep);
    sampler_builder & min_p(float p, size_t min_keep);
    sampler_builder & grammar(const std::string & _grammar, const std::string & _root);
    sampler_builder & temp(float t);
    sampler_builder & greedy();
    sampler_builder & dist(uint32_t seed);
  private:
    struct data;
    std::unique_ptr<data> d;
};
class context;

class model {
    friend class iterator;
    friend class context;
    friend class chat_template;
  public:
    /**
     * Attempt to load a model from a file.
     *
     * This function can trigger an exception.
     */
    static model load_from_file(const std::string & _name, const model_params & _params);
    /**
     * Create a context that can be used to generate text based on a prompt.
     */
    context      create_context(const context_params & _context_params, const sampler_builder & _sampler_builder) const;
    /**
     * Create a chat template that can be used to generate the prompt for a chat bot.
     */
    chat_template create_chat_template(bool _add_assistant = true) const;
  private:
    struct data;
    std::shared_ptr<data> d;
};

class iterator {
    friend class context;
    iterator();
  public:
    iterator(iterator && _rhs);
    ~iterator();
    /**
     * Return the next token, or null, if no more tokens.
     *
     * This function can trigger an exception.
     */
    std::optional<std::string> next();
  private:
    struct data;
    std::unique_ptr<data> d;
};

class context {
    friend class model;
    friend class iterator;
  public:
    /**
     * Prompt the llm.
     */
    iterator prompt(const std::string & _prompt);
  private:
    struct data;
    std::shared_ptr<data> d;
};
enum class chat_message_role { system, user, assistant };

struct chat_message {
    chat_message_role role;
    std::string       content;
};

class chat_template {
    friend class model;
  public:
    /**
     * Generate a prompt based on a template and a set of messages.
     */
    std::string generate(const std::vector<chat_message> & _messages);
  private:
    struct data;
    std::shared_ptr<data> d;
};
}  // namespace llamax
