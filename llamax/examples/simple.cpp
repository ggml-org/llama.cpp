#include <llamax.h>

#include <iostream>

int main(int _argc, const char ** _argv) {
    if (_argc != 3) {
        std::cerr << "llamax_simple [model] \"What is up doctor?\"" << std::endl;
        return -1;
    }

    llamax::model   model = llamax::model::load_from_file(_argv[1], llamax::model_params::default_params());
    llamax::context ctx =
        model.create_context(llamax::context_params::default_params(), llamax::sampler_builder().greedy());
    llamax::iterator it = ctx.prompt(_argv[2]);

    while (std::optional<std::string> s = it.next()) {
        std::cout << s.value();
    }
    std::cout << std::endl;

    return 0;
}
