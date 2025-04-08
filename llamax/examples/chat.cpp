#include <llamax.h>

#include <iostream>

int main(int _argc, const char ** _argv) {
    if (_argc != 2) {
        std::cerr << "llamax_chat [model]" << std::endl;
        return -1;
    }

    llamax::model   model = llamax::model::load_from_file(_argv[1], llamax::model_params::default_params());
    llamax::context ctx =
        model.create_context(llamax::context_params::default_params(),
                             llamax::sampler_builder().min_p(0.05, 1).temp(0.8f).dist(llamax::default_seed()));
    llamax::chat_template             ct = model.create_chat_template();
    std::vector<llamax::chat_message> messages;

    messages.push_back({ llamax::chat_message_role::system, "You are an assistant." });

    int offset = 0;

    while (true) {
        printf("\033[32m> \033[0m");
        std::string user;
        std::getline(std::cin, user);

        if (user.empty()) {
            break;
        }

        messages.push_back({ llamax::chat_message_role::user, user });

        std::string prompt = ct.generate(messages);

        std::string      answer;
        llamax::iterator it = ctx.prompt(prompt.substr(offset));
        offset              = prompt.size();

        while (std::optional<std::string> s = it.next()) {
            answer += s.value();
            std::cout << s.value();
        }
        std::cout << std::endl;

        messages.push_back({ llamax::chat_message_role::assistant, answer });
    }

    return 0;
}
