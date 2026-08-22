#include "arg.h"
#include "cli-params.h"
#include "common.h"

#undef NDEBUG
#include <cassert>

static void test_cli_params_finalize(void) {
    {
        common_params params;
        params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        params.prompt = "hello";
        params.single_turn = false;

        cli_params_finalize(params);

        assert(params.single_turn);
    }

    {
        common_params params;
        params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        params.prompt.clear();
        params.single_turn = false;

        cli_params_finalize(params);

        assert(!params.single_turn);
    }

    {
        common_params params;
        params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        params.prompt = "hello";
        params.single_turn = false;

        cli_params_finalize(params);

        assert(!params.single_turn);
    }

    {
        common_params params;
        params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        params.prompt = "hello";
        params.single_turn = true;

        cli_params_finalize(params);

        assert(params.single_turn);
    }
}

static void test_cli_run_requires_prompt(void) {
    {
        common_params params;
        params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        params.prompt.clear();

        assert(cli_run_requires_prompt(params));
    }

    {
        common_params params;
        params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        params.prompt = "hello";

        assert(!cli_run_requires_prompt(params));
    }

    {
        common_params params;
        params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        params.prompt.clear();

        assert(!cli_run_requires_prompt(params));
    }
}

static void test_cli_no_cnv_arg_parse(void) {
    std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "-no-cnv", "-p", "hello"};
    std::vector<char *> argv_ptr;
    for (auto & arg : argv) {
        argv_ptr.push_back(arg.data());
    }

    common_params params;
    assert(common_params_parse(argv_ptr.size(), argv_ptr.data(), params, LLAMA_EXAMPLE_CLI));
    assert(params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED);
    assert(params.prompt == "hello");

    cli_params_finalize(params);
    assert(params.single_turn);
}

int main(void) {
    test_cli_params_finalize();
    test_cli_run_requires_prompt();
    test_cli_no_cnv_arg_parse();
    return 0;
}
