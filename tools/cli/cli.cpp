#include "arg.h"
#include "common.h"
#include "log.h"

#include "cli-context.h"

#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (cli_context::interrupted().load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    cli_context::interrupted().store(true);
}
#endif

// satisfies -Wmissing-declarations
int llama_cli(int argc, char ** argv);

int llama_cli(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    cli_context ctx_cli(params);

    if (!ctx_cli.init()) {
        return 1;
    }

    return ctx_cli.run();
}
