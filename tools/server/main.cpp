int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
    return llama_server(argc, argv);
}

// satisfies -Wmissing-declarations
void server_signal_handler(int signal);

void server_signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}
