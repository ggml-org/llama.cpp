// main() of tests/test-backend-ops.cpp is compiled as test_backend_ops_main()
int test_backend_ops_main(int argc, char ** argv);

extern "C" int ggml_run_test_backend_ops(int argc, char ** argv) {
    return test_backend_ops_main(argc, argv);
}
