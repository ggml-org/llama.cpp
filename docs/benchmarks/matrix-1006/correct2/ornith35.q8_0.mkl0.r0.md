[New LWP 845376 (id 2)]
[New LWP 845375 (id 3)]
[New LWP 845374 (id 4)]
[New LWP 845373 (id 5)]
[New LWP 845372 (id 6)]
[New LWP 845371 (id 7)]
[New LWP 845370 (id 8)]
[New LWP 845369 (id 9)]
[New LWP 845368 (id 10)]
[New LWP 845367 (id 11)]
[New LWP 845366 (id 12)]
[New LWP 844562 (id 13)]
[New LWP 844552 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x7b950cd22e00 (LWP 844544))]
0x00007b951ad6f8f2 in ?? () from /usr/lib/libc.so.6
#0  0x00007b951ad6f8f2 in ?? () from /usr/lib/libc.so.6
#1  0x00007b951add44cb in wait4 () from /usr/lib/libc.so.6
#2  0x00007b951b340d8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x00007b951b33feb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x00007b951bbb1f78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x00007b951bc19117 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#6  0x00007b951bbdf7d5 in ?? () from /usr/lib/libggml-sycl.so.0
#7  0x00007b951bbd4bf2 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x00007b951bbcce7c in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x00007b951bbc9048 in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x00007b951b36bf15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#11 0x00007b953e0af1a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#12 0x00007b953e0ae4c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#13 0x00007b953e0b0d89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#14 0x00007b953e0b67c9 in llama_decode () from /usr/lib/libllama.so.0
#15 0x00007b953e7a0f35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#16 0x00007b953e7956ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#17 0x00007b951acf6781 in ?? () from /usr/lib/libc.so.6
#18 0x00007b951acf68b9 in __libc_start_main () from /usr/lib/libc.so.6
#19 0x0000000000406405 in ?? ()
[Inferior 1 (process 844544) detached]
