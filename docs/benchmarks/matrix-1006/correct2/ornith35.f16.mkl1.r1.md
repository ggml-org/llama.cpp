[New LWP 867168 (id 2)]
[New LWP 867167 (id 3)]
[New LWP 867166 (id 4)]
[New LWP 867165 (id 5)]
[New LWP 867164 (id 6)]
[New LWP 867163 (id 7)]
[New LWP 867162 (id 8)]
[New LWP 867161 (id 9)]
[New LWP 867160 (id 10)]
[New LWP 867159 (id 11)]
[New LWP 867158 (id 12)]
[New LWP 865887 (id 13)]
[New LWP 865879 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x7c82ebcf6e00 (LWP 865871))]
0x00007c82f9d428f2 in ?? () from /usr/lib/libc.so.6
#0  0x00007c82f9d428f2 in ?? () from /usr/lib/libc.so.6
#1  0x00007c82f9da74cb in wait4 () from /usr/lib/libc.so.6
#2  0x00007c82fa313d8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x00007c82fa312eb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x00007c82fab84f78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x00007c82fabe7f2e in ggml_sycl_pool_vmm::alloc(unsigned long, unsigned long*) () from /usr/lib/libggml-sycl.so.0
#6  0x00007c82fabebb31 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#7  0x00007c82fabb27d5 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x00007c82faba7bf2 in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x00007c82fab9fe7c in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x00007c82fab9c048 in ?? () from /usr/lib/libggml-sycl.so.0
#11 0x00007c82fa33ef15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#12 0x00007c831d0821a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#13 0x00007c831d0814c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#14 0x00007c831d083d89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#15 0x00007c831d0897c9 in llama_decode () from /usr/lib/libllama.so.0
#16 0x00007c831d773f35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#17 0x00007c831d7686ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#18 0x00007c82f9cc9781 in ?? () from /usr/lib/libc.so.6
#19 0x00007c82f9cc98b9 in __libc_start_main () from /usr/lib/libc.so.6
#20 0x0000000000406405 in ?? ()
[Inferior 1 (process 865871) detached]
