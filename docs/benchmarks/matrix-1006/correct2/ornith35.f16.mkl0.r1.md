[New LWP 869090 (id 2)]
[New LWP 869089 (id 3)]
[New LWP 869088 (id 4)]
[New LWP 869087 (id 5)]
[New LWP 869086 (id 6)]
[New LWP 869085 (id 7)]
[New LWP 869084 (id 8)]
[New LWP 869083 (id 9)]
[New LWP 869082 (id 10)]
[New LWP 869081 (id 11)]
[New LWP 869080 (id 12)]
[New LWP 867988 (id 13)]
[New LWP 867980 (id 14)]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
[Switching to thread 1 (Thread 0x74277ee9ce00 (LWP 867972))]
0x00007427815828f2 in ?? () from /usr/lib/libc.so.6
#0  0x00007427815828f2 in ?? () from /usr/lib/libc.so.6
#1  0x00007427815e74cb in wait4 () from /usr/lib/libc.so.6
#2  0x0000742781b53d8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x0000742781b52eb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x00007427823c4f78 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x0000742782427f2e in ggml_sycl_pool_vmm::alloc(unsigned long, unsigned long*) () from /usr/lib/libggml-sycl.so.0
#6  0x000074278242bb31 in ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, sycl::_V1::queue* const&) () from /usr/lib/libggml-sycl.so.0
#7  0x00007427823f27d5 in ?? () from /usr/lib/libggml-sycl.so.0
#8  0x00007427823e7bf2 in ?? () from /usr/lib/libggml-sycl.so.0
#9  0x00007427823dfe7c in ?? () from /usr/lib/libggml-sycl.so.0
#10 0x00007427823dc048 in ?? () from /usr/lib/libggml-sycl.so.0
#11 0x0000742781b7ef15 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#12 0x00007427a48c21a1 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#13 0x00007427a48c14c9 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#14 0x00007427a48c3d89 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#15 0x00007427a48c97c9 in llama_decode () from /usr/lib/libllama.so.0
#16 0x00007427a4fb3f35 in ?? () from /usr/lib/libllama-perplexity-impl.so
#17 0x00007427a4fa86ee in llama_perplexity(int, char**) () from /usr/lib/libllama-perplexity-impl.so
#18 0x0000742781509781 in ?? () from /usr/lib/libc.so.6
#19 0x00007427815098b9 in __libc_start_main () from /usr/lib/libc.so.6
#20 0x0000000000406405 in ?? ()
[Inferior 1 (process 867972) detached]
