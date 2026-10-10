| model                          |       size |     params | backend    | ngl |   fit | type_k | type_v |  fa |       fitt |        fitc |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ----: | -----: | -----: | --: | ---------: | ----------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |           pp512 |        200.42 ± 9.03 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |            tg64 |         12.87 ± 0.08 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.21 GiB |    35.51 B | SYCL       |  41 |     1 |   q8_0 |   q8_0 |   1 |       1024 |       32768 |   pp512 @ d8192 |        168.44 ± 2.07 |
[New LWP 260429]
[New LWP 260428]
[New LWP 260427]
[New LWP 260426]
[New LWP 260425]
[New LWP 260424]
[New LWP 260423]
[New LWP 260422]
[New LWP 260421]
[New LWP 260420]
[New LWP 260419]
[New LWP 260418]
[New LWP 260413]
Registering SYCL extensions for gdb
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/libthread_db.so.1".
0x000075b5035cb8f2 in ?? () from /usr/lib/libc.so.6
#0  0x000075b5035cb8f2 in ?? () from /usr/lib/libc.so.6
#1  0x000075b5036304cb in wait4 () from /usr/lib/libc.so.6
#2  0x000075b503b9cd8a in ggml_print_backtrace () from /usr/lib/libggml-base.so.0
#3  0x000075b503b9beb9 in ggml_abort () from /usr/lib/libggml-base.so.0
#4  0x000075b5043bc818 in ?? () from /usr/lib/libggml-sycl.so.0
#5  0x000075b5043d1a28 in ?? () from /usr/lib/libggml-sycl.so.0
#6  0x000075b503bc7b23 in ggml_backend_sched_graph_compute_async () from /usr/lib/libggml-base.so.0
#7  0x000075b523138c51 in llama_context::graph_compute(ggml_cgraph*, bool) () from /usr/lib/libllama.so.0
#8  0x000075b523137f79 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /usr/lib/libllama.so.0
#9  0x000075b52313a662 in llama_context::decode(llama_batch_ext const&) () from /usr/lib/libllama.so.0
#10 0x000075b523140049 in llama_decode () from /usr/lib/libllama.so.0
#11 0x000075b523837648 in ?? () from /usr/lib/libllama-bench-impl.so
#12 0x000075b52383090c in llama_bench(int, char**) () from /usr/lib/libllama-bench-impl.so
#13 0x000075b503552781 in ?? () from /usr/lib/libc.so.6
#14 0x000075b5035528b9 in __libc_start_main () from /usr/lib/libc.so.6
#15 0x0000000000405405 in ?? ()
[Inferior 1 (process 260404) detached]
