# llama.cpp/example/batched

The example demonstrates batched generation from a given prompt

```bash
./llama-batched -m ./models/llama-7b-v2/ggml-model-f16.gguf -p "Hello my name is" -np 4 --kv-unified

...

main: n_len = 32, n_ctx = 2048, n_parallel = 4, n_kv_req = 113

 Hello my name is

main: generating 4 sequences ...

main: stream 0 finished
main: stream 1 finished
main: stream 2 finished
main: stream 3 finished

sequence 0:

Hello my name is Shirley. I am a 25-year-old female who has been working for over 5 years as a b

sequence 1:

Hello my name is Renee and I'm a 32 year old female from the United States. I'm looking for a man between

sequence 2:

Hello my name is Diana. I am looking for a housekeeping job. I have experience with children and have my own transportation. I am

sequence 3:

Hello my name is Cody. I am a 3 year old neutered male. I am a very friendly cat. I am very playful and

main: decoded 108 tokens in 3.57 s, speed: 30.26 t/s

llama_print_timings:        load time =   587.00 ms
llama_print_timings:      sample time =     2.56 ms /   112 runs   (    0.02 ms per token, 43664.72 tokens per second)
llama_print_timings: prompt eval time =  4089.11 ms /   118 tokens (   34.65 ms per token,    28.86 tokens per second)
llama_print_timings:        eval time =     0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =  4156.04 ms
```

### Using backend samplers
It is possible to run this example using backend samplers so that sampling is
performed on the backend device, like a GPU.
```bash
./llama-batched \
    -m models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf -p "Hello my name is" \
    -np 4 -kvu \
    --backend_sampling --top-k 80 --backend_dist
```
The `--verbose` flag can be added to see more detailed output and also show
that the backend samplers are being used.
