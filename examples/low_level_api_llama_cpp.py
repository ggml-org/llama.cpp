import llama_cpp

import multiprocessing

import llama_cpp

N_THREADS = multiprocessing.cpu_count()

prompt = b"\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"

lparams = llama_cpp.llama_context_default_params()
ctx = llama_cpp.llama_init_from_file(b"../models/7B/ggml-model.bin", lparams)

# determine the required inference memory per token:
tmp = [0, 1, 2, 3]
llama_cpp.llama_eval(ctx, (llama_cpp.c_int * len(tmp))(*tmp), len(tmp), 0, N_THREADS)

n_past = 0

prompt = b" " + prompt

embd_inp = (llama_cpp.llama_token * (len(prompt) + 1))()
n_of_tok = llama_cpp.llama_tokenize(ctx, prompt, embd_inp, len(embd_inp), True)
embd_inp = embd_inp[:n_of_tok]

n_ctx = llama_cpp.llama_n_ctx(ctx)

n_predict = 20
n_predict = min(n_predict, n_ctx - len(embd_inp))

input_consumed = 0
input_noecho = False

remaining_tokens = n_predict

embd = []
last_n_size = 64
last_n_tokens_data = [0] * last_n_size
n_batch = 24
last_n_repeat = 64
repeat_penalty = 1
frequency_penalty = 0.0
presence_penalty = 0.0

while remaining_tokens > 0:
    if len(embd) > 0:
        llama_cpp.llama_eval(
            ctx, (llama_cpp.c_int * len(embd))(*embd), len(embd), n_past, N_THREADS
        )

    n_past += len(embd)
    embd = []
    if len(embd_inp) <= input_consumed:
        logits = llama_cpp.llama_get_logits(ctx)
        n_vocab = llama_cpp.llama_n_vocab(ctx)

        _arr = (llama_cpp.llama_token_data * n_vocab)(*[
            llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
            for token_id in range(n_vocab)
        ])
        candidates_p = llama_cpp.ctypes.pointer(llama_cpp.llama_token_data_array(_arr, len(_arr), False))

        _arr = (llama_cpp.c_int * len(last_n_tokens_data))(*last_n_tokens_data)
        llama_cpp.llama_sample_repetition_penalty(ctx, candidates_p,
            _arr,
            last_n_repeat, repeat_penalty)
        llama_cpp.llama_sample_frequency_and_presence_penalties(ctx, candidates_p,
            _arr,
            last_n_repeat, frequency_penalty, presence_penalty)

        llama_cpp.llama_sample_top_k(ctx, candidates_p, 40, min_keep=llama_cpp.c_size_t(1))
        llama_cpp.llama_sample_top_p(ctx, candidates_p, 0.8, min_keep=llama_cpp.c_size_t(1))
        llama_cpp.llama_sample_temperature(ctx, candidates_p, 0.2)
        id = llama_cpp.llama_sample_token(ctx, candidates_p)

        last_n_tokens_data = last_n_tokens_data[1:] + [id]
        embd.append(id)
        input_noecho = False
        remaining_tokens -= 1
    else:
        while len(embd_inp) > input_consumed:
            embd.append(embd_inp[input_consumed])
            last_n_tokens_data = last_n_tokens_data[1:] + [embd_inp[input_consumed]]
            input_consumed += 1
            if len(embd) >= n_batch:
                break
    if not input_noecho:
        for id in embd:
            print(
                llama_cpp.llama_token_to_str(ctx, id).decode("utf-8", errors="ignore"),
                end="",
                flush=True,
            )

    if len(embd) > 0 and embd[-1] == llama_cpp.llama_token_eos():
        break

print()

llama_cpp.llama_print_timings(ctx)

llama_cpp.llama_free(ctx)
