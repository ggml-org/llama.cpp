"""
This is an example implementation of main.cpp from llama.cpp
Quirks:
 * Its not exactly alike since this port is designed around programmatic I/O
 * Input is always echoed if on, so it should be turned off when using "input()"
 * The first antiprompt should be the userprompt like "\nUser:",
   because its added when n_predict is reached (aka generation ended prematurely)
 * n_predict can be set to -1 for unlimited length responses (or just a really high value)
 * Instruction mode adds its own antiprompt.
   You should also still be feeding the model with a "primer" prompt that
   shows it the expected format.
"""
import ctypes
import sys
from time import time
from os import cpu_count, path

import llama_cpp
from common import GptParams, gpt_params_parse, gpt_random_prompt

ANSI_COLOR_RESET = "\x1b[0m"
ANSI_COLOR_YELLOW = "\x1b[33m"
ANSI_BOLD = "\x1b[1m"
ANSI_COLOR_GREEN = "\x1b[32m"

CONSOLE_COLOR_DEFAULT = ANSI_COLOR_RESET
CONSOLE_COLOR_PROMPT = ANSI_COLOR_YELLOW
CONSOLE_COLOR_USER_INPUT = ANSI_BOLD + ANSI_COLOR_GREEN

# Iterative search
# Actively searches and prevents a pattern from being returned
class IterSearch:
    def __init__(self, pattern):
        self.pattern = list(pattern)
        self.buffer = []

    def __call__(self, char):
        self.buffer += [char]

        if (self.pattern[:len(self.buffer)] == self.buffer):
            if (len(self.buffer) >= len(self.pattern)):
                self.buffer.clear()
            return []

        _tmp = self.buffer[:]
        self.buffer.clear()
        return _tmp

# A LLaMA interactive session
class LLaMAInteract:
    def __init__(self, params: GptParams) -> None:
        # input args
        self.params = params

        if (self.params.perplexity):
            raise NotImplementedError("""************
please use the 'perplexity' tool for perplexity calculations
************""")

        if (self.params.embedding):
            raise NotImplementedError("""************
please use the 'embedding' tool for embedding calculations
************""")

        if (self.params.n_ctx > 2048):
            print(f"""warning: model does not support \
context sizes greater than 2048 tokens ({self.params.n_ctx} \
specified) expect poor results""", file=sys.stderr)

        if (self.params.seed <= 0):
            self.params.seed = int(time())

        print(f"seed = {self.params.seed}", file=sys.stderr)

        if (self.params.random_prompt):
            self.params.prompt = gpt_random_prompt(self.params.seed)

        # runtime args
        self.input_consumed = 0
        self.n_past = 0
        self.n_session_consumed = 0
        self.first_antiprompt = []
        self.remaining_tokens = self.params.n_predict
        self.output_echo = self.params.input_echo

        # model load
        self.lparams = llama_cpp.llama_context_default_params()
        self.lparams.n_ctx = self.params.n_ctx
        self.lparams.n_parts = self.params.n_parts
        self.lparams.seed = self.params.seed
        self.lparams.memory_f16 = self.params.memory_f16
        self.lparams.use_mlock = self.params.use_mlock
        self.lparams.use_mmap = self.params.use_mmap

        self.ctx = llama_cpp.llama_init_from_file(self.params.model.encode("utf8"), self.lparams)
        if (not self.ctx):
            raise RuntimeError(f"error: failed to load model '{self.params.model}'")

        if (self.params.ignore_eos):
            self.params.logit_bias[llama_cpp.llama_token_eos()] = -float("inf")

        if (len(self.params.lora_adapter) > 0):
            if (llama_cpp.llama_apply_lora_from_file(
                self.ctx,
                self.params.lora_adapter.encode("utf8"),
                self.params.lora_base.encode("utf8") if len(self.params.lora_base) > 0 else None,
                self.params.n_threads
            ) != 0):
                print("error: failed to apply lora adapter")
                return

        print(file=sys.stderr)
        print(f"system_info: n_threads = {self.params.n_threads} / {cpu_count()} \
| {llama_cpp.llama_print_system_info().decode('utf8')}", file=sys.stderr)

        # determine the required inference memory per token:
        if (self.params.mem_test):
            tmp = [0, 1, 2, 3]
            llama_cpp.llama_eval(self.ctx, (llama_cpp.c_int * len(tmp))(*tmp), len(tmp), 0, self.n_threads)
            llama_cpp.llama_print_timings(self.ctx)
            self.exit()
            return

        # create internal context
        self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)

        # Add a space in front of the first character to match OG llama tokenizer behavior
        self.params.prompt = " " + self.params.prompt

        # Load prompt file
        if (self.params.file):
            with open(self.params.file) as f:
                self.params.prompt = f.read()

        self.session_tokens: list[llama_cpp.llama_token] = []
        if (len(self.params.path_session) > 0):
            print(f"attempting to load saved session from '{self.params.path_session}'", file=sys.stderr)

            if (path.exists(self.params.path_session)):
                _session_tokens = (llama_cpp.llama_token * (self.params.n_ctx))()
                _n_token_count_out = llama_cpp.c_size_t()
                if (llama_cpp.llama_load_session_file(
                    self.ctx,
                    self.params.path_session.encode("utf8"),
                    _session_tokens,
                    self.params.n_ctx,
                    ctypes.byref(_n_token_count_out)
                ) != 1):
                    print(f"error: failed to load session file '{self.params.path_session}'", file=sys.stderr)
                    return
                _n_token_count_out = _n_token_count_out.value
                self.session_tokens = _session_tokens[:_n_token_count_out]
                print(f"loaded a session with prompt size of {_n_token_count_out} tokens", file=sys.stderr)
            else:
                print(f"session file does not exist, will create", file=sys.stderr)

        # tokenize the prompt
        self.embd = []
        self.embd_inp = self._tokenize(self.params.prompt)

        if (len(self.embd_inp) > self.n_ctx - 4):
            raise RuntimeError(f"error: prompt is too long ({len(self.embd_inp)} tokens, max {self.params.n_ctx - 4})")

        # debug message about similarity of saved session, if applicable
        self.n_matching_session_tokens = 0
        if len(self.session_tokens) > 0:
            for id in self.session_tokens:
                if self.n_matching_session_tokens >= len(self.embd_inp) or id != self.embd_inp[self.n_matching_session_tokens]:
                    break
                self.n_matching_session_tokens += 1

            if self.n_matching_session_tokens >= len(self.embd_inp):
                print(f"session file has exact match for prompt!")
            elif self.n_matching_session_tokens < (len(self.embd_inp) / 2):
                print(f"warning: session file has low similarity to prompt ({self.n_matching_session_tokens} / {len(self.embd_inp)} tokens); will mostly be reevaluated")
            else:
                print(f"session file matches {self.n_matching_session_tokens} / {len(self.embd_inp)} tokens of prompt")

        self.need_to_save_session = len(self.params.path_session) > 0 and self.n_matching_session_tokens < (len(self.embd_inp) * 3 / 4)

        # number of tokens to keep when resetting context
        if (self.params.n_keep < 0 or self.params.n_keep > len(self.embd_inp) or self.params.instruct):
            self.params.n_keep = len(self.embd_inp)

        self.inp_prefix = self._tokenize(self.params.instruct_inp_prefix)
        self.inp_suffix = self._tokenize(self.params.instruct_inp_suffix, False)

        # in instruct mode, we inject a prefix and a suffix to each input by the user
        self.antiecho = None
        if (self.params.instruct):
            self.params.interactive_start = True
            _ptn = self._tokenize(self.params.instruct_inp_prefix.strip(), False)
            self.first_antiprompt.append(_ptn)
            self.antiecho = IterSearch(_ptn)

        # enable interactive mode if reverse prompt or interactive start is specified
        if (len(self.params.antiprompt) != 0 or self.params.interactive_start):
            self.params.interactive = True

        # determine newline token
        self.llama_token_newline = self._tokenize("\n", False)
        self.llama_token_eot = self._tokenize(" [end of text]\n", False)

        if (self.params.verbose_prompt):
            print(f"""
prompt: '{self.params.prompt}'
number of tokens in prompt = {len(self.embd_inp)}""", file=sys.stderr)

            for i in range(len(self.embd_inp)):
                print(f"{self.embd_inp[i]} -> '{llama_cpp.llama_token_to_str(self.ctx, self.embd_inp[i])}'", file=sys.stderr)

            if (self.params.n_keep > 0):
                print("static prompt based on n_keep: '")
                for i in range(self.params.n_keep):
                    print(llama_cpp.llama_token_to_str(self.ctx, self.embd_inp[i]), file=sys.stderr)
                print("'", file=sys.stderr)
            print(file=sys.stderr)

        if (self.params.interactive):
            print("interactive mode on.", file=sys.stderr)

            if (len(self.params.antiprompt) > 0):
                for antiprompt in self.params.antiprompt:
                    print(f"Reverse prompt: '{antiprompt}'", file=sys.stderr)

            if len(self.params.input_prefix) > 0:
                print(f"Input prefix: '{self.params.input_prefix}'", file=sys.stderr)

        print(f"""sampling: repeat_last_n = {self.params.repeat_last_n},\
repeat_penalty = {self.params.repeat_penalty},\
presence_penalty = {self.params.presence_penalty},\
frequency_penalty = {self.params.frequency_penalty},\
top_k = {self.params.top_k},\
tfs_z = {self.params.tfs_z},\
top_p = {self.params.top_p},\
typical_p = {self.params.typical_p},\
temp = {self.params.temp},\
mirostat = {self.params.mirostat},\
mirostat_lr = {self.params.mirostat_eta},\
mirostat_ent = {self.params.mirostat_tau},\

generate: n_ctx = {self.n_ctx},\
n_batch = {self.params.n_batch},\
n_predict = {self.params.n_predict},\
n_keep = {self.params.n_keep}

""", file=sys.stderr)

        # determine antiprompt tokens
        for i in self.params.antiprompt:
            self.first_antiprompt.append(self._tokenize(i, False))

        self.last_n_tokens = [0]*self.n_ctx #TODO: deque doesnt support slices

        if (params.interactive):
            print("""== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to LLaMa.
 - If you want to submit another line, end your input in '\\'.

""", file=sys.stderr)
        self.set_color(CONSOLE_COLOR_PROMPT)

    # tokenize a prompt
    def _tokenize(self, prompt, bos=True):
        _arr = (llama_cpp.llama_token * (len(prompt) + 1))()
        _n = llama_cpp.llama_tokenize(self.ctx, prompt.encode("utf8", errors="ignore"), _arr, len(_arr), bos)
        return _arr[:_n]

    def set_color(self, c):
        if (self.params.use_color):
            print(c, end="")

    def use_antiprompt(self):
        return len(self.first_antiprompt) > 0

    # generate tokens
    def generate(self):
        while self.remaining_tokens > 0 or self.params.interactive or self.params.n_predict == -1:
            # predict
            if len(self.embd) > 0:
                # infinite text generation via context swapping
                # if we run out of context:
                # - take the n_keep first tokens from the original prompt (via n_past)
                # - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
                if (self.n_past + len(self.embd) > self.n_ctx):
                    n_left = self.n_past - self.params.n_keep
                    self.n_past = self.params.n_keep

                    # insert n_left/2 tokens at the start of embd from last_n_tokens
                    _insert = self.last_n_tokens[
                        self.n_ctx - int(n_left/2) - len(self.embd):-len(self.embd)
                    ]
                    self.embd = _insert + self.embd
                    self.params.path_session = ""

                # try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
                # REVIEW
                if self.n_session_consumed < len(self.session_tokens):
                    for i in range(len(self.embd)):
                        if self.embd[i] != self.session_tokens[self.n_session_consumed]:
                            self.session_tokens = self.session_tokens[:self.n_session_consumed]
                            break

                        self.n_past += 1
                        self.n_session_consumed += 1

                        if self.n_session_consumed >= len(self.session_tokens):
                            i += 1
                            break

                    if i > 0:
                        self.embd = self.embd[i:]

                # evaluate tokens in batches
                # embd is typically prepared beforehand to fit within a batch, but not always
                #TODO BUG: The batching code causes nonsensical generation
                """for i in range(0, len(self.embd), self.params.n_batch):
                    n_eval = self.params.n_batch
                    _arr = (llama_cpp.llama_token * n_eval)(*self.embd[i:i + n_eval])
                    if llama_cpp.llama_eval(self.ctx, _arr, n_eval, self.n_past, self.params.n_threads) != 0:
                        print(f"failed to eval")
                        return

                    self.n_past += n_eval"""

                if (llama_cpp.llama_eval(
                    self.ctx, (llama_cpp.llama_token * len(self.embd))(*self.embd), len(self.embd), self.n_past, self.params.n_threads
                ) != 0):
                    raise Exception("Failed to llama_eval!")

                if len(self.embd) > 0 and len(self.params.path_session) > 0:
                    self.session_tokens.extend(self.embd)
                    self.n_session_consumed = len(self.session_tokens)

            self.n_past += len(self.embd)
            self.embd = []
            if len(self.embd_inp) <= self.input_consumed: #&& !is_interacting
                # out of user input, sample next token
                top_k = llama_cpp.llama_n_vocab(self.ctx) if self.params.top_k <= 0 else self.params.top_k
                repeat_last_n = self.n_ctx if self.params.repeat_last_n < 0 else self.params.repeat_last_n

                # optionally save the session on first sample (for faster prompt loading next time)
                if len(self.params.path_session) > 0 and self.need_to_save_session:
                    self.need_to_save_session = False
                    llama_cpp.llama_save_session_file(
                        self.ctx,
                        self.params.path_session.encode("utf8"),
                        (llama_cpp.llama_token * len(self.session_tokens))(*self.session_tokens),
                        len(self.session_tokens)
                    )

                id = 0

                logits = llama_cpp.llama_get_logits(self.ctx)
                n_vocab = llama_cpp.llama_n_vocab(self.ctx)

                # Apply params.logit_bias map
                for key, value in self.params.logit_bias.items():
                    logits[key] += value

                _arr = (llama_cpp.llama_token_data * n_vocab)(*[
                    llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
                    for token_id in range(n_vocab)
                ])
                candidates_p = llama_cpp.ctypes.pointer(llama_cpp.llama_token_data_array(_arr, len(_arr), False))

                # Apply penalties
                nl_logit = logits[llama_cpp.llama_token_nl()]
                last_n_repeat = min(len(self.last_n_tokens), repeat_last_n, self.n_ctx)

                _arr = (llama_cpp.llama_token * last_n_repeat)(*self.last_n_tokens[len(self.last_n_tokens) - last_n_repeat:])
                llama_cpp.llama_sample_repetition_penalty(self.ctx, candidates_p,
                    _arr,
                    last_n_repeat, llama_cpp.c_float(self.params.repeat_penalty))
                llama_cpp.llama_sample_frequency_and_presence_penalties(self.ctx, candidates_p,
                    _arr,
                    last_n_repeat, llama_cpp.c_float(self.params.frequency_penalty), llama_cpp.c_float(self.params.presence_penalty))

                if not self.params.penalize_nl:
                    logits[llama_cpp.llama_token_nl()] = nl_logit

                if self.params.temp <= 0:
                    # Greedy sampling
                    id = llama_cpp.llama_sample_token_greedy(self.ctx, candidates_p)
                else:
                    if self.params.mirostat == 1:
                        mirostat_mu = 2.0 * self.params.mirostat_tau
                        mirostat_m = 100
                        llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
                        id = llama_cpp.llama_sample_token_mirostat(self.ctx, candidates_p, llama_cpp.c_float(self.params.mirostat_tau), llama_cpp.c_float(self.params.mirostat_eta), llama_cpp.c_int(mirostat_m), llama_cpp.c_float(mirostat_mu))
                    elif self.params.mirostat == 2:
                        mirostat_mu = 2.0 * self.params.mirostat_tau
                        llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
                        id = llama_cpp.llama_sample_token_mirostat_v2(self.ctx, candidates_p, llama_cpp.c_float(self.params.mirostat_tau), llama_cpp.c_float(self.params.mirostat_eta), llama_cpp.c_float(mirostat_mu))
                    else:
                        # Temperature sampling
                        llama_cpp.llama_sample_top_k(self.ctx, candidates_p, top_k, min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_tail_free(self.ctx, candidates_p, llama_cpp.c_float(self.params.tfs_z),min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_typical(self.ctx, candidates_p, llama_cpp.c_float(self.params.typical_p),min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_top_p(self.ctx, candidates_p, llama_cpp.c_float(self.params.top_p),min_keep=llama_cpp.c_size_t(1))
                        llama_cpp.llama_sample_temperature(self.ctx, candidates_p, llama_cpp.c_float(self.params.temp))
                        id = llama_cpp.llama_sample_token(self.ctx, candidates_p)
                # print("`{}`".format(candidates_p.size))

                self.last_n_tokens.pop(0)
                self.last_n_tokens.append(id)

                # replace end of text token with newline token when in interactive mode
                if (id == llama_cpp.llama_token_eos() and self.params.interactive and not self.params.instruct):
                    id = self.llama_token_newline[0]
                    self.embd.append(id)
                    if (self.use_antiprompt()):
                        # tokenize and inject first reverse prompt
                        self.embd_inp += self.first_antiprompt[0]
                        for id in self.first_antiprompt[0]:
                            self.embd.append(id)
                else:
                    # add it to the context
                    self.embd.append(id)

                # echo this to console
                self.output_echo = True

                # decrement remaining sampling budget
                self.remaining_tokens -= 1
            else:
                # output to console if input echo is on
                self.output_echo = self.params.input_echo

                # some user input remains from prompt or interaction, forward it to processing
                while len(self.embd_inp) > self.input_consumed:
                    self.embd.append(self.embd_inp[self.input_consumed])
                    self.last_n_tokens.pop(0)
                    self.last_n_tokens.append(self.embd_inp[self.input_consumed])
                    self.input_consumed += 1
                    if len(self.embd) >= self.params.n_batch:
                        break

            # display tokens
            if self.output_echo:
                for id in self.embd:
                    if self.antiecho != None:
                        for r in self.antiecho(id):
                            yield r
                    else:
                        yield id

            # reset color to default if we there is no pending user input
            if (self.params.input_echo and len(self.embd_inp) == self.input_consumed):
                self.set_color(CONSOLE_COLOR_DEFAULT)

            if (self.params.interactive and len(self.embd_inp) <= self.input_consumed):
                # if antiprompt is present, stop
                if (self.use_antiprompt()):
                    if True in [
                        i == self.last_n_tokens[-len(i):]
                        for i in self.first_antiprompt
                    ]:
                        break

                # if we are using instruction mode, and we have processed the initial prompt
                if (self.params.interactive_start):
                    break

            # end of text token
            if len(self.embd) > 0 and self.embd[-1] == llama_cpp.llama_token_eos():
                if (not self.params.instruct):
                    for i in self.llama_token_eot:
                        yield i
                    break

            # respect n_predict even if antiprompt is present
            if (self.params.interactive and self.remaining_tokens <= 0 and self.params.n_predict != -1):
                # If we arent in instruction mode, fix the current generation by appending the antiprompt.
                # Makes it so if chat ends prematurely you dont append the AI's text etc.
                if not self.params.instruct:
                    self.embd_inp += self.first_antiprompt[0]
                self.n_remain = self.params.n_predict
                break

        self.params.interactive_start = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.exit()

    def exit(self):
        llama_cpp.llama_free(self.ctx)
        self.set_color(CONSOLE_COLOR_DEFAULT)

    # return past text
    def past(self):
        for id in self.last_n_tokens[-self.n_past:]:
            yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf-8", errors="ignore")

    # write input
    def input(self, prompt: str):
        if (self.params.instruct and self.last_n_tokens[-len(self.inp_prefix):] != self.inp_prefix):
            self.embd_inp += self.inp_prefix
        self.embd_inp += self._tokenize(prompt)
        if (self.params.instruct):
            self.embd_inp += self.inp_suffix

    # write output
    def output(self):
        self.remaining_tokens = self.params.n_predict
        for id in self.generate():
            yield llama_cpp.llama_token_to_str(self.ctx, id).decode("utf-8")

    # read user input
    def read_input(self):
        out = ""
        while (t := input()).endswith("\\"):
            out += t[:-1] + "\n"
        return out + t + "\n"

    # interactive mode
    def interact(self):
        for i in self.output():
            print(i,end="",flush=True)
        self.params.input_echo = False

        while self.params.interactive:
            self.set_color(CONSOLE_COLOR_USER_INPUT)
            if (self.params.instruct):
                print('\n> ', end="")
                self.input(self.read_input())
            else:
                print(self.params.input_prefix, end="")
                self.input(f"{self.params.input_prefix}{self.read_input()}{self.params.input_suffix}")
                print(self.params.input_suffix,end="")
            self.set_color(CONSOLE_COLOR_DEFAULT)

            try:
                for i in self.output():
                    print(i,end="",flush=True)
            except KeyboardInterrupt:
                self.set_color(CONSOLE_COLOR_DEFAULT)
                if not self.params.instruct:
                    print(self.params.fix_prefix,end="")
                    self.input(self.params.fix_prefix)

if __name__ == "__main__":
    from datetime import datetime

    USER_NAME="User"
    AI_NAME="ChatLLaMa"

    time_now = datetime.now()
    prompt = f"""Text transcript of a never ending dialog, where {USER_NAME} interacts with an AI assistant named {AI_NAME}.
{AI_NAME} is helpful, kind, honest, friendly, good at writing and never fails to answer {USER_NAME}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {USER_NAME} and {AI_NAME} say aloud to each other.
The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.
The transcript only includes text, it does not include markup like HTML and Markdown.

{USER_NAME}: Hello, {AI_NAME}!
{AI_NAME}: Hello {USER_NAME}! How may I help you today?
{USER_NAME}: What time is it?
{AI_NAME}: It is {time_now.strftime("%H:%M")}.
{USER_NAME}: What year is it?
{AI_NAME}: We are in {time_now.strftime("%Y")}.
{USER_NAME}: What is a cat?
{AI_NAME}: A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{USER_NAME}: Name a color.
{AI_NAME}: Blue
{USER_NAME}:"""
    params = gpt_params_parse()

    with LLaMAInteract(params) as m:
        m.interact()
