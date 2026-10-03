import os
import argparse
import re

from dataclasses import dataclass, field
from typing import List

# Based on https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp


@dataclass
class GptParams:
    seed: int = -1
    n_threads: int = min(4, os.cpu_count() or 1)
    n_predict: int = 128
    n_parts: int = -1
    n_ctx: int = 512
    n_batch: int = 8
    n_keep: int = 0

    ignore_eos: bool = False
    logit_bias: dict[int, float] = field(default_factory=dict)
    top_k: int = 40
    top_p: float = 0.95
    tfs_z: float = 1.00
    typical_p: float = 1.00
    temp: float = 0.80
    repeat_penalty: float = 1.10
    repeat_last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    model: str = "./models/llama-7B/ggml-model.bin"
    prompt: str = ""
    path_session: str = ""
    input_prefix: str = " "
    input_suffix: str = ""
    antiprompt: List[str] = field(default_factory=list)

    lora_adapter: str = ""
    lora_base: str = ""

    memory_f16: bool = True
    random_prompt: bool = False
    use_color: bool = False
    interactive: bool = False

    embedding: bool = False
    interactive_start: bool = False

    instruct: bool = False
    penalize_nl: bool = True
    perplexity: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    mem_test: bool = False
    verbose_prompt: bool = False

    file: str = None

    # If chat ended prematurely, append this to the conversation to fix it.
    # Set to "\nUser:" etc.
    # This is an alternative to input_prefix which always adds it, so it potentially duplicates "User:""
    fix_prefix: str = ""
    input_echo: bool = True,

    # Default instructions for Alpaca
    # switch to "Human" and "Assistant" for Vicuna.
    # TODO: TBD how they are gonna handle this upstream
    instruct_inp_prefix: str="\n\n### Instruction:\n\n"
    instruct_inp_suffix: str="\n\n### Response:\n\n"


def gpt_params_parse(argv = None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", type=int, default=-1, help="RNG seed (use random seed for <= 0)",dest="seed")
    parser.add_argument("-t", "--threads", type=int, default=min(4, os.cpu_count() or 1), help="number of threads to use during computation",dest="n_threads")
    parser.add_argument("-n", "--n_predict", type=int, default=128, help="number of tokens to predict (-1 = infinity)",dest="n_predict")
    parser.add_argument("--n_parts", type=int, default=-1, help="number of model parts", dest="n_parts")
    parser.add_argument("-c", "--ctx_size", type=int, default=512, help="size of the prompt context",dest="n_ctx")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size for prompt processing",dest="n_batch")
    parser.add_argument("--keep", type=int, default=0, help="number of tokens to keep from the initial prompt",dest="n_keep")

    parser.add_argument(
        "-l",
        "--logit-bias",
        type=str,
        action='append',
        help="--logit-bias TOKEN_ID(+/-)BIAS",
        dest="logit_bias_str"
    )
    parser.add_argument("--ignore-eos", action="store_true", help="ignore end of stream token and continue generating", dest="ignore_eos")
    parser.add_argument("--top_k", type=int, default=40, help="top-k sampling",dest="top_k")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p samplin",dest="top_p")
    parser.add_argument("--tfs", type=float, default=1.0, help="tail free sampling, parameter z (1.0 = disabled)",dest="tfs_z")
    parser.add_argument("--temp", type=float, default=0.80, help="temperature",dest="temp")
    parser.add_argument("--repeat_penalty", type=float, default=1.10, help="penalize repeat sequence of tokens",dest="repeat_penalty")
    parser.add_argument("--repeat_last_n", type=int, default=64, help="last n tokens to consider for penalize ",dest="repeat_last_n")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="repeat alpha frequency penalty (0.0 = disabled)",dest="tfs_z")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="repeat alpha presence penalty (0.0 = disabled)",dest="presence_penalty")
    parser.add_argument("--mirostat", type=float, default=1.0, help="use Mirostat sampling.",dest="mirostat")
    parser.add_argument("--mirostat_ent", type=float, default=5.0, help="Mirostat target entropy, parameter tau represents the average surprise value",dest="mirostat_tau")
    parser.add_argument("--mirostat_lr", type=float, default=0.1, help="Mirostat learning rate, parameter eta",dest="mirostat_eta")

    parser.add_argument("-m", "--model", type=str, default="./models/llama-7B/ggml-model.bin", help="model path",dest="model")
    parser.add_argument("-p", "--prompt", type=str, default="", help="initial prompt",dest="prompt")
    parser.add_argument("-f", "--file", type=str, default=None, help="file containing initial prompt to load",dest="file")
    parser.add_argument("--session", type=str, default="", help="file to cache model state in (may be large!)",dest="path_session")
    parser.add_argument("--in-prefix", type=str, default="", help="string to prefix user inputs with", dest="input_prefix")
    parser.add_argument("--in-suffix", type=str, default="", help="append to input", dest="input_suffix")
    parser.add_argument(
        "-r",
        "--reverse-prompt",
        type=str,
        action='append',
        help="poll user input upon seeing PROMPT (can be\nspecified more than once for multiple prompts).",
        dest="antiprompt"
    )

    parser.add_argument("--lora", type=str, default="", help="apply LoRA adapter (implies --no-mmap)", dest="lora_adapter")
    parser.add_argument("--lora-base", type=str, default="", help="optional model to use as a base for the layers modified by the LoRA adapter", dest="lora_base")

    parser.add_argument("--memory_f32", action="store_false", help="use f32 instead of f16 for memory key+value",dest="memory_f16")
    parser.add_argument("--random-prompt", action="store_true", help="start with a randomized prompt.", dest="random_prompt")
    parser.add_argument(
        "--color",
        action="store_true",
        help="colorise output to distinguish prompt and user input from generations",
        dest="use_color"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="run in interactive mode", dest="interactive"
    )

    parser.add_argument("--embedding", action="store_true", help="", dest="embedding")
    parser.add_argument(
        "--interactive-first",
        action="store_true",
        help="run in interactive mode and wait for input right away",
        dest="interactive_start"
    )

    parser.add_argument(
        "-ins",
        "--instruct",
        action="store_true",
        help="run in instruction mode (use with Alpaca or Vicuna models)",
        dest="instruct"
    )
    parser.add_argument("--no-penalize-nl", action="store_false", help="do not penalize newline token", dest="penalize_nl")
    parser.add_argument("--perplexity", action="store_true", help="compute perplexity over the prompt", dest="perplexity")
    parser.add_argument("--no-mmap", action="store_false",help="do not memory-map model (slower load but may reduce pageouts if not using mlock)",dest="use_mmap")
    parser.add_argument("--mlock", action="store_true",help="force system to keep model in RAM rather than swapping or compressing",dest="use_mlock")
    parser.add_argument("--mtest", action="store_true",help="compute maximum memory usage",dest="mem_test")
    parser.add_argument("--verbose-prompt", action="store_true",help="print prompt before generation",dest="verbose_prompt")

    #Custom args
    parser.add_argument("--fix-prefix", type=str, default="", help="append to input when generated n_predict tokens", dest="fix_prefix")
    parser.add_argument("--input-noecho", action="store_false", help="dont output the input", dest="input_echo")

    parser.add_argument(
        "--interactive-start",
        action="store_true",
        help="run in interactive mode",
        dest="interactive"
    )

    args = parser.parse_args(argv)

    logit_bias_str = args.logit_bias_str
    delattr(args, "logit_bias_str")
    params = GptParams(**vars(args))

    if (params.lora_adapter):
        params.use_mmap = False

    if (logit_bias_str != None):
        for i in logit_bias_str:
            if (m := re.match(r"(\d+)([-+]\d+)", i)):
                params.logit_bias[int(m.group(1))] = float(m.group(2))

    return params

def gpt_random_prompt(rng):
    return [
        "So",
        "Once upon a time",
        "When",
        "The",
        "After",
        "If",
        "import",
        "He",
        "She",
        "They",
    ][rng % 10]

if __name__ == "__main__":
    print(gpt_params_parse())
