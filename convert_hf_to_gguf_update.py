#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import pathlib
import re

import requests
import json
import shutil
import argparse

from hashlib import sha256
from enum import IntEnum, auto
from transformers import AutoTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("convert_hf_to_gguf_update")
sess = requests.Session()

convert_py_pth = pathlib.Path("convert_hf_to_gguf.py")
convert_py = convert_py_pth.read_text(encoding="utf-8")
hf_token_pth = pathlib.Path.home() / ".cache" / "huggingface" / "token"
hf_token = hf_token_pth.read_text(encoding="utf-8").strip() if hf_token_pth.exists() else None


class TOKENIZER_TYPE(IntEnum):
    SPM = auto()
    BPE = auto()
    WPM = auto()
    UGM = auto()


DOC_STRING = """
This script downloads the tokenizer models of the specified models from Huggingface and
generates the get_vocab_base_pre() function for convert_hf_to_gguf.py

/!\\ It is intended to be used by contributors and is not meant to be run by end users

This is necessary in order to analyze the type of pre-tokenizer used by the model and
provide the necessary information to llama.cpp via the GGUF header in order to implement
the same pre-tokenizer.

ref: https://github.com/ggml-org/llama.cpp/pull/6920

Instructions:

- Add a new model to the "models" list
- Run the script with your huggingface token
    By default, token will be read from ~/.cache/huggingface/token
- The convert_hf_to_gguf.py script will have had its get_vocab_base_pre() function updated
- Update llama.cpp with the new pre-tokenizer if necessary
"""
# TODO: generate tokenizer tests for llama.cpp

parser = argparse.ArgumentParser(description=DOC_STRING, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--full", action="store_true",
    help="download full list of models - make sure you have access to all of them",
)
parser.add_argument(
    "hf_token",
    help="optional HF token",
    nargs="?",
)
args = parser.parse_args()
hf_token = args.hf_token if args.hf_token is not None else hf_token

if hf_token is None:
    logger.warning("HF token not found. You can provide it as an argument or set it in ~/.cache/huggingface/token")

# TODO: this string has to exercise as much pre-tokenizer functionality as possible
#       will be updated with time - contributions welcome
CHK_TXT = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````\"\"\"\"......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

# TODO: add models here, base models preferred
models = [
    {"name": "llama-spm",        "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/meta-llama/Llama-2-7b-hf", },
    {"name": "llama-bpe",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/meta-llama/Meta-Llama-3-8B", },
    {"name": "phi-3",            "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct", },
    {"name": "deepseek-llm",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-llm-7b-base", },
    {"name": "deepseek-coder",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base", },
    {"name": "falcon",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/falcon-7b", },
    {"name": "bert-bge",         "tokt": TOKENIZER_TYPE.WPM, "repo": "https://huggingface.co/BAAI/bge-small-en-v1.5", },
    {"name": "falcon3",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/Falcon3-7B-Base", },
    {"name": "bert-bge-large",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/BAAI/bge-large-zh-v1.5", },
    {"name": "mpt",              "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mosaicml/mpt-7b", },
    {"name": "starcoder",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/bigcode/starcoder2-3b", },
    {"name": "gpt-2",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/openai-community/gpt2", },
    {"name": "stablelm2",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b", },
    {"name": "refact",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/smallcloudai/Refact-1_6-base", },
    {"name": "command-r",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/CohereForAI/c4ai-command-r-v01", },
    {"name": "qwen2",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Qwen/Qwen1.5-7B", },
    {"name": "olmo",             "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/allenai/OLMo-1.7-7B-hf", },
    {"name": "dbrx",             "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/databricks/dbrx-base", },
    {"name": "jina-v1-en",       "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-reranker-v1-tiny-en", },
    {"name": "jina-v2-en",       "tokt": TOKENIZER_TYPE.WPM, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-en", }, # WPM!
    {"name": "jina-v2-es",       "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-es", },
    {"name": "jina-v2-de",       "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-de", },
    {"name": "smaug-bpe",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct", },
    {"name": "poro-chat",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LumiOpen/Poro-34B-chat", },
    {"name": "jina-v2-code",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-code", },
    {"name": "viking",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LumiOpen/Viking-7B", }, # Also used for Viking 13B and 33B
    {"name": "gemma",            "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/google/gemma-2b", },
    {"name": "gemma-2",          "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/google/gemma-2-9b", },
    {"name": "jais",             "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/core42/jais-13b", },
    {"name": "t5",               "tokt": TOKENIZER_TYPE.UGM, "repo": "https://huggingface.co/google-t5/t5-small", },
    {"name": "codeshell",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/WisdomShell/CodeShell-7B", },
    {"name": "tekken",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mistralai/Mistral-Nemo-Base-2407", },
    {"name": "smollm",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/HuggingFaceTB/SmolLM-135M", },
    {'name': "bloom",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/bigscience/bloom", },
    {'name': "gpt3-finnish",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/TurkuNLP/gpt3-finnish-small", },
    {"name": "exaone",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", },
    {"name": "phi-2",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/microsoft/phi-2", },
    {"name": "chameleon",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/facebook/chameleon-7b", },
    {"name": "roberta-bpe",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/sentence-transformers/stsb-roberta-base"},
    {"name": "gigachat",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct"},
    {"name": "megrez",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Infinigence/Megrez-3B-Instruct"},
    {"name": "deepseek-v3",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V3"},
    {"name": "deepseek-r1-qwen", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    {"name": "gpt-4o",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Xenova/gpt-4o", },
    {"name": "superbpe",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/UW/OLMo2-8B-SuperBPE-t180k", },
    {"name": "trillion",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/trillionlabs/Trillion-7B-preview", },
    {"name": "bailingmoe",       "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/inclusionAI/Ling-lite", },
    {"name": "llama4",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct", },
    {"name": "pixtral",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mistral-community/pixtral-12b", },
    {"name": "seed-coder",       "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base", },
    {"name": "a.x-4.0",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/skt/A.X-4.0", },
    {"name": "midm-2.0",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/K-intelligence/Midm-2.0-Base-Instruct", },
    {"name": "lfm2",             "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LiquidAI/LFM2-Tokenizer"},
    {"name": "exaone4",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B", },
]

# some models are known to be broken upstream, so we will skip them as exceptions
pre_computed_hashes = [
    # chatglm-bpe has 2 hashes, why?
    {"name": "chatglm-bpe", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/THUDM/glm-4-9b-chat", "chkhsh": "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b"},
    {"name": "chatglm-bpe", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/THUDM/glm-4-9b-chat", "chkhsh": "81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516"},
    {"name": "glm4", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/THUDM/glm-4-9b-hf", "chkhsh": "a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2"},
    {"name": "minerva-7b", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0", "chkhsh": "1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35"},
    {"name": "hunyuan", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tencent/Hunyuan-A13B-Instruct", "chkhsh": "7e57df22b1fe23a7b1e1c7f3dc4e3f96d43a4eb0836d0c6bdc3436d7b2f1c664"},
    # falcon-h1 series uses 4 different tokenizers across model sizes (0.5b - 34b), hence we need to define 4 different hashes
    {"name": "falcon-h1", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base", "chkhsh": "a6b57017d60e6edb4d88ecc2845188e0eb333a70357e45dcc9b53964a73bbae6"},
    {"name": "falcon-h1", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/Falcon-H1-1B-Base", "chkhsh": "60476e1243776c4fb1b993dbd7a5f15ac22f83c80afdf425fa5ae01c8d44ef86"},
    {"name": "falcon-h1", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/Falcon-H1-7B-Base", "chkhsh": "3eda48b4c4dc7de733d1a8b3e3b4a85243dbbf704da2ee9d42c6beced8897896"},
    {"name": "falcon-h1", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/Falcon-H1-34B-Base", "chkhsh": "48f8e02c0359c0bbdd82f26909171fac1c18a457bb47573ed1fe3bbb2c1cfd4b"},
    {"name": "kimi-k2",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/moonshotai/Kimi-K2-Base",   "chkhsh": "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890"},
]


def download_file_with_auth(url, token, save_path):
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = sess.get(url, headers=headers)
    response.raise_for_status()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as downloaded_file:
        downloaded_file.write(response.content)
    logger.info(f"File {save_path} downloaded successfully")


def download_model(model):
    name = model["name"]
    repo = model["repo"]
    tokt = model["tokt"]

    os.makedirs(f"models/tokenizers/{name}", exist_ok=True)

    files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    if name == "gpt-4o":
        # Xenova/gpt-4o is tokenizer-only, it does not contain config.json
        files = ["tokenizer.json", "tokenizer_config.json"]

    if tokt == TOKENIZER_TYPE.SPM:
        files.append("tokenizer.model")

    if tokt == TOKENIZER_TYPE.UGM:
        files.append("spiece.model")

    if os.path.isdir(repo):
        # If repo is a path on the file system, copy the directory
        for file in files:
            src_path = os.path.join(repo, file)
            dst_path = f"models/tokenizers/{name}/{file}"
            if os.path.isfile(dst_path):
                logger.info(f"{name}: File {dst_path} already exists - skipping")
                continue
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"{name}: Copied {src_path} to {dst_path}")
            else:
                logger.warning(f"{name}: Source file {src_path} does not exist")
    else:
        # If repo is a URL, download the files
        for file in files:
            save_path = f"models/tokenizers/{name}/{file}"
            if os.path.isfile(save_path):
                logger.info(f"{name}: File {save_path} already exists - skipping")
                continue
            download_file_with_auth(f"{repo}/resolve/main/{file}", hf_token, save_path)


# get list of existing models and chkhsh from the convert_hf_to_gguf.py file
# returns mapping res --> chkhsh
def get_existing_models(convert_py):
    pattern = r'if chkhsh == "([a-f0-9]{64})":\s*\n\s*.*\s*res = "([^"]+)"'
    matches = re.findall(pattern, convert_py)
    output = {}
    for chkhsh, res in matches:
        output[res] = chkhsh
    return output


existing_models = {}
all_models = models.copy()
if not args.full:
    # Filter out models that already exist in convert_hf_to_gguf.py
    existing_models = get_existing_models(convert_py)
    all_models = models.copy()
    models = [model for model in all_models if model["name"] not in existing_models]

logging.info(f"Downloading {len(models)} models...")
for model in models:
    try:
        download_model(model)
    except Exception as e:
        logger.error(f"Failed to download model {model['name']}. Error: {e}")


# generate the source code for the convert_hf_to_gguf.py:get_vocab_base_pre() function:

src_ifs = ""
for model in [*pre_computed_hashes, *all_models]:
    name = model["name"]
    tokt = model["tokt"]
    chkhsh = model.get("chkhsh")

    if tokt == TOKENIZER_TYPE.SPM or tokt == TOKENIZER_TYPE.UGM:
        continue

    # create the tokenizer
    if chkhsh is not None:
        # if the model has a pre-computed hash, use it
        logger.info(f"Using pre-computed hash for model {name}: {chkhsh}")
    elif name in existing_models:
        # if the model already exists in convert_hf_to_gguf.py, skip compute hash
        chkhsh = existing_models[name]
    else:
        # otherwise, compute the hash of the tokenizer

        # Fail if the tokenizer folder with config does not exist or there are other download issues previously
        if not os.path.isfile(f"models/tokenizers/{name}/tokenizer_config.json"):
            raise OSError(f"Config for tokenizer {name} not found. The model may not exist or is not accessible with the provided token.")

        try:
            logger.info(f"Loading tokenizer from {f'models/tokenizers/{name}'}...")
            if name == "t5":
                tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}", use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")
        except Exception as e:
            raise OSError(f"Error loading tokenizer for model {name}.") from e

        chktok = tokenizer.encode(CHK_TXT)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.info(f"model: {name}")
        logger.info(f"tokt: {tokt}")
        logger.info(f"repo: {model['repo']}")
        logger.info(f"chktok: {chktok}")
        logger.info(f"chkhsh: {chkhsh}")

        # print the "pre_tokenizer" content from the tokenizer.json
        with open(f"models/tokenizers/{name}/tokenizer.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            normalizer = cfg["normalizer"]
            logger.info("normalizer: " + json.dumps(normalizer, indent=4))
            pre_tokenizer = cfg["pre_tokenizer"]
            logger.info("pre_tokenizer: " + json.dumps(pre_tokenizer, indent=4))
            if "ignore_merges" in cfg["model"]:
                logger.info("ignore_merges: " + json.dumps(cfg["model"]["ignore_merges"], indent=4))

        logger.info("")

    src_ifs += f"        if chkhsh == \"{chkhsh}\":\n"
    src_ifs += f"            # ref: {model['repo']}\n"
    src_ifs += f"            res = \"{name}\"\n"

src_func = f"""
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = {repr(CHK_TXT)}

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {{chktok}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
{src_ifs}
        if res is None:
            logger.warning("\\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggml-org/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {{chkhsh}}")
            logger.warning("**************************************************************************************")
            logger.warning("\\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {{repr(res)}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        return res
"""

convert_py = re.sub(
    r"(# Marker: Start get_vocab_base_pre)(.+?)( +# Marker: End get_vocab_base_pre)",
    lambda m: m.group(1) + src_func + m.group(3),
    convert_py,
    flags=re.DOTALL | re.MULTILINE,
)

convert_py_pth.write_text(convert_py, encoding="utf-8")

logger.info("+++ convert_hf_to_gguf.py was updated")

# generate tests for each tokenizer model

tests = [
    "ied 4 ½ months",
    "Äpfel",
    "",
    " ",
    "  ",
    "   ",
    "\t",
    "\n",
    "\n\n",
    "\n\n\n",
    "\t\n",
    "Hello world",
    " Hello world",
    "Hello World",
    " Hello World",
    " Hello World!",
    "Hello, world!",
    " Hello, world!",
    " this is 🦙.cpp",
    "w048 7tuijk dsdfhu",
    "нещо на Български",
    "កាន់តែពិសេសអាចខលចេញ",
    "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
    "Hello",
    " Hello",
    "  Hello",
    "   Hello",
    "    Hello",
    "    Hello\n    Hello",
    " (",
    "\n =",
    "' era",
    "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
    "!!!!!!",
    "3",
    "33",
    "333",
    "3333",
    "33333",
    "333333",
    "3333333",
    "33333333",
    "333333333",
    "Cửa Việt", # llama-bpe fails on this
    " discards",
    CHK_TXT,
]

# write the tests to ./models/ggml-vocab-{name}.gguf.inp
# the format is:
#
# test0
# __ggml_vocab_test__
# test1
# __ggml_vocab_test__
# ...
#

# with each model, encode all tests and write the results in ./models/ggml-vocab-{name}.gguf.out
# for each test, write the resulting tokens on a separate line

for model in models:
    name = model["name"]
    tokt = model["tokt"]

    # Skip if the tokenizer folder does not exist or there are other download issues previously
    if not os.path.exists(f"models/tokenizers/{name}"):
        logger.warning(f"Directory for tokenizer {name} not found. Skipping...")
        continue

    # create the tokenizer
    try:
        if name == "t5":
            tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}", use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")
    except OSError as e:
        logger.error(f"Failed to load tokenizer for model {name}. Error: {e}")
        continue  # Skip this model and continue with the next one in the loop

    if not os.path.exists(f"models/ggml-vocab-{name}.gguf"):
        logger.info(f"Skip vocab files for model {name}, no GGUF file found")
        continue

    with open(f"models/ggml-vocab-{name}.gguf.inp", "w", encoding="utf-8") as f:
        for text in tests:
            f.write(f"{text}")
            f.write("\n__ggml_vocab_test__\n")

    with open(f"models/ggml-vocab-{name}.gguf.out", "w") as f:
        for text in tests:
            res = tokenizer.encode(text, add_special_tokens=False)
            for r in res:
                f.write(f" {r}")
            f.write("\n")

    logger.info(f"Tests for {name} written in ./models/ggml-vocab-{name}.gguf.*")

# generate commands for creating vocab files

logger.info("\nRun the following commands to generate the vocab files for testing:\n")

for model in models:
    name = model["name"]

    print(f"python3 convert_hf_to_gguf.py models/tokenizers/{name}/ --outfile models/ggml-vocab-{name}.gguf --vocab-only") # noqa: NP100

logger.info("\n")
