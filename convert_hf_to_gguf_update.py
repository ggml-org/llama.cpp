#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script downloads the tokenizer models of the specified models from Huggingface and
# generates the get_vocab_base_pre() function for convert_hf_to_gguf.py
#
# This is necessary in order to analyze the type of pre-tokenizer used by the model and
# provide the necessary information to llama.cpp via the GGUF header in order to implement
# the same pre-tokenizer.
#
# ref: https://github.com/ggerganov/llama.cpp/pull/6920
#
# Instructions:
#
# - Add a new model to the "models" list
# - Run the script with your huggingface token:
#
#   python3 convert_hf_to_gguf_update.py <huggingface_token>
#
# - The convert_hf_to_gguf.py script will have had its get_vocab_base_pre() function updated
# - Update llama.cpp with the new pre-tokenizer if necessary
#
# TODO: generate tokenizer tests for llama.cpp
#

import logging
import os
import pathlib
import re

import requests
import sys
import json
import shutil

from hashlib import sha256
from enum import IntEnum, auto
from transformers import AutoTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("convert_hf_to_gguf_update")
sess = requests.Session()


class TOKENIZER_TYPE(IntEnum):
    SPM = auto()
    BPE = auto()
    WPM = auto()
    UGM = auto()


# TODO: this string has to exercise as much pre-tokenizer functionality as possible
#       will be updated with time - contributions welcome
CHK_TXT = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````\"\"\"\"......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

if len(sys.argv) == 2:
    token = sys.argv[1]
    if not token.startswith("hf_"):
        logger.info("Huggingface token seems invalid")
        logger.info("Usage: python convert_hf_to_gguf_update.py <huggingface_token>")
        sys.exit(1)
else:
    logger.info("Usage: python convert_hf_to_gguf_update.py <huggingface_token>")
    sys.exit(1)

# TODO: add models here, base models preferred
models = [
    {"name": "llama-spm",      "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/meta-llama/Llama-2-7b-hf", },
    {"name": "llama-bpe",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/meta-llama/Meta-Llama-3-8B", },
    {"name": "phi-3",          "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct", },
    {"name": "deepseek-llm",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-llm-7b-base", },
    {"name": "deepseek-coder", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base", },
    {"name": "falcon",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/falcon-7b", },
    {"name": "bert-bge",       "tokt": TOKENIZER_TYPE.WPM, "repo": "https://huggingface.co/BAAI/bge-small-en-v1.5", },
    {"name": "falcon3",        "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/Falcon3-7B-Base", },
    {"name": "bert-bge-large", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/BAAI/bge-large-zh-v1.5", },
    {"name": "mpt",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mosaicml/mpt-7b", },
    {"name": "starcoder",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/bigcode/starcoder2-3b", },
    {"name": "gpt-2",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/openai-community/gpt2", },
    {"name": "stablelm2",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b", },
    {"name": "refact",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/smallcloudai/Refact-1_6-base", },
    {"name": "command-r",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/CohereForAI/c4ai-command-r-v01", },
    {"name": "qwen2",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Qwen/Qwen1.5-7B", },
    {"name": "olmo",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/allenai/OLMo-1.7-7B-hf", },
    {"name": "dbrx",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/databricks/dbrx-base", },
    {"name": "jina-v1-en",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-reranker-v1-tiny-en", },
    {"name": "jina-v2-en",     "tokt": TOKENIZER_TYPE.WPM, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-en", }, # WPM!
    {"name": "jina-v2-es",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-es", },
    {"name": "jina-v2-de",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-de", },
    {"name": "smaug-bpe",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct", },
    {"name": "poro-chat",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LumiOpen/Poro-34B-chat", },
    {"name": "jina-v2-code",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/jinaai/jina-embeddings-v2-base-code", },
    {"name": "viking",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LumiOpen/Viking-7B", }, # Also used for Viking 13B and 33B
    {"name": "gemma",          "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/google/gemma-2b", },
    {"name": "gemma-2",        "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/google/gemma-2-9b", },
    {"name": "jais",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/core42/jais-13b", },
    {"name": "t5",             "tokt": TOKENIZER_TYPE.UGM, "repo": "https://huggingface.co/google-t5/t5-small", },
    {"name": "codeshell",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/WisdomShell/CodeShell-7B", },
    {"name": "tekken",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mistralai/Mistral-Nemo-Base-2407", },
    {"name": "smollm",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/HuggingFaceTB/SmolLM-135M", },
    {'name': "bloom",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/bigscience/bloom", },
    {'name': "gpt3-finnish",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/TurkuNLP/gpt3-finnish-small", },
    {"name": "exaone",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", },
    {"name": "phi-2",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/microsoft/phi-2", },
    {"name": "chameleon",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/facebook/chameleon-7b", },
    {"name": "minerva-7b",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0", },
    {"name": "roberta-bpe",    "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/sentence-transformers/stsb-roberta-base"},
    {"name": "gigachat",       "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct"},
    {"name": "megrez",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Infinigence/Megrez-3B-Instruct"},
    {"name": "deepseek-v3",    "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V3"},
]


def download_file_with_auth(url, token, save_path):
    headers = {"Authorization": f"Bearer {token}"}
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
            download_file_with_auth(f"{repo}/resolve/main/{file}", token, save_path)


for model in models:
    try:
        download_model(model)
    except Exception as e:
        logger.error(f"Failed to download model {model['name']}. Error: {e}")


# generate the source code for the convert_hf_to_gguf.py:get_vocab_base_pre() function:

src_ifs = ""
for model in models:
    name = model["name"]
    tokt = model["tokt"]

    if tokt == TOKENIZER_TYPE.SPM or tokt == TOKENIZER_TYPE.UGM:
        continue

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
        logger.error(f"Error loading tokenizer for model {name}. The model may not exist or is not accessible with the provided token. Error: {e}")
        continue  # Skip to the next model if the tokenizer can't be loaded

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
            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {{chkhsh}}")
            logger.warning("**************************************************************************************")
            logger.warning("\\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {{repr(res)}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        return res
"""

convert_py_pth = pathlib.Path("convert_hf_to_gguf.py")
convert_py = convert_py_pth.read_text(encoding="utf-8")
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
    "Führer",
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
