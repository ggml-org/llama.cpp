import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import requests
from collections.abc import Sequence
from contextlib import closing
from re import RegexFlag
from typing import Any, Literal, cast

import aiohttp
import numpy as np
import openai
from openai.types.chat import ChatCompletionChunk
from behave import step  # pyright: ignore[reportAttributeAccessIssue]
from behave.api.async_step import async_run_until_complete
from prometheus_client import parser

# pyright: reportRedeclaration=false

@step("a server listening on {server_fqdn}:{server_port}")
def step_server_config(context, server_fqdn: str, server_port: str):
    context.server_fqdn = server_fqdn
    context.server_port = int(server_port)
    context.n_threads = None
    context.n_gpu_layer = None
    if 'PORT' in os.environ:
        context.server_port = int(os.environ['PORT'])
        print(f"$PORT set, overriding server port with to {context.server_port}")
    if 'FQDN' in os.environ:
        context.server_fqdn = os.environ['FQDN']
        print(f"$FQDN set, overriding server fqdn with to {context.server_fqdn}")
    if 'N_GPU_LAYERS' in os.environ:
        context.n_gpu_layer = int(os.environ['N_GPU_LAYERS'])
        print(f"$N_GPU_LAYERS set, overriding n_gpu_layer with to {context.n_gpu_layer}")

    context.base_url = f'http://{context.server_fqdn}:{context.server_port}'

    context.model_alias = None
    context.model_file = None
    context.model_hf_repo = None
    context.model_hf_file = None
    context.model_url = None
    context.n_batch = None
    context.n_ubatch = None
    context.n_ctx = None
    context.n_ga = None
    context.n_ga_w = None
    context.n_predict = None
    context.n_prompts = 0
    context.n_server_predict = None
    context.slot_save_path = None
    context.id_slot = None
    context.cache_prompt = None
    context.n_slots = None
    context.prompt_prefix = None
    context.prompt_suffix = None
    context.server_api_key = None
    context.server_continuous_batching = False
    context.server_embeddings = False
    context.server_metrics = False
    context.server_process = None
    context.seed = None
    context.draft = None
    context.server_seed = None
    context.user_api_key = None
    context.response_format = None
    context.temperature = None
    context.lora_file = None

    context.tasks_result = []
    context.concurrent_tasks = []
    context.prompts = []


@step('a model file {hf_file} from HF repo {hf_repo}')
def step_download_hf_model(context, hf_file: str, hf_repo: str):
    context.model_hf_repo = hf_repo
    context.model_hf_file = hf_file
    context.model_file = os.path.basename(hf_file)

@step('a lora adapter file from {lora_file_url}')
def step_download_lora_file(context, lora_file_url: str):
    file_name = lora_file_url.split('/').pop()
    context.lora_file = f'../../../{file_name}'
    with open(context.lora_file, 'wb') as f:
        f.write(requests.get(lora_file_url).content)

@step('a model file {model_file}')
def step_model_file(context, model_file: str):
    context.model_file = model_file


@step('a model url {model_url}')
def step_model_url(context, model_url: str):
    context.model_url = model_url


@step('a model alias {model_alias}')
def step_model_alias(context, model_alias: str):
    context.model_alias = model_alias


@step('{seed:d} as server seed')
def step_seed(context, seed: int):
    context.server_seed = seed


@step('{ngl:d} GPU offloaded layers')
def step_n_gpu_layer(context, ngl: int):
    if 'N_GPU_LAYERS' in os.environ:
        new_ngl = int(os.environ['N_GPU_LAYERS'])
        if context.debug:
            print(f"-ngl upgraded from {ngl} to {new_ngl}")
        ngl = new_ngl
    context.n_gpu_layer = ngl


@step('{n_threads:d} threads')
def step_n_threads(context, n_threads: int):
    context.n_thread = n_threads


@step('{draft:d} as draft')
def step_draft(context, draft: int):
    context.draft = draft


@step('{n_ctx:d} KV cache size')
def step_n_ctx(context, n_ctx: int):
    context.n_ctx = n_ctx


@step('{n_slots:d} slots')
def step_n_slots(context, n_slots: int):
    context.n_slots = n_slots


@step('{n_predict:d} server max tokens to predict')
def step_server_n_predict(context, n_predict: int):
    context.n_server_predict = n_predict


@step('{slot_save_path} as slot save path')
def step_slot_save_path(context, slot_save_path: str):
    context.slot_save_path = slot_save_path


@step('using slot id {id_slot:d}')
def step_id_slot(context, id_slot: int):
    context.id_slot = id_slot


@step('prompt caching is enabled')
def step_enable_prompt_cache(context):
    context.cache_prompt = True


@step('continuous batching')
def step_server_continuous_batching(context):
    context.server_continuous_batching = True


@step('embeddings extraction')
def step_server_embeddings(context):
    context.server_embeddings = True


@step('prometheus compatible metrics exposed')
def step_server_metrics(context):
    context.server_metrics = True


@step("the server is starting")
def step_start_server(context):
    start_server_background(context)
    attempts = 0
    max_attempts = 20
    if 'GITHUB_ACTIONS' in os.environ:
        max_attempts *= 2

    addrs = socket.getaddrinfo(context.server_fqdn, context.server_port, type=socket.SOCK_STREAM)
    family, typ, proto, _, sockaddr = addrs[0]

    while True:
        with closing(socket.socket(family, typ, proto)) as sock:
            result = sock.connect_ex(sockaddr)
            if result == 0:
                print("\x1b[33;46mserver started!\x1b[0m")
                return
            attempts += 1
            if attempts > max_attempts:
                assert False, "server not started"
            print(f"waiting for server to start, connect error code = {result}...")
            time.sleep(0.1)


@step("the server is {expecting_status}")
@async_run_until_complete
async def step_wait_for_the_server_to_be_started(context, expecting_status: Literal['healthy', 'ready', 'idle', 'busy'] | str):
    match expecting_status:
        case 'healthy':
            await wait_for_slots_status(context, context.base_url, 200,
                                        timeout=30)

        case 'ready' | 'idle':
            await wait_for_slots_status(context, context.base_url, 200,
                                        timeout=30,
                                        params={'fail_on_no_slot': 1},
                                        slots_idle=context.n_slots,
                                        slots_processing=0)
        case 'busy':
            await wait_for_slots_status(context, context.base_url, 503,
                                        params={'fail_on_no_slot': 1},
                                        slots_idle=0,
                                        slots_processing=context.n_slots)
        case _:
            assert False, "unknown status"


@step('all slots are {expected_slot_status_string}')
@async_run_until_complete
async def step_all_slots_status(context, expected_slot_status_string: Literal['idle', 'busy'] | str):
    match expected_slot_status_string:
        case 'idle':
            expected_slot_status = 0
        case 'busy':
            expected_slot_status = 1
        case _:
            assert False, "unknown status"

    expected_slots = [{'id': slot_id, 'state': expected_slot_status}
                      for slot_id in range(context.n_slots)]
    await request_slots_status(context, expected_slots)


@step('a completion request with {api_error} api error')
@async_run_until_complete
async def step_request_completion(context, api_error: Literal['raised'] | str):
    expect_api_error = api_error == 'raised'
    seeds = await completions_seed(context, num_seeds=1)
    completion = await request_completion(context.prompts.pop(),
                                          seeds[0] if seeds is not None else seeds,
                                          context.base_url,
                                          debug=context.debug,
                                          n_predict=context.n_predict,
                                          cache_prompt=context.cache_prompt,
                                          id_slot=context.id_slot,
                                          expect_api_error=expect_api_error,
                                          user_api_key=context.user_api_key,
                                          temperature=context.temperature)
    context.tasks_result.append(completion)
    if context.debug:
        print(f"Completion response: {completion}")
    if expect_api_error:
        assert completion == 401, f"completion must be an 401 status code: {completion}"


@step('{predicted_n:d} tokens are predicted matching {re_content}')
def step_n_tokens_predicted_with_content(context, predicted_n, re_content):
    context.completion = context.tasks_result.pop()
    assert_n_tokens_predicted(context.completion, predicted_n, re_content)


@step('{predicted_n:d} tokens are predicted')
def step_n_tokens_predicted(context, predicted_n):
    context.completion = context.tasks_result.pop()
    assert_n_tokens_predicted(context.completion, predicted_n)


@step('all predictions are equal')
@async_run_until_complete
async def step_predictions_equal(context):
    n_completions = await gather_tasks_results(context)
    assert n_completions >= 2, "need at least 2 completions"
    assert_all_predictions_equal(context.tasks_result)
    context.tasks_result = []


@step('all predictions are different')
@async_run_until_complete
async def step_predictions_different(context):
    n_completions = await gather_tasks_results(context)
    assert n_completions >= 2, "need at least 2 completions"
    assert_all_predictions_different(context.tasks_result)
    context.tasks_result = []


@step('all token probabilities are equal')
@async_run_until_complete
async def step_token_probabilities_equal(context):
    n_completions = await gather_tasks_results(context)
    assert n_completions >= 2, "need at least 2 completions"
    assert_all_token_probabilities_equal(context.tasks_result)
    context.tasks_result = []


@step('the completion is  truncated')
def step_assert_completion_truncated(context):
    step_assert_completion_truncated(context, '')


@step('the completion is {truncated} truncated')
def step_assert_completion_truncated(context, truncated):
    truncated = truncated != "not"
    assert context.completion['truncated'] == truncated, f'{context.completion}'


@step('{n_prompt:d} prompt tokens are processed')
def step_impl(context, n_prompt):
    assert n_prompt < 0 or n_prompt == context.completion['timings']['prompt_n'], f"n_prompt={context.completion['timings']['prompt_n']}"


@step('a user prompt {user_prompt}')
def step_user_prompt(context, user_prompt):
    context.prompts.append(user_prompt)
    context.n_prompts = len(context.prompts)


@step('a system prompt {system_prompt}')
def step_system_prompt(context, system_prompt):
    context.system_prompt = system_prompt


@step('a model {model}')
def step_model(context, model):
    context.model = model


@step('{max_tokens:d} max tokens to predict')
def step_max_tokens(context, max_tokens):
    context.n_predict = max_tokens


@step('a response format {response_format}')
def step_response_format(context, response_format):
    context.response_format = json.loads(response_format)


@step('{temperature:f} temperature')
def step_temperature(context, temperature):
    context.temperature = temperature


@step('streaming is {enable_streaming}')
def step_streaming(context, enable_streaming):
    context.enable_streaming = enable_streaming == 'enabled'


@step('a user api key {user_api_key}')
def step_user_api_key(context, user_api_key):
    context.user_api_key = user_api_key


@step('no user api key')
def step_no_user_api_key(context):
    context.user_api_key = None


@step('a user api key ')
def step_no_user_api_key_space(context):
    context.user_api_key = None


@step('a server api key {server_api_key}')
def step_server_api_key(context, server_api_key):
    context.server_api_key = server_api_key


@step('{n_junk:d} as number of junk')
def step_n_junk(context, n_junk):
    context.n_junk = n_junk


@step('{n_batch:d} as batch size')
def step_n_batch(context, n_batch):
    context.n_batch = n_batch


@step('{n_ubatch:d} as ubatch size')
def step_n_ubatch(context, n_ubatch):
    context.n_ubatch = n_ubatch


@step('{seed:d} as seed')
def step_seed(context, seed):
    if context.seed is None:
        context.seed = [seed]
    else:
        context.seed.append(seed)


@step('BOS token is {bos:d}')
def step_bos_token(context, bos):
    context.bos = bos


@step('a prefix prompt')
def step_prompt_prefix(context):
    context.prompt_prefix = context_text(context)


@step('a junk suffix prompt')
def step_prompt_junk_suffix(context):
    context.prompt_junk_suffix = context_text(context)


@step('a suffix prompt')
def step_prompt_suffix(context):
    context.prompt_suffix = context_text(context)


@step('{n_ga:d} group attention factor'
      ' to extend context size through self-extend')
def step_impl(context, n_ga):
    context.n_ga = n_ga


@step('{n_ga_w:d} group attention width to extend context size through self-extend')
def step_impl(context, n_ga_w):
    context.n_ga_w = n_ga_w


@step('a passkey prompt template')
def step_prompt_passkey(context):
    context.prompt_passkey = context_text(context)


@step('{n_prompts:d} fixed prompts')
def step_fixed_prompts(context, n_prompts):
    context.prompts.extend([str(0)*(context.n_batch if context.n_batch is not None else 512) for i in range(n_prompts)])
    context.n_prompts = n_prompts


@step('a "{passkey}" passkey challenge prompt with the passkey inserted every {i_pos:d} junk')
def step_prompt_passkey(context, passkey, i_pos):
    prompt = ""
    for i in range(context.n_junk):
        if i % context.n_junk == i_pos:
            prompt += context.prompt_passkey # the passkey is already substituted
        prompt += context.prompt_junk_suffix
    if context.debug:
        passkey_highlight = "\x1b[33m" + passkey + "\x1b[0m"
        print(f"Passkey challenge:\n```{prompt.replace(passkey, passkey_highlight)}```")
    context.prompts.append(context.prompt_prefix + prompt + context.prompt_suffix)
    context.n_prompts = len(context.prompts)


@step('an OAI compatible chat completions request with {api_error} api error')
@async_run_until_complete
async def step_oai_chat_completions(context, api_error):
    if context.debug:
        print(f"Submitting OAI compatible completions request...")
    expect_api_error = api_error == 'raised'
    seeds = await completions_seed(context, num_seeds=1),
    completion = await oai_chat_completions(context.prompts.pop(),
                                            seeds[0] if seeds is not None else seeds,
                                            context.system_prompt,
                                            context.base_url,
                                            '/v1/chat',
                                            False,
                                            model=context.model if hasattr(context, 'model') else None,

                                            n_predict=context.n_predict
                                            if hasattr(context, 'n_predict') else None,

                                            enable_streaming=context.enable_streaming
                                            if hasattr(context, 'enable_streaming') else None,

                                            response_format=context.response_format
                                            if hasattr(context, 'response_format') else None,

                                            user_api_key=context.user_api_key
                                            if hasattr(context, 'user_api_key') else None,

                                            expect_api_error=expect_api_error)
    context.tasks_result.append(completion)
    if context.debug:
        print(f"Completion response: {completion}")
    if expect_api_error:
        assert completion == 401, f"completion must be an 401 status code: {completion}"

    if context.debug:
        print(f"Completion response: {completion}")


@step('a prompt')
def step_a_prompt(context):
    context.prompts.append(context_text(context))
    context.n_prompts = len(context.prompts)


@step('a prompt {prompt}')
def step_a_prompt_prompt(context, prompt):
    context.prompts.append(prompt)
    context.n_prompts = len(context.prompts)


@step('{num_prompts:d} prompts {prompt} with seed {seed:d}')
def step_many_prompts(context, num_prompts, prompt, seed):
    if context.seed is None:
        context.seed = []
    for _ in range(num_prompts):
        context.seed.append(seed)
        context.prompts.append(prompt)
    context.n_prompts = len(context.prompts)


@step('concurrent completion requests')
@async_run_until_complete()
async def step_concurrent_completion_requests(context):
    await concurrent_requests(
        context,
        request_completion,
        # prompt is inserted automatically
        context.base_url,
        debug=context.debug,
        prompt_prefix=context.prompt_prefix,
        prompt_suffix=context.prompt_suffix,
        n_predict=context.n_predict if hasattr(context, 'n_predict') else None,
        user_api_key=context.user_api_key if hasattr(context, 'user_api_key') else None,
        temperature=context.temperature,
    )


@step('concurrent OAI completions requests')
@async_run_until_complete
async def step_oai_chat_completions(context):
    await concurrent_requests(context, oai_chat_completions,
                              # user_prompt is inserted automatically
                              context.system_prompt,
                              context.base_url,
                              '/v1/chat/completions',
                              True,  # async_client
                              model=context.model
                              if hasattr(context, 'model') else None,
                              n_predict=context.n_predict
                              if hasattr(context, 'n_predict') else None,
                              enable_streaming=context.enable_streaming
                              if hasattr(context, 'enable_streaming') else None,
                              response_format=context.response_format
                              if hasattr(context, 'response_format') else None,
                              user_api_key=context.user_api_key
                              if hasattr(context, 'user_api_key') else None)


@step('concurrent OAI completions requests no v1')
@async_run_until_complete
async def step_oai_chat_completions(context):
    await concurrent_requests(context, oai_chat_completions,
                              # user_prompt is inserted automatically
                              context.system_prompt,
                              context.base_url,
                              '/chat/completions',
                              True,  # async_client
                              model=context.model
                              if hasattr(context, 'model') else None,
                              n_predict=context.n_predict
                              if hasattr(context, 'n_predict') else None,
                              enable_streaming=context.enable_streaming
                              if hasattr(context, 'enable_streaming') else None,
                              response_format=context.response_format
                              if hasattr(context, 'response_format') else None,
                              user_api_key=context.user_api_key
                              if hasattr(context, 'user_api_key') else None)


@step('all prompts are predicted')
@async_run_until_complete
async def step_all_prompts_are_predicted(context):
    await all_prompts_are_predicted(context)


@step('all prompts are predicted with {n_expected_predicted:d} tokens')
@async_run_until_complete
async def step_all_prompts_are_predicted_with_n_tokens(context, n_expected_predicted):
    await all_prompts_are_predicted(context, n_expected_predicted)


async def all_prompts_are_predicted(context, expected_predicted_n=None):
    n_completions = await gather_tasks_results(context)
    assert n_completions > 0
    for i in range(n_completions):
        assert_n_tokens_predicted(context.tasks_result.pop(), expected_predicted_n=expected_predicted_n)
    assert len(context.concurrent_tasks) == 0, f"{len(context.concurrent_tasks)} pending requests"


@step('embeddings are computed for')
@async_run_until_complete
async def step_compute_embedding(context):
    context.n_prompts = 1
    context.embeddings = await request_embedding(context_text(context), None, base_url=context.base_url)


@step('all embeddings are the same')
@async_run_until_complete
async def step_all_embeddings_are_the_same(context):
    n_embedding_requests = await gather_tasks_results(context)
    assert n_embedding_requests > 0
    embeddings = []
    for i in range(n_embedding_requests):
        embedding = context.tasks_result.pop().pop()
        embeddings.append(embedding)
        assert_embeddings(embedding)
    n = len(embeddings)
    for i in range(n-1):
        for j in range(i+1, n):
            embedding1 = np.array(embeddings[i])
            embedding2 = np.array(embeddings[j])
            if context.debug:
                print(f"embedding1: {embedding1[-8:]}")
                print(f"embedding2: {embedding2[-8:]}")
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            msg = f"Similarity between {i} and {j}: {similarity:.10f}"
            if context.debug:
                print(f"{msg}")
            assert np.isclose(similarity, 1.0, rtol=1e-05, atol=1e-08, equal_nan=False), msg


@step('embeddings are generated')
def step_assert_embeddings(context):
    assert context.n_prompts == len(context.embeddings), (f"unexpected response:\n"
                                                             f"context.n_prompts={context.n_prompts}\n"
                                                             f"context.embeddings={context.embeddings}")
    for embedding in context.embeddings:
        assert_embeddings(embedding)


@step('an OAI compatible embeddings computation request for')
@async_run_until_complete
async def step_oai_compute_embeddings(context):
    context.n_prompts = 1
    context.embeddings = await request_oai_embeddings(context_text(context), None,
                                                      base_url=context.base_url,
                                                      user_api_key=context.user_api_key,
                                                      model=context.model)


@step('an OAI compatible embeddings computation request for multiple inputs')
@async_run_until_complete
async def step_oai_compute_embeddings_multiple_inputs(context):
    context.embeddings = await request_oai_embeddings(context.prompts, None,
                                                      base_url=context.base_url,
                                                      user_api_key=context.user_api_key,
                                                      model=context.model)
    context.prompts.clear()


@step('concurrent embedding requests')
@async_run_until_complete()
async def step_concurrent_embedding_requests(context):
    await concurrent_requests(context,
                              request_embedding,
                              # prompt is inserted automatically
                              base_url=context.base_url)


@step('concurrent OAI embedding requests')
@async_run_until_complete()
async def step_concurrent_oai_embedding_requests(context):
    await concurrent_requests(context,
                              request_oai_embeddings,
                              # prompt is inserted automatically
                              base_url=context.base_url,
                              async_client=True,
                              model=context.model)


@step('all embeddings are generated')
@async_run_until_complete()
async def all_embeddings_are_generated(context):
    n_embedding_requests = await gather_tasks_results(context)
    assert n_embedding_requests == context.n_prompts
    for i in range(n_embedding_requests):
        assert_embeddings(context.tasks_result.pop().pop())


@step('adding special tokens')
def step_tokenize_set_add_special(context):
    context.tokenize_add_special = True


@step('tokenizing')
@async_run_until_complete
async def step_tokenize(context):
    context.tokenized_text = context_text(context)
    async with aiohttp.ClientSession() as session:
        tokenize_args = {
            "content": context.tokenized_text,
        }
        if getattr(context, 'tokenize_add_special', None) is not None:
            tokenize_args['add_special'] = context.tokenize_add_special
        async with session.post(f'{context.base_url}/tokenize',
                                json=tokenize_args) as response:
            assert response.status == 200
            tokenize_json = await response.json()
            context.tokens = tokenize_json['tokens']


@step('tokens can be detokenized')
@async_run_until_complete
async def step_detokenize(context):
    assert len(context.tokens) > 0
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{context.base_url}/detokenize',
                                json={
                                    "tokens": context.tokens,
                                }) as response:
            assert response.status == 200
            detokenize_json = await response.json()
            # SPM tokenizer adds a whitespace prefix: https://github.com/google/sentencepiece/issues/15
            assert context.tokenized_text == detokenize_json['content'].strip()


@step('tokens begin with BOS')
def step_strings_for_tokenization(context):
    assert context.tokens[0] == context.bos


@step('tokens do not begin with BOS')
def step_strings_for_tokenization(context):
    assert context.tokens[0] != context.bos


@step('first token is removed')
def step_strings_for_tokenization(context):
    context.tokens = context.tokens[1:]


@step('an OPTIONS request is sent from {origin}')
@async_run_until_complete
async def step_options_request(context, origin):
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {context.user_api_key}', 'Origin': origin}
        async with session.options(f'{context.base_url}/v1/chat/completions',
                                    headers=headers) as response:
            assert response.status == 200
            context.options_response = response


@step('CORS header {cors_header} is set to {cors_header_value}')
def step_check_options_header_value(context, cors_header, cors_header_value):
    assert context.options_response.headers[cors_header] == cors_header_value


@step('prometheus metrics are exposed')
@async_run_until_complete
async def step_prometheus_metrics_exported(context):
    async with aiohttp.ClientSession() as session:
        async with await session.get(f'{context.base_url}/metrics') as metrics_response:
            assert metrics_response.status == 200
            assert metrics_response.headers['Content-Type'] == "text/plain; version=0.0.4"
            metrics_raw = await metrics_response.text()
            metric_exported = False
            if context.debug:
                print(f"/metrics answer:\n{metrics_raw}")
            context.metrics = {}
            for metric in parser.text_string_to_metric_families(metrics_raw):
                match metric.name:
                    case "llamacpp:kv_cache_usage_ratio":
                        assert len(metric.samples) > 0
                        metric_exported = True
                context.metrics[metric.name] = metric
            assert int(metrics_response.headers["Process-Start-Time-Unix"]) > 0, "no header process start time"
            assert metric_exported, "No metrics exported"


@step('metric {metric_name} is {metric_value:d}')
def step_assert_metric_value(context, metric_name, metric_value):
    if metric_name not in context.metrics:
        assert False, f"no metric {metric_name} in {context.metrics.keys()}"
    assert context.metrics[metric_name].samples[0].value == metric_value, f"metric: {context.metrics[metric_name]}"


@step('available models')
def step_available_models(context):
    # openai client always expects an api_key
    openai.api_key = context.user_api_key if context.user_api_key is not None else 'nope'
    openai.base_url = f'{context.base_url}/v1/'
    context.models = openai.models.list().data


@step('{n_model:d} models are supported')
def step_supported_models(context, n_model):
    if context.debug:
        print("server models available:", context.models)
    assert len(context.models) == n_model


@step('model {i_model:d} is {param} {preposition} {param_value}')
def step_supported_models(context, i_model: int, param: Literal['identified', 'trained'] | str, preposition: str, param_value: str):
    assert i_model < len(context.models)
    model = context.models[i_model]

    param_value = param_value.split(' ', 1)[0]
    match param:
        case 'identified':
            value = model.id
        case 'trained':
            value = str(model.meta["n_ctx_train"])
        case _:
            assert False, "param {param} not supported"
    assert param_value == value, f"model param {param} {value} != {param_value}"


async def concurrent_requests(context, f_completion, *args, **kwargs):
    context.n_prompts = len(context.prompts)
    if context.debug:
        print(f"starting {context.n_prompts} concurrent completion requests...")
    assert context.n_prompts > 0
    seeds = await completions_seed(context)
    assert seeds is not None
    for prompt_no in range(context.n_prompts):
        shifted_args = [context.prompts.pop(), seeds[prompt_no], *args]
        context.concurrent_tasks.append(asyncio.create_task(f_completion(*shifted_args, **kwargs)))
    await asyncio.sleep(0.1)


@step('the slot {slot_id:d} is saved with filename "{filename}"')
@async_run_until_complete
async def step_save_slot(context, slot_id, filename):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{context.base_url}/slots/{slot_id}?action=save',
                                json={"filename": filename},
                                headers={"Content-Type": "application/json"}) as response:
            context.response = response


@step('the slot {slot_id:d} is restored with filename "{filename}"')
@async_run_until_complete
async def step_restore_slot(context, slot_id, filename):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{context.base_url}/slots/{slot_id}?action=restore',
                                json={"filename": filename},
                                headers={"Content-Type": "application/json"}) as response:
            context.response = response


@step('the slot {slot_id:d} is erased')
@async_run_until_complete
async def step_erase_slot(context, slot_id):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{context.base_url}/slots/{slot_id}?action=erase',
                                headers={"Content-Type": "application/json"}) as response:
            context.response = response


@step('switch {on_or_off} lora adapter {lora_id:d}')
@async_run_until_complete
async def toggle_lora_adapter(context, on_or_off: str, lora_id: int):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{context.base_url}/lora-adapters',
                                json=[{'id': lora_id, 'scale': 1 if on_or_off == 'on' else 0}],
                                headers={"Content-Type": "application/json"}) as response:
            context.response = response
            print([{'id': lora_id, 'scale': 1 if on_or_off == 'on' else 0}])


@step('the server responds with status code {status_code:d}')
def step_server_responds_with_status_code(context, status_code):
    assert context.response.status == status_code


async def request_completion(prompt,
                             seed,
                             base_url,
                             debug=False,
                             prompt_prefix=None,
                             prompt_suffix=None,
                             n_predict=None,
                             cache_prompt=False,
                             id_slot=None,
                             expect_api_error=None,
                             user_api_key=None,
                             temperature=None) -> int | dict[str, Any]:
    if debug:
        print(f"Sending completion request: {prompt}")
    origin = "my.super.domain"
    headers = {
        'Origin': origin
    }
    if user_api_key is not None:
        if debug:
            print(f"Set user_api_key: {user_api_key}")
        headers['Authorization'] = f'Bearer {user_api_key}'

    async with aiohttp.ClientSession() as session:
        async with session.post(f'{base_url}/completion',
                                json={
                                    "input_prefix": prompt_prefix,
                                    "prompt": prompt,
                                    "input_suffix": prompt_suffix,
                                    "n_predict": n_predict if n_predict is not None else -1,
                                    "cache_prompt": cache_prompt,
                                    "id_slot": id_slot,
                                    "seed": seed if seed is not None else 42,
                                    "temperature": temperature if temperature is not None else 0.8,
                                    "n_probs": 2,
                                },
                                headers=headers,
                                timeout=3600) as response:
            if expect_api_error is None or not expect_api_error:
                assert response.status == 200
                assert response.headers['Access-Control-Allow-Origin'] == origin
                return await response.json()
            else:
                return response.status


async def oai_chat_completions(user_prompt,
                               seed,
                               system_prompt,
                               base_url: str,
                               base_path: str,
                               async_client,
                               debug=False,
                               temperature=None,
                               model=None,
                               n_predict=None,
                               enable_streaming=None,
                               response_format=None,
                               user_api_key=None,
                               expect_api_error=None) -> int | dict[str, Any]:
    if debug:
        print(f"Sending OAI Chat completions request: {user_prompt}")
    # openai client always expects an api key
    user_api_key = user_api_key if user_api_key is not None else 'nope'
    seed = seed if seed is not None else 42
    enable_streaming = enable_streaming if enable_streaming is not None else False
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "model": model,
        "max_tokens": n_predict,
        "stream": enable_streaming,
        "temperature": temperature if temperature is not None else 0.0,
        "seed": seed,
    }
    if response_format is not None:
        payload['response_format'] = response_format
    completion_response = {
        'content': '',
        'timings': {
            'predicted_n': 0,
            'prompt_n': 0
        }
    }
    if async_client:
        origin = 'llama.cpp'
        headers = {'Authorization': f'Bearer {user_api_key}', 'Origin': origin}
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{base_url}{base_path}',
                                    json=payload,
                                    headers=headers) as response:
                if enable_streaming:
                    assert response.status == 200
                    assert response.headers['Access-Control-Allow-Origin'] == origin
                    assert response.headers['Content-Type'] == "text/event-stream"
                    event_received = True
                    while event_received:
                        event_received = False
                        async for line_in_bytes in response.content:
                            line = line_in_bytes.decode('utf-8')
                            line = line.rstrip('\n').rstrip('\r')
                            if line == '':
                                continue
                            event_data = line.split(': ', 1)
                            assert event_data[0] == 'data', f'Bad event code received: ```{event_data}```'
                            chunk_raw = event_data[1]

                            chunk = json.loads(chunk_raw)
                            assert len(chunk['choices']) == 1, f"no choices provided, line ```{line}```"
                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta:
                                completion_response['content'] += delta['content']
                                completion_response['timings']['predicted_n'] += 1
                else:
                    if expect_api_error is None or not expect_api_error:
                        assert response.status == 200
                        assert response.headers['Access-Control-Allow-Origin'] == origin
                        assert response.headers['Content-Type'] == "application/json; charset=utf-8"
                        chat_completion_raw = await response.json()
                        completion_response = {
                            'content': chat_completion_raw['choices'][0]['message'],
                            'timings': {
                                'predicted_n': chat_completion_raw['usage']['completion_tokens'],
                                'prompt_n': chat_completion_raw['usage']['prompt_tokens']
                            }
                        }
                    else:
                        return response.status
    else:
        try:
            openai.api_key = user_api_key
            openai.base_url = f'{base_url}{base_path.removesuffix("chat")}'
            assert model is not None
            chat_completion = openai.chat.completions.create(
                messages=payload['messages'],
                model=model,
                max_tokens=n_predict,
                stream=enable_streaming,
                response_format=payload.get('response_format') or openai.NOT_GIVEN,
                seed=seed,
                temperature=payload['temperature']
            )
        except openai.AuthenticationError as e:
            if expect_api_error is not None and expect_api_error:
                return 401
            else:
                assert False, f'error raised: {e}'

        if enable_streaming:
            chat_completion = cast(openai.Stream[ChatCompletionChunk], chat_completion)
            for chunk in chat_completion:
                assert len(chunk.choices) == 1
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    completion_response['content'] += delta.content
                    completion_response['timings']['predicted_n'] += 1
                completion_response['truncated'] = chunk.choices[0].finish_reason != 'stop'
        else:
            assert len(chat_completion.choices) == 1
            assert chat_completion.usage is not None
            completion_response = {
                'content': chat_completion.choices[0].message.content,
                'timings': {
                    'predicted_n': chat_completion.usage.completion_tokens,
                    'prompt_n': chat_completion.usage.prompt_tokens
                    },
                'truncated': chat_completion.choices[0].finish_reason != 'stop'
            }
    if debug:
        print("OAI response formatted to llama.cpp:", completion_response)
    return completion_response


async def request_embedding(content, seed, base_url=None) -> list[list[float]]:
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{base_url}/embedding',
                                json={
                                    "content": content,
                                }) as response:
            assert response.status == 200
            response_json = await response.json()
            return [response_json['embedding']]


async def request_oai_embeddings(input, seed,
                                 base_url=None, user_api_key=None,
                                 model=None, async_client=False) -> list[list[float]]:
    # openai client always expects an api_key
    user_api_key = user_api_key if user_api_key is not None else 'nope'
    if async_client:
        origin = 'llama.cpp'
        headers=[]
        if user_api_key is not None:
            headers = {'Authorization': f'Bearer {user_api_key}', 'Origin': origin}
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{base_url}/v1/embeddings',
                                    json={
                                        "input": input,
                                        "model": model,
                                    },
                                    headers=headers,
                                    timeout=3600) as response:
                assert response.status == 200, f"received status code not expected: {response.status}"
                assert response.headers['Access-Control-Allow-Origin'] == origin
                assert response.headers['Content-Type'] == "application/json; charset=utf-8"
                response_json = await response.json()
                assert response_json['model'] == model, f"invalid model received: {response_json['model']}"
                assert response_json['object'] == 'list'
                if isinstance(input, Sequence):
                    embeddings = []
                    for an_oai_embeddings in response_json['data']:
                        embeddings.append(an_oai_embeddings['embedding'])
                else:
                    embeddings = [response_json['data']['embedding']]
                return embeddings
    else:
        openai.api_key = user_api_key
        openai.base_url = f'{base_url}/v1/'
        assert model is not None
        oai_embeddings = openai.embeddings.create(
            model=model,
            input=input,
        )

        return [e.embedding for e in oai_embeddings.data]


def assert_n_tokens_predicted(completion_response, expected_predicted_n=None, re_content=None):
    content = completion_response['content']
    n_predicted = completion_response['timings']['predicted_n']
    assert len(content) > 0, "no token predicted"
    if re_content is not None:
        p = re.compile(re_content, flags=RegexFlag.IGNORECASE | RegexFlag.MULTILINE | RegexFlag.DOTALL)
        matches = p.finditer(content)
        last_match = 0
        highlighted = ''
        for match in matches:
            start, end = match.span()
            highlighted += content[last_match: start]
            highlighted += '\x1b[33m'
            highlighted += content[start: end]
            highlighted += '\x1b[0m'
            last_match = end
        highlighted += content[last_match:]
        if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'ON':
          print(f"Checking completion response: {highlighted}")
        assert last_match > 0, f'/{re_content}/ must match ```{highlighted}```'
    if expected_predicted_n and expected_predicted_n > 0:
        assert n_predicted == expected_predicted_n, (f'invalid number of tokens predicted:'
                                                     f' {n_predicted} <> {expected_predicted_n}')

def assert_all_predictions_equal(completion_responses):
    if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'ON':
        for i, response_i in enumerate(completion_responses):
            content_i = response_i['content']
            print(f"content {i}: {content_i}")
    for i, response_i in enumerate(completion_responses):
        content_i = response_i['content']
        for j, response_j in enumerate(completion_responses):
            if i == j:
                continue
            content_j = response_j['content']
            assert content_i == content_j, "contents not equal"


def assert_all_predictions_different(completion_responses):
    if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'ON':
        for i, response_i in enumerate(completion_responses):
            content_i = response_i['content']
            print(f"content {i}: {content_i}")
    for i, response_i in enumerate(completion_responses):
        content_i = response_i['content']
        for j, response_j in enumerate(completion_responses):
            if i == j:
                continue
            content_j = response_j['content']
            assert content_i != content_j, "contents not different"


def assert_all_token_probabilities_equal(completion_responses):
    n_predict = len(completion_responses[0]['completion_probabilities'])
    if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'ON':
        for pos in range(n_predict):
            for i, response_i in enumerate(completion_responses):
                probs_i = response_i['completion_probabilities'][pos]['probs']
                print(f"pos {pos}, probs {i}: {probs_i}")
    for pos in range(n_predict):
        for i, response_i in enumerate(completion_responses):
            probs_i = response_i['completion_probabilities'][pos]['probs']
            for j, response_j in enumerate(completion_responses):
                if i == j:
                    continue
                probs_j = response_j['completion_probabilities'][pos]['probs']
                assert probs_i == probs_j, "contents not equal"


async def gather_tasks_results(context):
    n_tasks = len(context.concurrent_tasks)
    if context.debug:
        print(f"Waiting for all {n_tasks} tasks results...")
    for task_no in range(n_tasks):
        context.tasks_result.append(await context.concurrent_tasks.pop())
    n_completions = len(context.tasks_result)
    return n_completions


async def wait_for_slots_status(context,
                                base_url,
                                expected_http_status_code,
                                timeout=3,
                                params=None,
                                slots_idle=None,
                                slots_processing=None):
    if context.debug:
        print(f"Starting checking for health for expected_http_status_code={expected_http_status_code}")
    interval = 0.5
    counter = 0
    if 'GITHUB_ACTIONS' in os.environ:
        timeout *= 2

    async with aiohttp.ClientSession() as session:
        while True:
            async with await session.get(f'{base_url}/slots', params=params) as slots_response:
                status_code = slots_response.status
                slots = await slots_response.json()
                if context.debug:
                    print(f"slots responses {slots}\n")
                if status_code == 503 and status_code == expected_http_status_code:
                    return
                if status_code == 200 and status_code == expected_http_status_code:
                    n_slots_idle = sum(1 if slot["state"] == 0 else 0 for slot in slots)
                    n_slots_processing = sum(1 if slot["state"] != 0 else 0 for slot in slots)
                    if ((slots_idle is None or slots_idle == n_slots_idle)
                        and (slots_processing is None or slots_processing == n_slots_processing)):
                        return
            await asyncio.sleep(interval)

            counter += interval
            if counter >= timeout:
                # Sometimes health requests are triggered after completions are predicted
                if expected_http_status_code == 503:
                    if len(context.tasks_result) == 0:
                        print("\x1b[5;37;43mWARNING: forcing concurrent tasks,"
                              " busy health check missed, probably too fast inference\x1b[0m\n")
                        n_completions = await gather_tasks_results(context)
                        if n_completions > 0:
                            return

                assert False, f'slots check timeout exceeded {counter}s>={timeout}'


def assert_embeddings(embeddings):
    assert len(embeddings) > 0
    embeddings_computed = False
    for emb in embeddings:
        if not isinstance(emb, float):
            assert False, f"Bad embeddings: {embeddings}"
        if emb != 0:
            embeddings_computed = True
    assert embeddings_computed, f"Embeddings: {embeddings}"


async def request_slots_status(context, expected_slots):
    async with aiohttp.ClientSession() as session:
        async with await session.get(f'{context.base_url}/slots') as slots_response:
            assert slots_response.status == 200
            slots = await slots_response.json()
            assert_slots_status(slots, expected_slots)


def assert_slots_status(slots, expected_slots):
    assert len(slots) == len(expected_slots)
    for slot_id, (expected, slot) in enumerate(zip(expected_slots, slots)):
        for key in expected:
            assert expected[key] == slot[key], (f"invalid slot {slot_id}"
                                                f" expected[{key}] != slot[{key}]"
                                                f" = {expected[key]} != {slot[key]}")


async def completions_seed(context, num_seeds=None):
    if hasattr(context, "seed") and context.seed is not None:
        assert len(context.seed) == context.n_prompts
        if num_seeds is None:
            num_seeds = context.n_prompts
        assert num_seeds <= context.n_prompts
        seeds = context.seed[:num_seeds]
        context.seed = context.seed[num_seeds:] if num_seeds < context.n_prompts else None
        return seeds

    if hasattr(context, "server_seed") and context.server_seed is not None:
        if num_seeds is None:
            return [context.server_seed] * context.n_prompts
        else:
            return [context.server_seed] * num_seeds
    return None


def context_text(context):
    return context.text.replace('\r', '')


def start_server_background(context):
    if os.name == 'nt':
        context.server_path = '../../../build/bin/Release/llama-server.exe'
    else:
        context.server_path = '../../../build/bin/llama-server'
    if 'LLAMA_SERVER_BIN_PATH' in os.environ:
        context.server_path = os.environ['LLAMA_SERVER_BIN_PATH']
    server_listen_addr = context.server_fqdn
    server_args = [
        '--host', server_listen_addr,
        '--port', context.server_port,
    ]
    if context.model_file:
        server_args.extend(['--model', context.model_file])
    if context.model_url:
        server_args.extend(['--model-url', context.model_url])
    if context.model_hf_repo:
        server_args.extend(['--hf-repo', context.model_hf_repo])
    if context.model_hf_file:
        server_args.extend(['--hf-file', context.model_hf_file])
    if context.n_batch:
        server_args.extend(['--batch-size', context.n_batch])
    if context.n_ubatch:
        server_args.extend(['--ubatch-size', context.n_ubatch])
    if context.n_threads:
        server_args.extend(['--threads', context.threads])
    if context.n_gpu_layer:
        server_args.extend(['--n-gpu-layers', context.n_gpu_layer])
    if context.draft is not None:
        server_args.extend(['--draft', context.draft])
    if context.server_continuous_batching:
        server_args.append('--cont-batching')
    if context.server_embeddings:
        server_args.append('--embedding')
    if context.server_metrics:
        server_args.append('--metrics')
    if context.model_alias:
        server_args.extend(['--alias', context.model_alias])
    if context.n_ctx:
        server_args.extend(['--ctx-size', context.n_ctx])
    if context.n_slots:
        server_args.extend(['--parallel', context.n_slots])
    if context.n_server_predict:
        server_args.extend(['--n-predict', context.n_server_predict])
    if context.slot_save_path:
        server_args.extend(['--slot-save-path', context.slot_save_path])
    if context.server_api_key:
        server_args.extend(['--api-key', context.server_api_key])
    if context.n_ga:
        server_args.extend(['--grp-attn-n', context.n_ga])
    if context.n_ga_w:
        server_args.extend(['--grp-attn-w', context.n_ga_w])
    if context.debug:
        server_args.append('--verbose')
    if context.lora_file:
        server_args.extend(['--lora', context.lora_file])
    if 'SERVER_LOG_FORMAT_JSON' not in os.environ:
        server_args.extend(['--log-format', "text"])

    args = [str(arg) for arg in [context.server_path, *server_args]]
    print(f"bench: starting server with: {' '.join(args)}")

    flags = 0
    if 'nt' == os.name:
        flags |= subprocess.DETACHED_PROCESS
        flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        flags |= subprocess.CREATE_NO_WINDOW

    pkwargs = {
        'creationflags': flags,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE
    }
    context.server_process = subprocess.Popen(
        [str(arg) for arg in [context.server_path, *server_args]],
        **pkwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    def server_log(in_stream, out_stream):
        for line in iter(in_stream.readline, b''):
            print(line.decode('utf-8'), end='', file=out_stream)

    thread_stdout = threading.Thread(target=server_log, args=(context.server_process.stdout, sys.stdout))
    thread_stdout.start()

    thread_stderr = threading.Thread(target=server_log, args=(context.server_process.stderr, sys.stderr))
    thread_stderr.start()

    print(f"server pid={context.server_process.pid}, behave pid={os.getpid()}")
