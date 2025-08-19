import base64
import struct
import pytest
from openai import OpenAI
from utils import *

server = ServerPreset.bert_bge_small()

EPSILON = 1e-3

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.bert_bge_small()


def test_embedding_single():
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "I believe the meaning of life is",
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 1
    assert 'embedding' in res.body['data'][0]
    assert len(res.body['data'][0]['embedding']) > 1

    # make sure embedding vector is normalized
    assert abs(sum([x ** 2 for x in res.body['data'][0]['embedding']]) - 1) < EPSILON


def test_embedding_multiple():
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "Write a joke about AI from a very long prompt which will not be truncated",
            "This is a test",
            "This is another test",
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 4
    for d in res.body['data']:
        assert 'embedding' in d
        assert len(d['embedding']) > 1


def test_embedding_multiple_with_fa():
    server = ServerPreset.bert_bge_small_with_fa()
    server.pooling = 'last'
    server.start()
    # one of these should trigger the FA branch (i.e. context size % 256 == 0)
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "a "*253,
            "b "*254,
            "c "*255,
            "d "*256,
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 4
    for d in res.body['data']:
        assert 'embedding' in d
        assert len(d['embedding']) > 1


@pytest.mark.parametrize(
    "input,is_multi_prompt",
    [
        # do not crash on empty input
        ("", False),
        # single prompt
        ("string", False),
        ([12, 34, 56], False),
        ([12, 34, "string", 56, 78], False),
        # multiple prompts
        (["string1", "string2"], True),
        (["string1", [12, 34, 56]], True),
        ([[12, 34, 56], [12, 34, 56]], True),
        ([[12, 34, 56], [12, "string", 34, 56]], True),
    ]
)
def test_embedding_mixed_input(input, is_multi_prompt: bool):
    global server
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": input})
    assert res.status_code == 200
    data = res.body['data']
    if is_multi_prompt:
        assert len(data) == len(input)
        for d in data:
            assert 'embedding' in d
            assert len(d['embedding']) > 1
    else:
        assert 'embedding' in data[0]
        assert len(data[0]['embedding']) > 1


def test_embedding_pooling_none():
    global server
    server.pooling = 'none'
    server.start()
    res = server.make_request("POST", "/embeddings", data={
        "input": "hello hello hello",
    })
    assert res.status_code == 200
    assert 'embedding' in res.body[0]
    assert len(res.body[0]['embedding']) == 5 # 3 text tokens + 2 special

    # make sure embedding vector is not normalized
    for x in res.body[0]['embedding']:
        assert abs(sum([x ** 2 for x in x]) - 1) > EPSILON


def test_embedding_pooling_none_oai():
    global server
    server.pooling = 'none'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "hello hello hello",
    })

    # /v1/embeddings does not support pooling type 'none'
    assert res.status_code == 400
    assert "error" in res.body


def test_embedding_openai_library_single():
    global server
    server.pooling = 'last'
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.embeddings.create(model="text-embedding-3-small", input="I believe the meaning of life is")
    assert len(res.data) == 1
    assert len(res.data[0].embedding) > 1


def test_embedding_openai_library_multiple():
    global server
    server.pooling = 'last'
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.embeddings.create(model="text-embedding-3-small", input=[
        "I believe the meaning of life is",
        "Write a joke about AI from a very long prompt which will not be truncated",
        "This is a test",
        "This is another test",
    ])
    assert len(res.data) == 4
    for d in res.data:
        assert len(d.embedding) > 1


def test_embedding_error_prompt_too_long():
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "This is a test " * 512,
    })
    assert res.status_code != 200
    assert "too large" in res.body["error"]["message"]


def test_same_prompt_give_same_result():
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 5
    for i in range(1, len(res.body['data'])):
        v0 = res.body['data'][0]['embedding']
        vi = res.body['data'][i]['embedding']
        for x, y in zip(v0, vi):
            assert abs(x - y) < EPSILON


@pytest.mark.parametrize(
    "content,n_tokens",
    [
        ("I believe the meaning of life is", 9),
        ("This is a test", 6),
    ]
)
def test_embedding_usage_single(content, n_tokens):
    global server
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": content})
    assert res.status_code == 200
    assert res.body['usage']['prompt_tokens'] == res.body['usage']['total_tokens']
    assert res.body['usage']['prompt_tokens'] == n_tokens


def test_embedding_usage_multiple():
    global server
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "I believe the meaning of life is",
        ],
    })
    assert res.status_code == 200
    assert res.body['usage']['prompt_tokens'] == res.body['usage']['total_tokens']
    assert res.body['usage']['prompt_tokens'] == 2 * 9


def test_embedding_openai_library_base64():
    server.start()
    test_input = "Test base64 embedding output"

    # get embedding in default format
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": test_input
    })
    assert res.status_code == 200
    vec0 = res.body["data"][0]["embedding"]

    # get embedding in base64 format
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": test_input,
        "encoding_format": "base64"
    })

    assert res.status_code == 200
    assert "data" in res.body
    assert len(res.body["data"]) == 1

    embedding_data = res.body["data"][0]
    assert "embedding" in embedding_data
    assert isinstance(embedding_data["embedding"], str)

    # Verify embedding is valid base64
    decoded = base64.b64decode(embedding_data["embedding"])
    # Verify decoded data can be converted back to float array
    float_count = len(decoded) // 4  # 4 bytes per float
    floats = struct.unpack(f'{float_count}f', decoded)
    assert len(floats) > 0
    assert all(isinstance(x, float) for x in floats)
    assert len(floats) == len(vec0)

    # make sure the decoded data is the same as the original
    for x, y in zip(floats, vec0):
        assert abs(x - y) < EPSILON
