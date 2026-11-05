import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_tokenize_detokenize():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content
    })
    assert res_tok.status_code == 200
    assert len(res_tok.body["tokens"]) > 5
    # detokenize
    res_detok = server.make_request("POST", "/detokenize", data={
        "tokens": res_tok.body["tokens"],
    })
    assert res_detok.status_code == 200
    assert res_detok.body["content"].strip() == content


def test_tokenize_with_bos():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    bosId = 1
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "add_special": True,
    })
    assert res_tok.status_code == 200
    assert res_tok.body["tokens"][0] == bosId


def test_tokenize_with_pieces():
    global server
    server.start()
    # tokenize
    content = "This is a test string with unicode 媽 and emoji 🤗"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "with_pieces": True,
    })
    assert res_tok.status_code == 200
    for token in res_tok.body["tokens"]:
        assert "id" in token
        assert token["id"] > 0
        assert "piece" in token
        assert len(token["piece"]) > 0


def test_tokenize_array_not_batched():
    global server
    server.start()
    # without batch, an array content is merged into a single token array
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": ["Hello", "world"],
    })
    assert res_tok.status_code == 200
    assert all(isinstance(t, int) for t in res_tok.body["tokens"])


def test_tokenize_batch():
    global server
    server.start()
    texts = ["What is the capital of France ?", "Hello world"]
    res_batch = server.make_request("POST", "/tokenize", data={
        "content": texts,
        "batch": True,
    })
    assert res_batch.status_code == 200
    tokens = res_batch.body["tokens"]
    assert len(tokens) == len(texts)
    for seq in tokens:
        assert isinstance(seq, list) and len(seq) > 0
        assert all(isinstance(t, int) for t in seq)
    # each sequence matches tokenizing it on its own
    for text, seq in zip(texts, tokens):
        res_single = server.make_request("POST", "/tokenize", data={
            "content": text,
        })
        assert res_single.status_code == 200
        assert res_single.body["tokens"] == seq


def test_tokenize_batch_with_bos():
    global server
    server.start()
    bosId = 1
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": ["Hello", "world"],
        "add_special": True,
        "batch": True,
    })
    assert res_tok.status_code == 200
    for seq in res_tok.body["tokens"]:
        assert seq[0] == bosId


def test_tokenize_batch_with_pieces():
    global server
    server.start()
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": ["Hello", "world"],
        "batch": True,
        "with_pieces": True,
    })
    assert res_tok.status_code == 200
    tokens = res_tok.body["tokens"]
    assert len(tokens) == 2
    for seq in tokens:
        for token in seq:
            assert "id" in token and "piece" in token


def test_tokenize_batch_invalid_content():
    global server
    server.start()
    # batch requires an array
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": "Hello",
        "batch": True,
    })
    assert res_tok.status_code == 400
    # and each element must be a string
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": ["Hello", 123],
        "batch": True,
    })
    assert res_tok.status_code == 400
