import pytest
from utils import *

server = ServerPreset.jina_reranker_tiny()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.jina_reranker_tiny()


TEST_DOCUMENTS = [
    "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
    "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
    "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
    "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine."
]


def test_rerank():
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "documents": TEST_DOCUMENTS,
    })
    assert res.status_code == 200
    assert len(res.body["results"]) == 4

    most_relevant = res.body["results"][0]
    least_relevant = res.body["results"][0]
    for doc in res.body["results"]:
        if doc["relevance_score"] > most_relevant["relevance_score"]:
            most_relevant = doc
        if doc["relevance_score"] < least_relevant["relevance_score"]:
            least_relevant = doc

    assert most_relevant["relevance_score"] > least_relevant["relevance_score"]
    assert most_relevant["index"] == 2
    assert least_relevant["index"] == 3


@pytest.mark.parametrize("kv_unified", [False, True])
def test_rerank_does_not_save_idle_slot_to_prompt_cache(kv_unified, tmp_path):
    global server
    server.n_slots = 2
    server.cache_ram = 100
    server.kv_unified = kv_unified
    server.server_slots = True
    server.debug = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    res = server.make_request("POST", "/rerank", data={
        "query": "first query",
        "documents": ["first document", "second document"],
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/rerank", data={
        "query": "second query",
        "documents": ["third document with a different length"],
    })
    assert res.status_code == 200

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    n_slots_with_prompt = sum(slot.get("n_prompt_tokens", 0) > 0 for slot in res.body)
    assert n_slots_with_prompt == (1 if kv_unified else 2)

    server.stop()

    with open(server.log_path) as log_file:
        log = log_file.read()

    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log


def test_rerank_tei_format():
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "texts": TEST_DOCUMENTS,
    })
    assert res.status_code == 200
    assert len(res.body) == 4

    most_relevant = res.body[0]
    least_relevant = res.body[0]
    for doc in res.body:
        if doc["score"] > most_relevant["score"]:
            most_relevant = doc
        if doc["score"] < least_relevant["score"]:
            least_relevant = doc

    assert most_relevant["score"] > least_relevant["score"]
    assert most_relevant["index"] == 2
    assert least_relevant["index"] == 3


@pytest.mark.parametrize("documents", [
    [],
    None,
    123,
    [1, 2, 3],
])
def test_invalid_rerank_req(documents):
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "documents": documents,
    })
    assert res.status_code == 400
    assert "error" in res.body


@pytest.mark.parametrize(
    "query,doc1,doc2,n_tokens",
    [
        ("Machine learning is", "A machine", "Learning is", 19),
        ("Which city?", "Machine learning is ", "Paris, capitale de la", 26),
    ]
)
def test_rerank_usage(query, doc1, doc2, n_tokens):
    global server
    server.start()

    res = server.make_request("POST", "/rerank", data={
        "query": query,
        "documents": [
            doc1,
            doc2,
        ]
    })
    assert res.status_code == 200
    assert res.body['usage']['prompt_tokens'] == res.body['usage']['total_tokens']
    assert res.body['usage']['prompt_tokens'] == n_tokens


@pytest.mark.parametrize("top_n,expected_len", [
    (None, len(TEST_DOCUMENTS)),  # no top_n parameter
    (2, 2),
    (4, 4),
    (99, len(TEST_DOCUMENTS)),    # higher than available docs
])
def test_rerank_top_n(top_n, expected_len):
    global server
    server.start()
    data = {
        "query": "Machine learning is",
        "documents": TEST_DOCUMENTS,
    }
    if top_n is not None:
        data["top_n"] = top_n

    res = server.make_request("POST", "/rerank", data=data)
    assert res.status_code == 200
    assert len(res.body["results"]) == expected_len


@pytest.mark.parametrize("top_n,expected_len", [
    (None, len(TEST_DOCUMENTS)),  # no top_n parameter
    (2, 2),
    (4, 4),
    (99, len(TEST_DOCUMENTS)),    # higher than available docs
])
def test_rerank_tei_top_n(top_n, expected_len):
    global server
    server.start()
    data = {
        "query": "Machine learning is",
        "texts": TEST_DOCUMENTS,
    }
    if top_n is not None:
        data["top_n"] = top_n

    res = server.make_request("POST", "/rerank", data=data)
    assert res.status_code == 200
    assert len(res.body) == expected_len
