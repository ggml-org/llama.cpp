import pytest
from utils import *

server = ServerPreset.jina_reranker_tiny()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.jina_reranker_tiny()


def test_rerank():
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "documents": [
            "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
            "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
            "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
            "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine."
        ]
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
