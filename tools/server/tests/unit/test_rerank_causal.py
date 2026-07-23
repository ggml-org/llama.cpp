import pytest
from utils import *

TEST_DOCUMENTS = [
    "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
    "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
    "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
    "Paris, capitale de la France, est une grande ville europeenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture.",
]


# These tests verify that the /v1/rerank endpoint correctly detects causal-LM
# reranker models (those without a classification head) and routes them through
# logit-margin scoring instead of the pooling path.
#
# To run locally with a causal-LM reranker GGUF:
#   LLAMA_SERVER_BIN_PATH=./build/bin/llama-server \
#   LLAMA_MODEL_PATH=path/to/causal-reranker.gguf \
#   python -m pytest tools/server/tests/unit/test_rerank_causal.py -v


def create_causal_reranker_server():
    srv = ServerProcess()
    srv.server_reranking = True
    model_path = os.environ.get("LLAMA_MODEL_PATH", "")
    if model_path:
        srv.model_path = model_path
    else:
        pytest.skip("LLAMA_MODEL_PATH not set (need a causal-LM reranker GGUF)")
    return srv


@pytest.mark.skipif(
    not os.environ.get("LLAMA_MODEL_PATH"),
    reason="Requires LLAMA_MODEL_PATH pointing to a causal-LM reranker GGUF",
)
class TestCausalRerank:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.server = create_causal_reranker_server()

    def test_causal_rerank_scores_discriminate(self):
        """Causal-LM reranker must produce discriminating scores, not near-zero garbage."""
        self.server.start()
        res = self.server.make_request(
            "POST",
            "/v1/rerank",
            data={
                "query": "Machine learning is",
                "documents": TEST_DOCUMENTS,
            },
        )
        assert res.status_code == 200
        results = res.body["results"]
        assert len(results) == len(TEST_DOCUMENTS)

        scores = {r["index"]: r["relevance_score"] for r in results}

        # Doc 0 (machine learning) must score highest
        assert scores[0] > scores[1], "relevant doc should outscore related doc"
        assert scores[0] > scores[3], (
            "relevant doc should outscore foreign-language doc"
        )

        # Scores must be in [0, 1] (calibrated sigmoid output)
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0, f"doc {idx} score {score} out of [0,1] range"

        # Scores must not be near-zero garbage (the pre-fix failure mode)
        assert scores[0] > 0.5, (
            f"relevant doc score {scores[0]} too low (pre-fix garbage?)"
        )

        # Spread between best and worst must be meaningful
        spread = scores[0] - min(scores.values())
        assert spread > 0.1, f"score spread {spread} too small for discrimination"

    def test_causal_rerank_top_n(self):
        """top_n parameter must limit results for causal-LM rerankers."""
        self.server.start()
        res = self.server.make_request(
            "POST",
            "/v1/rerank",
            data={
                "query": "Machine learning is",
                "documents": TEST_DOCUMENTS,
                "top_n": 2,
            },
        )
        assert res.status_code == 200
        assert len(res.body["results"]) == 2

    def test_causal_rerank_tei_format(self):
        """TEI format (texts instead of documents) must work for causal-LM rerankers."""
        self.server.start()
        res = self.server.make_request(
            "POST",
            "/v1/rerank",
            data={
                "query": "Machine learning is",
                "texts": TEST_DOCUMENTS,
            },
        )
        assert res.status_code == 200
        assert len(res.body) == len(TEST_DOCUMENTS)

        scores = {r["index"]: r["score"] for r in res.body}
        assert scores[0] > scores[3], (
            "relevant doc should outscore foreign-language doc (TEI format)"
        )

    def test_causal_rerank_single_document(self):
        """Single-document rerank must work without errors."""
        self.server.start()
        res = self.server.make_request(
            "POST",
            "/v1/rerank",
            data={
                "query": "Machine learning is",
                "documents": [TEST_DOCUMENTS[0]],
            },
        )
        assert res.status_code == 200
        assert len(res.body["results"]) == 1
        assert 0.0 <= res.body["results"][0]["relevance_score"] <= 1.0


# Verify the existing pooling-based reranker still works unchanged
class TestPoolingRerankUnchanged:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.server = ServerPreset.jina_reranker_tiny()

    def test_pooling_rerank_still_works(self):
        """BERT-style rerankers must still use the pooling path and produce correct results."""
        self.server.start()
        res = self.server.make_request(
            "POST",
            "/rerank",
            data={
                "query": "Machine learning is",
                "documents": [
                    "A machine is a physical system that uses power to apply forces.",
                    "Machine learning is a field of study in artificial intelligence.",
                    "Paris is the capital of France.",
                ],
            },
        )
        assert res.status_code == 200
        results = res.body["results"]
        assert len(results) == 3

        scores = {r["index"]: r["relevance_score"] for r in results}
        assert scores[1] > scores[2], "ML doc should outscore Paris doc"
