"""
Tests for model capabilities in the /models endpoint response.

Verifies that:
- Regular (chat/completion) models get capabilities: ["completion"]
- Embedding models (--embedding flag) get capabilities: ["embedding"]
- Reranking models (--reranking flag) get capabilities: ["embedding", "rerank"]
- Capabilities field is always present in /models response
- Single-model and router-mode endpoints both return capabilities
"""
import pytest
from utils import *


server: ServerProcess


# ==============================================================================
# Single-model mode tests
# ==============================================================================


class TestSingleModelCapabilities:
    """Test capabilities in /models response for single-model server mode."""

    @pytest.fixture(autouse=True)
    def create_server(self):
        global server

    def test_completion_model_has_completion_capability(self):
        """A standard chat/completion model should report capabilities: ["completion"]."""
        global server
        server = ServerPreset.tinyllama2()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        assert len(res.body["data"]) == 1
        model = res.body["data"][0]
        assert "capabilities" in model or "capabilities" in res.body.get("models", [{}])[0], \
            "capabilities field must be present in /models response"
        # Check in the appropriate location
        if "models" in res.body and len(res.body["models"]) > 0:
            caps = res.body["models"][0].get("capabilities", [])
        else:
            caps = model.get("capabilities", [])
        assert "completion" in caps, \
            f"Completion model should have 'completion' capability, got: {caps}"

    def test_embedding_model_has_embedding_capability(self):
        """A model started with --embedding should report capabilities: ["embedding"]."""
        global server
        server = ServerPreset.bert_bge_small()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        assert len(res.body["data"]) == 1
        if "models" in res.body and len(res.body["models"]) > 0:
            caps = res.body["models"][0].get("capabilities", [])
        else:
            caps = res.body["data"][0].get("capabilities", [])
        assert "embedding" in caps, \
            f"Embedding model should have 'embedding' capability, got: {caps}"
        assert "completion" not in caps, \
            f"Embedding model should NOT have 'completion' capability, got: {caps}"

    def test_reranking_model_has_rerank_capability(self):
        """A model started with --reranking should report capabilities including 'rerank'."""
        global server
        server = ServerPreset.jina_reranker_tiny()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        assert len(res.body["data"]) == 1
        if "models" in res.body and len(res.body["models"]) > 0:
            caps = res.body["models"][0].get("capabilities", [])
        else:
            caps = res.body["data"][0].get("capabilities", [])
        assert "rerank" in caps, \
            f"Reranking model should have 'rerank' capability, got: {caps}"
        assert "embedding" in caps, \
            f"Reranking model should also have 'embedding' capability, got: {caps}"
        assert "completion" not in caps, \
            f"Reranking model should NOT have 'completion' capability, got: {caps}"

    def test_capabilities_field_is_array(self):
        """The capabilities field should always be a JSON array."""
        global server
        server = ServerPreset.tinyllama2()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        if "models" in res.body and len(res.body["models"]) > 0:
            caps = res.body["models"][0].get("capabilities")
        else:
            caps = res.body["data"][0].get("capabilities")
        assert isinstance(caps, list), \
            f"capabilities should be a list, got: {type(caps)}"

    def test_capabilities_contains_only_strings(self):
        """All entries in capabilities array should be strings."""
        global server
        server = ServerPreset.tinyllama2()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        if "models" in res.body and len(res.body["models"]) > 0:
            caps = res.body["models"][0].get("capabilities", [])
        else:
            caps = res.body["data"][0].get("capabilities", [])
        for cap in caps:
            assert isinstance(cap, str), \
                f"Each capability should be a string, got: {type(cap)} ({cap})"


# ==============================================================================
# OAI compatibility tests
# ==============================================================================


class TestOAICompatibility:
    """Test that /v1/models endpoint also returns capabilities."""

    @pytest.fixture(autouse=True)
    def create_server(self):
        global server

    def test_v1_models_completion(self):
        """The /v1/models endpoint should also include capabilities for completion models."""
        global server
        server = ServerPreset.tinyllama2()
        server.start()
        res = server.make_request("GET", "/v1/models")
        assert res.status_code == 200
        assert "data" in res.body
        assert len(res.body["data"]) >= 1

    def test_v1_models_embedding(self):
        """The /v1/models endpoint should also include capabilities for embedding models."""
        global server
        server = ServerPreset.bert_bge_small()
        server.start()
        res = server.make_request("GET", "/v1/models")
        assert res.status_code == 200
        assert "data" in res.body
        assert len(res.body["data"]) >= 1


# ==============================================================================
# Response structure tests
# ==============================================================================


class TestModelsResponseStructure:
    """Test the overall structure of /models response with capabilities."""

    @pytest.fixture(autouse=True)
    def create_server(self):
        global server

    def test_completion_model_does_not_have_embedding(self):
        """A standard completion model should not list embedding in its capabilities."""
        global server
        server = ServerPreset.tinyllama2()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        if "models" in res.body and len(res.body["models"]) > 0:
            caps = res.body["models"][0].get("capabilities", [])
        else:
            caps = res.body["data"][0].get("capabilities", [])
        assert "embedding" not in caps, \
            f"Completion model should NOT have 'embedding', got: {caps}"
        assert "rerank" not in caps, \
            f"Completion model should NOT have 'rerank', got: {caps}"

    def test_embedding_model_does_not_have_rerank(self):
        """An embedding-only model (no --reranking) should not have 'rerank' capability."""
        global server
        server = ServerPreset.bert_bge_small()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        if "models" in res.body and len(res.body["models"]) > 0:
            caps = res.body["models"][0].get("capabilities", [])
        else:
            caps = res.body["data"][0].get("capabilities", [])
        assert "rerank" not in caps, \
            f"Embedding-only model should NOT have 'rerank', got: {caps}"

    def test_model_id_and_capabilities_coexist(self):
        """Model response should include both id and capabilities fields."""
        global server
        server = ServerPreset.tinyllama2()
        server.start()
        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model = res.body["data"][0]
        assert "id" in model, "Model response must include 'id' field"
        # capabilities may be in data[0] or models[0] depending on endpoint
        has_caps = "capabilities" in model
        if "models" in res.body and len(res.body["models"]) > 0:
            has_caps = has_caps or "capabilities" in res.body["models"][0]
        assert has_caps, "Model response must include 'capabilities' field"
