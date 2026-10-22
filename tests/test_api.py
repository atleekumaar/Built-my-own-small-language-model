"""Unit tests for the FastAPI SLM inference and tokenization API."""

import pytest
from fastapi.testclient import TestClient

from slm.api import app, initialize_server_state


@pytest.fixture(scope="module")
def client():
    # Initialize state before running API tests
    initialize_server_state()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200 and expected metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["total_parameters"] == 3358094
    assert data["vocabulary_size"] == 5262
    assert data["max_length"] == 299


def test_config_endpoint(client: TestClient):
    """Verify /config returns verified architecture parameters."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "decoder_only_transformer"
    assert data["embedding_dim"] == 256
    assert data["num_heads"] == 2
    assert data["num_blocks"] == 1


def test_tokenize_endpoint(client: TestClient):
    """Verify /tokenize returns tokens, token IDs, and unknown token detection."""
    payload = {"text": "Abeni lived in Lagos"}
    response = client.post("/tokenize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tokens"] == ["Abeni", "lived", "in", "Lagos"]
    assert len(data["token_ids"]) == 4
    assert data["unknown_token_count"] == 0
    assert data["has_unknown_tokens"] is False

    # Text with unknown word
    payload_unk = {"text": "SuperUnknownToken12345"}
    response_unk = client.post("/tokenize", json=payload_unk)
    assert response_unk.status_code == 200
    data_unk = response_unk.json()
    assert data_unk["has_unknown_tokens"] is True
    assert data_unk["unknown_token_count"] == 1


def test_generate_endpoint_success(client: TestClient):
    """Verify /generate returns generated text, candidates, and metadata."""
    payload = {
        "prompt": "Abeni,",
        "max_new_tokens": 5,
        "sampling_mode": "greedy",
        "top_k_probabilities": 3,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == "Abeni,"
    assert isinstance(data["generated_text"], str)
    assert isinstance(data["full_text"], str)
    assert len(data["generated_token_ids"]) == 5
    assert len(data["step_top_candidates"]) == 5

    # Check top candidate format
    first_step_top3 = data["step_top_candidates"][0]
    assert len(first_step_top3) == 3
    for candidate in first_step_top3:
        assert "token" in candidate
        assert "token_id" in candidate
        assert "probability" in candidate
        assert 0.0 <= candidate["probability"] <= 1.0


def test_generate_endpoint_validation_errors(client: TestClient):
    """Verify FastAPI Pydantic schema validation rejects invalid inputs."""
    # Empty prompt
    resp_empty = client.post("/generate", json={"prompt": "", "max_new_tokens": 5})
    assert resp_empty.status_code == 422

    # Negative tokens
    resp_neg = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": -1})
    assert resp_neg.status_code == 422

    # Too many tokens (> 100)
    resp_large = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": 500})
    assert resp_large.status_code == 422

    # Invalid sampling mode
    resp_mode = client.post("/generate", json={"prompt": "Hello", "sampling_mode": "invalid"})
    assert resp_mode.status_code == 422
