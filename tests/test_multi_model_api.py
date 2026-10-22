"""
Unit and integration tests for multi-model serving, BPE integration, and experiment tracking.
"""

import pytest
from fastapi.testclient import TestClient

from slm.api import app, initialize_server_state


@pytest.fixture(scope="module")
def client():
    initialize_server_state()
    with TestClient(app) as test_client:
        yield test_client


def test_list_models_endpoint(client: TestClient):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "active_model_id" in data
    model_ids = [m["id"] for m in data["models"]]
    assert "slm_africa_galore" in model_ids
    assert "slm_baseline_val" in model_ids
    assert "slm_bpe_val" in model_ids


def test_switch_to_bpe_model(client: TestClient):
    # Switch to BPE model
    resp = client.post("/models/select", json={"model_id": "slm_bpe_val"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["active_model_id"] == "slm_bpe_val"
    assert data["tokenizer_type"] == "bpe"
    assert data["vocabulary_size"] == 2000

    # Verify /health reflects BPE model
    health = client.get("/health").json()
    assert health["active_model_id"] == "slm_bpe_val"
    assert health["tokenizer_type"] == "bpe"
    assert health["vocabulary_size"] == 2000

    # Verify BPE Tokenization on unseen words has 0 unknown tokens
    tok_resp = client.post("/tokenize", json={"text": "SuperQuantum12345"})
    assert tok_resp.status_code == 200
    tok_data = tok_resp.json()
    assert tok_data["unknown_token_count"] == 0
    assert tok_data["has_unknown_tokens"] is False
    assert len(tok_data["tokens"]) > 0

    # Verify generation works on BPE model
    gen_resp = client.post("/generate", json={"prompt": "Abeni,", "max_new_tokens": 5, "sampling_mode": "greedy"})
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()
    assert gen_data["active_model_id"] == "slm_bpe_val"
    assert len(gen_data["generated_token_ids"]) == 5
    assert len(gen_data["step_top_candidates"]) == 5


def test_switch_back_to_original_model(client: TestClient):
    resp = client.post("/models/select", json={"model_id": "slm_africa_galore"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_model_id"] == "slm_africa_galore"
    assert data["tokenizer_type"] == "word"
    assert data["vocabulary_size"] == 5262


def test_experiments_endpoint(client: TestClient):
    resp = client.get("/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert "experiment_a_word_baseline" in data
    assert "experiment_b_bpe_subword" in data
    assert "summary" in data
    assert data["summary"]["param_reduction_pct"] > 30.0
