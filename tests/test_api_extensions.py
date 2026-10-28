"""Tests for new Phase 3 & 4 API endpoints:

- POST /tokenize/compare (dual-tokenizer side-by-side analysis)
- POST /next-token (real model next-token logit distribution)
- GET /evaluation/export (JSON and CSV benchmark downloads)
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

from fastapi.testclient import TestClient
import pytest

from slm.api import app

client = TestClient(app)


def test_tokenize_compare_endpoint():
    payload = {"text": "Abeni visited Lagos to buy colorful beads and fabric."}
    response = client.post("/tokenize/compare", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert "word_tokenizer" in data
    assert "bpe_tokenizer" in data
    assert data["raw_bytes"] == len(payload["text"].encode("utf-8"))

    # Word breakdown assertions
    w = data["word_tokenizer"]
    assert w["tokenizer_type"] == "word"
    assert w["token_count"] > 0
    assert w["vocabulary_size"] > 4000
    assert len(w["tokens"]) == len(w["token_ids"])
    assert "bytes_per_token" in w

    # BPE breakdown assertions
    b = data["bpe_tokenizer"]
    assert b["tokenizer_type"] == "bpe"
    assert b["token_count"] > 0
    assert b["vocabulary_size"] == 2000
    assert b["unknown_token_count"] == 0  # Zero UNK guarantee
    assert len(b["tokens"]) == len(b["token_ids"])

    # Difference and warning assertions
    assert "token_count_difference" in data
    assert "comparability_warning" in data
    assert "Per-token perplexity" in data["comparability_warning"]


def test_next_token_endpoint_success():
    payload = {
        "text": "Abeni went to the",
        "temperature": 1.0,
        "top_k": 5,
    }
    response = client.post("/next-token", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["context"] == payload["text"]
    assert len(data["top_candidates"]) == 5
    assert data["temperature"] == 1.0

    # Probabilities should sum close to cumulative
    candidates = data["top_candidates"]
    for i, c in enumerate(candidates, 1):
        assert c["rank"] == i
        assert isinstance(c["token"], str)
        assert isinstance(c["token_id"], int)
        assert isinstance(c["raw_logit"], float)
        assert 0.0 <= c["probability"] <= 1.0

    # Highest prob should be rank 1
    assert candidates[0]["probability"] >= candidates[1]["probability"]


def test_next_token_temperature_scaling():
    """Lower temperature (0.1) should peak/sharpen probabilities compared to high temperature (2.0)."""
    text = "The lion roamed the"
    res_cold = client.post("/next-token", json={"text": text, "temperature": 0.1, "top_k": 5}).json()
    res_hot = client.post("/next-token", json={"text": text, "temperature": 2.0, "top_k": 5}).json()

    # In cold temperature, top-1 probability should be much higher than in hot temperature
    p1_cold = res_cold["top_candidates"][0]["probability"]
    p1_hot = res_hot["top_candidates"][0]["probability"]
    assert p1_cold > p1_hot, f"Cold p1 ({p1_cold}) should be higher than hot p1 ({p1_hot})"


def test_next_token_empty_input():
    response = client.post("/next-token", json={"text": "", "temperature": 1.0, "top_k": 5})
    assert response.status_code == 422  # Pydantic validation error for min_length=1


def test_evaluation_export_json():
    response = client.get("/evaluation/export?format=json")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    assert "attachment; filename=slm_evaluation_export.json" in response.headers["content-disposition"]
    data = response.json()
    assert "scientific_validation" in data
    assert "cpu_inference_benchmarks" in data
    assert "sliding_window_bpb" in data


def test_evaluation_export_csv():
    response = client.get("/evaluation/export?format=csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment; filename=slm_evaluation_benchmarks.csv" in response.headers["content-disposition"]
    text = response.text
    lines = text.strip().split("\n")
    assert len(lines) == 4  # Header + 3 models
    header = lines[0]
    assert "model_id" in header
    assert "sliding_window_raw_bpb" in header
    assert "slm_bpe_val" in text
    assert "slm_baseline_val" in text
