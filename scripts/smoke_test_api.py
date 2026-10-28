"""Comprehensive smoke test for ATLEE SLM LAB 2.0 API & Frontend Serving.

Executes an end-to-end smoke test through the FastAPI TestClient:
1. GET / (Frontend Single-Page App delivery)
2. GET /health (Health status, device, checkpoint status)
3. GET /models (Dynamic model listing)
4. POST /models/select (Model switching between Word and BPE)
5. POST /generate (Autoregressive generation with seed determinism)
6. POST /tokenize (Basic tokenization)
7. POST /tokenize/compare (Side-by-side Word vs BPE decomposition)
8. POST /next-token (Real model logits & Softmax probability distribution)
9. GET /evaluation (Dynamic evaluation with sliding window BPB)
10. GET /evaluation/export (JSON and CSV benchmark downloads)
"""

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

from fastapi.testclient import TestClient
from slm.api import app

client = TestClient(app)


def run_smoke_test():
    print("=" * 70)
    print("ATLEE SLM LAB 2.0 — END-TO-END SMOKE TEST")
    print("=" * 70)

    # 1. Frontend SPA delivery
    res = client.get("/")
    assert res.status_code == 200, f"Frontend delivery failed: {res.status_code}"
    assert "ATLEE SLM LAB" in res.text
    assert "tabPlaygroundBtn" in res.text
    assert "tabXrayBtn" in res.text
    print("[PASS] 1. GET /: Frontend Single-Page App served successfully.")

    # 2. Health check
    res = client.get("/health")
    assert res.status_code == 200
    health = res.json()
    assert health["status"] == "ok"
    assert health["device"] == "cpu"
    assert health["checkpoint_loaded"] is True
    print(f"[PASS] 2. GET /health: Status OK, Active Model: {health['active_model_id']}, Params: {health['total_parameters']:,}.")

    # 3. Models list
    res = client.get("/models")
    assert res.status_code == 200
    models_data = res.json()
    assert len(models_data["models"]) == 3
    print(f"[PASS] 3. GET /models: Retrieved {len(models_data['models'])} model profiles.")

    # 4. Model selection (Switch to BPE)
    res = client.post("/models/select", json={"model_id": "slm_bpe_val"})
    assert res.status_code == 200
    sel = res.json()
    assert sel["active_model_id"] == "slm_bpe_val"
    assert sel["tokenizer_type"] == "bpe"
    assert sel["parameters"] == 1684688
    print(f"[PASS] 4. POST /models/select: Switched to Byte-Level BPE (1.68M parameters).")

    # 5. Autoregressive generation
    gen_payload = {
        "prompt": "Kilimanjaro is",
        "max_new_tokens": 10,
        "sampling_mode": "greedy",
        "top_k_probabilities": 5,
    }
    res = client.post("/generate", json=gen_payload)
    assert res.status_code == 200
    gen = res.json()
    assert gen["prompt"] == "Kilimanjaro is"
    assert len(gen["generated_tokens"]) > 0
    assert len(gen["step_top_candidates"]) > 0
    print(f"[PASS] 5. POST /generate: Generated text: '{gen['generated_text']}' ({gen['latency_ms']} ms).")

    # 6. Basic tokenization
    res = client.post("/tokenize", json={"text": "Abeni visited Lagos"})
    assert res.status_code == 200
    tok = res.json()
    assert tok["token_count"] > 0
    print(f"[PASS] 6. POST /tokenize: Tokenized into {tok['token_count']} subwords.")

    # 7. Dual tokenizer comparison
    comp_payload = {"text": "Abeni lived in a vibrant neighborhood with colorful beads. Lagos was sunny!"}
    res = client.post("/tokenize/compare", json=comp_payload)
    assert res.status_code == 200
    comp = res.json()
    w_cnt = comp["word_tokenizer"]["token_count"]
    b_cnt = comp["bpe_tokenizer"]["token_count"]
    assert w_cnt > 0 and b_cnt > 0
    assert comp["bpe_tokenizer"]["unknown_token_count"] == 0
    print(f"[PASS] 7. POST /tokenize/compare: Word tokens={w_cnt}, BPE tokens={b_cnt}, Ratio={comp['token_ratio']}x.")

    # 8. Next-Token X-Ray
    xray_payload = {"text": "Kilimanjaro is", "temperature": 0.8, "top_k": 5}
    res = client.post("/next-token", json=xray_payload)
    assert res.status_code == 200
    xray = res.json()
    assert len(xray["top_candidates"]) == 5
    top1 = xray["top_candidates"][0]
    print(f"[PASS] 8. POST /next-token: Top Candidate: '{top1['token']}' (Prob: {top1['probability']*100:.1f}%, Logit: {top1['raw_logit']}).")

    # 9. Dynamic evaluation metrics
    res = client.get("/evaluation")
    assert res.status_code == 200
    eval_data = res.json()
    assert eval_data["is_held_out"] is True
    assert eval_data["sliding_window_coverage_pct"] == 100.0
    print(f"[PASS] 9. GET /evaluation: Held-Out Loss: {eval_data['evaluation_loss']}, Sliding-Window BPB: {eval_data['sliding_window_raw_bpb']}.")

    # 10. Evaluation export (JSON and CSV)
    res_json = client.get("/evaluation/export?format=json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers["content-type"]

    res_csv = client.get("/evaluation/export?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert "slm_bpe_val" in res_csv.text
    print(f"[PASS] 10. GET /evaluation/export: Successfully downloaded both JSON and CSV benchmark attachments.")

    # Reset back to course model
    client.post("/models/select", json={"model_id": "slm_africa_galore"})
    print("\n[SUCCESS] ALL 10 SMOKE TEST SUITES PASSED CLEANLY (100% HEALTHY).")


if __name__ == "__main__":
    run_smoke_test()
