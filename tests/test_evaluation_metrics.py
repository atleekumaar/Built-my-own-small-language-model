"""Unit tests for metric calculations, bits-per-byte normalization, and benchmark protocols."""

import math
import os
from pathlib import Path
import pytest

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np


from scripts.train_experiments import compute_per_token_loss, compute_repetition_metrics
from slm.config import ModelConfig
from slm.model import build_model
from slm.tokenizer import SimpleWordTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


class MockModelUniform:
    """Mock model that outputs uniform logits across vocabulary."""
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, x, training=False):
        batch, seq_len = x.shape
        # Logits all 0.0 -> uniform distribution with prob = 1/vocab_size
        return np.zeros((batch, seq_len, self.vocab_size), dtype=np.float32)


class MockModelPerfect:
    """Mock model that places high logit exclusively on target class."""
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, x, training=False):
        batch, seq_len = x.shape
        logits = np.zeros((batch, seq_len, self.vocab_size), dtype=np.float32)
        return logits


def test_cross_entropy_and_perplexity_uniform_distribution():
    """Verify that uniform predictions over V classes yield cross_entropy=ln(V) and perplexity=V."""
    vocab_size = 8
    model = MockModelUniform(vocab_size=vocab_size)

    # 1 sequence, 4 non-pad tokens
    inputs = np.array([[1, 2, 3, 4]], dtype=np.int32)
    targets = np.array([[2, 3, 4, 5]], dtype=np.int32)

    avg_loss, ppl, n_tokens = compute_per_token_loss(
        model=model,
        inputs=inputs,
        targets=targets,
        pad_token_id=0,
    )

    expected_loss = math.log(vocab_size)
    expected_ppl = float(vocab_size)

    assert n_tokens == 4
    np.testing.assert_almost_equal(avg_loss, expected_loss, decimal=4)
    np.testing.assert_almost_equal(ppl, expected_ppl, decimal=2)


def test_cross_entropy_perfect_prediction():
    """Verify that perfect target predictions yield cross-entropy near 0 and perplexity near 1."""
    vocab_size = 5
    batch, seq_len = 1, 3
    inputs = np.array([[1, 2, 3]], dtype=np.int32)
    targets = np.array([[1, 2, 3]], dtype=np.int32)

    class PerfectLogitsModel:
        def __call__(self, x, training=False):
            logits = np.full((batch, seq_len, vocab_size), -100.0, dtype=np.float32)
            for t in range(seq_len):
                logits[0, t, targets[0, t]] = 100.0
            return logits

    model = PerfectLogitsModel()
    avg_loss, ppl, n_tokens = compute_per_token_loss(
        model=model,
        inputs=inputs,
        targets=targets,
        pad_token_id=0,
    )

    assert n_tokens == 3
    assert avg_loss < 1e-4
    np.testing.assert_almost_equal(ppl, 1.0, decimal=2)


def test_bits_per_byte_calculation_toy_example():
    """Verify that Bits Per Byte (BPB) calculation uses consistent byte denominator."""
    from scripts.scientific_audit import evaluate_model_comprehensive

    toy_text = ["Hello world!"]
    raw_bytes = len(toy_text[0].encode("utf-8"))  # 12 UTF-8 bytes
    assert raw_bytes == 12

    word_tok = SimpleWordTokenizer(corpus=toy_text)
    config = ModelConfig(
        vocabulary_size=word_tok.vocab_size,
        max_length=9,  # Effective input length after autoregressive shifting (maxlen=10 - 1)
        embedding_dim=16,
        mlp_dim=16,
        num_heads=1,
        num_blocks=1,
    )
    model = build_model(config)

    metrics = evaluate_model_comprehensive(model, word_tok, toy_text, max_length=10)

    assert metrics["total_bytes"] == 12
    assert metrics["raw_bpb"] > 0.0
    assert metrics["adjusted_bpb"] >= metrics["raw_bpb"]

    # Verify manual formula: total_nats / ln(2) / total_bytes
    expected_bits = (metrics["per_token_ce_nats"] * metrics["total_tokens"]) / math.log(2.0)
    manual_raw_bpb = expected_bits / 12.0
    np.testing.assert_almost_equal(metrics["raw_bpb"], manual_raw_bpb, decimal=3)


def test_open_vocab_adjusted_bpb_formula():
    """Verify that open-vocabulary adjusted BPB properly penalizes UNK tokens."""
    total_nats = 100.0
    total_bytes = 200
    unk_tokens = 5
    extra_bits_per_unk = 15.61

    total_bits = total_nats / math.log(2.0)
    raw_bpb = total_bits / total_bytes
    adjusted_bpb = (total_bits + (unk_tokens * extra_bits_per_unk)) / total_bytes

    assert adjusted_bpb > raw_bpb
    assert np.isclose(adjusted_bpb - raw_bpb, (5 * 15.61) / 200, atol=1e-5)


def test_repetition_metrics():
    """Verify repetition rate and unique bigrams calculation on synthetic texts."""
    # 1. High-diversity text: 11 words -> 10 bigrams
    diverse_text = "The quick brown fox jumps over the lazy dog in autumn"
    res_div = compute_repetition_metrics(diverse_text)
    assert res_div["bigram_repetition_rate"] == 0.0  # all bigrams unique
    assert res_div["unique_bigrams"] == 10


    # 2. Degenerate repeating text
    rep_text = "traditional and traditional and traditional and traditional"
    res_rep = compute_repetition_metrics(rep_text)
    assert res_rep["bigram_repetition_rate"] > 0.5  # significant bigram repetition


def test_benchmark_warmup_protocol():
    """Verify benchmark helper executes warm-up runs and reports steady-state telemetry."""
    from scripts.benchmark_comparison import benchmark_model

    config = ModelConfig(
        vocabulary_size=20,
        max_length=10,
        embedding_dim=16,
        mlp_dim=16,
        num_heads=1,
        num_blocks=1,
    )
    model = build_model(config)
    tok = SimpleWordTokenizer(corpus=["word " + str(i) for i in range(18)])

    metrics = benchmark_model(
        model=model,
        tokenizer=tok,
        model_name="toy_test_model",
        prompts=["word 0", "word 1"],
        tokens_per_prompt=3,
        num_runs=2,
        warmup_runs=1,
    )

    assert "median_latency_ms" in metrics
    assert "p95_latency_ms" in metrics
    assert "throughput_tok_per_sec" in metrics
    assert metrics["total_tokens_generated"] == 2 * 2 * 3  # 2 runs x 2 prompts x 3 tokens = 12
    assert metrics["throughput_tok_per_sec"] > 0.0
