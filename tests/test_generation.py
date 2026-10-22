"""Unit tests for text generation functions and safety validation."""

import pytest
import numpy as np

from slm.config import ModelConfig
from slm.generate import generate_text
from slm.model import build_model
from slm.tokenizer import SimpleWordTokenizer


@pytest.fixture(scope="module")
def setup_generation():
    corpus = [
        "The Lagos air was thick with humidity",
        "Abeni lived in a vibrant neighborhood with colorful beads",
    ]
    tokenizer = SimpleWordTokenizer(corpus=corpus)
    config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        max_length=20,
        embedding_dim=32,
        mlp_dim=32,
        num_heads=2,
        num_blocks=1,
    )
    model = build_model(config)
    return model, tokenizer, config


def test_generation_input_validation(setup_generation):
    """Verify input validation handles invalid parameter values gracefully."""
    model, tokenizer, _ = setup_generation

    # Invalid prompt type
    with pytest.raises(TypeError, match="Prompt must be a string"):
        generate_text(12345, 5, model, tokenizer)  # type: ignore

    # Invalid n_tokens (zero or negative)
    with pytest.raises(ValueError, match="n_tokens must be greater than 0"):
        generate_text("Abeni", 0, model, tokenizer)

    with pytest.raises(ValueError, match="n_tokens must be greater than 0"):
        generate_text("Abeni", -5, model, tokenizer)

    # Invalid sampling mode
    with pytest.raises(ValueError, match="sampling_mode must be 'greedy' or 'random'"):
        generate_text("Abeni", 5, model, tokenizer, sampling_mode="invalid_mode")  # type: ignore

    # Invalid temperature
    with pytest.raises(ValueError, match="temperature must be a positive number"):
        generate_text("Abeni", 5, model, tokenizer, sampling_mode="random", temperature=0.0)


def test_generation_output_validity(setup_generation):
    """Verify generated token IDs, length, and probability shapes."""
    model, tokenizer, config = setup_generation

    n_tokens = 4
    result = generate_text(
        prompt="Abeni lived",
        n_tokens=n_tokens,
        model=model,
        tokenizer=tokenizer,
        sampling_mode="greedy",
    )

    assert len(result.generated_token_ids) == n_tokens
    assert len(result.generated_tokens) == n_tokens
    assert len(result.step_probabilities) == n_tokens
    assert len(result.all_token_ids) == len(result.prompt_token_ids) + n_tokens

    # Verify all generated IDs are within the vocabulary range
    for tid in result.generated_token_ids:
        assert 0 <= tid < config.vocabulary_size

    # Verify each probability vector sums to 1.0
    for probs in result.step_probabilities:
        assert probs.shape == (config.vocabulary_size,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)


def test_generation_determinism_with_seed(setup_generation):
    """Verify reproducibility when providing random seed."""
    model, tokenizer, _ = setup_generation

    res1 = generate_text(
        prompt="Abeni",
        n_tokens=5,
        model=model,
        tokenizer=tokenizer,
        sampling_mode="random",
        temperature=0.8,
        seed=42,
    )

    res2 = generate_text(
        prompt="Abeni",
        n_tokens=5,
        model=model,
        tokenizer=tokenizer,
        sampling_mode="random",
        temperature=0.8,
        seed=42,
    )

    assert res1.generated_token_ids == res2.generated_token_ids
    assert res1.generated_text == res2.generated_text
