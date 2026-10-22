"""Unit tests for SimpleWordTokenizer."""

import json
from pathlib import Path
import pytest

from slm.tokenizer import SimpleWordTokenizer


@pytest.fixture
def sample_corpus() -> list[str]:
    return [
        "The Lagos air was thick with humidity",
        "Abeni lived in a vibrant neighborhood",
        "Table Mountain overlooks the ocean",
    ]


@pytest.fixture
def tokenizer(sample_corpus: list[str]) -> SimpleWordTokenizer:
    return SimpleWordTokenizer(corpus=sample_corpus)


def test_special_tokens_presence(tokenizer: SimpleWordTokenizer):
    """Verify special tokens exist and have expected default indices."""
    assert "<PAD>" in tokenizer.vocabulary
    assert "<UNK>" in tokenizer.vocabulary
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unknown_token_id == tokenizer.vocabulary_size - 1
    assert tokenizer.vocabulary[0] == "<PAD>"
    assert tokenizer.vocabulary[-1] == "<UNK>"


def test_space_tokenize_whitespace_handling():
    """Verify space_tokenize handles varying spaces and whitespace cleanly."""
    tokens = SimpleWordTokenizer.space_tokenize("  Lagos   air   was   thick  ")
    assert tokens == ["Lagos", "air", "was", "thick"]

    empty_tokens = SimpleWordTokenizer.space_tokenize("    ")
    assert empty_tokens == []


def test_encode_known_words(tokenizer: SimpleWordTokenizer):
    """Verify known words map to valid non-UNK indices."""
    text = "Lagos air was thick"
    token_ids = tokenizer.encode(text)
    assert len(token_ids) == 4
    for tid in token_ids:
        assert isinstance(tid, int)
        assert 0 <= tid < tokenizer.vocabulary_size
        assert tid != tokenizer.unknown_token_id
        assert tid != tokenizer.pad_token_id


def test_encode_unknown_words(tokenizer: SimpleWordTokenizer):
    """Verify unseen words map to the <UNK> token index."""
    unseen_text = "UnknownWord123 SuperUnseen"
    token_ids = tokenizer.encode(unseen_text)
    assert token_ids == [tokenizer.unknown_token_id, tokenizer.unknown_token_id]


def test_decode_round_trip(tokenizer: SimpleWordTokenizer):
    """Verify that encoding and decoding known words round-trips exactly."""
    phrase = "Lagos air was thick"
    encoded = tokenizer.encode(phrase)
    decoded = tokenizer.decode(encoded)
    assert decoded == phrase


def test_decode_single_int(tokenizer: SimpleWordTokenizer):
    """Verify decode works on a single integer."""
    token = tokenizer.decode(tokenizer.pad_token_id)
    assert token == "<PAD>"


def test_decode_skip_special(tokenizer: SimpleWordTokenizer):
    """Verify skip_special omits <PAD> tokens."""
    seq = [tokenizer.pad_token_id, tokenizer.encode("Lagos")[0], tokenizer.pad_token_id]
    decoded = tokenizer.decode(seq, skip_special=True)
    assert decoded == "Lagos"


def test_vocabulary_serialization(tokenizer: SimpleWordTokenizer, tmp_path: Path):
    """Verify vocabulary can be saved to and loaded from JSON cleanly."""
    vocab_file = tmp_path / "vocab_test.json"
    tokenizer.save_vocabulary(vocab_file)

    loaded_tokenizer = SimpleWordTokenizer.from_vocabulary_file(vocab_file)
    assert loaded_tokenizer.vocabulary_size == tokenizer.vocabulary_size
    assert loaded_tokenizer.vocabulary == tokenizer.vocabulary
    assert loaded_tokenizer.pad_token_id == tokenizer.pad_token_id
    assert loaded_tokenizer.unknown_token_id == tokenizer.unknown_token_id

    # Verify identical encoding
    test_str = "Abeni lived in a vibrant neighborhood"
    assert loaded_tokenizer.encode(test_str) == tokenizer.encode(test_str)


def test_invalid_input_types(tokenizer: SimpleWordTokenizer):
    """Verify that non-string inputs raise TypeError."""
    with pytest.raises(TypeError):
        tokenizer.encode(12345)  # type: ignore

    with pytest.raises(TypeError):
        tokenizer.space_tokenize(None)  # type: ignore
