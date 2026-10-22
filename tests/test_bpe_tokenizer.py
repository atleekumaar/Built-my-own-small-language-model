"""
Unit tests for BPETokenizer (Byte-Level Byte-Pair Encoding).
"""

import tempfile
from pathlib import Path
import pytest
from slm.bpe_tokenizer import BPETokenizer

CORPUS = [
    "Abeni lived in a vibrant neighborhood with colorful beads in Lagos.",
    "Jide was hungry so she went looking for food and thirst quenchers.",
    "The savanna climate of Nigeria features dry seasons and lush wetlands.",
    "Afrobeat music blends traditional Yoruba rhythms with jazz and funk.",
]


@pytest.fixture
def trained_bpe():
    tok = BPETokenizer()
    tok.train_from_iterator(CORPUS, vocab_size=300, min_frequency=1)
    return tok


def test_special_tokens(trained_bpe):
    assert trained_bpe.pad_token_id == 0
    assert trained_bpe.unk_token_id == 1
    assert trained_bpe.bos_token_id == 2
    assert trained_bpe.eos_token_id == 3
    assert trained_bpe.vocab_size >= 256  # Includes byte alphabet


def test_encode_decode_roundtrip(trained_bpe):
    text = "Abeni visited Lagos with colorful beads!"
    ids = trained_bpe.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0

    decoded = trained_bpe.decode(ids)
    assert decoded == text


def test_unicode_and_unseen_words(trained_bpe):
    # Unseen words and special Unicode characters
    text = "QuantumCyberX 🚀 ₦5,000 café"
    ids = trained_bpe.encode(text)
    assert len(ids) > 0
    # No token should crash or be dropped
    decoded = trained_bpe.decode(ids)
    assert decoded == text


def test_punctuation_handling(trained_bpe):
    text = "Hello, world! How's it going? (Great/Fine)."
    ids = trained_bpe.encode(text)
    decoded = trained_bpe.decode(ids)
    assert decoded == text


def test_save_and_load(trained_bpe):
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_path = Path(tmpdir) / "bpe_tokenizer.json"
        vocab_path = Path(tmpdir) / "vocab_bpe.json"

        trained_bpe.save(tok_path)
        trained_bpe.save_vocabulary(vocab_path)

        assert tok_path.exists()
        assert vocab_path.exists()

        loaded = BPETokenizer.from_file(tok_path)
        assert loaded.vocab_size == trained_bpe.vocab_size

        text = "Abeni went to Lagos."
        assert loaded.encode(text) == trained_bpe.encode(text)
        assert loaded.decode(loaded.encode(text)) == text


def test_invalid_inputs(trained_bpe):
    with pytest.raises(TypeError):
        trained_bpe.encode(123)

    with pytest.raises(TypeError):
        trained_bpe.decode("not a list")

    assert trained_bpe.encode("") == []
    assert trained_bpe.decode([]) == ""
