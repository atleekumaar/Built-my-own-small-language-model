"""Unit tests for dataset splitting reproducibility, leakage prevention, and model integrity."""

import json
import os
from pathlib import Path
import pytest

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import keras


from slm.bpe_tokenizer import BPETokenizer
from slm.config import ModelConfig
from slm.generate import generate_text
from slm.losses import CustomMaskPadLoss
from slm.model import build_model, load_checkpoint, save_checkpoint
from slm.tokenizer import SimpleWordTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dataset_split_reproducibility():
    """Verify that train/val splitting with seed=42 produces identical, disjoint subsets."""
    from scripts.create_splits import create_and_save_splits

    split1 = create_and_save_splits(seed=42, train_ratio=0.8)
    split2 = create_and_save_splits(seed=42, train_ratio=0.8)

    assert split1["train_indices"] == split2["train_indices"]
    assert split1["val_indices"] == split2["val_indices"]
    assert len(split1["train_indices"]) == 185
    assert len(split1["val_indices"]) == 47

    # Strict disjointness
    set_train = set(split1["train_indices"])
    set_val = set(split1["val_indices"])
    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train) + len(set_val) == 232


def test_no_vocabulary_leakage():
    """Verify that vocabulary fitted on train split does not leak validation-exclusive words."""
    data_path = REPO_ROOT / "data" / "africa_galore.json"
    splits_path = REPO_ROOT / "data" / "dataset_splits.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_paras = [d["description"] for d in data]

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_paras = [all_paras[i] for i in splits["train_indices"]]
    val_paras = [all_paras[i] for i in splits["val_indices"]]

    # Fit word tokenizer strictly on train
    word_tok = SimpleWordTokenizer(corpus=train_paras)

    # Find words unique to validation set
    val_words = set()
    for p in val_paras:
        val_words.update(SimpleWordTokenizer.space_tokenize(p))

    train_words = set(word_tok.vocabulary)
    unseen_words = val_words - train_words
    assert len(unseen_words) > 0, "Expected some words in validation set to be out-of-vocabulary"

    # All unseen words must map to unknown_token_id (<UNK>)
    for w in unseen_words:
        enc = word_tok.encode(w)
        assert enc == [word_tok.unknown_token_id]

    # Verify BPE tokenizer handles unseen words with zero <UNK>
    bpe_tok = BPETokenizer()
    bpe_tok.train_from_iterator(train_paras, vocab_size=2000, min_frequency=2)
    for w in unseen_words:
        enc_bpe = bpe_tok.encode(w)
        assert len(enc_bpe) > 0
        assert bpe_tok.unk_token_id not in enc_bpe, f"BPE should not produce UNK for {w}"


def test_autoregressive_input_target_shifting():
    """Verify that autoregressive sequence preparation shifts by exactly one token."""
    synthetic_tokens = [
        [10, 20, 30, 40, 50],
        [1, 2, 3, 4, 5],
    ]
    padded = keras.preprocessing.sequence.pad_sequences(
        synthetic_tokens,
        maxlen=5,
        padding="post",
        truncating="post",
        value=0,
    )

    inputs = padded[:, :-1]
    targets = padded[:, 1:]

    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)

    # Target at t is input at t+1
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])

    # First timestep input is 10 and 1; first timestep target is 20 and 2
    assert inputs[0, 0] == 10
    assert targets[0, 0] == 20
    assert inputs[1, 0] == 1
    assert targets[1, 0] == 2


def test_padding_mask_loss_behavior():
    """Verify that padding tokens (pad_token_id=0) are completely masked in CustomMaskPadLoss."""
    loss_fn = CustomMaskPadLoss(pad_token_id=0)

    # Target with 2 real tokens and 2 pad tokens
    y_true_padded = np.array([[1, 2, 0, 0]], dtype=np.int32)
    y_pred_padded = np.zeros((1, 4, 10), dtype=np.float32)
    y_pred_padded[0, 0, 1] = 5.0
    y_pred_padded[0, 1, 2] = 5.0
    # Arbitrary predictions on pad positions
    y_pred_padded[0, 2, 3] = 999.0
    y_pred_padded[0, 3, 4] = -999.0

    loss_padded = float(loss_fn(y_true_padded, y_pred_padded))

    # Same target with different predictions on pad positions
    y_pred_padded_diff = np.copy(y_pred_padded)
    y_pred_padded_diff[0, 2, :] = 0.0
    y_pred_padded_diff[0, 3, :] = 0.0

    loss_diff = float(loss_fn(y_true_padded, y_pred_padded_diff))

    # Loss must be strictly identical because pad positions are ignored
    np.testing.assert_almost_equal(loss_padded, loss_diff, decimal=5)


def test_tokenizer_encode_decode_consistency():
    """Verify bidirectional consistency across both word and subword tokenizers."""
    text_samples = [
        "Abeni visited the market in Lagos.",
        "Table Mountain overlooks Cape Town with scenic beauty.",
        "Testing numbers 12345 and punctuation: (hello-world)!",
    ]

    word_tok = SimpleWordTokenizer(corpus=text_samples)
    bpe_tok = BPETokenizer()
    bpe_tok.train_from_iterator(text_samples, vocab_size=200, min_frequency=1)

    for text in text_samples:
        # Word tokenizer round-trip (whitespace-delimited)
        enc_w = word_tok.encode(text)
        dec_w = word_tok.decode(enc_w)
        # Decoded words should match space-tokenized input joined by space
        expected_w = " ".join(SimpleWordTokenizer.space_tokenize(text))
        assert dec_w == expected_w

        # BPE tokenizer round-trip (exact byte reconstruction)
        enc_b = bpe_tok.encode(text)
        dec_b = bpe_tok.decode(enc_b)
        assert dec_b == text


def test_checkpoint_save_and_reload_fresh_process(tmp_path: Path):
    """Verify that saved weights reloaded in a fresh model instance yield identical outputs."""
    config = ModelConfig(
        vocabulary_size=64,
        max_length=16,
        embedding_dim=32,
        mlp_dim=32,
        num_heads=2,
        num_blocks=1,
    )
    model1 = build_model(config)
    dummy_input = np.random.randint(0, 64, size=(2, 16), dtype=np.int32)
    output1 = np.array(model1(dummy_input))

    ckpt_path = tmp_path / "fresh_test.weights.h5"
    save_checkpoint(model1, ckpt_path)

    # Fresh instance
    model2 = build_model(config)
    load_checkpoint(model2, ckpt_path)
    output2 = np.array(model2(dummy_input))

    np.testing.assert_allclose(output1, output2, atol=1e-5)


def test_deterministic_generation_fixed_seed():
    """Verify that sampling with fixed random seed generates identical token sequences."""
    config = ModelConfig(
        vocabulary_size=50,
        max_length=20,
        embedding_dim=32,
        mlp_dim=32,
        num_heads=2,
        num_blocks=1,
    )
    model = build_model(config)
    tok = SimpleWordTokenizer(corpus=["word " + str(i) for i in range(48)])

    res_a = generate_text(
        prompt="word",
        n_tokens=8,
        model=model,
        tokenizer=tok,
        sampling_mode="random",
        temperature=0.8,
        seed=1234,
    )

    res_b = generate_text(
        prompt="word",
        n_tokens=8,
        model=model,
        tokenizer=tok,
        sampling_mode="random",
        temperature=0.8,
        seed=1234,
    )

    assert res_a.generated_token_ids == res_b.generated_token_ids
    assert res_a.generated_text == res_b.generated_text


def test_consistent_evaluation_documents_across_models():
    """Verify that Word Baseline and BPE models are evaluated on the exact same 47 documents."""
    splits_path = REPO_ROOT / "data" / "dataset_splits.json"
    data_path = REPO_ROOT / "data" / "africa_galore.json"

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    val_indices = splits["val_indices"]
    assert len(val_indices) == 47

    val_paragraphs = [data[i]["description"] for i in val_indices]
    total_val_bytes = sum(len(p.encode("utf-8")) for p in val_paragraphs)

    # Must match the audited byte count
    assert total_val_bytes == 26159


def test_missing_artifact_and_invalid_config_handling(tmp_path: Path):
    """Verify clear error handling when required files or configurations are missing."""
    config = ModelConfig(vocabulary_size=20, max_length=10, embedding_dim=16)
    model = build_model(config)

    # Missing weights file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint(model, tmp_path / "non_existent.weights.h5")

    # Missing tokenizer file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
        BPETokenizer.from_file(tmp_path / "non_existent_tok.json")

    # Unknown model in API load raises ValueError
    from slm.api import load_model_by_id
    with pytest.raises(ValueError, match="Unknown model_id"):
        load_model_by_id("invalid_model_id_123")
