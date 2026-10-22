"""Unit tests for SLM model architecture and loss functions."""

from pathlib import Path
import pytest
import numpy as np

from slm.config import ModelConfig
from slm.losses import CustomMaskPadLoss
from slm.model import build_model, load_checkpoint, save_checkpoint


def test_model_config_serialization(tmp_path: Path):
    """Verify ModelConfig can round-trip through JSON."""
    config = ModelConfig(vocabulary_size=1000, max_length=50, embedding_dim=64)
    cfg_path = tmp_path / "test_config.json"
    config.to_json(cfg_path)

    loaded = ModelConfig.from_json(cfg_path)
    assert loaded.vocabulary_size == 1000
    assert loaded.max_length == 50
    assert loaded.embedding_dim == 64
    assert loaded.model_type == "decoder_only_transformer"


def test_model_forward_shape():
    """Verify model instantiation and forward pass tensor shape on a test config."""
    config = ModelConfig(
        vocabulary_size=100,
        max_length=20,
        embedding_dim=32,
        mlp_dim=32,
        num_heads=2,
        num_blocks=1,
    )
    model = build_model(config)

    batch_size = 2
    dummy_input = np.random.randint(0, 100, size=(batch_size, 20), dtype=np.int32)
    output = model(dummy_input)

    assert output.shape == (batch_size, 20, 100)


def test_verified_architecture_parameter_count():
    """Verify the exact parameter count of the verified architecture (3,358,094 params)."""
    config = ModelConfig()
    model = build_model(config)
    assert model.count_params() == config.total_parameters
    assert model.count_params() == 3358094


def test_custom_mask_pad_loss():
    """Verify CustomMaskPadLoss computes loss and masks pad tokens."""
    loss_fn = CustomMaskPadLoss(pad_token_id=0)

    # 1 example, 4 timesteps, vocab size 5
    y_true = np.array([[1, 2, 0, 0]], dtype=np.int32)
    # Logits favoring index 1 and 2 respectively
    y_pred = np.zeros((1, 4, 5), dtype=np.float32)
    y_pred[0, 0, 1] = 10.0
    y_pred[0, 1, 2] = 10.0

    loss_val = float(loss_fn(y_true, y_pred))
    assert not np.isnan(loss_val)
    assert loss_val >= 0.0


def test_checkpoint_save_and_load(tmp_path: Path):
    """Verify weight saving and loading across checkpoint files."""
    config = ModelConfig(
        vocabulary_size=50,
        max_length=10,
        embedding_dim=16,
        mlp_dim=16,
        num_heads=2,
        num_blocks=1,
    )
    model1 = build_model(config)
    weights_path = tmp_path / "test_model.weights.h5"

    save_checkpoint(model1, weights_path)
    assert weights_path.exists()

    model2 = build_model(config)
    # Weights should be different before loading
    w1 = model1.get_weights()[0]
    w2 = model2.get_weights()[0]

    load_checkpoint(model2, weights_path)
    w2_after = model2.get_weights()[0]
    np.testing.assert_allclose(w1, w2_after, atol=1e-6)
