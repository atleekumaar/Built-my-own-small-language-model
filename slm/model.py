"""Model construction, compilation, and checkpoint loading/saving utilities."""

import os
from pathlib import Path
from typing import Any

import keras
from keras import layers

from slm.config import ModelConfig
from slm.layers import TokenAndPositionEmbedding, TransformerBlock
from slm.losses import CustomMaskPadLoss


def get_optimizer(optimizer_name: str, learning_rate: float, weight_decay: float = 0.005) -> keras.optimizers.Optimizer:
    """Helper to instantiate the specified Keras optimizer."""
    opt_lower = optimizer_name.lower()
    if opt_lower == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_accumulation_steps=None,
        )
    elif opt_lower == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt_lower == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not supported.")


def build_model(config: ModelConfig | None = None) -> keras.Model:
    """Construct and compile the Transformer Small Language Model.

    Args:
        config: ModelConfig instance. If None, default verified config is used.

    Returns:
        Compiled keras.Model instance.
    """
    if config is None:
        config = ModelConfig()

    # Input layer accepting token ID sequences
    inputs = layers.Input(shape=(config.max_length,), dtype="int32", name="token_input")

    # Combined token and sinusoidal positional embedding
    embedding_layer = TokenAndPositionEmbedding(
        max_length=config.max_length,
        vocabulary_size=config.vocabulary_size,
        embedding_dim=config.embedding_dim,
        positional_embedding_type=config.positional_embedding_type,
        name="token_and_position_embedding",
    )
    x = embedding_layer(inputs)

    # Stack of Transformer blocks
    for i in range(config.num_blocks):
        block = TransformerBlock(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            dropout_rate=config.dropout_rate,
            activation_function=config.activation_function,
            name=f"transformer_block_{i}",
        )
        x = block(x)

    # Output projection to vocabulary logits
    outputs = layers.Dense(config.vocabulary_size, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="slm_transformer")

    # Configure optimizer and loss
    optimizer = get_optimizer(
        optimizer_name=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = CustomMaskPadLoss(pad_token_id=config.pad_token_id)
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model


def load_model(
    config: ModelConfig | None = None,
    weights_path: str | Path | None = None,
) -> keras.Model:
    """Instantiate model and optionally load weights from checkpoint.

    Checkpoint formats supported:
    - Keras `.weights.h5` via `model.load_weights(weights_path)`
    - Full Keras `.keras` via `keras.models.load_model(weights_path)`

    Args:
        config: ModelConfig instance.
        weights_path: Optional path to weights file. If None, model has initialized weights.

    Returns:
        keras.Model instance.
    """
    if weights_path is not None:
        w_path = Path(weights_path)
        if not w_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

        if str(w_path).endswith(".keras"):
            # Load complete serialized model
            return keras.models.load_model(
                w_path,
                custom_objects={
                    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                    "TransformerBlock": TransformerBlock,
                    "CustomMaskPadLoss": CustomMaskPadLoss,
                },
            )

    model = build_model(config)

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def save_checkpoint(model: keras.Model, filepath: str | Path) -> None:
    """Save model checkpoint to either .keras or .weights.h5 format.

    Args:
        model: keras.Model instance to save.
        filepath: Destination path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if str(path).endswith(".weights.h5"):
        model.save_weights(path)
    else:
        model.save(path)


def load_checkpoint(model: keras.Model, filepath: str | Path) -> None:
    """Load weights into an existing model instance from a checkpoint.

    Args:
        model: keras.Model instance to populate with weights.
        filepath: Path to .weights.h5 or .keras checkpoint.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if str(path).endswith(".weights.h5"):
        model.load_weights(path)
    elif str(path).endswith(".keras"):
        # Load weights from .keras file
        loaded = keras.models.load_model(
            path,
            custom_objects={
                "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                "TransformerBlock": TransformerBlock,
                "CustomMaskPadLoss": CustomMaskPadLoss,
            },
        )
        model.set_weights(loaded.get_weights())
    else:
        model.load_weights(path)

