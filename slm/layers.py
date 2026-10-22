# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core architectural layers for the custom Transformer Small Language Model (SLM).

Derived from Google DeepMind AI Research Foundations (ai_foundations.transformers.layers)
with Apache 2.0 attribution and registered Keras 3 serialization.
"""

from typing import Any, Callable

import jax
from jax import numpy as jnp
import keras
from keras import layers
from keras import ops

ANGLE_RATE_MULTIPLIER = 10000


@keras.saving.register_keras_serializable(package="slm")
class TokenAndPositionEmbedding(layers.Layer):
    """Combines token embeddings with positional embeddings."""

    def __init__(
        self,
        max_length: int,
        vocabulary_size: int,
        embedding_dim: int,
        positional_embedding_type: str = "sinusoidal",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        self.positional_embedding_type = positional_embedding_type

        # Set mask_zero=True so Keras handles masked tokens
        self.token_emb = layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True
        )

        if self.positional_embedding_type == "simple":
            self.pos_emb = layers.Embedding(
                input_dim=max_length, output_dim=embedding_dim
            )
        elif self.positional_embedding_type == "sinusoidal":
            self.pos_emb = self._positional_encoding(
                length=max_length, depth=embedding_dim
            )
        else:
            raise NotImplementedError(
                f"Positional embedding type '{self.positional_embedding_type}' not implemented."
            )

    def _positional_encoding(self, length: int, depth: int) -> Callable[[Any], jax.Array]:
        depth = depth // 2
        positions = jnp.arange(length)[:, jnp.newaxis]
        depths = jnp.arange(depth)[jnp.newaxis, :] / depth

        angle_rates = 1.0 / (ANGLE_RATE_MULTIPLIER**depths)
        angle_rads = positions * angle_rates

        pos_encoding = jnp.concatenate(
            [jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1
        )
        pos_encoding_matrix = ops.cast(pos_encoding, dtype="float32")

        def apply(*args: Any) -> jax.Array:
            return pos_encoding_matrix[jnp.newaxis, :, :]

        return apply

    def call(self, x: jax.Array) -> jax.Array:
        token_embeddings = self.token_emb(x)

        if self.positional_embedding_type == "sinusoidal":
            token_embeddings *= ops.sqrt(
                ops.cast(self.embedding_dim, dtype="float32")
            )
            position_embeddings = self.pos_emb(None)
        else:
            positions = ops.arange(0, self.max_length, 1)
            position_embeddings = self.pos_emb(positions)

        return token_embeddings + position_embeddings

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "positional_embedding_type": self.positional_embedding_type,
        })
        return config


@keras.saving.register_keras_serializable(package="slm")
class FeedForwardNetwork(layers.Layer):
    """Feed forward network with residual connection and layer normalization."""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.ffn = keras.Sequential([
            layers.Dense(mlp_dim, activation=activation),
            layers.Dense(embedding_dim),
        ])
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization()

    def call(self, x: jax.Array) -> jax.Array:
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        return self.layernorm(x + ffn_output)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        })
        return config


@keras.saving.register_keras_serializable(package="slm")
class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer with causal mask, dropout, and layer normalization."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization()

    def call(self, x: jax.Array) -> jax.Array:
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        attn_output = self.dropout(attn_output)
        return self.layernorm(x + attn_output)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.saving.register_keras_serializable(package="slm")
class TransformerBlock(layers.Layer):
    """A single Transformer decoder block combining self-attention and FFN."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        activation_function: str = "relu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function

        self.self_attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            activation=activation_function,
        )

    def call(self, inputs: jax.Array) -> jax.Array:
        attn_output = self.self_attention(inputs)
        ffn_output = self.feed_forward(attn_output)
        return ffn_output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "activation_function": self.activation_function,
        })
        return config
