"""Small Language Model (SLM) package."""

import os

# Ensure Keras defaults to the JAX backend as established in the notebook
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

from slm.config import ModelConfig
from slm.generate import GenerationResult, generate_text
from slm.layers import TokenAndPositionEmbedding, TransformerBlock
from slm.losses import CustomMaskPadLoss
from slm.model import build_model, load_checkpoint, load_model, save_checkpoint
from slm.tokenizer import SimpleWordTokenizer

__all__ = [
    "ModelConfig",
    "SimpleWordTokenizer",
    "TokenAndPositionEmbedding",
    "TransformerBlock",
    "CustomMaskPadLoss",
    "build_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    "generate_text",
    "GenerationResult",
]
