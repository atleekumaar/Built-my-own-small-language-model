"""Configuration definitions for the Small Language Model (SLM)."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal


@dataclass
class ModelConfig:
    """Verified architecture, tokenizer, and training configuration."""

    # Architecture hyperparameters
    model_type: str = "decoder_only_transformer"
    vocabulary_size: int = 5262
    max_length: int = 299  # Effective sequence length after autoregressive shifting
    embedding_dim: int = 256
    mlp_dim: int = 256
    num_heads: int = 2
    num_blocks: int = 1
    dropout_rate: float = 0.0
    activation_function: str = "relu"
    positional_embedding_type: Literal["sinusoidal", "simple"] = "sinusoidal"

    # Optimizer & training hyperparameters
    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.005
    pad_token_id: int = 0
    unknown_token_id: int = 5261

    # Tokenizer constants
    pad_token: str = "<PAD>"
    unknown_token: str = "<UNK>"

    # Verified metadata
    total_parameters: int = 3358094

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Instantiate configuration from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "ModelConfig":
        """Load configuration from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, json_path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
