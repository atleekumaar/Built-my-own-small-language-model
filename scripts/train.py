"""Reproducible training script matching the original Colab notebook.

Hyperparameters match the verified settings:
- Optimizer: AdamW (lr=1e-4, weight_decay=0.005)
- Seed: 812
- Batch Size: 32
- Sequence Length: 300 (effective max_length=299)
- Epochs: 200
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys

# Add repository root to python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure Keras JAX backend
os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

from slm.config import ModelConfig
from slm.model import build_model, save_checkpoint
from slm.tokenizer import SimpleWordTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def load_dataset(dataset_path: str | Path) -> list[str]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["description"] for item in data]


def prepare_sequences(
    paragraphs: list[str],
    tokenizer: SimpleWordTokenizer,
    target_length: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize, pad, and create autoregressive input/target pairs."""
    encoded_tokens = [tokenizer.encode(p) for p in paragraphs]

    # Post-padding with pad_token_id (0)
    padded = keras.preprocessing.sequence.pad_sequences(
        encoded_tokens,
        maxlen=target_length,
        padding="post",
        truncating="post",
        value=tokenizer.pad_token_id,
    )

    # Autoregressive shift: inputs = tokens[:-1], targets = tokens[1:]
    input_seqs = padded[:, :-1]
    target_seqs = padded[:, 1:]
    return input_seqs, target_seqs


def main():
    parser = argparse.ArgumentParser(description="Train Small Language Model (SLM)")
    parser.add_argument("--data-path", type=str, default="data/africa_galore.json")
    parser.add_argument("--vocab-path", type=str, default="data/vocab.json")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--val-split", type=float, default=0.0, help="Fraction for held-out validation (0.0 = original notebook)")
    parser.add_argument("--print-every", type=int, default=10, help="Print generated text every N epochs")
    parser.add_argument("--verbose", type=int, default=2, help="Keras fit verbose (1=progress bar, 2=one line per epoch)")
    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data_path}...")
    paragraphs = load_dataset(args.data_path)
    print(f"Loaded {len(paragraphs)} paragraphs.")

    # Initialize tokenizer
    if os.path.exists(args.vocab_path):
        tokenizer = SimpleWordTokenizer.from_vocabulary_file(args.vocab_path)
    else:
        tokenizer = SimpleWordTokenizer(corpus=paragraphs)
        tokenizer.save_vocabulary(args.vocab_path)

    input_seqs, target_seqs = prepare_sequences(paragraphs, tokenizer, target_length=300)
    effective_max_length = input_seqs.shape[1]

    # Initialize model config
    config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        max_length=effective_max_length,
        learning_rate=args.lr,
    )
    model = build_model(config)
    print(f"Constructed model with {model.count_params():,} parameters.")

    indices = np.arange(len(input_seqs))
    np.random.shuffle(indices)
    input_seqs = input_seqs[indices]
    target_seqs = target_seqs[indices]

    if args.val_split > 0.0:
        val_count = int(len(input_seqs) * args.val_split)
        train_inputs, val_inputs = input_seqs[val_count:], input_seqs[:val_count]
        train_targets, val_targets = target_seqs[val_count:], target_seqs[:val_count]
        val_data = (val_inputs, val_targets)
        print(f"Split data: {len(train_inputs)} train, {len(val_inputs)} validation.")
    else:
        train_inputs = input_seqs
        train_targets = target_seqs
        val_data = None
        print(f"Training on all {len(train_inputs)} examples (matching original notebook).")

    # Periodic generation callback matching notebook cell 48
    from slm.generate import generate_text

    class TextGeneratorCallback(keras.callbacks.Callback):
        def __init__(self, prompt: str = "Abeni,", max_tokens: int = 10, print_every: int = 10):
            super().__init__()
            self.prompt = prompt
            self.max_tokens = max_tokens
            self.print_every = print_every

        def on_epoch_end(self, epoch: int, logs: dict | None = None):
            current_epoch = epoch + 1
            if self.print_every > 0 and current_epoch % self.print_every == 0:
                res = generate_text(
                    self.prompt,
                    self.max_tokens,
                    model=self.model,
                    tokenizer=tokenizer,
                    sampling_mode="greedy",
                )
                print(f"\n--- Epoch {current_epoch} Generation: '{res.full_text}' ---\n")

    callbacks = [TextGeneratorCallback(prompt="Abeni,", max_tokens=10, print_every=args.print_every)]

    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        x=train_inputs,
        y=train_targets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    # Save artifacts
    weights_path = save_dir / "slm_africa_galore.weights.h5"
    save_checkpoint(model, weights_path)
    print(f"Saved weights to {weights_path}")

    keras_path = save_dir / "slm_africa_galore.keras"
    save_checkpoint(model, keras_path)
    print(f"Saved full model to {keras_path}")

    hist_path = save_dir / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        # Cast float32 items to standard float for clean JSON serialization
        serializable_hist = {
            k: [float(val) for val in v] for k, v in history.history.items()
        }
        json.dump(serializable_hist, f, indent=2)
    print(f"Saved training history to {hist_path}")


if __name__ == "__main__":
    main()
