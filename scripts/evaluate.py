"""Model evaluation script for loss, perplexity, and prompt inspection.

Clearly distinguishes:
1. Recorded baseline outputs (from notebook logs)
2. Measured live metrics (from current weights/model)
3. Proposed production targets (for playground readiness)
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

# Add repository root to python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np

from slm.config import ModelConfig
from slm.generate import generate_text
from slm.model import build_model, load_checkpoint
from slm.tokenizer import SimpleWordTokenizer


def calculate_per_token_loss_and_perplexity(
    model: keras.Model,
    input_seqs: np.ndarray,
    target_seqs: np.ndarray,
    pad_token_id: int = 0,
    batch_size: int = 32,
) -> tuple[float, float, int]:
    """Calculate exact per-token Cross-Entropy loss and Perplexity, excluding padding tokens."""
    total_non_pad_tokens = 0
    total_cross_entropy = 0.0

    n_samples = len(input_seqs)
    n_batches = math.ceil(n_samples / batch_size)

    for b in range(n_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_inputs = input_seqs[start_idx:end_idx]
        batch_targets = target_seqs[start_idx:end_idx]

        # Forward pass (logits)
        logits = np.array(model(batch_inputs, training=False))  # shape: (batch, seq, vocab)

        # Log softmax
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        log_probs = (logits - max_logits) - np.log(np.sum(exp_logits, axis=-1, keepdims=True))

        # Extract target log probs
        for i in range(len(batch_inputs)):
            for t in range(batch_inputs.shape[1]):
                target_id = batch_targets[i, t]
                if target_id != pad_token_id:
                    total_cross_entropy -= log_probs[i, t, target_id]
                    total_non_pad_tokens += 1

    if total_non_pad_tokens == 0:
        return 0.0, 1.0, 0

    per_token_loss = total_cross_entropy / total_non_pad_tokens
    perplexity = math.exp(min(per_token_loss, 50.0))  # Avoid float overflow
    return float(per_token_loss), float(perplexity), int(total_non_pad_tokens)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Small Language Model")
    parser.add_argument("--weights-path", type=str, default="checkpoints/slm_africa_galore.weights.h5")
    parser.add_argument("--data-path", type=str, default="data/africa_galore.json")
    parser.add_argument("--vocab-path", type=str, default="data/vocab.json")
    args = parser.parse_args()

    print("=" * 70)
    print("SLM MODEL EVALUATION & METRICS REPORT")
    print("=" * 70)

    # 1. Load tokenizer and data
    tokenizer = SimpleWordTokenizer.from_vocabulary_file(args.vocab_path)
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    paragraphs = [d["description"] for d in data]

    # 2. Tokenize and pad
    encoded_tokens = [tokenizer.encode(p) for p in paragraphs]
    padded = keras.preprocessing.sequence.pad_sequences(
        encoded_tokens, maxlen=300, padding="post", truncating="post", value=0
    )
    input_seqs = padded[:, :-1]
    target_seqs = padded[:, 1:]

    # 3. Model construction and weights loading
    config = ModelConfig()
    model = build_model(config)

    weights_exist = os.path.exists(args.weights_path)
    if weights_exist:
        load_checkpoint(model, args.weights_path)
        print(f"[STATUS] Loaded trained weights from: {args.weights_path}")
    else:
        print(f"[STATUS] No weights file found at {args.weights_path}.")
        print("[STATUS] Evaluating model with initialized weights (reproduction baseline).")

    # 4. Compute metrics
    per_token_ce, perplexity, n_tokens = calculate_per_token_loss_and_perplexity(
        model, input_seqs, target_seqs, pad_token_id=0
    )

    print("\n--- MEASURED METRICS ---")
    print(f"Total Evaluated Non-Pad Tokens: {n_tokens:,}")
    print(f"Per-Token Cross-Entropy Loss:   {per_token_ce:.4f}")
    print(f"Perplexity:                    {perplexity:.2f}")

    print("\n--- RECORDED NOTEBOOK BASELINES (for comparison) ---")
    print("Final Recorded Notebook Loss:  0.0159 (CustomMaskPadLoss, sum_over_batch_size)")
    print("Training Epochs:               200")
    print("Optimizer:                     AdamW (lr=1e-4, weight_decay=0.005)")
    print("Held-out Validation Split:     None (0% - whole corpus was training set)")

    print("\n--- PROPOSED PLAYGROUND TARGETS ---")
    print("Target P95 Latency (CPU):      < 150 ms per token")
    print("Target Perplexity on In-Domain: < 25.0")
    print("Sampling Modes:                Greedy & Temperature-Scaled Random Sampling")

    print("\n--- GENERATION VERIFICATION ---")
    test_prompts = [
        ("Abeni,", 10, "greedy"),
        ("Jide was hungry so she went looking for", 10, "greedy"),
        (" ATLEE KUMAAR 2072 ", 10, "greedy"),
    ]

    for p, n, mode in test_prompts:
        res = generate_text(p, n, model, tokenizer, sampling_mode=mode)
        print(f"\nPrompt:    '{p}'")
        print(f"Generated: '{res.generated_text}'")
        print(f"Full Text: '{res.full_text}'")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
