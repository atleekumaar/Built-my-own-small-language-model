"""
Reproducible ML Experiments Runner for Phase 4.

Executes:
- Experiment A: Word-level baseline trained exclusively on the 185-paragraph train split.
- Experiment B: Byte-Level BPE model trained exclusively on the 185-paragraph train split.
- Fair Comparison: Evaluated on the exact same 47-paragraph held-out validation set.
- Diagnostics: Repetition rates, throughput, latency, parameter counts, loss curves.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
import keras
import numpy as np

from slm.bpe_tokenizer import BPETokenizer
from slm.config import ModelConfig
from slm.generate import generate_text
from slm.losses import CustomMaskPadLoss
from slm.model import build_model, save_checkpoint
from slm.tokenizer import SimpleWordTokenizer


def set_seed(seed: int = 812) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def load_data_and_splits():
    data_path = REPO_ROOT / "data" / "africa_galore.json"
    splits_path = REPO_ROOT / "data" / "dataset_splits.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_paras = [d["description"] for d in data]

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_indices = splits["train_indices"]
    val_indices = splits["val_indices"]

    train_paras = [all_paras[i] for i in train_indices]
    val_paras = [all_paras[i] for i in val_indices]

    return train_paras, val_paras, splits["metadata"]


def prepare_sequences(paragraphs, tokenizer, max_length=300):
    """Tokenize and pad sequences with pad_token_id=0."""
    encoded = [tokenizer.encode(p) for p in paragraphs]
    padded = keras.preprocessing.sequence.pad_sequences(
        encoded,
        maxlen=max_length,
        padding="post",
        truncating="post",
        value=tokenizer.pad_token_id,
    )
    # Autoregressive shift
    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    return inputs, targets


def compute_per_token_loss(model, inputs, targets, pad_token_id=0, batch_size=32):
    """Compute per-token cross entropy and perplexity excluding padding."""
    total_tokens = 0
    total_loss = 0.0
    n = len(inputs)
    n_batches = math.ceil(n / batch_size)

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n)
        b_in = inputs[s:e]
        b_tg = targets[s:e]

        logits = np.array(model(b_in, training=False))
        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        log_probs = (logits - max_l) - np.log(np.sum(exp_l, axis=-1, keepdims=True))

        for i in range(len(b_in)):
            for t in range(b_in.shape[1]):
                tid = b_tg[i, t]
                if tid != pad_token_id:
                    total_loss -= log_probs[i, t, tid]
                    total_tokens += 1

    if total_tokens == 0:
        return 0.0, 1.0, 0
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 50.0))
    return float(avg_loss), float(ppl), int(total_tokens)


def compute_repetition_metrics(text: str) -> dict:
    """Measure repetition metrics (unique 2-grams / total 2-grams)."""
    words = text.strip().split()
    if len(words) < 2:
        return {"bigram_repetition_rate": 0.0, "total_words": len(words), "unique_bigrams": 0}
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    unique_bigrams = len(set(bigrams))
    rep_rate = 1.0 - (unique_bigrams / len(bigrams))
    return {
        "bigram_repetition_rate": round(float(rep_rate), 4),
        "total_words": len(words),
        "unique_bigrams": unique_bigrams,
    }


def run_experiment_a_word(train_paras, val_paras, epochs=30, batch_size=32):
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT A: WORD-LEVEL BASELINE (TRAIN SPLIT ONLY)")
    print("=" * 70)
    set_seed(812)

    # 1. Fit tokenizer ONLY on train paragraphs
    print(f"[Word Baseline] Fitting SimpleWordTokenizer on {len(train_paras)} train paragraphs...")
    tokenizer = SimpleWordTokenizer(corpus=train_paras)
    vocab_path = REPO_ROOT / "data" / "vocab_baseline.json"
    tokenizer.save_vocabulary(vocab_path)
    vocab_size = tokenizer.vocab_size
    print(f"[Word Baseline] Fitted vocab size: {vocab_size} (saved to {vocab_path})")

    # 2. Prepare data
    train_in, train_tg = prepare_sequences(train_paras, tokenizer, max_length=300)
    val_in, val_tg = prepare_sequences(val_paras, tokenizer, max_length=300)

    # 3. Model setup
    config = ModelConfig(
        vocabulary_size=vocab_size,
        max_length=299,
        embedding_dim=256,
        mlp_dim=256,
        num_heads=2,
        num_blocks=1,
        unknown_token_id=tokenizer.unknown_token_id,
        dropout_rate=0.0,
    )
    config_path = REPO_ROOT / "configs" / "model_config_baseline.json"
    config.to_json(config_path)

    model = build_model(config)
    optimizer = keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.005)
    loss_fn = CustomMaskPadLoss(pad_token_id=0)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # 4. Training loop with per-epoch train and val loss logging
    history = {"epochs": [], "train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        # Epoch training
        res = model.fit(train_in, train_tg, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tr_k_loss = float(res.history["loss"][0])

        # Evaluate at epoch end
        tr_ce, tr_ppl, _ = compute_per_token_loss(model, train_in, train_tg)
        val_ce, val_ppl, _ = compute_per_token_loss(model, val_in, val_tg)

        history["epochs"].append(ep)
        history["train_loss"].append(round(tr_ce, 4))
        history["val_loss"].append(round(val_ce, 4))
        history["train_ppl"].append(round(tr_ppl, 2))
        history["val_ppl"].append(round(val_ppl, 2))

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:2d}/{epochs} | Train CE: {tr_ce:.4f} (PPL: {tr_ppl:7.2f}) | Val CE: {val_ce:.4f} (PPL: {val_ppl:7.2f})")

    duration = time.time() - t0

    # 5. Save checkpoints
    weights_path = REPO_ROOT / "checkpoints" / "slm_baseline_val.weights.h5"
    keras_path = REPO_ROOT / "checkpoints" / "slm_baseline_val.keras"
    save_checkpoint(model, weights_path)
    model.save(keras_path)
    print(f"[Word Baseline] Checkpoints saved to {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")

    # 6. Benchmark Inference throughput
    bench_start = time.time()
    n_tokens = 0
    test_prompts = [
        "Abeni,",
        "Jide was hungry so she went looking for",
        "Jide was thirsty so she went looking for",
        "The Lagos air was thick with",
    ]
    gen_results = {}
    for p in test_prompts:
        g = generate_text(p, max_new_tokens=15, model=model, tokenizer=tokenizer, sampling_mode="greedy")
        gen_results[p] = {
            "generated": g.generated_text,
            "full_text": g.full_text,
            "repetition": compute_repetition_metrics(g.full_text),
        }
        n_tokens += len(g.generated_tokens)
    bench_duration = time.time() - bench_start
    throughput = n_tokens / bench_duration if bench_duration > 0 else 0

    return {
        "model_name": "slm_baseline_val",
        "tokenizer_type": "word",
        "vocab_size": vocab_size,
        "parameters": model.count_params(),
        "training_time_s": round(duration, 2),
        "checkpoint_mb": round(weights_path.stat().st_size / 1e6, 2),
        "final_train_ce": history["train_loss"][-1],
        "final_train_ppl": history["train_ppl"][-1],
        "final_val_ce": history["val_loss"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "inference_tokens_per_sec": round(throughput, 1),
        "history": history,
        "sample_generations": gen_results,
    }


def run_experiment_b_bpe(train_paras, val_paras, epochs=30, batch_size=32, target_vocab=2000):
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT B: BYTE-LEVEL BPE (VOCAB={target_vocab}, TRAIN SPLIT ONLY)")
    print("=" * 70)
    set_seed(812)

    # 1. Fit BPE tokenizer ONLY on train paragraphs
    print(f"[BPE Subword] Training Byte-Level BPE on {len(train_paras)} train paragraphs...")
    tokenizer = BPETokenizer()
    tokenizer.train_from_iterator(train_paras, vocab_size=target_vocab, min_frequency=2)
    tok_path = REPO_ROOT / "checkpoints" / "bpe_tokenizer.json"
    vocab_path = REPO_ROOT / "data" / "vocab_bpe.json"
    tokenizer.save(tok_path)
    tokenizer.save_vocabulary(vocab_path)
    vocab_size = tokenizer.vocab_size
    print(f"[BPE Subword] Fitted BPE vocab size: {vocab_size} (saved to {tok_path})")

    # 2. Prepare data
    train_in, train_tg = prepare_sequences(train_paras, tokenizer, max_length=300)
    val_in, val_tg = prepare_sequences(val_paras, tokenizer, max_length=300)

    # 3. Model setup
    config = ModelConfig(
        vocabulary_size=vocab_size,
        max_length=299,
        embedding_dim=256,
        mlp_dim=256,
        num_heads=2,
        num_blocks=1,
        unknown_token_id=tokenizer.unk_token_id,
        dropout_rate=0.0,
    )
    config_path = REPO_ROOT / "configs" / "model_config_bpe.json"
    config.to_json(config_path)

    model = build_model(config)
    optimizer = keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.005)
    loss_fn = CustomMaskPadLoss(pad_token_id=0)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # 4. Training loop with per-epoch train and val loss logging
    history = {"epochs": [], "train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        res = model.fit(train_in, train_tg, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

        tr_ce, tr_ppl, _ = compute_per_token_loss(model, train_in, train_tg)
        val_ce, val_ppl, _ = compute_per_token_loss(model, val_in, val_tg)

        history["epochs"].append(ep)
        history["train_loss"].append(round(tr_ce, 4))
        history["val_loss"].append(round(val_ce, 4))
        history["train_ppl"].append(round(tr_ppl, 2))
        history["val_ppl"].append(round(val_ppl, 2))

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:2d}/{epochs} | Train CE: {tr_ce:.4f} (PPL: {tr_ppl:7.2f}) | Val CE: {val_ce:.4f} (PPL: {val_ppl:7.2f})")

    duration = time.time() - t0

    # 5. Save checkpoints
    weights_path = REPO_ROOT / "checkpoints" / "slm_bpe_val.weights.h5"
    keras_path = REPO_ROOT / "checkpoints" / "slm_bpe_val.keras"
    save_checkpoint(model, weights_path)
    model.save(keras_path)
    print(f"[BPE Subword] Checkpoints saved to {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")

    # 6. Benchmark Inference throughput
    bench_start = time.time()
    n_tokens = 0
    test_prompts = [
        "Abeni,",
        "Jide was hungry so she went looking for",
        "Jide was thirsty so she went looking for",
        "The Lagos air was thick with",
    ]
    gen_results = {}
    for p in test_prompts:
        g = generate_text(p, max_new_tokens=15, model=model, tokenizer=tokenizer, sampling_mode="greedy")
        gen_results[p] = {
            "generated": g.generated_text,
            "full_text": g.full_text,
            "repetition": compute_repetition_metrics(g.full_text),
        }
        n_tokens += len(g.generated_tokens)
    bench_duration = time.time() - bench_start
    throughput = n_tokens / bench_duration if bench_duration > 0 else 0

    return {
        "model_name": "slm_bpe_val",
        "tokenizer_type": "bpe",
        "vocab_size": vocab_size,
        "parameters": model.count_params(),
        "training_time_s": round(duration, 2),
        "checkpoint_mb": round(weights_path.stat().st_size / 1e6, 2),
        "final_train_ce": history["train_loss"][-1],
        "final_train_ppl": history["train_ppl"][-1],
        "final_val_ce": history["val_loss"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "inference_tokens_per_sec": round(throughput, 1),
        "history": history,
        "sample_generations": gen_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Phase 4 experiments")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bpe-vocab", type=int, default=2000)
    args = parser.parse_args()

    train_paras, val_paras, meta = load_data_and_splits()
    print(f"Loaded dataset: {len(train_paras)} train, {len(val_paras)} val paragraphs.")

    exp_a = run_experiment_a_word(train_paras, val_paras, epochs=args.epochs, batch_size=args.batch_size)
    exp_b = run_experiment_b_bpe(train_paras, val_paras, epochs=args.epochs, batch_size=args.batch_size, target_vocab=args.bpe_vocab)

    comparison = {
        "metadata": meta,
        "experiment_a_word_baseline": exp_a,
        "experiment_b_bpe_subword": exp_b,
        "summary": {
            "word_params": exp_a["parameters"],
            "bpe_params": exp_b["parameters"],
            "param_reduction_pct": round((1 - exp_b["parameters"] / exp_a["parameters"]) * 100, 2),
            "word_val_ppl": exp_a["final_val_ppl"],
            "bpe_val_ppl": exp_b["final_val_ppl"],
            "word_speed_tok_sec": exp_a["inference_tokens_per_sec"],
            "bpe_speed_tok_sec": exp_b["inference_tokens_per_sec"],
        },
    }

    out_file = REPO_ROOT / "checkpoints" / "experiment_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Full results saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
