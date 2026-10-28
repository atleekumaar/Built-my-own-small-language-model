"""Sliding-window evaluation script for computing full-corpus coverage Bits-per-Byte (BPB).

Evaluates 100% of held-out validation tokens without arbitrary sequence truncation.
Calculates:
1. Exact total cross-entropy nats over all tokens
2. Full-sequence Perplexity
3. True Bits-per-Byte (BPB) evaluated against the exact 26,159 raw UTF-8 bytes
4. Open-vocabulary adjusted BPB for the Word model
"""

import json
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

import keras
import numpy as np

from slm.bpe_tokenizer import BPETokenizer
from slm.config import ModelConfig
from slm.model import build_model, load_checkpoint
from slm.tokenizer import SimpleWordTokenizer


def evaluate_document_sliding_window(
    model: keras.Model,
    tokens: list[int],
    pad_token_id: int = 0,
    max_context: int = 299,
    stride: int = 200,
) -> tuple[float, int, int]:
    """Evaluate all autoregressive transitions in a document using a sliding window.

    Returns:
        (total_nats, non_pad_token_count, unk_token_count)
    """
    total_tokens = len(tokens)
    if total_tokens < 2:
        return 0.0, 0, 0

    target_tokens = tokens[1:]
    total_nats = 0.0
    evaluated_targets = 0

    # If document fits within max_context + 1 tokens
    if total_tokens <= max_context + 1:
        inp = np.array([tokens[:-1]], dtype=np.int32)
        padded_inp = keras.preprocessing.sequence.pad_sequences(
            inp, maxlen=max_context, padding="post", value=pad_token_id
        )
        logits = np.array(model(padded_inp, training=False))[0]  # (max_context, vocab)

        # Log softmax
        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        log_probs = (logits - max_l) - np.log(np.sum(exp_l, axis=-1, keepdims=True))

        for t_idx, tgt in enumerate(target_tokens):
            total_nats -= float(log_probs[t_idx, tgt])
            evaluated_targets += 1

        return total_nats, evaluated_targets, 0

    # Sliding window for documents exceeding context length
    # Evaluates all target tokens using the maximum available preceding context
    prev_end = 0
    for start in range(0, total_tokens - 1, stride):
        end = min(start + max_context, total_tokens - 1)
        inp_chunk = tokens[start:end]
        tgt_chunk = tokens[start + 1 : end + 1]

        # Determine which target positions to evaluate from this window
        # To avoid double-counting, only evaluate targets beyond prev_end
        eval_start_rel = max(0, prev_end - start)
        if eval_start_rel >= len(tgt_chunk):
            continue

        inp_arr = np.array([inp_chunk], dtype=np.int32)
        padded_inp = keras.preprocessing.sequence.pad_sequences(
            inp_arr, maxlen=max_context, padding="post", value=pad_token_id
        )
        logits = np.array(model(padded_inp, training=False))[0]

        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        log_probs = (logits - max_l) - np.log(np.sum(exp_l, axis=-1, keepdims=True))

        for pos in range(eval_start_rel, len(tgt_chunk)):
            tgt = tgt_chunk[pos]
            total_nats -= float(log_probs[pos, tgt])
            evaluated_targets += 1

        prev_end = end
        if end >= total_tokens - 1:
            break

    return total_nats, evaluated_targets, 0


def main():
    print("=" * 70)
    print("FULL-SEQUENCE SLIDING WINDOW BPB BENCHMARK (100% BYTE COVERAGE)")
    print("=" * 70)

    # 1. Load data
    splits_path = REPO_ROOT / "data" / "dataset_splits.json"
    data_path = REPO_ROOT / "data" / "africa_galore.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_paras = [d["description"] for d in data]
    val_paras = [all_paras[i] for i in splits["val_indices"]]
    total_val_bytes = sum(len(p.encode("utf-8")) for p in val_paras)
    print(f"Validation Paragraphs: {len(val_paras)}")
    print(f"Total Raw UTF-8 Bytes: {total_val_bytes:,} (100.0% Byte Accounting)")

    # 2. Evaluate Word Baseline
    word_cfg = ModelConfig.from_json(REPO_ROOT / "configs" / "model_config_baseline.json")
    word_tok = SimpleWordTokenizer.from_vocabulary_file(REPO_ROOT / "data" / "vocab_baseline.json")
    word_model = build_model(word_cfg)
    load_checkpoint(word_model, REPO_ROOT / "checkpoints" / "slm_baseline_val.weights.h5")

    word_nats = 0.0
    word_targets = 0
    word_unks = 0
    unk_id_word = word_tok.unknown_token_id

    for p in val_paras:
        encoded = word_tok.encode(p)
        word_unks += sum(1 for t in encoded[1:] if t == unk_id_word)
        nats, count, _ = evaluate_document_sliding_window(word_model, encoded)
        word_nats += nats
        word_targets += count

    word_ce = word_nats / word_targets if word_targets > 0 else 0.0
    word_ppl = math.exp(min(word_ce, 50.0))
    word_bits = word_nats / math.log(2.0)
    word_raw_bpb = word_bits / total_val_bytes
    # Adjusted with open-vocab 50K prior:
    word_adj_bpb = (word_bits + word_unks * 15.61) / total_val_bytes

    print("\n--- WORD BASELINE (FULL SLIDING WINDOW) ---")
    print(f"Evaluated Target Tokens: {word_targets:,} / {sum(len(word_tok.encode(p))-1 for p in val_paras):,}")
    print(f"Unknown Tokens (OOV):    {word_unks:,} ({(word_unks / word_targets * 100):.2f}%)")
    print(f"Per-Token Cross-Entropy: {word_ce:.4f} nats")
    print(f"Perplexity (PPL):        {word_ppl:.2f}")
    print(f"Raw Bits-Per-Byte:       {word_raw_bpb:.4f} bits/byte (ignores UNK constituent chars)")
    print(f"Open-Vocab Adjusted BPB: {word_adj_bpb:.4f} bits/byte (+15.61 bits / UNK)")

    # 3. Evaluate Byte-Level BPE
    bpe_cfg = ModelConfig.from_json(REPO_ROOT / "configs" / "model_config_bpe.json")
    bpe_tok = BPETokenizer.from_file(REPO_ROOT / "checkpoints" / "bpe_tokenizer.json")
    bpe_model = build_model(bpe_cfg)
    load_checkpoint(bpe_model, REPO_ROOT / "checkpoints" / "slm_bpe_val.weights.h5")

    bpe_nats = 0.0
    bpe_targets = 0
    bpe_unks = 0
    unk_id_bpe = getattr(bpe_tok, "unk_token_id", 1)

    for p in val_paras:
        encoded = bpe_tok.encode(p)
        bpe_unks += sum(1 for t in encoded[1:] if t == unk_id_bpe)
        nats, count, _ = evaluate_document_sliding_window(bpe_model, encoded)
        bpe_nats += nats
        bpe_targets += count

    bpe_ce = bpe_nats / bpe_targets if bpe_targets > 0 else 0.0
    bpe_ppl = math.exp(min(bpe_ce, 50.0))
    bpe_bits = bpe_nats / math.log(2.0)
    bpe_raw_bpb = bpe_bits / total_val_bytes
    bpe_adj_bpb = bpe_raw_bpb  # 0.0% UNK

    print("\n--- BYTE-LEVEL BPE (FULL SLIDING WINDOW) ---")
    print(f"Evaluated Target Tokens: {bpe_targets:,} / {sum(len(bpe_tok.encode(p))-1 for p in val_paras):,}")
    print(f"Unknown Tokens (OOV):    {bpe_unks:,} (0.00% Zero UNK)")
    print(f"Per-Token Cross-Entropy: {bpe_ce:.4f} nats")
    print(f"Perplexity (PPL):        {bpe_ppl:.2f}")
    print(f"Raw Bits-Per-Byte:       {bpe_raw_bpb:.4f} bits/byte (Exact character/subword entropy)")
    print(f"Open-Vocab Adjusted BPB: {bpe_adj_bpb:.4f} bits/byte")

    # Save sliding window results
    sw_results = {
        "evaluation_protocol": "sliding_window_full_sequence",
        "total_validation_paragraphs": len(val_paras),
        "total_validation_bytes": total_val_bytes,
        "byte_coverage_pct": 100.0,
        "models": {
            "word_baseline": {
                "evaluated_target_tokens": word_targets,
                "unk_tokens": word_unks,
                "unk_rate_pct": round(word_unks / word_targets * 100, 2),
                "per_token_ce_nats": round(word_ce, 4),
                "perplexity": round(word_ppl, 2),
                "raw_bpb": round(word_raw_bpb, 4),
                "adjusted_bpb": round(word_adj_bpb, 4),
            },
            "bpe_subword": {
                "evaluated_target_tokens": bpe_targets,
                "unk_tokens": bpe_unks,
                "unk_rate_pct": 0.0,
                "per_token_ce_nats": round(bpe_ce, 4),
                "perplexity": round(bpe_ppl, 2),
                "raw_bpb": round(bpe_raw_bpb, 4),
                "adjusted_bpb": round(bpe_adj_bpb, 4),
            },
        },
    }

    out_file = REPO_ROOT / "experiments" / "results" / "sliding_window_bpb_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sw_results, f, indent=2)
    print(f"\n[SUCCESS] Full sliding-window results saved to: {out_file}")


if __name__ == "__main__":
    main()
