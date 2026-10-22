"""
Phase 5: Scientific Validation, Cross-Tokenizer Normalization (Bits Per Byte),
Data Contamination Audit, and Reproducibility Verification.
"""

import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
import keras
import numpy as np

from slm.bpe_tokenizer import BPETokenizer
from slm.config import ModelConfig
from slm.generate import generate_text
from slm.model import build_model, load_checkpoint
from slm.tokenizer import SimpleWordTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def audit_dataset_contamination():
    """Audit train vs validation split for exact duplicates, near-duplicates, and n-gram overlap."""
    print("=" * 70)
    print("1. DATASET CONTAMINATION & SPLIT INTEGRITY AUDIT")
    print("=" * 70)

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

    print(f"Total corpus paragraphs: {len(all_paras)}")
    print(f"Train split paragraphs:  {len(train_paras)} (indices: {len(train_indices)})")
    print(f"Val split paragraphs:    {len(val_paras)} (indices: {len(val_indices)})")

    # Check index overlap
    idx_overlap = set(train_indices).intersection(set(val_indices))
    print(f"Index overlap between train and val: {len(idx_overlap)} (Expected: 0)")
    assert len(idx_overlap) == 0, "Train and Val indices overlap!"

    # Check exact paragraph string equality
    train_set = set(train_paras)
    val_set = set(val_paras)
    exact_overlap = train_set.intersection(val_set)
    print(f"Exact string overlap between train and val: {len(exact_overlap)} paragraphs")

    # Near-duplicate analysis using 5-gram Jaccard similarity
    def get_ngrams(text, n=5):
        words = text.lower().split()
        if len(words) < n:
            return set([" ".join(words)])
        return set([" ".join(words[i:i+n]) for i in range(len(words) - n + 1)])

    max_jaccard = 0.0
    highest_pair = (None, None, 0.0)
    high_similarity_count = 0

    for v_idx, v_p in enumerate(val_paras):
        v_ngrams = get_ngrams(v_p, n=5)
        for t_idx, t_p in enumerate(train_paras):
            t_ngrams = get_ngrams(t_p, n=5)
            intersection = v_ngrams.intersection(t_ngrams)
            union = v_ngrams.union(t_ngrams)
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                highest_pair = (v_idx, t_idx, jaccard)
            if jaccard > 0.3:  # Significant phrase borrowing
                high_similarity_count += 1

    print(f"Max 5-gram Jaccard similarity across train/val: {max_jaccard:.4f}")
    print(f"Pairs with >0.3 5-gram Jaccard similarity: {high_similarity_count}")
    if max_jaccard > 0:
        v_i, t_i, score = highest_pair
        print(f"  Sample highest overlap (Val #{v_i} vs Train #{t_i}, score={score:.4f}):")
        print(f"  Val:   '{val_paras[v_i][:80]}...'")
        print(f"  Train: '{train_paras[t_i][:80]}...'")

    return {
        "train_count": len(train_paras),
        "val_count": len(val_paras),
        "index_overlap": len(idx_overlap),
        "exact_overlap": len(exact_overlap),
        "max_5gram_jaccard": round(max_jaccard, 4),
        "pairs_above_30pct_jaccard": high_similarity_count,
    }


def evaluate_model_comprehensive(model, tokenizer, paragraphs, pad_token_id=0, max_length=300, batch_size=32):
    """
    Computes:
    - Token-level Cross-Entropy (nats) & Perplexity
    - Bits Per UTF-8 Byte (BPB)
    - Total tokens, total bytes, target sequence lengths
    - Out-Of-Vocabulary (<UNK>) counts and percentage
    """
    total_tokens = 0
    total_nats = 0.0
    total_bytes = 0
    total_unk_tokens = 0

    # Total UTF-8 bytes of raw paragraphs evaluated
    encoded_list = [tokenizer.encode(p) for p in paragraphs]

    # Calculate raw bytes
    for p in paragraphs:
        total_bytes += len(p.encode("utf-8"))

    padded = keras.preprocessing.sequence.pad_sequences(
        encoded_list,
        maxlen=max_length,
        padding="post",
        truncating="post",
        value=pad_token_id,
    )
    inputs = padded[:, :-1]
    targets = padded[:, 1:]

    n = len(inputs)
    n_batches = math.ceil(n / batch_size)

    unk_id = getattr(tokenizer, "unknown_token_id", None) or getattr(tokenizer, "unk_token_id", 1)

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
                    total_nats -= log_probs[i, t, tid]
                    total_tokens += 1
                    if tid == unk_id:
                        total_unk_tokens += 1

    per_token_ce_nats = total_nats / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(min(per_token_ce_nats, 50.0))

    # Bits per UTF-8 byte
    # NLL_bits = total_nats / ln(2)
    # BPB = NLL_bits / total_bytes
    total_bits = total_nats / math.log(2.0)
    raw_bpb = total_bits / total_bytes if total_bytes > 0 else 0.0

    # Open-vocabulary adjusted BPB:
    # If a model outputs UNK, it only predicted that some word exists.
    # In an open vocabulary, predicting an unknown word from a conservative estimate of
    # English/African geographical vocabulary (~50,000 words not in the 4,409 train vocab)
    # requires an extra log2(50,000) = 15.6 bits per UNK token!
    extra_bits_per_unk = 15.61  # log2(50000)
    adjusted_bits = total_bits + (total_unk_tokens * extra_bits_per_unk)
    adjusted_bpb = adjusted_bits / total_bytes if total_bytes > 0 else 0.0

    unk_pct = (total_unk_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0

    def to_py_float(val, ndigits=4):
        try:
            return float(str(round(float(val), ndigits)))
        except Exception:
            return 0.0

    return {
        "total_tokens": int(total_tokens),
        "total_bytes": int(total_bytes),
        "total_unk_tokens": int(total_unk_tokens),
        "unk_rate_pct": to_py_float(unk_pct, 2),
        "per_token_ce_nats": to_py_float(per_token_ce_nats, 4),
        "perplexity": to_py_float(ppl, 2),
        "raw_bpb": to_py_float(raw_bpb, 4),
        "adjusted_bpb": to_py_float(adjusted_bpb, 4),
        "bytes_per_token": to_py_float(total_bytes / total_tokens if total_tokens > 0 else 0.0, 3),
    }


def main():
    contamination_res = audit_dataset_contamination()

    # Load splits
    splits_path = REPO_ROOT / "data" / "dataset_splits.json"
    data_path = REPO_ROOT / "data" / "africa_galore.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_paras = [d["description"] for d in data]
    train_paras = [all_paras[i] for i in splits["train_indices"]]
    val_paras = [all_paras[i] for i in splits["val_indices"]]

    print("\n" + "=" * 70)
    print("2. VERIFYING & REPRODUCING MODEL METRICS")
    print("=" * 70)

    # 1. Load Word Baseline
    word_cfg_path = REPO_ROOT / "configs" / "model_config_baseline.json"
    word_weights_path = REPO_ROOT / "checkpoints" / "slm_baseline_val.weights.h5"
    word_vocab_path = REPO_ROOT / "data" / "vocab_baseline.json"
    word_tok = SimpleWordTokenizer.from_vocabulary_file(word_vocab_path)
    word_cfg = ModelConfig.from_json(word_cfg_path)
    word_model = build_model(word_cfg)
    load_checkpoint(word_model, word_weights_path)
    print(f"[Word Model] Loaded successfully. Vocab: {word_tok.vocab_size}, Params: {word_model.count_params():,}")

    # 2. Load BPE Model
    bpe_cfg_path = REPO_ROOT / "configs" / "model_config_bpe.json"
    bpe_weights_path = REPO_ROOT / "checkpoints" / "slm_bpe_val.weights.h5"
    bpe_tok_path = REPO_ROOT / "checkpoints" / "bpe_tokenizer.json"
    bpe_tok = BPETokenizer.from_file(bpe_tok_path)
    bpe_cfg = ModelConfig.from_json(bpe_cfg_path)
    bpe_model = build_model(bpe_cfg)
    load_checkpoint(bpe_model, bpe_weights_path)
    print(f"[BPE Model]  Loaded successfully. Vocab: {bpe_tok.vocab_size}, Params: {bpe_model.count_params():,}")

    # Evaluate Word Model on Train & Val
    print("\nEvaluating Word Model on Train Split (185 paras)...")
    word_train_eval = evaluate_model_comprehensive(word_model, word_tok, train_paras)
    print("Evaluating Word Model on Held-out Val Split (47 paras)...")
    word_val_eval = evaluate_model_comprehensive(word_model, word_tok, val_paras)

    # Evaluate BPE Model on Train & Val
    print("\nEvaluating BPE Model on Train Split (185 paras)...")
    bpe_train_eval = evaluate_model_comprehensive(bpe_model, bpe_tok, train_paras)
    print("Evaluating BPE Model on Held-out Val Split (47 paras)...")
    bpe_val_eval = evaluate_model_comprehensive(bpe_model, bpe_tok, val_paras)

    print("\n" + "=" * 70)
    print("3. RIGOROUS CROSS-TOKENIZER EVALUATION MATRIX")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Word Baseline':<16} | {'Byte-Level BPE':<16}")
    print("-" * 72)
    print(f"{'Total Parameters':<35} | {word_model.count_params():<16,} | {bpe_model.count_params():<16,}")
    print(f"{'Vocabulary Size':<35} | {word_tok.vocab_size:<16} | {bpe_tok.vocab_size:<16}")
    print(f"{'Val Total Tokens':<35} | {word_val_eval['total_tokens']:<16} | {bpe_val_eval['total_tokens']:<16}")
    print(f"{'Val Total UTF-8 Bytes':<35} | {word_val_eval['total_bytes']:<16} | {bpe_val_eval['total_bytes']:<16}")
    print(f"{'Bytes per Token (Granularity)':<35} | {word_val_eval['bytes_per_token']:<16} | {bpe_val_eval['bytes_per_token']:<16}")
    print(f"{'Val UNK Count (OOV)':<35} | {word_val_eval['total_unk_tokens']:<16} | {bpe_val_eval['total_unk_tokens']:<16}")
    print(f"{'Val UNK Rate (%)':<35} | {word_val_eval['unk_rate_pct']:<16} | {bpe_val_eval['unk_rate_pct']:<16}")
    print(f"{'Tokenizer-Specific Val CE (nats)':<35} | {word_val_eval['per_token_ce_nats']:<16} | {bpe_val_eval['per_token_ce_nats']:<16}")
    print(f"{'Tokenizer-Specific Val Perplexity':<35} | {word_val_eval['perplexity']:<16} | {bpe_val_eval['perplexity']:<16}")
    print("-" * 72)
    print(f"{'COMMON METRIC: Raw BPB (bits/byte)':<35} | {word_val_eval['raw_bpb']:<16} | {bpe_val_eval['raw_bpb']:<16}")
    print(f"{'COMMON METRIC: Open-Vocab Adj. BPB':<35} | {word_val_eval['adjusted_bpb']:<16} | {bpe_val_eval['adjusted_bpb']:<16}")

    print("\n" + "=" * 70)
    print("4. QUALITATIVE GENERATION & REPETITION DIAGNOSTICS")
    print("=" * 70)
    prompts = [
        "Kilimanjaro is",
        "The Sahara desert",
        "Ancient civilizations",
        "Wildlife in Africa",
        "Abeni visited",
        "Jide was thirsty so she went looking for",
    ]

    qual_results = []
    for p in prompts:
        # Word model greedy
        t0 = time.time()
        g_w = generate_text(p, max_new_tokens=25, model=word_model, tokenizer=word_tok, sampling_mode="greedy")
        w_lat = (time.time() - t0) * 1000 / len(g_w.generated_tokens) if g_w.generated_tokens else 0

        # BPE model greedy
        t0 = time.time()
        g_b = generate_text(p, max_new_tokens=25, model=bpe_model, tokenizer=bpe_tok, sampling_mode="greedy")
        b_lat = (time.time() - t0) * 1000 / len(g_b.generated_tokens) if g_b.generated_tokens else 0

        # Word model sampling
        g_w_samp = generate_text(p, max_new_tokens=25, model=word_model, tokenizer=word_tok, sampling_mode="random", temperature=0.7, seed=42)
        # BPE model sampling
        g_b_samp = generate_text(p, max_new_tokens=25, model=bpe_model, tokenizer=bpe_tok, sampling_mode="random", temperature=0.7, seed=42)

        def analyze_text(text):
            words = text.strip().split()
            n_words = len(words)
            if n_words < 2:
                return {"rep_rate": 0.0, "words": n_words, "unk_count": text.count("<UNK>")}
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(n_words - 1)]
            rep = 1.0 - (len(set(bigrams)) / len(bigrams))
            return {
                "rep_rate": round(rep, 3),
                "words": n_words,
                "unk_count": text.count("<UNK>"),
            }

        w_stats = analyze_text(g_w.generated_text)
        b_stats = analyze_text(g_b.generated_text)

        print(f"\n--- Prompt: '{p}' ---")
        print(f"  [Word Greedy] ({w_lat:.1f}ms/tok, rep={w_stats['rep_rate']}, UNK={w_stats['unk_count']}):\n    '{g_w.full_text}'")
        print(f"  [BPE  Greedy] ({b_lat:.1f}ms/tok, rep={b_stats['rep_rate']}, UNK={b_stats['unk_count']}):\n    '{g_b.full_text}'")
        print(f"  [BPE  T=0.7 ]:\n    '{g_b_samp.full_text}'")

        qual_results.append({
            "prompt": p,
            "word_greedy": {
                "text": g_w.full_text,
                "latency_ms_per_tok": round(w_lat, 2),
                "stats": w_stats,
            },
            "bpe_greedy": {
                "text": g_b.full_text,
                "latency_ms_per_tok": round(b_lat, 2),
                "stats": b_stats,
            },
            "bpe_sampling_t07": {
                "text": g_b_samp.full_text,
            }
        })

    # Save scientific validation report artifact
    audit_data = {
        "contamination_audit": contamination_res,
        "word_baseline": {
            "train": word_train_eval,
            "val": word_val_eval,
        },
        "bpe_subword": {
            "train": bpe_train_eval,
            "val": bpe_val_eval,
        },
        "qualitative_evaluation": qual_results,
    }

    out_path = REPO_ROOT / "checkpoints" / "scientific_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nFull scientific results saved to: {out_path}")


if __name__ == "__main__":
    main()
