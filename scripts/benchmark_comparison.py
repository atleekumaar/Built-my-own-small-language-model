"""
Rigorous CPU Inference Benchmark for Word Baseline vs. Byte-Level BPE.
Uses warm-up passes (JIT exclusion), 5 repetitions x 4 prompts, and reports median, mean, std, and throughput.
"""

import json
import math
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np

from slm.bpe_tokenizer import BPETokenizer
from slm.config import ModelConfig
from slm.generate import generate_text
from slm.model import build_model, load_checkpoint
from slm.tokenizer import SimpleWordTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def benchmark_model(model, tokenizer, model_name, prompts, tokens_per_prompt=20, num_runs=5, warmup_runs=2):
    print(f"\n--- Benchmarking {model_name} ---")
    print(f"Warm-up iterations (JIT compile exclusion): {warmup_runs}")
    for _ in range(warmup_runs):
        _ = generate_text("Warmup prompt for JIT", n_tokens=5, model=model, tokenizer=tokenizer, sampling_mode="greedy")

    latencies_ms = []
    total_tokens = 0
    start_total_t = time.perf_counter()

    for r in range(num_runs):
        for p in prompts:
            t0 = time.perf_counter()
            res = generate_text(p, n_tokens=tokens_per_prompt, model=model, tokenizer=tokenizer, sampling_mode="greedy")
            elapsed = time.perf_counter() - t0
            n_gen = len(res.generated_token_ids)
            total_tokens += n_gen
            latencies_ms.append((elapsed / n_gen) * 1000.0)

    total_time = time.perf_counter() - start_total_t
    latencies = np.array(latencies_ms)
    throughput = total_tokens / total_time if total_time > 0 else 0.0

    def to_py(v):
        return float(str(round(float(v), 2)))

    metrics = {
        "model_name": model_name,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "parameters": int(model.count_params()),
        "total_tokens_generated": int(total_tokens),
        "total_time_seconds": to_py(total_time),
        "mean_latency_ms": to_py(np.mean(latencies)),
        "std_latency_ms": to_py(np.std(latencies)),
        "median_latency_ms": to_py(np.median(latencies)),
        "p95_latency_ms": to_py(np.percentile(latencies, 95)),
        "min_latency_ms": to_py(np.min(latencies)),
        "max_latency_ms": to_py(np.max(latencies)),
        "throughput_tok_per_sec": to_py(throughput),
    }

    print(f"Results for {model_name}:")
    print(f"  Median Latency: {metrics['median_latency_ms']} ms/token (Mean: {metrics['mean_latency_ms']} +/- {metrics['std_latency_ms']} ms)")
    print(f"  P95 Latency:    {metrics['p95_latency_ms']} ms/token")
    print(f"  Throughput:     {metrics['throughput_tok_per_sec']} tokens/sec")
    return metrics


def main():
    prompts = [
        "Abeni visited",
        "The Sahara desert is",
        "Ancient civilizations of",
        "Wildlife in Africa",
    ]

    # Load Word Baseline
    w_cfg = ModelConfig.from_json(REPO_ROOT / "configs" / "model_config_baseline.json")
    w_model = build_model(w_cfg)
    load_checkpoint(w_model, REPO_ROOT / "checkpoints" / "slm_baseline_val.weights.h5")
    w_tok = SimpleWordTokenizer.from_vocabulary_file(REPO_ROOT / "data" / "vocab_baseline.json")

    # Load BPE Model
    b_cfg = ModelConfig.from_json(REPO_ROOT / "configs" / "model_config_bpe.json")
    b_model = build_model(b_cfg)
    load_checkpoint(b_model, REPO_ROOT / "checkpoints" / "slm_bpe_val.weights.h5")
    b_tok = BPETokenizer.from_file(REPO_ROOT / "checkpoints" / "bpe_tokenizer.json")

    w_res = benchmark_model(w_model, w_tok, "slm_baseline_val (Word)", prompts, tokens_per_prompt=20, num_runs=5, warmup_runs=2)
    b_res = benchmark_model(b_model, b_tok, "slm_bpe_val (Byte-Level BPE)", prompts, tokens_per_prompt=20, num_runs=5, warmup_runs=2)

    speedup_throughput = b_res["throughput_tok_per_sec"] / w_res["throughput_tok_per_sec"] if w_res["throughput_tok_per_sec"] > 0 else 0
    speedup_latency = w_res["median_latency_ms"] / b_res["median_latency_ms"] if b_res["median_latency_ms"] > 0 else 0

    print("\n" + "=" * 65)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 65)
    print(f"Word Baseline:  Median {w_res['median_latency_ms']} ms/tok | Throughput: {w_res['throughput_tok_per_sec']} tok/s")
    print(f"Byte-Level BPE: Median {b_res['median_latency_ms']} ms/tok | Throughput: {b_res['throughput_tok_per_sec']} tok/s")
    print(f"Measured Speedup (Throughput): {speedup_throughput:.2f}x")
    print(f"Measured Speedup (Latency):    {speedup_latency:.2f}x")

    bench_output = {
        "word_baseline": w_res,
        "bpe_subword": b_res,
        "speedup_throughput_ratio": float(str(round(float(speedup_throughput), 2))),
        "speedup_latency_ratio": float(str(round(float(speedup_latency), 2))),
        "benchmark_protocol": {
            "warmup_runs": 2,
            "measured_runs": 5,
            "prompts_count": len(prompts),
            "tokens_per_prompt": 20,
            "total_tokens_per_model": 400,
            "environment": "CPU (JAX backend), Intel Core i5-8250U, Windows 11",
        }
    }

    out_path = REPO_ROOT / "checkpoints" / "benchmark_rigorous_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bench_output, f, indent=2)
    print(f"Benchmark results saved to: {out_path}")


if __name__ == "__main__":
    main()
