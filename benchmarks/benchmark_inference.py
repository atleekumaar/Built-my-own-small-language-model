"""Inference latency and throughput benchmark script for the Small Language Model."""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Sequence

# Add repository root to python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["KERAS_BACKEND"] = "jax"
import numpy as np

from slm.config import ModelConfig
from slm.generate import generate_text
from slm.model import build_model, load_checkpoint
from slm.tokenizer import SimpleWordTokenizer


def run_benchmark(
    model,
    tokenizer,
    prompts: Sequence[str],
    tokens_per_prompt: int = 15,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> dict[str, float]:
    """Run warm-up passes followed by repeated latency and throughput measurements."""
    print("Executing warm-up passes (JIT compilation)...")
    for _ in range(warmup_runs):
        _ = generate_text(
            prompt="Abeni lived in",
            n_tokens=5,
            model=model,
            tokenizer=tokenizer,
            sampling_mode="greedy",
        )

    print(f"Running benchmark: {len(prompts)} prompts x {num_runs} repetitions ({tokens_per_prompt} tokens each)...")
    latencies_per_token: list[float] = []
    total_generated_tokens = 0
    total_time_seconds = 0.0

    for run_idx in range(num_runs):
        for prompt in prompts:
            start_t = time.perf_counter()
            result = generate_text(
                prompt=prompt,
                n_tokens=tokens_per_prompt,
                model=model,
                tokenizer=tokenizer,
                sampling_mode="greedy",
            )
            elapsed = time.perf_counter() - start_t

            token_count = len(result.generated_token_ids)
            total_generated_tokens += token_count
            total_time_seconds += elapsed
            latencies_per_token.append((elapsed / token_count) * 1000.0)  # in ms

    latencies = np.array(latencies_per_token)
    throughput = total_generated_tokens / total_time_seconds

    metrics = {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "throughput_tokens_per_sec": float(throughput),
        "total_tokens_evaluated": total_generated_tokens,
        "total_time_sec": float(total_time_seconds),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SLM Benchmark Script")
    parser.add_argument("--vocab-path", type=str, default="data/vocab.json")
    parser.add_argument("--weights-path", type=str, default="checkpoints/slm_africa_galore.weights.h5")
    parser.add_argument("--tokens", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print("=" * 65)
    print("SLM INFERENCE BENCHMARK")
    print("=" * 65)

    tokenizer = SimpleWordTokenizer.from_vocabulary_file(args.vocab_path)
    config = ModelConfig()
    model = build_model(config)

    if os.path.exists(args.weights_path):
        load_checkpoint(model, args.weights_path)
        print(f"Loaded weights from {args.weights_path}")
    else:
        print("Using initialized weights (architecture baseline benchmark)")

    test_prompts = [
        "Abeni lived in",
        "The Lagos air was thick with humidity",
        "Table Mountain is a famous landmark",
    ]

    metrics = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        tokens_per_prompt=args.tokens,
        num_runs=args.runs,
        warmup_runs=2,
    )

    print("\n" + "-" * 40)
    print("BENCHMARK RESULTS (JAX CPU Backend)")
    print("-" * 40)
    print(f"Total Tokens Generated:    {metrics['total_tokens_evaluated']}")
    print(f"Total Elapsed Time:        {metrics['total_time_sec']:.2f} s")
    print(f"Throughput:                {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Mean Latency per Token:    {metrics['mean_latency_ms']:.2f} ms")
    print(f"Median Latency per Token:  {metrics['median_latency_ms']:.2f} ms")
    print(f"P95 Latency per Token:     {metrics['p95_latency_ms']:.2f} ms")
    print(f"Min / Max Latency:         {metrics['min_latency_ms']:.2f} ms / {metrics['max_latency_ms']:.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
