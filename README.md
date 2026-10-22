# Built Your Own Small Language Model (SLM)

This repository contains a step-by-step guide and notebook for training a custom Transformer-based Small Language Model (SLM) from scratch using **Keras** and **JAX** on the **Africa Galore** dataset.

---

## 📌 Overview

Large Language Models (LLMs) like Gemma or GPT-4 contain billions of parameters and require massive compute resources. This project focuses on building a **Small Language Model (SLM)** (~3.5 million parameters) to demonstrate the fundamental end-to-end pipeline of preparing sequence data, configuring autoregressive inputs/targets, and training a transformer model.

---

## 🎯 Key Learnings & Features

- **Text Preprocessing & Tokenization:** Custom whitespace-based tokenization (`SimpleWordTokenizer`) with special tokens (`<PAD>` and `<UNK>`).
- **Sequence Padding & Truncation:** Managing varying paragraph lengths using Keras sequence utilities.
- **Data Pipeline:** Shuffling, batching, and dynamic input-target sequence generation (shifting tokens left for next-token prediction).
- **Model Training:** Training an autoregressive Transformer model powered by Keras and the JAX backend.
- **Generation & Evaluation:** Prompting the trained model and inspecting token output probabilities.

---

## 🛠️ Tech Stack & Requirements

- **Python 3.x**
- **Frameworks & Libraries:**
  - `keras` (JAX Backend)
  - `jax` / `jaxlib`
  - `tensorflow` (for dataset batching & shuffling)
  - `pandas` (for loading raw data)
  - `ai_foundations` library

---

## 🚀 Quick Start

### 1. Open in Google Colab

Run the full notebook directly in Google Colab with GPU acceleration:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1huNVuV63-3FR54VNFqxEOefTEM1J-p4d#scrollTo=HiJciyOo5gYU)

---

## 📊 Dataset

The model is trained on the **Africa Galore** dataset, consisting of short descriptive paragraphs:
- **Total Paragraphs:** 232
- **Shortest Paragraph:** 26 tokens
- **Longest Paragraph:** 318 tokens
- **Default Max Sequence Length:** 300 tokens

---
## 📂 Project Structure

```text
.
├── training_Atlee's_own_small_language_model.ipynb   # Original training notebook (preserved)
├── README.md                                          # Project documentation & guide
├── pytest.ini                                         # Pytest configuration
├── configs/
│   └── model_config.json                             # Canonical model architecture & hyperparams
├── data/
│   ├── africa_galore.json                            # 232 paragraphs training corpus
│   └── vocab.json                                    # 5,262-word vocabulary mapping
├── checkpoints/
│   ├── training_history.json                         # 30-epoch training loss trajectory
│   ├── slm_africa_galore.weights.h5                  # Reproducible weights (git-ignored)
│   └── slm_africa_galore.keras                       # Keras v3 bundle (git-ignored)
├── docs/
│   ├── TECHNICAL_AUDIT.md                            # Comprehensive architecture & layer audit
│   ├── CHECKPOINT_PROCEDURE.md                       # Weight serialization & loading guide
│   └── DESIGN_SPEC.md                                # ATLEE SLM LAB UI/UX design specification
├── slm/
│   ├── __init__.py                                   # Package exports
│   ├── config.py                                     # ModelConfig dataclass
│   ├── tokenizer.py                                  # SimpleWordTokenizer implementation
│   ├── layers.py                                     # Keras 3 TokenAndPositionEmbedding & TransformerBlock
│   ├── losses.py                                     # CustomMaskPadLoss (<PAD>=0 masking)
│   ├── model.py                                      # Model builder and checkpoint loaders
│   ├── generate.py                                   # Autoregressive generation & top-k extraction
│   └── api.py                                        # FastAPI application with static lab serving
├── frontend/
│   ├── index.html                                    # ATLEE SLM LAB single-page application
│   ├── styles.css                                    # Dark research laboratory design system
│   └── app.js                                        # Interactive client logic & API bindings
├── scripts/
│   ├── train.py                                      # Deterministic CPU training script (30 epochs)
│   └── evaluate.py                                   # In-sample cross-entropy & perplexity measurement
├── benchmarks/
│   └── benchmark_inference.py                        # CPU latency and throughput benchmark
└── tests/
    ├── test_tokenizer.py                             # Tokenizer unit tests
    ├── test_model.py                                 # Layer and parameter validation tests
    ├── test_generation.py                            # Generation and top-k probability tests
    └── test_api.py                                   # FastAPI endpoint & lifespan integration tests
```

---

## 🔬 ATLEE SLM LAB (Interactive Playground & Multi-Model Lab)

**ATLEE SLM LAB** is a research-oriented interactive web playground and diagnostic suite for the trained Small Language Models. It connects directly to an authentic FastAPI backend and Keras 3/JAX checkpoints with dynamic multi-model switching.

### Key Capabilities
1. **Multi-Model Architecture Switching:** Seamlessly switch between the original Course Model (`slm_africa_galore`), Word-Level Baseline with split (`slm_baseline_val`), and the new **Byte-Level BPE Model** (`slm_bpe_val`).
2. **Byte-Level BPE Tokenizer (Phase 4):** True subword tokenization with zero `<UNK>` tokens on rare/unseen vocabulary, 42.3% smaller parameter footprint, and 4.7× faster CPU generation.
3. **Real Generation Playground:** Autoregressive text generation with authentic token sampling, temperature scaling, seed reproducibility, latency breakdown, and token throughput.
4. **Next-Token Probability Explorer:** Step-by-step scrubber inspecting the exact top-5 candidate tokens, percentages, and selected token at each forward step.
5. **Tokenizer Inspector:** Word-level and BPE subword decomposition, token IDs, vocabulary lookup, and `<UNK>` token flagging.
6. **Model Architecture Dashboard:** Full parameter accounting, tensor dimensions, embedding matrices, attention head topology, and layer breakdown.
7. **Empirical Evaluation & Experiment Suite:** Direct comparison table in Tab 4 displaying train/val cross-entropy, perplexity on held-out paragraphs, parameter counts, and inference throughput across all models.

---

## ⚡ Quick Start & Run Commands

### 1. Requirements
- Windows 11 / Linux / macOS
- Python 3.10+ (Tested on Python 3.14.5)
- Dependencies: `pip install keras jax jaxlib fastapi uvicorn pytest requests tokenizers`

### 2. Launch the Application (Windows PowerShell)

To start the FastAPI backend and serve ATLEE SLM LAB:

```powershell
# Set JAX to run on CPU backend
$env:KERAS_BACKEND="jax"
$env:JAX_PLATFORMS="cpu"

# Start the uvicorn server (mounts frontend at http://127.0.0.1:8000)
python -m uvicorn slm.api:app --host 127.0.0.1 --port 8000
```

Once running, open your web browser to:
👉 **`http://127.0.0.1:8000/`**

### 3. Run Automated Tests

```powershell
python -m pytest tests/ -v
```
*(Runs 32 unit and regression tests including BPE tokenizer and multi-model API tests)*

### 4. Reproduce Phase 4 Experiments

```powershell
# 1. Create deterministic train (185) / validation (47) paragraph split
python scripts/create_splits.py

# 2. Train Word Baseline and Byte-Level BPE models on train split & evaluate on validation split
python scripts/train_experiments.py
```

### 5. Reproduce Original Model (Phase 1 & 2)

```powershell
# Retrain original single-corpus model for 30 epochs
python scripts/train.py

# In-sample evaluation
python scripts/evaluate.py

# Benchmark CPU latency
python benchmarks/benchmark_inference.py
```

---

## 📡 API Specification

| Endpoint | Method | Description | Request Body / Query |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Serves ATLEE SLM LAB web frontend | None |
| `/health` | `GET` | Health check, device, active model status | None |
| `/models` | `GET` | List available models, architectures, and active status | None |
| `/models/select` | `POST` | Switch active in-memory model and tokenizer | `{"model_id": "slm_bpe_val"}` |
| `/experiments` | `GET` | Machine-readable Phase 4 experiment comparison data | None |
| `/config` | `GET` | Full architecture and hyperparameters of active model | None |
| `/evaluation`| `GET` | Training loss history and metrics of active model | None |
| `/tokenize` | `POST` | Tokenizes input string into IDs and chips | `{"text": "Abeni visited..."}` |
| `/generate` | `POST` | Autoregressive generation & top-k probs | `{"prompt": "Abeni,", "max_new_tokens": 10, "temperature": 0.7, "sampling_mode": "temperature"}` |

---

## 🔬 Phase 5 Scientific Validation & Fair Comparison Summary

For full methodology and analysis, see [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md).

| Metric | Word-Level Baseline (`slm_baseline_val`) | Experimental Byte-Level BPE (`slm_bpe_val`) | Original Course Model (`slm_africa_galore`) |
| :--- | :--- | :--- | :--- |
| **Split** | 185 Train / 47 Val (Seed 42) | 185 Train / 47 Val (Seed 42) | 232 (100% In-sample) |
| **Tokenizer** | Word-Level (Whitespace) | Byte-Level BPE (Subword) | Word-Level (Whitespace) |
| **Vocab Size** | 4,409 words | **2,000 subwords** | 5,262 words |
| **Total Parameters** | 2,920,505 (~2.92M) | **1,684,688 (~1.68M) [-42.3%]** | 3,358,094 (~3.36M) |
| **Checkpoint Size** | 35.12 MB | **20.29 MB [-42.2%]** | 40.37 MB |
| **Val Cross-Entropy** | 6.9738 nats (Coarse words) | **5.9727 nats** (Fine subwords) | 5.7778 nats (In-sample fit) |
| **Val Perplexity (PPL)**| **1,068.32** *(Coarse tokens)* | **392.55** *(Fine tokens)* | 323.04 (In-sample memorization) |
| **Held-Out OOV Rate** | **22.74% (944 UNK tokens)** | **0.00% (Zero UNK)** | 0.00% (Full memorization) |
| **Raw BPB (bits/byte)** | **1.5969** *(Omits UNK words)* | **2.4623** *(Exact byte NLL)* | *N/A (No split)* |
| **Adj. BPB (bits/byte)** | **2.1602** *(50K word prior)* | **2.4623** *(True open-vocab)* | *N/A (No split)* |
| **CPU Throughput** | 27.94 tok/s (Median: 35.1ms) | **34.88 tok/s (Median: 27.8ms)** | 53.39 tok/s |
| **Measured Speedup** | Baseline | **1.25x Throughput (1.26x Latency)**| - |

---

## ⚠️ Scientific Integrity & Evaluation Disclosure

- **Cross-Tokenizer Perplexity Comparability:** Per-token cross-entropy and perplexity ($PPL = \exp(CE)$) evaluate fundamentally different token granularities (6.30 bytes/token for Word vs. 3.50 bytes/token for BPE). Do not conclude that BPE is "63% better" based solely on perplexity.
- **The `<UNK>` Evaluation Paradox:** The Word model's raw Bits Per Byte (1.5969) is an optimistic lower-bound because 22.74% of held-out tokens are assigned to class 1 (`<UNK>`) without predicting the actual constituent characters. Accounting for open-vocabulary unseen word entropy (+15.6 bits/token) raises the Word model's requirement to 2.1602 BPB.
- **Corpus Limitation & Generative Loops:** The entire Africa Galore dataset contains only 232 short paragraphs (185 training paragraphs). Consequently, both models exhibit cyclic phrase repetition under greedy decoding ($T=0$). Temperature sampling ($T=0.7$) introduces token diversity but cannot compensate for the limited training distribution.
- **Held-Out Validation vs. In-Sample Memorization:** Phase 4 & 5 models are evaluated strictly on 47 held-out paragraphs. The original course model's perplexity (323.04) was evaluated in-sample on its own training corpus and does not represent generalization.
- **Zero Simulation:** All outputs in ATLEE SLM LAB are computed in real time from Keras 3 with the JAX CPU backend.

