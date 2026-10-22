# Technical Audit: Small Language Model (SLM) Repository

**Repository:** `https://github.com/atleekumaar/Built-my-own-small-language-model`  
**Date of Audit:** October 2026  
**Auditor Role:** Senior ML Engineer  
**Objective:** Technical evaluation and readiness audit to convert the existing notebook-based Keras/JAX SLM project into a reproducible, testable model package suitable for productization and an interactive language model playground.

---

## 1. Executive Summary

This repository represents an educational Small Language Model (SLM) project derived from Google DeepMind's "AI Research Foundations" curriculum (Course 1, Lab 1.5: *Train Your Own Small Language Model*). The model is a 1-layer decoder-only causal Transformer trained on the **Africa Galore** dataset (232 paragraphs) using Keras 3 with the JAX backend.

### Key Audit Findings:
1. **Model & Architecture:** The model consists of 3,358,094 parameters (~3.36M, documented as ~3.5M). It uses a custom whitespace tokenizer with 5,262 tokens, sinusoidal positional embeddings (dim 256), a single causal multi-head self-attention block (2 heads, key dimension 256), a 2-layer MLP (hidden dim 256, ReLU activation), and a dense projection to the vocabulary.
2. **Missing Checkpoint Artifacts:** **No trained model weights, checkpoints, or exported model files exist in the git repository.** The notebook outputs show a completed 200-epoch training run in an ephemeral Google Colab environment, but no `model.save()`, `model.save_weights()`, or artifact upload was performed.
3. **Upstream Code vs. Independent Implementation:** The foundational model building blocks (`TokenAndPositionEmbedding`, `TransformerBlock`, `MultiHeadSelfAttention`, `FeedForwardNetwork`), the masked loss (`CustomMaskPadLoss`), and generation utilities (`generate_text`) were imported from the upstream external package `google-deepmind/ai-foundations` (`git+https://github.com/google-deepmind/ai-foundations.git@main`). The independent contributions in this repository are the parameter configuration, min/max sequence length calculation, student prompt experiments, and the repository documentation (`README.md`).
4. **Data & Evaluation Methodology:** The model was trained on the entire Africa Galore dataset without a train/validation/test split. There are no held-out validation metrics, no perplexity measurements, and no checkpoint format defined.
5. **Licensing:** The upstream code from Google DeepMind is licensed under the Apache License 2.0 (Copyright 2026 Google LLC). Any extraction or redistribution of derivative components must preserve Apache 2.0 attribution notices and indicate modifications.

---

## 2. Existing Repository Structure

```text
Built-my-own-small-language-model/
├── .git/
├── README.md                                          # High-level overview and Colab quick-start badge
└── training_Atlee's_own_small_language_model.ipynb   # 63-cell Colab notebook containing execution history
```

### Git History
- `fd01816` (Mon Sep 21 16:01:00 2026): *Update project title in README.md*
- `2278d5a` (Mon Sep 21 03:05:46 2026): *Create README for Small Language Model project*
- `719268c` (Mon Sep 21 03:01:35 2026): *Add files via upload* (committed initial notebook)

---

## 3. Architecture & Data Pipeline Audit

### 3.1 Data Pipeline
- **Raw Dataset:** `africa_galore.json` downloaded from `https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json`.
- **Corpus Volume:** Exactly 232 paragraphs, totaling 19,065 words/tokens.
- **Paragraph Lengths:** Minimum sequence length = 26 tokens; Maximum sequence length = 318 tokens; First paragraph = 118 tokens.
- **Tokenizer Class:** `SimpleWordTokenizer` implemented inside notebook cell 17.
  - Splitting rule: Regular expression `re.split(" +", text)` (whitespace-delimited).
  - Vocabulary creation: `sorted(list(set(tokens)))`.
  - Special tokens: `<PAD>` inserted at index `0`, `<UNK>` appended at index `len(vocabulary) - 1`.
  - Joined representation: `" ".join(tokens)`.
  - Total vocabulary size: 5,262 tokens (5,260 unique corpus words + 2 special tokens).
- **Padding & Truncation:**
  - Pad sequence length target: `max_length = 300` tokens (`padding="post"`, `truncating="post"`, pad value = `0`).
- **Autoregressive Shift (Inputs vs. Targets):**
  - Inputs: `input_sequences = padded_sequences[:, :-1]` (shape: `(232, 299)`).
  - Targets: `target_sequences = padded_sequences[:, 1:]` (shape: `(232, 299)`).
  - Effective sequence length for model: `max_length = 299`.
- **Batching & Shuffling:**
  - `tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))`
  - `shuffle(buffer_size=232)`
  - `batch(batch_size=32)`
  - Total batches per epoch: 8 batches (7 batches of 32 examples, 1 final batch of 8 examples).

### 3.2 Model Architecture Breakdown
The model is constructed via `ai_foundations.training.create_model(...)` in cell 46:

| Component | Layer Type | Parameters | Dimensions / Config |
| :--- | :--- | :--- | :--- |
| **Input** | `keras.layers.Input` | 0 | `(batch_size, 299)`, `dtype="int32"` |
| **Embedding** | `TokenAndPositionEmbedding` | 1,347,072 | Token Embedding: `(5262, 256)` with `mask_zero=True`<br>Positional: Sinusoidal encoding, dim 256, scale $\sqrt{256}$ |
| **Transformer Block** | `TransformerBlock` (1 block) | 658,688 | Stack count: 1 block |
| ↳ *Self-Attention* | `MultiHeadSelfAttention` | 263,424 | 2 attention heads, `key_dim=256`, `use_causal_mask=True`, `dropout=0.0`, `LayerNorm(256)` |
| ↳ *FeedForward* | `FeedForwardNetwork` | 132,096 | Dense 1: `(256 -> 256, relu)`, Dense 2: `(256 -> 256)`, `dropout=0.0`, `LayerNorm(256)` |
| **Output Logits** | `keras.layers.Dense` | 1,352,334 | Linear projection: `(256 -> 5262)` (includes 5,262 bias terms) |
| **Total Model** | Functional Keras Model | **3,358,094** | 100% Trainable (~3.36M params; 12.81 MB float32 weights) |

---

## 4. Verified vs. Unverified Specifications

| Specification | Status | Value / Finding | Source |
| :--- | :--- | :--- | :--- |
| **Model Framework** | **Verified** | Keras 3 with JAX backend (`KERAS_BACKEND="jax"`) | Notebook cell 13 |
| **Architecture** | **Verified** | 1-block Causal Transformer Decoder with sinusoidal position encodings | `upstream_ai_foundations_training_model.py` |
| **Total Parameters** | **Verified** | 3,358,094 trainable parameters | Exact calculation from layer weights |
| **Tokenizer Type** | **Verified** | Simple word-level whitespace regex tokenizer (`re.split(" +", text)`) | Notebook cell 17 |
| **Vocabulary Size** | **Verified** | 5,262 tokens (5,260 dataset tokens + `<PAD>` + `<UNK>`) | Dataset inspection & tokenizer build |
| **Maximum Context Length** | **Verified** | 299 tokens (from pad max 300, truncated by 1 for autoregressive shift) | Notebook cell 36 |
| **Embedding Dimension ($d_{model}$)** | **Verified** | 256 | Default argument in `create_model` |
| **Attention Heads** | **Verified** | 2 heads (`key_dim=256`) | Default argument in `create_model` |
| **MLP Intermediate Dimension** | **Verified** | 256 | Default argument in `create_model` |
| **Activation Function** | **Verified** | ReLU (`"relu"`) | Default argument in `create_model` |
| **Dropout Rate** | **Verified** | 0.0 | Default argument in `create_model` |
| **Optimizer** | **Verified** | AdamW (`learning_rate=1e-4`, `weight_decay=0.005`) | Notebook cell 46 & `get_optimizer` |
| **Loss Function** | **Verified** | Masked Sparse Categorical Crossentropy (`CustomMaskPadLoss`, `ignore_class=0`) | `upstream_ai_foundations_training_losses.py` |
| **Training Epochs** | **Verified** | 200 epochs | Notebook cell 50 |
| **Batch Size** | **Verified** | 32 (8 batches per epoch) | Notebook cell 40, 42 |
| **Checkpoint File on Disk** | **Verified Absent** | **No checkpoint exists anywhere in the repository** | Repo file listing & git log |
| **Checkpoint Format** | **Unverified / Unknown** | None specified in notebook (neither `.keras`, `.weights.h5`, nor `.safetensors`) | No serialization code in notebook |
| **Validation Loss / Metrics** | **Unverified / Unknown** | **None** (100% of data used for training, 0 validation samples) | Notebook cell 40 |
| **Held-out Perplexity** | **Unverified / Unknown** | **None recorded** (no train/val/test split) | Notebook cell 40 |

---

## 5. Recorded Training Logs & Generated Examples

All figures in this section are extracted directly from the execution state persisted in `training_Atlee's_own_small_language_model.ipynb`.

### 5.1 Training Loss Progression
- **Epoch 1:** Loss = `0.0212` (8 batches, ~49 ms/step)
- **Epoch 10:** Loss = `0.0208`
- **Epoch 20:** Loss = `0.0203`
- **Epoch 50:** Loss = `0.0191`
- **Epoch 80:** Loss = `0.0181`
- **Epoch 100:** Loss = `0.0178`
- **Epoch 108 (Spike):** Loss = `0.0736` (loss anomaly observed during training)
- **Epoch 110:** Loss = `0.0371`
- **Epoch 150:** Loss = `0.0167`
- **Epoch 180:** Loss = `0.0161`
- **Epoch 200 (Final):** Loss = `0.0159`

*(Note on loss magnitude: In `CustomMaskPadLoss`, reduction is `"sum_over_batch_size"`. Because padding tokens are masked out and not accumulated, the scalar loss appears very low, ~0.016-0.021, rather than per-token cross entropy).*

### 5.2 Recorded Callback Generations (Prompt: `"Abeni,"`)
- **Epoch 10:** `"Abeni, a bright-eyed girl with braids adorned with colorful beads, lived"`
- **Epoch 20:** `"Abeni, a fatty made beverage for exploration, had heard tales of"`
- **Epoch 30:** `"Abeni, a bright-eyed girl with braids adorned with colorful beads, lived"`
- **Epoch 90:** `"Abeni, a bright-eyed girl with braids adorned with colorful rich street"`
- **Epoch 200:** `"Abeni, a bright-eyed girl with braids adorned with colorful beads, lived"`

### 5.3 Recorded Evaluation Cell Outputs
1. **Prompt 1 (Student Custom Prompt):** `" ATLEE KUMAAR 2072 "`
   - Sampling Mode: `greedy`
   - Generated: ` <UNK> <UNK> <UNK>  windows of the to the most spectacular waterfalls, straddles the`
   - Top 3 candidate probabilities:
     - `'windows'`: $0.0123$
     - `'meat.'`: $0.0116$
     - `'tallest'`: $0.0075$
   - Observation: All prompt tokens were out-of-vocabulary (`<UNK>`). The model degenerated into low-probability ungrounded completions.

2. **Prompt 2:** `"Jide was hungry so she went looking for"`
   - Sampling Mode: `random`
   - Generated: `"Jide was hungry so she went looking for a quaint tea and rich, deep Nelion, are enjoying the"`
   - Top candidate probability: `'a'`: $0.9806$

3. **Prompt 3:** `"Jide was thirsty so she went looking for"`
   - Sampling Mode: `random`
   - Generated: `"Jide was thirsty so she went looking for a long day of framework for electric. The shrimp in"`
   - Top candidate probability: `'a'`: $0.9797$

---

## 6. Upstream Provenance & Licensing Constraints

### 6.1 Upstream Origin
The notebook was published as part of Google DeepMind's educational repository:
- **Upstream Repository:** `https://github.com/google-deepmind/ai-foundations`
- **Course & Lab:** Course 1, Lab 1.5 (`course_1/gdm_lab_1_5_train_your_own_small_language_model.ipynb`)
- **Upstream Modules:**
  - `ai_foundations.training.create_model`
  - `ai_foundations.training.callbacks.TextGenerator`
  - `ai_foundations.training.losses.CustomMaskPadLoss`
  - `ai_foundations.transformers.layers.TokenAndPositionEmbedding`
  - `ai_foundations.transformers.layers.TransformerBlock`
  - `ai_foundations.generation.generate.generate_text`
  - `ai_foundations.feedback.course_1.slm`

### 6.2 Licensing Analysis
- Upstream repo is governed by the **Apache License, Version 2.0** (January 2004).
- **Compliance Requirements:**
  - Section 4(a): Must provide recipients a copy of the Apache 2.0 license.
  - Section 4(b): Modified files must carry prominent notices stating that files were modified.
  - Section 4(c): All copyright notices (`Copyright 2026 Google LLC`) and attribution must be retained in source forms.
  - Section 7 & 8: Disclaimer of Warranty and Limitation of Liability apply.

---

## 7. Dependency, Portability & Reproduction Blockers

### Blocker 1: No Trained Weights Committed
- **Impact: Critical Blocker.**
- Because the training occurred on Google Colab without saving weights to Google Drive or committing them to git, no checkpoint exists to load for instant inference.
- **Resolution Strategy:** The package must support loading existing weights if present, and provide a deterministic, reproducible local training script matching the exact notebook training procedure (seed 812, AdamW lr=1e-4, 200 epochs, Africa Galore dataset) that outputs a standard `.keras` and `.weights.h5` checkpoint file.

### Blocker 2: Dependency on External Git Package `ai_foundations`
- **Impact: High Portability Risk.**
- Installing directly from `git+https://github.com/google-deepmind/ai-foundations.git@main` introduces build fragility, version drift, and dependency conflicts on local environments (especially Windows / Python 3.14).
- **Resolution Strategy:** Vendor and modularize the necessary model layers (`TokenAndPositionEmbedding`, `TransformerBlock`), loss function, and generation routine directly within an internal, self-contained `slm` package with full Apache 2.0 attribution notices.

### Blocker 3: Tokenizer Fragility
- **Impact: Medium Reliability Risk.**
- The `SimpleWordTokenizer` relies on simple whitespace splitting (`re.split(" +", text)`). It leaves attached punctuation as part of the token (e.g. `'Nelion,'`, `'meat.'`). Any word with different punctuation maps to `<UNK>`.
- The tokenizer must be saved and loaded deterministically (vocabulary JSON).

### Blocker 4: Dataset Availability & Offline Independence
- **Impact: Low-Medium Risk.**
- The dataset is fetched dynamically via HTTP from Google Cloud Storage (`dm-educational/assets/ai_foundations/africa_galore.json`). If the URL becomes unavailable or network access is offline, the pipeline fails.
- **Resolution Strategy:** Cache the dataset locally in `data/africa_galore.json` for offline reproducibility.

---

## 8. Prioritized Fixes & Implementation Plan

| Priority | Action Item | Target Location | Description |
| :--- | :--- | :--- | :--- |
| **P0** | **Vendor Model & Layers** | `slm/model.py`, `slm/layers.py` | Create pure Keras 3 modules with JAX backend compatibility and Apache 2.0 headers. |
| **P0** | **Modular Tokenizer** | `slm/tokenizer.py` | Decouple `SimpleWordTokenizer`, support serialization/deserialization to `vocab.json`. |
| **P0** | **Configuration Module** | `slm/config.py` | Formalize verified architecture hyperparameters in a structured dataclass / JSON config. |
| **P0** | **Inference & Generation** | `slm/generate.py` | Robust autoregressive generator supporting greedy & random sampling, temperature, and input validation. |
| **P1** | **Deterministic Training / Checkpoint Pipeline** | `scripts/train.py` | Reproducible script to generate the missing `.keras` / `.weights.h5` checkpoint matching notebook seed `812`. |
| **P1** | **Unit & Round-Trip Tests** | `tests/test_tokenizer.py`, `tests/test_model.py` | Validate tokenizer encoding/decoding, special tokens, model forward pass, shapes, and generation. |
| **P2** | **Latency & Throughput Benchmark** | `benchmarks/benchmark_inference.py` | Measure CPU/GPU warmup latency, per-token latency, and generation throughput. |
