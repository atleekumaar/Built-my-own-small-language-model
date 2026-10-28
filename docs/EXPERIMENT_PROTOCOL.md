# ATLEE SLM LAB 2.0 — Controlled ML Experiment Protocol & Scientific Benchmark

**Author:** Atlee Kumar
**Auditor Role:** Senior ML Engineer & Applied AI Researcher
**Framework:** Keras 3 (v3.15.1) with JAX (v0.7.1) CPU Backend
**Corpus:** Africa Galore (232 paragraphs, 19,065 words)
**Document Classification:** Empirical Research Protocol & Benchmark Reconciliation

---

## 1. Research Question & Theoretical Motivation

> **Primary Research Question:**
> *How does word-level tokenization compare with byte-level Byte-Pair Encoding (BPE) in vocabulary coverage, model footprint, and inference efficiency for a resource-constrained Small Language Model trained on limited domain data?*

In extreme low-data regimes ($< 250$ training paragraphs), conventional natural language processing techniques encounter severe trade-offs:
1. **Word-Level Tokenization (Coarse Granularity):** Assigns one token per whitespace-delimited word. While sequence lengths are short (averaging ~88 words per paragraph), the vocabulary grows rapidly and fails to generalize to held-out texts, causing catastrophic Out-Of-Vocabulary (`<UNK>`) failure.
2. **Byte-Level Subword BPE (Fine Granularity):** Tokenizes text into character and byte-level subwords. While sequence lengths increase (averaging ~159 subwords per paragraph), the vocabulary is compact (2,000 tokens), parameter counts in the embedding and projection layers shrink by $>40\%$, and out-of-vocabulary failures are mathematically eliminated ($0.00\%$ OOV rate).

This protocol defines the experimental methodology used to compare these two paradigms on the **Africa Galore** corpus.

---

## 2. Dataset Specification & Leakage Prevention

### 2.1 Corpus Volume & Partitioning
- **Total Corpus:** 232 paragraphs, 19,065 words, 138,889 bytes.
- **Partition Strategy:** Deterministic pseudo-random shuffle with fixed seed (`seed=42`).
  - **Training Split:** Exactly 185 paragraphs ($79.74\%$).
  - **Validation Split:** Exactly 47 held-out paragraphs ($20.26\%$).
- **Partition Disjointness:** Verified via `scripts/create_splits.py`:
  - Index overlap: $0$ indices ($\text{Train} \cap \text{Val} = \emptyset$).
  - Exact string paragraph overlap: $0$ duplicate paragraphs.
  - Maximum 5-gram Jaccard similarity across train and val: $0.1132$ (confirming absence of near-duplicate borrowing).

### 2.2 Strict Tokenizer Fitting (Zero Leakage)
To guarantee experimental integrity:
- `SimpleWordTokenizer` was fitted **exclusively on the 185 training paragraphs**, yielding a 4,409-token vocabulary (`data/vocab_baseline.json`). Zero validation tokens leaked into the dictionary.
- `BPETokenizer` was trained **exclusively on the 185 training paragraphs** using HuggingFace Byte-Level BPE (`target_vocab=2000`, `min_frequency=2`), yielding a 2,000-token vocabulary (`checkpoints/bpe_tokenizer.json`).
- Competing models evaluate the **exact same 47 validation documents**, representing **26,159 UTF-8 bytes**.

---

## 3. Controlled Experimental Architecture & Training Protocol

### 3.1 Model Architecture Matrix
Both models share an identical 1-block causal Transformer decoder architecture, differing only in vocabulary-dependent layers:

| Layer Component | Word Baseline (`slm_baseline_val`) | Byte-Level BPE (`slm_bpe_val`) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Tokenizer Type** | Whitespace Word Regex | Byte-Level BPE (GPT-2 style) | Subword vs Word |
| **Vocabulary Size ($V$)** | 4,409 tokens | 2,000 tokens | **-54.64%** |
| **Context Window ($L$)** | 299 effective tokens | 299 effective tokens | Identical |
| **Embedding Dimension ($d_{model}$)** | 256 | 256 | Identical |
| **Attention Heads / Key Dim** | 2 heads / 256 dim | 2 heads / 256 dim | Identical |
| **Transformer Blocks** | 1 block | 1 block | Identical |
| **MLP Hidden Dimension** | 256 (ReLU activation) | 256 (ReLU activation) | Identical |
| **Token & Position Embedding Params** | 1,128,704 | 512,000 | -54.64% |
| **Transformer Block Params** | 658,688 | 658,688 | Identical |
| **Dense Output Logits Params** | 1,133,113 | 514,000 | -54.64% |
| **Total Trainable Parameters** | **2,920,505 (~2.92M)** | **1,684,688 (~1.68M)** | **-42.31% parameter reduction** |
| **Serialized Weights File Size** | 35.12 MB | 20.29 MB | **-42.23% disk footprint** |

### 3.2 Training Budget & Hyperparameters
- **Optimizer:** AdamW (`learning_rate=1e-4`, `weight_decay=0.005`).
- **Loss Function:** `CustomMaskPadLoss` (Masked Sparse Categorical Cross-Entropy, ignoring `pad_token_id=0`).
- **Autoregressive Shifting:** Input sequences `padded[:, :-1]`, Target sequences `padded[:, 1:]`.
- **Batch Size:** 32 examples per batch.
- **Training Epochs:** 30 epochs for controlled comparative evaluation.
- **Random Seed:** Fixed seed `seed=812` for weight initialization and dataset shuffling.

---

## 4. Empirical Benchmark Matrix & Cross-Tokenizer Normalization

All evaluation measurements were executed on a 12-core CPU using the Keras 3 **JAX CPU backend**.

```text
Evaluating Word Model on Held-out Val Split (47 paras)...
Evaluating BPE Model on Held-out Val Split (47 paras)...
```

| Benchmark Metric | Word Baseline (`slm_baseline_val`) | Byte-Level BPE (`slm_bpe_val`) | Scientific Finding |
| :--- | :---: | :---: | :--- |
| **Held-Out Document Count** | 47 paragraphs | 47 paragraphs | Identical evaluation set |
| **Held-Out Raw UTF-8 Bytes** | 26,159 bytes | 26,159 bytes | Identical byte denominator |
| **Held-Out Evaluated Tokens** | 4,152 tokens | 7,475 tokens | BPE uses 1.80x more tokens |
| **Token Granularity** | 6.30 bytes / token | 3.50 bytes / token | Fundamental scale difference |
| **Held-Out `<UNK>` Tokens** | **944 tokens** | **0 tokens** | BPE eliminates OOV completely |
| **Held-Out Out-Of-Vocabulary Rate** | **22.74%** | **0.00%** | Word model fails on 1 in 4.4 tokens |
| **Val Cross-Entropy Loss** | 6.9738 nats | 5.9727 nats | Token-level cross-entropy |
| **Val Perplexity ($PPL = \exp(CE)$)** | **1,068.32** *(Coarse words)* | **392.56** *(Fine subwords)* | **NON-COMPARABLE across tokenizers** |
| **Raw Bits Per Byte (BPB)** | **1.5969 bits/byte** | **2.4623 bits/byte** | Provisional (Word omits UNK characters) |
| **Open-Vocab Adjusted BPB** | **2.1602 bits/byte** | **2.4623 bits/byte** | Word model penalized for 944 UNKs |
| **CPU Inference Throughput** | 27.94 tokens / sec | **34.88 tokens / sec** | **1.25x speedup in token generation** |
| **CPU Median Latency** | 35.07 ms / token | **27.83 ms / token** | **1.26x faster step latency** |
| **CPU P95 Latency** | 42.10 ms / token | **32.50 ms / token** | **1.30x lower tail latency** |

---

## 5. Scientific Findings & Metric Reconciliation

### 5.1 Why Perplexity Cannot Be Compared Across Tokenizers
A common pitfall in machine learning evaluation is comparing perplexity values between models with different tokenizers. Perplexity is defined as:
$$\text{PPL} = \exp\left(\frac{1}{N} \sum_{i=1}^N -\ln P(w_i \mid w_{<i})\right)$$
Because $N$ is 4,152 tokens for the word model and 7,475 tokens for the BPE model on the exact same 26,159 bytes of text, the per-token perplexity assesses fundamentally different token granularities:
- Each word token represents on average **6.30 bytes**.
- Each BPE subword represents on average **3.50 bytes**.

Therefore, reporting that the BPE model has a perplexity of 392.56 versus 1,068.32 for the Word model does **NOT** mean BPE is $63\%$ better in predicting language; it reflects the differing entropy of word vs. subword distributions.

### 5.2 The `<UNK>` Evaluation Paradox & Adjusted BPB
The raw Bits-Per-Byte formula evaluates:
$$\text{Raw BPB} = \frac{\sum \text{nats} / \ln(2)}{\text{Total UTF-8 Bytes}}$$
On raw BPB, the Word model appears to achieve 1.5969 bits/byte compared to BPE's 2.4623 bits/byte. However, this is an artifact of the `<UNK>` paradox:
1. When the Word model encounters an unseen word in the held-out validation set (which occurs for $22.74\%$ of all tokens), it is only required to predict the single token class `1` (`<UNK>`).
2. The Word model is **never required to predict the characters that compose the unknown word**.
3. In contrast, the Byte-Level BPE model must predict every single byte and subword of that unseen word, incurring non-zero entropy for every constituent character.

To establish fair open-vocabulary accounting:
- If a word is unknown, an open-vocabulary language model must identify that word from the remainder of the English/African vocabulary.
- Estimating a modest vocabulary of 50,000 words not present in the 4,409 training vocabulary requires an additional:
$$\log_2(50,000) \approx 15.61\text{ bits per UNK token}$$
- Adding this entropy penalty to the 944 `<UNK>` tokens increases the Word model's adjusted requirement to **2.1602 bits/byte**.

### 5.3 5-Fold Cross-Validation Stability Analysis
To verify whether the $22.74\%$ OOV rate was an anomaly of the specific split seed, a 5-fold cross-validation analysis was executed across the entire 232-paragraph corpus (`scripts/cross_validation.py`):

| Fold Index | Train Paras | Val Paras | Word Vocab Size | Word OOV Rate (%) | BPE OOV Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 186 | 46 | 4,374 words | 22.18% | 0.00% |
| **Fold 2** | 186 | 46 | 4,389 words | 20.89% | 0.00% |
| **Fold 3** | 186 | 46 | 4,395 words | 23.51% | 0.00% |
| **Fold 4** | 186 | 46 | 4,412 words | 19.12% | 0.00% |
| **Fold 5** | 186 | 46 | 4,438 words | 19.16% | 0.00% |
| **Summary** | **186** | **46** | **4,401.6 ± 24.3** | **20.97% ± 1.84%** | **0.00% ± 0.00%** |

**Conclusion:** The word-level model consistently exhibits a $\sim 21\%$ out-of-vocabulary rate across arbitrary held-out subsets of the corpus, confirming that whitespace word tokenization is mathematically unviable for generalization on small corpora.

---

## 6. Qualitative Generation & Generative Loop Diagnostics

Due to the limited volume of the training data (185 paragraphs), both models exhibit cyclic phrase repetition under greedy decoding ($T=0$):

```text
Prompt: 'Kilimanjaro is'
[Word Greedy]: 'Kilimanjaro is a traditional and a traditional and a traditional and a traditional...'
[BPE Greedy]:  'Kilimanjaro is a a a a a a is a a a a a the the the the the the the the the...'
[BPE T=0.7]:   'Kilimanjaro is a a consistoconut life,a, and food co preing, fizz, and set...'
```

### Analysis of Generative Loops
1. **Low Support in Causal Horizon:** In a corpus of 185 paragraphs, phrases like `"is a traditional"` and `"and the"` appear with high transition probabilities across disparate contexts. Once greedy decoding enters a high-probability bigram loop, the single self-attention head lacks the context horizon to break the cycle.
2. **Temperature Sampling ($T=0.7$):** Temperature scaling introduces entropy, breaking cyclic bigram repetition, but produces lexical hallucinations because the 1.68M model has not mastered complex multi-token grammatical agreement on 185 paragraphs.
3. **Research Implication:** Future work will integrate presence and frequency penalties (Keskar et al., CTRL) directly into the autoregressive sampler to suppress repeated n-grams.

---

## 7. Machine-Readable Schema & Organization

All experiment outputs are preserved in `experiments/results/`:
- `experiments/results/scientific_validation_results.json`: Complete Cross-Tokenizer Evaluation Matrix, held-out validation statistics, and sample prompt generations.
- `experiments/results/benchmark_rigorous_results.json`: P95 latency, mean, std, median, and throughput for both models.
- `experiments/results/cross_validation_results.json`: 5-fold cross-validation metrics and OOV variance.
- `experiments/results/experiment_comparison.json`: Training duration, parameter counts, and validation loss curves.
- `experiments/manifests/artifact_manifest.json`: Cryptographic SHA-256 hashes for all 18 artifacts.

---

## 8. Exact Commands for Reproduction

### 1. Run the Full Test Suite
```bash
py -3.12 -m pytest tests/ -v
```

### 2. Generate Deterministic Splits
```bash
py -3.12 scripts/create_splits.py
```

### 3. Execute Controlled Training Experiments
```bash
py -3.12 scripts/train_experiments.py --epochs 30 --batch-size 32 --bpe-vocab 2000
```

### 4. Execute the Scientific Audit & BPB Reconciliation
```bash
py -3.12 scripts/scientific_audit.py
```

### 5. Execute Rigorous CPU Inference Benchmarking
```bash
py -3.12 scripts/benchmark_comparison.py
```

### 6. Execute 5-Fold Cross-Validation Analysis
```bash
py -3.12 scripts/cross_validation.py
```

### 7. Re-generate Cryptographic Manifest
```bash
py -3.12 scripts/generate_manifest.py
```
