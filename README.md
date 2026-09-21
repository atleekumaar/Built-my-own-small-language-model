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
├── training_Atlee's_own_small_language_model.ipynb   # Main Google Colab notebook
└── README.md                                          # Project documentation
