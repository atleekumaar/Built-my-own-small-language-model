# Model Checkpoint & Serialization Guide

## 1. Checkpoint Status in the Repository
As determined in the technical audit ([`docs/TECHNICAL_AUDIT.md`](file:///c:/Users/atuls/OneDrive/Desktop/my%20slm/docs/TECHNICAL_AUDIT.md)), the original repository contained **no pre-saved model checkpoints or weights files**. The original training was executed inside an interactive Google Colab runtime without persistent checkpoint serialization.

To support productization and reproducible inference without fabricating weights or substituting third-party models, this repository provides:
1. Native Keras 3 / JAX checkpoint loading and saving routines.
2. An exact, deterministic reproduction script (`scripts/train.py`) matching the notebook seed (`812`), hyperparameter settings, and Africa Galore dataset.

---

## 2. Supported Checkpoint Formats

The `slm` package supports two primary checkpoint serialization formats:

### A. Full Model Archive (`.keras`)
- **Format:** Modern Keras zip-archive format containing architecture config, weights, and optimizer state.
- **Saving:**
  ```python
  from slm import save_checkpoint
  save_checkpoint(model, "checkpoints/slm_africa_galore.keras")
  ```
- **Loading:**
  ```python
  from slm import load_model
  model = load_model(weights_path="checkpoints/slm_africa_galore.keras")
  ```

### B. Weights-Only HDF5 Archive (`.weights.h5`)
- **Format:** Lightweight HDF5 archive storing only tensor weight matrices.
- **Saving:**
  ```python
  from slm import save_checkpoint
  save_checkpoint(model, "checkpoints/slm_africa_galore.weights.h5")
  ```
- **Loading into an existing architecture:**
  ```python
  from slm import build_model, load_checkpoint
  model = build_model()
  load_checkpoint(model, "checkpoints/slm_africa_galore.weights.h5")
  ```

---

## 3. End-to-End Loading Procedure

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

from slm import ModelConfig, SimpleWordTokenizer, build_model, generate_text, load_checkpoint

# 1. Load configuration and tokenizer
config = ModelConfig.from_json("configs/model_config.json")
tokenizer = SimpleWordTokenizer.from_vocabulary_file("data/vocab.json")

# 2. Build model architecture
model = build_model(config)

# 3. Load weights if checkpoint exists
checkpoint_path = "checkpoints/slm_africa_galore.weights.h5"
if os.path.exists(checkpoint_path):
    load_checkpoint(model, checkpoint_path)
    print(f"Loaded weights from {checkpoint_path}")
else:
    print(f"No checkpoint found at {checkpoint_path}. Running with initialized weights.")

# 4. Generate text
result = generate_text(
    prompt="Abeni,",
    n_tokens=10,
    model=model,
    tokenizer=tokenizer,
    sampling_mode="greedy"
)
print("Generated:", result.generated_text)
```

---

## 4. Reproducing Checkpoint via Training

To reproduce the exact weights demonstrated in the original notebook:
```bash
python scripts/train.py --epochs 200 --batch-size 32 --lr 1e-4 --seed 812 --save-dir checkpoints/
```
This produces:
- `checkpoints/slm_africa_galore.keras`
- `checkpoints/slm_africa_galore.weights.h5`
- `checkpoints/training_history.json`
