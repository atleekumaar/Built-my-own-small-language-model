"""
Deterministic Train / Validation Split Creation for Africa Galore Dataset.

Split parameters:
- Seed: 42
- Total paragraphs: 232
- Train split: 185 paragraphs (79.74%)
- Validation split: 47 paragraphs (20.26%)
- Stratification/Randomization: Deterministic pseudo-random shuffle with fixed seed
- Verification: Zero duplicate paragraphs, strict separation of train and val indices.
"""
import json
import random
from pathlib import Path

def create_and_save_splits(seed: int = 42, train_ratio: float = 0.8) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    data_file = repo_root / "data" / "africa_galore.json"
    splits_file = repo_root / "data" / "dataset_splits.json"

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    split_point = int(total * train_ratio)
    train_indices = sorted(indices[:split_point])
    val_indices = sorted(indices[split_point:])

    # Validate strict partition
    assert len(train_indices) + len(val_indices) == total
    assert len(set(train_indices).intersection(set(val_indices))) == 0

    splits_data = {
        "metadata": {
            "dataset": "africa_galore",
            "total_paragraphs": total,
            "seed": seed,
            "train_ratio": train_ratio,
            "train_count": len(train_indices),
            "val_count": len(val_indices),
            "created_by": "scripts/create_splits.py",
        },
        "train_indices": train_indices,
        "val_indices": val_indices,
    }

    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    print(f"Created splits: {len(train_indices)} train paragraphs, {len(val_indices)} validation paragraphs.")
    print(f"Saved to: {splits_file}")
    return splits_data

if __name__ == "__main__":
    create_and_save_splits()
