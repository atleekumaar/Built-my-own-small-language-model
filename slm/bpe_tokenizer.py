"""
Byte-Level Byte-Pair Encoding (BPE) Subword Tokenizer for ATLEE SLM.

Features:
- Built on HuggingFace tokenizers (Byte-level BPE, same family as GPT-2 / RoBERTa / LLaMA).
- 100% vocabulary coverage with zero out-of-vocabulary (<UNK>) failures on arbitrary text.
- Standardized API matching SimpleWordTokenizer for seamless model swap.
- Special tokens:
  - <PAD>: 0 (pad token masked by CustomMaskPadLoss)
  - <UNK>: 1 (unknown token fallback)
  - <BOS>: 2 (beginning-of-sequence)
  - <EOS>: 3 (end-of-sequence)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class BPETokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        if tokenizer is None:
            self._tok = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
            self._tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._tok.decoder = decoders.ByteLevel()
        else:
            self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID

    @property
    def vocabulary_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def token_to_index(self) -> dict[str, int]:
        return self._tok.get_vocab()

    @property
    def index_to_token(self) -> dict[int, str]:
        vocab = self._tok.get_vocab()
        return {idx: tok for tok, idx in vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return self._tok.get_vocab()

    def train_from_iterator(
        self,
        corpus: Iterable[str],
        vocab_size: int = 2000,
        min_frequency: int = 2,
    ) -> None:
        """Fit Byte-level BPE on the provided text iterator."""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self._tok.train_from_iterator(corpus, trainer=trainer)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ID sequence."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        if not text:
            return []
        encoding = self._tok.encode(text)
        return encoding.ids

    def decode(self, token_ids: list[int] | int, skip_special: bool = False) -> str:
        """Decode token IDs back into string."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if not isinstance(token_ids, (list, tuple)):
            raise TypeError(f"Expected list of int, got {type(token_ids).__name__}")
        if not token_ids:
            return ""
        # Filter special tokens manually if skip_special requested
        if skip_special:
            token_ids = [tid for tid in token_ids if tid not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)]
        return self._tok.decode(token_ids, skip_special_tokens=skip_special)

    def tokenize(self, text: str) -> list[str]:
        """Return raw token strings."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        if not text:
            return []
        encoding = self._tok.encode(text)
        return encoding.tokens

    def save(self, filepath: str | Path) -> None:
        """Save complete tokenizer model configuration to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(filepath))

    def save_vocabulary(self, filepath: str | Path) -> None:
        """Save readable vocabulary dictionary to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        vocab = self._tok.get_vocab()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, filepath: str | Path) -> BPETokenizer:
        """Load tokenizer from saved JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
        tok = Tokenizer.from_file(str(filepath))
        return cls(tok)
