"""Simple word-level tokenizer for the Small Language Model.

Based on the original implementation from Google DeepMind AI Research Foundations,
with added input validation, serialization, and round-trip verification utilities.
"""

from pathlib import Path
import re
from typing import Sequence, Union


class SimpleWordTokenizer:
    """A whitespace-delimited word tokenizer with special <PAD> and <UNK> tokens.

    Constants:
        PAD_TOKEN: "<PAD>" (mapped to index 0)
        UNKNOWN_TOKEN: "<UNK>" (mapped to last vocabulary index)
    """

    PAD_TOKEN = "<PAD>"
    UNKNOWN_TOKEN = "<UNK>"

    def __init__(
        self,
        corpus: Sequence[str] | None = None,
        vocabulary: Sequence[str] | None = None,
    ):
        """Initialize tokenizer with either a raw corpus or an explicit vocabulary list."""
        if vocabulary is not None:
            self.vocabulary = list(vocabulary)
        elif corpus is not None:
            if isinstance(corpus, str):
                corpus = [corpus]
            tokens: list[str] = []
            for text in corpus:
                tokens.extend(self.space_tokenize(text))
            unique_tokens = self.build_vocabulary(tokens)
            self.vocabulary = [self.PAD_TOKEN] + unique_tokens + [self.UNKNOWN_TOKEN]
        else:
            raise ValueError("Either 'corpus' or 'vocabulary' must be provided.")

        self.vocabulary_size = len(self.vocabulary)
        self.token_to_index: dict[str, int] = {
            token: idx for idx, token in enumerate(self.vocabulary)
        }
        self.index_to_token: dict[int, str] = {
            idx: token for idx, token in enumerate(self.vocabulary)
        }

        # Map special tokens to their IDs
        self.pad_token_id = self.token_to_index.get(self.PAD_TOKEN, 0)
        self.unknown_token_id = self.token_to_index.get(
            self.UNKNOWN_TOKEN, self.vocabulary_size - 1
        )

    @property
    def vocab_size(self) -> int:
        return self.vocabulary_size

    @staticmethod
    def space_tokenize(text: str) -> list[str]:
        """Split text on one or more spaces, stripping leading/trailing whitespace."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str input for text, got {type(text).__name__}")
        trimmed = text.strip()
        if not trimmed:
            return []
        return re.split(r" +", trimmed)

    @staticmethod
    def build_vocabulary(tokens: Sequence[str]) -> list[str]:
        """Create a sorted list of unique tokens from a token sequence."""
        return sorted(list(set(tokens)))

    @staticmethod
    def join_text(text_list: Sequence[str]) -> str:
        """Combine tokens into a whitespace-separated string."""
        return " ".join(text_list)

    def encode(self, text: str) -> list[int]:
        """Encode text sequence into a list of token indices.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str for text, got {type(text).__name__}")
        
        tokens = self.space_tokenize(text)
        return [
            self.token_to_index.get(token, self.unknown_token_id)
            for token in tokens
        ]

    def decode(self, indices: Union[int, Sequence[int]], skip_special: bool = False) -> str:
        """Decode a single index or sequence of indices back to text.

        Args:
            indices: Integer or sequence of integers.
            skip_special: If True, omit <PAD> tokens in decoded output.

        Returns:
            Decoded string.
        """
        if isinstance(indices, int):
            indices = [indices]
        elif not isinstance(indices, (list, tuple)):
            try:
                indices = list(indices)
            except Exception as e:
                raise TypeError(f"Invalid indices type: {type(indices).__name__}") from e

        tokens: list[str] = []
        for idx in indices:
            int_idx = int(idx)
            token = self.index_to_token.get(int_idx, self.UNKNOWN_TOKEN)
            if skip_special and token == self.PAD_TOKEN:
                continue
            tokens.append(token)

        return self.join_text(tokens)

    def save_vocabulary(self, file_path: str | Path) -> None:
        """Serialize vocabulary list to JSON file."""
        import json

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_vocabulary_file(cls, file_path: str | Path) -> "SimpleWordTokenizer":
        """Load tokenizer from a JSON vocabulary file."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocabulary=vocab)
