"""Text generation functions for the Small Language Model with input validation and temperature scaling."""

from dataclasses import dataclass
import random
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import keras
from keras import ops
import numpy as np

from slm.tokenizer import SimpleWordTokenizer


@dataclass
class GenerationResult:
    """Encapsulates the outputs and metadata of a text generation call."""

    prompt: str
    generated_text: str
    full_text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    all_token_ids: list[int]
    generated_tokens: list[str]
    step_probabilities: list[np.ndarray]
    sampling_mode: str
    temperature: float


def validate_generation_inputs(
    prompt: str,
    n_tokens: int,
    model: keras.Model,
    tokenizer: SimpleWordTokenizer,
    sampling_mode: str,
    temperature: float,
) -> None:
    """Validate all generation parameters prior to running inference."""
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")

    if not isinstance(n_tokens, int):
        raise TypeError(f"n_tokens must be an integer, got {type(n_tokens).__name__}")

    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be greater than 0, got {n_tokens}")

    if n_tokens > 1000:
        raise ValueError(f"n_tokens ({n_tokens}) exceeds maximum safety limit of 1000")

    if not isinstance(model, keras.Model):
        raise TypeError(f"model must be a keras.Model instance, got {type(model).__name__}")

    if not (isinstance(tokenizer, SimpleWordTokenizer) or (hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"))):
        raise TypeError(f"tokenizer must have encode and decode methods, got {type(tokenizer).__name__}")

    if sampling_mode not in ("greedy", "random"):
        raise ValueError(f"sampling_mode must be 'greedy' or 'random', got '{sampling_mode}'")

    if not isinstance(temperature, (int, float)) or temperature <= 0.0:
        raise ValueError(f"temperature must be a positive number, got {temperature}")


def greedy_decoding(probs: np.ndarray) -> int:
    """Select the token index with maximum probability."""
    return int(np.argmax(probs))


def sample_with_temperature(logits: np.ndarray, temperature: float, rng_seed: int | None = None) -> tuple[int, np.ndarray]:
    """Apply temperature scaling to logits, compute softmax probabilities, and sample."""
    scaled_logits = logits / max(temperature, 1e-6)
    # Numerical stability shift
    shifted_logits = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits)

    if rng_seed is not None:
        np.random.seed(rng_seed)
    sampled_id = int(np.random.choice(len(probs), p=probs))
    return sampled_id, probs


def generate_text(
    prompt: str,
    n_tokens: int | None = None,
    model: keras.Model | None = None,
    tokenizer: Any = None,
    pad_token_id: int = 0,
    sampling_mode: Literal["greedy", "random"] = "greedy",
    temperature: float = 1.0,
    seed: int | None = None,
    max_new_tokens: int | None = None,
) -> GenerationResult:
    """Generate tokens autoregressively from a starting prompt."""
    if n_tokens is None:
        n_tokens = max_new_tokens if max_new_tokens is not None else 15

    validate_generation_inputs(
        prompt=prompt,
        n_tokens=n_tokens,
        model=model,
        tokenizer=tokenizer,
        sampling_mode=sampling_mode,
        temperature=temperature,
    )

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Get maximum sequence length from the model's input shape
    max_length = model.layers[0].output.shape[1]

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    current_tokens = list(prompt_tokens)
    generated_token_ids: list[int] = []
    step_probs: list[np.ndarray] = []

    for step in range(n_tokens):
        pad_len = max_length - len(current_tokens)

        if pad_len < 0:
            # Truncate left or right: model processes the trailing max_length tokens
            model_input = current_tokens[-max_length:]
            sample_index = max_length - 1
        elif pad_len > 0:
            # Post-pad with pad_token_id
            model_input = current_tokens + [pad_token_id] * pad_len
            sample_index = len(current_tokens) - 1
        else:
            model_input = current_tokens
            sample_index = max_length - 1

        # Feed to model
        input_tensor = np.array([model_input], dtype=np.int32)
        logits = model.predict(input_tensor, verbose="0")
        step_logits = np.array(logits[0, sample_index, :], dtype=np.float32)

        if sampling_mode == "greedy":
            # Softmax for reporting probabilities
            shifted = step_logits - np.max(step_logits)
            exp_l = np.exp(shifted)
            probs = exp_l / np.sum(exp_l)
            next_token_id = greedy_decoding(probs)
        else:
            rng_s = (seed + step) if seed is not None else None
            next_token_id, probs = sample_with_temperature(step_logits, temperature, rng_seed=rng_s)

        step_probs.append(probs)
        generated_token_ids.append(next_token_id)
        current_tokens.append(next_token_id)

    # Decode outputs
    if hasattr(tokenizer, "index_to_token"):
        unk_token = getattr(tokenizer, "UNKNOWN_TOKEN", "<UNK>")
        generated_tokens = [tokenizer.index_to_token.get(t_id, unk_token) for t_id in generated_token_ids]
    else:
        generated_tokens = [tokenizer.decode([t_id]) for t_id in generated_token_ids]

    generated_text = tokenizer.decode(generated_token_ids, skip_special=True)
    full_text = tokenizer.decode(current_tokens, skip_special=True)

    return GenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        full_text=full_text,
        prompt_token_ids=prompt_tokens,
        generated_token_ids=generated_token_ids,
        all_token_ids=current_tokens,
        generated_tokens=generated_tokens,
        step_probabilities=step_probs,
        sampling_mode=sampling_mode,
        temperature=temperature,
    )
