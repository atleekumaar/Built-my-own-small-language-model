"""FastAPI application for SLM model serving, tokenization, and generation.

Exposes:
- GET /health
- GET /config
- POST /tokenize
- POST /generate
"""

import os
from pathlib import Path
import sys
import time
from typing import Any, Literal

# Add repository root to python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field

from slm.bpe_tokenizer import BPETokenizer
from slm.config import ModelConfig
from slm.generate import generate_text
from slm.model import build_model, load_checkpoint
from slm.tokenizer import SimpleWordTokenizer


AVAILABLE_MODELS: dict[str, dict[str, Any]] = {
    "slm_africa_galore": {
        "id": "slm_africa_galore",
        "name": "Africa Galore Original (Word-Level, 3.36M)",
        "description": "Trained on all 232 paragraphs with SimpleWordTokenizer. Historical course model.",
        "tokenizer_type": "word",
        "config_path": "configs/model_config.json",
        "vocab_path": "data/vocab.json",
        "weights_path": "checkpoints/slm_africa_galore.weights.h5",
        "parameters": 3358094,
        "vocabulary_size": 5262,
        "max_length": 299,
        "train_paragraphs": 232,
        "val_paragraphs": 0,
        "in_sample_perplexity": 323.04,
        "val_perplexity": None,
        "split_description": "Full Corpus (232 paragraphs, in-sample fitting)",
    },
    "slm_baseline_val": {
        "id": "slm_baseline_val",
        "name": "Experiment A: Word Baseline (2.92M, 80/20 Split)",
        "description": "Trained exclusively on 185 paragraphs with word tokenizer; evaluated on 47 held-out paragraphs.",
        "tokenizer_type": "word",
        "config_path": "configs/model_config_baseline.json",
        "vocab_path": "data/vocab_baseline.json",
        "weights_path": "checkpoints/slm_baseline_val.weights.h5",
        "parameters": 2920505,
        "vocabulary_size": 4409,
        "max_length": 299,
        "train_paragraphs": 185,
        "val_paragraphs": 47,
        "in_sample_perplexity": 398.66,
        "val_perplexity": 1068.32,
        "split_description": "185 Train / 47 Val (Strict held-out validation)",
    },
    "slm_bpe_val": {
        "id": "slm_bpe_val",
        "name": "Experiment B: Byte-Level BPE (1.68M, 80/20 Split)",
        "description": "Trained on 185 paragraphs with Byte-Level BPE subwords (2,000 vocab). Zero OOV; 42% smaller.",
        "tokenizer_type": "bpe",
        "config_path": "configs/model_config_bpe.json",
        "tokenizer_path": "checkpoints/bpe_tokenizer.json",
        "weights_path": "checkpoints/slm_bpe_val.weights.h5",
        "parameters": 1684688,
        "vocabulary_size": 2000,
        "max_length": 299,
        "train_paragraphs": 185,
        "val_paragraphs": 47,
        "in_sample_perplexity": 269.23,
        "val_perplexity": 392.55,
        "split_description": "185 Train / 47 Val (Strict held-out validation)",
    },
}


# Global server state
class ServerState:
    active_model_id: str = "slm_africa_galore"
    config: ModelConfig
    tokenizer: Any
    model: Any = None
    checkpoint_loaded: bool = False
    checkpoint_path: str | None = None
    device: str = "cpu"

state = ServerState()


def load_model_by_id(model_id: str) -> None:
    """Load a specific model and its tokenizer into server state."""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model_id: '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}")

    meta = AVAILABLE_MODELS[model_id]
    cfg_p = Path(meta["config_path"])
    if cfg_p.exists():
        state.config = ModelConfig.from_json(cfg_p)
    else:
        state.config = ModelConfig()

    # Load tokenizer
    if meta["tokenizer_type"] == "bpe":
        tok_p = Path(meta.get("tokenizer_path", "checkpoints/bpe_tokenizer.json"))
        state.tokenizer = BPETokenizer.from_file(tok_p)
    else:
        v_p = Path(meta["vocab_path"])
        state.tokenizer = SimpleWordTokenizer.from_vocabulary_file(v_p)

    state.model = build_model(state.config)
    state.config.total_parameters = state.model.count_params()
    w_p = Path(meta["weights_path"])
    if w_p.exists():
        load_checkpoint(state.model, str(w_p))
        state.checkpoint_loaded = True
        state.checkpoint_path = str(w_p)
    else:
        state.checkpoint_loaded = False
        state.checkpoint_path = None

    state.active_model_id = model_id


def initialize_server_state(
    config_path: str = "configs/model_config.json",
    vocab_path: str = "data/vocab.json",
    checkpoint_path: str = "checkpoints/slm_africa_galore.weights.h5",
) -> None:
    """Initialize server state with default original model."""
    load_model_by_id("slm_africa_galore")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_server_state()
    yield


app = FastAPI(
    title="Small Language Model (SLM) API",
    description="Interactive inference and tokenization API powered by Keras 3 and JAX.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for playground frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request / Response schemas
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    checkpoint_loaded: bool
    checkpoint_path: str | None
    device: str
    total_parameters: int
    vocabulary_size: int
    max_length: int
    active_model_id: str = "slm_africa_galore"
    tokenizer_type: str = "word"


class SelectModelRequest(BaseModel):
    model_id: str


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to tokenize.")


class TokenizeResponse(BaseModel):
    tokens: list[str]
    token_ids: list[int]
    token_count: int
    vocabulary_size: int
    unknown_token_count: int
    has_unknown_tokens: bool


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Input text prompt.")
    max_new_tokens: int = Field(default=15, ge=1, le=100, description="Number of tokens to generate.")
    sampling_mode: Literal["greedy", "random"] = Field(default="greedy", description="Sampling strategy.")
    temperature: float = Field(default=1.0, gt=0.01, le=5.0, description="Sampling temperature.")
    seed: int | None = Field(default=None, description="Optional seed for deterministic generation.")
    top_k_probabilities: int = Field(default=5, ge=0, le=20, description="Number of top candidates per step.")


class TokenProbability(BaseModel):
    token: str
    token_id: int
    probability: float


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: list[str]
    generated_tokens: list[str]
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    step_top_candidates: list[list[TokenProbability]]
    sampling_mode: str
    temperature: float
    latency_ms: float
    checkpoint_loaded: bool
    active_model_id: str = "slm_africa_galore"


@app.get("/models")
def list_models():
    """List available SLM models and indicate the active model."""
    models_list = []
    for mid, info in AVAILABLE_MODELS.items():
        item = dict(info)
        item["is_active"] = (mid == state.active_model_id)
        models_list.append(item)
    return {
        "active_model_id": state.active_model_id,
        "models": models_list,
    }


@app.post("/models/select")
def select_model(payload: SelectModelRequest):
    """Dynamically switch the active model and tokenizer."""
    if payload.model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{payload.model_id}' not found. Available: {list(AVAILABLE_MODELS.keys())}",
        )
    load_model_by_id(payload.model_id)
    return {
        "status": "ok",
        "active_model_id": state.active_model_id,
        "name": AVAILABLE_MODELS[state.active_model_id]["name"],
        "tokenizer_type": AVAILABLE_MODELS[state.active_model_id]["tokenizer_type"],
        "parameters": state.config.total_parameters,
        "vocabulary_size": state.tokenizer.vocabulary_size,
    }


@app.get("/experiments")
def get_experiments():
    """Retrieve machine-readable Phase 4 experiment comparisons."""
    exp_path = Path("checkpoints/experiment_comparison.json")
    if exp_path.exists():
        import json
        with open(exp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "not_found", "message": "No experiment comparison data found."}


@app.get("/health", response_model=HealthResponse)
def get_health():
    """Service health and model status endpoint."""
    if state.model is None:
        initialize_server_state()

    tok_type = AVAILABLE_MODELS.get(state.active_model_id, {}).get("tokenizer_type", "word")
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        checkpoint_loaded=state.checkpoint_loaded,
        checkpoint_path=state.checkpoint_path,
        device=state.device,
        total_parameters=state.config.total_parameters,
        vocabulary_size=state.tokenizer.vocabulary_size,
        max_length=state.config.max_length,
        active_model_id=state.active_model_id,
        tokenizer_type=tok_type,
    )


@app.get("/config")
def get_config():
    """Retrieve verified architecture and tokenizer configuration."""
    if state.model is None:
        initialize_server_state()
    data = state.config.to_dict()
    data["active_model_id"] = state.active_model_id
    data["tokenizer_type"] = AVAILABLE_MODELS.get(state.active_model_id, {}).get("tokenizer_type", "word")
    return data


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize_text(payload: TokenizeRequest):
    """Tokenize a string into words or subwords and vocabulary token IDs."""
    if state.tokenizer is None:
        initialize_server_state()

    if hasattr(state.tokenizer, "space_tokenize"):
        tokens = state.tokenizer.space_tokenize(payload.text)
    else:
        tokens = state.tokenizer.tokenize(payload.text)

    token_ids = state.tokenizer.encode(payload.text)
    unk_id = getattr(state.tokenizer, "unknown_token_id", getattr(state.tokenizer, "unk_token_id", 1))
    unk_count = sum(1 for tid in token_ids if tid == unk_id)

    return TokenizeResponse(
        tokens=tokens,
        token_ids=token_ids,
        token_count=len(token_ids),
        vocabulary_size=state.tokenizer.vocabulary_size,
        unknown_token_count=unk_count,
        has_unknown_tokens=unk_count > 0,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest):
    """Autoregressively generate next tokens from an initial prompt."""
    if state.model is None:
        initialize_server_state()

    start_time = time.perf_counter()
    try:
        result = generate_text(
            prompt=payload.prompt,
            n_tokens=payload.max_new_tokens,
            model=state.model,
            tokenizer=state.tokenizer,
            pad_token_id=state.config.pad_token_id,
            sampling_mode=payload.sampling_mode,
            temperature=payload.temperature,
            seed=payload.seed,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Generation failed: {str(exc)}",
        ) from exc

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Extract top k candidates for each step
    step_candidates: list[list[TokenProbability]] = []
    unk_token_str = getattr(state.tokenizer, "UNKNOWN_TOKEN", "<UNK>")
    if payload.top_k_probabilities > 0:
        for prob_vec in result.step_probabilities:
            top_indices = np.argsort(prob_vec)[::-1][: payload.top_k_probabilities]
            step_candidates.append([
                TokenProbability(
                    token=state.tokenizer.index_to_token.get(int(idx), unk_token_str),
                    token_id=int(idx),
                    probability=float(prob_vec[idx]),
                )
                for idx in top_indices
            ])

    if hasattr(state.tokenizer, "space_tokenize"):
        prompt_tokens = state.tokenizer.space_tokenize(payload.prompt)
    else:
        prompt_tokens = state.tokenizer.tokenize(payload.prompt)

    return GenerateResponse(
        prompt=payload.prompt,
        generated_text=result.generated_text,
        full_text=result.full_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=result.generated_tokens,
        prompt_token_ids=result.prompt_token_ids,
        generated_token_ids=result.generated_token_ids,
        step_top_candidates=step_candidates,
        sampling_mode=payload.sampling_mode,
        temperature=payload.temperature,
        latency_ms=round(latency_ms, 2),
        checkpoint_loaded=state.checkpoint_loaded,
        active_model_id=state.active_model_id,
    )


@app.get("/evaluation")
def get_evaluation():
    """Retrieve actual evaluation and benchmark metrics."""
    history_path = Path("checkpoints/training_history.json")
    loss_history = []
    if history_path.exists():
        import json
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            loss_history = data.get("loss", [])

    return {
        "training_epochs_recorded": len(loss_history),
        "training_loss_history": loss_history,
        "latest_training_loss": loss_history[-1] if loss_history else None,
        "evaluation_loss": 5.7778,
        "evaluation_perplexity": 323.04,
        "evaluation_split": "in_sample_training_corpus (232 paragraphs, 18,803 tokens)",
        "is_held_out": False,
        "generalization_note": (
            "The measured perplexity of 323.04 is an in-sample training corpus metric. "
            "Because the full 232-paragraph dataset was used for training, this is NOT a held-out generalization score."
        ),
        "benchmark": {
            "hardware": "12-Core CPU (AMD64, Windows)",
            "device": "CpuDevice(id=0)",
            "throughput_tokens_per_sec": 53.39,
            "mean_latency_ms": 18.73,
            "median_latency_ms": 18.36,
            "p95_latency_ms": 21.26,
            "min_latency_ms": 17.39,
            "max_latency_ms": 22.43,
            "measurement_conditions": "Warmup: 2 runs; Evaluation: 3 prompts x 3 runs (10 tokens each); Keras 3 JAX CPU backend",
        },
    }


# Mount static frontend files if directory exists
from fastapi.staticfiles import StaticFiles

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

