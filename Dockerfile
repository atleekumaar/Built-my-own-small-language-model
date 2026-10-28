# ==============================================================================
# ATLEE SLM LAB — Dockerfile for Hugging Face Spaces & Container Deployment
# Backend: Keras 3 with JAX (CPU Platform)
# Exposed Port: 7860 (Hugging Face Spaces default)
# ==============================================================================

FROM python:3.11-slim

# Prevent interactive prompts and Python bytecode / buffering delays
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KERAS_BACKEND=jax \
    JAX_PLATFORMS=cpu \
    PORT=7860

# Install minimal OS dependencies needed for compiling or fetching libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Step 1: Install Python dependencies (leveraging layer cache)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Copy application code, configs, tokenizers, and frontend
COPY --chown=user:user configs ./configs
COPY --chown=user:user data ./data
COPY --chown=user:user frontend ./frontend
COPY --chown=user:user slm ./slm
COPY --chown=user:user experiments ./experiments
COPY --chown=user:user pytest.ini ./pytest.ini

# Step 3: Copy checkpoints
# NOTE ON MODEL WEIGHTS:
# By default, binary checkpoint files (*.weights.h5, *.keras) are listed in .gitignore.
# When building locally, Docker will copy them from the checkpoints/ directory.
# If deploying to a remote host (e.g., Hugging Face Spaces via Git), checkpoints must either:
#   a) Be tracked via Git LFS, or
#   b) Downloaded in a startup entrypoint script, or
#   c) Supplied via volume mount.
COPY --chown=user:user checkpoints ./checkpoints

# Expose Hugging Face default port
EXPOSE 7860

# Healthcheck to verify FastAPI responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start Uvicorn serving FastAPI and the mounted frontend
CMD ["uvicorn", "slm.api:app", "--host", "0.0.0.0", "--port", "7860"]
