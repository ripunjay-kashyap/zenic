FROM python:3.12-slim

# Run as a dedicated non-root user.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    ENV=production \
    LOG_FORMAT=json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl is needed by the healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR $HOME/app

# Install CPU-only torch first (much smaller image), then the pinned tree.
# The production lock excludes indexing/evaluation dependencies and their
# documented advisories. CPU PyTorch is installed separately.
COPY --chown=user requirements.runtime.lock.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.runtime.lock.txt && \
    pip check

# Pre-bake the embedding + reranker models into the image (no cold-start download).
# Kept as its own layer so it is not invalidated by application code changes.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

# The models are baked into the image; avoid Hub probes during live requests.
ENV HF_HUB_OFFLINE=1

# App code, plus data/bm25_corpus.json and data/mock_weekly.json, which are read
# at startup and by the weekly-summary workflow respectively.
COPY --chown=user . .

EXPOSE 7860

# The API endpoint reports unhealthy if the server is unconfigured.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/api/health || exit 1

CMD ["uvicorn", "zenic.web.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
