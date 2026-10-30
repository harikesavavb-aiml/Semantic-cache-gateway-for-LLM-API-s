FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
    --index-url https://pypi.org/simple \
    --no-cache-dir \
    --retries 10 \
    --timeout 120 \
    -r requirements.txt

# Bake the embedding + re-rank models into the image so a container starts
# without downloading anything (faster, deterministic, works air-gapped).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/quora-distilroberta-base')"

COPY . .

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 gateway && chown -R gateway /app /opt/hf-cache
USER gateway

EXPOSE 8000

# Liveness only: never depends on Redis, so a cache outage does not get the
# container restarted (the gateway is built to keep serving through it).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# WEB_CONCURRENCY (uvicorn reads it) sets worker processes; each loads the models.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]
