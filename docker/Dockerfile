FROM python:3.12-slim-bookworm
LABEL org.opencontainers.image.title="Rudeus VinAI-LegalMT" \
      org.opencontainers.image.description="Offline bidirectional English-Vietnamese legal translation"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/rudeus
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY models/ ./models/
COPY app/ ./app/
COPY verify_models.py ./verify_models.py
COPY README.md LICENSE ./
RUN python verify_models.py
RUN useradd --create-home --uid 10001 rudeus && chown -R rudeus:rudeus /opt/rudeus
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false HF_HOME=/tmp/huggingface \
    RUDEUS_MODEL_ROOT=/opt/rudeus/models RUDEUS_BATCH_SIZE=32 RUDEUS_THREADS=2
USER rudeus
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).read()"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
