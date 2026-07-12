FROM python:3.11-slim

WORKDIR /app

# System deps needed by some optional extras (bcrypt, sqlite headers).
# Kept minimal on purpose - this image targets the lightweight web app
# (requirements-web.txt), not the full torch/whisper/diffusers stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Optional: build with `--build-arg INSTALL_FULL=1` to also pull in
# RAG/speech/image/local-LLM deps (torch, chromadb, sentence-transformers,
# etc). This makes the image several GB larger, so it's opt-in.
ARG INSTALL_FULL=0
COPY requirements.txt .
RUN if [ "$INSTALL_FULL" = "1" ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY . .

RUN mkdir -p agent_workspace chroma_db models checkpoints

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
