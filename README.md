# AI-Master (Nexus)

[![CI](https://github.com/<your-username>/<repo-name>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<repo-name>/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI/ML framework implementing neural networks, transformers, LLMs, RAG, agents, and multimodal capabilities from scratch - plus a full-stack web app (FastAPI + vanilla HTML/JS "Nexus" UI) that ties them together.

> Replace `<your-username>/<repo-name>` in the badge URL above with your actual GitHub path once pushed.

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Running the web app](#running-the-web-app)
- [Docker](#docker)
- [Testing](#testing)
- [API endpoints](#api-endpoints-prefixed-apiv1)
- [Library usage](#library-usage-outside-the-web-app)
- [Security notes](#security-notes-before-deploying-anywhere-public)
- [License](#license)

## Quickstart

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
cp .env.example .env        # add at least one provider key, e.g. GROQ_API_KEY
pip install -r requirements-web.txt
python -m app.backend.main
```
Open **http://localhost:8000/ui/index.html** and start chatting. See [Docker](#docker) for a one-command alternative.

## Features

- **Neural Networks**: Perceptron, Backpropagation, CNN, RNN, LSTM
- **Transformers**: Full transformer architecture, GPT implementation
- **LLM**: Training, inference, fine-tuning, evaluation pipelines, and a real multi-provider `LLMClient` (Groq/Gemini/OpenRouter/HuggingFace/Claude/OpenAI/local model, with automatic fallback)
- **LangChain**: Chatbots, PDF Q&A, SQL agents, RAG pipelines
- **Agents**: a working ReAct coding agent (read/write files, list dirs, run Python, sandboxed workspace, persistent per-session memory). `agents/planner.py`, `executor.py`, `crewai.py`, `autogen.py`, `browser_agent.py` are standalone modules not currently wired into the web app.
- **RAG**: per-user document upload + chunking + embedding + Chroma retrieval, answered through the same multi-provider LLMClient as chat
- **Speech**: Whisper transcription, Coqui TTS synthesis
- **Image**: Stable Diffusion generation, BLIP captioning
- **Full-Stack App**: FastAPI backend + a themed HTML/JS frontend (Chat, Coding Agent, and a Studio page for RAG/Speech/Image), plus an alternative Streamlit chat UI (`interface.py`)

## Installation

```bash
# Full install (everything above, incl. torch/whisper/diffusers/chroma):
pip install -r requirements.txt

# OR a lightweight install if you only want the chat/auth web app:
pip install -r requirements-web.txt

cp .env.example .env
# Edit .env with your API keys (only ONE cloud provider key is required
# for /chat to give real responses - see comments in .env.example)
```

## Running the web app

```bash
python -m app.backend.main
# or: uvicorn app.backend.api:app --reload
```

Then open **http://localhost:8000/ui/index.html** (Chat), **/ui/agent.html** (Coding Agent), or **/ui/studio.html** (RAG / Speech / Image). All three share the same login.

### Alternative UI: Streamlit

`interface.py` is a second, simpler chat frontend built with Streamlit instead of the HTML/JS "Nexus" UI - same backend, same `/api/v1` endpoints, just a different client. Handy if you want a quick local UI without touching the frontend files.

```bash
# backend must already be running (see above)
streamlit run interface.py
```

### Docker

```bash
cp .env.example .env   # add at least one provider key first
docker compose up --build
```

Serves the same app at **http://localhost:8000**, using `requirements-web.txt` by default (chat/auth/RAG-query work; RAG document embedding, speech, and image endpoints return 503 until you build with the full extras). The sqlite DB, Chroma store, and agent workspace persist across restarts via a named volume. To build with everything (torch/whisper/diffusers/chromadb included):

```bash
INSTALL_FULL=1 docker compose up --build
```

Without Docker, the same image can be built/run directly:
```bash
docker build -t nexus .
docker run -p 8000:8000 --env-file .env nexus
```

## Testing

```bash
pip install -r requirements-web.txt -r requirements-dev.txt
pytest
```

Covers auth (register/login/token enforcement), chat + streaming + conversation persistence (against the deterministic offline fallback, so no real API calls or keys are needed), RAG chunking logic, and the coding agent's sandbox (path-traversal blocking, file read/write, Python execution). Tests that need heavier deps (embeddings, speech, image, local-model inference) aren't included here since they'd require the full `requirements.txt` install and real model downloads - exercise those manually through the Studio UI instead.

CI (`.github/workflows/ci.yml`) runs this same suite on every push/PR against Python 3.11 and 3.12.

### API endpoints (prefixed `/api/v1`)

| Area | Endpoints |
|---|---|
| Auth | `POST /auth/register`, `POST /auth/login` |
| Chat | `POST /chat`, `POST /chat/stream` (NDJSON), `GET /models`, `GET /conversations`, `GET /conversations/{id}` |
| Files | `POST /upload/document` |
| Image | `POST /generate/image`, `POST /image/caption` |
| Speech | `POST /speech/transcribe`, `POST /speech/speak` |
| RAG | `POST /rag/upload`, `POST /rag/query`, `GET /rag/stats` |
| Agent | `GET /agent/status`, `POST /agent/run` (NDJSON, unauthenticated by design - see comment in `routes.py`), `GET /agent/memory/{session_id}` |

Speech/image/RAG endpoints lazy-load their models on first request, so a `requirements-web.txt`-only install still boots - those specific endpoints will just return a 503 until the full `requirements.txt` extras are installed.

## Library usage (outside the web app)

### Neural Network
```python
from neural_network import Perceptron
model = Perceptron(input_size=2)
model.train(X, y, epochs=100)
```

### Transformer
```python
from transformers import Transformer
model = Transformer(vocab_size=10000, d_model=512, num_heads=8)
output = model(src, tgt)
```

### LLM Inference
```python
from llm import LLMInference
llm = LLMInference(model_path="path/to/model")
response = llm.generate("Hello, how are you?")
```

## Security notes before deploying anywhere public

- **Set a real `SECRET_KEY`** in `.env` (`python -c "import secrets; print(secrets.token_hex(32))"`). If it's unset or left as the `.env.example` placeholder, `auth.py` now generates a random key per process instead of using an insecure hardcoded default - safer, but it also means every restart invalidates all existing login sessions until you set a real one.
- **`POST /api/v1/agent/run` is intentionally unauthenticated** (see the comment in `routes.py`) - it runs arbitrary Python inside `agent_workspace/`. Fine for local/dev use; add `Depends(AuthManager.verify_token)` before exposing it on a network anyone else can reach.
- CORS defaults to allow-all (`CORS_ORIGINS` unset) for zero-friction local dev. Set `CORS_ORIGINS` (comma-separated) in `.env` before deploying publicly, e.g. `CORS_ORIGINS=https://your-domain.com`.

## License

MIT License - see [LICENSE](LICENSE).
