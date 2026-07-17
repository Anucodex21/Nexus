<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1b26,50:7dcfff,100:bb9af7&height=200&section=header&text=NEXUS&fontSize=48&fontColor=c0caf5&animation=fadeIn&fontAlignY=35&desc=A%20Full-Stack%20AI%2FML%20Framework%20Built%20From%20Scratch&descAlignY=55&descSize=18" width="100%"/>

<a href="https://github.com/Anucodex21/ai-master-nexus">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=20&duration=2800&pause=1200&color=7DCFFF&center=true&vCenter=true&width=650&lines=Neural+Networks+%E2%80%A2+Transformers+%E2%80%A2+GPT+from+scratch;RAG+Pipelines+%E2%80%A2+Multi-Agent+Systems;Speech+%2B+Vision+%2B+FastAPI+Backend;Built+by+%40Anucodex21" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-7dcfff?style=for-the-badge&logo=python&logoColor=1a1b26&labelColor=24283b)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-bb9af7?style=for-the-badge&logo=fastapi&logoColor=1a1b26&labelColor=24283b)
![License](https://img.shields.io/badge/License-MIT-9ece6a?style=for-the-badge&labelColor=24283b)
![Status](https://img.shields.io/badge/Status-Alpha-f7768e?style=for-the-badge&labelColor=24283b)

</div>

---

## 📡 Overview

**AI-Master (Nexus)** is a full-stack AI/ML framework implementing core building blocks — neural networks, transformers, LLM pipelines, RAG, multi-agent systems, speech, and vision — largely **from scratch in Python**, wired up behind a FastAPI backend with a lightweight web frontend.

It's built as a learning-by-building project: instead of only calling library APIs, the core architectures (perceptrons, RNNs, LSTMs, attention, GPT blocks) are implemented directly, alongside integration layers for real providers (Groq, Gemini, OpenRouter, Hugging Face, Anthropic, OpenAI, NVIDIA NIM) where using a hosted model makes more sense than a from-scratch one.

> ⚠️ **Alpha status.** APIs and module layout may change.

---

## ✨ Features

| Module | What's inside |
|---|---|
| 🧠 `neural_network/` | Perceptron, backprop, CNN, RNN, LSTM — implemented from first principles |
| 🔤 `transformers/` | Tokenizer, embeddings, attention, encoder/decoder, full `Transformer`, and a from-scratch `GPT` with `.generate()` |
| 🤖 `llm/` | Training, fine-tuning, inference, dataset/dataloader, and evaluation pipelines |
| 🔗 `langchain/` | `ChatBot`, PDF Q&A, SQL agent, prompt templates, memory, tool-calling |
| 🕵️ `agents/` | Task planner, executor, memory, CrewAI/AutoGen integration, browser automation, coding agent |
| 📚 `rag/` | `RAGPipeline` with pluggable retrievers — Chroma, FAISS, Pinecone |
| 🎙️ `speech/` | `VoiceAssistant` — Whisper transcription + TTS in one loop |
| 🖼️ `image/` | `ImageGenerator` — Stable Diffusion backend, BLIP captioning, batch generation |
| 🌐 `app/` | FastAPI backend (auth, chat, streaming, RAG, agents) + HTML frontend |

---

## 🗂️ Project Structure

```
AI-Master (Nexus)/
├── neural_network/     # Perceptron, CNN, RNN, LSTM, backprop, optimizers
├── transformers/        # Attention, encoder/decoder, full Transformer, GPT
├── llm/                 # train / finetune / inference / evaluate pipelines
├── langchain/            # ChatBot, PDF Q&A, SQL agent, RAG, tools, memory
├── agents/              # Planner, executor, CrewAI, AutoGen, browser agent
├── rag/                 # Retriever + Chroma / FAISS / Pinecone pipelines
├── speech/               # Whisper + TTS voice assistant
├── image/                # Stable Diffusion, BLIP, captioning, batch gen
├── app/
│   ├── backend/         # FastAPI: auth, chat, streaming, RAG, agents
│   └── frontend/        # HTML/JS interface
├── interface.py          # CLI entry point
├── setup.py
└── requirements.txt
```

---

## 🚀 Installation

```bash
git clone https://github.com/Anucodex21/ai-master-nexus.git
cd ai-master-nexus

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Open `.env` and add **at least one** AI provider key (Groq and Gemini both have generous free tiers) so `/chat` returns real model responses instead of an offline fallback message. Provider priority is `groq → gemini → openrouter → huggingface → claude → openai`, overridable via `LLM_PROVIDER_ORDER`.

> 🔒 **Never commit your real `.env`** — only `.env.example` (with placeholder values) belongs in git. `.env` is already in `.gitignore`.

---

## ⚡ Quick Start

### Neural Network (from scratch)
```python
from neural_network import Perceptron

model = Perceptron(input_size=2, learning_rate=0.01, epochs=100)
model.train(X, y)
predictions = model.predict(X)
```

### Transformer
```python
from transformers import Transformer

model = Transformer(vocab_size=10000, d_model=512, num_heads=8)
output = model.forward(src, tgt)
```

### GPT (from scratch)
```python
from transformers.gpt import GPT

model = GPT(vocab_size=10000, d_model=768, num_heads=12, num_layers=12)
tokens = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
```

### LLM Inference (hosted providers)
```python
from llm.inference import LLMInference

llm = LLMInference(model_path="path/to/model")
response = llm.generate("Hello, how are you?")
```

### RAG Pipeline
```python
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever

retriever = Retriever(...)
rag = RAGPipeline(retriever=retriever)
answer = rag.run("What does this document say about X?", top_k=5)
```

### LangChain Chatbot
```python
from langchain.chatbot import ChatBot

bot = ChatBot(model_name="gpt-3.5-turbo", temperature=0.7)
reply = bot.chat("Summarize this in one sentence.")
```

### Voice Assistant
```python
from speech.voice_assistant import VoiceAssistant

assistant = VoiceAssistant()
assistant.run_interaction(duration=5)
```

### Image Generation
```python
from image.image_generation import ImageGenerator

gen = ImageGenerator(backend="stable_diffusion")
image = gen.generate("a cyberpunk city at night")
```

---

## 🌐 Running the Full-Stack App

```bash
python -m app.backend.main
```

FastAPI serves on `http://localhost:8000`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/auth/register`, `/auth/login` | POST | User auth (JWT) |
| `/chat` | POST | Single-turn chat |
| `/chat/stream` | POST | Streamed chat response |
| `/models` | GET | List available providers/models |
| `/conversations` | GET | List conversations |
| `/conversations/{id}` | GET | Fetch a conversation |
| `/upload/document` | POST | Upload a doc for RAG |
| `/rag/query` | POST | Query the RAG pipeline |
| `/generate/image` | POST | Image generation |
| `/agent/status`, `/agent/run` | GET / POST | Agent system control |

---

## 🛠️ Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-24283b?style=flat-square&logo=pytorch&logoColor=7dcfff)
![FastAPI](https://img.shields.io/badge/FastAPI-24283b?style=flat-square&logo=fastapi&logoColor=bb9af7)
![LangChain](https://img.shields.io/badge/LangChain-24283b?style=flat-square&logoColor=7dcfff)
![Whisper](https://img.shields.io/badge/Whisper-24283b?style=flat-square&logo=openai&logoColor=bb9af7)

---

## 🤝 Contributing

Issues and PRs are welcome — this is an actively evolving learning project. Fork it, open an issue, or send a PR.

## 📄 License

MIT License — see [`LICENSE`](LICENSE).

---

<div align="center">

**Built by [@Anucodex21](https://github.com/Anucodex21)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1b26,50:7dcfff,100:bb9af7&height=100&section=footer" width="100%"/>

</div>
