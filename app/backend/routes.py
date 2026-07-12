from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
import json
import os
import tempfile
import uuid
from .auth import AuthManager
from .database import DatabaseManager
from .llm_client import LLMClient
from .rag_service import RAGService

router = APIRouter()
llm_client = LLMClient()
rag_service = RAGService(llm_client)

# Directory generated media (TTS wav files, generated images) gets written
# to before being streamed back. Not user-facing - only ever read right
# back out in the same request.
_MEDIA_DIR = os.path.join(tempfile.gettempdir(), "nexus_media")
os.makedirs(_MEDIA_DIR, exist_ok=True)


# Request/Response models
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    conversation_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    response: str
    model: str
    conversation_id: str

class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = "default"
    num_images: Optional[int] = 1

class RAGQueryRequest(BaseModel):
    question: str
    model: Optional[str] = None
    top_k: Optional[int] = 5

class SpeakRequest(BaseModel):
    text: str
    speaker: Optional[str] = None


# ---------------- Auth ----------------

@router.post("/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Create a new user account and return a session token."""
    if DatabaseManager.get_user_by_username(request.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    hashed = AuthManager.hash_password(request.password)
    DatabaseManager.create_user(request.username, request.email, hashed)
    token = AuthManager.create_access_token({"sub": request.username})
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Verify credentials and return a session token."""
    user = DatabaseManager.get_user_by_username(request.username)
    if not user or not AuthManager.verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = AuthManager.create_access_token({"sub": request.username})
    return TokenResponse(access_token=token)


# ---------------- Chat ----------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: str = Depends(AuthManager.verify_token)):
    """Chat with a real configured AI provider (falls back to an offline
    notice if no provider keys are set)."""
    conversation_id = request.conversation_id
    history = []
    if conversation_id:
        past = DatabaseManager.get_conversation_messages(user, conversation_id)
        for turn in past[-10:]:
            history.append({"role": "user", "content": turn["message"]})
            history.append({"role": "assistant", "content": turn["response"]})
    history.append({"role": "user", "content": request.message})

    reply, provider_used = llm_client.chat(history, preferred=request.model)

    saved_conversation_id = DatabaseManager.save_conversation(
        user_id=user,
        message=request.message,
        response=reply,
        model_used=provider_used,
        conversation_id=conversation_id,
    )

    return ChatResponse(response=reply, model=provider_used, conversation_id=saved_conversation_id)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, user: str = Depends(AuthManager.verify_token)):
    """Same as /chat, but streams the reply as NDJSON lines (one JSON
    object per line) as soon as each piece of text is generated, instead
    of waiting for the full response. Line types:
      {"type": "start", "model": <provider>, "conversation_id": <id>}
      {"type": "delta", "text": <chunk>}          - zero or more
      {"type": "error", "detail": <message>}      - only on failure
      {"type": "done",  "model": <provider>, "conversation_id": <id>}
    The full reply is still saved to the conversation history once
    streaming finishes, exactly like the non-streaming endpoint does.
    """
    conversation_id = request.conversation_id or uuid.uuid4().hex[:12]
    history = []
    if request.conversation_id:
        past = DatabaseManager.get_conversation_messages(user, request.conversation_id)
        for turn in past[-10:]:
            history.append({"role": "user", "content": turn["message"]})
            history.append({"role": "assistant", "content": turn["response"]})
    history.append({"role": "user", "content": request.message})

    def event_stream():
        parts = []
        provider_used = "offline"
        try:
            for event in llm_client.chat_stream(history, preferred=request.model):
                if event["type"] == "start":
                    provider_used = event["provider"]
                    yield json.dumps({
                        "type": "start",
                        "model": provider_used,
                        "conversation_id": conversation_id,
                    }) + "\n"
                elif event["type"] == "delta":
                    parts.append(event["text"])
                    yield json.dumps({"type": "delta", "text": event["text"]}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

        full_text = "".join(parts) or "No response generated."
        DatabaseManager.save_conversation(
            user_id=user,
            message=request.message,
            response=full_text,
            model_used=provider_used,
            conversation_id=conversation_id,
        )
        yield json.dumps({
            "type": "done",
            "model": provider_used,
            "conversation_id": conversation_id,
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/models")
async def list_models(user: str = Depends(AuthManager.verify_token)):
    """List AI providers actually usable right now (i.e. configured with a key)."""
    return {"models": llm_client.available_providers()}


@router.get("/conversations")
async def get_conversations(user: str = Depends(AuthManager.verify_token)):
    """List this user's saved conversations, most recently active first."""
    return {"conversations": DatabaseManager.list_conversations(user)}


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user: str = Depends(AuthManager.verify_token)):
    """Full message history for one conversation."""
    return {"messages": DatabaseManager.get_conversation_messages(user, conversation_id)}


# ---------------- Files ----------------

@router.post("/upload/document")
async def upload_document(file: UploadFile = File(...), user: str = Depends(AuthManager.verify_token)):
    """Upload a small text-like file so its content can be attached to a chat
    message. Binary formats like PDF/images aren't parsed here yet."""
    contents = await file.read()
    try:
        text = contents.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't read file as text")
    truncated = len(text) > 6000
    if truncated:
        text = text[:6000]
    return {"filename": file.filename, "content": text, "truncated": truncated}


# ---------------- Image (generation + captioning) ----------------

# Lazy-loaded singletons, same pattern LLMClient uses for the local model -
# these pull in torch/diffusers/transformers, which only requirements.txt
# (not requirements-web.txt) installs, so importing them at module load
# time would break the whole API for anyone running the lightweight setup.
_image_generator = None
_image_captioner = None


def _get_image_generator():
    global _image_generator
    if _image_generator is None:
        from image.image_generation import ImageGenerator
        _image_generator = ImageGenerator()
    return _image_generator


def _get_image_captioner():
    global _image_captioner
    if _image_captioner is None:
        from image.image_caption import ImageCaptioner
        _image_captioner = ImageCaptioner()
    return _image_captioner


@router.post("/generate/image")
async def generate_image(request: GenerateRequest, user: str = Depends(AuthManager.verify_token)):
    """Generate image(s) from a text prompt using Stable Diffusion
    (image/stable_diffusion.py), returned as base64 PNGs so the frontend
    can render them directly without a separate file-serving endpoint."""
    try:
        generator = _get_image_generator()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Image generation model unavailable: {e}")

    try:
        result = generator.generate(request.prompt, num_images=request.num_images or 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    images = result if isinstance(result, list) else [result]
    encoded = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    return {"prompt": request.prompt, "images": encoded, "count": len(encoded)}


@router.post("/image/caption")
async def caption_image(
    file: UploadFile = File(...),
    style: str = "descriptive",
    user: str = Depends(AuthManager.verify_token),
):
    """Caption an uploaded image using BLIP (image/blip.py via
    image/image_caption.py). style: descriptive | creative | technical."""
    try:
        captioner = _get_image_captioner()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Captioning model unavailable: {e}")

    contents = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    tmp_path = os.path.join(_MEDIA_DIR, f"{uuid.uuid4().hex[:12]}{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)
        caption = captioner.generate_caption(tmp_path, style=style)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't caption image: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"filename": file.filename, "caption": caption, "style": style}


# ---------------- Speech (transcription + synthesis) ----------------

_whisper_transcriber = None
_tts_engine = None


def _get_transcriber():
    global _whisper_transcriber
    if _whisper_transcriber is None:
        from speech.whisper import WhisperTranscriber
        _whisper_transcriber = WhisperTranscriber()
    return _whisper_transcriber


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        from speech.tts import TextToSpeech
        _tts_engine = TextToSpeech()
    return _tts_engine


@router.post("/speech/transcribe")
async def speech_transcribe(file: UploadFile = File(...), user: str = Depends(AuthManager.verify_token)):
    """Transcribe an uploaded audio file to text using Whisper
    (speech/whisper.py)."""
    try:
        transcriber = _get_transcriber()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Transcription model unavailable: {e}")

    contents = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_path = os.path.join(_MEDIA_DIR, f"{uuid.uuid4().hex[:12]}{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)
        result = transcriber.transcribe(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't transcribe audio: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"text": result["text"], "language": result.get("language", "unknown")}


@router.post("/speech/speak")
async def speech_speak(request: SpeakRequest, user: str = Depends(AuthManager.verify_token)):
    """Synthesize text to speech using Coqui TTS (speech/tts.py) and stream
    the resulting WAV file back."""
    try:
        tts = _get_tts()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"TTS model unavailable: {e}")

    out_path = os.path.join(_MEDIA_DIR, f"{uuid.uuid4().hex[:12]}.wav")
    try:
        tts.synthesize(request.text, output_path=out_path, speaker=request.speaker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")

    def stream_and_cleanup():
        try:
            with open(out_path, "rb") as f:
                yield from f
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    return StreamingResponse(stream_and_cleanup(), media_type="audio/wav")


# ---------------- RAG ----------------

@router.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...), user: str = Depends(AuthManager.verify_token)):
    """Upload a document (.txt, .md, or .pdf) to be chunked, embedded, and
    stored for this user. Ask questions about it afterward with /rag/query."""
    contents = await file.read()
    filename = file.filename or "upload"

    if filename.lower().endswith(".pdf"):
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(contents))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Couldn't read PDF: {e}")
    else:
        try:
            text = contents.decode("utf-8", errors="replace")
        except Exception:
            raise HTTPException(status_code=400, detail="Couldn't read file as text")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in this file")

    chunks_stored = rag_service.add_document(user, filename, text)
    return {"filename": filename, "chunks_stored": chunks_stored}


@router.post("/rag/query")
async def rag_query(request: RAGQueryRequest, user: str = Depends(AuthManager.verify_token)):
    """Answer a question using this user's uploaded documents, via
    retrieval + your real multi-provider LLMClient (same fallback chain
    /chat uses - not a separate hardcoded OpenAI call)."""
    result = rag_service.query(user, request.question, top_k=request.top_k, preferred=request.model)
    return result


@router.get("/rag/stats")
async def rag_stats(user: str = Depends(AuthManager.verify_token)):
    """How many chunks this user currently has stored."""
    return rag_service.stats(user)


# ---------------- Coding Agent ----------------

class AgentRunRequest(BaseModel):
    task: str
    session_id: Optional[str] = None


# In-memory store of live CodingAgent instances, keyed by session_id, so an
# agent's AgentMemory (short-term/long-term/episodic) survives across
# multiple /agent/run calls in the same session instead of being thrown
# away as soon as one request finishes. This is process-local (resets on
# server restart) - swap for Redis/DB if you need it to survive that too.
agent_sessions: dict = {}


@router.get("/agent/status")
async def agent_status():
    """Which LLM providers the coding agent can actually use right now."""
    from agents.coding_agent import CodingAgent
    try:
        agent = CodingAgent()
        return {"ready": True, "providers": agent.llm.available_providers()}
    except RuntimeError as e:
        return {"ready": False, "providers": [], "error": str(e)}


@router.post("/agent/run")
async def agent_run(request: AgentRunRequest):
    """Run the coding agent on a task, streaming each Thought/Action/
    Observation step as a line of JSON (NDJSON) so a UI can render the
    agent's reasoning live instead of waiting for the final answer.

    Pass back the returned session_id on the next call to resume the same
    agent - it'll remember earlier tasks/results from this session and
    include them as context for the new one.

    NOTE: intentionally left unauthenticated for local/dev use. Add
    Depends(AuthManager.verify_token) before exposing this publicly.
    """
    from agents.coding_agent import CodingAgent

    session_id = request.session_id or uuid.uuid4().hex[:12]

    def event_stream():
        agent = agent_sessions.get(session_id)
        if agent is None:
            try:
                agent = CodingAgent()
            except RuntimeError as e:
                yield json.dumps({"type": "error", "content": str(e)}) + "\n"
                return
            agent_sessions[session_id] = agent

        yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
        try:
            for event in agent.run_steps(request.task):
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/agent/memory/{session_id}")
async def agent_memory(session_id: str):
    """Inspect a session's agent memory - useful for confirming persistence
    is actually working, or for a future UI panel showing what the agent
    remembers."""
    agent = agent_sessions.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="No agent session with that id")
    return {
        "session_id": session_id,
        "summary": agent.memory.get_summary(),
        "recent_short_term": agent.memory.recall_recent(),
        "episodic": agent.memory.episodic,
    }
