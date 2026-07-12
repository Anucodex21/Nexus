from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import uvicorn

from .routes import router
from .database import DatabaseManager
from .auth import AuthManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    await DatabaseManager.connect()
    yield
    # Shutdown
    await DatabaseManager.disconnect()

app = FastAPI(
    title="AI-Master API",
    description="API for AI-Master framework",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware. Defaults to allow-all for zero-friction local dev;
# set CORS_ORIGINS (comma-separated) in .env to restrict this before
# deploying anywhere public, e.g. CORS_ORIGINS=https://your-domain.com
_cors_origins_env = os.getenv("CORS_ORIGINS")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",")] if _cors_origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to AI-Master API", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")
