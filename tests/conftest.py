"""
Shared pytest fixtures.

IMPORTANT: env vars that affect module-level state (DATABASE_URL,
SECRET_KEY, CHROMA_PERSIST_DIR) must be set here, before any test module
imports anything under app.backend - those values are read once at import
time (e.g. database.py creates its engine at module load), not per-request.
Because pytest imports conftest.py before any test file, this is the one
place that's guaranteed to run first.
"""
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_tmp_db_fd, _tmp_db_path = tempfile.mkstemp(suffix=".db")
os.close(_tmp_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_db_path}")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp())
os.environ.setdefault("ENABLE_LOCAL_MODEL", "false")
# Deliberately no provider keys set (GROQ_API_KEY etc.) unless the running
# shell already has them exported - tests that care about "no provider
# configured" behavior rely on that; tests that need a real provider
# should monkeypatch LLMClient directly instead of hitting real APIs.

import pytest
from fastapi.testclient import TestClient

from app.backend.api import app


@pytest.fixture()
def client():
    """TestClient as a context manager so FastAPI's lifespan (which
    creates the DB tables via DatabaseManager.connect()) actually runs."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def unique_username():
    return f"testuser_{uuid.uuid4().hex[:10]}"


@pytest.fixture()
def auth_headers(client, unique_username):
    """Register a fresh random user and return Authorization headers for it."""
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "username": unique_username,
            "email": f"{unique_username}@example.com",
            "password": "correct-horse-battery-staple",
        },
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
