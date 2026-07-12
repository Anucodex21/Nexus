"""Intentionally empty (no eager imports).

app.backend.api, .routes, .auth, .database import each other directly
where needed. Keeping this __init__.py empty means importing a single
lightweight submodule (e.g. app.backend.llm_client or
app.backend.rag_service, both of which only need `requests`/stdlib at
module load time) doesn't force fastapi/sqlalchemy/jose/passlib to be
installed too - the same lazy-dependency principle used throughout this
codebase (see llm_client.py's _call_local, rag_service.py's
_get_embedder, routes.py's _get_image_generator, etc).

If you want the FastAPI app object, import it directly:
    from app.backend.api import app
"""
