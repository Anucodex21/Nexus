import json


def test_chat_offline_fallback(client, auth_headers):
    """With no provider keys configured in the test environment, /chat
    should still succeed and clearly report it used the offline path -
    never silently pretend to be a real model."""
    resp = client.post(
        "/api/v1/chat",
        json={"message": "hello there"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "offline"
    assert data["conversation_id"]
    assert "hello there" in data["response"]


def test_chat_requires_auth(client):
    resp = client.post("/api/v1/chat", json={"message": "hello"})
    assert resp.status_code in (401, 403)


def test_conversation_history_persists(client, auth_headers):
    first = client.post(
        "/api/v1/chat", json={"message": "message one"}, headers=auth_headers
    )
    cid = first.json()["conversation_id"]

    client.post(
        "/api/v1/chat",
        json={"message": "message two", "conversation_id": cid},
        headers=auth_headers,
    )

    resp = client.get(f"/api/v1/conversations/{cid}", headers=auth_headers)
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["message"] == "message one"
    assert messages[1]["message"] == "message two"


def test_conversations_list_shows_new_conversation(client, auth_headers):
    resp = client.post(
        "/api/v1/chat", json={"message": "a fresh conversation"}, headers=auth_headers
    )
    cid = resp.json()["conversation_id"]

    listing = client.get("/api/v1/conversations", headers=auth_headers)
    ids = [c["conversation_id"] for c in listing.json()["conversations"]]
    assert cid in ids


def test_chat_stream_ndjson(client, auth_headers):
    resp = client.post(
        "/api/v1/chat/stream",
        json={"message": "stream this"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    lines = [json.loads(l) for l in resp.text.strip().splitlines() if l.strip()]
    types = [l["type"] for l in lines]
    # Offline path still goes through start -> delta -> done.
    assert types[0] == "start"
    assert "delta" in types
    assert types[-1] == "done"


def test_models_endpoint(client, auth_headers):
    resp = client.get("/api/v1/models", headers=auth_headers)
    assert resp.status_code == 200
    assert "models" in resp.json()
