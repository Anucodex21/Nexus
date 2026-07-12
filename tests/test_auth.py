def test_register_returns_token(client, unique_username):
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "username": unique_username,
            "email": f"{unique_username}@example.com",
            "password": "correct-horse-battery-staple",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["token_type"] == "bearer"
    assert data["access_token"]


def test_duplicate_username_rejected(client, unique_username):
    payload = {
        "username": unique_username,
        "email": f"{unique_username}@example.com",
        "password": "correct-horse-battery-staple",
    }
    first = client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 200

    second = client.post("/api/v1/auth/register", json=payload)
    assert second.status_code == 400


def test_login_success(client, unique_username):
    client.post(
        "/api/v1/auth/register",
        json={
            "username": unique_username,
            "email": f"{unique_username}@example.com",
            "password": "correct-horse-battery-staple",
        },
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": unique_username, "password": "correct-horse-battery-staple"},
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_wrong_password(client, unique_username):
    client.post(
        "/api/v1/auth/register",
        json={
            "username": unique_username,
            "email": f"{unique_username}@example.com",
            "password": "correct-horse-battery-staple",
        },
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": unique_username, "password": "wrong-password"},
    )
    assert resp.status_code == 401


def test_login_unknown_user(client):
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "does-not-exist", "password": "whatever"},
    )
    assert resp.status_code == 401


def test_protected_route_requires_token(client):
    resp = client.get("/api/v1/conversations")
    assert resp.status_code in (401, 403)


def test_protected_route_with_token(client, auth_headers):
    resp = client.get("/api/v1/conversations", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["conversations"] == []
