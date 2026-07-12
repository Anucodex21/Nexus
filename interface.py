"""
AI-Master Chat - Streamlit frontend for the FastAPI backend.

Run the backend first:  python -m app.backend.main   (or: uvicorn app.backend.api:app --reload)
Then run this:           streamlit run interface.py
"""

import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

st.set_page_config(page_title="AI Master", page_icon="🤖", layout="wide")

# ---------------- session state ----------------

if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "attached_file" not in st.session_state:
    st.session_state.attached_file = None


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ---------------- auth screen ----------------

def show_auth_screen():
    st.title("🤖 AI-Master")
    st.write("Sign in or create an account to start chatting.")

    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            try:
                res = requests.post(f"{API_BASE}/auth/login",
                                     json={"username": username, "password": password}, timeout=10)
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Login failed"))
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Is it running on http://127.0.0.1:8000 ?")

    with tab_register:
        with st.form("register_form"):
            username = st.text_input("Username", key="reg_username")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            try:
                res = requests.post(f"{API_BASE}/auth/register",
                                     json={"username": username, "email": email, "password": password},
                                     timeout=10)
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Registration failed"))
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Is it running on http://127.0.0.1:8000 ?")


# ---------------- main chat screen ----------------

def load_conversations():
    try:
        res = requests.get(f"{API_BASE}/conversations", headers=auth_headers(), timeout=10)
        if res.status_code == 200:
            return res.json().get("conversations", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


def load_models():
    try:
        res = requests.get(f"{API_BASE}/models", headers=auth_headers(), timeout=10)
        if res.status_code == 200:
            return res.json().get("models", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


def load_conversation_messages(conversation_id):
    try:
        res = requests.get(f"{API_BASE}/conversations/{conversation_id}",
                            headers=auth_headers(), timeout=10)
        if res.status_code == 200:
            history = []
            for turn in res.json().get("messages", []):
                history.append({"role": "user", "content": turn["message"]})
                history.append({"role": "assistant", "content": turn["response"]})
            return history
    except requests.exceptions.ConnectionError:
        pass
    return []


def show_chat_screen():
    with st.sidebar:
        st.markdown(f"**{st.session_state.username}**")
        if st.button("Log out"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        if st.button("➕ New chat", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        st.caption("Recent chats")
        conversations = load_conversations()
        if not conversations:
            st.caption("No chats yet.")
        for conv in conversations:
            label = conv["title"] or "New chat"
            if st.button(label, key=f"conv_{conv['conversation_id']}", use_container_width=True):
                st.session_state.conversation_id = conv["conversation_id"]
                st.session_state.messages = load_conversation_messages(conv["conversation_id"])
                st.rerun()

    st.title("🤖 AI-Master")

    models = load_models()
    model_options = ["auto (configured order)"] + models
    selected_model = st.selectbox("Model", model_options, label_visibility="collapsed")
    preferred_model = None if selected_model == "auto (configured order)" else selected_model
    if not models:
        st.caption("No AI provider keys configured yet - responses will be offline notices "
                   "until you add one (e.g. GROQ_API_KEY) to your .env.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    uploaded_file = st.file_uploader("Attach a text file (optional)", type=None,
                                      label_visibility="collapsed")
    if uploaded_file is not None and st.session_state.attached_file is None:
        try:
            content = uploaded_file.read().decode("utf-8", errors="replace")
            st.session_state.attached_file = {"filename": uploaded_file.name, "content": content[:6000]}
        except Exception:
            st.warning("Couldn't read that file as text.")
    if st.session_state.attached_file:
        st.caption(f"📎 Attached: {st.session_state.attached_file['filename']}")
        if st.button("Remove attachment"):
            st.session_state.attached_file = None
            st.rerun()

    if user_input := st.chat_input("Ask me anything..."):
        outgoing = user_input
        if st.session_state.attached_file:
            f = st.session_state.attached_file
            outgoing = f"[Attached file: {f['filename']}]\n{f['content']}\n\n{user_input}"

        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.attached_file = None

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        f"{API_BASE}/chat",
                        headers=auth_headers(),
                        json={
                            "message": outgoing,
                            "model": preferred_model,
                            "conversation_id": st.session_state.conversation_id,
                        },
                        timeout=30,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.conversation_id = data["conversation_id"]
                        reply = data["response"]
                    elif res.status_code == 401:
                        reply = "Your session expired. Please log out and log back in."
                    else:
                        reply = f"Error from server: {res.status_code}"
                except requests.exceptions.ConnectionError:
                    reply = "Can't reach the backend. Is it running on http://127.0.0.1:8000 ?"
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


# ---------------- entry point ----------------

if st.session_state.token is None:
    show_auth_screen()
else:
    show_chat_screen()
