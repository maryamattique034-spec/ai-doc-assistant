"""
Streamlit UI: register / login, upload PDF, chat, load history from API.
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("AI Document Assistant")
st.caption("Per-user accounts · chat history via SQLAlchemy · embeddings in ChromaDB")


def auth_headers():
    token = st.session_state.get("token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def api_post(path, **kwargs):
    return requests.post(f"{API_URL}{path}", timeout=kwargs.pop("timeout", 120), **kwargs)


def api_get(path, **kwargs):
    return requests.get(f"{API_URL}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)


# ---------- session defaults ----------
for key, default in [
    ("token", None),
    ("user", None),
    ("messages", []),
    ("active_source", None),
    ("document_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------- login / register ----------
if not st.session_state.token:
    st.subheader("Login or Register")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            try:
                res = api_post(
                    "/login",
                    json={"email": email, "password": password},
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.user = data["user"]
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error(res.json().get("detail", res.text))
            except requests.exceptions.ConnectionError:
                st.error("API not running. Start: uvicorn server:app --reload --port 8000")

    with tab_register:
        email_r = st.text_input("Email", key="reg_email")
        password_r = st.text_input("Password (min 6 chars)", type="password", key="reg_password")
        if st.button("Create account"):
            try:
                res = api_post(
                    "/register",
                    json={"email": email_r, "password": password_r},
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.user = data["user"]
                    st.success("Account created — you are logged in.")
                    st.rerun()
                else:
                    st.error(res.json().get("detail", res.text))
            except requests.exceptions.ConnectionError:
                st.error("API not running. Start: uvicorn server:app --reload --port 8000")

    st.stop()


# ---------- logged-in app ----------
user = st.session_state.user or {}
st.sidebar.write(f"Signed in as **{user.get('email', '')}**")
if st.sidebar.button("Log out"):
    for key in ["token", "user", "messages", "active_source", "document_id"]:
        st.session_state[key] = None if key != "messages" else []
    st.rerun()

st.sidebar.header("Your documents")
try:
    docs_res = api_get("/documents", headers=auth_headers())
    docs = docs_res.json().get("documents", []) if docs_res.status_code == 200 else []
except requests.exceptions.ConnectionError:
    docs = []
    st.sidebar.error("Cannot reach API")

if docs:
    labels = {f"{d['filename']} (id {d['id']})": d for d in docs}
    choice = st.sidebar.selectbox("Select a document", list(labels.keys()))
    selected = labels[choice]
    if st.sidebar.button("Load this document + chat history"):
        st.session_state.active_source = selected["filename"]
        st.session_state.document_id = selected["id"]
        hist = api_get(
            f"/chats?document_id={selected['id']}",
            headers=auth_headers(),
        )
        if hist.status_code == 200:
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in hist.json().get("messages", [])
            ]
        st.rerun()
else:
    st.sidebar.caption("No documents yet — upload one below.")

st.sidebar.header("Upload Document")
if st.session_state.active_source:
    st.sidebar.info(
        f"Active: **{st.session_state.active_source}** "
        f"(doc id: {st.session_state.document_id})"
    )
else:
    st.sidebar.warning("No active PDF yet.")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    st.sidebar.caption("Click the button to index (required).")
    if st.sidebar.button("Process & Index PDF", type="primary"):
        with st.spinner("Indexing PDF..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                res = api_post(
                    "/upload",
                    headers=auth_headers(),
                    files=files,
                    timeout=600,
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.active_source = data.get("file")
                    st.session_state.document_id = data.get("document_id")
                    st.session_state.messages = []
                    st.sidebar.success(
                        f"Indexed **{data.get('file')}** ({data.get('chunks')} chunks)."
                    )
                    st.rerun()
                else:
                    detail = res.json().get("detail", res.text) if res.content else res.text
                    st.sidebar.error(f"Failed: {detail}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("Cannot reach API.")

# Chat area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.active_source:
        st.warning("Index or select a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching {st.session_state.active_source}..."):
            try:
                res = api_post(
                    "/chat",
                    headers=auth_headers(),
                    json={
                        "question": prompt,
                        "source": st.session_state.active_source,
                        "document_id": st.session_state.document_id,
                    },
                )
                if res.status_code == 200:
                    payload = res.json()
                    answer = payload["response"]
                    used = payload.get("source")
                    if used:
                        answer = f"{answer}\n\n_Source: `{used}`_"
                else:
                    answer = f"Error: {res.json().get('detail', res.text)}"
            except requests.exceptions.ConnectionError:
                answer = "Cannot reach API."
            except Exception as e:
                answer = str(e)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
