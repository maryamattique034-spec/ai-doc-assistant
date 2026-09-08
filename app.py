import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("AI Document Assistant")
st.caption("Upload a PDF, index it, then ask questions. Answers use only the active PDF.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_source" not in st.session_state:
    st.session_state.active_source = None

with st.sidebar:
    st.header("Upload Document")
    if st.session_state.active_source:
        st.info(f"Active PDF: **{st.session_state.active_source}**")
    else:
        st.warning("No active PDF yet. Index one below.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        st.caption("Selecting a file is not enough — click the button to index it.")
        if st.button("Process & Index PDF", type="primary"):
            with st.spinner("Processing PDF and creating vectors..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    res = requests.post(f"{API_URL}/upload", files=files, timeout=600)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.active_source = data.get("file")
                        st.session_state.messages = []
                        st.success(
                            f"Indexed **{data.get('file')}** "
                            f"({data.get('chunks')} chunks). Chat will use this PDF only."
                        )
                        st.rerun()
                    else:
                        st.error(f"Failed ({res.status_code}): {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API. Start the server first.")
                except Exception as e:
                    st.error(str(e))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.active_source:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Index a PDF first (sidebar → Process & Index PDF).",
            }
        )
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching {st.session_state.active_source}..."):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "source": st.session_state.active_source,
                    },
                    timeout=120,
                )
                if res.status_code == 200:
                    payload = res.json()
                    answer = payload["response"]
                    used = payload.get("source")
                    if used:
                        answer = f"{answer}\n\n_Source: `{used}`_"
                else:
                    answer = f"Error ({res.status_code}): {res.text}"
            except requests.exceptions.ConnectionError:
                answer = "Cannot reach API. Start `uvicorn server:app --reload --port 8000` first."
            except Exception as e:
                answer = str(e)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
