"""
Streamlit chat UI for the Procurement AI Assistant.

This app does NOT contain any AI logic. It only talks to the FastAPI
backend (POST /query), so UI and backend stay separate:

    Browser -> Streamlit (this file) -> FastAPI -> RAG chain -> Groq LLM

Run (backend must be running in another terminal first):
    streamlit run streamlit_app/app.py
"""

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

EXAMPLE_QUESTIONS = [
    "Who needs to approve a purchase of $20,000?",
    "What is the penalty if Acme delivers late?",
    "What are the payment terms in the Acme contract?",
    "When must we give notice to stop auto-renewal?",
    "What is the capital of France?",
]

st.set_page_config(page_title="Procurement AI Assistant", page_icon="📄", layout="centered")


# ------------------------------- helpers -------------------------------
def check_backend() -> dict | None:
    """Return the /health response, or None if the backend is not reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def ask_backend(question: str, top_k: int, session_id: str) -> dict:
    r = requests.post(
        f"{API_URL}/query",
        json={"question": question, "top_k": top_k, "session_id": session_id},
        timeout=90,
    )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"Backend error {r.status_code}: {detail}")
    return r.json()


def upload_document(file, doc_type: str) -> dict:
    r = requests.post(
        f"{API_URL}/upload",
        files={"file": (file.name, file.getvalue())},
        data={"doc_type": doc_type},
        timeout=300,  # re-indexing can take a while
    )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


def list_backend_documents() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        return r.json().get("documents", []) if r.ok else []
    except requests.RequestException:
        return []


def render_sources(sources: list[dict]) -> None:
    cited = [s for s in sources if s["cited"]]
    other = [s for s in sources if not s["cited"]]

    if cited:
        with st.expander(f"Sources used in this answer ({len(cited)})", expanded=False):
            for s in cited:
                st.markdown(f"**[{s['n']}] {s['source']}** · _{s['doc_type']}_")
                st.caption(s["snippet"] + " ...")
    if other:
        with st.expander(f"Other passages that were retrieved ({len(other)})", expanded=False):
            for s in other:
                st.markdown(f"**[{s['n']}] {s['source']}** · _{s['doc_type']}_")
                st.caption(s["snippet"] + " ...")


# ------------------------------- session state -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role", "content", "sources"?}
if "pending" not in st.session_state:
    st.session_state.pending = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0  # changing the key empties the file picker after an upload
if "upload_msg" not in st.session_state:
    st.session_state.upload_msg = None
if "session_id" not in st.session_state:
    # One id per browser chat. The backend uses it to remember this conversation.
    st.session_state.session_id = str(uuid.uuid4())


# ------------------------------- sidebar -------------------------------
with st.sidebar:
    st.header("Settings")

    health = check_backend()
    if health is None:
        st.error("Backend not reachable.\n\nStart it in another terminal:\n\n"
                 "`uvicorn src.api.main:app --port 8000`")
    elif health.get("faiss_index_found"):
        st.success(f"Backend online · LLM: {health.get('llm_provider')}")
    else:
        st.warning("Backend online but FAISS index is missing. "
                   "Run `python -m src.vectorstore.faiss_store`.")

    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, 4)

    st.subheader("Try an example")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

    if st.button("Clear chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())  # new id = backend forgets the old chat
        st.rerun()

    st.subheader("Add your documents")
    doc_type = st.selectbox("Document type", ["contracts", "policies", "supplier_manuals"])
    uploaded = st.file_uploader(
        "PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        key=f"uploader_{st.session_state.upload_key}",
    )
    if st.button("Upload & index", use_container_width=True, disabled=uploaded is None):
        with st.spinner("Uploading and re-indexing..."):
            try:
                res = upload_document(uploaded, doc_type)
                st.session_state.upload_msg = (
                    "success",
                    f"Added {res['filename']}. Index now has {res['documents_indexed']} sections "
                    f"({res['chunks_indexed']} chunks).",
                )
                st.session_state.upload_key += 1
            except Exception as exc:  # noqa: BLE001
                st.session_state.upload_msg = ("error", f"Upload failed: {exc}")
        st.rerun()

    if st.session_state.upload_msg:
        kind, text = st.session_state.upload_msg
        (st.success if kind == "success" else st.error)(text)

    docs_now = list_backend_documents()
    if docs_now:
        with st.expander(f"Indexed documents ({len(docs_now)})"):
            for d in docs_now:
                st.caption(f"{d['filename']} · {d['doc_type']}")


# ------------------------------- main page -------------------------------
st.title("📄 Procurement AI Assistant")
st.caption("Ask questions about contracts, supplier manuals and procurement policies. "
           "Answers are grounded in your documents and cite their sources.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rewritten"):
            st.caption(f"🔎 Understood as: {msg['rewritten']}")
        if msg.get("sources"):
            render_sources(msg["sources"])

typed = st.chat_input("Ask a procurement question...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask_backend(question, top_k, st.session_state.session_id)
                st.markdown(result["answer"])

                # Show the rewritten question only when memory actually changed it
                rewritten = result.get("standalone_question", "")
                if rewritten.strip().lower() == question.strip().lower():
                    rewritten = ""
                if rewritten:
                    st.caption(f"🔎 Understood as: {rewritten}")

                render_sources(result["sources"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "rewritten": rewritten,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                err = f"Sorry, something went wrong.\n\n`{exc}`"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
