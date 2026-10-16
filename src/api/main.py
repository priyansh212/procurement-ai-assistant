"""
FastAPI backend for the Procurement AI Assistant.

Endpoints:
  GET  /health   -> is the server up, is the FAISS index available?
  POST /query    -> ask a procurement question, get answer + sources
  POST /upload   -> upload a PDF/DOCX/TXT, it is saved and the index is rebuilt
  GET  /documents-> list the documents that are currently indexed

Start it (from the project root):
    uvicorn src.api.main:app --reload --port 8000

Then open  http://127.0.0.1:8000/docs  in your browser. FastAPI builds
an interactive test page there, so you can try /query without writing
any code.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.agent.graph import run_agent
from src.vectorstore.indexer import (
    MAX_UPLOAD_BYTES,
    list_documents,
    rebuild_index,
    save_document,
)


# ----------------------------- request / response models -----------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Who approves a $20,000 purchase?"])
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Chunks to retrieve (default from .env)")
    session_id: str = Field(
        "default",
        description="Same session_id = same conversation (memory). Use a new one to start fresh.",
    )


class Source(BaseModel):
    n: int
    source: str
    doc_type: str
    snippet: str
    cited: bool


class UploadResponse(BaseModel):
    filename: str
    doc_type: str
    documents_indexed: int
    chunks_indexed: int


class QueryResponse(BaseModel):
    question: str
    standalone_question: str  # the question after rewriting with chat context
    answer: str
    sources: List[Source]
    session_id: str


# ----------------------------------- app setup --------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load the models up front so the
    # first user request is not slow.
    from src.agent.graph import get_graph
    from src.rag.chain import _get_index
    from src.rag.llm import get_llm

    try:
        _get_index()
        get_llm()
        get_graph()
        print("[api] Index and LLM ready.")
    except Exception as exc:  # noqa: BLE001
        # Do not crash the server; /health will report the problem.
        print(f"[api] Startup warning: {exc}")
    yield


app = FastAPI(
    title="Procurement AI Assistant",
    description="RAG-based Q&A over contracts, supplier manuals and policy documents.",
    version="0.1.0",
    lifespan=lifespan,
)


# ------------------------------------- routes ---------------------------------------
@app.get("/", include_in_schema=False)
def root():
    # Opening http://127.0.0.1:8000/ in a browser goes straight to the test page.
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    index_exists = Path(settings.faiss_index_dir).exists()
    return {
        "status": "ok" if index_exists else "index_missing",
        "llm_provider": settings.llm_provider,
        "faiss_index_found": index_exists,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # Plain `def` (not async) on purpose: the LLM call blocks, and FastAPI
    # runs plain `def` routes in a worker thread so the server stays responsive.
    try:
        return run_agent(req.question, session_id=req.session_id, k=req.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}")


@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.post("/upload", response_model=UploadResponse)
def upload(
    file: UploadFile = File(...),
    doc_type: str = Form("policies", description="contracts | policies | supplier_manuals"),
):
    # Plain `def`: saving + re-indexing is blocking work, FastAPI runs it in a worker thread.
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")

    try:
        saved = save_document(file.filename or "", content, doc_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        stats = rebuild_index()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"File saved but indexing failed: {exc}")

    return UploadResponse(
        filename=saved.name,
        doc_type=doc_type,
        documents_indexed=stats["documents"],
        chunks_indexed=stats["chunks"],
    )
