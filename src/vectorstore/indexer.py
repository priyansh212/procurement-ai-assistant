"""
Document upload + re-indexing helpers for the Procurement AI Assistant.

Used by the API's /upload endpoint:
  1. save_document()  - validate the file name/type and store it in data/raw/<doc_type>/
  2. rebuild_index()  - load ALL documents again, chunk, embed, save the FAISS index and
                        hot-swap it into the running RAG chain (no server restart needed)

The whole corpus is re-indexed on every upload. That is fine for a few dozen documents
(takes a few seconds) and keeps the code simple and the index always consistent.
"""

import re
import threading
from pathlib import Path
from typing import Dict, List

from src.config import settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import SUPPORTED_EXTENSIONS, load_documents
from src.vectorstore.faiss_store import build_index, save_index

ALLOWED_DOC_TYPES = ("contracts", "policies", "supplier_manuals")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

_rebuild_lock = threading.Lock()  # two uploads at the same time must not rebuild together


def safe_filename(name: str) -> str:
    """Keep only the base name (no folders, so '../../x.txt' cannot escape) and safe characters."""
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    if not base:
        raise ValueError("Invalid file name.")
    if Path(base).suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Allowed: {allowed}")
    return base


def save_document(filename: str, content: bytes, doc_type: str) -> Path:
    """Store an uploaded file under data/raw/<doc_type>/ (same name = overwrite = updated version)."""
    if doc_type not in ALLOWED_DOC_TYPES:
        raise ValueError(f"doc_type must be one of: {', '.join(ALLOWED_DOC_TYPES)}")
    if not content:
        raise ValueError("The uploaded file is empty.")

    name = safe_filename(filename)
    target_dir = Path(settings.raw_data_dir) / doc_type
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    target.write_bytes(content)
    return target


def rebuild_index() -> Dict[str, int]:
    """Re-read every document in data/raw, rebuild + save the FAISS index, swap it into the chain."""
    with _rebuild_lock:
        docs = load_documents(Path(settings.raw_data_dir))
        if not docs:
            raise ValueError("No documents found to index.")
        chunks = chunk_documents(docs)
        index = build_index(chunks)
        save_index(index)

        # Hot swap: the running API uses this new index from the next question on.
        import src.rag.chain as chain

        chain._index = index
        return {"documents": len(docs), "chunks": len(chunks)}


def list_documents() -> List[Dict[str, str]]:
    """All documents currently in data/raw, with their doc_type."""
    raw = Path(settings.raw_data_dir)
    items = []
    for p in sorted(raw.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel = p.relative_to(raw)
            items.append({"filename": p.name, "doc_type": rel.parts[0] if len(rel.parts) > 1 else "unknown"})
    return items
