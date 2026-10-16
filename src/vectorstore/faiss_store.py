"""
FAISS vector store management for the Procurement AI Assistant.

Responsibilities:
  - build_index(): embed chunks and build a fresh FAISS index
  - save_index() / load_index(): persist the index to disk so we don't
    re-embed every document on every app restart
  - similarity_search(): semantic retrieval, returning chunks + metadata
    so the RAG layer can build citation-backed answers

The index is saved to `settings.faiss_index_dir` (default:
data/processed/faiss_index), which is gitignored since it's a build
artifact, not source-controlled data.
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import settings
from src.embeddings.embedder import get_embedding_model


def build_index(chunks: List[Document]) -> FAISS:
    """Embed all chunks and build a new in-memory FAISS index."""
    if not chunks:
        raise ValueError("No chunks provided -- did ingestion run first?")

    embedding_model = get_embedding_model()
    print(f"[vectorstore] Embedding {len(chunks)} chunk(s) and building FAISS index...")
    index = FAISS.from_documents(chunks, embedding_model)
    print("[vectorstore] Index built.")
    return index


def save_index(index: FAISS, index_dir: Optional[str] = None) -> None:
    """Persist a FAISS index to disk."""
    index_dir = index_dir or settings.faiss_index_dir
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index.save_local(index_dir)
    print(f"[vectorstore] Index saved to {index_dir}")


def load_index(index_dir: Optional[str] = None) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    index_dir = index_dir or settings.faiss_index_dir
    if not Path(index_dir).exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_dir}. "
            f"Run the ingestion + indexing pipeline first "
            f"(python -m src.vectorstore.faiss_store)."
        )
    embedding_model = get_embedding_model()
    print(f"[vectorstore] Loading FAISS index from {index_dir}")
    # allow_dangerous_deserialization=True is safe here because we only
    # ever load an index *we* built and saved -- never a third-party file.
    return FAISS.load_local(
        index_dir, embedding_model, allow_dangerous_deserialization=True
    )


def similarity_search(
    index: FAISS, query: str, k: Optional[int] = None
) -> List[Document]:
    """Retrieve the top-k most relevant chunks for a query."""
    k = k or settings.top_k_results
    results = index.similarity_search(query, k=k)
    print(f"[vectorstore] Retrieved {len(results)} chunk(s) for query: {query!r}")
    return results


if __name__ == "__main__":
    # End-to-end smoke test: ingest -> chunk -> embed -> index -> search
    # Run with: python -m src.vectorstore.faiss_store
    from src.ingestion.loaders import load_documents
    from src.ingestion.chunking import chunk_documents

    docs = load_documents(settings.raw_data_dir)
    chunks = chunk_documents(docs)
    idx = build_index(chunks)
    save_index(idx)

    reloaded = load_index()
    hits = similarity_search(reloaded, "What happens if a payment is late?")
    for h in hits:
        print("---")
        print(h.metadata)
        print(h.page_content[:200])
