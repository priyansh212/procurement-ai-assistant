"""
Text chunking for the Procurement AI Assistant.

We use LangChain's RecursiveCharacterTextSplitter because procurement
documents (contracts, policies) are mostly prose with clause/section
structure -- recursive splitting on paragraph -> sentence -> word
boundaries keeps clauses intact far better than a naive fixed-size split.

Every chunk keeps the parent document's metadata (source, doc_type, page)
plus a chunk_id, so the RAG layer can cite "Contract_ABC.pdf, chunk 3"
in its answers.

Contextual header: a chunk taken from the middle of a contract no longer says
which contract it belongs to (e.g. "Article 3 ... 98% on-time delivery" never
mentions "Globex"), so a question that names the supplier may not find it.
We therefore put a one-line header "[Document: <file> | Type: <doc_type>]" at the
top of every chunk BEFORE embedding. The RAG layer removes the header again
(strip_context_header) so the LLM prompt and the UI do not show it twice.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

HEADER_PREFIX = "[Document: "


def strip_context_header(text: str) -> str:
    """Remove the contextual header line (if present) from a chunk's text."""
    if text.startswith(HEADER_PREFIX) and "\n" in text:
        return text.split("\n", 1)[1]
    return text


def _add_context_header(chunk: Document) -> None:
    source = chunk.metadata.get("source", "unknown")
    doc_type = chunk.metadata.get("doc_type", "unknown")
    chunk.page_content = f"{HEADER_PREFIX}{source} | Type: {doc_type}]\n{chunk.page_content}"


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
    add_header: bool = True,
) -> List[Document]:
    """Split loaded documents into overlapping chunks for embedding."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        if add_header:
            _add_context_header(chunk)

    print(f"[chunking] Split {len(documents)} document section(s) into "
          f"{len(chunks)} chunk(s) (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.ingestion.chunking
    from src.ingestion.loaders import load_documents

    docs = load_documents(settings.raw_data_dir)
    chunks = chunk_documents(docs)
    if chunks:
        print("--- sample chunk ---")
        print(chunks[0].metadata)
        print(chunks[0].page_content[:300])
